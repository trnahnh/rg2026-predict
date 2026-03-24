"""Monte Carlo tournament simulation and backtesting."""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.config import (
    BACKTEST_YEARS,
    DATA_PROCESSED,
    MC_SIMULATIONS,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
)
from src.data.features import FeatureBuilder, build_feature_matrix, melt_matches
from src.elo.engine import compute_elo_history, get_current_ratings, get_surface_elo
from src.simulate.bracket import Bracket, build_seeded_bracket, get_rg_entrants


def precompute_player_features(
    builder: FeatureBuilder,
    player_ids: list[str],
    elo_players: dict,
    matches: pd.DataFrame,
    tourney_date: pd.Timestamp,
    surface: str = "Clay",
) -> dict[str, dict]:
    """Compute static per-player features for the tournament date."""
    player_features = {}
    for pid in player_ids:
        feats = {}
        feats.update(builder.rolling_features(pid, tourney_date, surface))

        serve = builder.serving_features(pid, tourney_date, surface=None)
        for k, v in serve.items():
            feats[f"serve_{k}"] = v

        serve_s = builder.serving_features(pid, tourney_date, surface=surface)
        for k, v in serve_s.items():
            feats[f"serve_surface_{k}"] = v

        if pid in elo_players:
            ep = elo_players[pid]
            feats["elo_overall"] = ep.overall
            feats["elo_surface"] = get_surface_elo(ep, surface)
        else:
            feats["elo_overall"] = np.nan
            feats["elo_surface"] = np.nan

        rank_rows = matches.loc[(matches["winner_id"] == pid) | (matches["loser_id"] == pid)]
        if len(rank_rows) > 0:
            last = rank_rows.iloc[-1]
            if str(last["winner_id"]) == pid:
                feats["rank"] = last["winner_rank"]
                feats["age"] = last["winner_age"]
                feats["height"] = last["winner_ht"]
            else:
                feats["rank"] = last["loser_rank"]
                feats["age"] = last["loser_age"]
                feats["height"] = last["loser_ht"]
        else:
            feats["rank"] = np.nan
            feats["age"] = np.nan
            feats["height"] = np.nan

        player_features[pid] = feats
    return player_features


def build_matchup_features(
    p_feats: dict,
    o_feats: dict,
    builder: FeatureBuilder,
    pid: str,
    oid: str,
    tourney_date: pd.Timestamp,
    round_name: str = "R128",
    tourney_level: str = "G",
) -> dict:
    """Build a single feature row for a matchup between two players."""
    row = {}

    row["elo_diff"] = (p_feats.get("elo_overall", np.nan) or np.nan) - (
        o_feats.get("elo_overall", np.nan) or np.nan
    )
    row["elo_surface_diff"] = (p_feats.get("elo_surface", np.nan) or np.nan) - (
        o_feats.get("elo_surface", np.nan) or np.nan
    )

    p_rank = p_feats.get("rank", np.nan)
    o_rank = o_feats.get("rank", np.nan)
    if pd.notna(p_rank) and pd.notna(o_rank):
        row["rank_diff"] = o_rank - p_rank
        row["log_rank_ratio"] = np.log(o_rank / p_rank) if p_rank > 0 and o_rank > 0 else np.nan
    else:
        row["rank_diff"] = np.nan
        row["log_rank_ratio"] = np.nan

    row["age"] = p_feats.get("age", np.nan)
    row["height"] = p_feats.get("height", np.nan)

    for key in [
        "win_rate_12m",
        "win_rate_3m",
        "win_rate_clay_12m",
        "win_rate_clay_3m",
        "win_rate_surface_12m",
        "matches_12m",
        "matches_3m",
        "matches_30d",
        "clay_matches_12m",
        "titles_12m",
        "titles_clay_12m",
        "serve_first_serve_pct",
        "serve_first_serve_win_pct",
        "serve_second_serve_win_pct",
        "serve_bp_saved_pct",
        "serve_ace_rate",
        "serve_df_rate",
        "serve_surface_first_serve_pct",
        "serve_surface_first_serve_win_pct",
        "serve_surface_second_serve_win_pct",
        "serve_surface_bp_saved_pct",
        "serve_surface_ace_rate",
        "serve_surface_df_rate",
    ]:
        row[key] = p_feats.get(key, np.nan)

    h2h = builder.h2h_features(pid, oid, tourney_date)
    row["h2h_win_pct"] = h2h["h2h_win_pct"]
    row["h2h_total"] = h2h["h2h_total"]

    ctx = builder.tournament_context(pid, tourney_date, round_name, np.nan, tourney_level)
    row["round_num"] = ctx["round_num"]
    row["is_seeded"] = ctx["is_seeded"]
    row["seed"] = ctx["seed"]
    row["is_grand_slam"] = ctx["is_grand_slam"]
    row["rg_matches_career"] = ctx["rg_matches_career"]
    row["rg_wins_career"] = ctx["rg_wins_career"]
    row["rg_best_round"] = ctx["rg_best_round"]

    return row


def precompute_pairwise_probabilities(
    player_ids: list[str],
    player_features: dict[str, dict],
    builder: FeatureBuilder,
    model: xgb.Booster,
    calibrator: IsotonicRegression | None,
    feature_cols: list[str],
    tourney_date: pd.Timestamp,
) -> np.ndarray:
    """Precompute NxN win probability matrix. prob_matrix[i][j] = P(i beats j)."""
    n = len(player_ids)
    prob_matrix = np.full((n, n), 0.5)

    rows = []
    pairs = []
    for i, pid in enumerate(player_ids):
        for j, oid in enumerate(player_ids):
            if i >= j:
                continue
            p_feats = player_features.get(pid, {})
            o_feats = player_features.get(oid, {})
            row = build_matchup_features(
                p_feats,
                o_feats,
                builder,
                pid,
                oid,
                tourney_date,
            )
            rows.append(row)
            pairs.append((i, j))

    if not rows:
        return prob_matrix

    df = pd.DataFrame(rows)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    dmat = xgb.DMatrix(df[feature_cols])
    raw_probs = model.predict(dmat)

    if calibrator is not None:
        probs = calibrator.predict(raw_probs)
    else:
        probs = raw_probs

    for (i, j), p in zip(pairs, probs):
        prob_matrix[i][j] = p
        prob_matrix[j][i] = 1.0 - p

    return prob_matrix


def simulate_tournament(
    bracket: Bracket,
    prob_matrix: np.ndarray,
    pid_to_idx: dict[str, int],
    rng: np.random.Generator,
) -> str:
    """Simulate one tournament run. Returns the winner's player_id."""
    current_round = [p.player_id for p in bracket.players]

    while len(current_round) > 1:
        next_round = []
        for k in range(0, len(current_round), 2):
            p1 = current_round[k]
            p2 = current_round[k + 1]

            if p1.startswith("BYE"):
                next_round.append(p2)
                continue
            if p2.startswith("BYE"):
                next_round.append(p1)
                continue

            i = pid_to_idx.get(p1)
            j = pid_to_idx.get(p2)
            if i is not None and j is not None:
                p1_wins_prob = prob_matrix[i][j]
            else:
                p1_wins_prob = 0.5

            winner = p1 if rng.random() < p1_wins_prob else p2
            next_round.append(winner)

        current_round = next_round

    return current_round[0]


def run_monte_carlo(
    bracket: Bracket,
    prob_matrix: np.ndarray,
    pid_to_idx: dict[str, int],
    n_simulations: int = MC_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run Monte Carlo simulation and return per-player win counts."""
    rng = np.random.default_rng(seed)
    win_counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    semi_counts: dict[str, int] = {}

    for _ in range(n_simulations):
        current_round = [p.player_id for p in bracket.players]
        round_num = 0

        while len(current_round) > 1:
            next_round = []
            round_num += 1
            for k in range(0, len(current_round), 2):
                p1 = current_round[k]
                p2 = current_round[k + 1]

                if p1.startswith("BYE"):
                    next_round.append(p2)
                    continue
                if p2.startswith("BYE"):
                    next_round.append(p1)
                    continue

                i = pid_to_idx.get(p1)
                j = pid_to_idx.get(p2)
                p1_prob = prob_matrix[i][j] if (i is not None and j is not None) else 0.5
                winner = p1 if rng.random() < p1_prob else p2
                next_round.append(winner)

            if len(next_round) == 2:
                for pid in next_round:
                    final_counts[pid] = final_counts.get(pid, 0) + 1
            if len(next_round) <= 4 and len(current_round) > 4:
                for pid in next_round:
                    semi_counts[pid] = semi_counts.get(pid, 0) + 1

            current_round = next_round

        champion = current_round[0]
        win_counts[champion] = win_counts.get(champion, 0) + 1

    results = []
    all_pids = set(win_counts) | set(final_counts) | set(semi_counts)
    for pid in all_pids:
        results.append(
            {
                "player_id": pid,
                "win_count": win_counts.get(pid, 0),
                "win_prob": win_counts.get(pid, 0) / n_simulations,
                "final_count": final_counts.get(pid, 0),
                "final_prob": final_counts.get(pid, 0) / n_simulations,
                "semi_count": semi_counts.get(pid, 0),
                "semi_prob": semi_counts.get(pid, 0) / n_simulations,
            }
        )

    df = pd.DataFrame(results).sort_values("win_prob", ascending=False).reset_index(drop=True)
    return df


def run_prediction() -> None:
    """Run Monte Carlo prediction for RG 2026."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and model...")
    matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    model = joblib.load(MODELS_DIR / "xgb_final.joblib")
    calibrator = joblib.load(MODELS_DIR / "calibrator.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")

    from src.elo.engine import PlayerElo, update_ratings

    elo_players: dict[str, PlayerElo] = {}
    for row in matches.sort_values(["tourney_date", "tourney_id", "match_num"]).itertuples():
        wid, lid = str(row.winner_id), str(row.loser_id)
        if wid not in elo_players:
            elo_players[wid] = PlayerElo()
        if lid not in elo_players:
            elo_players[lid] = PlayerElo()
        update_ratings(elo_players[wid], elo_players[lid], str(row.surface))

    elo_current = get_current_ratings(elo_players)

    print("Selecting entrants...")
    seeds, unseeded = get_rg_entrants(matches, elo_current)
    rng = np.random.default_rng(RANDOM_SEED)
    bracket = build_seeded_bracket(seeds, unseeded, rng)

    all_player_ids = [p.player_id for p in bracket.players]
    pid_to_idx = {pid: i for i, pid in enumerate(all_player_ids)}
    name_map = {}
    for p in bracket.players:
        name_map[p.player_id] = p.name

    print("Building player features...")
    melted = melt_matches(matches)
    builder = FeatureBuilder(melted)
    tourney_date = pd.Timestamp("2026-05-18")

    player_features = precompute_player_features(
        builder,
        all_player_ids,
        elo_players,
        matches,
        tourney_date,
    )

    print("Precomputing pairwise probabilities...")
    prob_matrix = precompute_pairwise_probabilities(
        all_player_ids,
        player_features,
        builder,
        model,
        calibrator,
        feature_cols,
        tourney_date,
    )

    print(f"Running {MC_SIMULATIONS:,} simulations...")
    results = run_monte_carlo(bracket, prob_matrix, pid_to_idx)

    results["name"] = results["player_id"].map(name_map).fillna("Unknown")
    cols = [
        "name",
        "player_id",
        "win_prob",
        "final_prob",
        "semi_prob",
        "win_count",
        "final_count",
        "semi_count",
    ]
    results = results[cols]

    results.to_parquet(OUTPUTS_DIR / "rg2026_predictions.parquet", index=False)
    results.to_csv(OUTPUTS_DIR / "rg2026_predictions.csv", index=False)

    print("\nTop 20 predicted RG 2026 winners:")
    print(f"{'Rank':<6} {'Name':<25} {'Win%':>8} {'Final%':>8} {'Semi%':>8}")
    print("-" * 60)
    for i, row in results.head(20).iterrows():
        print(
            f"{i + 1:<6} {row['name']:<25} "
            f"{row['win_prob'] * 100:>7.2f}% "
            f"{row['final_prob'] * 100:>7.2f}% "
            f"{row['semi_prob'] * 100:>7.2f}%"
        )

    print(f"\nSaved to {OUTPUTS_DIR / 'rg2026_predictions.csv'}")


def run_backtest() -> None:
    """Backtest: for each year, check if the actual RG winner is in the top-N predictions."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    best_params = joblib.load(MODELS_DIR / "best_params.joblib")
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")

    results = []
    for year in BACKTEST_YEARS:
        print(f"\n{'=' * 60}")
        print(f"Backtesting RG {year}...")

        rg_matches = matches[
            (matches["tourney_date"].dt.year == year)
            & matches["tourney_name"].str.contains(
                "Roland Garros|French Open",
                case=False,
                na=False,
            )
        ]
        if len(rg_matches) == 0:
            print(f"  No RG matches found for {year}, skipping.")
            continue

        final = rg_matches[rg_matches["round"] == "F"]
        if len(final) == 0:
            print(f"  No final found for {year}, skipping.")
            continue
        actual_winner = str(final.iloc[0]["winner_id"])
        winner_name = str(final.iloc[0]["winner_name"])

        train_matches = matches[matches["tourney_date"] < pd.Timestamp(f"{year}-05-01")]
        if len(train_matches) < 1000:
            print(f"  Insufficient training data for {year}, skipping.")
            continue

        elo_hist = compute_elo_history(train_matches)
        features = build_feature_matrix(train_matches, elo_hist)

        from src.elo.engine import PlayerElo, update_ratings

        elo_players: dict[str, PlayerElo] = {}
        for row in train_matches.sort_values(
            ["tourney_date", "tourney_id", "match_num"]
        ).itertuples():
            wid, lid = str(row.winner_id), str(row.loser_id)
            if wid not in elo_players:
                elo_players[wid] = PlayerElo()
            if lid not in elo_players:
                elo_players[lid] = PlayerElo()
            update_ratings(elo_players[wid], elo_players[lid], str(row.surface))

        elo_current = get_current_ratings(elo_players)

        val_year = year - 1
        train_year = features["tourney_date"].dt.year
        tr = features[train_year < val_year]
        va = features[train_year == val_year]

        if len(tr) < 100 or len(va) < 100:
            print(f"  Insufficient split data for {year}, skipping.")
            continue

        from src.model.train import calibrate_model, train_xgboost

        model = train_xgboost(tr, va, feature_cols, best_params)
        calibrator = calibrate_model(model, va, feature_cols)

        seeds, unseeded = get_rg_entrants(train_matches, elo_current)
        rng = np.random.default_rng(RANDOM_SEED)
        bracket = build_seeded_bracket(seeds, unseeded, rng)

        all_pids = [p.player_id for p in bracket.players]
        pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}

        melted = melt_matches(train_matches)
        builder = FeatureBuilder(melted)
        tourney_date = pd.Timestamp(f"{year}-05-20")

        player_feats = precompute_player_features(
            builder,
            all_pids,
            elo_players,
            train_matches,
            tourney_date,
        )

        prob_matrix = precompute_pairwise_probabilities(
            all_pids,
            player_feats,
            builder,
            model,
            calibrator,
            feature_cols,
            tourney_date,
        )

        predictions = run_monte_carlo(bracket, prob_matrix, pid_to_idx)

        if actual_winner in predictions["player_id"].values:
            winner_rank = predictions[predictions["player_id"] == actual_winner].index[0] + 1
            winner_prob = predictions.loc[
                predictions["player_id"] == actual_winner, "win_prob"
            ].iloc[0]
        else:
            winner_rank = -1
            winner_prob = 0.0

        print(f"  Actual winner: {winner_name} (ID: {actual_winner})")
        print(f"  Predicted rank: {winner_rank}, Win prob: {winner_prob * 100:.2f}%")

        results.append(
            {
                "year": year,
                "actual_winner": winner_name,
                "actual_winner_id": actual_winner,
                "predicted_rank": winner_rank,
                "predicted_win_prob": winner_prob,
                "in_top5": winner_rank <= 5 if winner_rank > 0 else False,
                "in_top10": winner_rank <= 10 if winner_rank > 0 else False,
            }
        )

    if results:
        df = pd.DataFrame(results)
        df.to_parquet(OUTPUTS_DIR / "backtest_results.parquet", index=False)

        print(f"\n{'=' * 60}")
        print("Backtest Summary:")
        print(f"  Years tested: {len(df)}")
        top5 = df["in_top5"]
        top10 = df["in_top10"]
        print(f"  Winner in top 5:  {top5.sum()}/{len(df)} ({top5.mean() * 100:.0f}%)")
        print(f"  Winner in top 10: {top10.sum()}/{len(df)} ({top10.mean() * 100:.0f}%)")
        print(f"  Mean predicted rank: {df['predicted_rank'].mean():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        run_prediction()
