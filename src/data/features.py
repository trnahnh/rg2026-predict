"""Engineer match-level features for the XGBoost classifier.

Strategy: melt each match into two player-perspective rows, pre-group by
player_id for O(1) lookups, compute rolling stats per player using only
pre-match data, then assemble symmetric feature pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATA_ELO,
    DATA_PROCESSED,
    FATIGUE_WINDOW_DAYS,
    FORM_WINDOW_3M_DAYS,
    FORM_WINDOW_12M_DAYS,
    LAPLACE_PRIOR,
    ROLLING_CLAY_MATCHES,
    ROLLING_MIN_PERIODS,
    ROUND_MAP,
)

SERVE_STAT_COLS = [
    "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
    "SvGms", "bpSaved", "bpFaced",
]


def melt_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert each match into two player-perspective rows (winner / loser)."""
    winner_rows = matches.assign(
        player_id=matches["winner_id"],
        opponent_id=matches["loser_id"],
        won=1,
        player_rank=matches["winner_rank"],
        player_rank_points=matches["winner_rank_points"],
        player_age=matches["winner_age"],
        player_ht=matches["winner_ht"],
        player_hand=matches["winner_hand"],
        player_seed=matches["winner_seed"],
        opponent_rank=matches["loser_rank"],
    )
    loser_rows = matches.assign(
        player_id=matches["loser_id"],
        opponent_id=matches["winner_id"],
        won=0,
        player_rank=matches["loser_rank"],
        player_rank_points=matches["loser_rank_points"],
        player_age=matches["loser_age"],
        player_ht=matches["loser_ht"],
        player_hand=matches["loser_hand"],
        player_seed=matches["loser_seed"],
        opponent_rank=matches["winner_rank"],
    )

    for prefix_from, prefix_to in [("w_", "p_"), ("l_", "o_")]:
        for stat in SERVE_STAT_COLS:
            winner_rows[f"{prefix_to}{stat}"] = matches[f"{prefix_from}{stat}"]

    for prefix_from, prefix_to in [("l_", "p_"), ("w_", "o_")]:
        for stat in SERVE_STAT_COLS:
            loser_rows[f"{prefix_to}{stat}"] = matches[f"{prefix_from}{stat}"]

    keep_cols = [
        "tourney_id", "tourney_name", "tourney_date", "tourney_level",
        "surface", "round", "match_num", "best_of", "draw_size", "minutes",
        "player_id", "opponent_id", "won",
        "player_rank", "player_rank_points", "player_age", "player_ht",
        "player_hand", "player_seed", "opponent_rank",
    ] + [f"p_{s}" for s in SERVE_STAT_COLS] + [f"o_{s}" for s in SERVE_STAT_COLS]

    melted = pd.concat(
        [winner_rows[keep_cols], loser_rows[keep_cols]],
        ignore_index=True,
    )
    melted = melted.sort_values(
        ["tourney_date", "tourney_id", "match_num", "won"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return melted


class FeatureBuilder:
    """Pre-indexes melted match data for fast per-player lookups."""

    def __init__(self, melted: pd.DataFrame) -> None:
        self._build_indices(melted)

    def _build_indices(self, melted: pd.DataFrame) -> None:
        self._player_data: dict[str, pd.DataFrame] = {}
        self._player_dates: dict[str, np.ndarray] = {}
        self._rg_data: dict[str, pd.DataFrame] = {}

        for pid, group in melted.groupby("player_id"):
            sorted_group = group.sort_values("tourney_date").reset_index(drop=True)
            self._player_data[pid] = sorted_group
            self._player_dates[pid] = sorted_group["tourney_date"].values

        rg_mask = melted["tourney_name"].str.contains(
            "Roland Garros|French Open", case=False, na=False
        )
        rg_melted = melted.loc[rg_mask]
        for pid, group in rg_melted.groupby("player_id"):
            self._rg_data[pid] = group.sort_values("tourney_date").reset_index(drop=True)

    def _get_history_before(
        self, player_id: str, match_date: pd.Timestamp
    ) -> pd.DataFrame:
        """All matches for a player strictly before match_date."""
        data = self._player_data.get(player_id)
        if data is None or len(data) == 0:
            return pd.DataFrame()
        dates = self._player_dates[player_id]
        idx = np.searchsorted(dates, np.datetime64(match_date), side="left")
        if idx == 0:
            return pd.DataFrame()
        return data.iloc[:idx]

    def _time_window(
        self,
        history: pd.DataFrame,
        match_date: pd.Timestamp,
        days: int,
        surface: str | None = None,
    ) -> pd.DataFrame:
        if len(history) == 0:
            return history
        cutoff = match_date - pd.Timedelta(days=1)
        start = cutoff - pd.Timedelta(days=days)
        mask = (history["tourney_date"] >= start) & (history["tourney_date"] <= cutoff)
        if surface is not None:
            mask = mask & (history["surface"] == surface)
        return history.loc[mask]

    def rolling_features(
        self,
        player_id: str,
        match_date: pd.Timestamp,
        surface: str,
    ) -> dict:
        """Compute all rolling features for a player using only pre-match data."""
        history = self._get_history_before(player_id, match_date)
        features: dict = {}

        if len(history) == 0:
            for key in [
                "win_rate_12m", "win_rate_3m", "win_rate_clay_12m", "win_rate_clay_3m",
                "win_rate_surface_12m",
            ]:
                features[key] = np.nan
            for key in [
                "matches_12m", "matches_3m", "matches_30d", "clay_matches_12m",
                "titles_12m", "titles_clay_12m",
            ]:
                features[key] = 0
            return features

        clay_12m = self._time_window(history, match_date, FORM_WINDOW_12M_DAYS, "Clay")
        clay_3m = self._time_window(history, match_date, FORM_WINDOW_3M_DAYS, "Clay")
        all_12m = self._time_window(history, match_date, FORM_WINDOW_12M_DAYS)
        all_3m = self._time_window(history, match_date, FORM_WINDOW_3M_DAYS)
        recent = self._time_window(history, match_date, FATIGUE_WINDOW_DAYS)

        features["win_rate_12m"] = (
            all_12m["won"].mean() if len(all_12m) >= ROLLING_MIN_PERIODS else np.nan
        )
        features["win_rate_3m"] = (
            all_3m["won"].mean() if len(all_3m) >= ROLLING_MIN_PERIODS else np.nan
        )
        features["win_rate_clay_12m"] = (
            clay_12m["won"].mean() if len(clay_12m) >= ROLLING_MIN_PERIODS else np.nan
        )
        features["win_rate_clay_3m"] = (
            clay_3m["won"].mean() if len(clay_3m) >= ROLLING_MIN_PERIODS else np.nan
        )

        features["matches_12m"] = len(all_12m)
        features["matches_3m"] = len(all_3m)
        features["matches_30d"] = len(recent)
        features["clay_matches_12m"] = len(clay_12m)

        surface_12m = self._time_window(history, match_date, FORM_WINDOW_12M_DAYS, surface)
        features["win_rate_surface_12m"] = (
            surface_12m["won"].mean() if len(surface_12m) >= ROLLING_MIN_PERIODS else np.nan
        )

        if len(all_12m) > 0:
            finals = all_12m[all_12m["round"] == "F"]
            features["titles_12m"] = int(finals["won"].sum())
        else:
            features["titles_12m"] = 0

        if len(clay_12m) > 0:
            clay_finals = clay_12m[clay_12m["round"] == "F"]
            features["titles_clay_12m"] = int(clay_finals["won"].sum())
        else:
            features["titles_clay_12m"] = 0

        return features

    def serving_features(
        self,
        player_id: str,
        match_date: pd.Timestamp,
        surface: str | None = None,
        n_matches: int = ROLLING_CLAY_MATCHES,
        min_periods: int = ROLLING_MIN_PERIODS,
    ) -> dict:
        """Compute rolling serving stats from the last n completed matches."""
        history = self._get_history_before(player_id, match_date)
        if surface is not None and len(history) > 0:
            history = history.loc[history["surface"] == surface]

        history = history.tail(n_matches) if len(history) > 0 else history

        nan_result = {
            k: np.nan for k in [
                "first_serve_pct", "first_serve_win_pct", "second_serve_win_pct",
                "bp_saved_pct", "ace_rate", "df_rate",
            ]
        }
        if len(history) < min_periods:
            return nan_result

        svpt = history["p_svpt"].sum()
        first_in = history["p_1stIn"].sum()
        first_won = history["p_1stWon"].sum()
        second_won = history["p_2ndWon"].sum()
        bp_saved = history["p_bpSaved"].sum()
        bp_faced = history["p_bpFaced"].sum()
        aces = history["p_ace"].sum()
        dfs = history["p_df"].sum()

        if svpt == 0 or np.isnan(svpt):
            return nan_result

        second_attempts = svpt - first_in
        return {
            "first_serve_pct": first_in / svpt if svpt > 0 else np.nan,
            "first_serve_win_pct": first_won / first_in if first_in > 0 else np.nan,
            "second_serve_win_pct": (
                second_won / second_attempts if second_attempts > 0 else np.nan
            ),
            "bp_saved_pct": bp_saved / bp_faced if bp_faced > 0 else np.nan,
            "ace_rate": aces / svpt,
            "df_rate": dfs / svpt,
        }

    def h2h_features(
        self,
        player_id: str,
        opponent_id: str,
        match_date: pd.Timestamp,
    ) -> dict:
        """Head-to-head record with Laplace smoothing, derived from player index."""
        no_history = {
            "h2h_wins": 0,
            "h2h_total": 0,
            "h2h_win_pct": LAPLACE_PRIOR / (2 * LAPLACE_PRIOR),
        }
        history = self._get_history_before(player_id, match_date)
        if len(history) == 0:
            return no_history

        vs_opponent = history.loc[history["opponent_id"] == opponent_id]
        if len(vs_opponent) == 0:
            return no_history

        wins = int(vs_opponent["won"].sum())
        total = len(vs_opponent)
        smoothed = (wins + LAPLACE_PRIOR) / (total + 2 * LAPLACE_PRIOR)
        return {"h2h_wins": wins, "h2h_total": total, "h2h_win_pct": smoothed}

    def tournament_context(
        self,
        player_id: str,
        match_date: pd.Timestamp,
        round_name: str,
        seed: float,
        tourney_level: str,
    ) -> dict:
        """Round number, seeding, historical RG performance."""
        features: dict = {
            "round_num": ROUND_MAP.get(round_name, 0),
            "is_seeded": int(pd.notna(seed) and seed > 0),
            "seed": seed if pd.notna(seed) else 0.0,
            "is_grand_slam": int(tourney_level == "G"),
        }

        rg = self._rg_data.get(player_id)
        if rg is None or len(rg) == 0:
            features["rg_matches_career"] = 0
            features["rg_wins_career"] = 0
            features["rg_best_round"] = 0
            return features

        before = rg.loc[rg["tourney_date"] < match_date]
        features["rg_matches_career"] = len(before)
        features["rg_wins_career"] = int(before["won"].sum()) if len(before) > 0 else 0

        if len(before) > 0:
            rg_rounds = before["round"].map(ROUND_MAP)
            features["rg_best_round"] = int(rg_rounds.max()) if rg_rounds.notna().any() else 0
        else:
            features["rg_best_round"] = 0

        return features


def build_feature_matrix(
    matches: pd.DataFrame,
    elo_history: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full symmetric feature matrix from matches and Elo history.

    Each match produces two rows: one from each player's perspective,
    with the label flipped and all relative features negated.
    """
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "match_num"]
    ).reset_index(drop=True)

    melted = melt_matches(matches)
    builder = FeatureBuilder(melted)

    elo_dict = {}
    for row in elo_history.itertuples():
        elo_dict[row.Index] = {
            "w_elo_overall": row.w_elo_overall,
            "w_elo_surface": row.w_elo_surface,
            "l_elo_overall": row.l_elo_overall,
            "l_elo_surface": row.l_elo_surface,
        }

    records: list[dict] = []
    total = len(matches)

    for i, match in enumerate(matches.itertuples()):
        if i % 5000 == 0:
            print(f"  Features: {i:,}/{total:,} matches processed...")

        elo_row = elo_dict.get(i, {
            "w_elo_overall": np.nan, "w_elo_surface": np.nan,
            "l_elo_overall": np.nan, "l_elo_surface": np.nan,
        })
        surface = str(match.surface)
        match_date = match.tourney_date
        round_name = str(match.round)
        tourney_level = str(match.tourney_level)

        for is_winner in (True, False):
            if is_winner:
                pid, oid = str(match.winner_id), str(match.loser_id)
                p_rank, o_rank = match.winner_rank, match.loser_rank
                p_age, p_ht = match.winner_age, match.winner_ht
                p_seed = match.winner_seed
                p_elo, p_elo_s = elo_row["w_elo_overall"], elo_row["w_elo_surface"]
                o_elo, o_elo_s = elo_row["l_elo_overall"], elo_row["l_elo_surface"]
            else:
                pid, oid = str(match.loser_id), str(match.winner_id)
                p_rank, o_rank = match.loser_rank, match.winner_rank
                p_age, p_ht = match.loser_age, match.loser_ht
                p_seed = match.loser_seed
                p_elo, p_elo_s = elo_row["l_elo_overall"], elo_row["l_elo_surface"]
                o_elo, o_elo_s = elo_row["w_elo_overall"], elo_row["w_elo_surface"]

            row: dict = {
                "match_idx": i,
                "tourney_id": match.tourney_id,
                "tourney_date": match_date,
                "surface": surface,
                "round": round_name,
                "tourney_level": tourney_level,
                "player_id": pid,
                "opponent_id": oid,
                "elo_overall": p_elo,
                "elo_surface": p_elo_s,
                "opp_elo_overall": o_elo,
                "opp_elo_surface": o_elo_s,
                "elo_diff": p_elo - o_elo,
                "elo_surface_diff": p_elo_s - o_elo_s,
                "rank": p_rank,
                "opp_rank": o_rank,
                "age": p_age,
                "height": p_ht,
                "label": int(is_winner),
            }

            row["rank_diff"] = (
                (o_rank - p_rank)
                if pd.notna(p_rank) and pd.notna(o_rank)
                else np.nan
            )
            row["log_rank_ratio"] = (
                np.log(o_rank / p_rank)
                if pd.notna(p_rank) and pd.notna(o_rank) and p_rank > 0 and o_rank > 0
                else np.nan
            )

            row.update(builder.rolling_features(pid, match_date, surface))

            serve = builder.serving_features(pid, match_date, surface=None)
            for k, v in serve.items():
                row[f"serve_{k}"] = v

            serve_s = builder.serving_features(pid, match_date, surface=surface)
            for k, v in serve_s.items():
                row[f"serve_surface_{k}"] = v

            row.update(builder.h2h_features(pid, oid, match_date))
            row.update(builder.tournament_context(
                pid, match_date, round_name, p_seed, tourney_level,
            ))

            records.append(row)

    features = pd.DataFrame(records)
    print(f"  Built {len(features):,} feature rows from {total:,} matches.")
    return features


FEATURE_COLS = [
    "elo_diff", "elo_surface_diff",
    "rank_diff", "log_rank_ratio",
    "age", "height",
    "win_rate_12m", "win_rate_3m",
    "win_rate_clay_12m", "win_rate_clay_3m",
    "win_rate_surface_12m",
    "matches_12m", "matches_3m", "matches_30d",
    "clay_matches_12m",
    "titles_12m", "titles_clay_12m",
    "serve_first_serve_pct", "serve_first_serve_win_pct",
    "serve_second_serve_win_pct", "serve_bp_saved_pct",
    "serve_ace_rate", "serve_df_rate",
    "serve_surface_first_serve_pct", "serve_surface_first_serve_win_pct",
    "serve_surface_second_serve_win_pct", "serve_surface_bp_saved_pct",
    "serve_surface_ace_rate", "serve_surface_df_rate",
    "h2h_win_pct", "h2h_total",
    "round_num", "is_seeded", "seed", "is_grand_slam",
    "rg_matches_career", "rg_wins_career", "rg_best_round",
]


def run_features() -> None:
    """Load processed data, compute features, and save to parquet."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    elo_history = pd.read_parquet(DATA_ELO / "elo_history.parquet")
    print(f"Building features for {len(matches):,} matches...")

    features = build_feature_matrix(matches, elo_history)
    features.to_parquet(DATA_PROCESSED / "features.parquet", index=False)
    print(f"Saved {len(features):,} rows to {DATA_PROCESSED / 'features.parquet'}")

    non_null_pcts = features[FEATURE_COLS].notna().mean() * 100
    print("\nFeature coverage (% non-null):")
    for col in FEATURE_COLS:
        print(f"  {col:40s} {non_null_pcts[col]:5.1f}%")


if __name__ == "__main__":
    run_features()
