"""Compute Elo ratings (overall + surface-specific) for every player at every match."""

from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    DATA_ELO,
    DATA_PROCESSED,
    ELO_INITIAL,
    ELO_K_ESTABLISHED,
    ELO_K_NEW,
    ELO_NEW_THRESHOLD,
)


@dataclass
class PlayerElo:
    overall: float = ELO_INITIAL
    clay: float = ELO_INITIAL
    hard: float = ELO_INITIAL
    grass: float = ELO_INITIAL
    match_count: int = 0
    surface_match_count: dict[str, int] = field(
        default_factory=lambda: {"Clay": 0, "Hard": 0, "Grass": 0}
    )


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(match_count: int) -> float:
    if match_count < ELO_NEW_THRESHOLD:
        return float(ELO_K_NEW)
    return float(ELO_K_ESTABLISHED)


def get_surface_elo(player: PlayerElo, surface: str) -> float:
    """Return surface Elo, falling back to overall if no matches on that surface."""
    if surface not in player.surface_match_count or player.surface_match_count[surface] == 0:
        return player.overall
    return getattr(player, surface.lower(), player.overall)


def update_ratings(
    winner: PlayerElo,
    loser: PlayerElo,
    surface: str,
) -> None:
    """Update Elo ratings in-place after a match result."""
    k_w = k_factor(winner.match_count)
    k_l = k_factor(loser.match_count)

    e_w = expected_score(winner.overall, loser.overall)
    winner.overall += k_w * (1.0 - e_w)
    loser.overall -= k_l * (1.0 - e_w)

    if surface in ("Clay", "Hard", "Grass"):
        w_surf = get_surface_elo(winner, surface)
        l_surf = get_surface_elo(loser, surface)
        e_surf = expected_score(w_surf, l_surf)

        setattr(winner, surface.lower(), w_surf + k_w * (1.0 - e_surf))
        setattr(loser, surface.lower(), l_surf - k_l * (1.0 - e_surf))

        winner.surface_match_count[surface] = winner.surface_match_count.get(surface, 0) + 1
        loser.surface_match_count[surface] = loser.surface_match_count.get(surface, 0) + 1

    winner.match_count += 1
    loser.match_count += 1


def compute_elo_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Process all matches chronologically, recording pre-match Elo for each row."""
    matches = matches.sort_values(
        ["tourney_date", "tourney_id", "match_num"]
    ).reset_index(drop=True)

    players: dict[str, PlayerElo] = {}
    records: list[dict] = []

    for row in matches.itertuples():
        wid = str(row.winner_id)
        lid = str(row.loser_id)
        surface = str(row.surface)

        if wid not in players:
            players[wid] = PlayerElo()
        if lid not in players:
            players[lid] = PlayerElo()

        w = players[wid]
        l = players[lid]

        records.append({
            "match_idx": row.Index,
            "tourney_date": row.tourney_date,
            "surface": surface,
            "winner_id": wid,
            "loser_id": lid,
            "w_elo_overall": w.overall,
            "w_elo_surface": get_surface_elo(w, surface),
            "l_elo_overall": l.overall,
            "l_elo_surface": get_surface_elo(l, surface),
        })

        update_ratings(w, l, surface)

    return pd.DataFrame(records)


def get_current_ratings(players: dict[str, PlayerElo]) -> pd.DataFrame:
    """Extract latest Elo rating for every player."""
    rows = []
    for pid, elo in players.items():
        rows.append({
            "player_id": pid,
            "elo_overall": elo.overall,
            "elo_clay": elo.clay if elo.surface_match_count.get("Clay", 0) > 0 else elo.overall,
            "elo_hard": elo.hard if elo.surface_match_count.get("Hard", 0) > 0 else elo.overall,
            "elo_grass": elo.grass if elo.surface_match_count.get("Grass", 0) > 0 else elo.overall,
            "match_count": elo.match_count,
        })
    return pd.DataFrame(rows).sort_values("elo_overall", ascending=False).reset_index(drop=True)


def run_elo() -> None:
    """Compute Elo history from processed matches and save to parquet."""
    DATA_ELO.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(DATA_PROCESSED / "matches.parquet")
    print(f"Computing Elo for {len(matches):,} matches...")

    elo_history = compute_elo_history(matches)
    elo_history.to_parquet(DATA_ELO / "elo_history.parquet", index=False)
    print(f"  Saved {len(elo_history):,} rows to {DATA_ELO / 'elo_history.parquet'}")

    print("Top 10 current Elo ratings:")
    players: dict[str, PlayerElo] = {}
    for row in matches.sort_values(["tourney_date", "tourney_id", "match_num"]).itertuples():
        wid, lid = str(row.winner_id), str(row.loser_id)
        if wid not in players:
            players[wid] = PlayerElo()
        if lid not in players:
            players[lid] = PlayerElo()
        update_ratings(players[wid], players[lid], str(row.surface))

    current = get_current_ratings(players)
    current.to_parquet(DATA_ELO / "current_ratings.parquet", index=False)

    player_names = pd.read_parquet(DATA_PROCESSED / "players.parquet")
    name_map = dict(zip(
        player_names["player_id"].astype(str),
        player_names["name_first"].fillna("") + " " + player_names["name_last"].fillna(""),
    ))
    top10 = current.head(10).copy()
    top10["name"] = top10["player_id"].map(name_map).fillna("Unknown")
    for _, row in top10.iterrows():
        print(f"  {row['name']:25s} Overall: {row['elo_overall']:.0f}  Clay: {row['elo_clay']:.0f}")


if __name__ == "__main__":
    run_elo()
