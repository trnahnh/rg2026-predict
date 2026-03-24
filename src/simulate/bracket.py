"""Grand Slam bracket construction with proper 32-seed placement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED_POSITIONS_128 = {
    1: 0,
    2: 127,
    3: 64,
    4: 63,
    5: 32,
    6: 95,
    7: 96,
    8: 31,
    9: 16,
    10: 111,
    11: 48,
    12: 79,
    13: 80,
    14: 47,
    15: 112,
    16: 15,
    17: 8,
    18: 119,
    19: 56,
    20: 71,
    21: 40,
    22: 87,
    23: 104,
    24: 23,
    25: 24,
    26: 103,
    27: 88,
    28: 39,
    29: 72,
    30: 55,
    31: 120,
    32: 7,
}


@dataclass
class Player:
    player_id: str
    name: str = ""
    seed: int | None = None


@dataclass
class Bracket:
    players: list[Player]
    size: int = 128
    rounds: list[list[str | None]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rounds = []
        current = [p.player_id for p in self.players]
        self.rounds.append(current)


def build_seeded_bracket(
    seeds: list[Player],
    unseeded: list[Player],
    rng: np.random.Generator,
) -> Bracket:
    """Build a 128-player bracket with proper Grand Slam seeding.

    Seeds 1-32 placed in fixed positions per SEED_POSITIONS_128.
    Remaining 96 slots filled randomly from unseeded players.
    """
    draw = [None] * 128

    for player in seeds[:32]:
        if player.seed is not None and player.seed in SEED_POSITIONS_128:
            draw[SEED_POSITIONS_128[player.seed]] = player

    empty_slots = [i for i, p in enumerate(draw) if p is None]
    rng.shuffle(empty_slots)

    fill_players = list(unseeded)
    rng.shuffle(fill_players)

    for slot, player in zip(empty_slots, fill_players):
        draw[slot] = player

    remaining_empty = [i for i, p in enumerate(draw) if p is None]
    if remaining_empty:
        for slot in remaining_empty:
            draw[slot] = Player(player_id=f"BYE_{slot}", name="BYE")

    return Bracket(players=draw)


def build_bracket_from_draw(draw_df: pd.DataFrame) -> Bracket:
    """Build bracket from an actual draw CSV with columns: position, player_id, name, seed."""
    draw_df = draw_df.sort_values("position").reset_index(drop=True)
    players = []
    for row in draw_df.itertuples():
        seed = int(row.seed) if pd.notna(row.seed) else None
        players.append(
            Player(
                player_id=str(row.player_id),
                name=str(row.name),
                seed=seed,
            )
        )
    return Bracket(players=players)


def get_rg_entrants(
    matches: pd.DataFrame,
    elo_current: pd.DataFrame,
    n_players: int = 128,
) -> tuple[list[Player], list[Player]]:
    """Select likely RG entrants based on ranking."""
    rankings = matches[["winner_id", "winner_rank", "winner_name"]].rename(
        columns={"winner_id": "player_id", "winner_rank": "rank", "winner_name": "name"}
    )
    rankings = rankings.dropna(subset=["rank"])
    latest = rankings.groupby("player_id").last().reset_index()
    latest = latest.sort_values("rank").drop_duplicates(subset="player_id")

    if elo_current is not None and len(elo_current) > 0:
        elo_map = dict(zip(elo_current["player_id"].astype(str), elo_current["elo_overall"]))
        latest["elo"] = latest["player_id"].astype(str).map(elo_map)

    candidates = latest.head(n_players * 2)
    selected = candidates.head(n_players)

    seeds = []
    unseeded = []
    for i, row in enumerate(selected.itertuples()):
        pid = str(row.player_id)
        name = str(row.name) if hasattr(row, "name") else pid
        if i < 32:
            seeds.append(Player(player_id=pid, name=name, seed=i + 1))
        else:
            unseeded.append(Player(player_id=pid, name=name))

    return seeds, unseeded
