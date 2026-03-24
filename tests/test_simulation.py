import numpy as np
import pytest

from src.simulate.bracket import (
    SEED_POSITIONS_128,
    Bracket,
    Player,
    build_seeded_bracket,
)
from src.simulate.montecarlo import run_monte_carlo, simulate_tournament


@pytest.fixture()
def players_128():
    return [Player(player_id=str(i), name=f"Player {i}") for i in range(128)]


@pytest.fixture()
def seeded_bracket(players_128):
    seeds = [Player(player_id=str(i), name=f"Seed {i + 1}", seed=i + 1) for i in range(32)]
    unseeded = [Player(player_id=str(i + 32), name=f"Player {i + 32}") for i in range(96)]
    rng = np.random.default_rng(42)
    return build_seeded_bracket(seeds, unseeded, rng)


@pytest.fixture()
def uniform_prob_matrix():
    return np.full((128, 128), 0.5)


@pytest.fixture()
def pid_to_idx():
    return {str(i): i for i in range(128)}


def test_bracket_has_128_players(seeded_bracket):
    assert len(seeded_bracket.players) == 128


def test_seed_placement(seeded_bracket):
    for seed_num, position in SEED_POSITIONS_128.items():
        player = seeded_bracket.players[position]
        assert player.seed == seed_num, (
            f"Seed {seed_num} should be at position {position}, "
            f"found {player.player_id} (seed={player.seed})"
        )


def test_seeds_opposite_halves(seeded_bracket):
    """Seeds 1 and 2 should be in opposite halves."""
    pos_1 = SEED_POSITIONS_128[1]
    pos_2 = SEED_POSITIONS_128[2]
    assert pos_1 < 64 and pos_2 >= 64


def test_no_duplicate_players(seeded_bracket):
    ids = [p.player_id for p in seeded_bracket.players]
    assert len(ids) == len(set(ids))


def test_simulate_produces_winner(seeded_bracket, uniform_prob_matrix, pid_to_idx):
    rng = np.random.default_rng(42)
    winner = simulate_tournament(seeded_bracket, uniform_prob_matrix, pid_to_idx, rng)
    assert winner is not None
    assert isinstance(winner, str)


def test_simulate_determinism(seeded_bracket, uniform_prob_matrix, pid_to_idx):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    w1 = simulate_tournament(seeded_bracket, uniform_prob_matrix, pid_to_idx, rng1)
    w2 = simulate_tournament(seeded_bracket, uniform_prob_matrix, pid_to_idx, rng2)
    assert w1 == w2


def test_monte_carlo_probabilities_sum(seeded_bracket, uniform_prob_matrix, pid_to_idx):
    results = run_monte_carlo(
        seeded_bracket,
        uniform_prob_matrix,
        pid_to_idx,
        n_simulations=1000,
        seed=42,
    )
    total_prob = results["win_prob"].sum()
    assert abs(total_prob - 1.0) < 0.01


def test_favorite_wins_more(seeded_bracket, pid_to_idx):
    """A strongly favored player should win more often than 50/50."""
    n = 128
    prob_matrix = np.full((n, n), 0.5)
    favorite_idx = pid_to_idx[seeded_bracket.players[0].player_id]
    for j in range(n):
        if j != favorite_idx:
            prob_matrix[favorite_idx][j] = 0.95
            prob_matrix[j][favorite_idx] = 0.05

    results = run_monte_carlo(
        seeded_bracket,
        prob_matrix,
        pid_to_idx,
        n_simulations=1000,
        seed=42,
    )

    fav_id = seeded_bracket.players[0].player_id
    fav_row = results[results["player_id"] == fav_id]
    assert len(fav_row) == 1
    assert fav_row.iloc[0]["win_prob"] > 0.3


def test_monte_carlo_determinism(seeded_bracket, uniform_prob_matrix, pid_to_idx):
    r1 = run_monte_carlo(
        seeded_bracket,
        uniform_prob_matrix,
        pid_to_idx,
        n_simulations=500,
        seed=42,
    )
    r2 = run_monte_carlo(
        seeded_bracket,
        uniform_prob_matrix,
        pid_to_idx,
        n_simulations=500,
        seed=42,
    )
    np.testing.assert_array_almost_equal(
        r1.sort_values("player_id")["win_prob"].values,
        r2.sort_values("player_id")["win_prob"].values,
    )


def test_bracket_round_structure():
    """128 players → 7 rounds (128→64→32→16→8→4→2→1)."""
    players = [Player(player_id=str(i)) for i in range(128)]
    bracket = Bracket(players=players)
    assert len(bracket.rounds[0]) == 128
