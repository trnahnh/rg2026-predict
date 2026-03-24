import pandas as pd

from src.config import ELO_INITIAL, ELO_K_ESTABLISHED, ELO_K_NEW, ELO_NEW_THRESHOLD
from src.elo.engine import (
    PlayerElo,
    compute_elo_history,
    expected_score,
    get_surface_elo,
    k_factor,
    update_ratings,
)


def test_initial_rating():
    p = PlayerElo()
    assert p.overall == ELO_INITIAL
    assert p.clay == ELO_INITIAL
    assert p.match_count == 0


def test_expected_score_equal_ratings():
    assert expected_score(1500.0, 1500.0) == 0.5


def test_expected_score_stronger_player():
    e = expected_score(1700.0, 1500.0)
    assert e > 0.5
    assert e < 1.0


def test_expected_scores_sum_to_one():
    e_a = expected_score(1600.0, 1400.0)
    e_b = expected_score(1400.0, 1600.0)
    assert abs(e_a + e_b - 1.0) < 1e-10


def test_k_factor_new_player():
    assert k_factor(0) == ELO_K_NEW
    assert k_factor(ELO_NEW_THRESHOLD - 1) == ELO_K_NEW


def test_k_factor_established_player():
    assert k_factor(ELO_NEW_THRESHOLD) == ELO_K_ESTABLISHED
    assert k_factor(100) == ELO_K_ESTABLISHED


def test_winner_gains_loser_loses():
    w = PlayerElo()
    lo = PlayerElo()
    w_before, lo_before = w.overall, lo.overall
    update_ratings(w, lo, "Clay")
    assert w.overall > w_before
    assert lo.overall < lo_before


def test_elo_conservation_same_k():
    """When both players have same K, total Elo is conserved."""
    w = PlayerElo()
    lo = PlayerElo()
    total_before = w.overall + lo.overall
    update_ratings(w, lo, "Hard")
    total_after = w.overall + lo.overall
    assert abs(total_before - total_after) < 1e-10


def test_upset_produces_larger_change():
    """Lower-rated beating higher-rated causes bigger rating change than expected win."""
    favorite = PlayerElo(overall=1800.0)
    underdog = PlayerElo(overall=1400.0)
    normal_w = PlayerElo(overall=1800.0)
    normal_l = PlayerElo(overall=1400.0)
    update_ratings(normal_w, normal_l, "Hard")
    normal_gain = normal_w.overall - 1800.0

    update_ratings(underdog, favorite, "Hard")
    upset_gain = underdog.overall - 1400.0

    assert upset_gain > normal_gain


def test_surface_elo_independence():
    """A hard-court match should not change clay Elo."""
    p = PlayerElo()
    p.surface_match_count["Clay"] = 5
    p.clay = 1550.0
    clay_before = p.clay

    opponent = PlayerElo()
    update_ratings(p, opponent, "Hard")

    assert p.clay == clay_before


def test_surface_cold_start_uses_overall():
    """First match on a surface should use overall Elo, not the default 1500."""
    p = PlayerElo(overall=1700.0)
    assert get_surface_elo(p, "Clay") == 1700.0


def test_surface_elo_after_matches():
    p = PlayerElo()
    p.clay = 1600.0
    p.surface_match_count["Clay"] = 10
    assert get_surface_elo(p, "Clay") == 1600.0


def test_pre_match_recording(sample_matches):
    """Recorded Elo values should be BEFORE the update, not after."""
    history = compute_elo_history(sample_matches)

    first_row = history.iloc[0]
    assert first_row["w_elo_overall"] == ELO_INITIAL
    assert first_row["l_elo_overall"] == ELO_INITIAL


def test_compute_elo_history_row_count(sample_matches):
    history = compute_elo_history(sample_matches)
    assert len(history) == len(sample_matches)


def test_determinism(sample_matches):
    h1 = compute_elo_history(sample_matches)
    h2 = compute_elo_history(sample_matches)
    pd.testing.assert_frame_equal(h1, h2)


def test_match_count_increments():
    w = PlayerElo()
    lo = PlayerElo()
    assert w.match_count == 0
    update_ratings(w, lo, "Clay")
    assert w.match_count == 1
    assert lo.match_count == 1
    update_ratings(w, lo, "Hard")
    assert w.match_count == 2
