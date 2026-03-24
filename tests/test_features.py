import numpy as np
import pandas as pd
import pytest

from src.config import LAPLACE_PRIOR, ROLLING_MIN_PERIODS, ROUND_MAP
from src.data.features import (
    FEATURE_COLS,
    FeatureBuilder,
    build_feature_matrix,
    melt_matches,
)
from src.elo.engine import compute_elo_history


@pytest.fixture()
def melted(sample_matches):
    return melt_matches(sample_matches)


@pytest.fixture()
def builder(melted):
    return FeatureBuilder(melted)


@pytest.fixture()
def elo_history(sample_matches):
    return compute_elo_history(sample_matches)


@pytest.fixture()
def feature_matrix(sample_matches, elo_history):
    return build_feature_matrix(sample_matches, elo_history)


def test_melt_doubles_rows(sample_matches):
    melted = melt_matches(sample_matches)
    assert len(melted) == 2 * len(sample_matches)


def test_melt_has_both_perspectives(sample_matches):
    melted = melt_matches(sample_matches)
    for _, match in sample_matches.iterrows():
        winner_rows = melted[
            (melted["player_id"] == match["winner_id"])
            & (melted["opponent_id"] == match["loser_id"])
            & (melted["tourney_date"] == match["tourney_date"])
            & (melted["match_num"] == match["match_num"])
        ]
        loser_rows = melted[
            (melted["player_id"] == match["loser_id"])
            & (melted["opponent_id"] == match["winner_id"])
            & (melted["tourney_date"] == match["tourney_date"])
            & (melted["match_num"] == match["match_num"])
        ]
        assert len(winner_rows) == 1
        assert len(loser_rows) == 1
        assert winner_rows.iloc[0]["won"] == 1
        assert loser_rows.iloc[0]["won"] == 0


def test_melt_serve_stats_correct(sample_matches):
    melted = melt_matches(sample_matches)
    first_match = sample_matches.iloc[0]
    winner_row = melted[
        (melted["player_id"] == first_match["winner_id"])
        & (melted["match_num"] == first_match["match_num"])
        & (melted["tourney_date"] == first_match["tourney_date"])
    ].iloc[0]
    assert winner_row["p_ace"] == first_match["w_ace"]
    assert winner_row["o_ace"] == first_match["l_ace"]


def test_no_future_leakage(builder, sample_matches):
    """Rolling features must not see data from the match date or later."""
    first_date = sample_matches["tourney_date"].min()
    player_id = str(sample_matches.iloc[0]["winner_id"])

    features = builder.rolling_features(player_id, first_date, "Clay")
    assert features["matches_12m"] == 0
    assert features["matches_3m"] == 0
    assert features["matches_30d"] == 0
    assert np.isnan(features["win_rate_12m"])


def test_rolling_uses_strict_before(builder, sample_matches):
    """Data from the same date should NOT be included in rolling features."""
    second_match = sample_matches.iloc[1]
    player_id = str(second_match["winner_id"])
    match_date = second_match["tourney_date"]

    features = builder.rolling_features(player_id, match_date, "Clay")
    assert features["matches_12m"] == 0


def test_rolling_sees_past_data(builder, sample_matches):
    """Third match should see data from the first two matches."""
    third_match = sample_matches.iloc[2]
    player_a = str(third_match["winner_id"])
    match_date = third_match["tourney_date"]

    features = builder.rolling_features(player_a, match_date, "Clay")
    assert features["matches_12m"] >= 1


def test_min_periods_returns_nan(builder, sample_matches):
    """Win rates should be NaN when fewer than ROLLING_MIN_PERIODS matches exist."""
    third_match = sample_matches.iloc[2]
    player_id = str(third_match["winner_id"])
    match_date = third_match["tourney_date"]

    features = builder.rolling_features(player_id, match_date, "Clay")
    if features["matches_12m"] < ROLLING_MIN_PERIODS:
        assert np.isnan(features["win_rate_12m"])


def test_h2h_no_history_returns_prior(builder):
    """No prior meetings → Laplace prior gives 0.5."""
    h2h = builder.h2h_features("999999", "888888", pd.Timestamp("2025-01-01"))
    assert h2h["h2h_wins"] == 0
    assert h2h["h2h_total"] == 0
    assert h2h["h2h_win_pct"] == LAPLACE_PRIOR / (2 * LAPLACE_PRIOR)


def test_h2h_no_future_leakage(builder, sample_matches):
    first_date = sample_matches["tourney_date"].min()
    p1 = str(sample_matches.iloc[0]["winner_id"])
    p2 = str(sample_matches.iloc[0]["loser_id"])

    h2h = builder.h2h_features(p1, p2, first_date)
    assert h2h["h2h_total"] == 0


def test_h2h_sees_past(builder, sample_matches):
    """After match 1, H2H should reflect that result for later matches."""
    third_match = sample_matches.iloc[2]
    p1 = str(sample_matches.iloc[0]["winner_id"])
    p2 = str(sample_matches.iloc[0]["loser_id"])
    match_date = third_match["tourney_date"]

    h2h = builder.h2h_features(p1, p2, match_date)
    assert h2h["h2h_total"] >= 1


def test_serving_features_bounds(builder, sample_matches):
    """Serving percentages should be in [0, 1] when not NaN."""
    last_match = sample_matches.iloc[-1]
    player_id = str(last_match["winner_id"])
    match_date = last_match["tourney_date"]

    serve = builder.serving_features(player_id, match_date, surface=None)
    for key in ["first_serve_pct", "first_serve_win_pct", "second_serve_win_pct",
                "bp_saved_pct", "ace_rate", "df_rate"]:
        val = serve[key]
        if not np.isnan(val):
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_serving_no_future_leakage(builder, sample_matches):
    first_date = sample_matches["tourney_date"].min()
    player_id = str(sample_matches.iloc[0]["winner_id"])

    serve = builder.serving_features(player_id, first_date)
    for val in serve.values():
        assert np.isnan(val)


def test_tournament_context_round_mapping(builder):
    ctx = builder.tournament_context("100001", pd.Timestamp("2030-01-01"), "QF", 1.0, "G")
    assert ctx["round_num"] == ROUND_MAP["QF"]
    assert ctx["is_seeded"] == 1
    assert ctx["is_grand_slam"] == 1


def test_tournament_context_unseeded(builder):
    ctx = builder.tournament_context(
        "100001", pd.Timestamp("2030-01-01"), "R128", float("nan"), "A",
    )
    assert ctx["is_seeded"] == 0
    assert ctx["seed"] == 0.0
    assert ctx["is_grand_slam"] == 0


def test_rg_context_no_future(builder, sample_matches):
    first_date = sample_matches["tourney_date"].min()
    player_id = str(sample_matches.iloc[0]["winner_id"])

    ctx = builder.tournament_context(player_id, first_date, "R128", 1.0, "G")
    assert ctx["rg_matches_career"] == 0


def test_feature_matrix_symmetry(feature_matrix):
    """Each match should produce exactly 2 rows with opposite labels."""
    for match_idx in feature_matrix["match_idx"].unique():
        pair = feature_matrix[feature_matrix["match_idx"] == match_idx]
        assert len(pair) == 2
        labels = sorted(pair["label"].tolist())
        assert labels == [0, 1]


def test_feature_matrix_elo_diff_antisymmetric(feature_matrix):
    """Elo diff should be negated between the two rows of each match."""
    for match_idx in feature_matrix["match_idx"].unique():
        pair = feature_matrix[feature_matrix["match_idx"] == match_idx].sort_values("label")
        loser_row = pair.iloc[0]
        winner_row = pair.iloc[1]
        assert abs(winner_row["elo_diff"] + loser_row["elo_diff"]) < 1e-10


def test_feature_matrix_rank_diff_antisymmetric(feature_matrix):
    for match_idx in feature_matrix["match_idx"].unique():
        pair = feature_matrix[feature_matrix["match_idx"] == match_idx].sort_values("label")
        loser_row = pair.iloc[0]
        winner_row = pair.iloc[1]
        if pd.notna(winner_row["rank_diff"]) and pd.notna(loser_row["rank_diff"]):
            assert abs(winner_row["rank_diff"] + loser_row["rank_diff"]) < 1e-10


def test_feature_matrix_has_all_feature_cols(feature_matrix):
    for col in FEATURE_COLS:
        assert col in feature_matrix.columns, f"Missing feature column: {col}"


def test_feature_matrix_row_count(sample_matches, feature_matrix):
    assert len(feature_matrix) == 2 * len(sample_matches)


def test_determinism(sample_matches, elo_history):
    f1 = build_feature_matrix(sample_matches, elo_history)
    f2 = build_feature_matrix(sample_matches, elo_history)
    pd.testing.assert_frame_equal(f1, f2)


def test_label_distribution(feature_matrix):
    """Symmetric enforcement means exactly 50% label=1."""
    assert feature_matrix["label"].mean() == 0.5
