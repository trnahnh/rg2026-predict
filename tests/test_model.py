import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.config import RANDOM_SEED, TRAIN_END, VAL_YEARS
from src.data.features import FEATURE_COLS, build_feature_matrix
from src.elo.engine import compute_elo_history
from src.model.baseline import rank_baseline, temporal_split
from src.model.evaluate import compute_metrics, expected_calibration_error


@pytest.fixture()
def feature_matrix(sample_matches):
    elo_history = compute_elo_history(sample_matches)
    return build_feature_matrix(sample_matches, elo_history)


@pytest.fixture()
def large_features(feature_matrix):
    """Replicate the small fixture enough times across years for split testing."""
    dfs = []
    for i, year in enumerate(range(2018, 2026)):
        chunk = feature_matrix.copy()
        chunk["tourney_date"] = pd.Timestamp(f"{year}-06-01")
        chunk["match_idx"] = chunk["match_idx"] + i * 1000
        dfs.append(chunk)
    return pd.concat(dfs, ignore_index=True)


def test_temporal_split_no_overlap(large_features):
    train, val, test = temporal_split(large_features, TRAIN_END, VAL_YEARS, 2025)
    train_years = set(train["tourney_date"].dt.year)
    val_years = set(val["tourney_date"].dt.year)
    test_years = set(test["tourney_date"].dt.year)
    assert train_years.isdisjoint(val_years)
    assert train_years.isdisjoint(test_years)
    assert val_years.isdisjoint(test_years)


def test_temporal_split_chronological(large_features):
    train, val, test = temporal_split(large_features, TRAIN_END, VAL_YEARS, 2025)
    assert train["tourney_date"].max() < val["tourney_date"].min()
    assert val["tourney_date"].max() < test["tourney_date"].min()


def test_temporal_split_covers_all(large_features):
    train, val, test = temporal_split(large_features, TRAIN_END, VAL_YEARS, 2025)
    total = len(train) + len(val) + len(test)
    assert total == len(large_features)


def test_symmetric_rows_same_split(large_features):
    """Both rows of the same match must land in the same split."""
    train, val, test = temporal_split(large_features, TRAIN_END, VAL_YEARS, 2025)
    for split_df in [train, val, test]:
        for match_idx in split_df["match_idx"].unique():
            pair = split_df[split_df["match_idx"] == match_idx]
            assert len(pair) == 2


def test_rank_baseline_returns_accuracy(feature_matrix):
    result = rank_baseline(feature_matrix)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["n"] > 0


def test_compute_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0
    assert m["auc_roc"] == 1.0
    assert m["brier_score"] < 0.1


def test_compute_metrics_bounds():
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    m = compute_metrics(y_true, y_prob)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["log_loss"] > 0
    assert 0.0 <= m["brier_score"] <= 1.0
    assert 0.0 <= m["auc_roc"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0


def test_ece_perfect_calibration():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert expected_calibration_error(y_true, y_prob) < 0.01


def test_predictions_in_unit_interval(feature_matrix):
    """XGBoost raw predictions should be in [0, 1]."""
    if len(feature_matrix) < 4:
        pytest.skip("Not enough data")

    X = feature_matrix[FEATURE_COLS]
    y = feature_matrix["label"]
    dtrain = xgb.DMatrix(X, label=y)

    params = {"objective": "binary:logistic", "max_depth": 2, "seed": RANDOM_SEED}
    model = xgb.train(params, dtrain, num_boost_round=10)

    preds = model.predict(dtrain)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)


def test_model_save_load_roundtrip(feature_matrix, tmp_path):
    X = feature_matrix[FEATURE_COLS]
    y = feature_matrix["label"]
    dtrain = xgb.DMatrix(X, label=y)

    params = {"objective": "binary:logistic", "max_depth": 2, "seed": RANDOM_SEED}
    model = xgb.train(params, dtrain, num_boost_round=10)

    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)

    preds_orig = model.predict(dtrain)
    preds_loaded = loaded.predict(dtrain)
    np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
