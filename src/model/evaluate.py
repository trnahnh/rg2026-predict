"""Model evaluation: metrics, calibration analysis, per-round breakdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": len(y_true),
    }
    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error — measures how well probabilities match outcomes."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def evaluate_model(
    model: xgb.Booster,
    calibrator: IsotonicRegression | None,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Evaluate model on a test set, print metrics for both raw and calibrated probs."""
    dtest = xgb.DMatrix(test_df[feature_cols])
    raw_probs = model.predict(dtest)
    y_true = test_df["label"].values

    raw_metrics = compute_metrics(y_true, raw_probs)
    print(
        f"  Raw       — Acc: {raw_metrics['accuracy']:.4f}  "
        f"LogLoss: {raw_metrics['log_loss']:.4f}  "
        f"Brier: {raw_metrics['brier_score']:.4f}  "
        f"AUC: {raw_metrics['auc_roc']:.4f}  "
        f"ECE: {raw_metrics['ece']:.4f}  "
        f"N: {raw_metrics['n']:,}"
    )

    if calibrator is not None:
        cal_probs = calibrator.predict(raw_probs)
        cal_metrics = compute_metrics(y_true, cal_probs)
        print(
            f"  Calibrated — Acc: {cal_metrics['accuracy']:.4f}  "
            f"LogLoss: {cal_metrics['log_loss']:.4f}  "
            f"Brier: {cal_metrics['brier_score']:.4f}  "
            f"AUC: {cal_metrics['auc_roc']:.4f}  "
            f"ECE: {cal_metrics['ece']:.4f}  "
            f"N: {cal_metrics['n']:,}"
        )
        return cal_metrics

    return raw_metrics


def evaluate_by_round(
    model: xgb.Booster,
    calibrator: IsotonicRegression | None,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Per-round accuracy breakdown."""
    dtest = xgb.DMatrix(test_df[feature_cols])
    raw_probs = model.predict(dtest)

    if calibrator is not None:
        probs = calibrator.predict(raw_probs)
    else:
        probs = raw_probs

    test_df = test_df.copy()
    test_df["prob"] = probs
    test_df["pred"] = (probs >= 0.5).astype(int)

    rows = []
    for rnd in ["R128", "R64", "R32", "R16", "QF", "SF", "F"]:
        subset = test_df[test_df["round"] == rnd]
        if len(subset) == 0:
            continue
        acc = accuracy_score(subset["label"], subset["pred"])
        ll = log_loss(subset["label"], subset["prob"]) if subset["label"].nunique() > 1 else np.nan
        rows.append({"round": rnd, "accuracy": acc, "log_loss": ll, "n": len(subset)})

    return pd.DataFrame(rows)


def evaluate_by_surface(
    model: xgb.Booster,
    calibrator: IsotonicRegression | None,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Per-surface evaluation breakdown."""
    dtest = xgb.DMatrix(test_df[feature_cols])
    raw_probs = model.predict(dtest)

    if calibrator is not None:
        probs = calibrator.predict(raw_probs)
    else:
        probs = raw_probs

    test_df = test_df.copy()
    test_df["prob"] = probs

    rows = []
    for surface in sorted(test_df["surface"].unique()):
        subset = test_df[test_df["surface"] == surface]
        y = subset["label"].values
        p = subset["prob"].values
        rows.append({"surface": surface, **compute_metrics(y, p)})

    return pd.DataFrame(rows)
