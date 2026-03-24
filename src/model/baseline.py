"""Baseline models for comparison: rank-based, Elo logistic, full logistic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from src.config import DATA_PROCESSED, RANDOM_SEED
from src.data.features import FEATURE_COLS


def temporal_split(
    features: pd.DataFrame,
    train_end: int,
    val_years: tuple[int, ...],
    test_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split features by year — strictly chronological."""
    year = features["tourney_date"].dt.year
    train = features[year <= train_end]
    val = features[year.isin(val_years)]
    test = features[year == test_year]
    return train, val, test


def rank_baseline(df: pd.DataFrame) -> dict:
    """Predict the higher-ranked player always wins (rank_diff > 0 → label 1)."""
    valid = df.dropna(subset=["rank_diff"])
    preds = (valid["rank_diff"] > 0).astype(int).values
    y = valid["label"].values
    acc = accuracy_score(y, preds)
    return {"model": "rank_baseline", "accuracy": acc, "log_loss": np.nan, "n": len(valid)}


def elo_logistic_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Logistic regression on elo_diff only."""
    feat = ["elo_diff"]
    tr = train.dropna(subset=feat + ["label"])
    te = test.dropna(subset=feat + ["label"])

    model = LogisticRegression(random_state=RANDOM_SEED)
    model.fit(tr[feat], tr["label"])

    probs = model.predict_proba(te[feat])[:, 1]
    preds = model.predict(te[feat])
    return {
        "model": "elo_logistic",
        "accuracy": accuracy_score(te["label"], preds),
        "log_loss": log_loss(te["label"], probs),
        "n": len(te),
    }


def full_logistic_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Logistic regression on all features (impute NaN with 0 for sklearn)."""
    tr = train[feature_cols + ["label"]].copy()
    te = test[feature_cols + ["label"]].copy()

    tr[feature_cols] = tr[feature_cols].fillna(0)
    te[feature_cols] = te[feature_cols].fillna(0)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(tr[feature_cols], tr["label"])

    probs = model.predict_proba(te[feature_cols])[:, 1]
    preds = model.predict(te[feature_cols])
    return {
        "model": "full_logistic",
        "accuracy": accuracy_score(te["label"], preds),
        "log_loss": log_loss(te["label"], probs),
        "n": len(te),
    }


def run_baselines() -> None:
    """Run all baselines and print comparison table."""
    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    train, val, test = temporal_split(features, 2022, (2023, 2024), 2025)

    print(f"Split sizes — Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
    print()

    results = []
    results.append(rank_baseline(test))
    results.append(elo_logistic_baseline(train, test))
    results.append(full_logistic_baseline(train, test, FEATURE_COLS))

    print(f"{'Model':<20s} {'Accuracy':>10s} {'Log-Loss':>10s} {'N':>8s}")
    print("-" * 50)
    for r in results:
        ll = f"{r['log_loss']:.4f}" if not np.isnan(r["log_loss"]) else "N/A"
        print(f"{r['model']:<20s} {r['accuracy']:>10.4f} {ll:>10s} {r['n']:>8,}")


if __name__ == "__main__":
    run_baselines()
