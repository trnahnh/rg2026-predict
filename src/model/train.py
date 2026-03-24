"""XGBoost training with Optuna HPO, expanding-window CV, and isotonic calibration."""

from __future__ import annotations

import joblib
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

from src.config import (
    DATA_PROCESSED,
    MODELS_DIR,
    OPTUNA_N_TRIALS,
    RANDOM_SEED,
    TRAIN_END,
    VAL_YEARS,
)
from src.data.features import FEATURE_COLS
from src.model.baseline import temporal_split
from src.model.evaluate import evaluate_model


def expanding_window_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    fold_years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022),
) -> float:
    """Expanding-window cross-validation: train on <year, validate on year."""
    scores = []
    year = train_df["tourney_date"].dt.year

    for val_year in fold_years:
        tr = train_df[year < val_year]
        va = train_df[year == val_year]
        if len(tr) < 100 or len(va) < 100:
            continue

        dtrain = xgb.DMatrix(tr[feature_cols], label=tr["label"], enable_categorical=False)
        dval = xgb.DMatrix(va[feature_cols], label=va["label"], enable_categorical=False)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        preds = model.predict(dval)
        scores.append(log_loss(va["label"], preds))

    return float(np.mean(scores)) if scores else float("inf")


def create_optuna_objective(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> callable:
    """Create an Optuna objective function for XGBoost HPO."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "seed": RANDOM_SEED,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }
        return expanding_window_cv(train_df, feature_cols, params)

    return objective


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
) -> xgb.Booster:
    """Train final XGBoost model on full training set with early stopping on val."""
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["label"])
    dval = xgb.DMatrix(val_df[feature_cols], label=val_df["label"])

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        **best_params,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=5000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )
    return model


def calibrate_model(
    model: xgb.Booster,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> IsotonicRegression:
    """Isotonic calibration on the validation set."""
    dval = xgb.DMatrix(val_df[feature_cols])
    raw_probs = model.predict(dval)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probs, val_df["label"].values)
    return calibrator


def run_training() -> None:
    """Full training pipeline: HPO → train → calibrate → evaluate → save."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    train, val, test = temporal_split(features, TRAIN_END, VAL_YEARS, 2025)
    print(f"Split sizes — Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    print(f"\nRunning Optuna HPO ({OPTUNA_N_TRIALS} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        create_optuna_objective(train, FEATURE_COLS),
        n_trials=OPTUNA_N_TRIALS,
    )
    best_params = study.best_params
    print(f"Best CV log-loss: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    print("\nTraining final model...")
    model = train_xgboost(train, val, FEATURE_COLS, best_params)

    print("\nCalibrating on validation set...")
    calibrator = calibrate_model(model, val, FEATURE_COLS)

    joblib.dump(model, MODELS_DIR / "xgb_final.joblib")
    joblib.dump(calibrator, MODELS_DIR / "calibrator.joblib")
    joblib.dump(best_params, MODELS_DIR / "best_params.joblib")
    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.joblib")
    print(f"Saved model artifacts to {MODELS_DIR}/")

    print("\n--- Test Set Evaluation ---")
    evaluate_model(model, calibrator, test, FEATURE_COLS)

    clay_test = test[test["surface"] == "Clay"]
    if len(clay_test) > 0:
        print("\n--- Clay-Only Test Evaluation ---")
        evaluate_model(model, calibrator, clay_test, FEATURE_COLS)


if __name__ == "__main__":
    run_training()
