"""Visualize model diagnostics: feature importance, calibration, backtest, Elo."""

from __future__ import annotations

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import xgboost as xgb

matplotlib.use("Agg")

from src.config import DATA_ELO, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR


def plot_feature_importance(
    model: xgb.Booster,
    feature_cols: list[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> go.Figure:
    """XGBoost gain-based feature importance."""
    importance = model.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame([{"feature": k, "gain": v} for k, v in importance.items()])
        .sort_values("gain", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    imp_sorted = imp_df.sort_values("gain", ascending=True)
    ax.barh(imp_sorted["feature"], imp_sorted["gain"], color=sns.color_palette("viridis", top_n))
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_n} Feature Importance (Gain)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / "feature_importance.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = px.bar(
        imp_df,
        x="gain",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Feature Importance (Gain)",
        labels={"gain": "Gain", "feature": "Feature"},
    )
    plotly_fig.update_layout(yaxis={"categoryorder": "total ascending"})
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_cal: np.ndarray,
    n_bins: int = 10,
    save_path: str | None = None,
) -> go.Figure:
    """Before/after calibration reliability diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")

    for probs, label, color in [
        (y_prob_raw, "Raw", "#fc8d59"),
        (y_prob_cal, "Calibrated", "#2166ac"),
    ]:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true = []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_means.append(probs[mask].mean())
                bin_true.append(y_true[mask].mean())
        ax.plot(bin_means, bin_true, "o-", label=label, color=color, markersize=8)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / "calibration_curve.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = go.Figure()
    plotly_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect",
            line={"dash": "dash", "color": "black"},
        )
    )
    for probs, label, color in [
        (y_prob_raw, "Raw", "#fc8d59"),
        (y_prob_cal, "Calibrated", "#2166ac"),
    ]:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bm, bt = [], []
        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bm.append(float(probs[mask].mean()))
                bt.append(float(y_true[mask].mean()))
        plotly_fig.add_trace(
            go.Scatter(
                x=bm,
                y=bt,
                mode="lines+markers",
                name=label,
                line={"color": color},
            )
        )

    plotly_fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Frequency",
    )
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def plot_backtest_results(
    backtest_df: pd.DataFrame,
    save_path: str | None = None,
) -> go.Figure:
    """Bar chart of actual winner's predicted rank per year."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [
        "#2166ac" if r <= 5 else "#fc8d59" if r <= 10 else "#d7301f"
        for r in backtest_df["predicted_rank"]
    ]
    ax.bar(backtest_df["year"].astype(str), backtest_df["predicted_rank"], color=colors)

    ax.axhline(y=5, color="green", linestyle="--", alpha=0.7, label="Top 5")
    ax.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="Top 10")

    for i, row in backtest_df.iterrows():
        ax.text(
            i,
            row["predicted_rank"] + 0.5,
            row["actual_winner"].split()[-1],
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Predicted Rank of Actual Winner")
    ax.set_title("Backtest: Where Did the Actual Winner Rank?")
    ax.legend()
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / "backtest_results.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = px.bar(
        backtest_df,
        x="year",
        y="predicted_rank",
        text="actual_winner",
        color="predicted_rank",
        color_continuous_scale="RdYlGn_r",
        title="Backtest: Where Did the Actual Winner Rank?",
        labels={"predicted_rank": "Predicted Rank", "year": "Year"},
    )
    plotly_fig.update_yaxes(autorange="reversed")
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def plot_elo_trajectories(
    elo_history: pd.DataFrame,
    player_ids: list[str],
    player_names: dict[str, str] | None = None,
    surface: str = "Clay",
    save_path: str | None = None,
) -> go.Figure:
    """Clay Elo rating over time for selected players."""
    surface_col = f"w_elo_{surface.lower()}"
    col = surface_col if surface_col in elo_history.columns else "w_elo_surface"

    fig, ax = plt.subplots(figsize=(14, 7))
    palette = sns.color_palette("tab10", n_colors=len(player_ids))

    for pid, color in zip(player_ids, palette):
        winner_rows = elo_history[elo_history["winner_id"] == pid].copy()
        if len(winner_rows) == 0:
            continue
        label = player_names.get(pid, pid) if player_names else pid
        ax.plot(winner_rows["tourney_date"], winner_rows[col], label=label, color=color, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{surface} Elo Rating")
    ax.set_title(f"{surface} Elo Trajectories")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / f"elo_trajectories_{surface.lower()}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = go.Figure()
    for pid in player_ids:
        winner_rows = elo_history[elo_history["winner_id"] == pid]
        if len(winner_rows) == 0:
            continue
        label = player_names.get(pid, pid) if player_names else pid
        plotly_fig.add_trace(
            go.Scatter(
                x=winner_rows["tourney_date"],
                y=winner_rows[col],
                mode="lines",
                name=label,
            )
        )

    plotly_fig.update_layout(
        title=f"{surface} Elo Trajectories",
        xaxis_title="Date",
        yaxis_title=f"{surface} Elo Rating",
    )
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def run_feature_viz() -> None:
    """Generate all model diagnostic visualizations."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgb_final.joblib"
    if not model_path.exists():
        print("No trained model found. Run `make train` first.")
        return

    model = joblib.load(model_path)
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
    calibrator = joblib.load(MODELS_DIR / "calibrator.joblib")

    print("Generating feature importance plot...")
    plot_feature_importance(model, feature_cols)

    features = pd.read_parquet(DATA_PROCESSED / "features.parquet")
    test = features[features["tourney_date"].dt.year == 2025]
    if len(test) > 0:
        print("Generating calibration curve...")
        dtest = xgb.DMatrix(test[feature_cols])
        raw_probs = model.predict(dtest)
        cal_probs = calibrator.predict(raw_probs)
        plot_calibration_curve(test["label"].values, raw_probs, cal_probs)

    backtest_path = OUTPUTS_DIR / "backtest_results.parquet"
    if backtest_path.exists():
        print("Generating backtest plot...")
        backtest_df = pd.read_parquet(backtest_path)
        plot_backtest_results(backtest_df)

    elo_path = DATA_ELO / "elo_history.parquet"
    if elo_path.exists():
        print("Generating Elo trajectory plot...")
        elo_history = pd.read_parquet(elo_path)
        predictions_path = OUTPUTS_DIR / "rg2026_predictions.parquet"
        if predictions_path.exists():
            predictions = pd.read_parquet(predictions_path)
            top_ids = predictions.head(10)["player_id"].tolist()
            name_map = dict(zip(predictions["player_id"], predictions["name"]))
        else:
            top_ids = (
                elo_history.groupby("winner_id")["w_elo_surface"].last().nlargest(10).index.tolist()
            )
            name_map = None
        plot_elo_trajectories(elo_history, top_ids, name_map)

    print("Feature/model visualizations complete.")


if __name__ == "__main__":
    run_feature_viz()
