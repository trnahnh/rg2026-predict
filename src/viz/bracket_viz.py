"""Visualize tournament predictions: win probability bars, round heatmap."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

matplotlib.use("Agg")

from src.config import OUTPUTS_DIR


def plot_win_probability_bar(
    predictions: pd.DataFrame,
    top_n: int = 20,
    save_path: str | None = None,
) -> go.Figure:
    """Horizontal bar chart of tournament win probabilities."""
    df = predictions.head(top_n).copy()
    df = df.sort_values("win_prob", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = sns.color_palette("YlOrRd_r", n_colors=top_n)
    bars = ax.barh(df["name"], df["win_prob"] * 100, color=colors)

    for bar, prob in zip(bars, df["win_prob"]):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Win Probability (%)")
    ax.set_title("Roland Garros 2026 — Predicted Win Probability")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / "win_probability_bar.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = px.bar(
        df.sort_values("win_prob", ascending=False),
        x="win_prob",
        y="name",
        orientation="h",
        labels={"win_prob": "Win Probability", "name": "Player"},
        title="Roland Garros 2026 — Predicted Win Probability",
        text=df.sort_values("win_prob", ascending=False)["win_prob"].apply(
            lambda x: f"{x * 100:.1f}%"
        ),
    )
    plotly_fig.update_layout(yaxis={"categoryorder": "total ascending"})
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def plot_round_heatmap(
    predictions: pd.DataFrame,
    top_n: int = 32,
    save_path: str | None = None,
) -> go.Figure:
    """Heatmap of advancement probabilities by round."""
    df = predictions.head(top_n).copy()

    prob_cols = []
    round_labels = []
    for col, label in [
        ("semi_prob", "SF"),
        ("final_prob", "Final"),
        ("win_prob", "Winner"),
    ]:
        if col in df.columns:
            prob_cols.append(col)
            round_labels.append(label)

    if not prob_cols:
        print("  No round probability columns found, skipping heatmap.")
        return None

    matrix = df[prob_cols].values * 100
    names = df["name"].tolist()

    fig, ax = plt.subplots(figsize=(8, max(10, top_n * 0.35)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=round_labels,
        yticklabels=names,
        cbar_kws={"label": "Probability (%)"},
        ax=ax,
    )
    ax.set_title("Round Advancement Probabilities (%)")
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / "round_heatmap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=round_labels,
            y=names,
            colorscale="YlOrRd",
            text=np.round(matrix, 1).astype(str),
            texttemplate="%{text}%",
            hovertemplate="Player: %{y}<br>Round: %{x}<br>Prob: %{z:.1f}%<extra></extra>",
        )
    )
    plotly_fig.update_layout(
        title="Round Advancement Probabilities",
        yaxis={"autorange": "reversed"},
    )
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def plot_top_contenders_comparison(
    predictions: pd.DataFrame,
    top_n: int = 10,
    save_path: str | None = None,
) -> go.Figure:
    """Grouped bar chart comparing win/final/semi probabilities."""
    df = predictions.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["semi_prob"] * 100, width, label="Semifinal", color="#fdd49e")
    ax.bar(x, df["final_prob"] * 100, width, label="Final", color="#fc8d59")
    ax.bar(x + width, df["win_prob"] * 100, width, label="Winner", color="#d7301f")

    ax.set_xticks(x)
    ax.set_xticklabels(df["name"], rotation=45, ha="right")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Top Contenders — Stage Probabilities")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = save_path or str(OUTPUTS_DIR / "top_contenders.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plotly_fig = go.Figure()
    for col, label, color in [
        ("semi_prob", "Semifinal", "#fdd49e"),
        ("final_prob", "Final", "#fc8d59"),
        ("win_prob", "Winner", "#d7301f"),
    ]:
        plotly_fig.add_trace(
            go.Bar(
                x=df["name"],
                y=df[col] * 100,
                name=label,
                marker_color=color,
            )
        )
    plotly_fig.update_layout(
        barmode="group",
        title="Top Contenders — Stage Probabilities",
        yaxis_title="Probability (%)",
    )
    html_path = path.replace(".png", ".html")
    plotly_fig.write_html(html_path)

    print(f"  Saved {path} and {html_path}")
    return plotly_fig


def run_bracket_viz() -> None:
    """Generate all bracket/prediction visualizations."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    pred_path = OUTPUTS_DIR / "rg2026_predictions.parquet"
    if not pred_path.exists():
        print("No predictions found. Run `make predict` first.")
        return

    predictions = pd.read_parquet(pred_path)
    print(f"Loaded {len(predictions)} player predictions.")

    plot_win_probability_bar(predictions)
    plot_round_heatmap(predictions)
    plot_top_contenders_comparison(predictions)

    print("Bracket visualizations complete.")


if __name__ == "__main__":
    run_bracket_viz()
