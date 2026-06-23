from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def plot_convergence(history: list[dict[str, Any]], path: str | Path) -> None:
    epochs = [row["epoch"] for row in history]
    train = [row["train_rmse"] for row in history]
    validation = [row["validation_rmse"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", markersize=3, label="Train RMSE")
    ax.plot(epochs, validation, marker="o", markersize=3, label="Validation RMSE")
    ax.set(xlabel="Epoch", ylabel="RMSE", title="PMF convergence")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_predicted_vs_actual(
    actual: np.ndarray,
    svd_predictions: np.ndarray,
    pmf_predictions: np.ndarray,
    path: str | Path,
    random_state: int = 42,
) -> None:
    rng = np.random.default_rng(random_state)
    sample_size = min(20_000, len(actual))
    selected = rng.choice(len(actual), size=sample_size, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, predicted, name in zip(
        axes, (svd_predictions, pmf_predictions), ("SVD", "PMF")
    ):
        density = ax.hexbin(
            actual[selected],
            predicted[selected],
            gridsize=35,
            mincnt=1,
            cmap="viridis",
        )
        ax.plot([1, 5], [1, 5], "--", color="tomato", linewidth=1)
        ax.set(title=name, xlabel="Actual rating", ylabel="Predicted rating")
        fig.colorbar(density, ax=ax, label="Count")
    fig.suptitle("Predicted vs actual test ratings")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rmse_comparison(svd_rmse: float, pmf_rmse: float, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["SVD", "PMF"], [svd_rmse, pmf_rmse], color=["#4c78a8", "#f58518"])
    ax.set(ylabel="Test RMSE", title="Model RMSE comparison", ylim=(0, max(svd_rmse, pmf_rmse) * 1.2))
    ax.bar_label(bars, fmt="%.4f")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_user_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    view = comparison.copy()
    view["label"] = view["title"].fillna("").str.slice(0, 42)
    view = view.head(15).iloc[::-1]
    positions = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(view["svd_predicted_rating"], positions, label="SVD", s=55)
    ax.scatter(view["pmf_predicted_rating"], positions, label="PMF", s=55)
    ax.set_yticks(positions, view["label"])
    ax.set(xlabel="Predicted rating", title="Recommendation score comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_recommendations(comparison: pd.DataFrame, path: str | Path) -> None:
    view = comparison.copy()
    view["best_score"] = view[
        ["svd_predicted_rating", "pmf_predicted_rating"]
    ].max(axis=1)
    view = view.nlargest(12, "best_score").sort_values("best_score")
    positions = np.arange(len(view))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        positions - width / 2,
        view["svd_predicted_rating"],
        height=width,
        label="SVD",
    )
    ax.barh(
        positions + width / 2,
        view["pmf_predicted_rating"],
        height=width,
        label="PMF",
    )
    ax.set_yticks(positions, view["title"].fillna("").str.slice(0, 42))
    ax.set(xlabel="Predicted rating", title="Top recommendation scores", xlim=(1, 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

