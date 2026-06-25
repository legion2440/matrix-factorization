from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


USER_ARTIFACT_PATTERN = re.compile(r"^user_(\d+)_.+")
REQUIRED_USER_ARTIFACT_SUFFIXES = (
    "recommendations.csv",
    "explanations.csv",
    "explanation.png",
    "ranking_case.csv",
    "ranking_case.png",
)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def cleanup_user_artifacts(
    reports_dir: str | Path,
    current_user_ids: set[int],
) -> list[Path]:
    reports_dir = Path(reports_dir)
    removed: list[Path] = []
    for path in sorted(reports_dir.glob("user_*")):
        match = USER_ARTIFACT_PATTERN.match(path.name)
        if match and int(match.group(1)) not in current_user_ids:
            path.unlink()
            removed.append(path)
    return removed


def prepare_pmf_convergence_payload(
    history: list[dict[str, Any]],
    *,
    selected_epoch: int,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    rows = []
    for row in history:
        train_rmse = float(row["train_rmse"])
        validation_rmse = float(row["validation_rmse"])
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_rmse": train_rmse,
                "validation_rmse": validation_rmse,
                "train_mse": train_rmse**2,
                "validation_mse": validation_rmse**2,
            }
        )
    return {
        "metric_source": "RMSE; MSE is computed as RMSE ** 2",
        "selected_epoch": int(selected_epoch),
        "early_stopping": {
            "patience": int(patience),
            "min_delta": float(min_delta),
            "epochs_run": len(rows),
            "triggered": len(rows) < 70,
        },
        "history": rows,
    }


def plot_pmf_convergence(
    payload: dict[str, Any],
    path: str | Path,
    *,
    metric: str,
) -> None:
    if metric not in {"rmse", "mse"}:
        raise ValueError("metric must be 'rmse' or 'mse'")
    history = payload["history"]
    epochs = [row["epoch"] for row in history]
    train = [row[f"train_{metric}"] for row in history]
    validation = [row[f"validation_{metric}"] for row in history]
    selected_epoch = int(payload["selected_epoch"])
    stopping = payload["early_stopping"]
    fig, ax = plt.subplots(figsize=(8, 5))
    label = metric.upper()
    ax.plot(epochs, train, marker="o", markersize=3, label=f"Train {label}")
    ax.plot(
        epochs,
        validation,
        marker="o",
        markersize=3,
        label=f"Validation {label}",
    )
    ax.axvline(
        selected_epoch,
        color="#e45756",
        linestyle="--",
        linewidth=1.5,
        label=f"Selected epoch {selected_epoch}",
    )
    if int(stopping["epochs_run"]) != selected_epoch:
        ax.axvline(
            int(stopping["epochs_run"]),
            color="#79706e",
            linestyle=":",
            linewidth=1.5,
            label=f"Stopped after epoch {stopping['epochs_run']}",
        )
    ax.set(
        xlabel="Epoch",
        ylabel=label,
        title=(
            f"PMF convergence ({label}; patience={stopping['patience']}, "
            f"min_delta={stopping['min_delta']:g})"
        ),
    )
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


def plot_model_metric_comparison(
    values_by_model: dict[str, float],
    path: str | Path,
    *,
    metric: str,
) -> None:
    if metric not in {"MSE", "RMSE"}:
        raise ValueError("metric must be MSE or RMSE")
    labels = ["BiasBaseline", "ItemKNN", "SVD", "PMF"]
    missing = set(labels) - set(values_by_model)
    if missing:
        raise ValueError(f"missing model values: {sorted(missing)}")
    values = [float(values_by_model[label]) for label in labels]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set(
        ylabel=f"Test {metric}",
        title=f"Model {metric} comparison",
        ylim=(0, max(values) * 1.2),
    )
    ax.bar_label(bars, fmt="%.4f")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def prepare_svd_rank_tuning(
    results: list[dict[str, Any]],
    *,
    selected_rank: int,
    selected_item_bias_regularization: float,
) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    required = {
        "n_factors",
        "item_bias_regularization",
        "validation_rmse",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SVD tuning results missing columns: {sorted(missing)}")
    selected = frame.loc[
        np.isclose(
            frame["item_bias_regularization"].astype(float),
            selected_item_bias_regularization,
            rtol=0.0,
            atol=0.0,
        )
    ].copy()
    if selected.empty:
        raise ValueError("selected SVD item-bias regularization is absent")
    selected["validation_mse"] = selected["validation_rmse"].astype(float) ** 2
    selected["selected"] = selected["n_factors"].astype(int).eq(selected_rank)
    if selected["selected"].sum() != 1:
        raise ValueError("selected SVD rank must appear exactly once")
    return selected.sort_values("n_factors", kind="mergesort").reset_index(
        drop=True
    )


def plot_svd_rank_tuning(
    tuning: pd.DataFrame,
    path: str | Path,
    *,
    metric: str,
) -> None:
    if metric not in {"rmse", "mse"}:
        raise ValueError("metric must be 'rmse' or 'mse'")
    value_column = f"validation_{metric}"
    selected = tuning.loc[tuning["selected"]].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        tuning["n_factors"],
        tuning[value_column],
        marker="o",
        color="#4c78a8",
    )
    ax.scatter(
        [selected["n_factors"]],
        [selected[value_column]],
        color="#e45756",
        s=80,
        zorder=3,
        label=f"Selected rank {int(selected['n_factors'])}",
    )
    ax.set(
        xlabel="Number of latent factors / rank",
        ylabel=f"Validation {metric.upper()}",
        title=f"SVD rank tuning curve ({metric.upper()})",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_user_comparison(comparison: pd.DataFrame, path: str | Path) -> None:
    view = comparison.copy()
    view["label"] = view["title"].fillna("").str.slice(0, 42)
    view = view.head(15).iloc[::-1]
    positions = np.arange(len(view))
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(view["svd_ranking_score"], positions, label="SVD", s=55)
    ax.scatter(view["pmf_ranking_score"], positions, label="PMF", s=55)
    ax.set_yticks(positions, view["label"])
    ax.set(xlabel="Raw ranking score", title="Recommendation score comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_top_recommendations(comparison: pd.DataFrame, path: str | Path) -> None:
    view = comparison.copy()
    view["best_score"] = view[
        ["svd_ranking_score", "pmf_ranking_score"]
    ].max(axis=1)
    view = view.nlargest(12, "best_score").sort_values("best_score")
    positions = np.arange(len(view))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        positions - width / 2,
        view["svd_ranking_score"],
        height=width,
        label="SVD",
    )
    ax.barh(
        positions + width / 2,
        view["pmf_ranking_score"],
        height=width,
        label="PMF",
    )
    ax.set_yticks(positions, view["title"].fillna("").str.slice(0, 42))
    ax.set(xlabel="Raw ranking score", title="Top recommendation scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
