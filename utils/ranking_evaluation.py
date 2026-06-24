from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.bias_baseline import BiasBaselineModel
from models.item_knn import ItemKNNModel
from models.pmf_model import PMFModel
from models.svd_model import SVDModel


MODEL_PREFIXES = {
    "BiasBaseline": "bias",
    "ItemKNN": "item_knn",
    "SVD": "svd",
    "PMF": "pmf",
}


def build_temporal_ranking_protocol(
    ratings: pd.DataFrame,
    positive_threshold: float = 4.0,
    min_prior_interactions: int = 20,
    min_target_item_support: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing columns: {sorted(missing)}")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("ratings must be unique by user and movie")
    if min_prior_interactions < 1 or min_target_item_support < 1:
        raise ValueError("minimum history and support must be positive")

    stable = ratings.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    )
    histories: list[pd.DataFrame] = []
    target_rows: list[dict[str, Any]] = []
    exclusion_counts = {
        "no_positive_target": 0,
        "insufficient_prior_history": 0,
        "target_item_below_min_support": 0,
    }
    same_timestamp_rows_excluded = 0
    later_rows_excluded = 0

    for user_id, group in stable.groupby("user_id", sort=True):
        positives = group.loc[group["rating"].ge(positive_threshold)]
        if positives.empty:
            exclusion_counts["no_positive_target"] += 1
            continue
        target_timestamp = int(positives["timestamp"].max())
        target = (
            positives.loc[positives["timestamp"].eq(target_timestamp)]
            .sort_values("movie_id", kind="mergesort")
            .iloc[0]
        )
        history = group.loc[group["timestamp"].lt(target_timestamp)].copy()
        same_timestamp_rows_excluded += int(
            group["timestamp"].eq(target_timestamp).sum()
        )
        later_rows_excluded += int(group["timestamp"].gt(target_timestamp).sum())
        if len(history) < min_prior_interactions:
            exclusion_counts["insufficient_prior_history"] += 1
            continue
        histories.append(history)
        target_rows.append(
            {
                "user_id": int(user_id),
                "movie_id": int(target["movie_id"]),
                "rating": float(target["rating"]),
                "timestamp": target_timestamp,
                "prior_history_count": int(len(history)),
            }
        )

    if not histories or not target_rows:
        raise ValueError("no users satisfy the temporal target/history requirements")
    ranking_train = pd.concat(histories, ignore_index=True)
    ranking_train = ranking_train.sort_values(
        ["user_id", "timestamp", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)
    targets = pd.DataFrame(target_rows)
    support = ranking_train.groupby("movie_id").size()
    targets["target_item_support"] = (
        targets["movie_id"].map(support).fillna(0).astype(int)
    )
    supported = targets["target_item_support"].ge(min_target_item_support)
    exclusion_counts["target_item_below_min_support"] = int((~supported).sum())
    targets = targets.loc[supported].sort_values("user_id", kind="mergesort")
    targets = targets.reset_index(drop=True)
    if targets.empty:
        raise ValueError("no target movies satisfy the ranking-training support minimum")

    target_pairs = set(
        map(tuple, targets[["user_id", "movie_id"]].to_numpy(dtype=np.int64))
    )
    training_pairs = set(
        map(
            tuple,
            ranking_train[["user_id", "movie_id"]].to_numpy(dtype=np.int64),
        )
    )
    if target_pairs & training_pairs:
        raise AssertionError("ranking targets leaked into ranking training")

    metadata = {
        "protocol": "next-positive recovery under temporal leave-one-positive-out",
        "positive_threshold": float(positive_threshold),
        "target_selection": (
            "latest rating >= threshold; movie_id ascending tie-break at latest timestamp"
        ),
        "history_rule": "timestamp < target_timestamp",
        "same_timestamp_and_later_interactions_excluded": True,
        "min_prior_interactions": int(min_prior_interactions),
        "min_target_item_support": int(min_target_item_support),
        "full_catalog_candidates": True,
        "sampled_negatives": False,
        "unknown_items_are_observed_negatives": False,
        "input_user_count": int(ratings["user_id"].nunique()),
        "history_qualified_user_count": int(len(target_rows)),
        "eligible_user_count": int(len(targets)),
        "ranking_training_user_count": int(ranking_train["user_id"].nunique()),
        "ranking_training_item_count": int(ranking_train["movie_id"].nunique()),
        "ranking_training_rows": int(len(ranking_train)),
        "same_timestamp_rows_excluded": int(same_timestamp_rows_excluded),
        "later_rows_excluded": int(later_rows_excluded),
        "exclusion_counts": exclusion_counts,
    }
    return ranking_train, targets, metadata


def candidate_movie_ids(
    supported_movie_ids: np.ndarray,
    history_movie_ids: np.ndarray,
    target_movie_id: int,
) -> np.ndarray:
    supported = np.asarray(supported_movie_ids, dtype=np.int64)
    history = np.asarray(history_movie_ids, dtype=np.int64)
    if supported.ndim != 1 or history.ndim != 1:
        raise ValueError("movie ID arrays must be one-dimensional")
    if np.unique(supported).size != supported.size:
        raise ValueError("supported_movie_ids must be unique")
    if not np.any(supported == int(target_movie_id)):
        raise ValueError("target movie is not supported by ranking training")
    if np.any(history == int(target_movie_id)):
        raise ValueError("target movie appears in ranking history")
    candidates = np.setdiff1d(supported, history, assume_unique=False)
    if candidates.size == 0 or int(target_movie_id) not in candidates:
        raise ValueError("target movie is absent from the candidate set")
    return candidates.astype(np.int32, copy=False)


def rank_target(
    candidate_movie_ids_array: np.ndarray,
    candidate_scores: np.ndarray,
    target_movie_id: int,
) -> int:
    movie_ids = np.asarray(candidate_movie_ids_array, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    if movie_ids.shape != scores.shape or movie_ids.ndim != 1:
        raise ValueError("candidate movie IDs and scores must be aligned vectors")
    if movie_ids.size == 0 or np.unique(movie_ids).size != movie_ids.size:
        raise ValueError("candidate movie IDs must be non-empty and unique")
    if not np.isfinite(scores).all():
        raise ValueError("candidate scores must be finite")
    if not np.any(movie_ids == int(target_movie_id)):
        raise ValueError("target movie is absent from candidates")
    order = np.lexsort((movie_ids, -scores))
    return int(np.flatnonzero(movie_ids[order] == int(target_movie_id))[0] + 1)


def single_target_metrics(target_rank: int, cutoff: int) -> dict[str, float | bool]:
    if target_rank <= 0 or cutoff <= 0:
        raise ValueError("target_rank and cutoff must be positive")
    hit = target_rank <= cutoff
    return {
        "hit": bool(hit),
        "ndcg": float(1.0 / np.log2(target_rank + 1)) if hit else 0.0,
        "mrr": float(1.0 / target_rank) if hit else 0.0,
    }


def aggregate_ranking_metrics(
    ranking_results: pd.DataFrame,
) -> dict[str, Any]:
    if ranking_results.empty:
        raise ValueError("ranking_results must not be empty")
    model_metrics: dict[str, dict[str, float | int]] = {}
    for model_name, prefix in MODEL_PREFIXES.items():
        rank_column = f"{prefix}_target_rank"
        if rank_column not in ranking_results:
            raise ValueError(f"ranking_results missing {rank_column}")
        ranks = ranking_results[rank_column].to_numpy(dtype=np.int64)
        metrics: dict[str, float | int] = {
            "eligible_user_count": int(len(ranks)),
            "mean_target_rank": float(np.mean(ranks)),
            "median_target_rank": float(np.median(ranks)),
        }
        for cutoff in (5, 10):
            hits = ranks <= cutoff
            metrics[f"HitRate@{cutoff}"] = float(np.mean(hits))
            metrics[f"NDCG@{cutoff}"] = float(
                np.mean(np.where(hits, 1.0 / np.log2(ranks + 1), 0.0))
            )
            metrics[f"MRR@{cutoff}"] = float(
                np.mean(np.where(hits, 1.0 / ranks, 0.0))
            )
        model_metrics[model_name] = metrics
    return {
        "protocol": "next-positive recovery under temporal leave-one-positive-out",
        "model_names": list(MODEL_PREFIXES),
        "eligible_user_count": int(len(ranking_results)),
        "candidate_policy": (
            "all ranking-training-supported items minus the user's strict-prefix history"
        ),
        "full_catalog_candidates": True,
        "sampled_negatives": False,
        "unknown_items_are_observed_negatives": False,
        "models": model_metrics,
    }


def _bias_scores(model: BiasBaselineModel, users: np.ndarray) -> np.ndarray:
    model._check_fitted()
    return (
        model.global_mean
        + model.user_bias[users, None]
        + model.item_bias[None, :]
    ).astype(np.float32)


def _svd_scores(model: SVDModel, users: np.ndarray) -> np.ndarray:
    model._check_fitted()
    scaled_users = model.user_factors[users] * model.singular_values
    return (
        scaled_users @ model.item_factors
        + model.user_means[users, None]
        + model.item_bias[None, :]
    ).astype(np.float32)


def _pmf_scores(model: PMFModel, users: np.ndarray) -> np.ndarray:
    model._check_fitted()
    return (
        model.global_mean
        + model.user_bias[users, None]
        + model.item_bias[None, :]
        + model.user_factors[users] @ model.item_factors.T
    ).astype(np.float32)


def evaluate_ranking_models(
    targets: pd.DataFrame,
    ranking_train: pd.DataFrame,
    movies: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    index_to_movie: np.ndarray,
    bias_model: BiasBaselineModel,
    item_knn_model: ItemKNNModel,
    svd_model: SVDModel,
    pmf_model: PMFModel,
    batch_size: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
        "prior_history_count",
    }
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"targets missing columns: {sorted(missing)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    supported_movie_ids = np.asarray(index_to_movie, dtype=np.int32)
    if supported_movie_ids.size != len(movie_to_index) or not np.all(
        supported_movie_ids[:-1] < supported_movie_ids[1:]
    ):
        raise ValueError("ranking movie mapping must be strictly sorted")
    if any(
        movie_to_index[int(movie_id)] != index
        for index, movie_id in enumerate(supported_movie_ids)
    ):
        raise ValueError("ranking movie mappings are inconsistent")
    histories = {
        int(user_id): group["movie_id"].to_numpy(dtype=np.int32)
        for user_id, group in ranking_train.groupby("user_id", sort=True)
    }
    movie_lookup = (
        movies[["movie_id", "title", "genres"]]
        .drop_duplicates("movie_id")
        .set_index("movie_id")
    )
    ordered_targets = targets.sort_values("user_id", kind="mergesort").reset_index(
        drop=True
    )
    rows: list[dict[str, Any]] = []

    for start in range(0, len(ordered_targets), batch_size):
        batch = ordered_targets.iloc[start : start + batch_size]
        raw_user_ids = batch["user_id"].to_numpy(dtype=np.int64)
        user_indices = np.asarray(
            [user_to_index[int(user_id)] for user_id in raw_user_ids],
            dtype=np.int32,
        )
        score_matrices = {
            "bias": _bias_scores(bias_model, user_indices),
            "item_knn": item_knn_model.predict_users(user_indices, clip=False),
            "svd": _svd_scores(svd_model, user_indices),
            "pmf": _pmf_scores(pmf_model, user_indices),
        }
        for batch_row, target in enumerate(batch.itertuples(index=False)):
            user_id = int(target.user_id)
            target_movie_id = int(target.movie_id)
            candidates = candidate_movie_ids(
                supported_movie_ids,
                histories[user_id],
                target_movie_id,
            )
            candidate_indices = np.searchsorted(
                supported_movie_ids, candidates
            ).astype(np.int32)
            movie = movie_lookup.loc[target_movie_id]
            result: dict[str, Any] = {
                "user_id": user_id,
                "target_movie_id": target_movie_id,
                "target_title": str(movie["title"]),
                "target_genres": str(movie["genres"]),
                "target_rating": float(target.rating),
                "target_timestamp": int(target.timestamp),
                "prior_history_count": int(target.prior_history_count),
                "candidate_count": int(candidates.size),
            }
            for prefix, all_scores in score_matrices.items():
                candidate_scores = all_scores[batch_row, candidate_indices].astype(
                    np.float64
                )
                target_rank = rank_target(
                    candidates, candidate_scores, target_movie_id
                )
                target_position = int(
                    np.flatnonzero(candidates == target_movie_id)[0]
                )
                result[f"{prefix}_target_rank"] = target_rank
                result[f"{prefix}_raw_target_score"] = float(
                    candidate_scores[target_position]
                )
                for cutoff in (5, 10):
                    values = single_target_metrics(target_rank, cutoff)
                    result[f"{prefix}_hit_at_{cutoff}"] = bool(values["hit"])
                    result[f"{prefix}_ndcg_at_{cutoff}"] = float(values["ndcg"])
                    result[f"{prefix}_mrr_at_{cutoff}"] = float(values["mrr"])
            rows.append(result)

    results = pd.DataFrame(rows).sort_values("user_id", kind="mergesort")
    results = results.reset_index(drop=True)
    return results, aggregate_ranking_metrics(results)


def plot_ranking_comparison(
    ranking_metrics: dict[str, Any],
    path: str,
) -> None:
    labels = list(MODEL_PREFIXES)
    metric_names = ["HitRate@10", "NDCG@10", "MRR@10"]
    values = np.asarray(
        [
            [ranking_metrics["models"][label][metric] for metric in metric_names]
            for label in labels
        ],
        dtype=np.float64,
    )
    positions = np.arange(len(metric_names))
    width = 0.18
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_index, label in enumerate(labels):
        ax.bar(
            positions + (model_index - 1.5) * width,
            values[model_index],
            width=width,
            label=label,
            color=colors[model_index],
        )
    ax.set_xticks(positions, metric_names)
    ax.set_ylim(0.0, max(0.05, float(values.max()) * 1.2))
    ax.set_ylabel("Metric value")
    ax.set_title("Full-catalog next-positive recovery")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
