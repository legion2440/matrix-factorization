from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.bias_baseline import BiasBaselineModel
from models.item_knn import ItemKNNModel
from models.pmf_model import PMFModel
from models.svd_model import SVDModel
from scripts.download_data import download_movielens
from utils.artifacts import (
    cleanup_user_artifacts,
    plot_model_metric_comparison,
    plot_pmf_convergence,
    plot_predicted_vs_actual,
    plot_svd_rank_tuning,
    plot_top_recommendations,
    plot_user_comparison,
    prepare_pmf_convergence_payload,
    prepare_svd_rank_tuning,
    save_json,
)
from utils.data_loader import load_movielens, validate_movielens
from utils.matrix_creation import (
    build_normalized_matrix,
    create_mappings,
    save_mappings,
    save_normalized_matrix_csv,
)
from utils.metrics import mse, rmse
from utils.interpretability import (
    build_local_pmf_explanations,
    build_pmf_factor_genre_profiles,
    build_pmf_factor_interpretation,
    build_pmf_movie_similarities,
    build_ranking_case_explanation,
    plot_pmf_latent_factor_heatmap,
    plot_ranking_case,
    plot_user_explanation,
    select_evaluation_users,
)
from utils.ranking_evaluation import (
    build_temporal_ranking_protocol,
    evaluate_ranking_models,
    plot_ranking_comparison,
)
from utils.recommendation import (
    PMFRecommendationModel,
    SVDRecommendationModel,
    compare_recommendations,
)
from utils.split import deterministic_user_split


RANDOM_STATE = 42
BIAS_BASELINE_REGULARIZATION_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
BIAS_BASELINE_ITERATIONS = 20
ITEM_KNN_K_GRID = [20, 40, 80]
ITEM_KNN_SHRINKAGE_GRID = [10.0, 50.0, 100.0]
ITEM_KNN_MIN_COMMON = 3
SVD_GRID = [5, 10, 20, 40, 60]
SVD_ITEM_BIAS_REGULARIZATION_GRID = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
PMF_FACTOR_GRID = [96, 112, 128]
PMF_FACTOR_REGULARIZATION_GRID = [0.05, 0.06, 0.07]
PMF_TUNING_EPOCHS = 70
PMF_TUNING_PATIENCE = 8
PMF_TUNING_MIN_DELTA = 5e-5
PMF_GRID = [
    {
        "n_factors": n_factors,
        "learning_rate": 0.006,
        "factor_regularization": factor_regularization,
        "bias_regularization": 0.02,
    }
    for n_factors in PMF_FACTOR_GRID
    for factor_regularization in PMF_FACTOR_REGULARIZATION_GRID
]


def _indexed(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        ratings["user_id"].map(user_to_index).to_numpy(np.int32),
        ratings["movie_id"].map(movie_to_index).to_numpy(np.int32),
        ratings["rating"].to_numpy(np.float32),
    )


def _select_showcase_users(train: pd.DataFrame) -> list[dict[str, int | float]]:
    counts = train.groupby("user_id").size().rename("train_ratings").reset_index()
    selected: list[dict[str, int | float]] = []
    used: set[int] = set()
    for quantile in (0.25, 0.5, 0.75):
        target = float(counts["train_ratings"].quantile(quantile))
        candidates = counts.loc[~counts["user_id"].isin(used)].copy()
        candidates["distance"] = (candidates["train_ratings"] - target).abs()
        row = candidates.sort_values(
            ["distance", "train_ratings", "user_id"], kind="mergesort"
        ).iloc[0]
        user_id = int(row["user_id"])
        used.add(user_id)
        selected.append(
            {
                "quantile": quantile,
                "target_train_count": target,
                "user_id": user_id,
                "train_ratings": int(row["train_ratings"]),
            }
        )
    return selected


def _tune_svd(
    train_matrix,
    train_means: np.ndarray,
    validation: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> tuple[dict[str, int | float], list[dict[str, int | float]]]:
    max_factors = min(max(SVD_GRID), min(train_matrix.shape) - 1)
    grid = [value for value in SVD_GRID if value <= max_factors]
    users, movies, actual = _indexed(validation, user_to_index, movie_to_index)
    results: list[dict[str, int | float]] = []
    for bias_regularization in SVD_ITEM_BIAS_REGULARIZATION_GRID:
        model = SVDModel(
            n_factors=max(grid),
            item_bias_regularization=bias_regularization,
            random_state=RANDOM_STATE,
        ).fit(train_matrix, train_means)
        for factors in grid:
            predicted = model.predict_pairs(users, movies, n_factors=factors)
            score = rmse(actual, predicted)
            results.append(
                {
                    "n_factors": factors,
                    "item_bias_regularization": bias_regularization,
                    "validation_mse": score**2,
                    "validation_rmse": score,
                }
            )
            print(
                f"SVD factors={factors}, item_bias_reg={bias_regularization:g}: "
                f"validation RMSE={score:.6f}"
            )
    best = min(
        results,
        key=lambda row: (
            row["validation_rmse"],
            row["n_factors"],
            row["item_bias_regularization"],
        ),
    )
    return best, results


def _tune_bias_baseline(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    train_users, train_items, train_ratings = train_arrays
    validation_users, validation_items, validation_ratings = validation_arrays
    results: list[dict[str, float | int]] = []
    for regularization in BIAS_BASELINE_REGULARIZATION_GRID:
        model = BiasBaselineModel(
            n_users=n_users,
            n_items=n_items,
            user_regularization=regularization,
            item_regularization=regularization,
            n_iterations=BIAS_BASELINE_ITERATIONS,
            random_state=RANDOM_STATE,
        ).fit(train_users, train_items, train_ratings)
        predictions = model.predict_pairs(validation_users, validation_items, clip=True)
        validation_mse = mse(validation_ratings, predictions)
        validation_rmse = rmse(validation_ratings, predictions)
        results.append(
            {
                "user_regularization": float(regularization),
                "item_regularization": float(regularization),
                "n_iterations": BIAS_BASELINE_ITERATIONS,
                "validation_mse": float(validation_mse),
                "validation_rmse": float(validation_rmse),
            }
        )
        print(
            "Bias baseline "
            f"bias_reg={regularization:g}: validation RMSE={validation_rmse:.6f}"
        )
    best = min(
        results,
        key=lambda row: (
            row["validation_rmse"],
            row["user_regularization"],
            row["item_regularization"],
        ),
    )
    return best, results


def _fit_final_bias_baseline(
    combined_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
    best_result: dict[str, float | int],
) -> BiasBaselineModel:
    model = BiasBaselineModel(
        n_users=n_users,
        n_items=n_items,
        user_regularization=float(best_result["user_regularization"]),
        item_regularization=float(best_result["item_regularization"]),
        n_iterations=int(best_result["n_iterations"]),
        random_state=RANDOM_STATE,
    )
    return model.fit(*combined_arrays)


def _tune_item_knn(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
    bias_best_result: dict[str, float | int],
    item_ids: np.ndarray,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    validation_users, validation_items, validation_ratings = validation_arrays
    results: list[dict[str, float | int]] = []
    for shrinkage in ITEM_KNN_SHRINKAGE_GRID:
        model = ItemKNNModel(
            n_users=n_users,
            n_items=n_items,
            n_neighbors=max(ITEM_KNN_K_GRID),
            shrinkage=shrinkage,
            min_common=ITEM_KNN_MIN_COMMON,
            baseline_user_regularization=float(
                bias_best_result["user_regularization"]
            ),
            baseline_item_regularization=float(
                bias_best_result["item_regularization"]
            ),
            baseline_iterations=int(bias_best_result["n_iterations"]),
            random_state=RANDOM_STATE,
            item_ids=item_ids,
        ).fit(*train_arrays)
        for n_neighbors in ITEM_KNN_K_GRID:
            predictions = model.predict_pairs(
                validation_users,
                validation_items,
                clip=True,
                n_neighbors=n_neighbors,
            )
            validation_mse = mse(validation_ratings, predictions)
            validation_rmse = rmse(validation_ratings, predictions)
            results.append(
                {
                    "k": int(n_neighbors),
                    "shrinkage": float(shrinkage),
                    "min_common": ITEM_KNN_MIN_COMMON,
                    "validation_mse": float(validation_mse),
                    "validation_rmse": float(validation_rmse),
                }
            )
            print(
                f"Item-kNN k={n_neighbors}, shrinkage={shrinkage:g}: "
                f"validation RMSE={validation_rmse:.6f}"
            )
    best = min(
        results,
        key=lambda row: (
            row["validation_rmse"],
            row["k"],
            -row["shrinkage"],
        ),
    )
    return best, results


def _fit_final_item_knn(
    combined_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
    bias_best_result: dict[str, float | int],
    item_knn_best_result: dict[str, float | int],
    item_ids: np.ndarray,
) -> ItemKNNModel:
    model = ItemKNNModel(
        n_users=n_users,
        n_items=n_items,
        n_neighbors=int(item_knn_best_result["k"]),
        shrinkage=float(item_knn_best_result["shrinkage"]),
        min_common=int(item_knn_best_result["min_common"]),
        baseline_user_regularization=float(
            bias_best_result["user_regularization"]
        ),
        baseline_item_regularization=float(
            bias_best_result["item_regularization"]
        ),
        baseline_iterations=int(bias_best_result["n_iterations"]),
        random_state=RANDOM_STATE,
        item_ids=item_ids,
    )
    return model.fit(*combined_arrays)


def _item_knn_diagnostics(model: ItemKNNModel) -> dict[str, object]:
    model._check_fitted()
    similarities = (
        np.concatenate(model.neighbor_similarities)
        if any(values.size for values in model.neighbor_similarities)
        else np.empty(0, dtype=np.float64)
    )
    common_counts = (
        np.concatenate(model.neighbor_common_counts)
        if any(values.size for values in model.neighbor_common_counts)
        else np.empty(0, dtype=np.int32)
    )
    ordering_valid = True
    self_neighbor_count = 0
    for item_index, (neighbors, values) in enumerate(
        zip(
            model.neighbor_indices,
            model.neighbor_similarities,
            strict=True,
        )
    ):
        self_neighbor_count += int(np.sum(neighbors == item_index))
        keys = [
            (-abs(float(value)), -float(value), int(model.item_ids[neighbor]))
            for neighbor, value in zip(neighbors, values, strict=True)
        ]
        ordering_valid = ordering_valid and keys == sorted(keys)
    return {
        "stored_neighbor_count": int(
            sum(len(values) for values in model.neighbor_indices)
        ),
        "similarities_finite": bool(np.isfinite(similarities).all()),
        "minimum_similarity": None
        if similarities.size == 0
        else float(similarities.min()),
        "maximum_similarity": None
        if similarities.size == 0
        else float(similarities.max()),
        "minimum_common_users": None
        if common_counts.size == 0
        else int(common_counts.min()),
        "self_neighbor_count": int(self_neighbor_count),
        "deterministic_ordering_verified": bool(ordering_valid),
    }


def _tune_pmf(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    train_users, train_items, train_ratings = train_arrays
    val_users, val_items, val_ratings = validation_arrays
    results: list[dict[str, object]] = []
    best_model: PMFModel | None = None
    best_result: dict[str, object] | None = None
    for number, params in enumerate(PMF_GRID, start=1):
        print(f"PMF tuning {number}/{len(PMF_GRID)}: {params}")
        started = time.perf_counter()
        model = PMFModel(
            n_users=n_users,
            n_items=n_items,
            epochs=PMF_TUNING_EPOCHS,
            patience=PMF_TUNING_PATIENCE,
            min_delta=PMF_TUNING_MIN_DELTA,
            random_state=RANDOM_STATE,
            **params,
        ).fit(
            train_users,
            train_items,
            train_ratings,
            val_users,
            val_items,
            val_ratings,
        )
        result = {
            **params,
            "best_epoch": int(model.best_epoch or len(model.history)),
            "validation_rmse": float(model.best_validation_rmse or np.inf),
            "epochs_run": len(model.history),
            "seconds": round(time.perf_counter() - started, 3),
            "hit_epoch_cap": int(model.best_epoch or len(model.history))
            == PMF_TUNING_EPOCHS,
            "hit_factor_boundary": int(params["n_factors"])
            == max(PMF_FACTOR_GRID),
        }
        results.append(result)
        print(
            f"PMF result: epoch={result['best_epoch']}, "
            f"validation RMSE={result['validation_rmse']:.6f}"
        )
        if best_result is None or _pmf_result_sort_key(result) < _pmf_result_sort_key(
            best_result
        ):
            best_result = result
            best_model = model
    assert best_result is not None and best_model is not None
    return best_result, results, best_model.history


def _pmf_result_sort_key(result: dict[str, object]) -> tuple[float, int, float, int]:
    return (
        float(result["validation_rmse"]),
        int(result["n_factors"]),
        -float(result["factor_regularization"]),
        int(result["best_epoch"]),
    )


def _pmf_search_diagnostics(best_result: dict[str, object]) -> dict[str, object]:
    return {
        "selected_at_factor_boundary": int(best_result["n_factors"])
        == max(PMF_FACTOR_GRID),
        "selected_at_epoch_boundary": int(best_result["best_epoch"])
        == PMF_TUNING_EPOCHS,
        "selected_early_stopping_triggered": int(best_result["epochs_run"])
        < PMF_TUNING_EPOCHS,
        "search_max_factors": max(PMF_FACTOR_GRID),
        "search_max_epochs": PMF_TUNING_EPOCHS,
    }


def _fit_final_pmf(
    combined_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
    best_result: dict[str, object],
) -> PMFModel:
    model = PMFModel(
        n_users=n_users,
        n_items=n_items,
        n_factors=int(best_result["n_factors"]),
        learning_rate=float(best_result["learning_rate"]),
        factor_regularization=float(best_result["factor_regularization"]),
        bias_regularization=float(best_result["bias_regularization"]),
        epochs=int(best_result["best_epoch"]),
        patience=int(best_result["best_epoch"]) + 1,
        random_state=RANDOM_STATE,
    )
    return model.fit(*combined_arrays)


def _add_pmf_search_metadata(
    metadata_path: Path,
    best_result: dict[str, object],
    diagnostics: dict[str, object],
) -> None:
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata["training_mode"] = "final_refit_train_plus_validation_without_holdout"
    metadata["selected_validation_result"] = {
        "n_factors": int(best_result["n_factors"]),
        "learning_rate": float(best_result["learning_rate"]),
        "factor_regularization": float(best_result["factor_regularization"]),
        "bias_regularization": float(best_result["bias_regularization"]),
        "best_epoch": int(best_result["best_epoch"]),
        "validation_rmse": float(best_result["validation_rmse"]),
        "epochs_run": int(best_result["epochs_run"]),
    }
    metadata["search_diagnostics"] = diagnostics
    save_json(metadata_path, metadata)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    processed_dir = root / "processed"
    mappings_dir = processed_dir / "mappings"
    reports_dir = root / "reports"
    pmf_dir = reports_dir / "pmf_factors"
    for directory in (data_dir, processed_dir, mappings_dir, reports_dir, pmf_dir):
        directory.mkdir(parents=True, exist_ok=True)

    download_movielens(data_dir)
    data = load_movielens(data_dir)
    summary = validate_movielens(data)
    save_json(reports_dir / "data_summary.json", summary)
    print(
        f"Loaded {summary['n_ratings']:,} ratings, "
        f"{summary['n_users']:,} users, {summary['n_movies']:,} movies."
    )

    split = deterministic_user_split(data.ratings, random_state=RANDOM_STATE)
    split.train.to_csv(processed_dir / "train_ratings.csv", index=False)
    split.validation.to_csv(processed_dir / "validation_ratings.csv", index=False)
    split.test.to_csv(processed_dir / "test_ratings.csv", index=False)
    print(
        f"Split sizes: train={len(split.train):,}, "
        f"validation={len(split.validation):,}, test={len(split.test):,}"
    )

    rated_user_ids = set(data.ratings["user_id"].astype(int))
    rated_movie_ids = set(data.ratings["movie_id"].astype(int))
    mapped_users = data.users.loc[data.users["user_id"].isin(rated_user_ids)]
    mapped_movies = data.movies.loc[data.movies["movie_id"].isin(rated_movie_ids)]
    user_to_index, movie_to_index, index_to_user, index_to_movie = create_mappings(
        mapped_users, mapped_movies
    )
    save_mappings(
        mappings_dir,
        user_to_index,
        movie_to_index,
        index_to_user,
        index_to_movie,
    )

    train_matrix, train_means = build_normalized_matrix(
        split.train, user_to_index, movie_to_index
    )
    np.save(processed_dir / "train_user_means.npy", train_means)
    save_normalized_matrix_csv(
        processed_dir / "user_item_matrix.csv",
        train_matrix,
        index_to_user,
        index_to_movie,
    )

    train_arrays = _indexed(split.train, user_to_index, movie_to_index)
    validation_arrays = _indexed(split.validation, user_to_index, movie_to_index)
    bias_best, bias_results = _tune_bias_baseline(
        train_arrays,
        validation_arrays,
        len(user_to_index),
        len(movie_to_index),
    )
    item_knn_best, item_knn_results = _tune_item_knn(
        train_arrays,
        validation_arrays,
        len(user_to_index),
        len(movie_to_index),
        bias_best,
        index_to_movie,
    )

    svd_best, svd_results = _tune_svd(
        train_matrix,
        train_means,
        split.validation,
        user_to_index,
        movie_to_index,
    )
    save_json(reports_dir / "svd_tuning.json", svd_results)
    svd_rank_tuning = prepare_svd_rank_tuning(
        svd_results,
        selected_rank=int(svd_best["n_factors"]),
        selected_item_bias_regularization=float(
            svd_best["item_bias_regularization"]
        ),
    )
    plot_svd_rank_tuning(
        svd_rank_tuning,
        reports_dir / "svd_rank_tuning_rmse.png",
        metric="rmse",
    )
    plot_svd_rank_tuning(
        svd_rank_tuning,
        reports_dir / "svd_rank_tuning_mse.png",
        metric="mse",
    )

    pmf_best, pmf_results, pmf_history = _tune_pmf(
        train_arrays,
        validation_arrays,
        len(user_to_index),
        len(movie_to_index),
    )
    save_json(reports_dir / "pmf_tuning.json", pmf_results)
    pmf_convergence = prepare_pmf_convergence_payload(
        pmf_history,
        selected_epoch=int(pmf_best["best_epoch"]),
        patience=PMF_TUNING_PATIENCE,
        min_delta=PMF_TUNING_MIN_DELTA,
    )
    save_json(reports_dir / "pmf_convergence.json", pmf_convergence)
    plot_pmf_convergence(
        pmf_convergence,
        reports_dir / "pmf_convergence_rmse.png",
        metric="rmse",
    )
    plot_pmf_convergence(
        pmf_convergence,
        reports_dir / "pmf_convergence_mse.png",
        metric="mse",
    )
    plot_pmf_convergence(
        pmf_convergence,
        reports_dir / "pmf_convergence.png",
        metric="rmse",
    )
    pmf_search_diagnostics = _pmf_search_diagnostics(pmf_best)

    train_validation = pd.concat(
        [split.train, split.validation], ignore_index=True
    ).sort_values(["user_id", "movie_id"], kind="mergesort")
    final_matrix, final_means = build_normalized_matrix(
        train_validation, user_to_index, movie_to_index
    )
    final_svd = SVDModel(
        n_factors=int(svd_best["n_factors"]),
        item_bias_regularization=float(svd_best["item_bias_regularization"]),
        random_state=RANDOM_STATE,
    ).fit(final_matrix, final_means)
    svd_prediction_matrix = final_svd.predict_all(clip=False)
    np.save(reports_dir / "svd_predictions.npy", svd_prediction_matrix.astype(np.float32))
    np.save(reports_dir / "svd_user_means.npy", final_means.astype(np.float32))
    save_json(
        reports_dir / "svd_metadata.json",
        {
            "n_factors": int(svd_best["n_factors"]),
            "item_bias_regularization": float(
                svd_best["item_bias_regularization"]
            ),
            "random_state": RANDOM_STATE,
            "shape": list(svd_prediction_matrix.shape),
            "prediction_scale": "raw_unclipped",
        },
    )

    combined_arrays = _indexed(train_validation, user_to_index, movie_to_index)
    final_bias_baseline = _fit_final_bias_baseline(
        combined_arrays,
        len(user_to_index),
        len(movie_to_index),
        bias_best,
    )
    final_item_knn = _fit_final_item_knn(
        combined_arrays,
        len(user_to_index),
        len(movie_to_index),
        bias_best,
        item_knn_best,
        index_to_movie,
    )
    final_pmf_params = {
        key: pmf_best[key]
        for key in (
            "n_factors",
            "learning_rate",
            "factor_regularization",
            "bias_regularization",
        )
    }
    final_pmf = _fit_final_pmf(
        combined_arrays,
        len(user_to_index),
        len(movie_to_index),
        pmf_best,
    )
    final_pmf.save(pmf_dir)
    _add_pmf_search_metadata(
        pmf_dir / "metadata.json",
        pmf_best,
        pmf_search_diagnostics,
    )

    test_users, test_movies, test_actual = _indexed(
        split.test, user_to_index, movie_to_index
    )
    bias_test_predictions = final_bias_baseline.predict_pairs(
        test_users, test_movies
    )
    item_knn_test_predictions = final_item_knn.predict_pairs(
        test_users, test_movies
    )
    svd_test_predictions = final_svd.predict_pairs(test_users, test_movies)
    pmf_test_predictions = final_pmf.predict_pairs(test_users, test_movies)
    bias_mse = mse(test_actual, bias_test_predictions)
    bias_rmse = rmse(test_actual, bias_test_predictions)
    item_knn_mse = mse(test_actual, item_knn_test_predictions)
    item_knn_rmse = rmse(test_actual, item_knn_test_predictions)
    svd_mse = mse(test_actual, svd_test_predictions)
    pmf_mse = mse(test_actual, pmf_test_predictions)
    svd_rmse = rmse(test_actual, svd_test_predictions)
    pmf_rmse = rmse(test_actual, pmf_test_predictions)
    item_knn_vs_bias = (bias_rmse - item_knn_rmse) / bias_rmse * 100.0
    svd_vs_bias = (bias_rmse - svd_rmse) / bias_rmse * 100.0
    pmf_vs_bias = (bias_rmse - pmf_rmse) / bias_rmse * 100.0
    svd_vs_item_knn = (item_knn_rmse - svd_rmse) / item_knn_rmse * 100.0
    pmf_vs_item_knn = (item_knn_rmse - pmf_rmse) / item_knn_rmse * 100.0
    pmf_vs_svd = (svd_rmse - pmf_rmse) / svd_rmse * 100.0

    save_json(
        reports_dir / "bias_baseline_tuning.json",
        {
            "model": "BiasBaseline",
            "prediction_formula": "global_mean + user_bias + item_bias",
            "random_state": RANDOM_STATE,
            "selection_metric": "validation_rmse",
            "uses_test_for_tuning": False,
            "regularization_grid": BIAS_BASELINE_REGULARIZATION_GRID,
            "results": bias_results,
            "selected": bias_best,
            "final_refit": {
                "training_rows": len(train_validation),
                "uses_train_plus_validation": True,
                "uses_validation_holdout": False,
            },
            "test_evaluation": {
                "test_rows": len(split.test),
                "mse": float(bias_mse),
                "rmse": float(bias_rmse),
            },
        },
    )
    save_json(
        reports_dir / "item_knn_tuning.json",
        {
            "model": "ItemKNN",
            "prediction_formula": (
                "bias_baseline + sum(similarity_ij * residual_uj) "
                "/ sum(abs(similarity_ij))"
            ),
            "similarity_definition": (
                "cosine(item residual vectors) * common_users "
                "/ (common_users + shrinkage)"
            ),
            "neighborhood_definition": (
                "global top-k eligible item neighbors intersected with the "
                "target user's fitted rating history"
            ),
            "neighborhood_ordering": [
                "absolute shrunk similarity descending",
                "signed shrunk similarity descending",
                "movie ID ascending",
            ],
            "parameter_grid": {
                "k": ITEM_KNN_K_GRID,
                "shrinkage": ITEM_KNN_SHRINKAGE_GRID,
                "min_common": ITEM_KNN_MIN_COMMON,
            },
            "selection_metric": "validation_rmse",
            "selection_tie_break": [
                "lower validation RMSE",
                "lower k",
                "higher shrinkage",
            ],
            "uses_test_for_tuning": False,
            "results": item_knn_results,
            "selected": item_knn_best,
            "final_refit": {
                "training_rows": len(train_validation),
                "uses_train_plus_validation": True,
                "uses_validation_holdout": False,
                "diagnostics": _item_knn_diagnostics(final_item_knn),
            },
            "test_evaluation": {
                "test_rows": len(split.test),
                "mse": float(item_knn_mse),
                "rmse": float(item_knn_rmse),
            },
        },
    )

    metrics = {
        "random_state": RANDOM_STATE,
        "split": {
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "actual_counts": {
                "train": len(split.train),
                "validation": len(split.validation),
                "test": len(split.test),
            },
        },
        "BiasBaseline_MSE": bias_mse,
        "BiasBaseline_RMSE": bias_rmse,
        "ItemKNN_MSE": item_knn_mse,
        "ItemKNN_RMSE": item_knn_rmse,
        "SVD_MSE": svd_mse,
        "SVD_RMSE": svd_rmse,
        "PMF_MSE": pmf_mse,
        "PMF_RMSE": pmf_rmse,
        "ItemKNN_vs_BiasBaseline_improvement_%": item_knn_vs_bias,
        "SVD_vs_BiasBaseline_improvement_%": svd_vs_bias,
        "PMF_vs_BiasBaseline_improvement_%": pmf_vs_bias,
        "SVD_vs_ItemKNN_improvement_%": svd_vs_item_knn,
        "PMF_vs_ItemKNN_improvement_%": pmf_vs_item_knn,
        "PMF_vs_SVD_improvement_%": pmf_vs_svd,
        "ItemKNN_beats_BiasBaseline": item_knn_rmse < bias_rmse,
        "SVD_beats_BiasBaseline": svd_rmse < bias_rmse,
        "PMF_beats_BiasBaseline": pmf_rmse < bias_rmse,
        "SVD_beats_ItemKNN": svd_rmse < item_knn_rmse,
        "PMF_beats_ItemKNN": pmf_rmse < item_knn_rmse,
        "SVD_target_met": svd_rmse <= 0.90,
        "PMF_target_met": pmf_rmse <= 0.85,
        "improvement_target_met": pmf_vs_svd >= 5.0,
        "bias_baseline_best_params": {
            "user_regularization": float(bias_best["user_regularization"]),
            "item_regularization": float(bias_best["item_regularization"]),
            "n_iterations": int(bias_best["n_iterations"]),
            "validation_rmse": float(bias_best["validation_rmse"]),
        },
        "item_knn_best_params": {
            "k": int(item_knn_best["k"]),
            "shrinkage": float(item_knn_best["shrinkage"]),
            "min_common": int(item_knn_best["min_common"]),
            "validation_rmse": float(item_knn_best["validation_rmse"]),
        },
        "svd_best_params": {
            "n_factors": int(svd_best["n_factors"]),
            "item_bias_regularization": float(
                svd_best["item_bias_regularization"]
            ),
        },
        "pmf_best_params": {
            **final_pmf_params,
            "selected_epoch": int(pmf_best["best_epoch"]),
            "validation_rmse": float(pmf_best["validation_rmse"]),
        },
        "pmf_search_diagnostics": pmf_search_diagnostics,
    }
    save_json(reports_dir / "model_metrics.json", metrics)

    plot_predicted_vs_actual(
        test_actual,
        svd_test_predictions,
        pmf_test_predictions,
        reports_dir / "predicted_vs_actual.png",
    )
    model_mse = {
        "BiasBaseline": bias_mse,
        "ItemKNN": item_knn_mse,
        "SVD": svd_mse,
        "PMF": pmf_mse,
    }
    model_rmse = {
        "BiasBaseline": bias_rmse,
        "ItemKNN": item_knn_rmse,
        "SVD": svd_rmse,
        "PMF": pmf_rmse,
    }
    plot_model_metric_comparison(
        model_mse,
        reports_dir / "model_mse_comparison.png",
        metric="MSE",
    )
    plot_model_metric_comparison(
        model_rmse,
        reports_dir / "model_rmse_comparison.png",
        metric="RMSE",
    )
    plot_model_metric_comparison(
        model_rmse,
        reports_dir / "rmse_comparison.png",
        metric="RMSE",
    )

    model_movies = data.movies.loc[
        data.movies["movie_id"].isin(index_to_movie)
    ].copy()

    ranking_train, ranking_targets, ranking_protocol = (
        build_temporal_ranking_protocol(data.ratings)
    )
    ranking_train.to_csv(
        processed_dir / "ranking_train_ratings.csv", index=False
    )
    ranking_targets.to_csv(processed_dir / "ranking_targets.csv", index=False)
    ranking_users = data.users.loc[
        data.users["user_id"].isin(ranking_train["user_id"])
    ]
    ranking_movies = data.movies.loc[
        data.movies["movie_id"].isin(ranking_train["movie_id"])
    ]
    (
        ranking_user_to_index,
        ranking_movie_to_index,
        ranking_index_to_user,
        ranking_index_to_movie,
    ) = create_mappings(ranking_users, ranking_movies)
    ranking_arrays = _indexed(
        ranking_train,
        ranking_user_to_index,
        ranking_movie_to_index,
    )
    ranking_bias = _fit_final_bias_baseline(
        ranking_arrays,
        len(ranking_user_to_index),
        len(ranking_movie_to_index),
        bias_best,
    )
    ranking_item_knn = _fit_final_item_knn(
        ranking_arrays,
        len(ranking_user_to_index),
        len(ranking_movie_to_index),
        bias_best,
        item_knn_best,
        ranking_index_to_movie,
    )
    ranking_matrix, ranking_user_means = build_normalized_matrix(
        ranking_train,
        ranking_user_to_index,
        ranking_movie_to_index,
    )
    ranking_svd = SVDModel(
        n_factors=int(svd_best["n_factors"]),
        item_bias_regularization=float(svd_best["item_bias_regularization"]),
        random_state=RANDOM_STATE,
    ).fit(ranking_matrix, ranking_user_means)
    ranking_pmf = _fit_final_pmf(
        ranking_arrays,
        len(ranking_user_to_index),
        len(ranking_movie_to_index),
        pmf_best,
    )
    ranking_results, ranking_metrics = evaluate_ranking_models(
        ranking_targets,
        ranking_train,
        ranking_movies,
        ranking_user_to_index,
        ranking_movie_to_index,
        ranking_index_to_movie,
        ranking_bias,
        ranking_item_knn,
        ranking_svd,
        ranking_pmf,
    )
    ranking_results.to_csv(reports_dir / "ranking_results.csv", index=False)
    save_json(reports_dir / "ranking_metrics.json", ranking_metrics)
    plot_ranking_comparison(
        ranking_metrics,
        str(reports_dir / "ranking_comparison.png"),
    )
    ranking_protocol["frozen_model_parameters"] = {
        "BiasBaseline": {
            "user_regularization": float(bias_best["user_regularization"]),
            "item_regularization": float(bias_best["item_regularization"]),
            "n_iterations": int(bias_best["n_iterations"]),
        },
        "ItemKNN": {
            "k": int(item_knn_best["k"]),
            "shrinkage": float(item_knn_best["shrinkage"]),
            "min_common": int(item_knn_best["min_common"]),
        },
        "SVD": {
            "n_factors": int(svd_best["n_factors"]),
            "item_bias_regularization": float(
                svd_best["item_bias_regularization"]
            ),
            "random_state": RANDOM_STATE,
        },
        "PMF": {
            **final_pmf_params,
            "epochs": int(pmf_best["best_epoch"]),
            "random_state": RANDOM_STATE,
            "uses_ranking_targets_for_tuning": False,
        },
    }
    save_json(reports_dir / "ranking_protocol.json", ranking_protocol)

    factor_interpretation = build_pmf_factor_interpretation(
        final_pmf.item_factors,
        index_to_movie,
        model_movies,
        n_factors=5,
        top_n=8,
    )
    factor_interpretation.to_csv(
        reports_dir / "pmf_factor_interpretation.csv", index=False
    )
    genre_profiles = build_pmf_factor_genre_profiles(factor_interpretation)
    genre_profiles.to_csv(
        reports_dir / "pmf_factor_genre_profiles.csv", index=False
    )
    plot_pmf_latent_factor_heatmap(
        factor_interpretation,
        final_pmf.item_factors,
        movie_to_index,
        reports_dir / "pmf_latent_factor_heatmap.png",
    )
    similarities = build_pmf_movie_similarities(
        final_pmf.item_factors,
        index_to_movie,
        model_movies,
        data.ratings,
        movie_to_index,
        n_anchors=5,
        top_n=10,
    )
    similarities.to_csv(reports_dir / "pmf_movie_similarities.csv", index=False)

    svd_rec_model = SVDRecommendationModel(
        svd_prediction_matrix,
        user_to_index,
        movie_to_index,
        index_to_movie,
        model_movies,
        data.ratings,
    )
    pmf_rec_model = PMFRecommendationModel(
        final_pmf,
        user_to_index,
        movie_to_index,
        index_to_movie,
        model_movies,
        data.ratings,
    )
    test_with_predictions = split.test.copy()
    test_with_predictions["svd_prediction"] = svd_test_predictions
    test_with_predictions["pmf_prediction"] = pmf_test_predictions
    evaluated_users = select_evaluation_users(
        split.train,
        split.validation,
        test_with_predictions,
        ranking_results,
    )
    save_json(reports_dir / "evaluated_users.json", evaluated_users)
    current_user_ids = {
        int(selection["user_id"]) for selection in evaluated_users
    }
    removed_user_artifacts = cleanup_user_artifacts(
        reports_dir, current_user_ids
    )
    for path in removed_user_artifacts:
        print(f"Removed stale user artifact: {path.name}")
    ranking_by_user = ranking_results.set_index("user_id")
    first_comparison: pd.DataFrame | None = None
    for selection in evaluated_users:
        user_id = int(selection["user_id"])
        comparison = compare_recommendations(
            user_id, svd_rec_model, pmf_rec_model, top_n=10
        )
        comparison.to_csv(
            reports_dir / f"user_{user_id}_recommendations.csv", index=False
        )
        explanations = build_local_pmf_explanations(
            user_id,
            str(selection["role"]),
            comparison,
            final_pmf,
            user_to_index,
            movie_to_index,
            data.ratings,
            model_movies,
        )
        explanations.to_csv(
            reports_dir / f"user_{user_id}_explanations.csv", index=False
        )
        plot_user_explanation(
            explanations,
            reports_dir / f"user_{user_id}_explanation.png",
        )
        ranking_case = build_ranking_case_explanation(
            selection,
            ranking_by_user.loc[user_id],
            ranking_train,
            ranking_pmf,
            ranking_user_to_index,
            ranking_movie_to_index,
            ranking_movies,
        )
        ranking_case.to_csv(
            reports_dir / f"user_{user_id}_ranking_case.csv", index=False
        )
        plot_ranking_case(
            ranking_case,
            reports_dir / f"user_{user_id}_ranking_case.png",
        )
        if first_comparison is None:
            first_comparison = comparison
    assert first_comparison is not None
    plot_user_comparison(first_comparison, reports_dir / "user_comparison.png")
    plot_top_recommendations(first_comparison, reports_dir / "top_recommendations.png")

    print("\nPipeline complete")
    print(f"Bias baseline test RMSE: {bias_rmse:.6f}")
    print(f"Item-kNN test RMSE: {item_knn_rmse:.6f}")
    print(f"SVD test RMSE: {svd_rmse:.6f}")
    print(f"PMF test RMSE: {pmf_rmse:.6f}")
    print(f"PMF vs SVD improvement: {pmf_vs_svd:.3f}%")
    print(f"Ranking eligible users: {ranking_metrics['eligible_user_count']}")
    print(f"Metrics: {reports_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
