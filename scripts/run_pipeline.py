from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.pmf_model import PMFModel
from models.svd_model import SVDModel
from scripts.download_data import download_movielens
from utils.artifacts import (
    plot_convergence,
    plot_predicted_vs_actual,
    plot_rmse_comparison,
    plot_top_recommendations,
    plot_user_comparison,
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
from utils.recommendation import (
    PMFRecommendationModel,
    SVDRecommendationModel,
    compare_recommendations,
)
from utils.split import deterministic_user_split


RANDOM_STATE = 42
SVD_GRID = [5, 10, 20, 40, 60]
SVD_ITEM_BIAS_REGULARIZATION_GRID = [0.0, 5.0, 10.0, 20.0, 40.0, 80.0]
PMF_GRID = [
    {
        "n_factors": 48,
        "learning_rate": 0.008,
        "factor_regularization": 0.04,
        "bias_regularization": 0.02,
    },
    {
        "n_factors": 64,
        "learning_rate": 0.008,
        "factor_regularization": 0.05,
        "bias_regularization": 0.02,
    },
    {
        "n_factors": 64,
        "learning_rate": 0.006,
        "factor_regularization": 0.04,
        "bias_regularization": 0.02,
    },
    {
        "n_factors": 64,
        "learning_rate": 0.006,
        "factor_regularization": 0.06,
        "bias_regularization": 0.02,
    },
    {
        "n_factors": 80,
        "learning_rate": 0.006,
        "factor_regularization": 0.05,
        "bias_regularization": 0.02,
    },
    {
        "n_factors": 96,
        "learning_rate": 0.006,
        "factor_regularization": 0.06,
        "bias_regularization": 0.02,
    },
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
            epochs=45,
            patience=6,
            min_delta=5e-5,
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
        }
        results.append(result)
        print(
            f"PMF result: epoch={result['best_epoch']}, "
            f"validation RMSE={result['validation_rmse']:.6f}"
        )
        if best_result is None or (
            result["validation_rmse"],
            result["n_factors"],
        ) < (
            best_result["validation_rmse"],
            best_result["n_factors"],
        ):
            best_result = result
            best_model = model
    assert best_result is not None and best_model is not None
    return best_result, results, best_model.history


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

    svd_best, svd_results = _tune_svd(
        train_matrix,
        train_means,
        split.validation,
        user_to_index,
        movie_to_index,
    )
    save_json(reports_dir / "svd_tuning.json", svd_results)

    train_arrays = _indexed(split.train, user_to_index, movie_to_index)
    validation_arrays = _indexed(split.validation, user_to_index, movie_to_index)
    pmf_best, pmf_results, pmf_history = _tune_pmf(
        train_arrays,
        validation_arrays,
        len(user_to_index),
        len(movie_to_index),
    )
    save_json(reports_dir / "pmf_tuning.json", pmf_results)
    plot_convergence(pmf_history, reports_dir / "pmf_convergence.png")

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
    final_pmf_params = {
        key: pmf_best[key]
        for key in (
            "n_factors",
            "learning_rate",
            "factor_regularization",
            "bias_regularization",
        )
    }
    final_pmf = PMFModel(
        n_users=len(user_to_index),
        n_items=len(movie_to_index),
        epochs=int(pmf_best["best_epoch"]),
        patience=int(pmf_best["best_epoch"]) + 1,
        random_state=RANDOM_STATE,
        **final_pmf_params,
    ).fit(*combined_arrays)
    final_pmf.save(pmf_dir)

    test_users, test_movies, test_actual = _indexed(
        split.test, user_to_index, movie_to_index
    )
    svd_test_predictions = final_svd.predict_pairs(test_users, test_movies)
    pmf_test_predictions = final_pmf.predict_pairs(test_users, test_movies)
    svd_mse = mse(test_actual, svd_test_predictions)
    pmf_mse = mse(test_actual, pmf_test_predictions)
    svd_rmse = rmse(test_actual, svd_test_predictions)
    pmf_rmse = rmse(test_actual, pmf_test_predictions)
    improvement = (svd_rmse - pmf_rmse) / svd_rmse * 100.0

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
        "SVD_MSE": svd_mse,
        "SVD_RMSE": svd_rmse,
        "PMF_MSE": pmf_mse,
        "PMF_RMSE": pmf_rmse,
        "PMF_vs_SVD_improvement_%": improvement,
        "SVD_target_met": svd_rmse <= 0.90,
        "PMF_target_met": pmf_rmse <= 0.85,
        "improvement_target_met": improvement >= 5.0,
        "svd_best_params": {
            "n_factors": int(svd_best["n_factors"]),
            "item_bias_regularization": float(
                svd_best["item_bias_regularization"]
            ),
        },
        "pmf_best_params": {
            **final_pmf_params,
            "selected_epoch": int(pmf_best["best_epoch"]),
        },
    }
    save_json(reports_dir / "model_metrics.json", metrics)

    plot_predicted_vs_actual(
        test_actual,
        svd_test_predictions,
        pmf_test_predictions,
        reports_dir / "predicted_vs_actual.png",
    )
    plot_rmse_comparison(svd_rmse, pmf_rmse, reports_dir / "rmse_comparison.png")

    model_movies = data.movies.loc[
        data.movies["movie_id"].isin(index_to_movie)
    ].copy()
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
    evaluated_users = _select_showcase_users(split.train)
    save_json(reports_dir / "evaluated_users.json", evaluated_users)
    first_comparison: pd.DataFrame | None = None
    for selection in evaluated_users:
        user_id = int(selection["user_id"])
        comparison = compare_recommendations(
            user_id, svd_rec_model, pmf_rec_model, top_n=10
        )
        comparison.to_csv(
            reports_dir / f"user_{user_id}_recommendations.csv", index=False
        )
        if first_comparison is None:
            first_comparison = comparison
    assert first_comparison is not None
    plot_user_comparison(first_comparison, reports_dir / "user_comparison.png")
    plot_top_recommendations(first_comparison, reports_dir / "top_recommendations.png")

    print("\nPipeline complete")
    print(f"SVD test RMSE: {svd_rmse:.6f}")
    print(f"PMF test RMSE: {pmf_rmse:.6f}")
    print(f"PMF improvement: {improvement:.3f}%")
    print(f"Metrics: {reports_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
