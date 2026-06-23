from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from models.baseline_cf import BaselineCFModel
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
from utils.interpretability import (
    build_local_pmf_explanations,
    build_pmf_factor_genre_profiles,
    build_pmf_factor_interpretation,
    build_pmf_movie_similarities,
    plot_pmf_latent_factor_heatmap,
    plot_user_explanation,
    select_audit_users,
)
from utils.recommendation import (
    PMFRecommendationModel,
    SVDRecommendationModel,
    compare_recommendations,
)
from utils.split import deterministic_user_split


RANDOM_STATE = 42
BASELINE_BIAS_REGULARIZATION_GRID = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
BASELINE_ITERATIONS = 20
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


def _tune_baseline_cf(
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    train_users, train_items, train_ratings = train_arrays
    validation_users, validation_items, validation_ratings = validation_arrays
    results: list[dict[str, float | int]] = []
    for regularization in BASELINE_BIAS_REGULARIZATION_GRID:
        model = BaselineCFModel(
            n_users=n_users,
            n_items=n_items,
            user_regularization=regularization,
            item_regularization=regularization,
            n_iterations=BASELINE_ITERATIONS,
            random_state=RANDOM_STATE,
        ).fit(train_users, train_items, train_ratings)
        predictions = model.predict_pairs(validation_users, validation_items, clip=True)
        validation_mse = mse(validation_ratings, predictions)
        validation_rmse = rmse(validation_ratings, predictions)
        results.append(
            {
                "user_regularization": float(regularization),
                "item_regularization": float(regularization),
                "n_iterations": BASELINE_ITERATIONS,
                "validation_mse": float(validation_mse),
                "validation_rmse": float(validation_rmse),
            }
        )
        print(
            "Baseline CF "
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


def _fit_final_baseline_cf(
    combined_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    n_users: int,
    n_items: int,
    best_result: dict[str, float | int],
) -> BaselineCFModel:
    model = BaselineCFModel(
        n_users=n_users,
        n_items=n_items,
        user_regularization=float(best_result["user_regularization"]),
        item_regularization=float(best_result["item_regularization"]),
        n_iterations=int(best_result["n_iterations"]),
        random_state=RANDOM_STATE,
    )
    return model.fit(*combined_arrays)


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
    baseline_best, baseline_results = _tune_baseline_cf(
        train_arrays,
        validation_arrays,
        len(user_to_index),
        len(movie_to_index),
    )

    svd_best, svd_results = _tune_svd(
        train_matrix,
        train_means,
        split.validation,
        user_to_index,
        movie_to_index,
    )
    save_json(reports_dir / "svd_tuning.json", svd_results)

    pmf_best, pmf_results, pmf_history = _tune_pmf(
        train_arrays,
        validation_arrays,
        len(user_to_index),
        len(movie_to_index),
    )
    save_json(reports_dir / "pmf_tuning.json", pmf_results)
    plot_convergence(pmf_history, reports_dir / "pmf_convergence.png")
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
    final_baseline = _fit_final_baseline_cf(
        combined_arrays,
        len(user_to_index),
        len(movie_to_index),
        baseline_best,
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
    baseline_test_predictions = final_baseline.predict_pairs(test_users, test_movies)
    svd_test_predictions = final_svd.predict_pairs(test_users, test_movies)
    pmf_test_predictions = final_pmf.predict_pairs(test_users, test_movies)
    baseline_mse = mse(test_actual, baseline_test_predictions)
    baseline_rmse = rmse(test_actual, baseline_test_predictions)
    svd_mse = mse(test_actual, svd_test_predictions)
    pmf_mse = mse(test_actual, pmf_test_predictions)
    svd_rmse = rmse(test_actual, svd_test_predictions)
    pmf_rmse = rmse(test_actual, pmf_test_predictions)
    improvement = (svd_rmse - pmf_rmse) / svd_rmse * 100.0
    svd_vs_baseline = (baseline_rmse - svd_rmse) / baseline_rmse * 100.0
    pmf_vs_baseline = (baseline_rmse - pmf_rmse) / baseline_rmse * 100.0

    save_json(
        reports_dir / "baseline_tuning.json",
        {
            "model": "regularized_bias_only_collaborative_filtering",
            "prediction_formula": "global_mean + user_bias + item_bias",
            "random_state": RANDOM_STATE,
            "selection_metric": "validation_rmse",
            "uses_test_for_tuning": False,
            "regularization_grid": BASELINE_BIAS_REGULARIZATION_GRID,
            "results": baseline_results,
            "selected": baseline_best,
            "final_refit": {
                "training_rows": len(train_validation),
                "uses_train_plus_validation": True,
                "uses_validation_holdout": False,
            },
            "test_evaluation": {
                "test_rows": len(split.test),
                "mse": float(baseline_mse),
                "rmse": float(baseline_rmse),
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
        "Baseline_CF_MSE": baseline_mse,
        "Baseline_CF_RMSE": baseline_rmse,
        "SVD_MSE": svd_mse,
        "SVD_RMSE": svd_rmse,
        "PMF_MSE": pmf_mse,
        "PMF_RMSE": pmf_rmse,
        "SVD_vs_Baseline_improvement_%": svd_vs_baseline,
        "PMF_vs_Baseline_improvement_%": pmf_vs_baseline,
        "PMF_vs_SVD_improvement_%": improvement,
        "SVD_beats_Baseline_CF": svd_rmse < baseline_rmse,
        "PMF_beats_Baseline_CF": pmf_rmse < baseline_rmse,
        "SVD_target_met": svd_rmse <= 0.90,
        "PMF_target_met": pmf_rmse <= 0.85,
        "improvement_target_met": improvement >= 5.0,
        "baseline_best_params": {
            "user_regularization": float(baseline_best["user_regularization"]),
            "item_regularization": float(baseline_best["item_regularization"]),
            "n_iterations": int(baseline_best["n_iterations"]),
            "validation_rmse": float(baseline_best["validation_rmse"]),
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
    plot_rmse_comparison(
        svd_rmse,
        pmf_rmse,
        reports_dir / "rmse_comparison.png",
        baseline_rmse=baseline_rmse,
    )

    model_movies = data.movies.loc[
        data.movies["movie_id"].isin(index_to_movie)
    ].copy()
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
    evaluated_users = select_audit_users(
        split.train,
        split.validation,
        test_with_predictions,
        min_train_ratings=50,
        min_test_ratings=10,
    )
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
        if first_comparison is None:
            first_comparison = comparison
    assert first_comparison is not None
    plot_user_comparison(first_comparison, reports_dir / "user_comparison.png")
    plot_top_recommendations(first_comparison, reports_dir / "top_recommendations.png")

    print("\nPipeline complete")
    print(f"Baseline CF test RMSE: {baseline_rmse:.6f}")
    print(f"SVD test RMSE: {svd_rmse:.6f}")
    print(f"PMF test RMSE: {pmf_rmse:.6f}")
    print(f"PMF improvement: {improvement:.3f}%")
    print(f"Metrics: {reports_dir / 'model_metrics.json'}")


if __name__ == "__main__":
    main()
