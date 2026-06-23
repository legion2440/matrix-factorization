from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from models.pmf_model import PMFModel
from utils.data_loader import load_movielens
from utils.matrix_creation import load_mappings


REQUIRED_PATHS = [
    "data/ratings.dat",
    "data/users.dat",
    "data/movies.dat",
    "processed/train_ratings.csv",
    "processed/validation_ratings.csv",
    "processed/test_ratings.csv",
    "processed/user_item_matrix.csv",
    "processed/mappings/user_to_index.json",
    "processed/mappings/movie_to_index.json",
    "processed/mappings/index_to_user.json",
    "processed/mappings/index_to_movie.json",
    "reports/model_metrics.json",
    "reports/svd_predictions.npy",
    "reports/svd_metadata.json",
    "reports/pmf_convergence.png",
    "reports/predicted_vs_actual.png",
    "reports/rmse_comparison.png",
    "reports/user_comparison.png",
    "reports/top_recommendations.png",
    "reports/evaluated_users.json",
    "reports/pmf_factors/user_factors.npy",
    "reports/pmf_factors/item_factors.npy",
    "reports/pmf_factors/user_bias.npy",
    "reports/pmf_factors/item_bias.npy",
    "reports/pmf_factors/metadata.json",
    "Movie_Recommender_System.ipynb",
    "README.md",
    "requirements.txt",
    "app.py",
]

RECOMMENDATION_COLUMNS = {
    "movie_id",
    "title",
    "genres",
    "svd_ranking_score",
    "svd_predicted_rating",
    "svd_rank",
    "pmf_ranking_score",
    "pmf_predicted_rating",
    "pmf_rank",
}


def _validate_recommendation_ranking(
    recommendations: pd.DataFrame,
    prefix: str,
    expected_count: int,
    user_id: int,
) -> list[str]:
    errors: list[str] = []
    score_column = f"{prefix}_ranking_score"
    rating_column = f"{prefix}_predicted_rating"
    rank_column = f"{prefix}_rank"
    rows = recommendations.dropna(subset=[score_column]).copy()

    if len(rows) != expected_count:
        errors.append(
            f"{prefix.upper()} recommendations for user {user_id}: "
            f"expected {expected_count}, found {len(rows)}"
        )
        return errors
    if not np.isfinite(rows[score_column]).all():
        errors.append(
            f"{prefix.upper()} recommendations for user {user_id} contain "
            "non-finite ranking scores"
        )
    displayed = rows[rating_column].to_numpy(np.float64)
    expected_displayed = np.clip(
        rows[score_column].to_numpy(np.float64), 1.0, 5.0
    )
    if not np.allclose(displayed, expected_displayed, rtol=0.0, atol=1e-6):
        errors.append(
            f"{prefix.upper()} displayed ratings for user {user_id} "
            "are not clipped raw ranking scores"
        )
    if not np.all((displayed >= 1.0) & (displayed <= 5.0)):
        errors.append(
            f"{prefix.upper()} displayed ratings for user {user_id} "
            "are outside [1, 5]"
        )

    ordered = rows.sort_values(
        [score_column, "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    expected_ranks = np.arange(1, expected_count + 1)
    actual_ranks = ordered[rank_column].to_numpy(np.float64)
    if not np.array_equal(actual_ranks, expected_ranks):
        errors.append(
            f"{prefix.upper()} ranks for user {user_id} do not match "
            "raw-score descending/movie-id ascending ordering"
        )
    return errors


def validate() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"Missing required file: {relative}")
    if errors:
        return errors

    try:
        with (root / "reports" / "model_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (root / "reports" / "evaluated_users.json").open(encoding="utf-8") as handle:
            evaluated_users = json.load(handle)
        with (root / "reports" / "svd_metadata.json").open(encoding="utf-8") as handle:
            svd_metadata = json.load(handle)
        with (root / "reports" / "pmf_tuning.json").open(encoding="utf-8") as handle:
            pmf_tuning = json.load(handle)
        with (root / "reports" / "pmf_factors" / "metadata.json").open(
            encoding="utf-8"
        ) as handle:
            pmf_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid JSON artifact: {exc}")
        return errors

    required_metric_keys = {
        "SVD_MSE",
        "SVD_RMSE",
        "PMF_MSE",
        "PMF_RMSE",
        "PMF_vs_SVD_improvement_%",
        "svd_best_params",
        "pmf_best_params",
        "pmf_search_diagnostics",
    }
    if required_metric_keys - set(metrics):
        errors.append("model_metrics.json has an incomplete schema")
    if metrics.get("SVD_RMSE", np.inf) > 0.90:
        errors.append("SVD RMSE target not met")
    if metrics.get("PMF_RMSE", np.inf) > 0.85:
        errors.append("PMF RMSE target not met")
    if metrics.get("PMF_vs_SVD_improvement_%", -np.inf) < 5.0:
        errors.append("PMF improvement target not met")

    required_tuning_fields = {
        "n_factors",
        "learning_rate",
        "factor_regularization",
        "bias_regularization",
        "best_epoch",
        "validation_rmse",
        "epochs_run",
        "seconds",
        "hit_epoch_cap",
        "hit_factor_boundary",
    }
    if not isinstance(pmf_tuning, list) or len(pmf_tuning) != 9:
        errors.append("PMF tuning artifact must contain exactly 9 configurations")
    else:
        combinations = {
            (row.get("n_factors"), row.get("factor_regularization"))
            for row in pmf_tuning
        }
        expected_combinations = {
            (factors, regularization)
            for factors in (96, 112, 128)
            for regularization in (0.05, 0.06, 0.07)
        }
        if combinations != expected_combinations:
            errors.append("PMF tuning artifact has an unexpected search grid")
        for row in pmf_tuning:
            if required_tuning_fields - set(row):
                errors.append("PMF tuning row has an incomplete diagnostic schema")
                break
            if row["hit_epoch_cap"] != (row["best_epoch"] == 70):
                errors.append("PMF hit_epoch_cap diagnostic is inconsistent")
                break
            if row["hit_factor_boundary"] != (row["n_factors"] == 128):
                errors.append("PMF hit_factor_boundary diagnostic is inconsistent")
                break

    diagnostics = metrics.get("pmf_search_diagnostics", {})
    required_diagnostics = {
        "selected_at_factor_boundary",
        "selected_at_epoch_boundary",
        "selected_early_stopping_triggered",
        "search_max_factors",
        "search_max_epochs",
    }
    if required_diagnostics - set(diagnostics):
        errors.append("PMF search diagnostics are missing from model metrics")
    elif diagnostics.get("search_max_factors") != 128 or diagnostics.get(
        "search_max_epochs"
    ) != 70:
        errors.append("PMF search boundary diagnostics are invalid")

    if pmf_metadata.get("search_diagnostics") != diagnostics:
        errors.append("PMF factor metadata search diagnostics do not match metrics")
    if (
        pmf_metadata.get("training_mode")
        != "final_refit_train_plus_validation_without_holdout"
    ):
        errors.append("PMF metadata does not declare validation-free final refit")
    selected_epoch = metrics.get("pmf_best_params", {}).get("selected_epoch")
    if pmf_metadata.get("config", {}).get("epochs") != selected_epoch:
        errors.append("PMF final refit epochs do not match selected best_epoch")
    if len(pmf_metadata.get("history", [])) != selected_epoch:
        errors.append("PMF final refit history length does not match selected best_epoch")
    if any(
        row.get("validation_rmse") is not None
        for row in pmf_metadata.get("history", [])
    ):
        errors.append("PMF final refit unexpectedly used validation data")

    user_to_index, movie_to_index, index_to_user, index_to_movie = load_mappings(
        root / "processed" / "mappings"
    )
    predictions = np.load(root / "reports" / "svd_predictions.npy", mmap_mode="r")
    expected_shape = (len(user_to_index), len(movie_to_index))
    if predictions.shape != expected_shape:
        errors.append(
            f"SVD prediction shape {predictions.shape} != expected {expected_shape}"
        )
    if not np.isfinite(predictions).all():
        errors.append("Raw SVD predictions contain non-finite values")
    if svd_metadata.get("prediction_scale") != "raw_unclipped":
        errors.append("SVD metadata does not declare raw_unclipped prediction scale")

    try:
        pmf = PMFModel.load(root / "reports" / "pmf_factors")
        if pmf.user_factors.shape[0] != len(index_to_user):
            errors.append("PMF user factors do not align with mappings")
        if pmf.item_factors.shape[0] != len(index_to_movie):
            errors.append("PMF item factors do not align with mappings")
        test = pd.read_csv(root / "processed" / "test_ratings.csv")
        test_users = test["user_id"].map(user_to_index).to_numpy()
        test_movies = test["movie_id"].map(movie_to_index).to_numpy()
        if pd.isna(test_users).any() or pd.isna(test_movies).any():
            errors.append("Test rows are not fully covered by mappings")
        else:
            test_user_indices = test_users.astype(np.int32)
            test_movie_indices = test_movies.astype(np.int32)
            svd_displayed_predictions = np.clip(
                predictions[test_user_indices, test_movie_indices], 1.0, 5.0
            )
            if not np.isfinite(svd_displayed_predictions).all():
                errors.append("Clipped SVD test predictions contain non-finite values")
            if (
                svd_displayed_predictions.min() < 1.0
                or svd_displayed_predictions.max() > 5.0
            ):
                errors.append("Clipped SVD test predictions are outside [1, 5]")
            pmf_predictions = pmf.predict_pairs(
                test_user_indices, test_movie_indices, clip=True
            )
            if not np.isfinite(pmf_predictions).all():
                errors.append("PMF predictions contain non-finite values")
            if pmf_predictions.min() < 1.0 or pmf_predictions.max() > 5.0:
                errors.append("PMF predictions are outside [1, 5]")
    except Exception as exc:
        errors.append(f"PMF artifact validation failed: {exc}")

    data = load_movielens(root / "data")
    for selection in evaluated_users:
        user_id = int(selection["user_id"])
        recommendation_path = root / "reports" / f"user_{user_id}_recommendations.csv"
        if not recommendation_path.exists():
            errors.append(f"Missing recommendation CSV for user {user_id}")
            continue
        recommendations = pd.read_csv(recommendation_path)
        if recommendations.empty:
            errors.append(f"Recommendation CSV for user {user_id} is empty")
            continue
        missing_columns = RECOMMENDATION_COLUMNS - set(recommendations.columns)
        if missing_columns:
            errors.append(
                f"Recommendation CSV for user {user_id} is missing columns: "
                f"{sorted(missing_columns)}"
            )
            continue
        seen = set(
            data.ratings.loc[data.ratings["user_id"].eq(user_id), "movie_id"].astype(int)
        )
        leaked = seen & set(recommendations["movie_id"].dropna().astype(int))
        if leaked:
            errors.append(
                f"Recommendations for user {user_id} contain rated movies: {sorted(leaked)}"
            )
        candidate_count = len(index_to_movie) - len(seen & set(index_to_movie.astype(int)))
        expected_count = min(10, candidate_count)
        errors.extend(
            _validate_recommendation_ranking(
                recommendations, "svd", expected_count, user_id
            )
        )
        errors.extend(
            _validate_recommendation_ranking(
                recommendations, "pmf", expected_count, user_id
            )
        )

    try:
        notebook = nbformat.read(
            root / "Movie_Recommender_System.ipynb", as_version=4
        )
        notebook_errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if notebook_errors:
            errors.append("Notebook contains error outputs")
    except Exception as exc:
        errors.append(f"Notebook validation failed: {exc}")

    try:
        importlib.import_module("app")
    except Exception as exc:
        errors.append(f"Streamlit app import failed: {exc}")
    return errors


def main() -> None:
    errors = validate()
    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Project validation passed.")


if __name__ == "__main__":
    main()
