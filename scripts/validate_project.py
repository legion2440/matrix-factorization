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
from utils.interpretability import (
    AUDIT_USER_ROLES,
    FACTOR_INTERPRETATION_COLUMNS,
    LOCAL_EXPLANATION_COLUMNS,
    SIMILARITY_COLUMNS,
)
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
    "reports/baseline_tuning.json",
    "reports/svd_predictions.npy",
    "reports/svd_metadata.json",
    "reports/pmf_convergence.png",
    "reports/predicted_vs_actual.png",
    "reports/rmse_comparison.png",
    "reports/user_comparison.png",
    "reports/top_recommendations.png",
    "reports/evaluated_users.json",
    "reports/pmf_factor_interpretation.csv",
    "reports/pmf_factor_genre_profiles.csv",
    "reports/pmf_latent_factor_heatmap.png",
    "reports/pmf_movie_similarities.csv",
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


def _validate_raw_svd_predictions(
    predictions: np.ndarray,
    expected_shape: tuple[int, int],
) -> list[str]:
    errors: list[str] = []
    if predictions.shape != expected_shape:
        errors.append(
            f"SVD prediction shape {predictions.shape} != expected {expected_shape}"
        )

    all_finite = bool(np.isfinite(predictions).all())
    if not all_finite:
        errors.append("Raw SVD predictions contain non-finite values")
    elif not np.any((predictions < 1.0) | (predictions > 5.0)):
        errors.append(
            "SVD prediction artifact appears clipped; "
            "expected raw values outside [1, 5]"
        )
    return errors


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


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and np.isfinite(float(value))


def _validate_benchmark_metrics(metrics: dict[str, object]) -> list[str]:
    errors: list[str] = []
    required = {
        "Baseline_CF_MSE",
        "Baseline_CF_RMSE",
        "SVD_vs_Baseline_improvement_%",
        "PMF_vs_Baseline_improvement_%",
        "baseline_best_params",
    }
    missing = required - set(metrics)
    if missing:
        errors.append(f"model_metrics.json missing benchmark fields: {sorted(missing)}")
        return errors
    for key in (
        "Baseline_CF_MSE",
        "Baseline_CF_RMSE",
        "SVD_vs_Baseline_improvement_%",
        "PMF_vs_Baseline_improvement_%",
    ):
        if not _is_finite_number(metrics.get(key)):
            errors.append(f"{key} must be finite")
    if _is_finite_number(metrics.get("Baseline_CF_RMSE")) and _is_finite_number(
        metrics.get("SVD_RMSE")
    ):
        if float(metrics["Baseline_CF_RMSE"]) <= float(metrics["SVD_RMSE"]):
            errors.append("Baseline_CF_RMSE must be greater than SVD_RMSE")
    if _is_finite_number(metrics.get("Baseline_CF_RMSE")) and _is_finite_number(
        metrics.get("PMF_RMSE")
    ):
        if float(metrics["Baseline_CF_RMSE"]) <= float(metrics["PMF_RMSE"]):
            errors.append("Baseline_CF_RMSE must be greater than PMF_RMSE")
    return errors


def _validate_baseline_tuning_artifact(payload: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["baseline_tuning.json must contain an object"]
    required = {
        "model",
        "prediction_formula",
        "selection_metric",
        "uses_test_for_tuning",
        "regularization_grid",
        "results",
        "selected",
        "final_refit",
        "test_evaluation",
    }
    missing = required - set(payload)
    if missing:
        errors.append(f"baseline_tuning.json missing fields: {sorted(missing)}")
        return errors
    if payload.get("uses_test_for_tuning") is not False:
        errors.append("Baseline tuning must declare that test data was not used")
    results = payload.get("results")
    selected = payload.get("selected")
    if not isinstance(results, list) or not results:
        errors.append("Baseline tuning results must be a non-empty list")
    else:
        required_row = {
            "user_regularization",
            "item_regularization",
            "n_iterations",
            "validation_mse",
            "validation_rmse",
        }
        for row in results:
            if not isinstance(row, dict) or required_row - set(row):
                errors.append("Baseline tuning row has an incomplete schema")
                break
            if not all(
                _is_finite_number(row.get(key))
                for key in required_row
                if key != "n_iterations"
            ):
                errors.append("Baseline tuning row contains non-finite values")
                break
    if not isinstance(selected, dict):
        errors.append("Baseline tuning selected result must be an object")
    elif results and selected not in results:
        errors.append("Baseline selected result is not present in tuning results")
    final_refit = payload.get("final_refit")
    if not isinstance(final_refit, dict) or final_refit.get(
        "uses_train_plus_validation"
    ) is not True:
        errors.append("Baseline final refit must use train plus validation")
    test_eval = payload.get("test_evaluation")
    if not isinstance(test_eval, dict) or not all(
        _is_finite_number(test_eval.get(key)) for key in ("mse", "rmse")
    ):
        errors.append("Baseline test evaluation must contain finite mse/rmse")
    return errors


def _validate_factor_interpretation(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = set(FACTOR_INTERPRETATION_COLUMNS) - set(frame.columns)
    if missing:
        return [f"Factor interpretation CSV missing columns: {sorted(missing)}"]
    if frame.empty:
        return ["Factor interpretation CSV is empty"]
    factor_ids = frame["factor_index"].astype(int)
    if factor_ids.nunique() < 1:
        errors.append("Factor interpretation must include at least one factor")
    if frame.duplicated(["factor_index", "polarity", "movie_id"]).any():
        errors.append("Factor interpretation contains duplicate factor/movie rows")
    for factor_id, group in frame.groupby("factor_index", sort=True):
        polarities = set(group["polarity"].astype(str))
        if polarities != {"positive", "negative"}:
            errors.append(
                f"Factor {factor_id} must contain positive and negative polarities"
            )
    numeric = frame[["factor_variance", "factor_loading"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        errors.append("Factor interpretation contains non-finite loadings")
    return errors


def _validate_similarity_artifact(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = set(SIMILARITY_COLUMNS) - set(frame.columns)
    if missing:
        return [f"Similarity CSV missing columns: {sorted(missing)}"]
    if frame.empty:
        return ["Similarity CSV is empty"]
    if (
        frame["anchor_movie_id"].astype(int).to_numpy()
        == frame["similar_movie_id"].astype(int).to_numpy()
    ).any():
        errors.append("Similarity CSV contains self matches")
    similarities = frame["cosine_similarity"].to_numpy(dtype=float)
    if not np.isfinite(similarities).all():
        errors.append("Similarity CSV contains non-finite values")
    if ((similarities < -1.000001) | (similarities > 1.000001)).any():
        errors.append("Similarity CSV contains values outside [-1, 1]")
    for anchor_id, group in frame.groupby("anchor_movie_id", sort=True):
        expected = group.sort_values(
            ["cosine_similarity", "similar_movie_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        if not expected.index.equals(group.index):
            errors.append(f"Similarity rows for anchor {anchor_id} are not sorted")
            break
        ranks = group["rank"].to_numpy(dtype=int)
        if not np.array_equal(ranks, np.arange(1, len(group) + 1)):
            errors.append(f"Similarity ranks for anchor {anchor_id} are invalid")
            break
    return errors


def _validate_audit_users_payload(
    evaluated_users: object,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(evaluated_users, list) or len(evaluated_users) != 3:
        return ["evaluated_users.json must contain exactly three records"]
    roles = [row.get("role") for row in evaluated_users if isinstance(row, dict)]
    user_ids = [row.get("user_id") for row in evaluated_users if isinstance(row, dict)]
    if set(roles) != set(AUDIT_USER_ROLES):
        errors.append(f"Audit user roles must be exactly {list(AUDIT_USER_ROLES)}")
    if len(set(user_ids)) != 3:
        errors.append("Audit user IDs must be unique")

    train_counts = train.groupby("user_id").size()
    validation_counts = validation.groupby("user_id").size()
    test_counts = test.groupby("user_id").size()
    by_role: dict[str, dict[str, object]] = {}
    required = {
        "user_id",
        "role",
        "selection_reason",
        "train_ratings",
        "validation_ratings",
        "test_ratings",
        "svd_test_rmse",
        "pmf_test_rmse",
    }
    for row in evaluated_users:
        if not isinstance(row, dict):
            errors.append("Audit user record must be an object")
            continue
        if required - set(row):
            errors.append("Audit user record has an incomplete schema")
            continue
        user_id = int(row["user_id"])
        role = str(row["role"])
        by_role[role] = row
        expected_counts = {
            "train_ratings": int(train_counts.get(user_id, 0)),
            "validation_ratings": int(validation_counts.get(user_id, 0)),
            "test_ratings": int(test_counts.get(user_id, 0)),
        }
        for key, expected in expected_counts.items():
            if int(row[key]) != expected or expected <= 0:
                errors.append(
                    f"Audit user {user_id} has invalid {key}: "
                    f"{row[key]} != {expected}"
                )
        if not _is_finite_number(row.get("svd_test_rmse")) or not _is_finite_number(
            row.get("pmf_test_rmse")
        ):
            errors.append(f"Audit user {user_id} has non-finite per-user RMSE")
    accurate = by_role.get("train_profile_accurate")
    less = by_role.get("train_profile_less_accurate")
    if accurate and less and float(accurate["pmf_test_rmse"]) >= float(
        less["pmf_test_rmse"]
    ):
        errors.append("Accurate audit user's PMF RMSE must be lower than less accurate")
    return errors


def _validate_explanation_artifact(
    explanations: pd.DataFrame,
    recommendations: pd.DataFrame,
    user_id: int,
    role: str,
    pmf: PMFModel,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    known_movie_ids: set[int],
    expected_count: int,
) -> list[str]:
    errors: list[str] = []
    missing = set(LOCAL_EXPLANATION_COLUMNS) - set(explanations.columns)
    if missing:
        return [f"Explanation CSV for user {user_id} missing columns: {sorted(missing)}"]
    if len(explanations) != expected_count:
        errors.append(
            f"Explanation CSV for user {user_id}: expected {expected_count}, "
            f"found {len(explanations)}"
        )
    if not explanations["user_id"].eq(user_id).all():
        errors.append(f"Explanation CSV for user {user_id} contains another user_id")
    if not explanations["role"].eq(role).all():
        errors.append(f"Explanation CSV for user {user_id} contains another role")
    if set(explanations["movie_id"].astype(int)) & known_movie_ids:
        errors.append(f"Explanation CSV for user {user_id} contains known movies")
    numeric_columns = [
        "raw_pmf_ranking_score",
        "clipped_displayed_rating",
        "global_mean_contribution",
        "user_bias_contribution",
        "item_bias_contribution",
        "total_latent_dot_product",
        "component_sum",
        "reconstruction_error",
        "nearest_known_similarity",
    ]
    numeric_values = explanations[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        errors.append(f"Explanation CSV for user {user_id} contains non-finite values")
        return errors
    raw = explanations["raw_pmf_ranking_score"].to_numpy(dtype=float)
    displayed = explanations["clipped_displayed_rating"].to_numpy(dtype=float)
    if not np.allclose(displayed, np.clip(raw, 1.0, 5.0), rtol=0.0, atol=1e-6):
        errors.append(f"Explanation displayed ratings for user {user_id} are invalid")
    component_sum = (
        explanations["global_mean_contribution"].to_numpy(dtype=float)
        + explanations["user_bias_contribution"].to_numpy(dtype=float)
        + explanations["item_bias_contribution"].to_numpy(dtype=float)
        + explanations["total_latent_dot_product"].to_numpy(dtype=float)
    )
    if not np.allclose(component_sum, raw, rtol=0.0, atol=1e-5):
        errors.append(f"Explanation decomposition for user {user_id} is broken")
    if not np.allclose(
        explanations["component_sum"].to_numpy(dtype=float), raw, rtol=0.0, atol=1e-5
    ):
        errors.append(f"Explanation component_sum for user {user_id} is invalid")
    if not np.all(np.abs(explanations["reconstruction_error"].to_numpy(float)) <= 1e-5):
        errors.append(f"Explanation reconstruction error for user {user_id} is too high")
    if (
        explanations["nearest_known_similarity"].to_numpy(dtype=float) < -1.000001
    ).any() or (
        explanations["nearest_known_similarity"].to_numpy(dtype=float) > 1.000001
    ).any():
        errors.append(f"Nearest-known similarity for user {user_id} is outside [-1, 1]")

    pmf._check_fitted()
    user_index = user_to_index[user_id]
    for row in explanations.itertuples(index=False):
        item_index = movie_to_index[int(row.movie_id)]
        contributions = (
            pmf.user_factors[user_index].astype(np.float64)
            * pmf.item_factors[item_index].astype(np.float64)
        )
        order = np.lexsort((np.arange(contributions.size), -np.abs(contributions)))[:3]
        expected_indices = [int(index) for index in order]
        actual_indices = [
            int(row.top_factor_1_index),
            int(row.top_factor_2_index),
            int(row.top_factor_3_index),
        ]
        if actual_indices != expected_indices:
            errors.append(f"Top factor indices for user {user_id} are inconsistent")
            break
        actual_values = np.array(
            [
                row.top_factor_1_contribution,
                row.top_factor_2_contribution,
                row.top_factor_3_contribution,
            ],
            dtype=float,
        )
        if not np.allclose(actual_values, contributions[order], rtol=0.0, atol=1e-6):
            errors.append(f"Top factor values for user {user_id} are inconsistent")
            break
    pmf_movies = set(
        recommendations.dropna(subset=["pmf_rank"])["movie_id"].astype(int).tolist()
    )
    if set(explanations["movie_id"].astype(int)) != pmf_movies:
        errors.append(f"Explanation movies for user {user_id} do not match PMF Top-N")
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
        with (root / "reports" / "baseline_tuning.json").open(
            encoding="utf-8"
        ) as handle:
            baseline_tuning = json.load(handle)
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
    errors.extend(_validate_benchmark_metrics(metrics))
    errors.extend(_validate_baseline_tuning_artifact(baseline_tuning))
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
    errors.extend(_validate_raw_svd_predictions(predictions, expected_shape))
    if svd_metadata.get("prediction_scale") != "raw_unclipped":
        errors.append("SVD metadata does not declare raw_unclipped prediction scale")

    train = pd.read_csv(root / "processed" / "train_ratings.csv")
    validation = pd.read_csv(root / "processed" / "validation_ratings.csv")
    test = pd.read_csv(root / "processed" / "test_ratings.csv")
    errors.extend(_validate_audit_users_payload(evaluated_users, train, validation, test))

    try:
        factor_interpretation = pd.read_csv(
            root / "reports" / "pmf_factor_interpretation.csv"
        )
        factor_genre_profiles = pd.read_csv(
            root / "reports" / "pmf_factor_genre_profiles.csv"
        )
        similarities = pd.read_csv(root / "reports" / "pmf_movie_similarities.csv")
        errors.extend(_validate_factor_interpretation(factor_interpretation))
        if factor_genre_profiles.empty:
            errors.append("PMF factor genre profile CSV is empty")
        errors.extend(_validate_similarity_artifact(similarities))
    except Exception as exc:
        errors.append(f"Interpretability artifact validation failed: {exc}")
    heatmap = root / "reports" / "pmf_latent_factor_heatmap.png"
    if heatmap.exists() and heatmap.stat().st_size == 0:
        errors.append("PMF latent factor heatmap is empty")

    pmf: PMFModel | None = None
    try:
        pmf = PMFModel.load(root / "reports" / "pmf_factors")
        if pmf.user_factors.shape[0] != len(index_to_user):
            errors.append("PMF user factors do not align with mappings")
        if pmf.item_factors.shape[0] != len(index_to_movie):
            errors.append("PMF item factors do not align with mappings")
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
        role = str(selection.get("role"))
        recommendation_path = root / "reports" / f"user_{user_id}_recommendations.csv"
        explanation_path = root / "reports" / f"user_{user_id}_explanations.csv"
        explanation_plot = root / "reports" / f"user_{user_id}_explanation.png"
        if not recommendation_path.exists():
            errors.append(f"Missing recommendation CSV for user {user_id}")
            continue
        if not explanation_path.exists():
            errors.append(f"Missing explanation CSV for user {user_id}")
            continue
        if not explanation_plot.exists():
            errors.append(f"Missing explanation PNG for user {user_id}")
            continue
        if explanation_plot.stat().st_size == 0:
            errors.append(f"Explanation PNG for user {user_id} is empty")
        recommendations = pd.read_csv(recommendation_path)
        explanations = pd.read_csv(explanation_path)
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
        if pmf is not None:
            errors.extend(
                _validate_explanation_artifact(
                    explanations,
                    recommendations,
                    user_id,
                    role,
                    pmf,
                    user_to_index,
                    movie_to_index,
                    seen,
                    expected_count,
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
