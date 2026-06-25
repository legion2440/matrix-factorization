from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from models.pmf_model import PMFModel
from utils.artifacts import (
    REQUIRED_USER_ARTIFACT_SUFFIXES,
    USER_ARTIFACT_PATTERN,
    prepare_svd_rank_tuning,
)
from utils.data_loader import load_movielens
from utils.interpretability import (
    EVALUATION_USER_ROLES,
    FACTOR_INTERPRETATION_COLUMNS,
    LOCAL_EXPLANATION_COLUMNS,
    RANKING_CASE_COLUMNS,
    SIMILARITY_COLUMNS,
)
from utils.matrix_creation import load_mappings
from utils.ranking_evaluation import MODEL_PREFIXES, aggregate_ranking_metrics
from utils.split import deterministic_user_split


REQUIRED_PATHS = [
    "data/ratings.dat",
    "data/users.dat",
    "data/movies.dat",
    "processed/train_ratings.csv",
    "processed/validation_ratings.csv",
    "processed/test_ratings.csv",
    "processed/ranking_train_ratings.csv",
    "processed/ranking_targets.csv",
    "processed/user_item_matrix.csv",
    "processed/mappings/user_to_index.json",
    "processed/mappings/movie_to_index.json",
    "processed/mappings/index_to_user.json",
    "processed/mappings/index_to_movie.json",
    "reports/model_metrics.json",
    "reports/bias_baseline_tuning.json",
    "reports/item_knn_tuning.json",
    "reports/ranking_protocol.json",
    "reports/ranking_metrics.json",
    "reports/ranking_results.csv",
    "reports/ranking_comparison.png",
    "reports/svd_predictions.npy",
    "reports/svd_metadata.json",
    "reports/svd_tuning.json",
    "reports/svd_rank_tuning_mse.png",
    "reports/svd_rank_tuning_rmse.png",
    "reports/pmf_convergence.json",
    "reports/pmf_convergence_mse.png",
    "reports/pmf_convergence_rmse.png",
    "reports/predicted_vs_actual.png",
    "reports/model_mse_comparison.png",
    "reports/model_rmse_comparison.png",
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
    "models/bias_baseline.py",
    "models/item_knn.py",
    "scripts/build_analysis_notebook.py",
    "utils/eda.py",
    "utils/rating_ranking_analysis.py",
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


def _validate_mse_rmse_pair(
    mse_value: object,
    rmse_value: object,
    label: str,
    *,
    tolerance: float = 1e-10,
) -> list[str]:
    if not _is_finite_number(mse_value) or not _is_finite_number(rmse_value):
        return [f"{label} MSE/RMSE values must be finite"]
    if not np.isclose(
        float(mse_value),
        float(rmse_value) ** 2,
        rtol=tolerance,
        atol=tolerance,
    ):
        return [f"{label} MSE is inconsistent with RMSE ** 2"]
    return []


def _validate_pmf_convergence_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["pmf_convergence.json must contain an object"]
    history = payload.get("history")
    if not isinstance(history, list) or not history:
        return ["pmf_convergence.json must contain non-empty history"]
    errors: list[str] = []
    required = {
        "epoch",
        "train_mse",
        "train_rmse",
        "validation_mse",
        "validation_rmse",
    }
    epochs: list[int] = []
    for row in history:
        if not isinstance(row, dict) or required - set(row):
            errors.append("PMF convergence history row has an incomplete schema")
            break
        epochs.append(int(row["epoch"]))
        errors.extend(
            _validate_mse_rmse_pair(
                row["train_mse"], row["train_rmse"], "PMF train convergence"
            )
        )
        errors.extend(
            _validate_mse_rmse_pair(
                row["validation_mse"],
                row["validation_rmse"],
                "PMF validation convergence",
            )
        )
    if epochs != list(range(1, len(epochs) + 1)):
        errors.append("PMF convergence epochs must be contiguous and 1-based")
    selected_epoch = payload.get("selected_epoch")
    if not isinstance(selected_epoch, int) or selected_epoch not in epochs:
        errors.append("PMF selected epoch is absent from convergence history")
    stopping = payload.get("early_stopping")
    if not isinstance(stopping, dict):
        errors.append("PMF convergence early-stopping context is missing")
    else:
        if int(stopping.get("epochs_run", -1)) != len(history):
            errors.append("PMF convergence epochs_run is inconsistent")
        if int(stopping.get("patience", -1)) != 8:
            errors.append("PMF convergence patience changed")
        if not np.isclose(
            float(stopping.get("min_delta", np.nan)),
            5e-5,
            rtol=0.0,
            atol=0.0,
        ):
            errors.append("PMF convergence min_delta changed")
    return errors


def _validate_svd_tuning_payload(
    payload: object,
    selected_rank: int,
    selected_item_bias_regularization: float,
) -> list[str]:
    if not isinstance(payload, list) or not payload:
        return ["svd_tuning.json must contain a non-empty list"]
    errors: list[str] = []
    required = {
        "n_factors",
        "item_bias_regularization",
        "validation_mse",
        "validation_rmse",
    }
    for row in payload:
        if not isinstance(row, dict) or required - set(row):
            errors.append("SVD tuning row has an incomplete schema")
            return errors
        errors.extend(
            _validate_mse_rmse_pair(
                row["validation_mse"],
                row["validation_rmse"],
                "SVD tuning validation",
            )
        )
    try:
        prepared = prepare_svd_rank_tuning(
            payload,
            selected_rank=selected_rank,
            selected_item_bias_regularization=selected_item_bias_regularization,
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"SVD rank tuning preparation failed: {exc}")
        return errors
    if prepared["n_factors"].astype(int).tolist() != [5, 10, 20, 40, 60]:
        errors.append("SVD rank tuning curve has an unexpected rank grid")
    return errors


def _validate_user_artifact_manifest(
    reports_dir: Path,
    evaluated_users: object,
) -> list[str]:
    if not isinstance(evaluated_users, list):
        return ["Cannot validate user artifacts without evaluated_users records"]
    current_ids = {
        int(row["user_id"])
        for row in evaluated_users
        if isinstance(row, dict) and "user_id" in row
    }
    errors: list[str] = []
    artifact_ids: set[int] = set()
    for path in reports_dir.glob("user_*"):
        match = USER_ARTIFACT_PATTERN.match(path.name)
        if not match:
            continue
        user_id = int(match.group(1))
        artifact_ids.add(user_id)
        if user_id not in current_ids:
            errors.append(f"Orphan user artifact is not in evaluated_users.json: {path.name}")
    for user_id in sorted(current_ids):
        for suffix in REQUIRED_USER_ARTIFACT_SUFFIXES:
            path = reports_dir / f"user_{user_id}_{suffix}"
            if not path.exists():
                errors.append(
                    f"Missing required user artifact for {user_id}: {path.name}"
                )
            elif path.stat().st_size == 0:
                errors.append(f"User artifact is empty: {path.name}")
    if artifact_ids != current_ids:
        errors.append(
            "User artifact IDs do not match evaluated_users.json: "
            f"artifacts={sorted(artifact_ids)}, manifest={sorted(current_ids)}"
        )
    return errors


def _validate_user_item_matrix(
    path: Path,
    index_to_user: np.ndarray,
    index_to_movie: np.ndarray,
) -> list[str]:
    errors: list[str] = []
    try:
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n\r").split(",")
            first_row = handle.readline().rstrip("\n\r").split(",")
    except OSError as exc:
        return [f"Could not read user_item_matrix.csv: {exc}"]
    expected_header = ["user_id", *[str(int(value)) for value in index_to_movie]]
    if header != expected_header:
        errors.append("user_item_matrix.csv header does not align with movie mappings")
    if len(first_row) != len(expected_header):
        errors.append("user_item_matrix.csv first row has the wrong width")
    elif first_row and int(float(first_row[0])) != int(index_to_user[0]):
        errors.append("user_item_matrix.csv first user does not align with mappings")
    return errors


def _validate_benchmark_metrics(metrics: dict[str, object]) -> list[str]:
    errors: list[str] = []
    required = {
        "BiasBaseline_MSE",
        "BiasBaseline_RMSE",
        "ItemKNN_MSE",
        "ItemKNN_RMSE",
        "SVD_MSE",
        "SVD_RMSE",
        "PMF_MSE",
        "PMF_RMSE",
        "ItemKNN_vs_BiasBaseline_improvement_%",
        "SVD_vs_BiasBaseline_improvement_%",
        "PMF_vs_BiasBaseline_improvement_%",
        "SVD_vs_ItemKNN_improvement_%",
        "PMF_vs_ItemKNN_improvement_%",
        "PMF_vs_SVD_improvement_%",
        "ItemKNN_beats_BiasBaseline",
        "SVD_beats_BiasBaseline",
        "PMF_beats_BiasBaseline",
        "SVD_beats_ItemKNN",
        "PMF_beats_ItemKNN",
        "bias_baseline_best_params",
        "item_knn_best_params",
    }
    missing = required - set(metrics)
    if missing:
        errors.append(f"model_metrics.json missing benchmark fields: {sorted(missing)}")
        return errors
    numeric_keys = [
        key
        for key in required
        if key.endswith("_MSE")
        or key.endswith("_RMSE")
        or key.endswith("improvement_%")
    ]
    for key in numeric_keys:
        if not _is_finite_number(metrics.get(key)):
            errors.append(f"{key} must be finite")
    if errors:
        return errors
    for model in ("BiasBaseline", "ItemKNN", "SVD", "PMF"):
        errors.extend(
            _validate_mse_rmse_pair(
                metrics[f"{model}_MSE"],
                metrics[f"{model}_RMSE"],
                model,
            )
        )

    comparisons = {
        "ItemKNN_beats_BiasBaseline": ("ItemKNN_RMSE", "BiasBaseline_RMSE"),
        "SVD_beats_BiasBaseline": ("SVD_RMSE", "BiasBaseline_RMSE"),
        "PMF_beats_BiasBaseline": ("PMF_RMSE", "BiasBaseline_RMSE"),
        "SVD_beats_ItemKNN": ("SVD_RMSE", "ItemKNN_RMSE"),
        "PMF_beats_ItemKNN": ("PMF_RMSE", "ItemKNN_RMSE"),
    }
    for flag, (model_key, reference_key) in comparisons.items():
        expected = float(metrics[model_key]) < float(metrics[reference_key])
        if metrics.get(flag) is not expected:
            errors.append(f"{flag} does not match the stored RMSE values")

    improvements = {
        "ItemKNN_vs_BiasBaseline_improvement_%": (
            "ItemKNN_RMSE",
            "BiasBaseline_RMSE",
        ),
        "SVD_vs_BiasBaseline_improvement_%": (
            "SVD_RMSE",
            "BiasBaseline_RMSE",
        ),
        "PMF_vs_BiasBaseline_improvement_%": (
            "PMF_RMSE",
            "BiasBaseline_RMSE",
        ),
        "SVD_vs_ItemKNN_improvement_%": ("SVD_RMSE", "ItemKNN_RMSE"),
        "PMF_vs_ItemKNN_improvement_%": ("PMF_RMSE", "ItemKNN_RMSE"),
        "PMF_vs_SVD_improvement_%": ("PMF_RMSE", "SVD_RMSE"),
    }
    for field, (model_key, reference_key) in improvements.items():
        expected = (
            float(metrics[reference_key]) - float(metrics[model_key])
        ) / float(metrics[reference_key]) * 100.0
        if not np.isclose(float(metrics[field]), expected, rtol=0.0, atol=1e-9):
            errors.append(f"{field} does not match the stored RMSE values")
    return errors


def _validate_bias_baseline_tuning_artifact(payload: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["bias_baseline_tuning.json must contain an object"]
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
        errors.append(
            f"bias_baseline_tuning.json missing fields: {sorted(missing)}"
        )
        return errors
    if payload.get("uses_test_for_tuning") is not False:
        errors.append(
            "Bias baseline tuning must declare that test data was not used"
        )
    results = payload.get("results")
    selected = payload.get("selected")
    if not isinstance(results, list) or not results:
        errors.append("Bias baseline tuning results must be a non-empty list")
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
                errors.append(
                    "Bias baseline tuning row has an incomplete schema"
                )
                break
            if not all(
                _is_finite_number(row.get(key))
                for key in required_row
                if key != "n_iterations"
            ):
                errors.append(
                    "Bias baseline tuning row contains non-finite values"
                )
                break
    if not isinstance(selected, dict):
        errors.append("Bias baseline selected result must be an object")
    elif results and selected not in results:
        errors.append(
            "Bias baseline selected result is not present in tuning results"
        )
    final_refit = payload.get("final_refit")
    if not isinstance(final_refit, dict) or final_refit.get(
        "uses_train_plus_validation"
    ) is not True:
        errors.append("Bias baseline final refit must use train plus validation")
    test_eval = payload.get("test_evaluation")
    if not isinstance(test_eval, dict) or not all(
        _is_finite_number(test_eval.get(key)) for key in ("mse", "rmse")
    ):
        errors.append(
            "Bias baseline test evaluation must contain finite mse/rmse"
        )
    return errors


def _validate_item_knn_tuning_artifact(payload: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["item_knn_tuning.json must contain an object"]
    required = {
        "model",
        "prediction_formula",
        "similarity_definition",
        "neighborhood_definition",
        "neighborhood_ordering",
        "parameter_grid",
        "selection_metric",
        "selection_tie_break",
        "uses_test_for_tuning",
        "results",
        "selected",
        "final_refit",
        "test_evaluation",
    }
    missing = required - set(payload)
    if missing:
        return [f"item_knn_tuning.json missing fields: {sorted(missing)}"]
    if payload.get("model") != "ItemKNN":
        errors.append("item-kNN tuning model name must be ItemKNN")
    if payload.get("uses_test_for_tuning") is not False:
        errors.append("item-kNN tuning must declare that test data was not used")
    expected_grid = {
        "k": [20, 40, 80],
        "shrinkage": [10.0, 50.0, 100.0],
        "min_common": 3,
    }
    if payload.get("parameter_grid") != expected_grid:
        errors.append("item-kNN tuning grid must be the required 3 x 3 grid")
    results = payload.get("results")
    expected_combinations = {
        (k, shrinkage, 3)
        for k in (20, 40, 80)
        for shrinkage in (10.0, 50.0, 100.0)
    }
    if not isinstance(results, list) or len(results) != 9:
        errors.append("item-kNN tuning must contain exactly 9 validation rows")
    else:
        combinations = {
            (row.get("k"), row.get("shrinkage"), row.get("min_common"))
            for row in results
            if isinstance(row, dict)
        }
        if combinations != expected_combinations:
            errors.append("item-kNN tuning rows do not match the required grid")
        for row in results:
            required_row = {
                "k",
                "shrinkage",
                "min_common",
                "validation_mse",
                "validation_rmse",
            }
            if not isinstance(row, dict) or required_row - set(row):
                errors.append("item-kNN tuning row has an incomplete schema")
                break
            if not all(_is_finite_number(row.get(key)) for key in required_row):
                errors.append("item-kNN tuning row contains invalid values")
                break
    selected = payload.get("selected")
    if not isinstance(selected, dict) or not isinstance(results, list):
        errors.append("item-kNN selected configuration must be an object")
    elif selected not in results:
        errors.append("item-kNN selected row is absent from validation results")
    elif selected != min(
        results,
        key=lambda row: (
            row["validation_rmse"],
            row["k"],
            -row["shrinkage"],
        ),
    ):
        errors.append("item-kNN selected row violates the deterministic tie-break")
    if payload.get("neighborhood_ordering") != [
        "absolute shrunk similarity descending",
        "signed shrunk similarity descending",
        "movie ID ascending",
    ]:
        errors.append("item-kNN neighborhood ordering declaration is invalid")
    final_refit = payload.get("final_refit")
    diagnostics = (
        final_refit.get("diagnostics") if isinstance(final_refit, dict) else None
    )
    if not isinstance(final_refit, dict) or final_refit.get(
        "uses_train_plus_validation"
    ) is not True:
        errors.append("item-kNN final refit must use train plus validation")
    if not isinstance(diagnostics, dict):
        errors.append("item-kNN final refit diagnostics are missing")
    else:
        if diagnostics.get("similarities_finite") is not True:
            errors.append("item-kNN similarities must be finite")
        if diagnostics.get("self_neighbor_count") != 0:
            errors.append("item-kNN self-neighbors must be absent")
        if diagnostics.get("deterministic_ordering_verified") is not True:
            errors.append("item-kNN stored neighbor ordering is invalid")
        minimum_common = diagnostics.get("minimum_common_users")
        if minimum_common is not None and int(minimum_common) < 3:
            errors.append("item-kNN min_common was not enforced")
    test_eval = payload.get("test_evaluation")
    if not isinstance(test_eval, dict) or not all(
        _is_finite_number(test_eval.get(key)) for key in ("mse", "rmse")
    ):
        errors.append("item-kNN test evaluation must contain finite mse/rmse")
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


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _validate_ranking_artifacts(
    ranking_train: pd.DataFrame,
    ranking_targets: pd.DataFrame,
    ranking_results: pd.DataFrame,
    ranking_metrics: object,
    ranking_protocol: object,
) -> list[str]:
    errors: list[str] = []
    target_required = {
        "user_id",
        "movie_id",
        "rating",
        "timestamp",
        "prior_history_count",
        "target_item_support",
    }
    if target_required - set(ranking_targets.columns):
        return ["ranking_targets.csv has an incomplete schema"]
    if ranking_targets.empty or ranking_targets["user_id"].duplicated().any():
        errors.append("ranking targets must contain one row per eligible user")

    result_required = {
        "user_id",
        "target_movie_id",
        "target_title",
        "target_genres",
        "target_rating",
        "target_timestamp",
        "prior_history_count",
        "candidate_count",
    }
    for prefix in MODEL_PREFIXES.values():
        result_required.update(
            {
                f"{prefix}_target_rank",
                f"{prefix}_raw_target_score",
                f"{prefix}_hit_at_5",
                f"{prefix}_hit_at_10",
                f"{prefix}_ndcg_at_5",
                f"{prefix}_ndcg_at_10",
                f"{prefix}_mrr_at_5",
                f"{prefix}_mrr_at_10",
            }
        )
    missing_results = result_required - set(ranking_results.columns)
    if missing_results:
        return [
            f"ranking_results.csv missing columns: {sorted(missing_results)}"
        ]

    if not isinstance(ranking_protocol, dict):
        errors.append("ranking_protocol.json must contain an object")
    else:
        if (
            ranking_protocol.get("protocol")
            != "next-positive recovery under temporal leave-one-positive-out"
        ):
            errors.append("ranking protocol name is invalid")
        if ranking_protocol.get("full_catalog_candidates") is not True:
            errors.append("ranking protocol must use full-catalog candidates")
        if ranking_protocol.get("sampled_negatives") is not False:
            errors.append("ranking protocol must declare no sampled negatives")
        if ranking_protocol.get("history_rule") != "timestamp < target_timestamp":
            errors.append("ranking protocol must use a strict temporal prefix")
        if int(ranking_protocol.get("min_prior_interactions", 0)) != 20:
            errors.append("ranking protocol minimum history must be 20")
        if int(ranking_protocol.get("min_target_item_support", 0)) != 10:
            errors.append("ranking protocol target support minimum must be 10")
        frozen = ranking_protocol.get("frozen_model_parameters", {})
        svd_frozen = frozen.get("SVD", {}) if isinstance(frozen, dict) else {}
        pmf_frozen = frozen.get("PMF", {}) if isinstance(frozen, dict) else {}
        if (
            svd_frozen.get("n_factors") != 20
            or float(svd_frozen.get("item_bias_regularization", np.nan)) != 5.0
            or svd_frozen.get("random_state") != 42
        ):
            errors.append("ranking SVD parameters are not the frozen selected values")
        expected_pmf = {
            "n_factors": 128,
            "learning_rate": 0.006,
            "factor_regularization": 0.06,
            "bias_regularization": 0.02,
            "epochs": 53,
            "random_state": 42,
            "uses_ranking_targets_for_tuning": False,
        }
        if any(pmf_frozen.get(key) != value for key, value in expected_pmf.items()):
            errors.append("ranking PMF parameters are not the frozen selected values")

    supported_movies = set(ranking_train["movie_id"].astype(int))
    support_counts = ranking_train.groupby("movie_id").size()
    histories = {
        int(user_id): group
        for user_id, group in ranking_train.groupby("user_id", sort=True)
    }
    targets_by_user = ranking_targets.set_index("user_id")
    results_by_user = ranking_results.set_index("user_id")
    target_users = set(targets_by_user.index.astype(int))
    result_users = set(results_by_user.index.astype(int))
    if target_users != result_users:
        errors.append("ranking result users do not match ranking target users")

    for user_id in sorted(target_users & result_users):
        target = targets_by_user.loc[user_id]
        result = results_by_user.loc[user_id]
        target_movie_id = int(target["movie_id"])
        target_timestamp = int(target["timestamp"])
        history = histories.get(user_id)
        if history is None or history.empty:
            errors.append(f"ranking user {user_id} has no prefix history")
            continue
        if not history["timestamp"].lt(target_timestamp).all():
            errors.append(
                f"ranking user {user_id} has same-timestamp or later training rows"
            )
        if len(history) != int(target["prior_history_count"]) or len(history) < 20:
            errors.append(f"ranking user {user_id} has an invalid history count")
        if ((history["movie_id"].astype(int)) == target_movie_id).any():
            errors.append(f"ranking target for user {user_id} leaked into history")
        if float(target["rating"]) < 4.0:
            errors.append(f"ranking target for user {user_id} is not positive")
        support = int(support_counts.get(target_movie_id, 0))
        if support < 10 or support != int(target["target_item_support"]):
            errors.append(f"ranking target for user {user_id} has invalid support")
        if target_movie_id not in supported_movies:
            errors.append(f"ranking target for user {user_id} is unsupported")

        history_movies = set(history["movie_id"].astype(int))
        expected_candidate_count = len(supported_movies - history_movies)
        if target_movie_id not in supported_movies - history_movies:
            errors.append(f"ranking target for user {user_id} left the candidate set")
        if int(result["candidate_count"]) != expected_candidate_count:
            errors.append(f"ranking candidate count for user {user_id} is invalid")
        if expected_candidate_count <= 0:
            errors.append(f"ranking user {user_id} has no candidates")
        if (
            int(result["target_movie_id"]) != target_movie_id
            or int(result["target_timestamp"]) != target_timestamp
            or float(result["target_rating"]) != float(target["rating"])
            or int(result["prior_history_count"]) != len(history)
        ):
            errors.append(f"ranking target fields for user {user_id} do not match")

        for prefix in MODEL_PREFIXES.values():
            rank = int(result[f"{prefix}_target_rank"])
            if not 1 <= rank <= expected_candidate_count:
                errors.append(f"{prefix} target rank for user {user_id} is invalid")
            if not _is_finite_number(result[f"{prefix}_raw_target_score"]):
                errors.append(
                    f"{prefix} target score for user {user_id} is non-finite"
                )
            for cutoff in (5, 10):
                expected_hit = rank <= cutoff
                expected_ndcg = (
                    float(1.0 / np.log2(rank + 1)) if expected_hit else 0.0
                )
                expected_mrr = float(1.0 / rank) if expected_hit else 0.0
                if _as_bool(result[f"{prefix}_hit_at_{cutoff}"]) != expected_hit:
                    errors.append(
                        f"{prefix} Hit@{cutoff} for user {user_id} is invalid"
                    )
                if not np.isclose(
                    float(result[f"{prefix}_ndcg_at_{cutoff}"]),
                    expected_ndcg,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    errors.append(
                        f"{prefix} NDCG@{cutoff} for user {user_id} is invalid"
                    )
                if not np.isclose(
                    float(result[f"{prefix}_mrr_at_{cutoff}"]),
                    expected_mrr,
                    rtol=0.0,
                    atol=1e-12,
                ):
                    errors.append(
                        f"{prefix} MRR@{cutoff} for user {user_id} is invalid"
                    )

    if not isinstance(ranking_metrics, dict):
        errors.append("ranking_metrics.json must contain an object")
    else:
        if ranking_metrics.get("model_names") != list(MODEL_PREFIXES):
            errors.append("ranking metric model names are invalid")
        if ranking_metrics.get("full_catalog_candidates") is not True:
            errors.append("ranking metrics must declare full-catalog candidates")
        if ranking_metrics.get("sampled_negatives") is not False:
            errors.append("ranking metrics must declare no sampled negatives")
        if any("Recall@" in key for key in ranking_metrics):
            errors.append("ranking metrics must not duplicate HitRate as Recall")
        try:
            recomputed = aggregate_ranking_metrics(ranking_results)
            for model_name in MODEL_PREFIXES:
                actual = ranking_metrics["models"][model_name]
                expected = recomputed["models"][model_name]
                for key, value in expected.items():
                    if key == "eligible_user_count":
                        if int(actual.get(key, -1)) != int(value):
                            errors.append(
                                f"{model_name} ranking user count is inconsistent"
                            )
                    elif not np.isclose(
                        float(actual.get(key, np.nan)),
                        float(value),
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        errors.append(
                            f"{model_name} aggregate ranking metric {key} is inconsistent"
                        )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"ranking metric reconstruction failed: {exc}")
    return errors


def _validate_evaluation_users_payload(
    evaluated_users: object,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    ranking_results: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(evaluated_users, list) or len(evaluated_users) != 3:
        return ["evaluated_users.json must contain exactly three records"]
    roles = [row.get("role") for row in evaluated_users if isinstance(row, dict)]
    user_ids = [row.get("user_id") for row in evaluated_users if isinstance(row, dict)]
    if set(roles) != set(EVALUATION_USER_ROLES):
        errors.append(
            "Evaluation profile roles must be exactly "
            f"{list(EVALUATION_USER_ROLES)}"
        )
    if len(set(user_ids)) != 3:
        errors.append("Evaluation profile user IDs must be unique")
    if set(user_ids) != {2739, 2505, 2210}:
        errors.append(
            "Evaluation profile users must remain exactly 2739, 2505, and 2210"
        )

    train_counts = train.groupby("user_id").size()
    validation_counts = validation.groupby("user_id").size()
    test_counts = test.groupby("user_id").size()
    by_role: dict[str, dict[str, object]] = {}
    required = {
        "user_id",
        "role",
        "ranking_case",
        "selection_reason",
        "train_ratings",
        "validation_ratings",
        "test_ratings",
        "svd_test_rmse",
        "pmf_test_rmse",
        "ranking_target_movie_id",
        "ranking_target_title",
        "ranking_target_rating",
        "ranking_target_timestamp",
        "ranking_history_count",
        "ranking_candidate_count",
        "bias_target_rank",
        "item_knn_target_rank",
        "svd_target_rank",
        "pmf_target_rank",
        "bias_hit_at_10",
        "item_knn_hit_at_10",
        "svd_hit_at_10",
        "pmf_hit_at_10",
    }
    ranking_by_user = ranking_results.set_index("user_id")
    for row in evaluated_users:
        if not isinstance(row, dict):
            errors.append("Evaluation profile record must be an object")
            continue
        if required - set(row):
            errors.append("Evaluation profile record has an incomplete schema")
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
                    f"Evaluation profile {user_id} has invalid {key}: "
                    f"{row[key]} != {expected}"
                )
        if not _is_finite_number(row.get("svd_test_rmse")) or not _is_finite_number(
            row.get("pmf_test_rmse")
        ):
            errors.append(
                f"Evaluation profile {user_id} has non-finite per-user RMSE"
            )
        if user_id not in ranking_by_user.index:
            errors.append(
                f"Evaluation profile {user_id} is absent from ranking results"
            )
            continue
        ranking_row = ranking_by_user.loc[user_id]
        expected_fields = {
            "ranking_target_movie_id": int(ranking_row["target_movie_id"]),
            "ranking_target_title": str(ranking_row["target_title"]),
            "ranking_target_rating": float(ranking_row["target_rating"]),
            "ranking_target_timestamp": int(ranking_row["target_timestamp"]),
            "ranking_history_count": int(ranking_row["prior_history_count"]),
            "ranking_candidate_count": int(ranking_row["candidate_count"]),
            "bias_target_rank": int(ranking_row["bias_target_rank"]),
            "item_knn_target_rank": int(ranking_row["item_knn_target_rank"]),
            "svd_target_rank": int(ranking_row["svd_target_rank"]),
            "pmf_target_rank": int(ranking_row["pmf_target_rank"]),
        }
        for key, expected in expected_fields.items():
            if row[key] != expected:
                errors.append(
                    f"Evaluation profile {user_id} has inconsistent {key}"
                )
        for prefix in ("bias", "item_knn", "svd", "pmf"):
            key = f"{prefix}_hit_at_10"
            if _as_bool(row[key]) != _as_bool(ranking_row[key]):
                errors.append(
                    f"Evaluation profile {user_id} has inconsistent {key}"
                )
    accurate = by_role.get("train_profile_accurate")
    less = by_role.get("train_profile_less_accurate")
    test_case = by_role.get("test_case")
    if accurate:
        if (
            accurate.get("ranking_case") != "pmf_hit_at_10"
            or not _as_bool(accurate.get("pmf_hit_at_10"))
            or int(accurate.get("pmf_target_rank", 0)) > 10
        ):
            errors.append("Accurate evaluation profile must be a PMF Hit@10")
    if less:
        if (
            less.get("ranking_case") != "pmf_miss_at_10"
            or _as_bool(less.get("pmf_hit_at_10"))
            or int(less.get("pmf_target_rank", 0)) <= 10
        ):
            errors.append("Less-accurate evaluation profile must be a PMF miss")
    if test_case and test_case.get("ranking_case") != "representative_target_rank":
        errors.append("Test case must use the representative target-rank role")
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


def _validate_ranking_case_artifact(
    ranking_case: pd.DataFrame,
    ranking_row: pd.Series,
    user_id: int,
    role: str,
) -> list[str]:
    errors: list[str] = []
    missing = set(RANKING_CASE_COLUMNS) - set(ranking_case.columns)
    if missing:
        return [
            f"Ranking case CSV for user {user_id} missing columns: {sorted(missing)}"
        ]
    if len(ranking_case) != 1:
        return [f"Ranking case CSV for user {user_id} must contain one row"]
    row = ranking_case.iloc[0]
    if int(row["user_id"]) != user_id or str(row["role"]) != role:
        errors.append(f"Ranking case CSV for user {user_id} has wrong identity")
    if (
        int(row["target_movie_id"]) != int(ranking_row["target_movie_id"])
        or int(row["target_timestamp"]) != int(ranking_row["target_timestamp"])
        or int(row["candidate_count"]) != int(ranking_row["candidate_count"])
        or int(row["prior_history_count"])
        != int(ranking_row["prior_history_count"])
    ):
        errors.append(f"Ranking case CSV for user {user_id} has wrong target fields")
    for prefix in ("bias", "item_knn", "svd", "pmf"):
        if int(row[f"{prefix}_target_rank"]) != int(
            ranking_row[f"{prefix}_target_rank"]
        ):
            errors.append(
                f"Ranking case {prefix} rank for user {user_id} is inconsistent"
            )
        if not np.isclose(
            float(row[f"{prefix}_raw_target_score"]),
            float(ranking_row[f"{prefix}_raw_target_score"]),
            rtol=0.0,
            atol=1e-6,
        ):
            errors.append(
                f"Ranking case {prefix} score for user {user_id} is inconsistent"
            )
        for cutoff in (5, 10):
            if _as_bool(row[f"{prefix}_hit_at_{cutoff}"]) != _as_bool(
                ranking_row[f"{prefix}_hit_at_{cutoff}"]
            ):
                errors.append(
                    f"Ranking case {prefix} Hit@{cutoff} for user {user_id} "
                    "is inconsistent"
                )
    component_sum = (
        float(row["pmf_global_mean_contribution"])
        + float(row["pmf_user_bias_contribution"])
        + float(row["pmf_item_bias_contribution"])
        + float(row["pmf_total_latent_dot_product"])
    )
    if not np.isclose(
        component_sum,
        float(row["pmf_raw_target_score"]),
        rtol=0.0,
        atol=1e-5,
    ):
        errors.append(
            f"Ranking case PMF decomposition for user {user_id} is broken"
        )
    if abs(float(row["pmf_reconstruction_error"])) > 1e-5:
        errors.append(
            f"Ranking case PMF reconstruction error for user {user_id} is too high"
        )
    nearest_similarity = float(row["nearest_known_similarity"])
    if not np.isfinite(nearest_similarity) or not -1.000001 <= nearest_similarity <= 1.000001:
        errors.append(
            f"Ranking case nearest similarity for user {user_id} is invalid"
        )
    return errors


def _validate_stale_source_references(root: Path) -> list[str]:
    errors: list[str] = []
    stale_paths = [
        root / "models" / "baseline_cf.py",
        root / "scripts" / "build_audit_notebook.py",
    ]
    for path in stale_paths:
        if path.exists():
            errors.append(f"Stale source file remains: {path.relative_to(root)}")
    stale_tokens = [
        "models.baseline_cf",
        "BaselineCFModel",
        "BaselineCFConfig",
        "reports/baseline_tuning.json",
        "scripts/build_audit_notebook.py",
        "AUDIT_USER_ROLES",
        "select_audit_users",
    ]
    source_paths = [
        root / "models",
        root / "utils",
        root / "app.py",
        root / "README.md",
        root / "scripts" / "run_pipeline.py",
        root / "scripts" / "build_analysis_notebook.py",
    ]
    files: list[Path] = []
    for path in source_paths:
        if path.is_dir():
            files.extend(path.glob("*.py"))
        elif path.exists():
            files.append(path)
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in stale_tokens:
            if token in text:
                errors.append(
                    f"Stale reference {token!r} remains in {path.relative_to(root)}"
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
    errors.extend(_validate_stale_source_references(root))
    for relative in REQUIRED_PATHS:
        path = root / relative
        if path.suffix == ".png" and path.stat().st_size == 0:
            errors.append(f"Generated plot is empty: {relative}")

    try:
        with (root / "reports" / "model_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (root / "reports" / "bias_baseline_tuning.json").open(
            encoding="utf-8"
        ) as handle:
            bias_tuning = json.load(handle)
        with (root / "reports" / "item_knn_tuning.json").open(
            encoding="utf-8"
        ) as handle:
            item_knn_tuning = json.load(handle)
        with (root / "reports" / "ranking_protocol.json").open(
            encoding="utf-8"
        ) as handle:
            ranking_protocol = json.load(handle)
        with (root / "reports" / "ranking_metrics.json").open(
            encoding="utf-8"
        ) as handle:
            ranking_metrics = json.load(handle)
        with (root / "reports" / "evaluated_users.json").open(encoding="utf-8") as handle:
            evaluated_users = json.load(handle)
        with (root / "reports" / "svd_metadata.json").open(encoding="utf-8") as handle:
            svd_metadata = json.load(handle)
        with (root / "reports" / "svd_tuning.json").open(encoding="utf-8") as handle:
            svd_tuning = json.load(handle)
        with (root / "reports" / "pmf_tuning.json").open(encoding="utf-8") as handle:
            pmf_tuning = json.load(handle)
        with (root / "reports" / "pmf_convergence.json").open(
            encoding="utf-8"
        ) as handle:
            pmf_convergence = json.load(handle)
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
    errors.extend(_validate_pmf_convergence_payload(pmf_convergence))
    errors.extend(_validate_bias_baseline_tuning_artifact(bias_tuning))
    errors.extend(_validate_item_knn_tuning_artifact(item_knn_tuning))
    svd_best = metrics.get("svd_best_params", {})
    if (
        svd_best.get("n_factors") != 20
        or float(svd_best.get("item_bias_regularization", np.nan)) != 5.0
    ):
        errors.append("SVD selected parameters changed from rank 20 / bias reg 5.0")
    else:
        errors.extend(
            _validate_svd_tuning_payload(
                svd_tuning,
                selected_rank=int(svd_best["n_factors"]),
                selected_item_bias_regularization=float(
                    svd_best["item_bias_regularization"]
                ),
            )
        )
    pmf_best = metrics.get("pmf_best_params", {})
    expected_pmf_best = {
        "n_factors": 128,
        "learning_rate": 0.006,
        "factor_regularization": 0.06,
        "bias_regularization": 0.02,
        "selected_epoch": 53,
    }
    if any(pmf_best.get(key) != value for key, value in expected_pmf_best.items()):
        errors.append("PMF selected parameters changed from the frozen configuration")
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
        "hit_epoch_cap",
        "hit_factor_boundary",
    }
    forbidden_timing_fields = {
        "sec" + "onds",
        "elapsed_time",
        "elapsed_seconds",
        "duration",
        "duration_seconds",
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
            timing_fields = forbidden_timing_fields.intersection(row)
            if timing_fields:
                errors.append(
                    "PMF tuning row contains runtime timing fields: "
                    f"{sorted(timing_fields)}"
                )
                break
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
    errors.extend(
        _validate_user_item_matrix(
            root / "processed" / "user_item_matrix.csv",
            index_to_user,
            index_to_movie,
        )
    )
    predictions = np.load(root / "reports" / "svd_predictions.npy", mmap_mode="r")
    expected_shape = (len(user_to_index), len(movie_to_index))
    errors.extend(_validate_raw_svd_predictions(predictions, expected_shape))
    if svd_metadata.get("prediction_scale") != "raw_unclipped":
        errors.append("SVD metadata does not declare raw_unclipped prediction scale")

    train = pd.read_csv(root / "processed" / "train_ratings.csv")
    validation = pd.read_csv(root / "processed" / "validation_ratings.csv")
    test = pd.read_csv(root / "processed" / "test_ratings.csv")
    ranking_train = pd.read_csv(
        root / "processed" / "ranking_train_ratings.csv"
    )
    ranking_targets = pd.read_csv(root / "processed" / "ranking_targets.csv")
    ranking_results = pd.read_csv(root / "reports" / "ranking_results.csv")
    errors.extend(
        _validate_ranking_artifacts(
            ranking_train,
            ranking_targets,
            ranking_results,
            ranking_metrics,
            ranking_protocol,
        )
    )
    for name, payload in (
        ("bias baseline", bias_tuning),
        ("item-kNN", item_knn_tuning),
    ):
        test_evaluation = payload.get("test_evaluation", {})
        if int(test_evaluation.get("test_rows", -1)) != len(test):
            errors.append(f"{name} test row count does not match the saved split")
    errors.extend(
        _validate_evaluation_users_payload(
            evaluated_users,
            train,
            validation,
            test,
            ranking_results,
        )
    )
    errors.extend(
        _validate_user_artifact_manifest(
            root / "reports",
            evaluated_users,
        )
    )

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
    expected_split = deterministic_user_split(data.ratings, random_state=42)
    for name, actual, expected in (
        ("train", train, expected_split.train),
        ("validation", validation, expected_split.validation),
        ("test", test, expected_split.test),
    ):
        actual_sorted = actual.sort_values(
            ["user_id", "movie_id"], kind="mergesort"
        ).reset_index(drop=True)
        expected_sorted = expected.sort_values(
            ["user_id", "movie_id"], kind="mergesort"
        ).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                actual_sorted,
                expected_sorted,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError:
            errors.append(f"Saved {name} rows do not match the deterministic split")

    ranking_by_user = ranking_results.set_index("user_id")
    for selection in evaluated_users:
        user_id = int(selection["user_id"])
        role = str(selection.get("role"))
        recommendation_path = root / "reports" / f"user_{user_id}_recommendations.csv"
        explanation_path = root / "reports" / f"user_{user_id}_explanations.csv"
        explanation_plot = root / "reports" / f"user_{user_id}_explanation.png"
        ranking_case_path = root / "reports" / f"user_{user_id}_ranking_case.csv"
        ranking_case_plot = root / "reports" / f"user_{user_id}_ranking_case.png"
        if not ranking_case_path.exists():
            errors.append(f"Missing ranking case CSV for user {user_id}")
        if not ranking_case_plot.exists():
            errors.append(f"Missing ranking case PNG for user {user_id}")
        elif ranking_case_plot.stat().st_size == 0:
            errors.append(f"Ranking case PNG for user {user_id} is empty")
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
        if ranking_case_path.exists() and user_id in ranking_by_user.index:
            ranking_case = pd.read_csv(ranking_case_path)
            errors.extend(
                _validate_ranking_case_artifact(
                    ranking_case,
                    ranking_by_user.loc[user_id],
                    user_id,
                    role,
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
        unexecuted = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and cell.get("execution_count") is None
        ]
        if unexecuted:
            errors.append(
                "Notebook contains unexecuted code cells: "
                f"{unexecuted[:10]}"
            )
        oversized = [
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and len(cell.source.splitlines()) > 150
        ]
        if oversized:
            errors.append(
                "Notebook contains oversized implementation cells: "
                f"{oversized}"
            )
        markdown_text = "\n".join(
            cell.source
            for cell in notebook.cells
            if cell.cell_type == "markdown"
        )
        normalized_markdown = " ".join(markdown_text.split())
        required_sections = [
            "## 1. Project goal",
            "## 2. MovieLens EDA and Insights",
            "### 2.1 Temporal EDA",
            "### 2.2 Genre EDA",
            "### 2.3 Demographic EDA",
            "## 3. Rating-prediction split",
            "## 4. Bias baseline",
            "## 5. Item-kNN neighborhood collaborative filtering",
            "## 6. SVD methodology and tuning",
            "## 7. PMF methodology and tuning",
            "## 8. Convergence, regularization and stopping",
            "## 9. Rating-prediction results",
            "## 10. Temporal leave-one-positive-out protocol",
            "## 11. Top-K ranking results",
            "## 12. Global latent-factor interpretation",
            "## 13. Movie similarity analysis",
            "## 14. User Case Studies",
            "## 15. Recommendation Hit vs Miss Analysis",
            "## 16. Local Recommendation Explanations",
            "## 17. Streamlit and artifact overview",
            "## 18. Limitations",
        ]
        for section in required_sections:
            if section not in markdown_text:
                errors.append(f"Notebook is missing required section: {section}")
        required_statements = [
            "RMSE is pointwise rating-prediction accuracy",
            "different tasks",
            "Unknown catalog items are not observed negatives",
            "one known future positive",
            "No sampled negatives are used",
            "defined by temporal ranking outcome, not per-user RMSE",
            "The full raw dataset is used only for descriptive EDA",
            "SVD is fitted through a direct truncated decomposition",
            "cold-start user",
        ]
        for statement in required_statements:
            if statement not in normalized_markdown:
                errors.append(
                    f"Notebook is missing required methodology statement: {statement}"
                )
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
