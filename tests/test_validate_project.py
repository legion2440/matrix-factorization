from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pmf_model import PMFModel
from scripts.validate_project import (
    _validate_benchmark_metrics,
    _validate_evaluation_users_payload,
    _validate_explanation_artifact,
    _validate_factor_interpretation,
    _validate_item_knn_tuning_artifact,
    _validate_ranking_artifacts,
    _validate_raw_svd_predictions,
    _validate_similarity_artifact,
)
from utils.ranking_evaluation import aggregate_ranking_metrics


@pytest.mark.parametrize("raw_value", [0.9, 5.1])
def test_raw_svd_prediction_matrix_accepts_values_outside_display_range(raw_value):
    predictions = np.array([[1.0, 3.0], [4.0, raw_value]], dtype=np.float32)

    assert _validate_raw_svd_predictions(predictions, predictions.shape) == []


def test_raw_svd_prediction_matrix_rejects_clipped_values():
    predictions = np.array([[1.0, 3.0], [4.0, 5.0]], dtype=np.float32)

    errors = _validate_raw_svd_predictions(predictions, predictions.shape)

    assert errors == [
        "SVD prediction artifact appears clipped; "
        "expected raw values outside [1, 5]"
    ]


def test_raw_svd_prediction_matrix_preserves_shape_and_finite_checks():
    predictions = np.array([[0.9, np.nan]], dtype=np.float32)

    errors = _validate_raw_svd_predictions(predictions, (2, 2))

    assert "SVD prediction shape (1, 2) != expected (2, 2)" in errors
    assert "Raw SVD predictions contain non-finite values" in errors


def test_benchmark_metric_validation_checks_honest_pairwise_flags():
    missing_errors = _validate_benchmark_metrics({"SVD_RMSE": 0.9, "PMF_RMSE": 0.8})
    assert any("missing benchmark fields" in error for error in missing_errors)

    metrics = {
        "BiasBaseline_MSE": 1.0,
        "BiasBaseline_RMSE": 1.0,
        "ItemKNN_MSE": 0.81,
        "ItemKNN_RMSE": 0.9,
        "SVD_MSE": 0.9025,
        "SVD_RMSE": 0.95,
        "PMF_MSE": 0.7225,
        "PMF_RMSE": 0.85,
        "ItemKNN_vs_BiasBaseline_improvement_%": 10.0,
        "SVD_vs_BiasBaseline_improvement_%": 5.0,
        "PMF_vs_BiasBaseline_improvement_%": 15.0,
        "SVD_vs_ItemKNN_improvement_%": (0.9 - 0.95) / 0.9 * 100.0,
        "PMF_vs_ItemKNN_improvement_%": (0.9 - 0.85) / 0.9 * 100.0,
        "PMF_vs_SVD_improvement_%": (0.95 - 0.85) / 0.95 * 100.0,
        "ItemKNN_beats_BiasBaseline": True,
        "SVD_beats_BiasBaseline": True,
        "PMF_beats_BiasBaseline": True,
        "SVD_beats_ItemKNN": False,
        "PMF_beats_ItemKNN": True,
        "bias_baseline_best_params": {},
        "item_knn_best_params": {},
    }
    assert _validate_benchmark_metrics(metrics) == []

    metrics["SVD_beats_ItemKNN"] = True
    errors = _validate_benchmark_metrics(metrics)
    assert "SVD_beats_ItemKNN does not match the stored RMSE values" in errors


def test_item_knn_tuning_validator_checks_grid_selection_and_neighbors():
    results = [
        {
            "k": k,
            "shrinkage": shrinkage,
            "min_common": 3,
            "validation_mse": 1.0,
            "validation_rmse": 1.0 + k / 10000 - shrinkage / 100000,
        }
        for k in (20, 40, 80)
        for shrinkage in (10.0, 50.0, 100.0)
    ]
    selected = min(
        results,
        key=lambda row: (
            row["validation_rmse"],
            row["k"],
            -row["shrinkage"],
        ),
    )
    payload = {
        "model": "ItemKNN",
        "prediction_formula": "x",
        "similarity_definition": "x",
        "neighborhood_definition": "x",
        "neighborhood_ordering": [
            "absolute shrunk similarity descending",
            "signed shrunk similarity descending",
            "movie ID ascending",
        ],
        "parameter_grid": {
            "k": [20, 40, 80],
            "shrinkage": [10.0, 50.0, 100.0],
            "min_common": 3,
        },
        "selection_metric": "validation_rmse",
        "selection_tie_break": [],
        "uses_test_for_tuning": False,
        "results": results,
        "selected": selected,
        "final_refit": {
            "uses_train_plus_validation": True,
            "diagnostics": {
                "similarities_finite": True,
                "self_neighbor_count": 0,
                "deterministic_ordering_verified": True,
                "minimum_common_users": 3,
            },
        },
        "test_evaluation": {"mse": 1.0, "rmse": 1.0},
    }
    assert _validate_item_knn_tuning_artifact(payload) == []
    payload["final_refit"]["diagnostics"]["self_neighbor_count"] = 1
    assert "item-kNN self-neighbors must be absent" in _validate_item_knn_tuning_artifact(
        payload
    )


def test_factor_interpretation_validation_rejects_missing_polarity_and_bad_values():
    frame = pd.DataFrame(
        {
            "factor_index": [0],
            "factor_variance": [0.1],
            "polarity": ["positive"],
            "polarity_rank": [1],
            "movie_id": [10],
            "title": ["A"],
            "genres": ["Drama"],
            "factor_loading": [np.nan],
        }
    )

    errors = _validate_factor_interpretation(frame)

    assert "Factor 0 must contain positive and negative polarities" in errors
    assert "Factor interpretation contains non-finite loadings" in errors


def test_similarity_validation_rejects_self_match_range_and_sorting():
    frame = pd.DataFrame(
        {
            "anchor_movie_id": [10, 10],
            "anchor_title": ["A", "A"],
            "anchor_genres": ["Drama", "Drama"],
            "similar_movie_id": [10, 20],
            "similar_title": ["A", "B"],
            "similar_genres": ["Drama", "Comedy"],
            "cosine_similarity": [1.0, 1.5],
            "rank": [1, 2],
        }
    )

    errors = _validate_similarity_artifact(frame)

    assert "Similarity CSV contains self matches" in errors
    assert "Similarity CSV contains values outside [-1, 1]" in errors


def test_evaluation_user_validation_rejects_duplicate_roles_users_and_bad_ordering():
    train = pd.DataFrame({"user_id": [1, 2, 3], "movie_id": [1, 1, 1]})
    validation = train.copy()
    test = train.copy()
    payload = [
        {
            "user_id": 1,
            "role": "train_profile_accurate",
            "selection_reason": "x",
            "train_ratings": 1,
            "validation_ratings": 1,
            "test_ratings": 1,
            "svd_test_rmse": 1.0,
            "pmf_test_rmse": 2.0,
        },
        {
            "user_id": 1,
            "role": "train_profile_accurate",
            "selection_reason": "x",
            "train_ratings": 1,
            "validation_ratings": 1,
            "test_ratings": 1,
            "svd_test_rmse": 1.0,
            "pmf_test_rmse": 1.0,
        },
        {
            "user_id": 3,
            "role": "test_case",
            "selection_reason": "x",
            "train_ratings": 1,
            "validation_ratings": 1,
            "test_ratings": 1,
            "svd_test_rmse": 1.0,
            "pmf_test_rmse": 1.0,
        },
    ]

    ranking = pd.DataFrame({"user_id": pd.Series(dtype=int)})
    errors = _validate_evaluation_users_payload(
        payload, train, validation, test, ranking
    )

    assert "Evaluation profile user IDs must be unique" in errors
    assert any("roles must be exactly" in error for error in errors)


def test_ranking_validator_reconstructs_full_catalog_metrics_and_strict_prefix():
    history_rows = [
        (1, 100 + index, 3.0, index + 1) for index in range(20)
    ]
    support_rows = [(user_id, 50, 4.0, 1) for user_id in range(2, 12)]
    ranking_train = pd.DataFrame(
        history_rows + support_rows,
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    ranking_targets = pd.DataFrame(
        {
            "user_id": [1],
            "movie_id": [50],
            "rating": [5.0],
            "timestamp": [30],
            "prior_history_count": [20],
            "target_item_support": [10],
        }
    )
    result = {
        "user_id": 1,
        "target_movie_id": 50,
        "target_title": "Target",
        "target_genres": "Drama",
        "target_rating": 5.0,
        "target_timestamp": 30,
        "prior_history_count": 20,
        "candidate_count": 1,
    }
    for prefix in ("bias", "item_knn", "svd", "pmf"):
        result[f"{prefix}_target_rank"] = 1
        result[f"{prefix}_raw_target_score"] = 4.0
        for cutoff in (5, 10):
            result[f"{prefix}_hit_at_{cutoff}"] = True
            result[f"{prefix}_ndcg_at_{cutoff}"] = 1.0
            result[f"{prefix}_mrr_at_{cutoff}"] = 1.0
    ranking_results = pd.DataFrame([result])
    ranking_metrics = aggregate_ranking_metrics(ranking_results)
    ranking_protocol = {
        "protocol": "next-positive recovery under temporal leave-one-positive-out",
        "full_catalog_candidates": True,
        "sampled_negatives": False,
        "history_rule": "timestamp < target_timestamp",
        "min_prior_interactions": 20,
        "min_target_item_support": 10,
        "frozen_model_parameters": {
            "SVD": {
                "n_factors": 20,
                "item_bias_regularization": 5.0,
                "random_state": 42,
            },
            "PMF": {
                "n_factors": 128,
                "learning_rate": 0.006,
                "factor_regularization": 0.06,
                "bias_regularization": 0.02,
                "epochs": 53,
                "random_state": 42,
                "uses_ranking_targets_for_tuning": False,
            },
        },
    }

    assert (
        _validate_ranking_artifacts(
            ranking_train,
            ranking_targets,
            ranking_results,
            ranking_metrics,
            ranking_protocol,
        )
        == []
    )

    ranking_train.loc[len(ranking_train)] = [1, 999, 2.0, 30]
    errors = _validate_ranking_artifacts(
        ranking_train,
        ranking_targets,
        ranking_results,
        ranking_metrics,
        ranking_protocol,
    )
    assert any("same-timestamp or later" in error for error in errors)


def _fake_pmf_for_validator() -> PMFModel:
    model = PMFModel(n_users=1, n_items=2, n_factors=2)
    model.global_mean = 3.0
    model.user_bias = np.array([0.5], dtype=np.float32)
    model.item_bias = np.array([0.25, 0.0], dtype=np.float32)
    model.user_factors = np.array([[1.0, -2.0]], dtype=np.float32)
    model.item_factors = np.array([[0.5, -0.25], [1.0, 1.0]], dtype=np.float32)
    return model


def test_explanation_validation_rejects_broken_decomposition_and_missing_columns():
    pmf = _fake_pmf_for_validator()
    recommendations = pd.DataFrame(
        {
            "movie_id": [10],
            "pmf_rank": [1],
        }
    )
    incomplete = pd.DataFrame({"movie_id": [10]})
    missing_errors = _validate_explanation_artifact(
        incomplete,
        recommendations,
        1,
        "test_case",
        pmf,
        {1: 0},
        {10: 0, 20: 1},
        set(),
        1,
    )
    assert any("missing columns" in error for error in missing_errors)

    broken = pd.DataFrame(
        {
            "user_id": [1],
            "role": ["test_case"],
            "recommendation_rank": [1],
            "movie_id": [10],
            "title": ["A"],
            "genres": ["Drama"],
            "raw_pmf_ranking_score": [9.0],
            "clipped_displayed_rating": [5.0],
            "global_mean_contribution": [3.0],
            "user_bias_contribution": [0.5],
            "item_bias_contribution": [0.25],
            "total_latent_dot_product": [1.0],
            "top_factor_1_index": [0],
            "top_factor_1_contribution": [0.5],
            "top_factor_2_index": [1],
            "top_factor_2_contribution": [0.5],
            "top_factor_3_index": [-1],
            "top_factor_3_contribution": [0.0],
            "top_factor_contributions": ["[]"],
            "component_sum": [4.75],
            "reconstruction_error": [4.25],
            "nearest_known_movie_id": [20],
            "nearest_known_title": ["B"],
            "nearest_known_genres": ["Comedy"],
            "nearest_known_rating": [5.0],
            "nearest_known_similarity": [0.5],
            "common_genres": [""],
        }
    )

    errors = _validate_explanation_artifact(
        broken,
        recommendations,
        1,
        "test_case",
        pmf,
        {1: 0},
        {10: 0, 20: 1},
        set(),
        1,
    )

    assert "Explanation decomposition for user 1 is broken" in errors
    assert "Explanation component_sum for user 1 is invalid" in errors
    assert "Explanation reconstruction error for user 1 is too high" in errors
