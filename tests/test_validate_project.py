from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pmf_model import PMFModel
from scripts.validate_project import (
    _validate_audit_users_payload,
    _validate_benchmark_metrics,
    _validate_explanation_artifact,
    _validate_factor_interpretation,
    _validate_raw_svd_predictions,
    _validate_similarity_artifact,
)


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


def test_benchmark_metric_validation_rejects_missing_and_too_strong_baseline():
    missing_errors = _validate_benchmark_metrics({"SVD_RMSE": 0.9, "PMF_RMSE": 0.8})
    assert any("missing benchmark fields" in error for error in missing_errors)

    errors = _validate_benchmark_metrics(
        {
            "Baseline_CF_MSE": 0.64,
            "Baseline_CF_RMSE": 0.8,
            "SVD_RMSE": 0.9,
            "PMF_RMSE": 0.85,
            "SVD_vs_Baseline_improvement_%": -12.5,
            "PMF_vs_Baseline_improvement_%": -6.25,
            "baseline_best_params": {},
        }
    )

    assert "Baseline_CF_RMSE must be greater than SVD_RMSE" in errors
    assert "Baseline_CF_RMSE must be greater than PMF_RMSE" in errors


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


def test_audit_user_validation_rejects_duplicate_roles_users_and_bad_ordering():
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

    errors = _validate_audit_users_payload(payload, train, validation, test)

    assert "Audit user IDs must be unique" in errors
    assert any("roles must be exactly" in error for error in errors)


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
