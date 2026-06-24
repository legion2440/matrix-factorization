from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.pmf_model import PMFModel
from utils.interpretability import (
    build_local_pmf_explanations,
    build_pmf_movie_similarities,
    build_ranking_case_explanation,
    cosine_similarity,
    decompose_pmf_score,
    nearest_known_liked_movie,
    select_evaluation_users,
)


def _fake_pmf() -> PMFModel:
    model = PMFModel(n_users=2, n_items=4, n_factors=3)
    model.global_mean = 3.0
    model.user_bias = np.array([0.2, -0.1], dtype=np.float32)
    model.item_bias = np.array([0.1, -0.2, 0.0, 0.3], dtype=np.float32)
    model.user_factors = np.array(
        [[1.0, -2.0, 0.5], [0.2, 0.1, -0.3]], dtype=np.float32
    )
    model.item_factors = np.array(
        [
            [0.5, -0.25, 2.0],
            [1.0, 1.0, 0.0],
            [0.0, -1.0, 0.5],
            [0.5, -0.25, 2.0],
        ],
        dtype=np.float32,
    )
    return model


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Action|Drama", "Action", "Comedy", "Action|Drama"],
        }
    )


def test_pmf_score_decomposition_and_top_factor_contributions():
    model = _fake_pmf()
    parts = decompose_pmf_score(model, user_index=0, item_index=0)

    expected_contributions = np.array([0.5, 0.5, 1.0])
    assert parts["factor_contributions"].tolist() == pytest.approx(
        expected_contributions.tolist()
    )
    assert parts["latent_dot"] == pytest.approx(expected_contributions.sum())
    assert parts["raw_score"] == pytest.approx(3.0 + 0.2 + 0.1 + 2.0)
    assert [row["factor_index"] for row in parts["top_factors"]] == [2, 0, 1]


def test_cosine_similarity_handles_zero_vectors_and_range():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0
    assert cosine_similarity(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_similarity_excludes_self_and_tie_breaks_by_movie_id():
    item_factors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.9, 0.1]],
        dtype=np.float32,
    )
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5, 6],
            "movie_id": [10, 10, 10, 20, 30, 40],
            "rating": [5.0] * 6,
        }
    )
    similarities = build_pmf_movie_similarities(
        item_factors,
        np.array([10, 20, 30, 40], dtype=np.int32),
        _movies(),
        ratings,
        {10: 0, 20: 1, 30: 2, 40: 3},
        n_anchors=1,
        top_n=3,
    )

    assert (similarities["anchor_movie_id"] != similarities["similar_movie_id"]).all()
    assert similarities["similar_movie_id"].tolist()[:2] == [20, 40]
    assert similarities["rank"].tolist() == [1, 2, 3]


def test_nearest_known_liked_movie_prefers_high_ratings_then_similarity():
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "movie_id": [20, 30, 40],
            "rating": [5.0, 5.0, 3.0],
        }
    )
    item_factors = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float32,
    )

    nearest = nearest_known_liked_movie(
        1,
        10,
        ratings,
        _movies(),
        {10: 0, 20: 1, 30: 2, 40: 3},
        item_factors,
    )

    assert nearest["movie_id"] == 20
    assert nearest["rating"] == 5.0
    assert nearest["common_genres"] == "Action"


def test_local_explanations_reconstruct_raw_scores_and_nearest_liked_movie():
    pmf = _fake_pmf()
    recommendations = pd.DataFrame(
        {
            "movie_id": [10, 30],
            "title": ["A", "C"],
            "genres": ["Action|Drama", "Comedy"],
            "svd_ranking_score": [4.0, 3.0],
            "svd_predicted_rating": [4.0, 3.0],
            "svd_rank": [1, 2],
            "pmf_ranking_score": [5.3, 4.0],
            "pmf_predicted_rating": [5.0, 4.0],
            "pmf_rank": [1, 2],
        }
    )
    ratings = pd.DataFrame(
        {
            "user_id": [1, 1],
            "movie_id": [20, 40],
            "rating": [5.0, 4.0],
        }
    )

    explanations = build_local_pmf_explanations(
        1,
        "test_case",
        recommendations,
        pmf,
        {1: 0},
        {10: 0, 20: 1, 30: 2, 40: 3},
        ratings,
        _movies(),
    )

    assert len(explanations) == 2
    assert np.allclose(
        explanations["component_sum"],
        explanations["raw_pmf_ranking_score"],
        atol=1e-6,
    )
    assert np.all(np.abs(explanations["reconstruction_error"]) <= 1e-6)
    assert explanations.loc[0, "nearest_known_movie_id"] in {20, 40}


def test_ranking_case_reconstructs_ranking_pmf_target_score():
    pmf = _fake_pmf()
    raw_score = float(pmf.predict_pairs(np.array([0]), np.array([0]), clip=False)[0])
    ranking_row = pd.Series(
        {
            "user_id": 1,
            "target_movie_id": 10,
            "target_title": "A",
            "target_genres": "Action|Drama",
            "target_rating": 5.0,
            "target_timestamp": 100,
            "prior_history_count": 2,
            "candidate_count": 3,
            "bias_target_rank": 3,
            "item_knn_target_rank": 2,
            "svd_target_rank": 2,
            "pmf_target_rank": 1,
            "bias_raw_target_score": 3.0,
            "item_knn_raw_target_score": 3.5,
            "svd_raw_target_score": 4.0,
            "pmf_raw_target_score": raw_score,
            "bias_hit_at_5": True,
            "bias_hit_at_10": True,
            "item_knn_hit_at_5": True,
            "item_knn_hit_at_10": True,
            "svd_hit_at_5": True,
            "svd_hit_at_10": True,
            "pmf_hit_at_5": True,
            "pmf_hit_at_10": True,
        }
    )
    ranking_row = ranking_row.drop(labels=["user_id"])
    ranking_row.name = 1
    ranking_train = pd.DataFrame(
        {
            "user_id": [1, 1],
            "movie_id": [20, 40],
            "rating": [5.0, 4.0],
            "timestamp": [10, 20],
        }
    )
    selection = {
        "user_id": 1,
        "role": "train_profile_accurate",
        "ranking_case": "pmf_hit_at_10",
    }

    case = build_ranking_case_explanation(
        selection,
        ranking_row,
        ranking_train,
        pmf,
        {1: 0, 2: 1},
        {10: 0, 20: 1, 30: 2, 40: 3},
        _movies(),
    )

    assert len(case) == 1
    assert abs(float(case.loc[0, "pmf_reconstruction_error"])) <= 1e-6
    assert int(case.loc[0, "pmf_target_rank"]) == 1
    assert int(case.loc[0, "nearest_known_movie_id"]) in {20, 40}


def test_evaluation_user_selection_roles_support_and_determinism():
    train = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "movie_id": [10, 10, 10, 10, 10],
            "rating": [4.0] * 5,
        }
    )
    validation = train.copy()
    rows = []
    errors = {1: 0.1, 2: 0.2, 3: 0.3, 4: 1.0, 5: 1.2}
    for user_id, error in errors.items():
        rows.append(
            {
                "user_id": user_id,
                "movie_id": 20,
                "rating": 4.0,
                "svd_prediction": 4.0,
                "pmf_prediction": 4.0 + error,
            }
        )
    test = pd.DataFrame(rows)
    ranking = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "target_movie_id": [30, 31, 32, 33, 34],
            "target_title": ["A", "B", "C", "D", "E"],
            "target_rating": [5.0, 4.0, 5.0, 4.0, 5.0],
            "target_timestamp": [10, 20, 30, 40, 50],
            "prior_history_count": [20, 21, 22, 23, 24],
            "candidate_count": [100] * 5,
            "bias_target_rank": [5, 6, 7, 8, 9],
            "item_knn_target_rank": [4, 5, 6, 7, 8],
            "svd_target_rank": [3, 4, 5, 6, 7],
            "pmf_target_rank": [1, 5, 9, 20, 40],
            "bias_hit_at_10": [True] * 5,
            "item_knn_hit_at_10": [True] * 5,
            "svd_hit_at_10": [True] * 5,
            "pmf_hit_at_10": [True, True, True, False, False],
        }
    )

    first = select_evaluation_users(
        train,
        validation,
        test,
        ranking,
        min_train_ratings=1,
        min_test_ratings=1,
    )
    second = select_evaluation_users(
        train,
        validation,
        test,
        ranking,
        min_train_ratings=1,
        min_test_ratings=1,
    )

    assert first == second
    assert {row["role"] for row in first} == {
        "train_profile_accurate",
        "train_profile_less_accurate",
        "test_case",
    }
    assert len({row["user_id"] for row in first}) == 3
    by_role = {row["role"]: row for row in first}
    assert by_role["train_profile_accurate"]["pmf_hit_at_10"] is True
    assert by_role["train_profile_accurate"]["pmf_target_rank"] == 5
    assert by_role["train_profile_less_accurate"]["pmf_hit_at_10"] is False
    assert by_role["train_profile_less_accurate"]["pmf_target_rank"] == 20
    assert by_role["test_case"]["user_id"] == 3
