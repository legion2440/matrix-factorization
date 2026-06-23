from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.recommendation import (
    PMFRecommendationModel,
    SVDRecommendationModel,
    compare_recommendations,
    generate_recommendations,
)


def _svd_model(
    movie_ids: list[int],
    scores: list[float],
    seen_movie_ids: list[int] | None = None,
):
    seen_movie_ids = seen_movie_ids or []
    movies = pd.DataFrame(
        {
            "movie_id": movie_ids,
            "title": [f"Movie {movie_id}" for movie_id in movie_ids],
            "genres": ["Drama"] * len(movie_ids),
        }
    )
    ratings = pd.DataFrame(
        {
            "user_id": [1] * len(seen_movie_ids),
            "movie_id": seen_movie_ids,
            "rating": [5.0] * len(seen_movie_ids),
        }
    )
    return SVDRecommendationModel(
        predictions=np.array([scores], dtype=np.float32),
        user_to_index={1: 0},
        movie_to_index={movie_id: index for index, movie_id in enumerate(movie_ids)},
        index_to_movie=np.array(movie_ids, dtype=np.int32),
        movies=movies,
        known_ratings=ratings,
    )


def test_raw_scores_rank_saturated_displayed_ratings_correctly():
    model = _svd_model([100, 200, 300], [5.1, 5.9, 5.6])
    recommendations = generate_recommendations(1, model, top_n=3)
    assert recommendations["movie_id"].tolist() == [200, 300, 100]
    assert recommendations["ranking_score"].tolist() == pytest.approx([5.9, 5.6, 5.1])
    assert recommendations["predicted_rating"].tolist() == [5.0, 5.0, 5.0]


def test_seen_movies_are_excluded_and_equal_raw_scores_tie_break_by_movie_id():
    model = _svd_model(
        [10, 20, 30, 40],
        [4.9, 4.8, 5.0, 4.8],
        seen_movie_ids=[10, 30],
    )
    recommendations = generate_recommendations(1, model, top_n=2)
    assert recommendations["movie_id"].tolist() == [20, 40]
    assert not {10, 30} & set(recommendations["movie_id"])
    assert recommendations["predicted_rating"].between(1.0, 5.0).all()


class _RecordingPMF:
    def __init__(self) -> None:
        self.clip_arguments: list[bool] = []

    def predict_user(self, user_index: int, clip: bool = True) -> np.ndarray:
        self.clip_arguments.append(clip)
        scores = np.array([5.1, 5.9, 5.6], dtype=np.float32)
        return np.clip(scores, 1.0, 5.0) if clip else scores


def test_pmf_recommendation_path_requests_unclipped_scores():
    pmf = _RecordingPMF()
    movie_ids = [100, 200, 300]
    model = PMFRecommendationModel(
        pmf=pmf,
        user_to_index={1: 0},
        movie_to_index={movie_id: index for index, movie_id in enumerate(movie_ids)},
        index_to_movie=np.array(movie_ids, dtype=np.int32),
        movies=pd.DataFrame(
            {
                "movie_id": movie_ids,
                "title": ["A", "B", "C"],
                "genres": ["Drama"] * 3,
            }
        ),
        known_ratings=pd.DataFrame(columns=["user_id", "movie_id", "rating"]),
    )
    recommendations = generate_recommendations(1, model, top_n=3)
    assert pmf.clip_arguments == [False]
    assert recommendations["movie_id"].tolist() == [200, 300, 100]


def test_comparison_preserves_raw_and_displayed_scores_for_both_models():
    svd = _svd_model([100, 200, 300], [5.1, 5.9, 5.6])
    pmf = _svd_model([100, 200, 300], [5.8, 5.2, 5.7])
    comparison = compare_recommendations(1, svd, pmf, top_n=3)
    expected_columns = {
        "svd_ranking_score",
        "svd_predicted_rating",
        "svd_rank",
        "pmf_ranking_score",
        "pmf_predicted_rating",
        "pmf_rank",
    }
    assert expected_columns.issubset(comparison.columns)
    assert np.allclose(
        comparison["svd_predicted_rating"],
        np.clip(comparison["svd_ranking_score"], 1.0, 5.0),
    )
    assert np.allclose(
        comparison["pmf_predicted_rating"],
        np.clip(comparison["pmf_ranking_score"], 1.0, 5.0),
    )


def test_recommendations_reject_unknown_user_and_invalid_top_n():
    with pytest.raises(ValueError, match="Unknown user_id"):
        generate_recommendations(
            99, _svd_model([10, 20], [4.0, 3.0]), top_n=2
        )
    with pytest.raises(ValueError, match="positive integer"):
        generate_recommendations(
            1, _svd_model([10, 20], [4.0, 3.0]), top_n=0
        )
