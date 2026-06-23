from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.recommendation import SVDRecommendationModel, generate_recommendations


def _model():
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Drama", "Comedy", "Action", "Drama"],
        }
    )
    ratings = pd.DataFrame(
        {"user_id": [1, 1], "movie_id": [10, 30], "rating": [5, 4]}
    )
    return SVDRecommendationModel(
        predictions=np.array([[4.9, 4.8, 4.7, 4.8]], dtype=np.float32),
        user_to_index={1: 0},
        movie_to_index={10: 0, 20: 1, 30: 2, 40: 3},
        index_to_movie=np.array([10, 20, 30, 40], dtype=np.int32),
        movies=movies,
        known_ratings=ratings,
    )


def test_recommendations_exclude_seen_and_break_ties_by_movie_id():
    recommendations = generate_recommendations(1, _model(), top_n=2)
    assert recommendations["movie_id"].tolist() == [20, 40]
    assert not {10, 30} & set(recommendations["movie_id"])


def test_recommendations_reject_unknown_user_and_invalid_top_n():
    with pytest.raises(ValueError, match="Unknown user_id"):
        generate_recommendations(99, _model(), top_n=2)
    with pytest.raises(ValueError, match="positive integer"):
        generate_recommendations(1, _model(), top_n=0)

