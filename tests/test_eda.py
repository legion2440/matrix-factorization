from __future__ import annotations

import pandas as pd

from utils.eda import (
    aggregate_genre_statistics,
    aggregate_temporal_ratings,
    explode_movie_genres,
)


def test_temporal_aggregation_groups_months_deterministically():
    ratings = pd.DataFrame(
        {
            "timestamp": [946684800, 946771200, 949363200],
            "rating": [4.0, 2.0, 5.0],
        }
    )

    result = aggregate_temporal_ratings(ratings)

    assert result["month"].dt.strftime("%Y-%m").tolist() == [
        "2000-01",
        "2000-02",
    ]
    assert result["rating_count"].tolist() == [2, 1]
    assert result["mean_rating"].tolist() == [3.0, 5.0]


def test_genre_explode_counts_multilabel_movies_and_ratings():
    movies = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "genres": ["Action|Comedy", "Comedy"],
        }
    )
    ratings = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "movie_id": [10, 10, 20],
            "rating": [4.0, 2.0, 5.0],
        }
    )

    exploded = explode_movie_genres(movies)
    summary = aggregate_genre_statistics(ratings, movies).set_index("genre")

    assert len(exploded) == 3
    assert summary.loc["Action", "movie_count"] == 1
    assert summary.loc["Action", "rating_count"] == 2
    assert summary.loc["Comedy", "movie_count"] == 2
    assert summary.loc["Comedy", "rating_count"] == 3
    assert summary.loc["Comedy", "mean_rating"] == 11 / 3
