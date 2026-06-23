from __future__ import annotations

import numpy as np
import pandas as pd

from models.pmf_model import PMFModel
from models.svd_model import SVDModel
from utils.matrix_creation import build_normalized_matrix, create_mappings


def test_mapping_consistency_and_svd_range(synthetic_ratings):
    users = pd.DataFrame({"user_id": sorted(synthetic_ratings["user_id"].unique())})
    movies = pd.DataFrame({"movie_id": sorted(synthetic_ratings["movie_id"].unique())})
    user_map, movie_map, reverse_users, reverse_movies = create_mappings(users, movies)
    matrix, means = build_normalized_matrix(synthetic_ratings, user_map, movie_map)
    model = SVDModel(n_factors=2, random_state=42).fit(matrix, means)
    predictions = model.predict_all()
    assert predictions.shape == (len(reverse_users), len(reverse_movies))
    assert np.all((predictions >= 1.0) & (predictions <= 5.0))
    assert all(reverse_users[index] == raw for raw, index in user_map.items())
    assert all(reverse_movies[index] == raw for raw, index in movie_map.items())


def test_pmf_prediction_shape_and_range(synthetic_ratings):
    users = synthetic_ratings["user_id"].to_numpy(np.int32) - 1
    items = synthetic_ratings["movie_id"].to_numpy(np.int32) - 1
    ratings = synthetic_ratings["rating"].to_numpy(np.float32)
    model = PMFModel(
        n_users=4,
        n_items=10,
        n_factors=4,
        learning_rate=0.01,
        epochs=2,
        patience=2,
        random_state=42,
    ).fit(users, items, ratings)
    predictions = model.predict_pairs(users, items)
    assert predictions.shape == ratings.shape
    assert np.all((predictions >= 1.0) & (predictions <= 5.0))

