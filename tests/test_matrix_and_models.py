from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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
    pair_predictions = model.predict_pairs(
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
    )
    assert predictions.shape == (len(reverse_users), len(reverse_movies))
    assert np.all((predictions >= 1.0) & (predictions <= 5.0))
    assert np.all((pair_predictions >= 1.0) & (pair_predictions <= 5.0))
    assert all(reverse_users[index] == raw for raw, index in user_map.items())
    assert all(reverse_movies[index] == raw for raw, index in movie_map.items())


def test_svd_clip_false_returns_raw_scores_outside_rating_range():
    model = SVDModel(n_factors=1, random_state=42)
    model.user_factors = np.array([[1.0]], dtype=np.float32)
    model.singular_values = np.array([1.0], dtype=np.float32)
    model.item_factors = np.array([[2.0, 1.0, -6.0]], dtype=np.float32)
    model.user_means = np.array([4.5], dtype=np.float32)
    model.item_bias = np.zeros(3, dtype=np.float32)
    model.shape = (1, 3)

    raw_all = model.predict_all(clip=False)
    clipped_all = model.predict_all()
    raw_pairs = model.predict_pairs(
        np.array([0, 0, 0]),
        np.array([0, 1, 2]),
        clip=False,
    )
    clipped_pairs = model.predict_pairs(
        np.array([0, 0, 0]),
        np.array([0, 1, 2]),
    )

    assert raw_all[0].tolist() == pytest.approx([6.5, 5.5, -1.5])
    assert raw_pairs.tolist() == pytest.approx([6.5, 5.5, -1.5])
    assert clipped_all[0].tolist() == [5.0, 5.0, 1.0]
    assert clipped_pairs.tolist() == [5.0, 5.0, 1.0]


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
