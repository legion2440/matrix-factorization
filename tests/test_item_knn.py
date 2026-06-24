from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import sparse

from models.item_knn import ItemKNNModel
from scripts import run_pipeline


def _fit_small_model(
    *,
    shrinkage: float = 2.0,
    min_common: int = 2,
    n_neighbors: int = 3,
) -> ItemKNNModel:
    users = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    items = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2], dtype=np.int32)
    ratings = np.array(
        [5.0, 4.0, 1.0, 4.0, 2.0, 5.0, 1.0, 2.0, 2.0, 4.0],
        dtype=np.float32,
    )
    return ItemKNNModel(
        n_users=4,
        n_items=3,
        n_neighbors=n_neighbors,
        shrinkage=shrinkage,
        min_common=min_common,
        baseline_user_regularization=1.0,
        baseline_item_regularization=1.0,
        baseline_iterations=10,
        item_ids=np.array([30, 10, 20], dtype=np.int32),
        similarity_chunk_size=2,
    ).fit(users, items, ratings)


def test_item_knn_builds_bias_residuals_and_shrunk_cosine_similarity():
    model = _fit_small_model()
    users = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    items = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 2], dtype=np.int32)
    ratings = np.array(
        [5.0, 4.0, 1.0, 4.0, 2.0, 5.0, 1.0, 2.0, 2.0, 4.0],
        dtype=np.float64,
    )
    expected_residuals = ratings - model.baseline.predict_pairs(
        users, items, clip=False
    )
    actual_residuals = np.asarray(model.residual_matrix[users, items]).ravel()
    assert actual_residuals == pytest.approx(expected_residuals, abs=1e-6)

    target = 0
    neighbor = 1
    position = np.flatnonzero(model.neighbor_indices[target] == neighbor)
    assert position.size == 1
    residual_dense = model.residual_matrix.toarray()
    observed_dense = model.observed_matrix.toarray()
    common = int(np.sum((observed_dense[:, target] > 0) & (observed_dense[:, neighbor] > 0)))
    cosine = float(
        residual_dense[:, target] @ residual_dense[:, neighbor]
        / (
            np.linalg.norm(residual_dense[:, target])
            * np.linalg.norm(residual_dense[:, neighbor])
        )
    )
    expected = cosine * common / (common + model.config.shrinkage)
    selected = int(position[0])
    assert int(model.neighbor_common_counts[target][selected]) == common
    assert model.neighbor_similarities[target][selected] == pytest.approx(expected)


def test_item_knn_enforces_min_common_self_exclusion_and_neighbor_ordering():
    model = _fit_small_model(min_common=3)
    repeated = _fit_small_model(min_common=3)

    for item_index, (neighbors, similarities, counts) in enumerate(
        zip(
            model.neighbor_indices,
            model.neighbor_similarities,
            model.neighbor_common_counts,
            strict=True,
        )
    ):
        assert item_index not in neighbors
        assert np.all(counts >= 3)
        keys = [
            (-abs(float(similarity)), -float(similarity), int(model.item_ids[neighbor]))
            for neighbor, similarity in zip(neighbors, similarities, strict=True)
        ]
        assert keys == sorted(keys)
        assert np.array_equal(neighbors, repeated.neighbor_indices[item_index])
        assert np.allclose(similarities, repeated.neighbor_similarities[item_index])


def test_item_knn_uses_signed_numerator_absolute_denominator_and_fallback():
    model = ItemKNNModel(n_users=2, n_items=3, n_neighbors=2)
    model.baseline.global_mean = 3.0
    model.baseline.user_bias = np.array([0.0, 0.0], dtype=np.float64)
    model.baseline.item_bias = np.array([0.0, 3.5, -5.0], dtype=np.float64)
    model.residual_matrix = sparse.csr_matrix(
        np.array([[0.0, 2.0, 2.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    )
    model.observed_matrix = sparse.csr_matrix(
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    )
    model.neighbor_indices = [
        np.array([1, 2], dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    ]
    model.neighbor_similarities = [
        np.array([-0.5, 0.25], dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    ]
    model.neighbor_common_counts = [
        np.array([3, 3], dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    ]

    raw = model.predict_pairs(
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 2]),
        clip=False,
    )
    clipped = model.predict_pairs(
        np.array([0, 1, 0, 0]),
        np.array([0, 0, 1, 2]),
        clip=True,
    )

    assert raw[0] == pytest.approx(3.0 + (-0.5 * 2.0 + 0.25 * 2.0) / 0.75)
    assert raw[1] == pytest.approx(3.0)
    assert raw[2] == pytest.approx(6.5)
    assert raw[3] == pytest.approx(-2.0)
    assert clipped.tolist() == pytest.approx([raw[0], 3.0, 5.0, 1.0])


def test_item_knn_invalid_indices_and_validation_only_tuning_signature():
    model = _fit_small_model()
    with pytest.raises(ValueError, match="user index out of range"):
        model.predict_pairs(np.array([4]), np.array([0]))
    with pytest.raises(ValueError, match="item index out of range"):
        model.predict_pairs(np.array([0]), np.array([3]))
    with pytest.raises(ValueError, match="user index out of range"):
        model.predict_user(4)

    parameters = inspect.signature(run_pipeline._tune_item_knn).parameters
    assert set(parameters) == {
        "train_arrays",
        "validation_arrays",
        "n_users",
        "n_items",
        "bias_best_result",
        "item_ids",
    }
