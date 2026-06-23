from __future__ import annotations

import inspect

import numpy as np
import pytest

from models.baseline_cf import BaselineCFModel
from scripts import run_pipeline


def _training_arrays():
    users = np.array([0, 0, 1, 1, 2], dtype=np.int32)
    items = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    ratings = np.array([5.0, 4.0, 2.0, 1.0, 5.0], dtype=np.float32)
    return users, items, ratings


def test_predict_pairs_uses_global_user_and_item_bias_formula():
    model = BaselineCFModel(n_users=2, n_items=2)
    model.global_mean = 3.0
    model.user_bias = np.array([0.5, -0.25], dtype=np.float64)
    model.item_bias = np.array([0.75, -1.0], dtype=np.float64)

    raw = model.predict_pairs(
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        clip=False,
    )

    assert raw.tolist() == pytest.approx([4.25, 1.75])


def test_baseline_fit_is_deterministic_and_regularized():
    arrays = _training_arrays()
    low_reg = BaselineCFModel(
        n_users=3,
        n_items=2,
        user_regularization=0.1,
        item_regularization=0.1,
        n_iterations=10,
    ).fit(*arrays)
    high_reg = BaselineCFModel(
        n_users=3,
        n_items=2,
        user_regularization=100.0,
        item_regularization=100.0,
        n_iterations=10,
    ).fit(*arrays)
    repeat = BaselineCFModel(
        n_users=3,
        n_items=2,
        user_regularization=0.1,
        item_regularization=0.1,
        n_iterations=10,
    ).fit(*arrays)

    assert np.allclose(low_reg.user_bias, repeat.user_bias)
    assert np.allclose(low_reg.item_bias, repeat.item_bias)
    assert np.linalg.norm(high_reg.user_bias) < np.linalg.norm(low_reg.user_bias)
    assert np.linalg.norm(high_reg.item_bias) < np.linalg.norm(low_reg.item_bias)


def test_baseline_clipping_and_invalid_indices():
    model = BaselineCFModel(n_users=1, n_items=2)
    model.global_mean = 3.0
    model.user_bias = np.array([2.5], dtype=np.float64)
    model.item_bias = np.array([1.0, -5.0], dtype=np.float64)

    clipped = model.predict_pairs(np.array([0, 0]), np.array([0, 1]))
    raw = model.predict_pairs(np.array([0, 0]), np.array([0, 1]), clip=False)

    assert clipped.tolist() == [5.0, 1.0]
    assert raw.tolist() == pytest.approx([6.5, 0.5])
    with pytest.raises(ValueError, match="user index out of range"):
        model.predict_pairs(np.array([1]), np.array([0]))
    with pytest.raises(ValueError, match="item index out of range"):
        model.predict_pairs(np.array([0]), np.array([2]))


def test_baseline_tuning_orchestration_does_not_accept_test_data(monkeypatch):
    parameters = inspect.signature(run_pipeline._tune_baseline_cf).parameters
    assert set(parameters) == {
        "train_arrays",
        "validation_arrays",
        "n_users",
        "n_items",
    }

    calls: dict[str, object] = {}

    class RecordingBaseline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, *arrays):
            calls["fit_arrays"] = arrays
            return self

    monkeypatch.setattr(run_pipeline, "BaselineCFModel", RecordingBaseline)
    arrays = (
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([4.0], dtype=np.float32),
    )
    best = {
        "user_regularization": 10.0,
        "item_regularization": 10.0,
        "n_iterations": 5,
    }
    run_pipeline._fit_final_baseline_cf(arrays, 1, 1, best)

    assert calls["init"]["random_state"] == 42
    assert len(calls["fit_arrays"]) == 3
    assert all(
        actual is expected
        for actual, expected in zip(calls["fit_arrays"], arrays, strict=True)
    )
