from __future__ import annotations

import numpy as np


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual.size == 0:
        raise ValueError("metrics require at least one observation")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite values")
    return float(np.mean(np.square(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mse(actual, predicted)))

