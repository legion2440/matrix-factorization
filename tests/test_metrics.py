from __future__ import annotations

import numpy as np

from utils.metrics import mse, rmse


def test_metric_calculations():
    actual = np.array([1.0, 3.0, 5.0])
    predicted = np.array([2.0, 3.0, 4.0])
    assert mse(actual, predicted) == 2.0 / 3.0
    assert np.isclose(rmse(actual, predicted), np.sqrt(2.0 / 3.0))

