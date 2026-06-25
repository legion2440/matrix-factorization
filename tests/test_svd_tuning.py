from __future__ import annotations

import numpy as np

from utils.artifacts import prepare_svd_rank_tuning


def test_svd_rank_tuning_preparation_filters_selected_regularization():
    results = [
        {
            "n_factors": rank,
            "item_bias_regularization": regularization,
            "validation_rmse": 0.9 + rank / 10000 + regularization / 100000,
        }
        for regularization in (0.0, 5.0)
        for rank in (5, 10, 20, 40, 60)
    ]

    prepared = prepare_svd_rank_tuning(
        results,
        selected_rank=20,
        selected_item_bias_regularization=5.0,
    )

    assert prepared["n_factors"].tolist() == [5, 10, 20, 40, 60]
    assert prepared["selected"].tolist() == [False, False, True, False, False]
    assert np.allclose(
        prepared["validation_mse"],
        prepared["validation_rmse"] ** 2,
        rtol=0.0,
        atol=0.0,
    )
