from __future__ import annotations

import json


def test_metric_artifact_schema(tmp_path):
    payload = {
        "BiasBaseline_MSE": 0.9,
        "BiasBaseline_RMSE": 0.95,
        "ItemKNN_MSE": 0.82,
        "ItemKNN_RMSE": 0.91,
        "SVD_MSE": 0.8,
        "SVD_RMSE": 0.89,
        "PMF_MSE": 0.7,
        "PMF_RMSE": 0.83,
        "PMF_vs_SVD_improvement_%": 6.7,
        "bias_baseline_best_params": {"user_regularization": 10.0},
        "item_knn_best_params": {"k": 40},
        "svd_best_params": {"n_factors": 40},
        "pmf_best_params": {"n_factors": 64},
    }
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload).issubset(loaded)
