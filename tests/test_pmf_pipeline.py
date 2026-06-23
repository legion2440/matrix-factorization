from __future__ import annotations

import inspect
import json

import numpy as np

from scripts import run_pipeline


def test_pmf_grid_is_bounded_joint_factor_regularization_search():
    assert len(run_pipeline.PMF_GRID) == 9
    assert {row["n_factors"] for row in run_pipeline.PMF_GRID} == {96, 112, 128}
    assert {
        row["factor_regularization"] for row in run_pipeline.PMF_GRID
    } == {0.05, 0.06, 0.07}
    assert {
        (row["n_factors"], row["factor_regularization"])
        for row in run_pipeline.PMF_GRID
    } == {
        (factors, regularization)
        for factors in (96, 112, 128)
        for regularization in (0.05, 0.06, 0.07)
    }
    assert all(row["learning_rate"] == 0.006 for row in run_pipeline.PMF_GRID)
    assert all(row["bias_regularization"] == 0.02 for row in run_pipeline.PMF_GRID)
    assert run_pipeline.PMF_TUNING_EPOCHS == 70
    assert run_pipeline.PMF_TUNING_PATIENCE == 8
    assert run_pipeline.PMF_TUNING_MIN_DELTA == 5e-5


def test_pmf_tuning_is_validation_only_and_svd_search_is_unchanged():
    parameters = inspect.signature(run_pipeline._tune_pmf).parameters
    assert set(parameters) == {
        "train_arrays",
        "validation_arrays",
        "n_users",
        "n_items",
    }
    assert run_pipeline.SVD_GRID == [5, 10, 20, 40, 60]
    assert run_pipeline.SVD_ITEM_BIAS_REGULARIZATION_GRID == [
        0.0,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
    ]


def test_pmf_selection_tie_break_is_deterministic():
    common = {
        "validation_rmse": 0.85,
        "learning_rate": 0.006,
        "bias_regularization": 0.02,
        "epochs_run": 50,
    }
    results = [
        {**common, "n_factors": 112, "factor_regularization": 0.07, "best_epoch": 30},
        {**common, "n_factors": 96, "factor_regularization": 0.05, "best_epoch": 20},
        {**common, "n_factors": 96, "factor_regularization": 0.07, "best_epoch": 35},
        {**common, "n_factors": 96, "factor_regularization": 0.07, "best_epoch": 25},
    ]
    selected = min(results, key=run_pipeline._pmf_result_sort_key)
    assert selected["n_factors"] == 96
    assert selected["factor_regularization"] == 0.07
    assert selected["best_epoch"] == 25


def test_final_refit_uses_selected_epoch_without_validation(monkeypatch):
    calls: dict[str, object] = {}

    class RecordingPMF:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def fit(self, *arrays):
            calls["fit_arrays"] = arrays
            return self

    monkeypatch.setattr(run_pipeline, "PMFModel", RecordingPMF)
    arrays = (
        np.array([0, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([4.0, 3.0], dtype=np.float32),
    )
    best = {
        "n_factors": 112,
        "learning_rate": 0.006,
        "factor_regularization": 0.07,
        "bias_regularization": 0.02,
        "best_epoch": 37,
    }
    run_pipeline._fit_final_pmf(arrays, 2, 2, best)

    assert calls["init"]["epochs"] == 37
    assert calls["init"]["random_state"] == 42
    assert len(calls["fit_arrays"]) == 3
    assert all(
        actual is expected
        for actual, expected in zip(calls["fit_arrays"], arrays, strict=True)
    )


def test_boundary_diagnostics_are_written_to_metadata(tmp_path):
    best = {
        "n_factors": 128,
        "learning_rate": 0.006,
        "factor_regularization": 0.06,
        "bias_regularization": 0.02,
        "best_epoch": 70,
        "validation_rmse": 0.84,
        "epochs_run": 70,
    }
    diagnostics = run_pipeline._pmf_search_diagnostics(best)
    assert diagnostics == {
        "selected_at_factor_boundary": True,
        "selected_at_epoch_boundary": True,
        "selected_early_stopping_triggered": False,
        "search_max_factors": 128,
        "search_max_epochs": 70,
    }

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"config": {}}', encoding="utf-8")
    run_pipeline._add_pmf_search_metadata(metadata_path, best, diagnostics)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["search_diagnostics"] == diagnostics
    assert metadata["selected_validation_result"]["best_epoch"] == 70
    assert (
        metadata["training_mode"]
        == "final_refit_train_plus_validation_without_holdout"
    )
