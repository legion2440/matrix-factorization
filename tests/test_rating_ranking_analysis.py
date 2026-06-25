from __future__ import annotations

import pandas as pd
import pytest

from utils.rating_ranking_analysis import (
    EXPECTED_MODELS,
    analyze_rating_ranking,
    build_comparison_table,
    build_rating_table,
    competition_positions,
    deterministic_model_order,
)


def _metrics() -> dict[str, float]:
    return {
        "BiasBaseline_MSE": 0.824118,
        "BiasBaseline_RMSE": 0.90781,
        "ItemKNN_MSE": 0.737614,
        "ItemKNN_RMSE": 0.858845,
        "SVD_MSE": 0.793518,
        "SVD_RMSE": 0.890796,
        "PMF_MSE": 0.712165,
        "PMF_RMSE": 0.843899,
    }


def _ranking_metrics() -> dict[str, object]:
    hit_rates = {
        "BiasBaseline": 0.03,
        "ItemKNN": 0.01,
        "SVD": 0.08,
        "PMF": 0.06,
    }
    return {
        "models": {
            model: {
                "HitRate@5": hit_rates[model] / 2,
                "HitRate@10": hit_rates[model],
                "median_target_rank": 100 + index,
            }
            for index, model in enumerate(EXPECTED_MODELS)
        }
    }


def _ranking_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bias_target_rank": [100, 2500, 300],
            "item_knn_target_rank": [200, 2600, 400],
            "svd_target_rank": [10, 2000, 50],
            "pmf_target_rank": [20, 1900, 40],
        }
    )


def test_ordering_and_reversal_match_rating_vs_ranking_result():
    analysis = analyze_rating_ranking(
        _metrics(), _ranking_metrics(), _ranking_results()
    )
    positions = analysis.comparison_table.set_index("model")

    assert positions.loc["ItemKNN", "rmse_position"] == 2
    assert positions.loc["ItemKNN", "hit_rate_10_position"] == 4
    assert positions.loc["SVD", "rmse_position"] == 3
    assert positions.loc["SVD", "hit_rate_10_position"] == 1
    assert {"ItemKNN", "SVD"}.issubset(set(analysis.reversal_table["model"]))


def test_ties_use_competition_positions_and_deterministic_model_order():
    frame = pd.DataFrame(
        {
            "model": list(EXPECTED_MODELS),
            "score": [1.0, 0.5, 0.5, 2.0],
        }
    )

    positions = competition_positions(frame, "score", ascending=True)

    assert positions.tolist() == [3, 1, 1, 4]
    assert deterministic_model_order(frame, "score", ascending=True) == [
        "ItemKNN",
        "SVD",
        "BiasBaseline",
        "PMF",
    ]


def test_missing_or_extra_models_fail_fast():
    rating = build_rating_table(_metrics()).iloc[:-1]
    with pytest.raises(ValueError, match="models must be exactly"):
        build_comparison_table(rating, _ranking_metrics(), _ranking_results())

    ranking_metrics = _ranking_metrics()
    ranking_metrics["models"].pop("PMF")
    with pytest.raises(ValueError, match="models must be exactly"):
        build_comparison_table(
            build_rating_table(_metrics()),
            ranking_metrics,
            _ranking_results(),
        )
