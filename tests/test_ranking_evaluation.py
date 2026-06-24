from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.ranking_evaluation import (
    aggregate_ranking_metrics,
    build_temporal_ranking_protocol,
    candidate_movie_ids,
    rank_target,
    single_target_metrics,
)


def _protocol_ratings() -> pd.DataFrame:
    rows = [
        (1, 10, 3.0, 1),
        (1, 11, 5.0, 2),
        (1, 3, 5.0, 30),
        (1, 5, 4.0, 30),
        (1, 12, 2.0, 40),
        (2, 3, 5.0, 1),
        (2, 20, 3.0, 2),
        (2, 21, 4.0, 20),
        (3, 3, 4.0, 1),
        (3, 30, 2.0, 2),
        (3, 31, 5.0, 20),
        (4, 3, 5.0, 1),
        (4, 40, 4.0, 2),
    ]
    return pd.DataFrame(
        rows, columns=["user_id", "movie_id", "rating", "timestamp"]
    )


def test_temporal_protocol_uses_latest_positive_tie_break_and_strict_prefix():
    ranking_train, targets, metadata = build_temporal_ranking_protocol(
        _protocol_ratings(),
        min_prior_interactions=2,
        min_target_item_support=2,
    )

    user_one = targets.loc[targets["user_id"].eq(1)].iloc[0]
    assert int(user_one["movie_id"]) == 3
    assert int(user_one["timestamp"]) == 30
    user_one_history = ranking_train.loc[ranking_train["user_id"].eq(1)]
    assert set(user_one_history["movie_id"]) == {10, 11}
    assert user_one_history["timestamp"].lt(30).all()
    assert 3 not in set(user_one_history["movie_id"])
    assert 5 not in set(user_one_history["movie_id"])
    assert 12 not in set(user_one_history["movie_id"])
    assert metadata["sampled_negatives"] is False
    assert metadata["full_catalog_candidates"] is True
    assert metadata["exclusion_counts"]["insufficient_prior_history"] == 1


def test_temporal_protocol_filters_unsupported_targets():
    _, targets, metadata = build_temporal_ranking_protocol(
        _protocol_ratings(),
        min_prior_interactions=2,
        min_target_item_support=2,
    )

    assert set(targets["user_id"]) == {1}
    assert metadata["exclusion_counts"]["target_item_below_min_support"] == 2
    assert int(targets.iloc[0]["target_item_support"]) == 2


def test_full_catalog_candidates_exclude_history_and_keep_target():
    candidates = candidate_movie_ids(
        np.array([3, 5, 10, 11, 12], dtype=np.int32),
        np.array([10, 11], dtype=np.int32),
        target_movie_id=3,
    )
    assert candidates.tolist() == [3, 5, 12]
    with pytest.raises(ValueError, match="target movie appears"):
        candidate_movie_ids(
            np.array([3, 5], dtype=np.int32),
            np.array([3], dtype=np.int32),
            target_movie_id=3,
        )


def test_target_rank_tie_break_and_single_target_metrics_are_exact():
    rank = rank_target(
        np.array([9, 3, 5], dtype=np.int32),
        np.array([1.0, 1.0, 0.5], dtype=np.float64),
        target_movie_id=9,
    )
    assert rank == 2
    at_five = single_target_metrics(rank, 5)
    assert at_five["hit"] is True
    assert at_five["ndcg"] == pytest.approx(1.0 / np.log2(3.0))
    assert at_five["mrr"] == pytest.approx(0.5)
    at_one = single_target_metrics(rank, 1)
    assert at_one == {"hit": False, "ndcg": 0.0, "mrr": 0.0}


def test_aggregate_metrics_reconstruct_from_ranks():
    frame = pd.DataFrame(
        {
            "bias_target_rank": [1, 10, 11],
            "item_knn_target_rank": [2, 6, 12],
            "svd_target_rank": [3, 7, 13],
            "pmf_target_rank": [4, 8, 14],
        }
    )
    metrics = aggregate_ranking_metrics(frame)

    bias = metrics["models"]["BiasBaseline"]
    assert bias["HitRate@5"] == pytest.approx(1 / 3)
    assert bias["HitRate@10"] == pytest.approx(2 / 3)
    assert bias["NDCG@10"] == pytest.approx(
        (1.0 + 1.0 / np.log2(11.0)) / 3
    )
    assert bias["MRR@10"] == pytest.approx((1.0 + 0.1) / 3)
    assert bias["mean_target_rank"] == pytest.approx(22 / 3)
    assert bias["median_target_rank"] == 10.0
