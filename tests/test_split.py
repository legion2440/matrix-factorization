from __future__ import annotations

import pandas as pd

from utils.split import deterministic_user_split


def _keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(map(tuple, frame[["user_id", "movie_id"]].to_numpy()))


def test_split_is_deterministic_and_non_overlapping(synthetic_ratings):
    first = deterministic_user_split(synthetic_ratings, random_state=42)
    second = deterministic_user_split(synthetic_ratings, random_state=42)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
    assert not (_keys(first.train) & _keys(first.validation))
    assert not (_keys(first.train) & _keys(first.test))
    assert not (_keys(first.validation) & _keys(first.test))


def test_split_sizes_and_train_coverage(synthetic_ratings):
    split = deterministic_user_split(synthetic_ratings, random_state=42)
    assert len(split.train) + len(split.validation) + len(split.test) == len(
        synthetic_ratings
    )
    assert set(synthetic_ratings["user_id"]) == set(split.train["user_id"])
    assert set(split.validation["movie_id"]).issubset(set(split.train["movie_id"]))
    assert set(split.test["movie_id"]).issubset(set(split.train["movie_id"]))

