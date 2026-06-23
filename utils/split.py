from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RatingSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _allocation(n: int) -> tuple[int, int, int]:
    if n < 3:
        return n, 0, 0
    n_validation = max(1, int(np.floor(n * 0.15)))
    n_test = max(1, int(np.floor(n * 0.15)))
    if n - n_validation - n_test < 1:
        n_validation, n_test = 1, 1
    return n - n_validation - n_test, n_validation, n_test


def deterministic_user_split(
    ratings: pd.DataFrame,
    random_state: int = 42,
) -> RatingSplit:
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings missing required columns: {sorted(missing)}")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Cannot split duplicate user/movie interactions")

    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []

    stable = ratings.sort_values(["user_id", "movie_id", "timestamp"], kind="mergesort")
    for _, group in stable.groupby("user_id", sort=True):
        indices = group.index.to_numpy(copy=True)
        indices = indices[rng.permutation(len(indices))]
        n_train, n_validation, _ = _allocation(len(indices))
        train_indices.extend(indices[:n_train])
        validation_indices.extend(indices[n_train : n_train + n_validation])
        test_indices.extend(indices[n_train + n_validation :])

    split_labels = pd.Series("train", index=ratings.index, dtype="object")
    split_labels.loc[validation_indices] = "validation"
    split_labels.loc[test_indices] = "test"

    # Guarantee movie coverage without leaking validation or test into each other.
    train_movies = set(ratings.loc[split_labels.eq("train"), "movie_id"])
    held_out = ratings.loc[~split_labels.eq("train")].sort_values(
        ["movie_id", "user_id", "timestamp"], kind="mergesort"
    )
    for movie_id, group in held_out.groupby("movie_id", sort=True):
        if movie_id not in train_movies:
            chosen_index = int(group.index[0])
            split_labels.loc[chosen_index] = "train"
            train_movies.add(int(movie_id))

    columns = list(ratings.columns)
    train = ratings.loc[split_labels.eq("train"), columns].sort_index().reset_index(drop=True)
    validation = (
        ratings.loc[split_labels.eq("validation"), columns].sort_index().reset_index(drop=True)
    )
    test = ratings.loc[split_labels.eq("test"), columns].sort_index().reset_index(drop=True)
    validate_split(ratings, RatingSplit(train, validation, test))
    return RatingSplit(train=train, validation=validation, test=test)


def validate_split(original: pd.DataFrame, split: RatingSplit) -> None:
    key = ["user_id", "movie_id"]
    train_keys = set(map(tuple, split.train[key].to_numpy()))
    validation_keys = set(map(tuple, split.validation[key].to_numpy()))
    test_keys = set(map(tuple, split.test[key].to_numpy()))

    if train_keys & validation_keys or train_keys & test_keys or validation_keys & test_keys:
        raise ValueError("Split partitions overlap")
    original_keys = set(map(tuple, original[key].to_numpy()))
    if train_keys | validation_keys | test_keys != original_keys:
        raise ValueError("Split does not preserve all original interactions")

    train_users = set(split.train["user_id"])
    train_movies = set(split.train["movie_id"])
    if set(split.validation["user_id"]) - train_users or set(split.test["user_id"]) - train_users:
        raise ValueError("Validation/test contains users absent from training")
    if set(split.validation["movie_id"]) - train_movies or set(split.test["movie_id"]) - train_movies:
        raise ValueError("Validation/test contains movies absent from training")
    if set(original["user_id"]) - train_users:
        raise ValueError("At least one user has no training interactions")

