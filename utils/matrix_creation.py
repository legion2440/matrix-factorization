from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class MatrixArtifacts:
    residual_matrix: sparse.csr_matrix
    user_means: np.ndarray
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_user: np.ndarray
    index_to_movie: np.ndarray


def create_mappings(
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
    user_ids = np.sort(users["user_id"].astype(int).unique())
    movie_ids = np.sort(movies["movie_id"].astype(int).unique())
    user_to_index = {int(raw): int(index) for index, raw in enumerate(user_ids)}
    movie_to_index = {int(raw): int(index) for index, raw in enumerate(movie_ids)}
    return user_to_index, movie_to_index, user_ids.astype(np.int32), movie_ids.astype(np.int32)


def build_normalized_matrix(
    ratings: pd.DataFrame,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
) -> tuple[sparse.csr_matrix, np.ndarray]:
    user_indices = ratings["user_id"].map(user_to_index)
    movie_indices = ratings["movie_id"].map(movie_to_index)
    if user_indices.isnull().any() or movie_indices.isnull().any():
        raise ValueError("Ratings contain IDs absent from mappings")

    means_by_id = ratings.groupby("user_id", sort=False)["rating"].mean()
    user_means = np.full(len(user_to_index), np.nan, dtype=np.float32)
    for user_id, mean in means_by_id.items():
        user_means[user_to_index[int(user_id)]] = np.float32(mean)
    if not np.isfinite(user_means).all():
        raise ValueError("Every mapped user must have at least one training rating")

    residuals = ratings["rating"].to_numpy(np.float32) - user_means[
        user_indices.to_numpy(np.int64)
    ]
    matrix = sparse.coo_matrix(
        (
            residuals.astype(np.float32),
            (
                user_indices.to_numpy(np.int32),
                movie_indices.to_numpy(np.int32),
            ),
        ),
        shape=(len(user_to_index), len(movie_to_index)),
        dtype=np.float32,
    ).tocsr()
    return matrix, user_means


def save_mappings(
    mappings_dir: str | Path,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    index_to_user: np.ndarray,
    index_to_movie: np.ndarray,
) -> None:
    mappings_dir = Path(mappings_dir)
    mappings_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "user_to_index.json": {str(k): v for k, v in user_to_index.items()},
        "movie_to_index.json": {str(k): v for k, v in movie_to_index.items()},
        "index_to_user.json": [int(v) for v in index_to_user],
        "index_to_movie.json": [int(v) for v in index_to_movie],
    }
    for filename, payload in values.items():
        with (mappings_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def load_mappings(
    mappings_dir: str | Path,
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray]:
    mappings_dir = Path(mappings_dir)
    with (mappings_dir / "user_to_index.json").open(encoding="utf-8") as handle:
        user_to_index = {int(k): int(v) for k, v in json.load(handle).items()}
    with (mappings_dir / "movie_to_index.json").open(encoding="utf-8") as handle:
        movie_to_index = {int(k): int(v) for k, v in json.load(handle).items()}
    with (mappings_dir / "index_to_user.json").open(encoding="utf-8") as handle:
        index_to_user = np.asarray(json.load(handle), dtype=np.int32)
    with (mappings_dir / "index_to_movie.json").open(encoding="utf-8") as handle:
        index_to_movie = np.asarray(json.load(handle), dtype=np.int32)
    return user_to_index, movie_to_index, index_to_user, index_to_movie


def save_normalized_matrix_csv(
    path: str | Path,
    residual_matrix: sparse.csr_matrix,
    index_to_user: np.ndarray,
    index_to_movie: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dense = residual_matrix.toarray().astype(np.float32)
    mask = np.zeros(residual_matrix.shape, dtype=bool)
    rows = np.repeat(
        np.arange(residual_matrix.shape[0], dtype=np.int32),
        np.diff(residual_matrix.indptr),
    )
    mask[rows, residual_matrix.indices] = True
    dense[~mask] = np.nan
    frame = pd.DataFrame(dense, index=index_to_user, columns=index_to_movie)
    frame.index.name = "user_id"
    frame.to_csv(path, float_format="%.4g", na_rep="")
