from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from models.bias_baseline import BiasBaselineModel


@dataclass(frozen=True)
class ItemKNNConfig:
    n_neighbors: int = 40
    shrinkage: float = 50.0
    min_common: int = 3
    similarity_chunk_size: int = 128


class ItemKNNModel:
    """Residualized item-kNN with shrunk cosine similarities."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_neighbors: int = 40,
        shrinkage: float = 50.0,
        min_common: int = 3,
        baseline_user_regularization: float = 10.0,
        baseline_item_regularization: float = 10.0,
        baseline_iterations: int = 20,
        random_state: int = 42,
        item_ids: np.ndarray | None = None,
        similarity_chunk_size: int = 128,
    ) -> None:
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be positive")
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        if min_common <= 0:
            raise ValueError("min_common must be positive")
        if similarity_chunk_size <= 0:
            raise ValueError("similarity_chunk_size must be positive")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.config = ItemKNNConfig(
            n_neighbors=int(n_neighbors),
            shrinkage=float(shrinkage),
            min_common=int(min_common),
            similarity_chunk_size=int(similarity_chunk_size),
        )
        self.item_ids = (
            np.arange(self.n_items, dtype=np.int64)
            if item_ids is None
            else np.asarray(item_ids, dtype=np.int64)
        )
        if self.item_ids.shape != (self.n_items,):
            raise ValueError("item_ids must align with n_items")
        if np.unique(self.item_ids).size != self.n_items:
            raise ValueError("item_ids must be unique")

        self.baseline = BiasBaselineModel(
            n_users=self.n_users,
            n_items=self.n_items,
            user_regularization=baseline_user_regularization,
            item_regularization=baseline_item_regularization,
            n_iterations=baseline_iterations,
            random_state=random_state,
        )
        self.residual_matrix: sparse.csr_matrix | None = None
        self.observed_matrix: sparse.csr_matrix | None = None
        self.neighbor_indices: list[np.ndarray] | None = None
        self.neighbor_similarities: list[np.ndarray] | None = None
        self.neighbor_common_counts: list[np.ndarray] | None = None
        self._neighbor_matrix_cache: dict[
            int, tuple[sparse.csr_matrix, sparse.csr_matrix]
        ] = {}

    def _check_indices(self, users: np.ndarray, items: np.ndarray) -> None:
        if users.shape != items.shape:
            raise ValueError("user and item arrays must have the same shape")
        if users.size and (users.min() < 0 or users.max() >= self.n_users):
            raise ValueError("user index out of range")
        if items.size and (items.min() < 0 or items.max() >= self.n_items):
            raise ValueError("item index out of range")

    def fit(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
    ) -> "ItemKNNModel":
        users = np.asarray(users, dtype=np.int32)
        items = np.asarray(items, dtype=np.int32)
        ratings = np.asarray(ratings, dtype=np.float64)
        self._check_indices(users, items)
        if users.shape != ratings.shape or ratings.size == 0:
            raise ValueError("training arrays must be non-empty and aligned")
        if not np.isfinite(ratings).all():
            raise ValueError("ratings must be finite")

        self.baseline.fit(users, items, ratings)
        baseline_scores = self.baseline.predict_pairs(users, items, clip=False).astype(
            np.float64
        )
        residuals = ratings - baseline_scores
        if not np.isfinite(residuals).all():
            raise FloatingPointError("item-kNN residuals contain non-finite values")

        shape = (self.n_users, self.n_items)
        self.residual_matrix = sparse.coo_matrix(
            (residuals, (users, items)), shape=shape, dtype=np.float64
        ).tocsr()
        self.residual_matrix.sort_indices()
        self.observed_matrix = sparse.coo_matrix(
            (np.ones(ratings.size, dtype=np.float64), (users, items)),
            shape=shape,
            dtype=np.float64,
        ).tocsr()
        self.observed_matrix.sort_indices()
        self._build_neighbors()
        return self

    def _build_neighbors(self) -> None:
        assert self.residual_matrix is not None
        assert self.observed_matrix is not None

        residual_for_similarity = self.residual_matrix.copy()
        residual_for_similarity.eliminate_zeros()
        item_residuals = residual_for_similarity.T.tocsr()
        item_observed = self.observed_matrix.T.tocsr()
        norms = np.sqrt(
            np.asarray(item_residuals.multiply(item_residuals).sum(axis=1)).ravel()
        )

        neighbor_indices: list[np.ndarray] = []
        neighbor_similarities: list[np.ndarray] = []
        neighbor_common_counts: list[np.ndarray] = []
        chunk_size = self.config.similarity_chunk_size
        for start in range(0, self.n_items, chunk_size):
            stop = min(start + chunk_size, self.n_items)
            dot_products = (item_residuals[start:stop] @ residual_for_similarity).toarray()
            common_counts = (item_observed[start:stop] @ self.observed_matrix).toarray()

            for offset, target_item in enumerate(range(start, stop)):
                common = common_counts[offset].astype(np.int32, copy=False)
                denominator = norms[target_item] * norms
                eligible = (
                    (common >= self.config.min_common)
                    & (denominator > 0.0)
                    & (np.arange(self.n_items) != target_item)
                )
                candidates = np.flatnonzero(eligible)
                if candidates.size == 0:
                    neighbor_indices.append(np.empty(0, dtype=np.int32))
                    neighbor_similarities.append(np.empty(0, dtype=np.float64))
                    neighbor_common_counts.append(np.empty(0, dtype=np.int32))
                    continue

                cosine = dot_products[offset, candidates] / denominator[candidates]
                cosine = np.clip(cosine, -1.0, 1.0)
                shrunk = cosine * (
                    common[candidates]
                    / (common[candidates].astype(np.float64) + self.config.shrinkage)
                )
                finite = np.isfinite(shrunk) & (shrunk != 0.0)
                candidates = candidates[finite]
                shrunk = shrunk[finite]
                counts = common[candidates]
                if candidates.size:
                    order = np.lexsort(
                        (
                            self.item_ids[candidates],
                            -shrunk,
                            -np.abs(shrunk),
                        )
                    )
                    order = order[: self.config.n_neighbors]
                    candidates = candidates[order]
                    shrunk = shrunk[order]
                    counts = counts[order]

                neighbor_indices.append(candidates.astype(np.int32, copy=False))
                neighbor_similarities.append(shrunk.astype(np.float64, copy=False))
                neighbor_common_counts.append(counts.astype(np.int32, copy=False))

        self.neighbor_indices = neighbor_indices
        self.neighbor_similarities = neighbor_similarities
        self.neighbor_common_counts = neighbor_common_counts
        self._neighbor_matrix_cache.clear()
        all_similarities = (
            np.concatenate(neighbor_similarities)
            if any(values.size for values in neighbor_similarities)
            else np.empty(0, dtype=np.float64)
        )
        if not np.isfinite(all_similarities).all():
            raise FloatingPointError("item-kNN similarities contain non-finite values")

    def _check_fitted(self) -> None:
        if (
            self.residual_matrix is None
            or self.observed_matrix is None
            or self.neighbor_indices is None
            or self.neighbor_similarities is None
            or self.neighbor_common_counts is None
        ):
            raise RuntimeError("ItemKNNModel has not been fitted")

    def _neighbor_matrices(
        self, n_neighbors: int | None
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        self._check_fitted()
        selected_k = (
            self.config.n_neighbors if n_neighbors is None else int(n_neighbors)
        )
        if not 1 <= selected_k <= self.config.n_neighbors:
            raise ValueError(
                f"n_neighbors must be in [1, {self.config.n_neighbors}]"
            )
        cached = self._neighbor_matrix_cache.get(selected_k)
        if cached is not None:
            return cached

        row_indices: list[np.ndarray] = []
        column_indices: list[np.ndarray] = []
        values: list[np.ndarray] = []
        for target_item, (neighbors, similarities) in enumerate(
            zip(
                self.neighbor_indices,
                self.neighbor_similarities,
                strict=True,
            )
        ):
            count = min(selected_k, neighbors.size)
            if count == 0:
                continue
            row_indices.append(np.full(count, target_item, dtype=np.int32))
            column_indices.append(neighbors[:count])
            values.append(similarities[:count])

        if values:
            rows = np.concatenate(row_indices)
            columns = np.concatenate(column_indices)
            data = np.concatenate(values)
        else:
            rows = np.empty(0, dtype=np.int32)
            columns = np.empty(0, dtype=np.int32)
            data = np.empty(0, dtype=np.float64)
        matrix = sparse.coo_matrix(
            (data, (rows, columns)),
            shape=(self.n_items, self.n_items),
            dtype=np.float64,
        ).tocsr()
        absolute = matrix.copy()
        absolute.data = np.abs(absolute.data)
        self._neighbor_matrix_cache[selected_k] = (matrix, absolute)
        return matrix, absolute

    def predict_users(
        self,
        user_indices: np.ndarray,
        n_neighbors: int | None = None,
        clip: bool = False,
    ) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(user_indices, dtype=np.int64)
        if users.ndim != 1:
            raise ValueError("user_indices must be one-dimensional")
        if users.size and (users.min() < 0 or users.max() >= self.n_users):
            raise ValueError("user index out of range")
        matrix, absolute = self._neighbor_matrices(n_neighbors)
        baseline = (
            self.baseline.global_mean
            + self.baseline.user_bias[users, None]
            + self.baseline.item_bias[None, :]
        )
        numerators = (self.residual_matrix[users] @ matrix.T).toarray()
        denominators = (self.observed_matrix[users] @ absolute.T).toarray()
        predictions = np.asarray(baseline, dtype=np.float64)
        usable = denominators > 0.0
        predictions[usable] += numerators[usable] / denominators[usable]
        if not np.isfinite(predictions).all():
            raise FloatingPointError("item-kNN predictions contain non-finite values")
        if clip:
            predictions = np.clip(predictions, 1.0, 5.0)
        return predictions.astype(np.float32)

    def predict_pairs(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        clip: bool = True,
        n_neighbors: int | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(user_indices, dtype=np.int64)
        items = np.asarray(item_indices, dtype=np.int64)
        self._check_indices(users, items)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        predictions = np.empty(users.size, dtype=np.float32)
        unique_users = np.unique(users)
        for start in range(0, unique_users.size, batch_size):
            batch_users = unique_users[start : start + batch_size]
            scores = self.predict_users(
                batch_users, n_neighbors=n_neighbors, clip=clip
            )
            positions = np.flatnonzero(np.isin(users, batch_users))
            row_lookup = np.searchsorted(batch_users, users[positions])
            predictions[positions] = scores[row_lookup, items[positions]]
        return predictions

    def predict_user(
        self,
        user_index: int,
        clip: bool = False,
        n_neighbors: int | None = None,
    ) -> np.ndarray:
        if not 0 <= int(user_index) < self.n_users:
            raise ValueError("user index out of range")
        return self.predict_users(
            np.asarray([int(user_index)], dtype=np.int64),
            n_neighbors=n_neighbors,
            clip=clip,
        )[0]
