from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


@dataclass
class SVDModel:
    n_factors: int = 80
    item_bias_regularization: float = 20.0
    random_state: int = 42

    def __post_init__(self) -> None:
        self.user_factors: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_means: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.shape: tuple[int, int] | None = None

    def fit(self, residual_matrix: sparse.spmatrix, user_means: np.ndarray) -> "SVDModel":
        matrix = sparse.csr_matrix(residual_matrix, dtype=np.float64)
        max_rank = min(matrix.shape) - 1
        if not 1 <= self.n_factors <= max_rank:
            raise ValueError(f"n_factors must be in [1, {max_rank}]")
        if user_means.shape != (matrix.shape[0],):
            raise ValueError("user_means shape does not align with matrix")

        counts = np.asarray(matrix.getnnz(axis=0), dtype=np.float64)
        residual_sums = np.asarray(matrix.sum(axis=0)).ravel()
        self.item_bias = (
            residual_sums / (counts + float(self.item_bias_regularization))
        ).astype(np.float32)
        adjusted = matrix.copy()
        adjusted.data -= self.item_bias[adjusted.indices]

        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(matrix.shape))
        user_factors, singular_values, item_factors = svds(
            adjusted,
            k=self.n_factors,
            v0=v0,
            solver="arpack",
        )
        order = np.argsort(singular_values)[::-1]
        self.user_factors = user_factors[:, order].astype(np.float32)
        self.singular_values = singular_values[order].astype(np.float32)
        self.item_factors = item_factors[order, :].astype(np.float32)
        self.user_means = np.asarray(user_means, dtype=np.float32)
        self.shape = matrix.shape
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.user_factors,
                self.singular_values,
                self.item_factors,
                self.user_means,
                self.item_bias,
            )
        ):
            raise RuntimeError("SVDModel has not been fitted")

    def predict_pairs(
        self,
        user_indices: np.ndarray,
        movie_indices: np.ndarray,
        n_factors: int | None = None,
    ) -> np.ndarray:
        self._check_fitted()
        user_indices = np.asarray(user_indices, dtype=np.int64)
        movie_indices = np.asarray(movie_indices, dtype=np.int64)
        if user_indices.shape != movie_indices.shape:
            raise ValueError("user_indices and movie_indices must have the same shape")
        k = self.n_factors if n_factors is None else int(n_factors)
        if not 1 <= k <= self.n_factors:
            raise ValueError(f"n_factors must be in [1, {self.n_factors}]")
        scaled_users = self.user_factors[user_indices, :k] * self.singular_values[:k]
        residuals = np.sum(
            scaled_users * self.item_factors[:k, movie_indices].T,
            axis=1,
        )
        predictions = (
            residuals
            + self.user_means[user_indices]
            + self.item_bias[movie_indices]
        )
        return np.clip(predictions, 1.0, 5.0).astype(np.float32)

    def predict_all(self, n_factors: int | None = None) -> np.ndarray:
        self._check_fitted()
        k = self.n_factors if n_factors is None else int(n_factors)
        if not 1 <= k <= self.n_factors:
            raise ValueError(f"n_factors must be in [1, {self.n_factors}]")
        reconstructed = (
            self.user_factors[:, :k] * self.singular_values[:k]
        ) @ self.item_factors[:k, :]
        reconstructed += self.user_means[:, None]
        reconstructed += self.item_bias[None, :]
        return np.clip(reconstructed, 1.0, 5.0).astype(np.float32)
