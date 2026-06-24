from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BiasBaselineConfig:
    user_regularization: float = 10.0
    item_regularization: float = 10.0
    n_iterations: int = 20
    random_state: int = 42


class BiasBaselineModel:
    """Regularized bias-only collaborative filtering baseline.

    The fitted score is:
        global_mean + user_bias[user] + item_bias[item]
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_regularization: float = 10.0,
        item_regularization: float = 10.0,
        n_iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        if n_users <= 0 or n_items <= 0:
            raise ValueError("n_users and n_items must be positive")
        if user_regularization < 0 or item_regularization < 0:
            raise ValueError("regularization values must be non-negative")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.config = BiasBaselineConfig(
            user_regularization=float(user_regularization),
            item_regularization=float(item_regularization),
            n_iterations=int(n_iterations),
            random_state=int(random_state),
        )
        self.global_mean = 0.0
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None

    def _check_indices(self, users: np.ndarray, items: np.ndarray) -> None:
        if users.shape != items.shape:
            raise ValueError("user and item arrays must have the same shape")
        if users.size and (users.min() < 0 or users.max() >= self.n_users):
            raise ValueError("user index out of range")
        if items.size and (items.min() < 0 or items.max() >= self.n_items):
            raise ValueError("item index out of range")

    def fit(
        self,
        train_users: np.ndarray,
        train_items: np.ndarray,
        train_ratings: np.ndarray,
    ) -> "BiasBaselineModel":
        users = np.asarray(train_users, dtype=np.int32)
        items = np.asarray(train_items, dtype=np.int32)
        ratings = np.asarray(train_ratings, dtype=np.float64)
        self._check_indices(users, items)
        if users.shape != ratings.shape or ratings.size == 0:
            raise ValueError("training arrays must be non-empty and aligned")
        if not np.isfinite(ratings).all():
            raise ValueError("ratings must be finite")

        self.global_mean = float(np.mean(ratings))
        self.user_bias = np.zeros(self.n_users, dtype=np.float64)
        self.item_bias = np.zeros(self.n_items, dtype=np.float64)
        user_counts = np.bincount(users, minlength=self.n_users).astype(np.float64)
        item_counts = np.bincount(items, minlength=self.n_items).astype(np.float64)

        for _ in range(self.config.n_iterations):
            user_residuals = ratings - self.global_mean - self.item_bias[items]
            user_sums = np.bincount(
                users, weights=user_residuals, minlength=self.n_users
            )
            self.user_bias = user_sums / (
                user_counts + self.config.user_regularization
            )

            item_residuals = ratings - self.global_mean - self.user_bias[users]
            item_sums = np.bincount(
                items, weights=item_residuals, minlength=self.n_items
            )
            self.item_bias = item_sums / (
                item_counts + self.config.item_regularization
            )

        if not np.isfinite(self.user_bias).all() or not np.isfinite(
            self.item_bias
        ).all():
            raise FloatingPointError("Bias baseline diverged to non-finite biases")
        return self

    def _check_fitted(self) -> None:
        if self.user_bias is None or self.item_bias is None:
            raise RuntimeError("BiasBaselineModel has not been fitted")

    def predict_pairs(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        clip: bool = True,
    ) -> np.ndarray:
        self._check_fitted()
        users = np.asarray(user_indices, dtype=np.int64)
        items = np.asarray(item_indices, dtype=np.int64)
        self._check_indices(users, items)
        predictions = self.global_mean + self.user_bias[users] + self.item_bias[items]
        if clip:
            predictions = np.clip(predictions, 1.0, 5.0)
        return predictions.astype(np.float32)

    def predict_user(self, user_index: int, clip: bool = True) -> np.ndarray:
        self._check_fitted()
        if not 0 <= int(user_index) < self.n_users:
            raise ValueError("user index out of range")
        predictions = self.global_mean + self.user_bias[int(user_index)] + self.item_bias
        if clip:
            predictions = np.clip(predictions, 1.0, 5.0)
        return predictions.astype(np.float32)
