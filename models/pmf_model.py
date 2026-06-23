from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - dependency is installed in the project environment
    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        if args and callable(args[0]):
            return args[0]
        return lambda function: function


@njit(cache=True)
def _sgd_epoch(
    user_indices: np.ndarray,
    movie_indices: np.ndarray,
    ratings: np.ndarray,
    order: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_bias: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    learning_rate: float,
    factor_regularization: float,
    bias_regularization: float,
) -> None:
    n_factors = user_factors.shape[1]
    for position in order:
        user = user_indices[position]
        movie = movie_indices[position]
        prediction = global_mean + user_bias[user] + item_bias[movie]
        for factor in range(n_factors):
            prediction += user_factors[user, factor] * item_factors[movie, factor]
        error = ratings[position] - prediction

        user_bias[user] += learning_rate * (
            error - bias_regularization * user_bias[user]
        )
        item_bias[movie] += learning_rate * (
            error - bias_regularization * item_bias[movie]
        )
        for factor in range(n_factors):
            old_user = user_factors[user, factor]
            old_item = item_factors[movie, factor]
            user_factors[user, factor] += learning_rate * (
                error * old_item - factor_regularization * old_user
            )
            item_factors[movie, factor] += learning_rate * (
                error * old_user - factor_regularization * old_item
            )


@dataclass
class PMFConfig:
    n_factors: int = 64
    learning_rate: float = 0.005
    factor_regularization: float = 0.05
    bias_regularization: float = 0.02
    epochs: int = 30
    patience: int = 5
    min_delta: float = 1e-4
    random_state: int = 42


class PMFModel:
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 64,
        learning_rate: float = 0.005,
        factor_regularization: float = 0.05,
        bias_regularization: float = 0.02,
        epochs: int = 30,
        patience: int = 5,
        min_delta: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        if n_users <= 0 or n_items <= 0 or n_factors <= 0:
            raise ValueError("n_users, n_items, and n_factors must be positive")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.config = PMFConfig(
            n_factors=n_factors,
            learning_rate=learning_rate,
            factor_regularization=factor_regularization,
            bias_regularization=bias_regularization,
            epochs=epochs,
            patience=patience,
            min_delta=min_delta,
            random_state=random_state,
        )
        self.global_mean = 0.0
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None
        self.history: list[dict[str, float | int | None]] = []
        self.best_epoch: int | None = None
        self.best_validation_rmse: float | None = None

    def _initialize(self, global_mean: float) -> None:
        rng = np.random.default_rng(self.config.random_state)
        scale = 0.08 / np.sqrt(max(1, self.config.n_factors / 32))
        self.user_factors = rng.normal(
            0.0, scale, size=(self.n_users, self.config.n_factors)
        ).astype(np.float32)
        self.item_factors = rng.normal(
            0.0, scale, size=(self.n_items, self.config.n_factors)
        ).astype(np.float32)
        self.user_bias = np.zeros(self.n_users, dtype=np.float32)
        self.item_bias = np.zeros(self.n_items, dtype=np.float32)
        self.global_mean = float(global_mean)
        self.history = []
        self.best_epoch = None
        self.best_validation_rmse = None

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
        validation_users: np.ndarray | None = None,
        validation_items: np.ndarray | None = None,
        validation_ratings: np.ndarray | None = None,
    ) -> "PMFModel":
        train_users = np.asarray(train_users, dtype=np.int32)
        train_items = np.asarray(train_items, dtype=np.int32)
        train_ratings = np.asarray(train_ratings, dtype=np.float32)
        self._check_indices(train_users, train_items)
        if train_users.shape != train_ratings.shape or train_ratings.size == 0:
            raise ValueError("training arrays must be non-empty and aligned")

        has_validation = validation_ratings is not None
        if has_validation:
            if validation_users is None or validation_items is None:
                raise ValueError("validation users and items are required")
            validation_users = np.asarray(validation_users, dtype=np.int32)
            validation_items = np.asarray(validation_items, dtype=np.int32)
            validation_ratings = np.asarray(validation_ratings, dtype=np.float32)
            self._check_indices(validation_users, validation_items)
            if validation_users.shape != validation_ratings.shape:
                raise ValueError("validation arrays must be aligned")

        self._initialize(float(np.mean(train_ratings)))
        rng = np.random.default_rng(self.config.random_state)
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        best_rmse = np.inf
        stale_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            order = rng.permutation(train_ratings.size).astype(np.int64)
            _sgd_epoch(
                train_users,
                train_items,
                train_ratings,
                order,
                self.user_factors,
                self.item_factors,
                self.user_bias,
                self.item_bias,
                self.global_mean,
                self.config.learning_rate,
                self.config.factor_regularization,
                self.config.bias_regularization,
            )
            train_predictions = self.predict_pairs(train_users, train_items, clip=True)
            train_rmse = float(
                np.sqrt(np.mean(np.square(train_ratings - train_predictions)))
            )
            validation_rmse: float | None = None
            if has_validation:
                validation_predictions = self.predict_pairs(
                    validation_users, validation_items, clip=True
                )
                validation_rmse = float(
                    np.sqrt(
                        np.mean(np.square(validation_ratings - validation_predictions))
                    )
                )
            if not np.isfinite(train_rmse) or (
                validation_rmse is not None and not np.isfinite(validation_rmse)
            ):
                raise FloatingPointError("PMF diverged to non-finite values")

            self.history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_rmse**2,
                    "train_rmse": train_rmse,
                    "validation_mse": None
                    if validation_rmse is None
                    else validation_rmse**2,
                    "validation_rmse": validation_rmse,
                }
            )

            if validation_rmse is not None:
                if validation_rmse < best_rmse - self.config.min_delta:
                    best_rmse = validation_rmse
                    self.best_epoch = epoch
                    self.best_validation_rmse = validation_rmse
                    best_state = (
                        self.user_factors.copy(),
                        self.item_factors.copy(),
                        self.user_bias.copy(),
                        self.item_bias.copy(),
                    )
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.config.patience:
                        break

        if best_state is not None:
            (
                self.user_factors,
                self.item_factors,
                self.user_bias,
                self.item_bias,
            ) = best_state
        elif self.history:
            self.best_epoch = int(self.history[-1]["epoch"])
        return self

    def _check_fitted(self) -> None:
        if any(
            value is None
            for value in (
                self.user_factors,
                self.item_factors,
                self.user_bias,
                self.item_bias,
            )
        ):
            raise RuntimeError("PMFModel has not been fitted")

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
        predictions = (
            self.global_mean
            + self.user_bias[users]
            + self.item_bias[items]
            + np.sum(self.user_factors[users] * self.item_factors[items], axis=1)
        )
        if clip:
            predictions = np.clip(predictions, 1.0, 5.0)
        return predictions.astype(np.float32)

    def predict_user(self, user_index: int, clip: bool = True) -> np.ndarray:
        self._check_fitted()
        if not 0 <= user_index < self.n_users:
            raise ValueError("user index out of range")
        predictions = (
            self.global_mean
            + self.user_bias[user_index]
            + self.item_bias
            + self.item_factors @ self.user_factors[user_index]
        )
        if clip:
            predictions = np.clip(predictions, 1.0, 5.0)
        return predictions.astype(np.float32)

    def save(self, directory: str | Path) -> None:
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "user_factors.npy", self.user_factors.astype(np.float32))
        np.save(directory / "item_factors.npy", self.item_factors.astype(np.float32))
        np.save(directory / "user_bias.npy", self.user_bias.astype(np.float32))
        np.save(directory / "item_bias.npy", self.item_bias.astype(np.float32))
        metadata = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "global_mean": self.global_mean,
            "config": asdict(self.config),
            "best_epoch": self.best_epoch,
            "best_validation_rmse": self.best_validation_rmse,
            "history": self.history,
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> "PMFModel":
        directory = Path(directory)
        with (directory / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        config = metadata["config"]
        model = cls(
            n_users=int(metadata["n_users"]),
            n_items=int(metadata["n_items"]),
            **config,
        )
        model.global_mean = float(metadata["global_mean"])
        model.user_factors = np.load(directory / "user_factors.npy").astype(np.float32)
        model.item_factors = np.load(directory / "item_factors.npy").astype(np.float32)
        model.user_bias = np.load(directory / "user_bias.npy").astype(np.float32)
        model.item_bias = np.load(directory / "item_bias.npy").astype(np.float32)
        model.best_epoch = metadata.get("best_epoch")
        model.best_validation_rmse = metadata.get("best_validation_rmse")
        model.history = metadata.get("history", [])
        return model

