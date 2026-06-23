from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]
USER_COLUMNS = ["user_id", "gender", "age", "occupation", "zip_code"]


@dataclass(frozen=True)
class MovieLensData:
    ratings: pd.DataFrame
    movies: pd.DataFrame
    users: pd.DataFrame


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required MovieLens file is missing: {path}")
    return pd.read_csv(
        path,
        sep="::",
        names=names,
        engine="python",
        encoding="latin-1",
        dtype=str,
        keep_default_na=False,
    )


def load_movielens(data_dir: str | Path) -> MovieLensData:
    data_dir = Path(data_dir)
    ratings = _read_dat(data_dir / "ratings.dat", RATING_COLUMNS)
    movies = _read_dat(data_dir / "movies.dat", MOVIE_COLUMNS)
    users = _read_dat(data_dir / "users.dat", USER_COLUMNS)

    for column in ("user_id", "movie_id", "rating", "timestamp"):
        ratings[column] = pd.to_numeric(ratings[column], errors="raise")
    ratings = ratings.astype(
        {"user_id": "int32", "movie_id": "int32", "rating": "float32", "timestamp": "int64"}
    )

    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="raise").astype("int32")
    for column in ("user_id", "age", "occupation"):
        users[column] = pd.to_numeric(users[column], errors="raise")
    users = users.astype({"user_id": "int32", "age": "int16", "occupation": "int16"})
    return MovieLensData(ratings=ratings, movies=movies, users=users)


def validate_movielens(data: MovieLensData) -> dict[str, Any]:
    ratings, movies, users = data.ratings, data.movies, data.users
    errors: list[str] = []

    for name, frame, required in (
        ("ratings", ratings, RATING_COLUMNS),
        ("movies", movies, MOVIE_COLUMNS),
        ("users", users, USER_COLUMNS),
    ):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            errors.append(f"{name}: missing columns {missing}")
        if frame.empty:
            errors.append(f"{name}: table is empty")
        if frame.isnull().any().any():
            errors.append(f"{name}: null values found")

    if not pd.api.types.is_integer_dtype(ratings["user_id"]):
        errors.append("ratings.user_id must be integer")
    if not pd.api.types.is_integer_dtype(ratings["movie_id"]):
        errors.append("ratings.movie_id must be integer")
    if not pd.api.types.is_numeric_dtype(ratings["rating"]):
        errors.append("ratings.rating must be numeric")
    if not ratings["rating"].between(1, 5).all():
        errors.append("ratings.rating must be within [1, 5]")
    if ratings.duplicated(["user_id", "movie_id"]).any():
        errors.append("ratings contain duplicate user/movie interactions")
    if movies["movie_id"].duplicated().any():
        errors.append("movies.movie_id must be unique")
    if users["user_id"].duplicated().any():
        errors.append("users.user_id must be unique")

    orphan_users = set(ratings["user_id"]) - set(users["user_id"])
    orphan_movies = set(ratings["movie_id"]) - set(movies["movie_id"])
    if orphan_users:
        errors.append(f"ratings reference {len(orphan_users)} unknown users")
    if orphan_movies:
        errors.append(f"ratings reference {len(orphan_movies)} unknown movies")

    if errors:
        raise ValueError("Invalid MovieLens data:\n- " + "\n- ".join(errors))

    user_counts = ratings.groupby("user_id").size()
    movie_counts = ratings.groupby("movie_id").size()
    n_users = int(users["user_id"].nunique())
    n_movies = int(movies["movie_id"].nunique())
    observed = int(len(ratings))
    possible = n_users * n_movies
    return {
        "n_users": n_users,
        "n_movies": n_movies,
        "n_ratings": observed,
        "rating_distribution": {
            str(int(k)): int(v)
            for k, v in ratings["rating"].value_counts().sort_index().items()
        },
        "user_interactions": {
            "min": int(user_counts.min()),
            "median": float(user_counts.median()),
            "max": int(user_counts.max()),
        },
        "movie_interactions": {
            "min": int(movie_counts.min()),
            "median": float(movie_counts.median()),
            "max": int(movie_counts.max()),
        },
        "matrix_sparsity": float(1.0 - observed / possible),
    }

