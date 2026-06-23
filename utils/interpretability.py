from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.pmf_model import PMFModel


AUDIT_USER_ROLES = (
    "train_profile_accurate",
    "train_profile_less_accurate",
    "test_case",
)

FACTOR_INTERPRETATION_COLUMNS = [
    "factor_index",
    "factor_variance",
    "polarity",
    "polarity_rank",
    "movie_id",
    "title",
    "genres",
    "factor_loading",
]

FACTOR_GENRE_PROFILE_COLUMNS = [
    "factor_index",
    "polarity",
    "genre",
    "movie_count",
    "genre_share",
    "mean_factor_loading",
]

SIMILARITY_COLUMNS = [
    "anchor_movie_id",
    "anchor_title",
    "anchor_genres",
    "similar_movie_id",
    "similar_title",
    "similar_genres",
    "cosine_similarity",
    "rank",
]

LOCAL_EXPLANATION_COLUMNS = [
    "user_id",
    "role",
    "recommendation_rank",
    "movie_id",
    "title",
    "genres",
    "raw_pmf_ranking_score",
    "clipped_displayed_rating",
    "global_mean_contribution",
    "user_bias_contribution",
    "item_bias_contribution",
    "total_latent_dot_product",
    "top_factor_1_index",
    "top_factor_1_contribution",
    "top_factor_2_index",
    "top_factor_2_contribution",
    "top_factor_3_index",
    "top_factor_3_contribution",
    "top_factor_contributions",
    "component_sum",
    "reconstruction_error",
    "nearest_known_movie_id",
    "nearest_known_title",
    "nearest_known_genres",
    "nearest_known_rating",
    "nearest_known_similarity",
    "common_genres",
]


def _split_genres(genres: str | float | None) -> list[str]:
    if not isinstance(genres, str) or not genres:
        return []
    return [genre for genre in genres.split("|") if genre]


def _movie_lookup(movies: pd.DataFrame) -> pd.DataFrame:
    return movies[["movie_id", "title", "genres"]].drop_duplicates("movie_id").set_index(
        "movie_id"
    )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 0.0
    value = float(np.dot(left, right) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def _normalized_factors(item_factors: np.ndarray) -> np.ndarray:
    factors = np.asarray(item_factors, dtype=np.float64)
    norms = np.linalg.norm(factors, axis=1)
    normalized = np.zeros_like(factors, dtype=np.float64)
    nonzero = norms > 0.0
    normalized[nonzero] = factors[nonzero] / norms[nonzero, None]
    return normalized


def decompose_pmf_score(
    pmf: PMFModel,
    user_index: int,
    item_index: int,
) -> dict[str, Any]:
    pmf._check_fitted()
    if not 0 <= int(user_index) < pmf.n_users:
        raise ValueError("user index out of range")
    if not 0 <= int(item_index) < pmf.n_items:
        raise ValueError("item index out of range")
    contributions = (
        pmf.user_factors[int(user_index)].astype(np.float64)
        * pmf.item_factors[int(item_index)].astype(np.float64)
    )
    latent_dot = float(np.sum(contributions))
    raw_score = float(
        pmf.global_mean
        + float(pmf.user_bias[int(user_index)])
        + float(pmf.item_bias[int(item_index)])
        + latent_dot
    )
    order = np.lexsort((np.arange(contributions.size), -np.abs(contributions)))[:3]
    top_factors = [
        {"factor_index": int(index), "contribution": float(contributions[index])}
        for index in order
    ]
    return {
        "raw_score": raw_score,
        "global_mean": float(pmf.global_mean),
        "user_bias": float(pmf.user_bias[int(user_index)]),
        "item_bias": float(pmf.item_bias[int(item_index)]),
        "latent_dot": latent_dot,
        "factor_contributions": contributions,
        "top_factors": top_factors,
    }


def build_pmf_factor_interpretation(
    item_factors: np.ndarray,
    index_to_movie: np.ndarray,
    movies: pd.DataFrame,
    n_factors: int = 5,
    top_n: int = 8,
) -> pd.DataFrame:
    factors = np.asarray(item_factors, dtype=np.float64)
    movie_ids = np.asarray(index_to_movie, dtype=np.int32)
    if factors.ndim != 2 or factors.shape[0] != movie_ids.size:
        raise ValueError("item factors and movie mapping are not aligned")
    variances = np.var(factors, axis=0)
    selected = np.lexsort((np.arange(variances.size), -variances))[:n_factors]
    lookup = _movie_lookup(movies)

    rows: list[dict[str, Any]] = []
    for factor_index in selected:
        loadings = factors[:, factor_index]
        positive_order = np.lexsort((movie_ids, -loadings))[:top_n]
        negative_order = np.lexsort((movie_ids, loadings))[:top_n]
        for polarity, ordered in (
            ("positive", positive_order),
            ("negative", negative_order),
        ):
            for rank, item_index in enumerate(ordered, start=1):
                movie_id = int(movie_ids[item_index])
                movie = lookup.loc[movie_id]
                rows.append(
                    {
                        "factor_index": int(factor_index),
                        "factor_variance": float(variances[factor_index]),
                        "polarity": polarity,
                        "polarity_rank": int(rank),
                        "movie_id": movie_id,
                        "title": str(movie["title"]),
                        "genres": str(movie["genres"]),
                        "factor_loading": float(loadings[item_index]),
                    }
                )
    return pd.DataFrame(rows, columns=FACTOR_INTERPRETATION_COLUMNS)


def build_pmf_factor_genre_profiles(
    factor_interpretation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (factor_index, polarity), group in factor_interpretation.groupby(
        ["factor_index", "polarity"], sort=True
    ):
        genre_rows: list[tuple[str, float]] = []
        for record in group.itertuples(index=False):
            for genre in _split_genres(record.genres):
                genre_rows.append((genre, float(record.factor_loading)))
        if not genre_rows:
            continue
        genre_frame = pd.DataFrame(genre_rows, columns=["genre", "factor_loading"])
        counts = genre_frame.groupby("genre", sort=True).agg(
            movie_count=("genre", "size"),
            mean_factor_loading=("factor_loading", "mean"),
        )
        total = float(counts["movie_count"].sum())
        for genre, row in counts.reset_index().sort_values(
            ["movie_count", "genre"], ascending=[False, True], kind="mergesort"
        ).iterrows():
            rows.append(
                {
                    "factor_index": int(factor_index),
                    "polarity": str(polarity),
                    "genre": str(row["genre"]),
                    "movie_count": int(row["movie_count"]),
                    "genre_share": float(row["movie_count"] / total),
                    "mean_factor_loading": float(row["mean_factor_loading"]),
                }
            )
    return pd.DataFrame(rows, columns=FACTOR_GENRE_PROFILE_COLUMNS)


def plot_pmf_latent_factor_heatmap(
    factor_interpretation: pd.DataFrame,
    item_factors: np.ndarray,
    movie_to_index: dict[int, int],
    path: str | Path,
    max_rank_per_polarity: int = 3,
) -> None:
    selected = factor_interpretation.loc[
        factor_interpretation["polarity_rank"].le(max_rank_per_polarity)
    ].copy()
    selected = selected.drop_duplicates("movie_id", keep="first")
    factor_ids = sorted(factor_interpretation["factor_index"].astype(int).unique())
    movie_ids = selected["movie_id"].astype(int).tolist()
    indices = [movie_to_index[movie_id] for movie_id in movie_ids]
    values = np.asarray(item_factors, dtype=np.float64)[indices][:, factor_ids]
    labels = [
        f"{title[:42]} ({movie_id})"
        for title, movie_id in zip(selected["title"].astype(str), movie_ids, strict=True)
    ]

    height = max(6.0, 0.34 * len(labels))
    fig, ax = plt.subplots(figsize=(9, height))
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(factor_ids)), [f"F{factor_id}" for factor_id in factor_ids])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("PMF factor ID")
    ax.set_ylabel("Representative movies")
    ax.set_title("PMF latent factor loadings for high-variance factors")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Item-factor loading")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def select_anchor_movies(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    movie_to_index: dict[int, int],
    n_anchors: int = 5,
) -> list[int]:
    mapped = set(movie_to_index)
    popularity = (
        ratings.loc[ratings["movie_id"].isin(mapped)]
        .groupby("movie_id")
        .size()
        .rename("rating_count")
        .reset_index()
    )
    candidates = movies.merge(popularity, on="movie_id", how="inner").sort_values(
        ["rating_count", "movie_id"], ascending=[False, True], kind="mergesort"
    )
    selected: list[int] = []
    selected_genres: list[set[str]] = []
    for row in candidates.itertuples(index=False):
        genres = set(_split_genres(row.genres))
        if selected_genres:
            overlaps = [
                len(genres & existing) / max(1, len(genres | existing))
                for existing in selected_genres
            ]
            if max(overlaps) > 0.55:
                continue
        selected.append(int(row.movie_id))
        selected_genres.append(genres)
        if len(selected) == n_anchors:
            return selected
    for row in candidates.itertuples(index=False):
        movie_id = int(row.movie_id)
        if movie_id not in selected:
            selected.append(movie_id)
        if len(selected) == n_anchors:
            return selected
    return selected


def build_pmf_movie_similarities(
    item_factors: np.ndarray,
    index_to_movie: np.ndarray,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    movie_to_index: dict[int, int],
    n_anchors: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
    movie_ids = np.asarray(index_to_movie, dtype=np.int32)
    lookup = _movie_lookup(movies)
    normalized = _normalized_factors(item_factors)
    anchors = select_anchor_movies(ratings, movies, movie_to_index, n_anchors=n_anchors)
    rows: list[dict[str, Any]] = []
    for anchor_movie_id in anchors:
        anchor_index = int(movie_to_index[anchor_movie_id])
        similarities = normalized @ normalized[anchor_index]
        frame = pd.DataFrame(
            {
                "movie_id": movie_ids,
                "cosine_similarity": np.clip(similarities, -1.0, 1.0),
            }
        )
        frame = frame.loc[frame["movie_id"].ne(anchor_movie_id)]
        frame = frame.sort_values(
            ["cosine_similarity", "movie_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(top_n)
        anchor = lookup.loc[anchor_movie_id]
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            similar = lookup.loc[int(row.movie_id)]
            rows.append(
                {
                    "anchor_movie_id": int(anchor_movie_id),
                    "anchor_title": str(anchor["title"]),
                    "anchor_genres": str(anchor["genres"]),
                    "similar_movie_id": int(row.movie_id),
                    "similar_title": str(similar["title"]),
                    "similar_genres": str(similar["genres"]),
                    "cosine_similarity": float(row.cosine_similarity),
                    "rank": int(rank),
                }
            )
    return pd.DataFrame(rows, columns=SIMILARITY_COLUMNS)


def _per_user_rmse(frame: pd.DataFrame, prediction_column: str) -> pd.Series:
    squared = (frame["rating"].astype(float) - frame[prediction_column].astype(float)) ** 2
    return squared.groupby(frame["user_id"]).mean().pow(0.5)


def select_audit_users(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test_with_predictions: pd.DataFrame,
    min_train_ratings: int = 50,
    min_test_ratings: int = 10,
) -> list[dict[str, Any]]:
    required = {"user_id", "rating", "svd_prediction", "pmf_prediction"}
    missing = required - set(test_with_predictions.columns)
    if missing:
        raise ValueError(f"test predictions missing columns: {sorted(missing)}")

    train_counts = train.groupby("user_id").size().rename("train_ratings")
    validation_counts = validation.groupby("user_id").size().rename("validation_ratings")
    test_counts = test_with_predictions.groupby("user_id").size().rename("test_ratings")
    profile = pd.concat([train_counts, validation_counts, test_counts], axis=1).fillna(0)
    profile = profile.astype(int)
    profile["svd_test_rmse"] = _per_user_rmse(test_with_predictions, "svd_prediction")
    profile["pmf_test_rmse"] = _per_user_rmse(test_with_predictions, "pmf_prediction")
    profile = profile.dropna(subset=["svd_test_rmse", "pmf_test_rmse"]).reset_index()

    eligible = profile.loc[
        profile["train_ratings"].ge(min_train_ratings)
        & profile["test_ratings"].ge(min_test_ratings)
    ].copy()
    if len(eligible) < 3:
        raise ValueError("not enough users satisfy audit support thresholds")

    lower_target = float(eligible["pmf_test_rmse"].quantile(0.25))
    upper_target = float(eligible["pmf_test_rmse"].quantile(0.75))
    accurate_candidates = eligible.assign(
        distance=(eligible["pmf_test_rmse"] - lower_target).abs()
    ).sort_values(["distance", "pmf_test_rmse", "user_id"], kind="mergesort")
    accurate = accurate_candidates.iloc[0]

    less_candidates = eligible.loc[eligible["user_id"].ne(accurate["user_id"])].assign(
        distance=(eligible.loc[eligible["user_id"].ne(accurate["user_id"]), "pmf_test_rmse"] - upper_target).abs()
    )
    less_candidates = less_candidates.loc[
        less_candidates["pmf_test_rmse"].gt(accurate["pmf_test_rmse"])
    ].sort_values(["distance", "pmf_test_rmse", "user_id"], kind="mergesort")
    if less_candidates.empty:
        raise ValueError("could not select a less accurate user above the accurate RMSE")
    less_accurate = less_candidates.iloc[0]

    median_test_count = float(eligible["test_ratings"].median())
    used = {int(accurate["user_id"]), int(less_accurate["user_id"])}
    test_case = (
        eligible.loc[~eligible["user_id"].isin(used)]
        .assign(distance=(eligible.loc[~eligible["user_id"].isin(used), "test_ratings"] - median_test_count).abs())
        .sort_values(["distance", "test_ratings", "user_id"], kind="mergesort")
        .iloc[0]
    )

    selections = [
        (
            accurate,
            "train_profile_accurate",
            f"Eligible user nearest the lower-quartile PMF test RMSE ({lower_target:.4f}) with at least {min_train_ratings} train and {min_test_ratings} test ratings.",
        ),
        (
            less_accurate,
            "train_profile_less_accurate",
            f"Eligible user nearest the upper-quartile PMF test RMSE ({upper_target:.4f}) and worse than the accurate profile.",
        ),
        (
            test_case,
            "test_case",
            f"Separate deterministic test case near the median eligible test support ({median_test_count:.1f}) and distinct from the two training-profile users.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for row, role, reason in selections:
        rows.append(
            {
                "user_id": int(row["user_id"]),
                "role": role,
                "selection_reason": reason,
                "train_ratings": int(row["train_ratings"]),
                "validation_ratings": int(row["validation_ratings"]),
                "test_ratings": int(row["test_ratings"]),
                "svd_test_rmse": float(row["svd_test_rmse"]),
                "pmf_test_rmse": float(row["pmf_test_rmse"]),
            }
        )
    return rows


def nearest_known_liked_movie(
    user_id: int,
    recommended_movie_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    movie_to_index: dict[int, int],
    item_factors: np.ndarray,
    high_rating_threshold: float = 4.0,
) -> dict[str, Any]:
    history = ratings.loc[
        ratings["user_id"].eq(user_id)
        & ratings["movie_id"].isin(movie_to_index)
        & ratings["movie_id"].ne(recommended_movie_id)
    ].copy()
    if history.empty:
        return {
            "movie_id": np.nan,
            "title": "",
            "genres": "",
            "rating": np.nan,
            "similarity": np.nan,
            "common_genres": "",
        }
    liked = history.loc[history["rating"].ge(high_rating_threshold)].copy()
    if liked.empty:
        max_rating = history["rating"].max()
        liked = history.loc[history["rating"].eq(max_rating)].copy()

    rec_index = movie_to_index[int(recommended_movie_id)]
    rec_vector = np.asarray(item_factors[rec_index], dtype=np.float64)
    liked["similarity"] = [
        cosine_similarity(rec_vector, item_factors[movie_to_index[int(movie_id)]])
        for movie_id in liked["movie_id"].astype(int)
    ]
    selected = liked.sort_values(
        ["similarity", "rating", "movie_id"],
        ascending=[False, False, True],
        kind="mergesort",
    ).iloc[0]
    lookup = _movie_lookup(movies)
    known = lookup.loc[int(selected["movie_id"])]
    recommended = lookup.loc[int(recommended_movie_id)]
    common = sorted(set(_split_genres(known["genres"])) & set(_split_genres(recommended["genres"])))
    return {
        "movie_id": int(selected["movie_id"]),
        "title": str(known["title"]),
        "genres": str(known["genres"]),
        "rating": float(selected["rating"]),
        "similarity": float(selected["similarity"]),
        "common_genres": "|".join(common),
    }


def build_local_pmf_explanations(
    user_id: int,
    role: str,
    recommendations: pd.DataFrame,
    pmf: PMFModel,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    if role not in AUDIT_USER_ROLES:
        raise ValueError(f"unknown audit role: {role}")
    user_index = user_to_index[int(user_id)]
    pmf_rows = recommendations.dropna(subset=["pmf_rank"]).copy()
    pmf_rows = pmf_rows.sort_values(["pmf_rank", "movie_id"], kind="mergesort")
    rows: list[dict[str, Any]] = []
    for record in pmf_rows.itertuples(index=False):
        movie_id = int(record.movie_id)
        item_index = movie_to_index[movie_id]
        parts = decompose_pmf_score(pmf, user_index, item_index)
        top = parts["top_factors"]
        while len(top) < 3:
            top.append({"factor_index": -1, "contribution": 0.0})
        nearest = nearest_known_liked_movie(
            int(user_id),
            movie_id,
            ratings,
            movies,
            movie_to_index,
            pmf.item_factors,
        )
        component_sum = float(
            parts["global_mean"]
            + parts["user_bias"]
            + parts["item_bias"]
            + parts["latent_dot"]
        )
        rows.append(
            {
                "user_id": int(user_id),
                "role": role,
                "recommendation_rank": int(record.pmf_rank),
                "movie_id": movie_id,
                "title": str(record.title),
                "genres": str(record.genres),
                "raw_pmf_ranking_score": float(parts["raw_score"]),
                "clipped_displayed_rating": float(np.clip(parts["raw_score"], 1.0, 5.0)),
                "global_mean_contribution": float(parts["global_mean"]),
                "user_bias_contribution": float(parts["user_bias"]),
                "item_bias_contribution": float(parts["item_bias"]),
                "total_latent_dot_product": float(parts["latent_dot"]),
                "top_factor_1_index": int(top[0]["factor_index"]),
                "top_factor_1_contribution": float(top[0]["contribution"]),
                "top_factor_2_index": int(top[1]["factor_index"]),
                "top_factor_2_contribution": float(top[1]["contribution"]),
                "top_factor_3_index": int(top[2]["factor_index"]),
                "top_factor_3_contribution": float(top[2]["contribution"]),
                "top_factor_contributions": json.dumps(top, sort_keys=True),
                "component_sum": component_sum,
                "reconstruction_error": float(component_sum - parts["raw_score"]),
                "nearest_known_movie_id": nearest["movie_id"],
                "nearest_known_title": nearest["title"],
                "nearest_known_genres": nearest["genres"],
                "nearest_known_rating": nearest["rating"],
                "nearest_known_similarity": nearest["similarity"],
                "common_genres": nearest["common_genres"],
            }
        )
    return pd.DataFrame(rows, columns=LOCAL_EXPLANATION_COLUMNS)


def plot_user_explanation(
    explanations: pd.DataFrame,
    path: str | Path,
) -> None:
    if explanations.empty:
        raise ValueError("explanations must not be empty")
    user_id = int(explanations["user_id"].iloc[0])
    top = explanations.sort_values("recommendation_rank", kind="mergesort").head(5)
    labels = [
        f"#{int(row.recommendation_rank)} {str(row.title)[:35]}\nliked: {str(row.nearest_known_title)[:32]} ({row.nearest_known_similarity:.2f})"
        for row in top.itertuples(index=False)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].barh(np.arange(len(top)), top["raw_pmf_ranking_score"], color="#4c78a8")
    axes[0].set_yticks(np.arange(len(top)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Raw PMF ranking score")
    axes[0].set_title(f"Why PMF recommends these movies for user {user_id}")
    axes[0].grid(axis="x", alpha=0.25)

    long_rows: list[dict[str, Any]] = []
    for row in top.itertuples(index=False):
        for position in (1, 2, 3):
            long_rows.append(
                {
                    "label": f"#{int(row.recommendation_rank)} F{getattr(row, f'top_factor_{position}_index')}",
                    "contribution": float(
                        getattr(row, f"top_factor_{position}_contribution")
                    ),
                }
            )
    contrib = pd.DataFrame(long_rows).iloc[::-1]
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in contrib["contribution"]]
    axes[1].barh(np.arange(len(contrib)), contrib["contribution"], color=colors)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_yticks(np.arange(len(contrib)), contrib["label"])
    axes[1].set_xlabel("Latent factor contribution")
    axes[1].set_title("Largest positive/negative PMF factor terms")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
