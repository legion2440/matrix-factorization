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


EVALUATION_USER_ROLES = (
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

RANKING_CASE_COLUMNS = [
    "user_id",
    "role",
    "ranking_case",
    "target_movie_id",
    "target_title",
    "target_genres",
    "target_rating",
    "target_timestamp",
    "prior_history_count",
    "candidate_count",
    "bias_target_rank",
    "item_knn_target_rank",
    "svd_target_rank",
    "pmf_target_rank",
    "bias_raw_target_score",
    "item_knn_raw_target_score",
    "svd_raw_target_score",
    "pmf_raw_target_score",
    "bias_hit_at_5",
    "bias_hit_at_10",
    "item_knn_hit_at_5",
    "item_knn_hit_at_10",
    "svd_hit_at_5",
    "svd_hit_at_10",
    "pmf_hit_at_5",
    "pmf_hit_at_10",
    "pmf_global_mean_contribution",
    "pmf_user_bias_contribution",
    "pmf_item_bias_contribution",
    "pmf_total_latent_dot_product",
    "pmf_component_sum",
    "pmf_reconstruction_error",
    "top_factor_1_index",
    "top_factor_1_contribution",
    "top_factor_2_index",
    "top_factor_2_contribution",
    "top_factor_3_index",
    "top_factor_3_contribution",
    "top_factor_contributions",
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


def select_evaluation_users(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test_with_predictions: pd.DataFrame,
    ranking_results: pd.DataFrame,
    min_train_ratings: int = 1,
    min_test_ratings: int = 1,
) -> list[dict[str, Any]]:
    required = {"user_id", "rating", "svd_prediction", "pmf_prediction"}
    missing = required - set(test_with_predictions.columns)
    if missing:
        raise ValueError(f"test predictions missing columns: {sorted(missing)}")
    ranking_required = {
        "user_id",
        "target_movie_id",
        "target_title",
        "target_rating",
        "target_timestamp",
        "prior_history_count",
        "candidate_count",
        "bias_target_rank",
        "item_knn_target_rank",
        "svd_target_rank",
        "pmf_target_rank",
        "bias_hit_at_10",
        "item_knn_hit_at_10",
        "svd_hit_at_10",
        "pmf_hit_at_10",
    }
    ranking_missing = ranking_required - set(ranking_results.columns)
    if ranking_missing:
        raise ValueError(
            f"ranking results missing columns: {sorted(ranking_missing)}"
        )

    train_counts = train.groupby("user_id").size().rename("train_ratings")
    validation_counts = validation.groupby("user_id").size().rename("validation_ratings")
    test_counts = test_with_predictions.groupby("user_id").size().rename("test_ratings")
    profile = pd.concat([train_counts, validation_counts, test_counts], axis=1).fillna(0)
    profile = profile.astype(int)
    profile["svd_test_rmse"] = _per_user_rmse(test_with_predictions, "svd_prediction")
    profile["pmf_test_rmse"] = _per_user_rmse(test_with_predictions, "pmf_prediction")
    profile = profile.dropna(subset=["svd_test_rmse", "pmf_test_rmse"]).reset_index()
    eligible = ranking_results.merge(profile, on="user_id", how="inner")
    eligible = eligible.loc[
        eligible["train_ratings"].ge(min_train_ratings)
        & eligible["test_ratings"].ge(min_test_ratings)
    ].copy()
    if len(eligible) < 3:
        raise ValueError("not enough users satisfy minimum profile support thresholds")

    accurate_pool = eligible.loc[eligible["pmf_target_rank"].le(10)].copy()
    if accurate_pool.empty:
        raise ValueError("ranking results contain no PMF Hit@10 users")
    non_extreme_hits = accurate_pool.loc[accurate_pool["pmf_target_rank"].gt(1)]
    if not non_extreme_hits.empty:
        accurate_pool = non_extreme_hits
    accurate_target = float(accurate_pool["pmf_target_rank"].median())
    accurate = (
        accurate_pool.assign(
            distance=(accurate_pool["pmf_target_rank"] - accurate_target).abs()
        )
        .sort_values(
            ["distance", "prior_history_count", "user_id"], kind="mergesort"
        )
        .iloc[0]
    )

    less_pool = eligible.loc[eligible["pmf_target_rank"].gt(10)].copy()
    if less_pool.empty:
        raise ValueError("ranking results contain no PMF Miss@10 users")
    less_target = float(less_pool["pmf_target_rank"].median())
    less_accurate = (
        less_pool.assign(
            distance=(less_pool["pmf_target_rank"] - less_target).abs()
        )
        .sort_values(
            ["distance", "prior_history_count", "user_id"], kind="mergesort"
        )
        .iloc[0]
    )

    overall_target = float(eligible["pmf_target_rank"].median())
    used = {int(accurate["user_id"]), int(less_accurate["user_id"])}
    representative_pool = eligible.loc[~eligible["user_id"].isin(used)].copy()
    if representative_pool.empty:
        raise ValueError("could not select a distinct representative ranking case")
    test_case = (
        representative_pool.assign(
            distance=(
                representative_pool["pmf_target_rank"] - overall_target
            ).abs()
        )
        .sort_values(
            ["distance", "prior_history_count", "user_id"], kind="mergesort"
        )
        .iloc[0]
    )

    selections = [
        (
            accurate,
            "train_profile_accurate",
            "pmf_hit_at_10",
            (
                "Supported PMF Hit@10 user nearest the median non-extreme PMF "
                f"hit rank ({accurate_target:.1f})."
            ),
        ),
        (
            less_accurate,
            "train_profile_less_accurate",
            "pmf_miss_at_10",
            (
                "Supported PMF miss user nearest the median PMF miss rank "
                f"({less_target:.1f})."
            ),
        ),
        (
            test_case,
            "test_case",
            "representative_target_rank",
            (
                "Distinct supported user nearest the overall median PMF target "
                f"rank ({overall_target:.1f})."
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for row, role, ranking_case, reason in selections:
        rows.append(
            {
                "user_id": int(row["user_id"]),
                "role": role,
                "ranking_case": ranking_case,
                "selection_reason": reason,
                "train_ratings": int(row["train_ratings"]),
                "validation_ratings": int(row["validation_ratings"]),
                "test_ratings": int(row["test_ratings"]),
                "svd_test_rmse": float(row["svd_test_rmse"]),
                "pmf_test_rmse": float(row["pmf_test_rmse"]),
                "ranking_target_movie_id": int(row["target_movie_id"]),
                "ranking_target_title": str(row["target_title"]),
                "ranking_target_rating": float(row["target_rating"]),
                "ranking_target_timestamp": int(row["target_timestamp"]),
                "ranking_history_count": int(row["prior_history_count"]),
                "ranking_candidate_count": int(row["candidate_count"]),
                "bias_target_rank": int(row["bias_target_rank"]),
                "item_knn_target_rank": int(row["item_knn_target_rank"]),
                "svd_target_rank": int(row["svd_target_rank"]),
                "pmf_target_rank": int(row["pmf_target_rank"]),
                "bias_hit_at_10": bool(row["bias_hit_at_10"]),
                "item_knn_hit_at_10": bool(row["item_knn_hit_at_10"]),
                "svd_hit_at_10": bool(row["svd_hit_at_10"]),
                "pmf_hit_at_10": bool(row["pmf_hit_at_10"]),
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
    if role not in EVALUATION_USER_ROLES:
        raise ValueError(f"unknown evaluation role: {role}")
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


def build_ranking_case_explanation(
    selection: dict[str, Any],
    ranking_row: pd.Series,
    ranking_train: pd.DataFrame,
    ranking_pmf: PMFModel,
    user_to_index: dict[int, int],
    movie_to_index: dict[int, int],
    movies: pd.DataFrame,
) -> pd.DataFrame:
    user_id = int(selection["user_id"])
    role = str(selection["role"])
    if role not in EVALUATION_USER_ROLES:
        raise ValueError(f"unknown evaluation role: {role}")
    ranking_user_id = (
        int(ranking_row["user_id"])
        if "user_id" in ranking_row.index
        else int(ranking_row.name)
    )
    if ranking_user_id != user_id:
        raise ValueError("selection and ranking row refer to different users")
    target_movie_id = int(ranking_row["target_movie_id"])
    user_index = user_to_index[user_id]
    item_index = movie_to_index[target_movie_id]
    parts = decompose_pmf_score(ranking_pmf, user_index, item_index)
    top = list(parts["top_factors"])
    while len(top) < 3:
        top.append({"factor_index": -1, "contribution": 0.0})
    nearest = nearest_known_liked_movie(
        user_id,
        target_movie_id,
        ranking_train,
        movies,
        movie_to_index,
        ranking_pmf.item_factors,
    )
    component_sum = float(
        parts["global_mean"]
        + parts["user_bias"]
        + parts["item_bias"]
        + parts["latent_dot"]
    )
    row: dict[str, Any] = {
        "user_id": user_id,
        "role": role,
        "ranking_case": str(selection["ranking_case"]),
        "target_movie_id": target_movie_id,
        "target_title": str(ranking_row["target_title"]),
        "target_genres": str(ranking_row["target_genres"]),
        "target_rating": float(ranking_row["target_rating"]),
        "target_timestamp": int(ranking_row["target_timestamp"]),
        "prior_history_count": int(ranking_row["prior_history_count"]),
        "candidate_count": int(ranking_row["candidate_count"]),
        "pmf_global_mean_contribution": float(parts["global_mean"]),
        "pmf_user_bias_contribution": float(parts["user_bias"]),
        "pmf_item_bias_contribution": float(parts["item_bias"]),
        "pmf_total_latent_dot_product": float(parts["latent_dot"]),
        "pmf_component_sum": component_sum,
        "pmf_reconstruction_error": float(
            component_sum - float(ranking_row["pmf_raw_target_score"])
        ),
        "top_factor_1_index": int(top[0]["factor_index"]),
        "top_factor_1_contribution": float(top[0]["contribution"]),
        "top_factor_2_index": int(top[1]["factor_index"]),
        "top_factor_2_contribution": float(top[1]["contribution"]),
        "top_factor_3_index": int(top[2]["factor_index"]),
        "top_factor_3_contribution": float(top[2]["contribution"]),
        "top_factor_contributions": json.dumps(top, sort_keys=True),
        "nearest_known_movie_id": nearest["movie_id"],
        "nearest_known_title": nearest["title"],
        "nearest_known_genres": nearest["genres"],
        "nearest_known_rating": nearest["rating"],
        "nearest_known_similarity": nearest["similarity"],
        "common_genres": nearest["common_genres"],
    }
    for prefix in ("bias", "item_knn", "svd", "pmf"):
        row[f"{prefix}_target_rank"] = int(ranking_row[f"{prefix}_target_rank"])
        row[f"{prefix}_raw_target_score"] = float(
            ranking_row[f"{prefix}_raw_target_score"]
        )
        for cutoff in (5, 10):
            row[f"{prefix}_hit_at_{cutoff}"] = bool(
                ranking_row[f"{prefix}_hit_at_{cutoff}"]
            )
    return pd.DataFrame([row], columns=RANKING_CASE_COLUMNS)


def _build_ranking_case_figure(ranking_case: pd.DataFrame) -> plt.Figure:
    if len(ranking_case) != 1:
        raise ValueError("ranking_case must contain exactly one row")
    row = ranking_case.iloc[0]
    labels = ["Bias baseline", "Item-kNN", "SVD", "PMF"]
    ranks = [
        int(row["bias_target_rank"]),
        int(row["item_knn_target_rank"]),
        int(row["svd_target_rank"]),
        int(row["pmf_target_rank"]),
    ]
    adjustments = [
        float(row["pmf_user_bias_contribution"]),
        float(row["pmf_item_bias_contribution"]),
        float(row["pmf_total_latent_dot_product"]),
    ]
    adjustment_labels = ["User bias", "Item bias", "Latent dot product"]
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#54a24b"]
    positions = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    rank_axis = axes[0]
    rank_axis.axvspan(1, 10, color="#54a24b", alpha=0.14, label="Top-10")
    rank_axis.hlines(
        positions,
        1,
        ranks,
        color=colors,
        linewidth=2.5,
        alpha=0.8,
    )
    rank_axis.scatter(ranks, positions, color=colors, s=85, zorder=3)
    rank_axis.set_xscale("log")
    rank_axis.set_xlim(left=1, right=max(ranks) * 1.55)
    rank_axis.set_yticks(positions, labels)
    rank_axis.invert_yaxis()
    rank_axis.set_xlabel("Held-out target rank (log scale; lower is better)")
    rank_axis.set_title(
        f"User {int(row['user_id'])}: {str(row['target_title'])[:42]}"
    )
    rank_axis.grid(axis="x", which="both", alpha=0.28)
    for position, rank in zip(positions, ranks, strict=True):
        rank_axis.annotate(
            f"{rank:,}",
            (rank, position),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    rank_axis.text(
        np.sqrt(10),
        -0.48,
        "Top-10",
        ha="center",
        va="center",
        color="#2f6b3c",
        fontsize=9,
    )

    decomposition_axis = axes[1]
    adjustment_colors = [
        "#54a24b" if value >= 0 else "#e45756" for value in adjustments
    ]
    adjustment_positions = np.arange(len(adjustment_labels))
    bars = decomposition_axis.barh(
        adjustment_positions,
        adjustments,
        color=adjustment_colors,
        alpha=0.9,
    )
    decomposition_axis.set_yticks(adjustment_positions, adjustment_labels)
    decomposition_axis.invert_yaxis()
    decomposition_axis.axvline(0.0, color="black", linewidth=1.0)
    decomposition_axis.set_xlabel("Adjustment to the global-mean baseline")
    decomposition_axis.set_title(
        "PMF target decomposition\n"
        "raw PMF score = global mean + adjustments"
    )
    decomposition_axis.grid(axis="x", alpha=0.25)

    max_adjustment = max(abs(value) for value in adjustments)
    limit = max(max_adjustment * 1.55, 0.1)
    decomposition_axis.set_xlim(-limit, limit)
    for bar, value in zip(bars, adjustments, strict=True):
        offset = 5 if value >= 0 else -5
        decomposition_axis.annotate(
            f"{value:+.4f}",
            (value, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0),
            textcoords="offset points",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )

    global_mean = float(row["pmf_global_mean_contribution"])
    raw_score = float(row["pmf_raw_target_score"])
    component_sum = float(row["pmf_component_sum"])
    decomposition_axis.text(
        0.98,
        0.97,
        f"Global mean baseline: {global_mean:.4f}",
        transform=decomposition_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "#777777",
            "alpha": 0.9,
        },
    )
    decomposition_axis.text(
        0.02,
        -0.12,
        f"Final raw PMF score: {raw_score:.4f}",
        transform=decomposition_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    decomposition_axis.text(
        0.98,
        -0.12,
        f"Reconstructed component sum: {component_sum:.4f}",
        transform=decomposition_axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()
    return fig


def plot_ranking_case(
    ranking_case: pd.DataFrame,
    path: str | Path,
) -> None:
    fig = _build_ranking_case_figure(ranking_case)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
