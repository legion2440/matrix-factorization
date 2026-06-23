from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from models.pmf_model import PMFModel


class RecommendationModel(Protocol):
    name: str
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: np.ndarray
    movies: pd.DataFrame
    known_ratings: pd.DataFrame

    def predict_all_for_user(self, user_id: int) -> np.ndarray:
        """Return raw, unclipped scores used only for ranking."""
        ...


@dataclass
class SVDRecommendationModel:
    predictions: np.ndarray
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: np.ndarray
    movies: pd.DataFrame
    known_ratings: pd.DataFrame
    name: str = "SVD"

    def predict_all_for_user(self, user_id: int) -> np.ndarray:
        return self.predictions[self.user_to_index[user_id]]


@dataclass
class PMFRecommendationModel:
    pmf: PMFModel
    user_to_index: dict[int, int]
    movie_to_index: dict[int, int]
    index_to_movie: np.ndarray
    movies: pd.DataFrame
    known_ratings: pd.DataFrame
    name: str = "PMF"

    def predict_all_for_user(self, user_id: int) -> np.ndarray:
        return self.pmf.predict_user(self.user_to_index[user_id], clip=False)


def generate_recommendations(
    user_id: int,
    model: RecommendationModel,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return deterministic top-N unseen movies for a known user."""
    if isinstance(top_n, bool) or not isinstance(top_n, (int, np.integer)) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    user_id = int(user_id)
    if user_id not in model.user_to_index:
        raise ValueError(f"Unknown user_id: {user_id}")

    ranking_scores = np.asarray(
        model.predict_all_for_user(user_id), dtype=np.float32
    )
    if ranking_scores.shape != (len(model.index_to_movie),):
        raise ValueError("Model returned an invalid score vector")
    if not np.isfinite(ranking_scores).all():
        raise ValueError("Model returned non-finite recommendation scores")
    seen = set(
        model.known_ratings.loc[
            model.known_ratings["user_id"].eq(user_id), "movie_id"
        ].astype(int)
    )
    candidates = pd.DataFrame(
        {
            "movie_id": model.index_to_movie.astype(np.int32),
            "ranking_score": ranking_scores,
            "predicted_rating": np.clip(ranking_scores, 1.0, 5.0),
        }
    )
    candidates = candidates.loc[~candidates["movie_id"].isin(seen)]
    candidates = candidates.merge(
        model.movies[["movie_id", "title", "genres"]],
        on="movie_id",
        how="left",
        validate="one_to_one",
    )
    candidates = candidates.sort_values(
        ["ranking_score", "movie_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    columns = [
        "movie_id",
        "title",
        "genres",
        "ranking_score",
        "predicted_rating",
    ]
    return candidates.head(int(top_n))[columns].reset_index(drop=True)


def compare_recommendations(
    user_id: int,
    svd_model: RecommendationModel,
    pmf_model: RecommendationModel,
    top_n: int = 10,
) -> pd.DataFrame:
    svd = generate_recommendations(user_id, svd_model, top_n).copy()
    pmf = generate_recommendations(user_id, pmf_model, top_n).copy()
    svd["svd_rank"] = np.arange(1, len(svd) + 1)
    pmf["pmf_rank"] = np.arange(1, len(pmf) + 1)
    svd = svd.rename(
        columns={
            "ranking_score": "svd_ranking_score",
            "predicted_rating": "svd_predicted_rating",
        }
    )
    pmf = pmf.rename(
        columns={
            "ranking_score": "pmf_ranking_score",
            "predicted_rating": "pmf_predicted_rating",
        }
    )
    comparison = svd.merge(
        pmf,
        on=["movie_id", "title", "genres"],
        how="outer",
        sort=False,
    )
    return comparison.sort_values(
        ["svd_rank", "pmf_rank", "movie_id"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
