from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from models.pmf_model import PMFModel
from utils.data_loader import load_movielens
from utils.matrix_creation import load_mappings
from utils.recommendation import (
    PMFRecommendationModel,
    SVDRecommendationModel,
    compare_recommendations,
    generate_recommendations,
)


ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_application_resources() -> dict[str, object]:
    required = [
        ROOT / "reports" / "svd_predictions.npy",
        ROOT / "reports" / "pmf_factors" / "metadata.json",
        ROOT / "reports" / "model_metrics.json",
        ROOT / "processed" / "mappings" / "user_to_index.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing generated artifacts. Run `python -m scripts.run_pipeline`. "
            f"Missing: {missing}"
        )

    data = load_movielens(ROOT / "data")
    user_to_index, movie_to_index, index_to_user, index_to_movie = load_mappings(
        ROOT / "processed" / "mappings"
    )
    movies = data.movies.loc[data.movies["movie_id"].isin(index_to_movie)].copy()
    svd_predictions = np.load(ROOT / "reports" / "svd_predictions.npy", mmap_mode="r")
    pmf = PMFModel.load(ROOT / "reports" / "pmf_factors")
    with (ROOT / "reports" / "model_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    return {
        "ratings": data.ratings,
        "movies": movies,
        "users": data.users,
        "user_to_index": user_to_index,
        "movie_to_index": movie_to_index,
        "index_to_user": index_to_user,
        "index_to_movie": index_to_movie,
        "svd": SVDRecommendationModel(
            svd_predictions,
            user_to_index,
            movie_to_index,
            index_to_movie,
            movies,
            data.ratings,
        ),
        "pmf": PMFRecommendationModel(
            pmf,
            user_to_index,
            movie_to_index,
            index_to_movie,
            movies,
            data.ratings,
        ),
        "metrics": metrics,
    }


def main() -> None:
    st.set_page_config(page_title="MovieLens Matrix Factorization", layout="wide")
    st.title("MovieLens 1M recommender")
    st.caption("Truncated SVD versus locally implemented biased PMF")

    try:
        resources = load_application_resources()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    metrics = resources["metrics"]
    metric_columns = st.columns(3)
    metric_columns[0].metric("SVD test RMSE", f"{metrics['SVD_RMSE']:.4f}")
    metric_columns[1].metric("PMF test RMSE", f"{metrics['PMF_RMSE']:.4f}")
    metric_columns[2].metric(
        "PMF improvement", f"{metrics['PMF_vs_SVD_improvement_%']:.2f}%"
    )

    user_ids = resources["index_to_user"].astype(int).tolist()
    controls = st.columns([2, 1])
    user_id = controls[0].selectbox("User ID", user_ids, index=0)
    top_n = controls[1].slider("Recommendations", min_value=5, max_value=25, value=10)
    if user_id not in resources["user_to_index"]:
        st.error(f"Unknown user ID: {user_id}")
        return

    ratings = resources["ratings"]
    movies = resources["movies"]
    history = (
        ratings.loc[ratings["user_id"].eq(user_id)]
        .merge(movies, on="movie_id", how="left")
        .sort_values(["rating", "timestamp", "movie_id"], ascending=[False, False, True])
        .head(10)[["movie_id", "title", "genres", "rating"]]
    )
    st.subheader("Highest-rated known films")
    st.dataframe(history, hide_index=True, use_container_width=True)

    svd_recommendations = generate_recommendations(
        user_id, resources["svd"], top_n=top_n
    )
    pmf_recommendations = generate_recommendations(
        user_id, resources["pmf"], top_n=top_n
    )
    left, right = st.columns(2)
    with left:
        st.subheader("SVD recommendations")
        st.dataframe(
            svd_recommendations.style.format(
                {
                    "ranking_score": "{:.3f}",
                    "predicted_rating": "{:.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.subheader("PMF recommendations")
        st.dataframe(
            pmf_recommendations.style.format(
                {
                    "ranking_score": "{:.3f}",
                    "predicted_rating": "{:.3f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    comparison = compare_recommendations(
        user_id, resources["svd"], resources["pmf"], top_n=top_n
    )
    st.subheader("Combined ranking")
    st.dataframe(
        comparison.style.format(
            {
                "svd_ranking_score": "{:.3f}",
                "svd_predicted_rating": "{:.3f}",
                "pmf_ranking_score": "{:.3f}",
                "pmf_predicted_rating": "{:.3f}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    chart_frames = []
    for prefix, label in (("svd", "SVD"), ("pmf", "PMF")):
        model_points = comparison[
            ["movie_id", "title", f"{prefix}_ranking_score", f"{prefix}_rank"]
        ].dropna(subset=[f"{prefix}_ranking_score"])
        model_points = model_points.rename(
            columns={
                f"{prefix}_ranking_score": "ranking_score",
                f"{prefix}_rank": "rank",
            }
        )
        model_points["model"] = label
        chart_frames.append(model_points)
    chart_data = pd.concat(chart_frames, ignore_index=True)
    chart_data["movie"] = (
        chart_data["title"].str.slice(0, 48)
        + " ("
        + chart_data["movie_id"].astype(int).astype(str)
        + ")"
    )
    st.subheader("Raw ranking score comparison")
    st.scatter_chart(
        chart_data,
        x="ranking_score",
        y="movie",
        color="model",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
