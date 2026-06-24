from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        ROOT / "reports" / "ranking_metrics.json",
        ROOT / "reports" / "ranking_protocol.json",
        ROOT / "reports" / "evaluated_users.json",
        ROOT / "reports" / "pmf_factor_interpretation.csv",
        ROOT / "reports" / "pmf_factor_genre_profiles.csv",
        ROOT / "reports" / "pmf_movie_similarities.csv",
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
    with (ROOT / "reports" / "ranking_metrics.json").open(encoding="utf-8") as handle:
        ranking_metrics = json.load(handle)
    with (ROOT / "reports" / "ranking_protocol.json").open(encoding="utf-8") as handle:
        ranking_protocol = json.load(handle)
    with (ROOT / "reports" / "evaluated_users.json").open(encoding="utf-8") as handle:
        evaluated_users = json.load(handle)

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
        "ranking_metrics": ranking_metrics,
        "ranking_protocol": ranking_protocol,
        "evaluated_users": evaluated_users,
        "factor_interpretation": pd.read_csv(
            ROOT / "reports" / "pmf_factor_interpretation.csv"
        ),
        "factor_genre_profiles": pd.read_csv(
            ROOT / "reports" / "pmf_factor_genre_profiles.csv"
        ),
        "similarities": pd.read_csv(ROOT / "reports" / "pmf_movie_similarities.csv"),
    }


def _format_frame(frame: pd.DataFrame, formats: dict[str, str]) -> Any:
    present = {key: value for key, value in formats.items() if key in frame.columns}
    return frame.style.format(present) if present else frame


def _artifact_image(relative_path: str) -> None:
    path = ROOT / relative_path
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Missing artifact: {relative_path}")


def _load_user_explanations(user_id: int) -> pd.DataFrame | None:
    path = ROOT / "reports" / f"user_{user_id}_explanations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_user_ranking_case(user_id: int) -> pd.DataFrame | None:
    path = ROOT / "reports" / f"user_{user_id}_ranking_case.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _evaluation_user_label(record: dict[str, object]) -> str:
    return f"{record['role']} - user {record['user_id']}"


def _sync_evaluation_profile_user_id() -> None:
    selected = st.session_state.get("evaluation_profile")
    if isinstance(selected, dict) and "user_id" in selected:
        st.session_state["user_id_input"] = str(selected["user_id"])


def main() -> None:
    st.set_page_config(page_title="MovieLens Matrix Factorization", layout="wide")
    st.title("MovieLens 1M recommender")
    st.caption(
        "Bias baseline, residualized item-kNN, truncated SVD, and biased PMF"
    )

    try:
        resources = load_application_resources()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    metrics = resources["metrics"]
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Bias baseline RMSE", f"{metrics['BiasBaseline_RMSE']:.4f}"
    )
    metric_columns[1].metric(
        "Item-kNN RMSE", f"{metrics['ItemKNN_RMSE']:.4f}"
    )
    metric_columns[2].metric("SVD RMSE", f"{metrics['SVD_RMSE']:.4f}")
    metric_columns[3].metric("PMF RMSE", f"{metrics['PMF_RMSE']:.4f}")

    evaluated_users = resources["evaluated_users"]
    evaluation_user_ids = [int(row["user_id"]) for row in evaluated_users]
    if "evaluation_profile" not in st.session_state:
        st.session_state["evaluation_profile"] = evaluated_users[0]
    if "user_id_input" not in st.session_state:
        st.session_state["user_id_input"] = str(
            st.session_state["evaluation_profile"]["user_id"]
        )
    controls = st.columns([2, 2, 1])
    controls[0].selectbox(
        "Evaluation profile shortcut",
        evaluated_users,
        format_func=_evaluation_user_label,
        key="evaluation_profile",
        on_change=_sync_evaluation_profile_user_id,
    )
    manual_user_id = controls[1].text_input(
        "User ID input",
        key="user_id_input",
    )
    top_n = controls[2].slider("Top N", min_value=5, max_value=25, value=10)
    try:
        user_id = int(manual_user_id.strip())
    except ValueError:
        st.error(f"Invalid user ID input: {manual_user_id!r}")
        return
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
    svd_recommendations = generate_recommendations(
        user_id, resources["svd"], top_n=top_n
    )
    pmf_recommendations = generate_recommendations(
        user_id, resources["pmf"], top_n=top_n
    )
    comparison = compare_recommendations(
        user_id, resources["svd"], resources["pmf"], top_n=top_n
    )

    tabs = st.tabs(
        [
            "Recommendations",
            "Why recommended",
            "Model evaluation",
            "Global latent factors",
        ]
    )

    with tabs[0]:
        st.subheader("Highest-rated known films")
        st.dataframe(history, hide_index=True, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("SVD recommendations")
            st.dataframe(
                _format_frame(
                    svd_recommendations,
                    {"ranking_score": "{:.3f}", "predicted_rating": "{:.3f}"},
                ),
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.subheader("PMF recommendations")
            st.dataframe(
                _format_frame(
                    pmf_recommendations,
                    {"ranking_score": "{:.3f}", "predicted_rating": "{:.3f}"},
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Combined ranking")
        st.dataframe(
            _format_frame(
                comparison,
                {
                    "svd_ranking_score": "{:.3f}",
                    "svd_predicted_rating": "{:.3f}",
                    "pmf_ranking_score": "{:.3f}",
                    "pmf_predicted_rating": "{:.3f}",
                },
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

    with tabs[1]:
        explanations = _load_user_explanations(user_id)
        if explanations is None:
            st.info(
                "Saved local explanation artifacts are available for the three "
                "evaluation profiles: "
                + ", ".join(str(value) for value in evaluation_user_ids)
            )
        else:
            st.subheader("Local PMF explanation table")
            st.dataframe(
                _format_frame(
                    explanations,
                    {
                        "raw_pmf_ranking_score": "{:.4f}",
                        "clipped_displayed_rating": "{:.4f}",
                        "global_mean_contribution": "{:.4f}",
                        "user_bias_contribution": "{:.4f}",
                        "item_bias_contribution": "{:.4f}",
                        "total_latent_dot_product": "{:.4f}",
                        "nearest_known_similarity": "{:.4f}",
                    },
                ),
                hide_index=True,
                use_container_width=True,
            )

            nearest_columns = [
                "recommendation_rank",
                "title",
                "nearest_known_title",
                "nearest_known_rating",
                "nearest_known_similarity",
                "common_genres",
            ]
            st.subheader("Nearest known liked movie")
            st.dataframe(
                _format_frame(
                    explanations[nearest_columns],
                    {
                        "nearest_known_rating": "{:.1f}",
                        "nearest_known_similarity": "{:.3f}",
                    },
                ),
                hide_index=True,
                use_container_width=True,
            )

            factor_columns = [
                "recommendation_rank",
                "top_factor_1_contribution",
                "top_factor_2_contribution",
                "top_factor_3_contribution",
            ]
            st.subheader("Top latent factor contributions")
            st.bar_chart(
                explanations[factor_columns].set_index("recommendation_rank"),
                use_container_width=True,
            )
            _artifact_image(f"reports/user_{user_id}_explanation.png")

    with tabs[2]:
        st.subheader("Rating prediction")
        metric_table = pd.DataFrame(
            [
                {
                    "model": "Bias baseline",
                    "mse": metrics["BiasBaseline_MSE"],
                    "rmse": metrics["BiasBaseline_RMSE"],
                },
                {
                    "model": "Item-kNN",
                    "mse": metrics["ItemKNN_MSE"],
                    "rmse": metrics["ItemKNN_RMSE"],
                },
                {"model": "SVD", "mse": metrics["SVD_MSE"], "rmse": metrics["SVD_RMSE"]},
                {"model": "PMF", "mse": metrics["PMF_MSE"], "rmse": metrics["PMF_RMSE"]},
            ]
        )
        st.dataframe(
            _format_frame(metric_table, {"mse": "{:.6f}", "rmse": "{:.6f}"}),
            hide_index=True,
            use_container_width=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            _artifact_image("reports/rmse_comparison.png")
            _artifact_image("reports/pmf_convergence.png")
        with col2:
            _artifact_image("reports/predicted_vs_actual.png")

        st.divider()
        st.subheader("Top-K next-positive recovery")
        st.caption(
            resources["ranking_protocol"]["protocol"]
            + ". Candidates are the full supported catalog minus the user's "
            "strict temporal-prefix history; unseen movies are not treated as "
            "observed negatives."
        )
        ranking_table = pd.DataFrame(
            [
                {
                    "model": model_name,
                    "HitRate@10": values["HitRate@10"],
                    "NDCG@10": values["NDCG@10"],
                    "MRR@10": values["MRR@10"],
                    "mean_target_rank": values["mean_target_rank"],
                    "median_target_rank": values["median_target_rank"],
                }
                for model_name, values in resources["ranking_metrics"]["models"].items()
            ]
        )
        st.dataframe(
            _format_frame(
                ranking_table,
                {
                    "HitRate@10": "{:.4f}",
                    "NDCG@10": "{:.4f}",
                    "MRR@10": "{:.4f}",
                    "mean_target_rank": "{:.2f}",
                    "median_target_rank": "{:.1f}",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )
        _artifact_image("reports/ranking_comparison.png")

        ranking_case = _load_user_ranking_case(user_id)
        if ranking_case is not None:
            st.subheader("Held-out target case")
            st.dataframe(
                _format_frame(
                    ranking_case,
                    {
                        "target_rating": "{:.1f}",
                        "bias_raw_target_score": "{:.4f}",
                        "item_knn_raw_target_score": "{:.4f}",
                        "svd_raw_target_score": "{:.4f}",
                        "pmf_raw_target_score": "{:.4f}",
                        "nearest_known_rating": "{:.1f}",
                        "nearest_known_similarity": "{:.4f}",
                    },
                ),
                hide_index=True,
                use_container_width=True,
            )
            _artifact_image(f"reports/user_{user_id}_ranking_case.png")

    with tabs[3]:
        st.subheader("High-variance PMF factors")
        st.dataframe(
            _format_frame(
                resources["factor_interpretation"],
                {"factor_variance": "{:.6f}", "factor_loading": "{:.6f}"},
            ),
            hide_index=True,
            use_container_width=True,
        )
        _artifact_image("reports/pmf_latent_factor_heatmap.png")

        st.subheader("Genre profiles by factor polarity")
        st.dataframe(
            _format_frame(
                resources["factor_genre_profiles"],
                {"genre_share": "{:.3f}", "mean_factor_loading": "{:.6f}"},
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("PMF item-factor similarity examples")
        st.dataframe(
            _format_frame(
                resources["similarities"],
                {"cosine_similarity": "{:.4f}"},
            ),
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
