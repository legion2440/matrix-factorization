from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Callable

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
        ROOT / "reports" / "rmse_comparison.png",
        ROOT / "reports" / "pmf_convergence.png",
        ROOT / "reports" / "svd_rank_tuning_rmse.png",
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


FormatValue = str | Callable[[Any], str]
CENTER_COLUMN_RATIOS = {
    "compact": (1, 5, 1),
    "medium": (1, 7, 1),
}
TABLE_HEADER_HEIGHT = 40
TABLE_ROW_HEIGHT = 35
TABLE_BOTTOM_ALLOWANCE = 4
PAGE_CONTENT_REFERENCE_WIDTH = 1400


def _apply_layout_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"],
        .block-container {
            width: 100%;
            max-width: 1920px;
            margin-left: auto;
            margin-right: auto;
            padding-left: clamp(1rem, 2.5vw, 2.5rem);
            padding-right: clamp(1rem, 2.5vw, 2.5rem);
        }
        .mf-expander-gap {
            height: 1rem;
        }
        .mf-table-wrap {
            width: fit-content;
            max-width: 100%;
            margin-left: auto;
            margin-right: auto;
            overflow-x: visible;
            overflow-y: visible;
            border: 1px solid rgba(49, 51, 63, 0.18);
            border-radius: 8px;
            background: white;
        }
        .mf-table-wrap--scroll {
            width: 100%;
            overflow: auto;
        }
        .mf-table {
            width: max-content;
            border-collapse: collapse;
            font-size: 0.875rem;
            line-height: 1.35;
        }
        .mf-table th,
        .mf-table td {
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid rgba(49, 51, 63, 0.12);
            text-align: left;
            vertical-align: top;
            white-space: normal;
            overflow-wrap: anywhere;
            max-width: 28rem;
        }
        .mf-table-wrap--scroll .mf-table th {
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .mf-table th {
            background: rgb(248, 249, 251);
            color: rgb(49, 51, 63);
            font-weight: 600;
        }
        .mf-table tr:last-child td {
            border-bottom: 0;
        }
        .mf-table tbody tr:nth-child(even) {
            background: rgba(49, 51, 63, 0.035);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _centered_column(size: str = "compact") -> Any:
    if size not in CENTER_COLUMN_RATIOS:
        raise ValueError(f"Unknown centered column size: {size}")
    _, center, _ = st.columns(CENTER_COLUMN_RATIOS[size], gap="small")
    return center


def _centered_width_column(target_width: int) -> Any:
    width = min(max(int(target_width), 1), PAGE_CONTENT_REFERENCE_WIDTH)
    side = max((PAGE_CONTENT_REFERENCE_WIDTH - width) / 2, 1)
    _, center, _ = st.columns([side, width, side], gap="small")
    return center


def _table_subheader(label: str) -> None:
    st.subheader(label, text_alignment="center")


def _table_height(row_count: int, maximum: int | None = None) -> int:
    height = (
        TABLE_HEADER_HEIGHT
        + max(int(row_count), 0) * TABLE_ROW_HEIGHT
        + TABLE_BOTTOM_ALLOWANCE
    )
    return min(height, maximum) if maximum is not None else height


def _show_dataframe(
    data: pd.DataFrame,
    *,
    row_count: int,
    maximum_height: int | None = None,
    key: str,
) -> None:
    max_height = (
        _table_height(row_count, maximum_height)
        if maximum_height is not None
        else None
    )
    st.markdown(
        _build_html_table(
            data,
            key=key,
            max_height=max_height,
        ),
        unsafe_allow_html=True,
    )


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _format_display_value(value: Any, format_spec: FormatValue | None) -> str:
    if _is_missing(value):
        return "n/a"
    if callable(format_spec):
        return str(format_spec(value))
    if format_spec:
        try:
            return str(format_spec.format(value))
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _format_integer_value(value: Any) -> str:
    if _is_missing(value):
        return "n/a"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def _format_table_cell_text(
    value: Any,
    column: str,
    formats: dict[str, FormatValue],
    integer_columns: tuple[str, ...],
) -> str:
    if column in integer_columns:
        return _format_integer_value(value)
    return _format_display_value(value, formats.get(column))


def _build_html_table(
    data: pd.DataFrame,
    *,
    key: str,
    max_height: int | None = None,
) -> str:
    formats = data.attrs.get("display_formats", {})
    integer_columns = data.attrs.get("integer_columns", ())
    table_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", key).strip("-") or "table"
    header_cells = "".join(
        f"<th>{html.escape(str(column))}</th>" for column in data.columns
    )

    rows = []
    for _, row in data.iterrows():
        cells = []
        for column in data.columns:
            text = _format_table_cell_text(
                row[column],
                column,
                formats,
                integer_columns,
            )
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    body = "".join(rows) or (
        f"<tr><td colspan=\"{len(data.columns)}\">No rows to display</td></tr>"
    )
    wrapper_class = "mf-table-wrap"
    wrapper_style = ""
    if max_height is not None:
        wrapper_class += " mf-table-wrap--scroll"
        wrapper_style = f" style=\"max-height: {int(max_height)}px;\""
    return (
        f"<div id=\"mf-table-{table_id}\" class=\"{wrapper_class}\"{wrapper_style}>"
        "<table class=\"mf-table\">"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</div>"
    )


def _build_grouped_anchor_similarity_table(
    data: pd.DataFrame,
    *,
    key: str,
) -> str:
    anchor_columns = ("anchor_movie_id", "anchor_title", "anchor_genres")
    missing_anchor_columns = [
        column for column in anchor_columns if column not in data.columns
    ]
    if missing_anchor_columns:
        raise ValueError(f"Missing anchor columns: {missing_anchor_columns}")

    formats = data.attrs.get("display_formats", {})
    integer_columns = data.attrs.get("integer_columns", ())
    table_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", key).strip("-") or "table"
    header_cells = "".join(
        f"<th>{html.escape(str(column))}</th>" for column in data.columns
    )

    rows = []
    records = list(data.iterrows())
    index = 0
    while index < len(records):
        _, first_row = records[index]
        anchor_key = tuple(first_row[column] for column in anchor_columns)
        group_end = index + 1
        while group_end < len(records):
            _, candidate = records[group_end]
            candidate_key = tuple(candidate[column] for column in anchor_columns)
            if candidate_key != anchor_key:
                break
            group_end += 1

        group_size = group_end - index
        for row_position in range(index, group_end):
            _, row = records[row_position]
            cells = []
            if row_position == index:
                for column in anchor_columns:
                    text = _format_table_cell_text(
                        row[column],
                        column,
                        formats,
                        integer_columns,
                    )
                    cells.append(
                        f"<td rowspan=\"{group_size}\">{html.escape(text)}</td>"
                    )
            for column in data.columns:
                if column in anchor_columns:
                    continue
                text = _format_table_cell_text(
                    row[column],
                    column,
                    formats,
                    integer_columns,
                )
                cells.append(f"<td>{html.escape(text)}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")

        index = group_end

    body = "".join(rows) or (
        f"<tr><td colspan=\"{len(data.columns)}\">No rows to display</td></tr>"
    )
    return (
        f"<div id=\"mf-table-{table_id}\" class=\"mf-table-wrap\">"
        "<table class=\"mf-table\">"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table>"
        "</div>"
    )


def _show_grouped_anchor_similarity_table(data: pd.DataFrame, *, key: str) -> None:
    st.markdown(
        _build_grouped_anchor_similarity_table(data, key=key),
        unsafe_allow_html=True,
    )


def _format_frame(
    frame: pd.DataFrame,
    formats: dict[str, FormatValue] | None = None,
    *,
    integer_columns: tuple[str, ...] = (),
) -> Any:
    view = frame.copy()
    present = {
        key: value
        for key, value in (formats or {}).items()
        if key in view.columns
    }
    view.attrs["display_formats"] = present
    view.attrs["integer_columns"] = tuple(
        column for column in integer_columns if column in view.columns
    )
    return view


def _combined_ranking_display(
    comparison: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    view = comparison.sort_values(
        ["svd_rank", "pmf_rank", "movie_id"],
        na_position="last",
        kind="mergesort",
    )[
        [
            "movie_id",
            "title",
            "svd_rank",
            "svd_ranking_score",
            "pmf_rank",
            "pmf_ranking_score",
        ]
    ].copy()
    view = view.rename(
        columns={
            "svd_ranking_score": "svd_score",
            "pmf_ranking_score": "pmf_score",
        }
    )
    for prefix in ("svd", "pmf"):
        rank_column = f"{prefix}_rank"
        score_column = f"{prefix}_score"
        view[rank_column] = view[rank_column].map(
            lambda value: f">{top_n}" if pd.isna(value) else str(int(value))
        )
        view[score_column] = view[score_column].map(
            lambda value: "n/a" if pd.isna(value) else f"{float(value):.3f}"
        )
    return view


def _recommendation_display(recommendations: pd.DataFrame) -> pd.DataFrame:
    return recommendations[
        ["movie_id", "title", "genres", "predicted_rating"]
    ].copy()


def _compact_explanation_display(explanations: pd.DataFrame) -> pd.DataFrame:
    return explanations[
        [
            "recommendation_rank",
            "title",
            "raw_pmf_ranking_score",
            "item_bias_contribution",
            "total_latent_dot_product",
        ]
    ].copy()


def _nearest_known_display(explanations: pd.DataFrame) -> pd.DataFrame:
    view = explanations[
        [
            "recommendation_rank",
            "title",
            "nearest_known_title",
            "nearest_known_rating",
            "nearest_known_similarity",
            "common_genres",
        ]
    ].copy()
    genres = view["common_genres"].fillna("").astype(str).str.strip()
    view["common_genres"] = genres.mask(genres.eq(""), "none")
    return view


def _compact_ranking_case_display(ranking_case: pd.DataFrame) -> pd.DataFrame:
    return ranking_case[
        [
            "target_title",
            "target_rating",
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
            "pmf_raw_target_score",
            "nearest_known_title",
        ]
    ].copy()


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
    _apply_layout_css()
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
    with _centered_column("medium"):
        metric_columns = st.columns(4, gap="small")
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
    with _centered_column("medium"):
        controls = st.columns([5, 5, 2], gap="small")
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
        st.caption(
            "Showcase roles are defined by temporal ranking outcomes, not per-user "
            "RMSE. These users have training history and are evaluated on held-out "
            "interactions; cold start is outside the project scope."
        )
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
        with st.container(horizontal_alignment="center"):
            _table_subheader("Highest-rated known films")
            _show_dataframe(
                _format_frame(
                    history,
                    {"rating": "{:.1f}"},
                    integer_columns=("movie_id",),
                ),
                row_count=len(history),
                key="history_grid",
            )

        left, right = st.columns(2)
        with left:
            _table_subheader("SVD recommendations")
            svd_display = _recommendation_display(svd_recommendations)
            _show_dataframe(
                _format_frame(
                    svd_display,
                    {"predicted_rating": "{:.3f}"},
                    integer_columns=("movie_id",),
                ),
                row_count=len(svd_recommendations),
                key="svd_recommendations_grid",
            )
        with right:
            _table_subheader("PMF recommendations")
            pmf_display = _recommendation_display(pmf_recommendations)
            _show_dataframe(
                _format_frame(
                    pmf_display,
                    {"predicted_rating": "{:.3f}"},
                    integer_columns=("movie_id",),
                ),
                row_count=len(pmf_recommendations),
                key="pmf_recommendations_grid",
            )

        with _centered_column("medium"):
            _table_subheader("Combined ranking")
            combined_display = _combined_ranking_display(comparison, top_n)
            _show_dataframe(
                combined_display,
                row_count=len(combined_display),
                key="combined_ranking_grid",
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
            compact_explanations = _compact_explanation_display(explanations)
            with _centered_column("compact"):
                _table_subheader("Local PMF explanation table")
                _show_dataframe(
                    _format_frame(
                        compact_explanations,
                        {
                            "raw_pmf_ranking_score": "{:.3f}",
                            "item_bias_contribution": "{:.4f}",
                            "total_latent_dot_product": "{:.4f}",
                        },
                        integer_columns=("recommendation_rank",),
                    ),
                    row_count=len(compact_explanations),
                    key="local_explanation_grid",
                )

                with st.expander("Full technical explanation", expanded=False):
                    _show_dataframe(
                        _format_frame(
                            explanations,
                            {
                                "raw_pmf_ranking_score": "{:.3f}",
                                "clipped_displayed_rating": "{:.3f}",
                                "global_mean_contribution": "{:.4f}",
                                "user_bias_contribution": "{:.4f}",
                                "item_bias_contribution": "{:.4f}",
                                "total_latent_dot_product": "{:.4f}",
                                "top_factor_1_contribution": "{:.4f}",
                                "top_factor_2_contribution": "{:.4f}",
                                "top_factor_3_contribution": "{:.4f}",
                                "component_sum": "{:.4f}",
                                "reconstruction_error": "{:.4f}",
                                "nearest_known_rating": "{:.1f}",
                                "nearest_known_similarity": "{:.3f}",
                            },
                            integer_columns=(
                                "user_id",
                                "recommendation_rank",
                                "movie_id",
                                "top_factor_1_index",
                                "top_factor_2_index",
                                "top_factor_3_index",
                                "nearest_known_movie_id",
                            ),
                        ),
                        row_count=len(explanations),
                        maximum_height=620,
                        key="full_explanation_grid",
                    )
            nearest_display = _nearest_known_display(explanations)
            with st.container(horizontal_alignment="center"):
                _table_subheader("Nearest known liked movie")
                _show_dataframe(
                    _format_frame(
                        nearest_display,
                        {
                            "nearest_known_rating": "{:.1f}",
                            "nearest_known_similarity": "{:.3f}",
                        },
                        integer_columns=("recommendation_rank",),
                    ),
                    row_count=len(nearest_display),
                    key="nearest_known_grid",
                )
            with _centered_width_column(1280):
                _artifact_image(f"reports/user_{user_id}_explanation.png")

    with tabs[2]:
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
        with st.container(horizontal_alignment="center"):
            _table_subheader("Rating prediction")
            _show_dataframe(
                _format_frame(metric_table, {"mse": "{:.6f}", "rmse": "{:.6f}"}),
                row_count=len(metric_table),
                key="rating_prediction_grid",
            )
        col1, col2 = st.columns(2)
        with col1:
            _artifact_image("reports/rmse_comparison.png")
            _artifact_image("reports/pmf_convergence.png")
        with col2:
            _artifact_image("reports/svd_rank_tuning_rmse.png")
        with _centered_width_column(830):
            _artifact_image("reports/predicted_vs_actual.png")

        st.divider()
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
        with st.container(horizontal_alignment="center"):
            _table_subheader("Top-K next-positive recovery")
            st.caption(
                resources["ranking_protocol"]["protocol"]
                + ". Candidates are the full supported catalog minus the user's "
                "strict temporal-prefix history; unseen movies are not treated as "
                "observed negatives."
            )
            _show_dataframe(
                _format_frame(
                    ranking_table,
                    {
                        "HitRate@10": "{:.4f}",
                        "NDCG@10": "{:.4f}",
                        "MRR@10": "{:.4f}",
                        "mean_target_rank": "{:.2f}",
                        "median_target_rank": "{:.0f}",
                    },
                ),
                row_count=len(ranking_table),
                key="ranking_metrics_grid",
            )
        with _centered_width_column(900):
            _artifact_image("reports/ranking_comparison.png")

        ranking_case = _load_user_ranking_case(user_id)
        if ranking_case is not None:
            compact_ranking_case = _compact_ranking_case_display(ranking_case)
            with st.container(horizontal_alignment="center"):
                _table_subheader("Held-out target case")
                _show_dataframe(
                    _format_frame(
                        compact_ranking_case,
                        {
                            "target_rating": "{:.1f}",
                            "pmf_raw_target_score": "{:.3f}",
                        },
                        integer_columns=(
                            "prior_history_count",
                            "candidate_count",
                            "bias_target_rank",
                            "item_knn_target_rank",
                            "svd_target_rank",
                            "pmf_target_rank",
                        ),
                    ),
                    row_count=len(compact_ranking_case),
                    key="held_out_target_grid",
                )

            st.markdown('<div class="mf-expander-gap"></div>', unsafe_allow_html=True)
            with st.expander("Full technical ranking case", expanded=False):
                _show_dataframe(
                    _format_frame(
                        ranking_case,
                        {
                            "target_rating": "{:.1f}",
                            "bias_raw_target_score": "{:.3f}",
                            "item_knn_raw_target_score": "{:.3f}",
                            "svd_raw_target_score": "{:.3f}",
                            "pmf_raw_target_score": "{:.3f}",
                            "pmf_global_mean_contribution": "{:.4f}",
                            "pmf_user_bias_contribution": "{:.4f}",
                            "pmf_item_bias_contribution": "{:.4f}",
                            "pmf_total_latent_dot_product": "{:.4f}",
                            "pmf_component_sum": "{:.4f}",
                            "pmf_reconstruction_error": "{:.4f}",
                            "top_factor_1_contribution": "{:.4f}",
                            "top_factor_2_contribution": "{:.4f}",
                            "top_factor_3_contribution": "{:.4f}",
                            "nearest_known_rating": "{:.1f}",
                            "nearest_known_similarity": "{:.3f}",
                        },
                        integer_columns=(
                            "user_id",
                            "target_movie_id",
                            "target_timestamp",
                            "prior_history_count",
                            "candidate_count",
                            "bias_target_rank",
                            "item_knn_target_rank",
                            "svd_target_rank",
                            "pmf_target_rank",
                            "top_factor_1_index",
                            "top_factor_2_index",
                            "top_factor_3_index",
                            "nearest_known_movie_id",
                        ),
                    ),
                    row_count=len(ranking_case),
                    maximum_height=400,
                    key="full_ranking_case_grid",
                )
            with _centered_width_column(1280):
                _artifact_image(f"reports/user_{user_id}_ranking_case.png")

    with tabs[3]:
        factor_interpretation = resources["factor_interpretation"]
        with st.container(horizontal_alignment="center"):
            _table_subheader("High-variance PMF factors")
            _show_dataframe(
                _format_frame(
                    factor_interpretation,
                    {"factor_variance": "{:.6f}", "factor_loading": "{:.4f}"},
                    integer_columns=(
                        "factor_index",
                        "polarity_rank",
                        "movie_id",
                    ),
                ),
                row_count=len(factor_interpretation),
                key="factor_interpretation_grid",
            )
        with _centered_width_column(900):
            _artifact_image("reports/pmf_latent_factor_heatmap.png")

        factor_genre_profiles = resources["factor_genre_profiles"]
        with st.container(horizontal_alignment="center"):
            _table_subheader("Genre profiles by factor polarity")
            _show_dataframe(
                _format_frame(
                    factor_genre_profiles,
                    {"genre_share": "{:.3f}", "mean_factor_loading": "{:.4f}"},
                    integer_columns=("factor_index", "movie_count"),
                ),
                row_count=len(factor_genre_profiles),
                key="factor_genre_profiles_grid",
            )

        similarities = resources["similarities"]
        with st.container(horizontal_alignment="center"):
            _table_subheader("PMF item-factor similarity examples")
            _show_grouped_anchor_similarity_table(
                _format_frame(
                    similarities,
                    {"cosine_similarity": "{:.3f}"},
                    integer_columns=(
                        "anchor_movie_id",
                        "similar_movie_id",
                        "rank",
                    ),
                ),
                key="factor_similarity_grid",
            )


if __name__ == "__main__":
    main()
