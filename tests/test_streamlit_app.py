from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from app import (
    CENTER_COLUMN_RATIOS,
    TABLE_BOTTOM_ALLOWANCE,
    TABLE_HEADER_HEIGHT,
    TABLE_ROW_HEIGHT,
    _build_grouped_anchor_similarity_table,
    _build_html_table,
    _combined_ranking_display,
    _compact_explanation_display,
    _format_frame,
    _nearest_known_display,
    _recommendation_display,
    _table_height,
)


ROOT = Path(__file__).resolve().parents[1]


def test_layout_helpers_keep_centered_widths_and_row_based_heights():
    assert CENTER_COLUMN_RATIOS == {
        "compact": (1, 5, 1),
        "medium": (1, 7, 1),
    }
    assert _table_height(10) == (
        TABLE_HEADER_HEIGHT
        + 10 * TABLE_ROW_HEIGHT
        + TABLE_BOTTOM_ALLOWANCE
    )
    assert _table_height(25) > _table_height(10)
    assert _table_height(50, maximum=620) == 620


def test_html_table_formats_and_escapes_without_mutating_table_values():
    source = pd.DataFrame(
        {
            "movie_id": [260],
            "title": ["Star Wars <Episode IV> & Friends"],
            "predicted_rating": [4.438717],
        }
    )
    original = source.copy(deep=True)

    display = _format_frame(
        source,
        {"predicted_rating": "{:.3f}"},
        integer_columns=("movie_id",),
    )
    html = _build_html_table(display, key="recommendations grid")
    scroll_html = _build_html_table(
        display,
        key="recommendations grid",
        max_height=200,
    )

    assert 'id="mf-table-recommendations-grid"' in html
    assert "max-height" not in html
    assert "mf-table-wrap--scroll" not in html
    assert 'style="max-height: 200px;"' in scroll_html
    assert "mf-table-wrap--scroll" in scroll_html
    assert "<table" in html
    assert "Star Wars &lt;Episode IV&gt; &amp; Friends" in html
    assert ">4.439<" in html
    assert ">260<" in html
    assert display["predicted_rating"].iloc[0] == source["predicted_rating"].iloc[0]
    assert display["movie_id"].iloc[0] == source["movie_id"].iloc[0]
    pd.testing.assert_frame_equal(source, original)


def test_grouped_anchor_similarity_table_uses_rowspan_without_mutating_source():
    source = pd.DataFrame(
        {
            "anchor_movie_id": [10, 10, 20],
            "anchor_title": ["Anchor <A>", "Anchor <A>", "Anchor B"],
            "anchor_genres": ["Drama", "Drama", "Comedy"],
            "similar_movie_id": [101, 102, 201],
            "similar_title": ["Similar A1", "Similar A2", "Similar B1"],
            "similar_genres": ["Drama", "Crime", "Comedy"],
            "cosine_similarity": [0.61234, 0.59876, 0.5],
            "rank": [1, 2, 1],
        }
    )
    original = source.copy(deep=True)

    display = _format_frame(
        source,
        {"cosine_similarity": "{:.3f}"},
        integer_columns=("anchor_movie_id", "similar_movie_id", "rank"),
    )
    html = _build_grouped_anchor_similarity_table(
        display,
        key="factor similarity grid",
    )

    assert 'id="mf-table-factor-similarity-grid"' in html
    assert html.count('rowspan="2"') == 3
    assert html.count('rowspan="1"') == 3
    assert '<td rowspan="2">10</td>' in html
    assert '<td rowspan="2">Anchor &lt;A&gt;</td>' in html
    assert ">Similar A1<" in html
    assert ">Similar A2<" in html
    assert ">0.612<" in html
    assert source["anchor_movie_id"].tolist() == [10, 10, 20]
    pd.testing.assert_frame_equal(source, original)


def test_evaluation_profile_shortcut_synchronizes_keyed_manual_input():
    profiles = json.loads(
        (ROOT / "reports" / "evaluated_users.json").read_text(encoding="utf-8")
    )
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)

    at.run()
    assert not at.exception
    assert at.text_input[0].value == str(profiles[0]["user_id"])

    at.selectbox[0].select(profiles[1]).run()
    assert not at.exception
    assert at.text_input[0].value == str(profiles[1]["user_id"])

    manual_value = str(profiles[2]["user_id"])
    at.text_input[0].input(manual_value).run()
    assert not at.exception
    assert at.text_input[0].value == manual_value

    at.text_input[0].input("not-a-number").run()
    assert not at.exception
    assert any("Invalid user ID input" in error.value for error in at.error)

    at.text_input[0].input("999999999").run()
    assert not at.exception
    assert any("Unknown user ID" in error.value for error in at.error)


def test_recommendation_and_combined_display_views_do_not_mutate_sources():
    recommendations = pd.DataFrame(
        {
            "movie_id": [10],
            "title": ["A"],
            "genres": ["Drama"],
            "ranking_score": [4.438717],
            "predicted_rating": [4.438717],
        }
    )
    comparison = pd.DataFrame(
        {
            "movie_id": [10, 20],
            "title": ["A", "B"],
            "svd_rank": [1.0, 2.0],
            "svd_ranking_score": [4.438717, 4.1],
            "pmf_rank": [1.0, np.nan],
            "pmf_ranking_score": [4.192632, np.nan],
        }
    )
    original_recommendations = recommendations.copy(deep=True)
    original_comparison = comparison.copy(deep=True)

    recommendation_view = _recommendation_display(recommendations)
    comparison_view = _combined_ranking_display(comparison, top_n=5)

    assert list(recommendation_view.columns) == [
        "movie_id",
        "title",
        "genres",
        "predicted_rating",
    ]
    assert list(comparison_view.columns) == [
        "movie_id",
        "title",
        "svd_rank",
        "svd_score",
        "pmf_rank",
        "pmf_score",
    ]
    assert comparison_view.loc[1, "pmf_rank"] == ">5"
    assert comparison_view.loc[1, "pmf_score"] == "n/a"
    pd.testing.assert_frame_equal(recommendations, original_recommendations)
    pd.testing.assert_frame_equal(comparison, original_comparison)


def test_explanation_display_views_separate_nearest_movie_context():
    explanations = pd.DataFrame(
        {
            "recommendation_rank": [1, 2],
            "title": ["A", "B"],
            "raw_pmf_ranking_score": [4.2, 4.1],
            "item_bias_contribution": [0.1, -0.1],
            "total_latent_dot_product": [0.2, 0.3],
            "nearest_known_title": ["Known A", "Known B"],
            "nearest_known_rating": [5.0, 4.0],
            "nearest_known_similarity": [0.8, 0.7],
            "common_genres": ["", np.nan],
        }
    )
    original = explanations.copy(deep=True)

    compact = _compact_explanation_display(explanations)
    nearest = _nearest_known_display(explanations)

    assert list(compact.columns) == [
        "recommendation_rank",
        "title",
        "raw_pmf_ranking_score",
        "item_bias_contribution",
        "total_latent_dot_product",
    ]
    assert "nearest_known_title" not in compact
    assert nearest["common_genres"].tolist() == ["none", "none"]
    pd.testing.assert_frame_equal(explanations, original)
