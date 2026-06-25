from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_MODELS = ("BiasBaseline", "ItemKNN", "SVD", "PMF")
MODEL_TO_RANK_COLUMN = {
    "BiasBaseline": "bias_target_rank",
    "ItemKNN": "item_knn_target_rank",
    "SVD": "svd_target_rank",
    "PMF": "pmf_target_rank",
}
MODEL_SORT_ORDER = {
    model: position for position, model in enumerate(EXPECTED_MODELS)
}
QUANTILE_LEVELS = {
    "p1": 0.01,
    "p5": 0.05,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}


@dataclass(frozen=True)
class RatingRankingAnalysis:
    comparison_table: pd.DataFrame
    reversal_table: pd.DataFrame
    rank_distribution_table: pd.DataFrame
    plot_data: pd.DataFrame
    pmf_better_share: float
    svd_better_share: float
    tie_share: float
    interpretation: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_expected_models(models: set[str], artifact_name: str) -> None:
    expected = set(EXPECTED_MODELS)
    missing = expected - models
    extra = models - expected
    if missing or extra:
        raise ValueError(
            f"{artifact_name} models must be exactly {list(EXPECTED_MODELS)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def build_rating_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for model in EXPECTED_MODELS:
        mse_key = f"{model}_MSE"
        rmse_key = f"{model}_RMSE"
        if mse_key not in metrics or rmse_key not in metrics:
            raise ValueError(f"model metrics missing {mse_key} or {rmse_key}")
        rows.append(
            {
                "model": model,
                "test_mse": float(metrics[mse_key]),
                "test_rmse": float(metrics[rmse_key]),
            }
        )
    return pd.DataFrame(rows)


def metric_value_groups(
    frame: pd.DataFrame,
    value_column: str,
    *,
    ascending: bool,
) -> list[list[dict[str, float | str]]]:
    _validate_expected_models(set(frame["model"]), "comparison table")
    ordered = sorted(
        (
            {
                "model": str(row["model"]),
                "value": float(row[value_column]),
            }
            for row in frame[["model", value_column]].to_dict("records")
        ),
        key=lambda row: (
            row["value"] if ascending else -float(row["value"]),
            MODEL_SORT_ORDER[str(row["model"])],
        ),
    )
    groups: list[list[dict[str, float | str]]] = []
    for row in ordered:
        if not groups or not np.isclose(
            float(groups[-1][0]["value"]),
            float(row["value"]),
            rtol=0.0,
            atol=0.0,
        ):
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def competition_positions(
    frame: pd.DataFrame,
    value_column: str,
    *,
    ascending: bool,
) -> pd.Series:
    positions: dict[str, int] = {}
    next_position = 1
    for group in metric_value_groups(
        frame, value_column, ascending=ascending
    ):
        for row in group:
            positions[str(row["model"])] = next_position
        next_position += len(group)
    return frame["model"].map(positions).astype(int)


def deterministic_model_order(
    frame: pd.DataFrame,
    value_column: str,
    *,
    ascending: bool,
) -> list[str]:
    return [
        str(row["model"])
        for group in metric_value_groups(
            frame, value_column, ascending=ascending
        )
        for row in group
    ]


def detect_metric_reversals(comparison_table: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "rmse_position", "hit_rate_10_position"}
    missing = required - set(comparison_table.columns)
    if missing:
        raise ValueError(f"comparison table missing columns: {sorted(missing)}")
    reversals = comparison_table.loc[
        comparison_table["rmse_position"].ne(
            comparison_table["hit_rate_10_position"]
        ),
        ["model", "rmse_position", "hit_rate_10_position"],
    ].copy()
    reversals["position_change"] = (
        reversals["hit_rate_10_position"] - reversals["rmse_position"]
    )
    return reversals.sort_values(
        ["rmse_position", "hit_rate_10_position", "model"],
        kind="mergesort",
    ).reset_index(drop=True)


def _validate_rank_columns(ranking_results: pd.DataFrame) -> None:
    missing = set(MODEL_TO_RANK_COLUMN.values()) - set(ranking_results.columns)
    if missing:
        raise ValueError(
            f"ranking results missing target-rank columns: {sorted(missing)}"
        )
    values = ranking_results[list(MODEL_TO_RANK_COLUMN.values())].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or (values < 1).any():
        raise ValueError("target ranks must be finite and at least 1")


def build_comparison_table(
    rating_table: pd.DataFrame,
    ranking_metrics: dict[str, Any],
    ranking_results: pd.DataFrame,
) -> pd.DataFrame:
    _validate_expected_models(set(rating_table["model"]), "rating metrics")
    ranking_models = ranking_metrics.get("models")
    if not isinstance(ranking_models, dict):
        raise ValueError("ranking metrics missing models object")
    _validate_expected_models(set(ranking_models), "ranking metrics")
    _validate_rank_columns(ranking_results)

    comparison = rating_table.copy()
    for metric in ("HitRate@5", "HitRate@10", "median_target_rank"):
        comparison[metric] = comparison["model"].map(
            lambda model: float(ranking_models[model][metric])
        )
    comparison["share_target_rank_gt_2000"] = comparison["model"].map(
        lambda model: float(
            ranking_results[MODEL_TO_RANK_COLUMN[model]].gt(2000).mean()
        )
    )
    comparison["rmse_position"] = competition_positions(
        comparison, "test_rmse", ascending=True
    )
    comparison["hit_rate_10_position"] = competition_positions(
        comparison, "HitRate@10", ascending=False
    )
    value_columns = [
        "test_mse",
        "test_rmse",
        "HitRate@5",
        "HitRate@10",
        "median_target_rank",
        "share_target_rank_gt_2000",
    ]
    if not np.isfinite(comparison[value_columns].to_numpy(dtype=float)).all():
        raise ValueError("comparison metrics must be finite")
    for column in ("HitRate@5", "HitRate@10", "share_target_rank_gt_2000"):
        if not comparison[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be within [0, 1]")
    return comparison.sort_values(
        ["rmse_position", "model"],
        key=lambda series: (
            series.map(MODEL_SORT_ORDER)
            if series.name == "model"
            else series
        ),
        kind="mergesort",
    ).reset_index(drop=True)


def build_rank_distribution(
    ranking_results: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float]:
    _validate_rank_columns(ranking_results)
    svd_ranks = ranking_results[MODEL_TO_RANK_COLUMN["SVD"]]
    pmf_ranks = ranking_results[MODEL_TO_RANK_COLUMN["PMF"]]
    pmf_better_share = float(pmf_ranks.lt(svd_ranks).mean())
    svd_better_share = float(svd_ranks.lt(pmf_ranks).mean())
    tie_share = float(pmf_ranks.eq(svd_ranks).mean())
    if not np.isclose(
        pmf_better_share + svd_better_share + tie_share,
        1.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("SVD wins, PMF wins, and ties must partition users")

    rows = []
    for model, ranks, better_share in (
        ("SVD", svd_ranks, svd_better_share),
        ("PMF", pmf_ranks, pmf_better_share),
    ):
        quantiles = ranks.quantile(
            list(QUANTILE_LEVELS.values()), interpolation="linear"
        )
        rows.append(
            {
                "model": model,
                **dict(zip(QUANTILE_LEVELS, quantiles.to_numpy(), strict=True)),
                "share_rank_gt_2000": float(ranks.gt(2000).mean()),
                "per_user_better_share": better_share,
            }
        )
    table = pd.DataFrame(rows)
    quantile_values = table[list(QUANTILE_LEVELS)].to_numpy(dtype=float)
    if not np.isfinite(quantile_values).all():
        raise ValueError("target-rank quantiles must be finite")
    if any(np.any(np.diff(row) < -1e-12) for row in quantile_values):
        raise ValueError("target-rank quantiles must be non-decreasing")
    return table, pmf_better_share, svd_better_share, tie_share


def _grouped_order_text(
    frame: pd.DataFrame,
    value_column: str,
    *,
    ascending: bool,
    decimals: int,
    percentage: bool = False,
) -> str:
    groups = metric_value_groups(frame, value_column, ascending=ascending)

    def format_value(value: float) -> str:
        return (
            f"{value:.{decimals}%}"
            if percentage
            else f"{value:.{decimals}f}"
        )

    group_text = [
        " = ".join(
            f"{row['model']} ({format_value(float(row['value']))})"
            for row in group
        )
        for group in groups
    ]
    return (" < " if ascending else " > ").join(group_text)


def build_interpretation(
    comparison: pd.DataFrame,
    reversals: pd.DataFrame,
    rank_distribution: pd.DataFrame,
    pmf_better_share: float,
    svd_better_share: float,
    tie_share: float,
) -> str:
    rmse_order = _grouped_order_text(
        comparison, "test_rmse", ascending=True, decimals=3
    )
    hit_order = _grouped_order_text(
        comparison,
        "HitRate@10",
        ascending=False,
        decimals=2,
        percentage=True,
    )
    if reversals.empty:
        reversal_text = "No model changes position between the two objectives."
    else:
        reversal_text = " ".join(
            f"{row.model} is #{row.rmse_position} by RMSE and "
            f"#{row.hit_rate_10_position} by HitRate@10."
            for row in reversals.itertuples(index=False)
        )

    distribution = rank_distribution.set_index("model")
    tail_text = (
        "Deep-tail shares above rank 2,000 are "
        f"SVD {distribution.loc['SVD', 'share_rank_gt_2000']:.2%} and "
        f"PMF {distribution.loc['PMF', 'share_rank_gt_2000']:.2%}."
    )
    return (
        f"By test RMSE: {rmse_order}. By HitRate@10: {hit_order}. "
        f"{reversal_text} {tail_text} Per user, PMF has the lower target rank "
        f"in {pmf_better_share:.1%} of cases, SVD in "
        f"{svd_better_share:.1%}, with {tie_share:.1%} ties. "
        "Rating prediction and top-K retrieval answer different questions, so "
        "the preferred model depends on the evaluation objective."
    )


def prepare_tradeoff_plot_data(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison[
        ["model", "test_rmse", "HitRate@10", "rmse_position", "hit_rate_10_position"]
    ].sort_values(
        ["rmse_position", "model"],
        key=lambda series: (
            series.map(MODEL_SORT_ORDER)
            if series.name == "model"
            else series
        ),
        kind="mergesort",
    ).reset_index(drop=True)


def plot_rating_ranking_tradeoff(
    plot_data: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    rmse_view = plot_data.sort_values(
        ["test_rmse", "model"],
        key=lambda series: (
            series.map(MODEL_SORT_ORDER)
            if series.name == "model"
            else series
        ),
        kind="mergesort",
    )
    hit_view = plot_data.sort_values(
        ["HitRate@10", "model"],
        ascending=[False, True],
        key=lambda series: (
            series.map(MODEL_SORT_ORDER)
            if series.name == "model"
            else series
        ),
        kind="mergesort",
    )
    axes[0].barh(rmse_view["model"], rmse_view["test_rmse"], color="#4c78a8")
    axes[0].invert_yaxis()
    axes[0].set(title="Rating accuracy", xlabel="Test RMSE")
    axes[1].barh(hit_view["model"], hit_view["HitRate@10"], color="#f58518")
    axes[1].invert_yaxis()
    axes[1].set(title="Top-K retrieval", xlabel="HitRate@10")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Rating accuracy vs ranking")
    fig.tight_layout()
    return fig


def analyze_rating_ranking(
    metrics: dict[str, Any],
    ranking_metrics: dict[str, Any],
    ranking_results: pd.DataFrame,
) -> RatingRankingAnalysis:
    rating_table = build_rating_table(metrics)
    comparison = build_comparison_table(
        rating_table, ranking_metrics, ranking_results
    )
    reversals = detect_metric_reversals(comparison)
    rank_distribution, pmf_better, svd_better, ties = build_rank_distribution(
        ranking_results
    )
    plot_data = prepare_tradeoff_plot_data(comparison)
    interpretation = build_interpretation(
        comparison,
        reversals,
        rank_distribution,
        pmf_better,
        svd_better,
        ties,
    )
    return RatingRankingAnalysis(
        comparison_table=comparison,
        reversal_table=reversals,
        rank_distribution_table=rank_distribution,
        plot_data=plot_data,
        pmf_better_share=pmf_better,
        svd_better_share=svd_better,
        tie_share=ties,
        interpretation=interpretation,
    )


def load_rating_ranking_analysis(
    reports_dir: str | Path,
) -> RatingRankingAnalysis:
    reports_dir = Path(reports_dir)
    return analyze_rating_ranking(
        _load_json(reports_dir / "model_metrics.json"),
        _load_json(reports_dir / "ranking_metrics.json"),
        pd.read_csv(reports_dir / "ranking_results.csv"),
    )
