from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def _md(notebook: nbf.NotebookNode, source: str) -> None:
    notebook.cells.append(nbf.v4.new_markdown_cell(dedent(source).strip()))


def _code(notebook: nbf.NotebookNode, source: str) -> None:
    notebook.cells.append(nbf.v4.new_code_cell(dedent(source).strip()))


def build_notebook(root: Path) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    _md(
        notebook,
        """
        # MovieLens 1M Matrix Factorization

        This notebook reads artifacts produced by `python -m scripts.run_pipeline`.
        It does not retrain or retune any model.
        """,
    )
    _code(
        notebook,
        """
        from pathlib import Path
        import json

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import Image, Markdown, display

        from utils.data_loader import load_movielens
        from utils.eda import (
            aggregate_genre_statistics,
            aggregate_temporal_ratings,
            demographic_distributions,
        )
        from utils.rating_ranking_analysis import (
            build_rating_table,
            load_rating_ranking_analysis,
            plot_rating_ranking_tradeoff,
        )

        ROOT = Path.cwd()
        REPORTS = ROOT / "reports"
        PROCESSED = ROOT / "processed"

        def load_json(path):
            return json.loads(path.read_text(encoding="utf-8"))

        data = load_movielens(ROOT / "data")
        train = pd.read_csv(PROCESSED / "train_ratings.csv")
        validation = pd.read_csv(PROCESSED / "validation_ratings.csv")
        test = pd.read_csv(PROCESSED / "test_ratings.csv")
        ranking_train = pd.read_csv(PROCESSED / "ranking_train_ratings.csv")
        ranking_targets = pd.read_csv(PROCESSED / "ranking_targets.csv")

        metrics = load_json(REPORTS / "model_metrics.json")
        bias_tuning = load_json(REPORTS / "bias_baseline_tuning.json")
        item_knn_tuning = load_json(REPORTS / "item_knn_tuning.json")
        ranking_protocol = load_json(REPORTS / "ranking_protocol.json")
        ranking_metrics = load_json(REPORTS / "ranking_metrics.json")
        evaluated_users = load_json(REPORTS / "evaluated_users.json")
        pmf_convergence = load_json(REPORTS / "pmf_convergence.json")

        svd_tuning = pd.read_json(REPORTS / "svd_tuning.json")
        pmf_tuning = pd.read_json(REPORTS / "pmf_tuning.json")
        ranking_results = pd.read_csv(REPORTS / "ranking_results.csv")
        factor_interpretation = pd.read_csv(
            REPORTS / "pmf_factor_interpretation.csv"
        )
        factor_genre_profiles = pd.read_csv(
            REPORTS / "pmf_factor_genre_profiles.csv"
        )
        similarities = pd.read_csv(REPORTS / "pmf_movie_similarities.csv")
        evaluated = pd.DataFrame(evaluated_users)
        """,
    )

    _md(
        notebook,
        """
        ## 1. Project goal

        Compare BiasBaseline, residualized ItemKNN, SVD, and PMF under two
        explicitly different evaluation protocols.

        - RMSE is pointwise rating-prediction accuracy on the deterministic
          interaction split; MSE is shown alongside it.
        - HitRate, NDCG, and MRR measure held-out next-positive recovery under a
          temporal leave-one-positive-out protocol.

        These are different tasks, so rating RMSE is not presented as Top-K
        recommendation accuracy.
        """,
    )

    _md(
        notebook,
        """
        ## 2. MovieLens EDA and Insights

        MovieLens 1M contains about one million ratings from 6,040 users for
        3,706 rated movies. The matrix is approximately 95.7% sparse. User
        activity is heterogeneous, movie popularity has a long tail, and the
        rating distribution is skewed upward.

        The full raw dataset is used only for descriptive EDA. Hyperparameter
        selection, stopping decisions, and model comparison use the predefined
        training and validation partitions, while the test partition remains
        untouched until final evaluation.
        """,
    )
    _code(
        notebook,
        """
        rating_counts = data.ratings["rating"].value_counts().sort_index()
        user_activity = data.ratings.groupby("user_id").size()
        movie_popularity = data.ratings.groupby("movie_id").size()
        sparsity = 1.0 - len(data.ratings) / (
            data.users["user_id"].nunique() * data.movies["movie_id"].nunique()
        )
        display(pd.DataFrame([
            {"metric": "ratings", "value": len(data.ratings)},
            {"metric": "users", "value": data.ratings["user_id"].nunique()},
            {"metric": "rated_movies", "value": data.ratings["movie_id"].nunique()},
            {"metric": "matrix_sparsity", "value": sparsity},
            {"metric": "median_user_activity", "value": user_activity.median()},
            {"metric": "median_movie_popularity", "value": movie_popularity.median()},
        ]))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        rating_counts.plot(kind="bar", ax=axes[0], color="#4c78a8")
        axes[0].set(title="Rating distribution", xlabel="Rating", ylabel="Count")
        axes[1].hist(user_activity, bins=50, color="#72b7b2")
        axes[1].set(title="User activity", xlabel="Ratings per user")
        axes[2].hist(movie_popularity, bins=60, log=True, color="#f58518")
        axes[2].set(title="Movie popularity", xlabel="Ratings per movie")
        fig.tight_layout()
        plt.show()
        """,
    )

    _md(
        notebook,
        """
        ### 2.1 Temporal EDA

        Interactions are distributed unevenly over time and timestamps preserve
        ordering. Therefore ranking evaluation uses temporal
        leave-one-positive-out. This does not change the main rating-prediction
        split, which remains deterministic 70/15/15.
        """,
    )
    _code(
        notebook,
        """
        temporal = aggregate_temporal_ratings(data.ratings)
        raw_dates = pd.to_datetime(data.ratings["timestamp"], unit="s", utc=True)
        display(pd.DataFrame([{
            "minimum_date": raw_dates.min(),
            "maximum_date": raw_dates.max(),
            "months": len(temporal),
        }]))
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(temporal["month"], temporal["rating_count"], color="#4c78a8")
        axes[0].set(title="Ratings per month", ylabel="Rating count")
        axes[1].plot(temporal["month"], temporal["mean_rating"], color="#f58518")
        axes[1].set(xlabel="Month", ylabel="Mean rating")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        """,
    )

    _md(
        notebook,
        """
        ### 2.2 Genre EDA

        Genres are multi-label: one movie and one rating can contribute to
        several genre rows, so genre counts are not mutually exclusive shares.
        Genres are not model features. They are used only for post-hoc
        interpretation, genre entropy, and latent-factor profiles.
        """,
    )
    _code(
        notebook,
        """
        genre_summary = aggregate_genre_statistics(data.ratings, data.movies)
        display(genre_summary)
        view = genre_summary.sort_values("rating_count").tail(15)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].barh(view["genre"], view["movie_count"], color="#72b7b2")
        axes[0].set(title="Movies by genre", xlabel="Movie count")
        axes[1].barh(view["genre"], view["rating_count"], color="#4c78a8")
        axes[1].set(title="Ratings by genre", xlabel="Rating count")
        fig.tight_layout()
        plt.show()
        """,
    )

    _md(
        notebook,
        """
        ### 2.3 Demographic EDA

        Demographic attributes are explored only as dataset context. The
        recommender models remain collaborative-only and do not use age, gender,
        or occupation as input features. Labels below use the documented
        MovieLens 1M age-group and occupation mappings.
        """,
    )
    _code(
        notebook,
        """
        demographics = demographic_distributions(data.users)
        display(demographics["gender"])
        display(demographics["age_group"])
        display(demographics["occupation"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        demographics["gender"].plot.bar(
            x="gender", y="user_count", ax=axes[0], legend=False, color="#4c78a8"
        )
        demographics["age_group"].plot.bar(
            x="age_group", y="user_count", ax=axes[1], legend=False, color="#72b7b2"
        )
        demographics["occupation"].head(10).sort_values("user_count").plot.barh(
            x="occupation_label", y="user_count", ax=axes[2],
            legend=False, color="#f58518"
        )
        axes[0].set(title="Gender distribution", xlabel="")
        axes[1].set(title="Age-group distribution", xlabel="")
        axes[2].set(title="Top occupations", ylabel="")
        fig.tight_layout()
        plt.show()
        """,
    )

    _md(
        notebook,
        """
        ## 3. Rating-prediction split

        The pointwise protocol preserves the deterministic 70/15/15
        interaction split. Validation selects hyperparameters and stopping; test
        rows remain untouched until the final train-plus-validation refit.
        """,
    )
    _code(
        notebook,
        """
        display(pd.DataFrame([
            {"split": "train", "rows": len(train)},
            {"split": "validation", "rows": len(validation)},
            {"split": "test", "rows": len(test)},
        ]))
        display(metrics["split"])
        """,
    )

    _md(
        notebook,
        """
        ## 4. Bias baseline

        BiasBaseline is a regularized ablation:
        `global_mean + user_bias + item_bias`. Its regularization is selected
        from validation data rather than hard-coded from a previous run.
        """,
    )
    _code(
        notebook,
        """
        display(pd.DataFrame(bias_tuning["results"]))
        print("Selected:", bias_tuning["selected"])
        print("Test MSE:", metrics["BiasBaseline_MSE"])
        print("Test RMSE:", metrics["BiasBaseline_RMSE"])
        """,
    )

    _md(
        notebook,
        """
        ## 5. Item-kNN neighborhood collaborative filtering

        ItemKNN subtracts the fitted bias baseline, computes sparse item residual
        vectors, applies cosine significance shrinkage, and keeps pairs with at
        least three common users. Prediction uses signed similarities in the
        numerator and absolute similarities in the denominator.
        """,
    )
    _code(
        notebook,
        """
        item_results = pd.DataFrame(item_knn_tuning["results"])
        display(item_results.sort_values(["validation_rmse", "k", "shrinkage"]).head(9))
        print("Selected:", item_knn_tuning["selected"])
        print("Test MSE:", metrics["ItemKNN_MSE"])
        print("Test RMSE:", metrics["ItemKNN_RMSE"])
        print("Neighbor diagnostics:", item_knn_tuning["final_refit"]["diagnostics"])
        """,
    )

    _md(
        notebook,
        """
        ## 6. SVD methodology and tuning

        SVD factorizes a user-mean-centered sparse matrix with a regularized item
        residual bias. Raw predictions are retained for ranking; clipping is used
        only for rating evaluation and display.

        SVD is fitted through a direct truncated decomposition and therefore has
        no epoch-based learning curve. These plots show validation error across
        the tested ranks and document the rank-selection decision.
        """,
    )
    _code(
        notebook,
        """
        display(svd_tuning.sort_values("validation_rmse").head(10))
        print("Selected SVD parameters:", metrics["svd_best_params"])
        display(Image(filename=str(REPORTS / "svd_rank_tuning_rmse.png")))
        display(Image(filename=str(REPORTS / "svd_rank_tuning_mse.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 7. PMF methodology and tuning

        PMF learns user/item biases and latent interactions with seeded SGD. The
        preserved grid searches 96/112/128 factors and factor regularization
        0.05/0.06/0.07 at learning rate 0.006 and bias regularization 0.02.
        """,
    )
    _code(
        notebook,
        """
        display(pmf_tuning.sort_values("validation_rmse"))
        print("Selected PMF parameters:", metrics["pmf_best_params"])
        """,
    )

    _md(
        notebook,
        """
        ## 8. Convergence, regularization and stopping

        PMF convergence is a genuine epoch-based learning curve. Both views show
        train and validation history, the selected epoch, and early-stopping
        context. MSE is derived unambiguously as `RMSE ** 2`.
        """,
    )
    _code(
        notebook,
        """
        display(Image(filename=str(REPORTS / "pmf_convergence.png")))
        display(Image(filename=str(REPORTS / "pmf_convergence_mse.png")))
        display(pd.DataFrame(pmf_convergence["history"]).tail())
        pmf_metadata = load_json(REPORTS / "pmf_factors" / "metadata.json")
        print("Final refit epochs:", pmf_metadata["config"]["epochs"])
        print("Early stopping:", pmf_convergence["early_stopping"])
        print("Search diagnostics:", metrics["pmf_search_diagnostics"])
        """,
    )

    _md(
        notebook,
        """
        ## 9. Rating-prediction results

        All four models use the same untouched rating test rows. MSE and RMSE are
        shown separately rather than sharing one axis.
        """,
    )
    _code(
        notebook,
        """
        rating_table = build_rating_table(metrics)
        display(rating_table)
        display(Image(filename=str(REPORTS / "model_mse_comparison.png")))
        display(Image(filename=str(REPORTS / "rmse_comparison.png")))
        display(Image(filename=str(REPORTS / "predicted_vs_actual.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 10. Temporal leave-one-positive-out protocol

        Each eligible user contributes one known future positive: the latest
        rating at least 4.0, with movie ID ascending as the same-timestamp
        tie-break. Ranking history is strictly `timestamp < target_timestamp`;
        same-timestamp and later interactions are excluded. Users need at least
        20 prior interactions, and targets need at least 10 ranking-training
        interactions.

        Candidate movies are the full ranking-training-supported catalog minus
        the user's prefix history. No sampled negatives are used. Unknown catalog
        items are not observed negatives.
        """,
    )
    _code(
        notebook,
        """
        display(pd.Series(ranking_protocol).to_frame("value"))
        display(ranking_targets.head())
        print("Ranking training rows:", len(ranking_train))
        """,
    )

    _md(
        notebook,
        """
        ## 11. Top-K ranking results

        HitRate/NDCG/MRR measure next-positive recovery for the single held-out
        target. Recall is not reported separately because with one target it is
        identical to HitRate.
        """,
    )
    _code(
        notebook,
        """
        ranking_table = pd.DataFrame(ranking_metrics["models"]).T.reset_index(
            names="model"
        )
        display(ranking_table)
        display(Image(filename=str(REPORTS / "ranking_comparison.png")))
        display(ranking_results.head())
        """,
    )

    _md(
        notebook,
        """
        ### Rating accuracy vs ranking

        The implementation below is reusable project code. It validates model
        coverage and rank-derived metrics, applies deterministic tie handling,
        computes metric reversals and target-rank quantiles, and prepares the
        comparison plot.
        """,
    )
    _code(
        notebook,
        """
        rating_ranking = load_rating_ranking_analysis(REPORTS)
        display(rating_ranking.comparison_table.style.hide(axis="index").format({
            "test_mse": "{:.3f}",
            "test_rmse": "{:.3f}",
            "HitRate@5": "{:.2%}",
            "HitRate@10": "{:.2%}",
            "median_target_rank": "{:.0f}",
            "share_target_rank_gt_2000": "{:.2%}",
        }))
        display(rating_ranking.reversal_table)
        display(rating_ranking.rank_distribution_table.style.hide(axis="index").format({
            **{column: "{:.1f}" for column in [
                "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"
            ]},
            "share_rank_gt_2000": "{:.2%}",
            "per_user_better_share": "{:.1%}",
        }))
        figure = plot_rating_ranking_tradeoff(rating_ranking.plot_data)
        display(figure)
        plt.close(figure)
        display(Markdown(rating_ranking.interpretation))
        """,
    )

    _md(
        notebook,
        """
        ## 12. Global latent-factor interpretation

        High-variance PMF factors are described from movies and genres on both
        poles. Factor sign is arbitrary, so these are descriptive patterns rather
        than objective semantic dimensions.
        """,
    )
    _code(
        notebook,
        """
        display(factor_interpretation.head(20))
        display(factor_genre_profiles.head(20))
        display(Image(filename=str(REPORTS / "pmf_latent_factor_heatmap.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 13. Movie similarity analysis

        Similarity uses cosine distance between final production PMF item-factor
        vectors. Self-matches are excluded and rows are deterministically sorted.
        """,
    )
    _code(
        notebook,
        """
        display(similarities.head(30))
        print(
            "Similarity range:",
            similarities["cosine_similarity"].min(),
            similarities["cosine_similarity"].max(),
        )
        """,
    )

    _md(
        notebook,
        """
        ## 14. User Case Studies

        The 70/15/15 split is performed over each user's interactions, so the
        same user may appear in train, validation, and test. `test_case` means a
        held-out interaction case, not an unseen or cold-start user. Cold start is
        outside scope; user 2210 has training history and is evaluated on a
        held-out future interaction.

        Roles `train_profile_accurate`, `train_profile_less_accurate`, and
        `test_case` are defined by temporal ranking outcome, not per-user RMSE.
        User 2739 is `accurate` because PMF achieves Hit@10; user 2505 is
        `less_accurate` because PMF misses Hit@10. These labels do not imply lower
        PMF per-user RMSE: PMF can be worse than SVD for user 2739, while user
        2505 can have very low SVD RMSE. This is expected because rating
        prediction and ranking answer different questions.
        """,
    )
    _code(
        notebook,
        """
        case_columns = [
            "role", "user_id", "ranking_case", "ranking_target_movie_id",
            "ranking_target_title", "ranking_target_rating",
            "ranking_history_count", "ranking_candidate_count",
            "bias_target_rank", "item_knn_target_rank", "svd_target_rank",
            "pmf_target_rank", "bias_hit_at_10", "item_knn_hit_at_10",
            "svd_hit_at_10", "pmf_hit_at_10", "svd_test_rmse", "pmf_test_rmse",
        ]
        display(evaluated[case_columns])
        """,
    )

    _md(
        notebook,
        """
        ## 15. Recommendation Hit vs Miss Analysis

        The accurate profile is a PMF Hit@10 and the less-accurate profile is a
        PMF miss. Score components and profile summaries may help interpret a
        result, but they do not establish causality.
        """,
    )
    _code(
        notebook,
        """
        hit_miss = evaluated.loc[evaluated["role"].isin([
            "train_profile_accurate", "train_profile_less_accurate"
        ])]
        display(hit_miss)
        for selected in hit_miss.itertuples(index=False):
            user_id = int(selected.user_id)
            case = pd.read_csv(REPORTS / f"user_{user_id}_ranking_case.csv")
            print(f"\\n{selected.role}: user {user_id}")
            display(case)
            display(Image(filename=str(REPORTS / f"user_{user_id}_ranking_case.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 16. Local Recommendation Explanations

        Production Top-10 recommendation explanations remain separate from the
        temporal target cases. Each PMF recommendation score is decomposed into
        global mean, biases, and latent dot product.
        """,
    )
    _code(
        notebook,
        """
        for selected in evaluated.itertuples(index=False):
            user_id = int(selected.user_id)
            explanations = pd.read_csv(
                REPORTS / f"user_{user_id}_explanations.csv"
            )
            print(
                f"User {user_id} max production explanation error:",
                explanations["reconstruction_error"].abs().max(),
            )
            display(explanations.head(3))
            display(Image(filename=str(REPORTS / f"user_{user_id}_explanation.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 17. Streamlit and artifact overview

        Streamlit reads saved artifacts for both rating and ranking evaluation,
        including separate MSE/RMSE comparisons and convergence views.
        """,
    )
    _code(
        notebook,
        """
        artifacts = [
            "model_metrics.json", "model_mse_comparison.png",
            "rmse_comparison.png", "pmf_convergence_mse.png",
            "pmf_convergence.png", "svd_rank_tuning_mse.png",
            "svd_rank_tuning_rmse.png", "ranking_protocol.json",
            "ranking_metrics.json", "ranking_results.csv",
            "ranking_comparison.png", "evaluated_users.json",
        ]
        display(pd.DataFrame({
            "artifact": artifacts,
            "exists": [(REPORTS / name).exists() for name in artifacts],
        }))
        """,
    )

    _md(
        notebook,
        """
        ## 18. Limitations

        The models use collaborative ratings only and do not solve cold start.
        RMSE, typical target rank, and extreme-head retrieval can produce
        different model orderings. The temporal protocol evaluates one known
        future positive and has no observed true negatives. PMF factor
        interpretations are descriptive, and the selected factor count remains
        at the searched boundary.
        """,
    )

    nbf.write(notebook, root / "Movie_Recommender_System.ipynb")


def main() -> None:
    build_notebook(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
