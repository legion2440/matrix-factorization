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
        from IPython.display import Image, display

        from utils.data_loader import load_movielens

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
          interaction split.
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

        MovieLens 1M is sparse, user activity is heterogeneous, movie popularity
        has a long tail, and ratings are skewed toward higher values. These
        properties motivate regularization and careful held-out evaluation.
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
        numerator and absolute similarities in the denominator. The global
        top-k list is intersected with the user's fitted history; an empty
        intersection falls back exactly to BiasBaseline.
        """,
    )
    _code(
        notebook,
        """
        item_results = pd.DataFrame(item_knn_tuning["results"])
        display(item_results.sort_values(["validation_rmse", "k", "shrinkage"]).head(9))
        print("Selected:", item_knn_tuning["selected"])
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
        """,
    )
    _code(
        notebook,
        """
        display(svd_tuning.sort_values("validation_rmse").head(10))
        print("Selected SVD parameters:", metrics["svd_best_params"])
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

        PMF tuning restores the best validation checkpoint. The final production
        refit uses the selected epoch count on train plus validation without a
        validation holdout. Ranking evaluation trains a separate PMF copy for
        exactly the same frozen epoch count and never tunes against ranking targets.
        """,
    )
    _code(
        notebook,
        """
        display(Image(filename=str(REPORTS / "pmf_convergence.png")))
        pmf_metadata = load_json(REPORTS / "pmf_factors" / "metadata.json")
        print("Final refit epochs:", pmf_metadata["config"]["epochs"])
        print("Search diagnostics:", metrics["pmf_search_diagnostics"])
        """,
    )

    _md(
        notebook,
        """
        ## 9. Rating-prediction results

        All four models use the same untouched rating test rows. The table reports
        pointwise MSE/RMSE only. Pairwise booleans are generated from the actual
        metrics; no comparison outcome is assumed in advance.
        """,
    )
    _code(
        notebook,
        """
        rating_table = pd.DataFrame([
            {"model": "BiasBaseline", "mse": metrics["BiasBaseline_MSE"], "rmse": metrics["BiasBaseline_RMSE"]},
            {"model": "ItemKNN", "mse": metrics["ItemKNN_MSE"], "rmse": metrics["ItemKNN_RMSE"]},
            {"model": "SVD", "mse": metrics["SVD_MSE"], "rmse": metrics["SVD_RMSE"]},
            {"model": "PMF", "mse": metrics["PMF_MSE"], "rmse": metrics["PMF_RMSE"]},
        ])
        display(rating_table)
        display(pd.Series({
            "SVD_beats_ItemKNN": metrics["SVD_beats_ItemKNN"],
            "PMF_beats_ItemKNN": metrics["PMF_beats_ItemKNN"],
            "ItemKNN_beats_BiasBaseline": metrics["ItemKNN_beats_BiasBaseline"],
        }))
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

        The persisted roles remain compatible with previous artifacts, but their
        selection now comes from actual temporal ranking outcomes. Per-user SVD
        and PMF RMSE remain secondary diagnostics.
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

        The accurate profile is a non-extreme PMF Hit@10 near the median hit
        rank. The less-accurate profile is near the median PMF miss rank. The
        comparison below includes target metadata, prefix support, candidate
        counts, all model ranks/Hit@10 values, pointwise RMSE diagnostics, profile
        statistics, and the target-specific PMF explanation.

        Score components and profile summaries may partially explain a result,
        but they do not establish a causal reason for the hit or miss.
        """,
    )
    _code(
        notebook,
        """
        movie_lookup = data.movies[["movie_id", "title", "genres"]]

        def genre_entropy(frame):
            genres = (
                frame.merge(movie_lookup, on="movie_id", how="left")["genres"]
                .str.split("|")
                .explode()
            )
            shares = genres.value_counts(normalize=True)
            return 0.0 if shares.empty else float(-(shares * np.log2(shares)).sum())

        def prefix_stats(user_id):
            prefix = ranking_train.loc[ranking_train["user_id"].eq(user_id)]
            return {
                "user_id": user_id,
                "prefix_interactions": len(prefix),
                "prefix_mean_rating": prefix["rating"].mean(),
                "prefix_positive_share": prefix["rating"].ge(4).mean(),
                "prefix_genre_entropy": genre_entropy(prefix),
            }

        hit_miss = evaluated.loc[evaluated["role"].isin([
            "train_profile_accurate", "train_profile_less_accurate"
        ])]
        display(hit_miss.merge(
            pd.DataFrame([prefix_stats(int(user_id)) for user_id in hit_miss["user_id"]]),
            on="user_id",
            how="left",
        ))

        for selected in hit_miss.itertuples(index=False):
            user_id = int(selected.user_id)
            case = pd.read_csv(REPORTS / f"user_{user_id}_ranking_case.csv")
            print(f"\\n{selected.role}: user {user_id}")
            display(case[[
                "target_movie_id", "target_title", "target_rating",
                "target_timestamp", "prior_history_count", "candidate_count",
                "bias_target_rank", "item_knn_target_rank", "svd_target_rank",
                "pmf_target_rank", "bias_hit_at_10", "item_knn_hit_at_10",
                "svd_hit_at_10", "pmf_hit_at_10",
                "bias_raw_target_score", "item_knn_raw_target_score",
                "svd_raw_target_score", "pmf_raw_target_score",
                "pmf_global_mean_contribution", "pmf_user_bias_contribution",
                "pmf_item_bias_contribution", "pmf_total_latent_dot_product",
                "nearest_known_title", "nearest_known_rating",
                "nearest_known_similarity", "common_genres",
            ]])
            display(Image(filename=str(REPORTS / f"user_{user_id}_ranking_case.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 16. Local Recommendation Explanations

        Production Top-10 recommendation explanations remain separate from the
        temporal target cases. Each PMF recommendation score is decomposed into
        global mean, biases, and latent dot product, with nearest known liked
        movies drawn from the production user's full observed history.
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

        Streamlit reads saved production recommendation artifacts and the two
        evaluation protocols. The evaluation-profile selectbox synchronizes a
        keyed manual user input through Session State; manual edits remain valid.
        """,
    )
    _code(
        notebook,
        """
        artifacts = [
            "bias_baseline_tuning.json", "item_knn_tuning.json",
            "model_metrics.json", "rmse_comparison.png",
            "ranking_protocol.json", "ranking_metrics.json",
            "ranking_results.csv", "ranking_comparison.png",
            "evaluated_users.json", "pmf_factor_interpretation.csv",
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
        RMSE and ranking metrics answer different questions. The temporal protocol
        evaluates one known future positive and has no observed true negatives,
        so unseen candidates cannot be interpreted as irrelevant. PMF factor
        interpretations are descriptive, and the selected factor count remains at
        the searched boundary.
        """,
    )

    nbf.write(notebook, root / "Movie_Recommender_System.ipynb")


def main() -> None:
    build_notebook(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
