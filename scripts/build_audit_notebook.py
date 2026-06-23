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

        This audit-oriented notebook reads the split files, trained model outputs,
        metrics, and interpretability reports produced by `python -m scripts.run_pipeline`.
        It does not rerun SVD/PMF training or tuning.
        """,
    )

    _md(
        notebook,
        """
        ## 1. Project goal

        Compare a local benchmark collaborative-filtering baseline with two
        matrix-factorization models on the same deterministic MovieLens 1M split,
        then inspect both global PMF latent-factor behavior and local recommendation
        explanations for three audit users.
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

        from models.pmf_model import PMFModel
        from utils.data_loader import load_movielens
        from utils.matrix_creation import load_mappings

        ROOT = Path.cwd()
        REPORTS = ROOT / "reports"
        PROCESSED = ROOT / "processed"

        with (REPORTS / "model_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (REPORTS / "baseline_tuning.json").open(encoding="utf-8") as handle:
            baseline_tuning = json.load(handle)
        with (REPORTS / "evaluated_users.json").open(encoding="utf-8") as handle:
            evaluated_users = json.load(handle)

        data = load_movielens(ROOT / "data")
        train = pd.read_csv(PROCESSED / "train_ratings.csv")
        validation = pd.read_csv(PROCESSED / "validation_ratings.csv")
        test = pd.read_csv(PROCESSED / "test_ratings.csv")
        factor_interpretation = pd.read_csv(REPORTS / "pmf_factor_interpretation.csv")
        factor_genre_profiles = pd.read_csv(REPORTS / "pmf_factor_genre_profiles.csv")
        similarities = pd.read_csv(REPORTS / "pmf_movie_similarities.csv")
        user_to_index, movie_to_index, index_to_user, index_to_movie = load_mappings(
            PROCESSED / "mappings"
        )
        pmf = PMFModel.load(REPORTS / "pmf_factors")

        print(f"Ratings: {len(data.ratings):,}")
        print(
            f"Split: train={len(train):,}, validation={len(validation):,}, "
            f"test={len(test):,}"
        )
        print(f"Mapped users={len(user_to_index):,}, mapped movies={len(movie_to_index):,}")
        """,
    )

    _md(
        notebook,
        """
        ## 2. MovieLens EDA and insights

        The compact EDA below focuses on audit-relevant properties: rating skew,
        uneven user activity, long-tail movie popularity, sparsity, and genre frequency.
        These properties motivate regularization and a reproducible user-level split.
        """,
    )

    _code(
        notebook,
        """
        rating_counts = data.ratings["rating"].value_counts().sort_index()
        user_activity = data.ratings.groupby("user_id").size()
        movie_popularity = data.ratings.groupby("movie_id").size()
        movie_genres = data.movies.assign(
            genre=data.movies["genres"].str.split("|")
        ).explode("genre")
        genre_counts = movie_genres["genre"].value_counts().head(15)
        sparsity = 1.0 - len(data.ratings) / (
            data.users["user_id"].nunique() * data.movies["movie_id"].nunique()
        )
        summary = pd.DataFrame(
            [
                {"metric": "ratings", "value": len(data.ratings)},
                {"metric": "users", "value": data.users["user_id"].nunique()},
                {"metric": "movies", "value": data.movies["movie_id"].nunique()},
                {"metric": "matrix_sparsity", "value": sparsity},
                {"metric": "median_user_activity", "value": user_activity.median()},
                {"metric": "median_movie_popularity", "value": movie_popularity.median()},
            ]
        )
        display(summary)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        rating_counts.plot(kind="bar", ax=axes[0, 0], color="#4c78a8")
        axes[0, 0].set(title="Rating distribution", xlabel="Rating", ylabel="Count")
        axes[0, 1].hist(user_activity, bins=50, color="#72b7b2")
        axes[0, 1].set(
            title="User activity distribution",
            xlabel="Ratings per user",
            ylabel="Users",
        )
        axes[1, 0].hist(movie_popularity, bins=60, color="#f58518", log=True)
        axes[1, 0].set(
            title="Movie popularity long tail",
            xlabel="Ratings per movie",
            ylabel="Movies (log scale)",
        )
        genre_counts.sort_values().plot(kind="barh", ax=axes[1, 1], color="#54a24b")
        axes[1, 1].set(title="Top genre frequencies", xlabel="Movie count", ylabel="Genre")
        fig.tight_layout()
        plt.show()
        """,
    )

    _md(
        notebook,
        """
        Ratings are skewed toward higher values: ratings 4 and 5 account for a
        large share of all observations. The user-item matrix is highly sparse,
        movie popularity has a pronounced long tail, and user activity is
        heterogeneous. Low-support movies are harder to model because they provide
        fewer observed residuals or item-factor updates. These data properties make
        regularization and a reproducible user-level split necessary for stable
        evaluation.
        """,
    )

    _md(
        notebook,
        """
        ## 3. Reproducible split

        The project uses a deterministic interaction-level split grouped by user.
        Every mapped user and held-out movie remains covered by training rows, while
        validation and test rows stay disjoint from train and from each other.
        """,
    )

    _code(
        notebook,
        """
        split_table = pd.DataFrame(
            [
                {"split": "train", "rows": len(train)},
                {"split": "validation", "rows": len(validation)},
                {"split": "test", "rows": len(test)},
            ]
        )
        display(split_table)
        print(metrics["split"])
        """,
    )

    _md(
        notebook,
        """
        ## 4. Benchmark collaborative filtering model

        The benchmark is a local regularized bias-only collaborative filtering model,
        not a third-party recommender library. It predicts `global_mean + user_bias +
        item_bias`, learns only from observed ratings, selects regularization on
        validation, then refits once on train plus validation before a single test
        evaluation.
        """,
    )

    _code(
        notebook,
        """
        baseline_results = pd.DataFrame(baseline_tuning["results"])
        display(baseline_results)
        print("Selected baseline:", baseline_tuning["selected"])
        print("Test RMSE:", metrics["Baseline_CF_RMSE"])
        """,
    )

    _md(
        notebook,
        """
        ## 5. SVD methodology and tuning

        The SVD model factorizes a user-mean-centered sparse matrix with a regularized
        item residual bias. Tuning uses validation rows only; final predictions are
        stored raw and unclipped for ranking, with clipping used only for
        evaluation/display.
        """,
    )

    _code(
        notebook,
        """
        svd_tuning = pd.read_json(REPORTS / "svd_tuning.json")
        display(svd_tuning.sort_values("validation_rmse").head(10))
        print("Selected SVD params:", metrics["svd_best_params"])
        """,
    )

    _md(
        notebook,
        """
        ## 6. PMF methodology and tuning

        PMF is a locally implemented biased matrix factorization model trained with
        SGD. The current stable grid is intentionally preserved: factors 96/112/128,
        learning rate 0.006, factor regularization 0.05/0.06/0.07, and bias
        regularization 0.02.
        """,
    )

    _code(
        notebook,
        """
        pmf_tuning = pd.read_json(REPORTS / "pmf_tuning.json")
        display(pmf_tuning.sort_values("validation_rmse"))
        print("Selected PMF params:", metrics["pmf_best_params"])
        """,
    )

    _md(
        notebook,
        """
        ## 7. Convergence, regularization and stopping

        The selected PMF configuration is 128 factors, learning rate 0.006, factor
        regularization 0.06, and bias regularization 0.02. Validation RMSE reaches
        its minimum at selected epoch 53. With patience 8, tuning for that
        configuration stops after 61 epochs. The final refit uses exactly 53 epochs
        on train plus validation and does not use the test set for stopping. The
        selected epoch is not at the 70-epoch boundary, while the selected factor
        count is at the searched factor boundary; that boundary result is disclosed
        as a limitation, not retuned here.
        """,
    )

    _code(
        notebook,
        """
        display(Image(filename=str(REPORTS / "pmf_convergence.png")))
        pmf_metadata = json.loads(
            (REPORTS / "pmf_factors" / "metadata.json").read_text(encoding="utf-8")
        )
        print("Final refit epochs:", pmf_metadata["config"]["epochs"])
        print(
            "Final history validation values:",
            sorted({row["validation_rmse"] for row in pmf_metadata["history"]}),
        )
        print("Search diagnostics:", metrics["pmf_search_diagnostics"])
        """,
    )

    _md(
        notebook,
        """
        ## 8. Test metrics and benchmark comparison

        All three models are evaluated on the same held-out test rows. Both
        matrix-factorization models beat the bias-only benchmark, and PMF also
        improves over SVD.
        """,
    )

    _code(
        notebook,
        """
        metric_table = pd.DataFrame(
            [
                {
                    "model": "Baseline CF",
                    "mse": metrics["Baseline_CF_MSE"],
                    "rmse": metrics["Baseline_CF_RMSE"],
                },
                {"model": "SVD", "mse": metrics["SVD_MSE"], "rmse": metrics["SVD_RMSE"]},
                {"model": "PMF", "mse": metrics["PMF_MSE"], "rmse": metrics["PMF_RMSE"]},
            ]
        )
        display(metric_table)
        print(f"SVD vs baseline improvement: {metrics['SVD_vs_Baseline_improvement_%']:.3f}%")
        print(f"PMF vs baseline improvement: {metrics['PMF_vs_Baseline_improvement_%']:.3f}%")
        print(f"PMF vs SVD improvement: {metrics['PMF_vs_SVD_improvement_%']:.3f}%")
        display(Image(filename=str(REPORTS / "rmse_comparison.png")))
        display(Image(filename=str(REPORTS / "predicted_vs_actual.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 9. Global latent factor interpretation

        The PMF item-factor analysis selects the top five factors by variance across
        item loadings. A factor's sign is arbitrary: positive and negative poles can
        be flipped without changing the model if the corresponding user factor is
        flipped too. The label or theme of a factor is therefore a descriptive
        interpretation from movies and genres on both poles, not ground truth
        semantics.
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
        ## 10. Movie similarity analysis

        Cosine similarity is computed between PMF item-factor vectors. Anchor movies
        are selected deterministically from popular mapped movies while encouraging
        genre diversity. Self-matches are excluded and ties are broken by similarity
        descending, then movie ID ascending.
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
        ## 11. Three audit users

        The split is interaction-level, so the selected users can have train,
        validation, and test rows simultaneously. The two training-profile users are
        selected from supported users near lower and upper quartiles of per-user PMF
        test RMSE. The third user is a separate deterministic test case.
        """,
    )

    _code(
        notebook,
        """
        evaluated = pd.DataFrame(evaluated_users)
        display(evaluated)
        """,
    )

    _md(
        notebook,
        """
        ## 12. Accurate vs less accurate user analysis

        The comparison below uses train-history statistics and held-out test errors.
        It avoids assuming that either user is train-only because the split is
        interaction-level.
        """,
    )

    _code(
        notebook,
        """
        movies = data.movies[["movie_id", "title", "genres"]]
        movie_popularity = data.ratings.groupby("movie_id").size().rename("movie_popularity")

        def genre_entropy(frame):
            genres = (
                frame.merge(movies, on="movie_id", how="left")["genres"]
                .str.split("|")
                .explode()
            )
            shares = genres.value_counts(normalize=True)
            if shares.empty:
                return 0.0
            return float(-(shares * np.log2(shares)).sum())

        def profile_stats(user_id):
            train_user = train.loc[train["user_id"].eq(user_id)].copy()
            test_user = test.loc[test["user_id"].eq(user_id)].copy()
            popularity = train_user.merge(
                movie_popularity, on="movie_id", how="left"
            )["movie_popularity"]
            return {
                "user_id": user_id,
                "train_interactions": len(train_user),
                "test_interactions": len(test_user),
                "train_mean_rating": train_user["rating"].mean(),
                "test_mean_rating": test_user["rating"].mean(),
                "train_high_rating_share": train_user["rating"].ge(4).mean(),
                "test_high_rating_share": test_user["rating"].ge(4).mean(),
                "genre_entropy": genre_entropy(train_user),
                "avg_train_movie_popularity": popularity.mean(),
                "distinct_train_movies": train_user["movie_id"].nunique(),
                "rating_distribution": train_user["rating"].value_counts().sort_index().to_dict(),
            }

        profile_roles = evaluated.loc[
            evaluated["role"].isin(
                ["train_profile_accurate", "train_profile_less_accurate"]
            )
        ]
        profile_table = profile_roles.merge(
            pd.DataFrame(
                [profile_stats(int(user_id)) for user_id in profile_roles["user_id"]]
            ),
            on="user_id",
            how="left",
        )
        display(profile_table)

        for row in profile_roles.itertuples(index=False):
            recs = pd.read_csv(REPORTS / f"user_{int(row.user_id)}_recommendations.csv")
            explanations = pd.read_csv(
                REPORTS / f"user_{int(row.user_id)}_explanations.csv"
            )
            print(f"\\n{row.role}: user {int(row.user_id)}")
            display(
                recs.dropna(subset=["pmf_rank"])
                .sort_values("pmf_rank")
                .head(5)[
                    ["movie_id", "title", "genres", "pmf_ranking_score", "pmf_rank"]
                ]
            )
            display(
                explanations.head(3)[
                    [
                        "recommendation_rank",
                        "title",
                        "raw_pmf_ranking_score",
                        "nearest_known_title",
                        "nearest_known_rating",
                        "nearest_known_similarity",
                        "common_genres",
                    ]
                ]
            )
        """,
    )

    _md(
        notebook,
        """
        User 3233 is the more accurate profile in this deterministic selection: PMF
        test RMSE is 0.6928 versus 0.9401 for user 119. The available profile
        statistics explain part of the difference: user 3233 has more train and test
        support (113/24 versus 75/15), giving the model a broader observed profile
        and a more stable held-out estimate. If genre entropy, high-rating share, or
        popularity statistics do not fully separate the users, the difference should
        be read as observed and only partially explained by these profile summaries
        rather than as a universal rule.
        """,
    )

    _md(
        notebook,
        """
        ## 13. Local recommendation explanations

        Each audit user has a PMF local explanation artifact. The decomposition
        reconstructs the raw PMF ranking score as `global_mean + user_bias +
        item_bias + sum(user_factor[k] * item_factor[k])`. The nearest known liked
        movie is searched in the user's full known history, preferring rating >= 4
        when available.
        """,
    )

    _code(
        notebook,
        """
        for row in evaluated.itertuples(index=False):
            user_id = int(row.user_id)
            explanations = pd.read_csv(REPORTS / f"user_{user_id}_explanations.csv")
            max_error = explanations["reconstruction_error"].abs().max()
            print(f"User {user_id} ({row.role}) max reconstruction error: {max_error:.2e}")
            display(
                explanations.head(5)[
                    [
                        "recommendation_rank",
                        "title",
                        "raw_pmf_ranking_score",
                        "global_mean_contribution",
                        "user_bias_contribution",
                        "item_bias_contribution",
                        "total_latent_dot_product",
                        "nearest_known_title",
                        "nearest_known_similarity",
                    ]
                ]
            )
            display(Image(filename=str(REPORTS / f"user_{user_id}_explanation.png")))
        """,
    )

    _md(
        notebook,
        """
        ## 14. Streamlit/artifact overview

        The Streamlit app reads saved artifacts and exposes four sections:
        recommendations, local explanations, model evaluation, and global latent
        factors. It accepts manual user ID input, reports unknown IDs gracefully, and
        does not train or tune models at app startup.
        """,
    )

    _code(
        notebook,
        """
        artifact_list = [
            "baseline_tuning.json",
            "model_metrics.json",
            "rmse_comparison.png",
            "predicted_vs_actual.png",
            "pmf_convergence.png",
            "pmf_factor_interpretation.csv",
            "pmf_factor_genre_profiles.csv",
            "pmf_latent_factor_heatmap.png",
            "pmf_movie_similarities.csv",
            "evaluated_users.json",
        ]
        artifact_table = pd.DataFrame(
            {
                "artifact": artifact_list,
                "exists": [(REPORTS / name).exists() for name in artifact_list],
            }
        )
        display(artifact_table)
        """,
    )

    _md(
        notebook,
        """
        ## 15. Limitations

        Latent factors are not objectively semantic labels; their interpretation is
        descriptive and depends on movies and genres at both poles. The chosen PMF
        factor count lies at the searched factor boundary, so larger factors might be
        worth investigating in future work, but this audit pass intentionally does
        not retune the stable PMF grid. Interaction-level splitting means selected
        audit users have train and held-out rows, not isolated train-only/test-only
        identities.
        """,
    )

    nbf.write(notebook, root / "Movie_Recommender_System.ipynb")


def main() -> None:
    build_notebook(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
