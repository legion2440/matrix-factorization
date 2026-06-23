from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from models.pmf_model import PMFModel
from utils.data_loader import load_movielens
from utils.matrix_creation import load_mappings


REQUIRED_PATHS = [
    "data/ratings.dat",
    "data/users.dat",
    "data/movies.dat",
    "processed/train_ratings.csv",
    "processed/validation_ratings.csv",
    "processed/test_ratings.csv",
    "processed/user_item_matrix.csv",
    "processed/mappings/user_to_index.json",
    "processed/mappings/movie_to_index.json",
    "processed/mappings/index_to_user.json",
    "processed/mappings/index_to_movie.json",
    "reports/model_metrics.json",
    "reports/svd_predictions.npy",
    "reports/pmf_convergence.png",
    "reports/predicted_vs_actual.png",
    "reports/rmse_comparison.png",
    "reports/user_comparison.png",
    "reports/top_recommendations.png",
    "reports/evaluated_users.json",
    "reports/pmf_factors/user_factors.npy",
    "reports/pmf_factors/item_factors.npy",
    "reports/pmf_factors/user_bias.npy",
    "reports/pmf_factors/item_bias.npy",
    "reports/pmf_factors/metadata.json",
    "Movie_Recommender_System.ipynb",
    "README.md",
    "requirements.txt",
    "app.py",
]


def validate() -> list[str]:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            errors.append(f"Missing required file: {relative}")
    if errors:
        return errors

    try:
        with (root / "reports" / "model_metrics.json").open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        with (root / "reports" / "evaluated_users.json").open(encoding="utf-8") as handle:
            evaluated_users = json.load(handle)
        with (root / "reports" / "pmf_factors" / "metadata.json").open(
            encoding="utf-8"
        ) as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Invalid JSON artifact: {exc}")
        return errors

    required_metric_keys = {
        "SVD_MSE",
        "SVD_RMSE",
        "PMF_MSE",
        "PMF_RMSE",
        "PMF_vs_SVD_improvement_%",
        "svd_best_params",
        "pmf_best_params",
    }
    if required_metric_keys - set(metrics):
        errors.append("model_metrics.json has an incomplete schema")
    if metrics.get("SVD_RMSE", np.inf) > 0.90:
        errors.append("SVD RMSE target not met")
    if metrics.get("PMF_RMSE", np.inf) > 0.85:
        errors.append("PMF RMSE target not met")
    if metrics.get("PMF_vs_SVD_improvement_%", -np.inf) < 5.0:
        errors.append("PMF improvement target not met")

    user_to_index, movie_to_index, index_to_user, index_to_movie = load_mappings(
        root / "processed" / "mappings"
    )
    predictions = np.load(root / "reports" / "svd_predictions.npy", mmap_mode="r")
    expected_shape = (len(user_to_index), len(movie_to_index))
    if predictions.shape != expected_shape:
        errors.append(
            f"SVD prediction shape {predictions.shape} != expected {expected_shape}"
        )
    if not np.isfinite(predictions).all():
        errors.append("SVD predictions contain non-finite values")
    if predictions.min() < 1.0 or predictions.max() > 5.0:
        errors.append("SVD predictions are outside [1, 5]")

    try:
        pmf = PMFModel.load(root / "reports" / "pmf_factors")
        if pmf.user_factors.shape[0] != len(index_to_user):
            errors.append("PMF user factors do not align with mappings")
        if pmf.item_factors.shape[0] != len(index_to_movie):
            errors.append("PMF item factors do not align with mappings")
        test = pd.read_csv(root / "processed" / "test_ratings.csv")
        test_users = test["user_id"].map(user_to_index).to_numpy()
        test_movies = test["movie_id"].map(movie_to_index).to_numpy()
        if pd.isna(test_users).any() or pd.isna(test_movies).any():
            errors.append("Test rows are not fully covered by mappings")
        else:
            pmf_predictions = pmf.predict_pairs(
                test_users.astype(np.int32), test_movies.astype(np.int32)
            )
            if not np.isfinite(pmf_predictions).all():
                errors.append("PMF predictions contain non-finite values")
            if pmf_predictions.min() < 1.0 or pmf_predictions.max() > 5.0:
                errors.append("PMF predictions are outside [1, 5]")
    except Exception as exc:
        errors.append(f"PMF artifact validation failed: {exc}")

    data = load_movielens(root / "data")
    for selection in evaluated_users:
        user_id = int(selection["user_id"])
        recommendation_path = root / "reports" / f"user_{user_id}_recommendations.csv"
        if not recommendation_path.exists():
            errors.append(f"Missing recommendation CSV for user {user_id}")
            continue
        recommendations = pd.read_csv(recommendation_path)
        if recommendations.empty:
            errors.append(f"Recommendation CSV for user {user_id} is empty")
        seen = set(
            data.ratings.loc[data.ratings["user_id"].eq(user_id), "movie_id"].astype(int)
        )
        leaked = seen & set(recommendations["movie_id"].dropna().astype(int))
        if leaked:
            errors.append(
                f"Recommendations for user {user_id} contain rated movies: {sorted(leaked)}"
            )

    try:
        notebook = nbformat.read(
            root / "Movie_Recommender_System.ipynb", as_version=4
        )
        notebook_errors = [
            output
            for cell in notebook.cells
            if cell.cell_type == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if notebook_errors:
            errors.append("Notebook contains error outputs")
    except Exception as exc:
        errors.append(f"Notebook validation failed: {exc}")

    try:
        importlib.import_module("app")
    except Exception as exc:
        errors.append(f"Streamlit app import failed: {exc}")
    return errors


def main() -> None:
    errors = validate()
    if errors:
        print("Project validation failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Project validation passed.")


if __name__ == "__main__":
    main()

