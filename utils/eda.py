from __future__ import annotations

import pandas as pd


MOVIELENS_AGE_GROUPS = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

MOVIELENS_OCCUPATIONS = {
    0: "other/not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


def aggregate_temporal_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing temporal EDA columns: {sorted(missing)}")
    temporal = ratings[["timestamp", "rating"]].copy()
    temporal["datetime"] = pd.to_datetime(
        temporal["timestamp"], unit="s", utc=True
    ).dt.tz_localize(None)
    temporal["month"] = temporal["datetime"].dt.to_period("M").dt.to_timestamp()
    return (
        temporal.groupby("month", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
        .sort_values("month", kind="mergesort")
        .reset_index(drop=True)
    )


def explode_movie_genres(movies: pd.DataFrame) -> pd.DataFrame:
    required = {"movie_id", "genres"}
    missing = required - set(movies.columns)
    if missing:
        raise ValueError(f"movies missing genre EDA columns: {sorted(missing)}")
    exploded = movies[["movie_id", "genres"]].copy()
    exploded["genre"] = exploded["genres"].astype(str).str.split("|")
    return (
        exploded.explode("genre", ignore_index=True)
        .drop(columns="genres")
        .sort_values(["movie_id", "genre"], kind="mergesort")
        .reset_index(drop=True)
    )


def aggregate_genre_statistics(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    required = {"movie_id", "rating"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing genre EDA columns: {sorted(missing)}")
    exploded = explode_movie_genres(movies)
    movie_counts = exploded.groupby("genre")["movie_id"].nunique()
    rating_genres = ratings[["movie_id", "rating"]].merge(
        exploded, on="movie_id", how="inner", validate="many_to_many"
    )
    rating_summary = rating_genres.groupby("genre").agg(
        rating_count=("rating", "size"),
        mean_rating=("rating", "mean"),
    )
    return (
        rating_summary.join(movie_counts.rename("movie_count"))
        .reset_index()
        [["genre", "movie_count", "rating_count", "mean_rating"]]
        .sort_values(["rating_count", "genre"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def demographic_distributions(
    users: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    required = {"gender", "age", "occupation"}
    missing = required - set(users.columns)
    if missing:
        raise ValueError(
            f"users missing demographic EDA columns: {sorted(missing)}"
        )
    enriched = users.copy()
    enriched["age_group"] = enriched["age"].map(MOVIELENS_AGE_GROUPS)
    enriched["occupation_label"] = enriched["occupation"].map(
        MOVIELENS_OCCUPATIONS
    )
    if enriched["age_group"].isna().any():
        unknown = sorted(enriched.loc[enriched["age_group"].isna(), "age"].unique())
        raise ValueError(f"unknown MovieLens age codes: {unknown}")
    if enriched["occupation_label"].isna().any():
        unknown = sorted(
            enriched.loc[
                enriched["occupation_label"].isna(), "occupation"
            ].unique()
        )
        raise ValueError(f"unknown MovieLens occupation codes: {unknown}")

    def counts(column: str) -> pd.DataFrame:
        return (
            enriched[column]
            .value_counts()
            .rename_axis(column)
            .rename("user_count")
            .reset_index()
        )

    return {
        "gender": counts("gender"),
        "age_group": counts("age_group"),
        "occupation": counts("occupation_label"),
    }
