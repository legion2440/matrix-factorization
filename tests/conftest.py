from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_ratings() -> pd.DataFrame:
    rows = []
    timestamp = 1
    for user_id in range(1, 5):
        for movie_id in range(1, 11):
            rows.append(
                {
                    "user_id": user_id,
                    "movie_id": movie_id,
                    "rating": float(1 + (user_id + movie_id) % 5),
                    "timestamp": timestamp,
                }
            )
            timestamp += 1
    return pd.DataFrame(rows)

