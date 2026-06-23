from __future__ import annotations

from utils.data_loader import load_movielens, validate_movielens


def test_movielens_parsing(tmp_path):
    (tmp_path / "ratings.dat").write_text("1::10::5::100\n", encoding="latin-1")
    (tmp_path / "movies.dat").write_text(
        "10::Example Movie (2000)::Drama\n", encoding="latin-1"
    )
    (tmp_path / "users.dat").write_text("1::F::25::3::12345\n", encoding="latin-1")
    data = load_movielens(tmp_path)
    summary = validate_movielens(data)
    assert data.ratings.iloc[0].to_dict()["rating"] == 5.0
    assert data.movies.iloc[0]["title"] == "Example Movie (2000)"
    assert summary["n_ratings"] == 1

