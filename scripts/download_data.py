from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REQUIRED_FILES = ("ratings.dat", "users.dat", "movies.dat")


def download_movielens(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / filename).exists() for filename in REQUIRED_FILES):
        print("MovieLens 1M raw files already exist.")
        return

    archive = data_dir / "ml-1m.zip"
    print(f"Downloading {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        for filename in REQUIRED_FILES:
            member = f"ml-1m/{filename}"
            with bundle.open(member) as source, (data_dir / filename).open("wb") as target:
                shutil.copyfileobj(source, target)
    archive.unlink(missing_ok=True)
    print("MovieLens 1M download complete.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    download_movielens(root / "data")


if __name__ == "__main__":
    main()

