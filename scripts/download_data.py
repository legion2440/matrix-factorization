from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REQUIRED_FILES = ("ratings.dat", "users.dat", "movies.dat")


def _raw_files_complete(data_dir: Path) -> bool:
    return all(
        (data_dir / filename).is_file()
        and (data_dir / filename).stat().st_size > 0
        for filename in REQUIRED_FILES
    )


def download_movielens(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if _raw_files_complete(data_dir):
        print("MovieLens 1M raw files already exist.")
        return

    archive = data_dir / "ml-1m.zip"
    print(f"Downloading {DATA_URL}")
    try:
        urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as bundle:
            for filename in REQUIRED_FILES:
                member = f"ml-1m/{filename}"
                info = bundle.getinfo(member)
                if info.file_size <= 0:
                    raise ValueError(f"Downloaded MovieLens file is empty: {member}")
                temporary = data_dir / f".{filename}.tmp"
                try:
                    with bundle.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    if temporary.stat().st_size <= 0:
                        raise ValueError(
                            f"Extracted MovieLens file is empty: {filename}"
                        )
                    temporary.replace(data_dir / filename)
                finally:
                    temporary.unlink(missing_ok=True)
    finally:
        archive.unlink(missing_ok=True)
    if not _raw_files_complete(data_dir):
        raise RuntimeError("MovieLens download did not produce all required raw files")
    print("MovieLens 1M download complete.")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    download_movielens(root / "data")


if __name__ == "__main__":
    main()
