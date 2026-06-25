from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from scripts.download_data import REQUIRED_FILES, download_movielens
from scripts.validate_project import (
    _validate_mse_rmse_pair,
    _validate_user_artifact_manifest,
)
from utils.artifacts import (
    REQUIRED_USER_ARTIFACT_SUFFIXES,
    cleanup_user_artifacts,
)


def _write_user_artifacts(reports_dir: Path, user_id: int) -> None:
    for suffix in REQUIRED_USER_ARTIFACT_SUFFIXES:
        (reports_dir / f"user_{user_id}_{suffix}").write_bytes(b"x")


def test_cleanup_user_artifacts_removes_only_orphans(tmp_path):
    _write_user_artifacts(tmp_path, 2739)
    _write_user_artifacts(tmp_path, 40)
    unrelated = tmp_path / "model_metrics.json"
    unrelated.write_text("{}", encoding="utf-8")

    removed = cleanup_user_artifacts(tmp_path, {2739})

    assert {path.name for path in removed} == {
        f"user_40_{suffix}" for suffix in REQUIRED_USER_ARTIFACT_SUFFIXES
    }
    assert all(
        (tmp_path / f"user_2739_{suffix}").exists()
        for suffix in REQUIRED_USER_ARTIFACT_SUFFIXES
    )
    assert unrelated.exists()


def test_user_artifact_validator_rejects_orphan(tmp_path):
    _write_user_artifacts(tmp_path, 2739)
    (tmp_path / "user_40_recommendations.csv").write_text(
        "movie_id\n1\n", encoding="utf-8"
    )

    errors = _validate_user_artifact_manifest(
        tmp_path, [{"user_id": 2739}]
    )

    assert any("Orphan user artifact" in error for error in errors)
    assert any("do not match evaluated_users.json" in error for error in errors)


def test_mse_rmse_consistency_validator():
    assert _validate_mse_rmse_pair(0.81, 0.9, "example") == []
    assert _validate_mse_rmse_pair(0.82, 0.9, "example") == [
        "example MSE is inconsistent with RMSE ** 2"
    ]


def _fake_download(url: str, archive: Path) -> tuple[str, object]:
    del url
    with zipfile.ZipFile(archive, "w") as bundle:
        for filename in REQUIRED_FILES:
            bundle.writestr(f"ml-1m/{filename}", f"{filename}\n")
    return str(archive), None


def test_downloader_skips_existing_complete_raw_files(tmp_path, monkeypatch):
    for filename in REQUIRED_FILES:
        (tmp_path / filename).write_text(filename, encoding="utf-8")

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called")

    monkeypatch.setattr("scripts.download_data.urllib.request.urlretrieve", fail_download)

    download_movielens(tmp_path)


def test_downloader_repairs_missing_or_incomplete_files(tmp_path, monkeypatch):
    (tmp_path / "ratings.dat").write_text("old", encoding="utf-8")
    (tmp_path / "users.dat").write_bytes(b"")
    monkeypatch.setattr(
        "scripts.download_data.urllib.request.urlretrieve", _fake_download
    )

    download_movielens(tmp_path)

    assert all((tmp_path / filename).stat().st_size > 0 for filename in REQUIRED_FILES)
    assert json.dumps(
        [path.name for path in tmp_path.glob(".*.tmp")]
    ) == "[]"
    assert not (tmp_path / "ml-1m.zip").exists()
    assert np.all(
        [
            (tmp_path / filename).read_text(encoding="utf-8").strip()
            == filename
            for filename in REQUIRED_FILES
        ]
    )
