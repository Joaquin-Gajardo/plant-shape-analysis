"""The downloader must not accept an incomplete dataset, nor leave one behind."""

import zipfile

import pytest

from plant_shape_analysis.utils import download_dataset as dd


@pytest.fixture
def fake_release(tmp_path, monkeypatch):
    """Serve a local zip as version "vt", which must contain vt/sparse/."""

    def make(dirs):
        zip_path = tmp_path / "release.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            for d in dirs:
                z.writestr(f"vt/{d}/a.ply", "x")
        return zip_path

    def config(zip_path):
        monkeypatch.setattr(dd, "_get_dataset_config", lambda version: {
            "url": zip_path.as_uri(), "filename": "vt.zip", "extract_dir": "vt",
            "size_mb": 0, "description": "test", "expected_dirs": ["sparse"],
        })

    return lambda dirs: config(make(dirs))


def test_fresh_download(tmp_path, fake_release):
    fake_release(["sparse"])
    path = dd.download_trackplant3d(tmp_path / "data", version="vt", verbose=False)
    assert (path / "sparse" / "a.ply").exists()


def test_incomplete_existing_dir_is_rejected_then_repaired(tmp_path, fake_release):
    fake_release(["sparse"])
    mine = tmp_path / "data" / "vt" / "my_labels"
    mine.mkdir(parents=True)  # the version dir exists, but without "sparse"
    with pytest.raises(RuntimeError, match="incomplete"):
        dd.download_trackplant3d(tmp_path / "data", version="vt", verbose=False)
    path = dd.download_trackplant3d(tmp_path / "data", version="vt", force=True, verbose=False)
    assert (path / "sparse" / "a.ply").exists() and mine.exists()  # user files kept


def test_bad_archive_leaves_nothing_behind(tmp_path, fake_release):
    fake_release(["other"])  # the archive lacks "sparse"
    with pytest.raises(RuntimeError, match="Missing"):
        dd.download_trackplant3d(tmp_path / "data", version="vt", verbose=False)
    assert list((tmp_path / "data").iterdir()) == []  # no version dir, temp dir, or zip
