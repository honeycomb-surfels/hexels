"""Input-pack boundaries and the published half-resolution image convention."""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
import tarfile

from PIL import Image
import pytest

from scripts import prepare_nvs_data
from scripts.prepare_nvs_data import destination_for, prepare, prepare_from_geometry


@pytest.mark.parametrize("name", ["train/../../outside", "/train/images/a.png", "truck/images_2/a.png"])
def test_archive_paths_stay_in_the_requested_scene(name: str) -> None:
    with pytest.raises(ValueError, match="archive path"):
        destination_for(tarfile.TarInfo(name), "Train")


def test_archive_links_are_not_extracted() -> None:
    member = tarfile.TarInfo("train/images_2/link")
    member.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="non-regular"):
        destination_for(member, "Train")


def test_geometry_input_keeps_sparse_frame_and_half_size_images(tmp_path: Path, monkeypatch) -> None:
    source, destination = tmp_path / "geometry/Barn", tmp_path / "nvs/Barn"
    (source / "images").mkdir(parents=True)
    (source / "sparse/0").mkdir(parents=True)
    Image.new("RGB", (9, 7), color=(20, 40, 80)).save(source / "images/frame.png")
    (source / "sparse/0/cameras.bin").write_bytes(b"original sparse frame")
    monkeypatch.setattr("scripts.prepare_nvs_budget_sweep.inspect_input", lambda settings: {"images": [{"name": "frame.png"}]})
    monkeypatch.setattr("scripts.prepare_nvs_data.validate_scene", lambda folder: {"registered_images": 1})
    prepare_from_geometry(source, destination)
    assert (destination / "sparse").resolve() == (source / "sparse").resolve()
    assert (destination / "sparse/0/cameras.bin").read_bytes() == b"original sparse frame"
    with Image.open(destination / "images_2/frame.png") as image:
        assert image.size == (4, 4)
        assert image.getpixel((0, 0)) == (20, 40, 80)
    record = json.loads((destination / "input_provenance.json").read_text())
    assert record["initialization"] == "existing_sparse_colmap_not_mvs"
    assert record["validation"] == {"registered_images": 1}
    assert not (tmp_path / "nvs/.Barn.partial").exists()


def test_geometry_input_is_staged_and_an_interrupted_run_is_actionable(tmp_path: Path, monkeypatch) -> None:
    source, destination = tmp_path / "geometry/Barn", tmp_path / "nvs/Barn"
    (source / "images").mkdir(parents=True)
    (source / "sparse/0").mkdir(parents=True)
    Image.new("RGB", (9, 7), color=(20, 40, 80)).save(source / "images/frame.jpg")
    monkeypatch.setattr("scripts.prepare_nvs_budget_sweep.inspect_input", lambda settings: {"images": [{"name": "frame.jpg"}]})

    def interrupted(folder: Path) -> dict:
        raise RuntimeError("interrupted")

    monkeypatch.setattr("scripts.prepare_nvs_data.validate_scene", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_from_geometry(source, destination)
    staging = tmp_path / "nvs/.Barn.partial"
    assert staging.is_dir() and not destination.exists()
    with pytest.raises(FileExistsError, match="Delete it and rerun"):
        prepare_from_geometry(source, destination)
    shutil.rmtree(staging)
    assert (source / "images/frame.jpg").exists()
    monkeypatch.setattr("scripts.prepare_nvs_data.validate_scene", lambda folder: {"registered_images": 1})
    prepare_from_geometry(source, destination)
    assert not staging.exists()
    with Image.open(source / "images/frame.jpg") as image:
        expected = io.BytesIO()
        image.resize((4, 4), Image.Resampling.LANCZOS).save(expected, format="JPEG")
    assert (destination / "images_2/frame.jpg").read_bytes() == expected.getvalue()


def test_archive_preparation_reports_leftover_staging_before_downloading(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "nvs/.Train.partial").mkdir(parents=True)

    def no_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("downloaded despite a leftover staging directory")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    with pytest.raises(FileExistsError, match=r"\.Train\.partial\. Delete it and rerun"):
        prepare("Train", tmp_path / "nvs", tmp_path / "downloads")
    assert not (tmp_path / "downloads").exists()


def test_gof_only_preparation_skips_the_archive_downloads(tmp_path: Path, monkeypatch) -> None:
    prepared = []

    def no_archive(*args: object) -> None:
        raise AssertionError("downloaded an archive scene")

    monkeypatch.setattr(prepare_nvs_data, "prepare", no_archive)
    monkeypatch.setattr(prepare_nvs_data, "prepare_from_geometry",
                        lambda source, destination: prepared.append((source.name, destination.name)))
    monkeypatch.setattr(sys, "argv", ["prepare_nvs_data", "--scenes", "--geometry-root", str(tmp_path / "gof"),
                                      "--output", str(tmp_path / "nvs")])
    prepare_nvs_data.main()
    assert prepared == [(scene, scene) for scene in prepare_nvs_data.GEOMETRY_SCENES]
    monkeypatch.setattr(sys, "argv", ["prepare_nvs_data", "--scenes"])
    with pytest.raises(SystemExit):
        prepare_nvs_data.main()
