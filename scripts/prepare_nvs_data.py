"""Acquire and validate the shared, half-resolution T&T SfM input packs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import shutil
import struct
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REVISION = "6a638951a007c8b47eb8a0301d48b04d18e955cf"
BASE = f"https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/{REVISION}/tanksandtemples"
ARCHIVES = {
    "Train": (182371491, "8f255e1b20da44db142e573819e8f3078ab012aaa0e059c2131f290164dddd09"),
    "Truck": (156007999, "20923dadb77abc7ee47c0b16c39f5fbd279466b3555eaec60eff8bc2548f6919"),
    "Francis": (138104721, "70504801033e20474655f43ede0cb989304abfa00c9f7a6a1313a5ce4e0c2dc1"),
    "Horse": (78064897, "ae6cd0b695f9b4e5054fc4fd1e0daab87daab155d3d37958b5d3edae75d46b8f"),
    "Lighthouse": (167718733, "9261e74bf6ec258220f402fd39f67aba2fc23ad622abe7dc77c00335b61b3aa5"),
}
GEOMETRY_SCENES = ("Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def destination_for(member: tarfile.TarInfo, scene: str) -> Path | None:
    """Accept only ordinary images and sparse files under the expected archive root."""
    parts = PurePosixPath(member.name).parts
    if not parts or parts[0] != scene.lower() or ".." in parts or PurePosixPath(member.name).is_absolute():
        raise ValueError(f"Unexpected archive path: {member.name}")
    if member.isdir():
        return None
    if not member.isfile():
        raise ValueError(f"Archive contains a non-regular file: {member.name}")
    relative = Path(*parts[1:])
    if relative.parts[0] == "sparse":
        return Path("sparse/0") / Path(*relative.parts[1:])
    if relative.parts[0] == "images_2":
        return relative
    return None


def validate_scene(root: Path) -> dict:
    specification = importlib.util.spec_from_file_location("input_colmap", ROOT / "2d-gaussian-splatting/scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    cameras = loader.read_intrinsics_binary(root / "sparse/0/cameras.bin")
    poses = loader.read_extrinsics_binary(root / "sparse/0/images.bin")
    names = sorted(pose.name for pose in poses.values())
    for pose in poses.values():
        camera = cameras[pose.camera_id]
        if camera.model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Non-pinhole input camera: {camera.model}")
        with Image.open(root / "images_2" / pose.name) as image:
            image.verify()
        with Image.open(root / "images_2" / pose.name) as image:
            if abs(image.width * 2 - camera.width) > 1 or abs(image.height * 2 - camera.height) > 1:
                raise ValueError(f"Unexpected input resolution at {pose.name}")
    with (root / "sparse/0/points3D.bin").open("rb") as handle:
        count, = struct.unpack("<Q", handle.read(8))
    files = [*root.joinpath("images_2").rglob("*"), *root.joinpath("sparse/0").glob("*.bin")]
    files = [path for path in files if path.is_file()]
    return {
        "registered_images": len(names), "sparse_points": count,
        "train_names": [name for index, name in enumerate(names) if index % 8],
        "test_names": names[::8], "images": "images_2", "resolution": 1,
        "resolution_relative_to_original": 2,
        "files": {str(path.relative_to(root)): sha256(path) for path in sorted(files)},
    }


def staging_directory(output: Path, scene: str) -> Path:
    """Return the scene's fresh staging path; a leftover one means an earlier run was interrupted."""
    staging = output / f".{scene}.partial"
    if staging.exists():
        raise FileExistsError(f"Unvalidated staging directory from an interrupted run: {staging}. "
                              "Delete it and rerun this command.")
    return staging


def prepare(scene: str, output: Path, downloads: Path) -> None:
    destination = output / scene
    record = destination / "input_provenance.json"
    if destination.exists():
        expected = json.loads(record.read_text())
        if validate_scene(destination) != expected["validation"]:
            raise ValueError(f"Existing input pack changed: {destination}")
        logging.info("Verified existing %s", destination)
        return
    staging = staging_directory(output, scene)
    downloads.mkdir(parents=True, exist_ok=True)
    archive_path = downloads / f"{scene.lower()}_2down.tar.gz"
    expected_size, expected_hash = ARCHIVES[scene]
    url = f"{BASE}/{archive_path.name}"
    if not archive_path.exists():
        partial = archive_path.with_suffix(".part")
        logging.info("Downloading %s", url)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1024 * 1024)
        if partial.stat().st_size != expected_size or sha256(partial) != expected_hash:
            raise ValueError(f"Incomplete or corrupt download: {partial}")
        partial.rename(archive_path)
    if archive_path.stat().st_size != expected_size or sha256(archive_path) != expected_hash:
        raise ValueError(f"Archive hash mismatch: {archive_path}")
    staging.mkdir(parents=True)
    with tarfile.open(archive_path) as archive:
        for member in archive:
            relative = destination_for(member, scene)
            if relative is None:
                continue
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
    validation = validate_scene(staging)
    (staging / "input_provenance.json").write_text(json.dumps({
        "source": url, "source_revision": REVISION, "archive_sha256": expected_hash,
        "archive_bytes": expected_size, "initialization": "published_sparse_colmap_not_mvs",
        "validation": validation,
    }, indent=2) + "\n")
    staging.rename(destination)
    logging.info("Prepared %s: %d cameras, %d sparse points", scene, validation["registered_images"], validation["sparse_points"])


def prepare_from_geometry(source: Path, destination: Path) -> None:
    """Preserve the GOF sparse frame while producing the paper's half-size NVS images.

    Frames keep their GOF names, so PIL re-encodes the halved `.jpg` images as JPEG with its
    default settings (quality 75). This lossy re-save is intentional: with the pinned Pillow it
    reproduces the paper's `images_2` files. The pack is built in `.<Scene>.partial` and only
    renamed into place after validation, like the archive packs.
    """
    from scripts.prepare_nvs_budget_sweep import inspect_input

    source = source.resolve()
    record = destination / "input_provenance.json"
    if destination.exists():
        if validate_scene(destination) != json.loads(record.read_text())["validation"]:
            raise ValueError(f"Existing input pack changed: {destination}")
        return
    staging = staging_directory(destination.parent, destination.name)
    evidence = inspect_input({"source_path": str(source), "images": "images", "resolution": 2})
    staging.mkdir(parents=True)
    (staging / "sparse").symlink_to(source / "sparse", target_is_directory=True)
    (staging / "images").symlink_to(source / "images", target_is_directory=True)
    (staging / "images_2").mkdir()
    for item in evidence["images"]:
        with Image.open(source / "images" / item["name"]) as image:
            image.resize((round(image.width / 2), round(image.height / 2)),
                         Image.Resampling.LANCZOS).save(staging / "images_2" / item["name"])
    (staging / "input_provenance.json").write_text(json.dumps({
        "source": str(source), "initialization": "existing_sparse_colmap_not_mvs",
        "resize": "PIL LANCZOS half resolution", "validation": validate_scene(staging)}, indent=2) + "\n")
    staging.rename(destination)
    logging.info("Prepared %s from %s", destination, source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", choices=ARCHIVES, nargs="*", default=list(ARCHIVES),
                        help="Archive scenes to download; pass --scenes with no names for GOF-only preparation")
    parser.add_argument("--output", type=Path, default=ROOT / "data/tnt_nvs")
    parser.add_argument("--downloads", type=Path, default=ROOT / "data/downloads/tnt_nvs")
    parser.add_argument("--geometry-root", type=Path,
                        help="Also prepare the five additional NVS scenes from the downloaded GOF geometry pack")
    args = parser.parse_args()
    if not args.scenes and args.geometry_root is None:
        parser.error("Nothing to prepare: name archive --scenes or pass --geometry-root")
    logging.basicConfig(level=logging.INFO)
    for scene in args.scenes:
        prepare(scene, args.output, args.downloads)
    if args.geometry_root is not None:
        for scene in GEOMETRY_SCENES:
            prepare_from_geometry(args.geometry_root / scene, args.output / scene)


if __name__ == "__main__":
    main()
