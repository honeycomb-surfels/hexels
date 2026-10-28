"""Camera, train-only initialization, and portable binary contract checks."""

from __future__ import annotations

import importlib.util
import io
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image
import pytest

MODULE = Path(__file__).resolve().parents[1] / "hexel_edge/tools/edge_bundle.py"
SPEC = importlib.util.spec_from_file_location("edge_bundle", MODULE)
edge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = edge
SPEC.loader.exec_module(edge)


def test_split_is_reserved_from_full_camera_list_before_subsetting() -> None:
    names = [f"{i:03d}.png" for i in range(49)]
    train, test = edge.select_views(list(reversed(names)), 24, 6)
    assert not set(train) & set(test)
    assert all(int(Path(name).stem) % 8 != 0 for name in train)
    assert all(int(Path(name).stem) % 8 == 0 for name in test)
    assert len(train) == 24 and len(test) == 6
    assert edge.select_views(names, 24, 6) == (train, test)


def test_half_pixel_intrinsics_preserve_projected_resampling_coordinates() -> None:
    k = np.array([[501., 0, 312.3], [0, 499.5, 239.2], [0, 0, 1]])
    resized = edge.resize_intrinsics(k, (640, 480), (317, 239))
    point = np.array([.3, -.1, 2.])
    old = k @ point
    old = old[:2] / old[2]
    new = resized @ point
    np.testing.assert_allclose(new[:2]/new[2], (old+.5)*[317/640, 239/480]-.5)


def test_projection_decomposition_round_trips_world_points() -> None:
    k = np.array([[700., 0, 310.], [0, 720., 239.], [0, 0, 1.]])
    rotation = edge.quaternion_matrix(np.array([.9, .1, -.3, .2]))
    translation = np.array([.1, -.3, 3.])
    p = k @ np.column_stack((rotation, translation))
    for scalar in (2.5, -3.):
        recovered, c2w = edge.decompose_projection(p*scalar)
        np.testing.assert_allclose(recovered, k, atol=1e-9)
        np.testing.assert_allclose(recovered @ np.linalg.inv(c2w)[:3], p, atol=1e-9)
        assert np.linalg.det(c2w[:3, :3]) == pytest.approx(1.)
    skewed = k.copy()
    skewed[0, 1] = 2.
    with pytest.raises(ValueError, match="Skewed"):
        edge.decompose_projection(skewed @ np.column_stack((rotation, translation)))


def test_antiparallel_and_general_normals_have_valid_wxyz_rotations() -> None:
    normals = np.array([[0, 0, -1], [1, 2, 3], [0, 0, 1.]])
    rotations = edge.rotations_from_normals(normals)
    np.testing.assert_allclose(np.linalg.norm(rotations, axis=1), 1.)
    for q, n in zip(rotations, normals, strict=True):
        np.testing.assert_allclose(edge.quaternion_matrix(q)[:, 2], n/np.linalg.norm(n), atol=1e-8)


def test_rgba_and_mask_are_composited_without_transparent_color_bleed(tmp_path: Path) -> None:
    image = np.array([[[255, 0, 0, 0], [0, 128, 0, 255]]], dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(np.array([[255, 0]], dtype=np.uint8)).save(tmp_path / "mask.png")
    frame = edge.Frame("image.png", tmp_path / "image.png", 2, 1, np.eye(3), np.eye(4), tmp_path / "mask.png")
    pixels, alpha, _ = edge.prepare_image(frame, 1, np.ones(3))
    np.testing.assert_array_equal(pixels, np.full((1, 1, 3), 255, dtype=np.uint8))
    assert not alpha.any()


def _write_colmap(root: Path, binary: bool) -> None:
    root.mkdir()
    model = root / "sparse/0"
    model.mkdir(parents=True)
    (root / "images").mkdir()
    Image.new("RGB", (64, 48), color=(10, 30, 50)).save(root / "images/frame.png")
    if binary:
        (model / "cameras.bin").write_bytes(struct.pack("<QiiQQ4d", 1, 1, 1, 64, 48, 50, 52, 31, 23))
        data = struct.pack("<Qi7di", 1, 7, 1, 0, 0, 0, -1, -2, -3, 1)
        data += b"frame.png\0" + struct.pack("<Qddq", 1, 12., 14., 9)
        (model / "images.bin").write_bytes(data)
    else:
        (model / "cameras.txt").write_text("1 PINHOLE 64 48 50 52 31 23\n")
        (model / "images.txt").write_text("7 1 0 0 0 -1 -2 -3 1 frame.png\n12 14 9\n")


def test_colmap_binary_and_text_agree_on_camera_direction_and_tracks(tmp_path: Path) -> None:
    _write_colmap(tmp_path / "text", False)
    _write_colmap(tmp_path / "binary", True)
    text = edge.load_colmap(tmp_path / "text")[0]
    binary = edge.load_colmap(tmp_path / "binary")[0]
    np.testing.assert_array_equal(text.camera_to_world, binary.camera_to_world)
    np.testing.assert_array_equal(text.camera_to_world[:3, 3], [1, 2, 3])
    np.testing.assert_array_equal(text.intrinsics, binary.intrinsics)
    np.testing.assert_array_equal(text.observations[9], [11.5, 13.5])
    np.testing.assert_array_equal(text.intrinsics[:2, 2], [30.5, 22.5])
    assert binary.name == "frame.png"


def test_disconnected_colmap_models_are_selected_explicitly_never_merged(tmp_path: Path) -> None:
    _write_colmap(tmp_path / "source", False)
    second = tmp_path / "source/sparse/1"
    second.mkdir()
    (second / "cameras.txt").write_text("1 PINHOLE 64 48 50 52 31 23\n")
    (second / "images.txt").write_text("8 1 0 0 0 0 0 -4 1 other.png\n\n")
    default = edge.load_colmap(tmp_path / "source")
    explicit = edge.load_colmap(tmp_path / "source", model_path="sparse/1")
    assert [frame.name for frame in default] == ["frame.png"]
    assert [frame.name for frame in explicit] == ["other.png"]
    np.testing.assert_array_equal(explicit[0].camera_to_world[:3, 3], [0, 0, 4])
    with pytest.raises(ValueError, match="escapes"):
        edge.load_colmap(tmp_path / "source", model_path="../../other_model")


def test_unsupported_distortion_and_unsafe_asset_paths_are_rejected(tmp_path: Path) -> None:
    _write_colmap(tmp_path / "source", False)
    model = tmp_path / "source/sparse/0"
    (model / "cameras.txt").write_text("1 OPENCV 64 48 50 52 31 23 0 0 0 0\n")
    with pytest.raises(ValueError, match="Unsupported COLMAP"):
        edge.load_colmap(tmp_path / "source")
    for name in ("../../escape.png", "/etc/passwd"):
        with pytest.raises(ValueError, match="escapes"):
            edge.safe_child(tmp_path, name)


def test_triangulation_uses_training_observations_and_ignores_missing_tracks() -> None:
    points = np.array([[-.3, -.2, 2], [.3, -.2, 2], [-.3, .2, 2], [.3, .2, 2]])
    frames = []
    k = np.array([[50., 0, 32], [0, 50., 24], [0, 0, 1.]])
    for x in (-.2, .2):
        pose = np.eye(4)
        pose[0, 3] = x
        frame = edge.Frame(str(x), Path("unused"), 64, 48, k, pose)
        xy, _ = edge.project(points, frame)
        frame.observations = dict(enumerate(xy))
        frames.append(frame)
    frames[0].observations[999] = np.array([0, 0])
    actual = edge.triangulate_training(frames, 50)
    np.testing.assert_allclose(actual, points, atol=1e-10)


def test_triangulation_rejects_single_camera_pure_rotation_and_parallel_rays() -> None:
    k = np.array([[50., 0, 32], [0, 50., 24], [0, 0, 1.]])
    one = edge.Frame("one", Path("unused"), 64, 48, k, np.eye(4),
                     observations={i: np.array([20.+i, 24.]) for i in range(4)})
    with pytest.raises(ValueError, match="two training cameras"):
        edge.triangulate_training([one], 10)
    other = edge.Frame("other", Path("unused"), 64, 48, k, np.eye(4),
                       observations=one.observations)
    with pytest.raises(ValueError, match="translational baseline"):
        edge.triangulate_training([one, other], 10)
    other.camera_to_world = np.eye(4)
    other.camera_to_world[0, 3] = .5
    with pytest.raises(ValueError, match="four valid seeds"):
        edge.triangulate_training([one, other], 10)


def test_unconstrained_pca_normals_use_a_finite_camera_facing_prior() -> None:
    points = np.column_stack((np.linspace(-1, 1, 8), np.zeros((8, 2))))
    normals, scales = edge.point_geometry(points, np.array([[0., 0., 3.]]))
    expected = np.array([0., 0., 3.])-points
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(normals, expected, atol=1e-12)
    assert np.isfinite(scales).all() and (scales > 0).all()
    with pytest.raises(ValueError, match="at least four"):
        edge.point_geometry(points[:1], np.array([[0., 0., 3.]]))


def test_dtu_normalization_and_source_index_keep_undownloaded_views_in_split(tmp_path: Path) -> None:
    (tmp_path / "image").mkdir()
    Image.new("RGB", (64, 48)).save(tmp_path / "image/000000.png")
    k = np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1.]])
    p = np.eye(4)
    p[:3] = k @ np.column_stack((np.eye(3), [0, 0, 3]))
    scale = np.diag([2., 2., 2., 1])
    np.savez(tmp_path / "cameras.npz", world_mat_0=p, scale_mat_0=scale,
             world_mat_1=p, scale_mat_1=scale)
    (tmp_path / "source_index.json").write_text(json.dumps({"image_names": ["000000.png", "000001.png"]}))
    frames = edge.load_dtu(tmp_path)
    assert len(frames) == 2 and frames[1].width == 0
    np.testing.assert_allclose(frames[0].camera_to_world[:3, 3], [0, 0, -1.5])


def test_visual_hull_is_bounded_finite_and_changes_with_training_silhouette() -> None:
    k = np.array([[30., 0, 31.5], [0, 30, 31.5], [0, 0, 1.]])
    pose = np.eye(4)
    pose[2, 3] = -3
    frame = edge.Frame("train", Path("unused"), 64, 64, k, pose)
    y, x = np.mgrid[:64, :64]
    mask = ((x-31.5)**2+(y-31.5)**2 < 10**2).astype(float)
    points, normals, scale = edge.visual_hull([frame], [mask], 20, 16)
    assert len(points) == 20 and scale > 0
    assert np.max(abs(points)) <= 1 and np.isfinite(normals).all()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.)
    with pytest.raises(ValueError, match="empty"):
        edge.visual_hull([frame], [np.zeros_like(mask)], 20, 16)


def test_synthetic_contract_hashes_split_and_binary_are_repeatable(tmp_path: Path) -> None:
    first = edge.synthetic(tmp_path / "a", long_edge=32, seed=7, steps=72)
    second = edge.synthetic(tmp_path / "b", long_edge=32, seed=7, steps=72)
    assert first == second
    assert first["schema_version"] == 1 and first["parameter_layout"] == edge.LAYOUT
    assert sum(f["split"] == "train" for f in first["frames"]) == 24
    assert sum(f["split"] == "test" for f in first["frames"]) == 6
    params = np.fromfile(tmp_path / "a/parameters.f32", dtype="<f4").reshape(-1, 20)
    assert params.shape == (63, 20) and np.isfinite(params).all()
    np.testing.assert_allclose(np.linalg.norm(params[:, 5:9], axis=1), 1.)
    order = np.fromfile(tmp_path / "a/view_order.u32", dtype="<u4")
    assert len(order) == 72 and all(first["frames"][i]["split"] == "train" for i in order)
    for relative, expected in first["sha256"].items():
        assert edge.sha256(tmp_path / "a" / relative) == expected
    with pytest.raises(FileExistsError):
        edge.synthetic(tmp_path / "a", long_edge=32)


def test_preparation_never_reads_heldout_pixels_or_masks_for_seeds(tmp_path: Path) -> None:
    source = tmp_path / "dtu"
    (source / "image").mkdir(parents=True)
    (source / "mask").mkdir()
    k = np.array([[30., 0, 31.5], [0, 30, 31.5], [0, 0, 1.]])
    p = np.eye(4)
    p[:3] = k @ np.column_stack((np.eye(3), [0., 0., 3.]))
    cameras = {}
    y, x = np.mgrid[:64, :64]
    mask = (((x-31.5)**2+(y-31.5)**2 < 10**2)*255).astype(np.uint8)
    for i in range(16):
        Image.new("RGB", (64, 64), color=(20, 80, 150)).save(source / f"image/{i:06d}.png")
        # Different stem padding is intentional: the public mirror uses this.
        Image.fromarray(mask).save(source / f"mask/{i:03d}.png")
        cameras[f"world_mat_{i}"] = p
        cameras[f"scale_mat_{i}"] = np.eye(4)
    np.savez(source / "cameras.npz", **cameras)
    kwargs = dict(kind="dtu", train_count=4, test_count=2, long_edge=32,
                  max_points=32, hull_resolution=16, steps=16)
    first = edge.prepare(source, tmp_path / "first", **kwargs)
    for i in (0, 8):
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(source / f"image/{i:06d}.png")
        Image.new("L", (64, 64), color=0).save(source / f"mask/{i:03d}.png")
    second = edge.prepare(source, tmp_path / "second", **kwargs)
    assert (tmp_path / "first/parameters.f32").read_bytes() == (tmp_path / "second/parameters.f32").read_bytes()
    assert first["provenance"]["heldout_views"] == ["000000.png", "000008.png"]
    assert first["provenance"]["initialization_uses_heldout_observations"] is False
    assert second["primitive_count"] <= 32


def test_downloader_checks_size_and_hash_and_associates_numeric_mask_stems(tmp_path: Path, monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location("fetch_dtu_subset", MODULE.with_name("fetch_dtu_subset.py"))
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    files = ["scan24/cameras.npz", "scan24/image/000000.png", "scan24/image/000001.png",
             "scan24/mask/000.png", "scan24/mask/001.png"]
    payload = b"downloaded bytes"
    entries = [{"type": "file", "path": name, "size": len(payload),
                "lfs": {"oid": hashlib.sha256(payload).hexdigest()}} for name in files]
    requests = []

    def fake_json(url: str) -> list:
        requests.append(url)
        return entries

    monkeypatch.setattr(fetch, "_json", fake_json)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    with pytest.raises(ValueError, match="exceeds"):
        fetch.fetch(tmp_path / "too_large", train_count=1, test_count=1, max_megabytes=0)
    assert not (tmp_path / "too_large").exists()
    index = fetch.fetch(tmp_path / "good", train_count=1, test_count=1)
    assert index["training_views"] == ["000001.png"]
    assert (tmp_path / "good/mask/001.png").read_bytes() == payload
    assert index["revision"] == fetch.DEFAULT_REVISION
    assert all(f"/tree/{fetch.DEFAULT_REVISION}/" in url for url in requests)
    assert not any("/main/" in url for url in requests)
    explicit_revision = "a"*40
    explicit = fetch.fetch(tmp_path / "explicit", train_count=1, test_count=1, revision=explicit_revision)
    assert explicit["revision"] == explicit_revision
    assert f"/tree/{explicit_revision}/" in requests[-1]
    count = len(requests)
    with pytest.raises(ValueError, match="commit SHA"):
        fetch.fetch(tmp_path / "moving", train_count=1, test_count=1, revision="main")
    assert len(requests) == count
    corrupted = b"X"*len(payload)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(corrupted))
    with pytest.raises(ValueError, match="identity mismatch"):
        fetch.fetch(tmp_path / "corrupt", train_count=1, test_count=1)
    assert not list((tmp_path / "corrupt").rglob("*.partial"))
    assert not (tmp_path / "corrupt/cameras.npz").exists()


def test_nonfinite_or_malformed_camera_metadata_is_rejected_before_export(tmp_path: Path) -> None:
    bad = edge.Frame("bad", Path("unused"), 32, 24, np.eye(3), np.eye(4))
    bad.camera_to_world[0, 3] = np.nan
    with pytest.raises(ValueError, match="rigid camera"):
        edge.validate_frame(bad)
    bad.camera_to_world = np.eye(4)
    bad.observations = {1: np.array([np.inf, 2])}
    with pytest.raises(ValueError, match="feature observations"):
        edge.validate_frame(bad)
    with pytest.raises(ValueError, match="tangent scales"):
        edge.pack_parameters(np.zeros((4, 3)), np.tile([0, 0, 1.], (4, 1)), 0, np.zeros((4, 3)))


def test_interrupted_export_is_not_published_and_retry_identifies_staging(tmp_path: Path, monkeypatch) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr(Image.Image, "save", interrupted)
    with pytest.raises(OSError, match="write failure"):
        edge.synthetic(tmp_path / "incomplete", long_edge=16, steps=8)
    assert not (tmp_path / "incomplete").exists()
    assert (tmp_path / ".incomplete.partial").is_dir()
    with pytest.raises(FileExistsError, match="Incomplete bundle"):
        edge.synthetic(tmp_path / "incomplete", long_edge=16, steps=8)
