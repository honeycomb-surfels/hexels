from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image as PILImage


@dataclass(frozen=True)
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2d_indices: np.ndarray


def _add_colmap_loader_path() -> None:
    root = Path(__file__).resolve().parents[1]
    dependency = str(root / "2d-gaussian-splatting")
    if dependency not in sys.path:
        sys.path.insert(0, dependency)


def _read_bytes(handle: Any, byte_count: int, format_string: str) -> tuple[Any, ...]:
    value = handle.read(byte_count)
    if len(value) != byte_count:
        raise ValueError("Unexpected end of COLMAP points3D binary")
    return struct.unpack(f"<{format_string}", value)


def read_points3d_binary(path: Path) -> dict[int, Point3D]:
    """Read COLMAP points and their observation tracks."""
    points: dict[int, Point3D] = {}
    with path.open("rb") as handle:
        count = _read_bytes(handle, 8, "Q")[0]
        for _ in range(count):
            values = _read_bytes(handle, 43, "QdddBBBd")
            track_length = _read_bytes(handle, 8, "Q")[0]
            track = _read_bytes(handle, 8 * track_length, "ii" * track_length)
            image_ids = np.asarray(track[0::2], dtype=np.int32)
            point2d_indices = np.asarray(track[1::2], dtype=np.int32)
            point = Point3D(
                id=int(values[0]),
                xyz=np.asarray(values[1:4], dtype=np.float64),
                rgb=np.asarray(values[4:7], dtype=np.uint8),
                error=float(values[7]),
                image_ids=image_ids,
                point2d_indices=point2d_indices,
            )
            points[point.id] = point
    return points


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _float_text(value: float) -> str:
    return format(float(value), ".17g")


def _scaled_pinhole_camera(camera: Any, image_size: tuple[int, int]) -> Any:
    if camera.model != "PINHOLE" or len(camera.params) != 4:
        raise ValueError(f"MVS input requires PINHOLE cameras, got {camera.model}")
    width, height = image_size
    scale_x = width / int(camera.width)
    scale_y = height / int(camera.height)
    params = np.asarray(camera.params, dtype=np.float64).copy()
    params[[0, 2]] *= scale_x
    params[[1, 3]] *= scale_y
    return camera._replace(width=width, height=height, params=params)


def _filter_points(
    points: Mapping[int, Point3D],
    selected_image_ids: set[int],
) -> dict[int, Point3D]:
    filtered: dict[int, Point3D] = {}
    for point_id, point in points.items():
        observations = {}
        for image_id, point2d_index in zip(
            point.image_ids,
            point.point2d_indices,
        ):
            image_id = int(image_id)
            if image_id in selected_image_ids and image_id not in observations:
                observations[image_id] = int(point2d_index)
        if len(observations) < 2:
            continue
        filtered[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=np.fromiter(observations, dtype=np.int32),
            point2d_indices=np.fromiter(observations.values(), dtype=np.int32),
        )
    return filtered


def _write_cameras(path: Path, cameras: Mapping[int, Any]) -> None:
    with path.open("w") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            params = " ".join(_float_text(value) for value in camera.params)
            handle.write(
                f"{camera.id} {camera.model} {camera.width} {camera.height} {params}\n"
            )


def _write_images(
    path: Path,
    images: Mapping[int, Any],
    points: Mapping[int, Point3D],
    coordinate_scales: Mapping[int, tuple[float, float]],
) -> None:
    kept_observations = {
        (int(image_id), int(point2d_index)): point.id
        for point in points.values()
        for image_id, point2d_index in zip(
            point.image_ids,
            point.point2d_indices,
        )
    }
    observations = len(kept_observations)
    mean_observations = observations / len(images)
    with path.open("w") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        handle.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(
            f"# Number of images: {len(images)}, mean observations per image: "
            f"{_float_text(mean_observations)}\n"
        )
        for image_id in sorted(images):
            image = images[image_id]
            pose = [*image.qvec, *image.tvec]
            pose_text = " ".join(_float_text(value) for value in pose)
            handle.write(
                f"{image.id} {pose_text} {image.camera_id} {image.name}\n"
            )
            scale_x, scale_y = coordinate_scales[image.camera_id]
            entries = []
            for point2d_index, (xy, point_id) in enumerate(
                zip(image.xys, image.point3D_ids)
            ):
                expected_id = kept_observations.get((image.id, point2d_index))
                output_id = int(point_id) if expected_id == int(point_id) else -1
                entries.extend(
                    (
                        _float_text(float(xy[0]) * scale_x),
                        _float_text(float(xy[1]) * scale_y),
                        str(output_id),
                    )
                )
            handle.write(" ".join(entries) + "\n")


def _write_points(path: Path, points: Mapping[int, Point3D]) -> None:
    mean_track = sum(len(point.image_ids) for point in points.values()) / len(points)
    with path.open("w") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        handle.write(
            f"# Number of points: {len(points)}, mean track length: "
            f"{_float_text(mean_track)}\n"
        )
        for point_id in sorted(points):
            point = points[point_id]
            fields = [
                str(point.id),
                *(_float_text(value) for value in point.xyz),
                *(str(int(value)) for value in point.rgb),
                _float_text(point.error),
            ]
            for image_id, point2d_index in zip(
                point.image_ids,
                point.point2d_indices,
            ):
                fields.extend((str(int(image_id)), str(int(point2d_index))))
            handle.write(" ".join(fields) + "\n")


def prepare_proposal_mvs_input(
    *,
    source_path: Path,
    camera_names: Iterable[str],
    output_path: Path,
    image_directory: str = "images",
) -> dict[str, Any]:
    """Create a proposal-pixel-only COLMAP model and image directory."""
    _add_colmap_loader_path()
    from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary

    sparse_path = source_path / "sparse" / "0"
    image_path = source_path / image_directory
    cameras = read_intrinsics_binary(str(sparse_path / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse_path / "images.bin"))
    points = read_points3d_binary(sparse_path / "points3D.bin")

    names = tuple(camera_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("camera_names must be nonempty and unique")
    images_by_stem = {Path(image.name).stem: image for image in images.values()}
    missing = sorted(set(names) - set(images_by_stem))
    if missing:
        raise ValueError(f"Camera manifest contains missing images: {missing[:3]}")
    selected_images = {
        images_by_stem[name].id: images_by_stem[name]
        for name in names
    }
    selected_image_ids = set(selected_images)
    selected_points = _filter_points(points, selected_image_ids)

    image_sizes = {}
    for image in selected_images.values():
        path = image_path / image.name
        with PILImage.open(path) as bitmap:
            image_sizes[path] = bitmap.size
    sizes_by_camera: dict[int, set[tuple[int, int]]] = {}
    for image in selected_images.values():
        sizes_by_camera.setdefault(image.camera_id, set()).add(
            image_sizes[image_path / image.name]
        )
    if any(len(sizes) != 1 for sizes in sizes_by_camera.values()):
        raise ValueError("Images sharing a camera must have identical dimensions")

    selected_camera_ids = {image.camera_id for image in selected_images.values()}
    selected_cameras = {
        camera_id: _scaled_pinhole_camera(
            cameras[camera_id],
            next(iter(sizes_by_camera[camera_id])),
        )
        for camera_id in selected_camera_ids
    }
    coordinate_scales = {
        camera_id: (
            selected_cameras[camera_id].width / cameras[camera_id].width,
            selected_cameras[camera_id].height / cameras[camera_id].height,
        )
        for camera_id in selected_camera_ids
    }
    scaled_points = {}
    for point_id, point in selected_points.items():
        error_scales = [
            (
                coordinate_scales[selected_images[int(image_id)].camera_id][0]
                * coordinate_scales[selected_images[int(image_id)].camera_id][1]
            )
            ** 0.5
            for image_id in point.image_ids
        ]
        scaled_points[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error * float(np.mean(error_scales)),
            image_ids=point.image_ids,
            point2d_indices=point.point2d_indices,
        )
    selected_points = scaled_points

    model_path = output_path / "input_model"
    linked_images_path = output_path / "input_images"
    if model_path.exists() or linked_images_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing MVS input at {output_path}")
    model_path.mkdir(parents=True)
    linked_images_path.mkdir()
    _write_cameras(model_path / "cameras.txt", selected_cameras)
    _write_images(
        model_path / "images.txt",
        selected_images,
        selected_points,
        coordinate_scales,
    )
    _write_points(model_path / "points3D.txt", selected_points)

    for image in selected_images.values():
        source = (image_path / image.name).resolve()
        (linked_images_path / image.name).symlink_to(source)

    selected_names = sorted(Path(image.name).stem for image in selected_images.values())
    observations_per_image = {image_id: 0 for image_id in selected_image_ids}
    neighbors_per_image = {image_id: set() for image_id in selected_image_ids}
    for point in selected_points.values():
        track = [int(image_id) for image_id in point.image_ids]
        for image_id in track:
            observations_per_image[image_id] += 1
            neighbors_per_image[image_id].update(set(track) - {image_id})
    linked_inventory = [
        f"{image.name}\t{image_sizes[image_path / image.name][0]}\t"
        f"{image_sizes[image_path / image.name][1]}\t"
        f"{(image_path / image.name).stat().st_size}"
        for image in sorted(selected_images.values(), key=lambda item: item.name)
    ]
    result = {
        "schema_version": 1,
        "source_path": str(source_path),
        "construction_boundary": (
            "Dense-MVS pixels and tracks are restricted to proposal cameras. "
            "World-space sparse points and poses originate from the shared dataset COLMAP model."
        ),
        "camera_count": len(selected_cameras),
        "image_count": len(selected_images),
        "point_count": len(selected_points),
        "source_point_count": len(points),
        "source_image_count": len(images),
        "excluded_image_count": len(images) - len(selected_images),
        "observation_count": sum(
            len(point.image_ids) for point in selected_points.values()
        ),
        "minimum_image_observations": min(observations_per_image.values()),
        "minimum_image_neighbors": min(
            len(neighbors) for neighbors in neighbors_per_image.values()
        ),
        "camera_names_sha256": _json_sha256(selected_names),
        "linked_image_inventory_sha256": hashlib.sha256(
            "\n".join(linked_inventory).encode()
        ).hexdigest(),
        "model": {
            name: {
                "path": str(model_path / name),
                "sha256": _sha256(model_path / name),
            }
            for name in ("cameras.txt", "images.txt", "points3D.txt")
        },
        "input_images_path": str(linked_images_path),
        "input_model_path": str(model_path),
    }
    artifact_path = output_path / "input_artifact.json"
    artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    result["artifact"] = {
        "path": str(artifact_path),
        "sha256": _sha256(artifact_path),
    }
    return result
