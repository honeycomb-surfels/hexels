"""Prepare bounded, calibrated V1 bundles for the native Apple prototype.

Only NumPy and Pillow are required. COLMAP seeds are re-triangulated from
selected training observations; DTU seeds are a training-mask visual hull.
Neither path uses held-out pixels, masks, points, or colors for initialization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)
LAYOUT = "hexels_v1_log_wxyz_20"


@dataclass
class Frame:
    """One calibrated CV-basis image; observations are source pixel centers."""

    name: str
    image: Path
    width: int
    height: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    mask: Path | None = None
    observations: dict[int, np.ndarray] = field(default_factory=dict)


def safe_child(root: Path, relative: str) -> Path:
    """Resolve an input asset without allowing absolute/traversing paths."""
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes scene root: {relative}")
    return candidate


def select_views(names: list[str], train_count: int, test_count: int,
                 hold_every: int = 8) -> tuple[list[str], list[str]]:
    """Reserve every eighth sorted source view BEFORE selecting either subset."""
    if train_count < 1 or test_count < 1 or hold_every < 2:
        raise ValueError("Positive train/test counts and hold_every >= 2 are required")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate camera names")
    ordered = sorted(names)
    test_pool = ordered[::hold_every]
    train_pool = [name for i, name in enumerate(ordered) if i % hold_every]

    def subset(pool: list[str], count: int) -> list[str]:
        if len(pool) < count:
            raise ValueError(f"Requested {count} views from pool containing {len(pool)}")
        indices = np.linspace(0, len(pool) - 1, count).round().astype(int)
        return [pool[i] for i in indices]

    return subset(train_pool, train_count), subset(test_pool, test_count)


def resize_intrinsics(k: np.ndarray, old_size: tuple[int, int],
                      new_size: tuple[int, int]) -> np.ndarray:
    """Map zero-based pixel centers through Pillow's half-pixel resize."""
    sx, sy = new_size[0] / old_size[0], new_size[1] / old_size[1]
    affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return affine @ np.asarray(k, dtype=np.float64)


def quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """Return a proper rotation matrix from a finite WXYZ quaternion."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid WXYZ quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def validate_frame(frame: Frame) -> None:
    """Reject malformed camera metadata before geometry or file publication."""
    k, pose = frame.intrinsics, frame.camera_to_world
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-8):
        raise ValueError(f"Invalid pinhole intrinsics for {frame.name}")
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f"Invalid rigid camera-to-world pose for {frame.name}")
    if frame.width < 0 or frame.height < 0 or bool(frame.width) != bool(frame.height):
        raise ValueError(f"Invalid image dimensions for {frame.name}")
    if any(np.shape(xy) != (2,) or not np.isfinite(xy).all() for xy in frame.observations.values()):
        raise ValueError(f"Invalid feature observations for {frame.name}")


def rotations_from_normals(normals: np.ndarray) -> np.ndarray:
    """Map local +Z to each unit normal, including the antiparallel case."""
    normals = np.asarray(normals, dtype=np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError("Normals must have shape (N,3), finite and nonzero")
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if not np.isfinite(normals).all() or np.any(lengths < 1e-12):
        raise ValueError("Normals must have shape (N,3), finite and nonzero")
    n = normals / lengths
    q = np.column_stack((1+n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))))
    q[np.linalg.norm(q, axis=1) < 1e-7] = [0, 1, 0, 0]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a 3x4 pinhole matrix using NumPy RQ; return K and CV c2w."""
    p = np.asarray(projection, dtype=np.float64).copy()
    if p.shape != (3, 4) or not np.isfinite(p).all() or abs(np.linalg.det(p[:, :3])) < 1e-12:
        raise ValueError("Projection must be finite, nonsingular and shape (3,4)")
    if np.linalg.det(p[:, :3]) < 0:
        p *= -1
    q, r = np.linalg.qr(np.flipud(p[:, :3]).T)
    k = np.fliplr(np.flipud(r.T))
    rotation = np.flipud(q.T)
    signs = np.diag(np.where(np.diag(k) < 0, -1.0, 1.0))
    k, rotation = k @ signs, signs @ rotation
    translation = np.linalg.solve(k, p[:, 3])
    k /= k[2, 2]
    # DTU's float32 stored projections acquire tiny numerical skew during RQ.
    # Reject material skew, but remove errors below one part per million of f.
    if abs(k[0, 1]) > 1e-6*max(k[0, 0], k[1, 1]) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("Skewed or nonrigid camera unsupported by native pinhole contract")
    k[0, 1] = 0
    c2w = np.eye(4)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ translation
    return k, c2w


def _read(handle: BinaryIO, fmt: str) -> tuple:
    size = struct.calcsize("<" + fmt)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("Truncated COLMAP binary model")
    return struct.unpack("<" + fmt, data)


def _camera(model: str, width: int, height: int, values: list[float]) -> np.ndarray:
    if model == "SIMPLE_PINHOLE" and len(values) == 3:
        f, cx, cy = values
        fx, fy = f, f
    elif model == "PINHOLE" and len(values) == 4:
        fx, fy, cx, cy = values
    else:
        raise ValueError(f"Unsupported COLMAP model {model}; undistort to PINHOLE first")
    if width < 1 or height < 1 or not np.isfinite(values).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid calibrated image dimensions or intrinsics")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def load_colmap(source: Path, images: str = "images", model_path: str | None = None) -> list[Frame]:
    """Read COLMAP text/binary cameras and image tracks without Torch/CUDA."""
    if model_path is not None:
        model = safe_child(source, model_path)
    else:
        model = source / "sparse/0"
        if not model.is_dir():
            model = source / "sparse"
        if not model.is_dir():
            model = source
    image_root = safe_child(source, images)
    cameras: dict[int, tuple[int, int, np.ndarray]] = {}
    records: list[tuple[int, np.ndarray, np.ndarray, int, str, dict]] = []
    if (model / "cameras.bin").exists():
        with (model / "cameras.bin").open("rb") as handle:
            for _ in range(_read(handle, "Q")[0]):
                camera_id, model_id, w, h = _read(handle, "iiQQ")
                if model_id not in (0, 1):
                    raise ValueError(f"Unsupported COLMAP model ID {model_id}; undistort first")
                values = list(_read(handle, "3d" if model_id == 0 else "4d"))
                k = _camera("SIMPLE_PINHOLE" if model_id == 0 else "PINHOLE", w, h, values)
                cameras[camera_id] = (w, h, k)
        with (model / "images.bin").open("rb") as handle:
            for _ in range(_read(handle, "Q")[0]):
                fields = _read(handle, "i7di")
                name_bytes = bytearray()
                while True:
                    char = handle.read(1)
                    if not char:
                        raise ValueError("Truncated COLMAP image name")
                    if char == b"\0":
                        break
                    name_bytes.extend(char)
                observations = {}
                for _ in range(_read(handle, "Q")[0]):
                    x, y, point_id = _read(handle, "ddq")
                    if point_id >= 0:
                        observations[point_id] = np.array([x, y])
                records.append((fields[0], np.array(fields[1:5]), np.array(fields[5:8]), fields[8],
                                name_bytes.decode("utf-8"), observations))
    else:
        for line in (model / "cameras.txt").read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split()
            camera_id, model_name, w, h = int(values[0]), values[1], int(values[2]), int(values[3])
            cameras[camera_id] = (w, h, _camera(model_name, w, h, list(map(float, values[4:]))))
        lines = iter((model / "images.txt").read_text().splitlines())
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = line.split(maxsplit=9)
            if len(values) != 10:
                raise ValueError("Malformed COLMAP image record")
            points = next(lines, "").split()
            if len(points) % 3:
                raise ValueError("Malformed COLMAP image observations")
            observations = {int(points[i+2]): np.array([float(points[i]), float(points[i+1])])
                            for i in range(0, len(points), 3) if int(points[i+2]) >= 0}
            records.append((int(values[0]), np.array(values[1:5], dtype=float),
                            np.array(values[5:8], dtype=float), int(values[8]), values[9], observations))
    frames = []
    for _, q, t, camera_id, name, observations in records:
        if camera_id not in cameras:
            raise ValueError(f"Image {name} references missing camera {camera_id}")
        w, h, source_k = cameras[camera_id]
        # COLMAP uses image corners: the first pixel center is (0.5, 0.5).
        # The native prototype instead evaluates rays at integer centers.
        k = source_k.copy()
        k[:2, 2] -= .5
        observations = {point_id: xy-.5 for point_id, xy in observations.items()}
        rotation = quaternion_matrix(q)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = rotation.T, -rotation.T @ t
        frames.append(Frame(name, safe_child(image_root, name), w, h, k, c2w,
                            observations=observations))
    if not frames:
        raise ValueError("No registered COLMAP cameras")
    for frame in frames:
        validate_frame(frame)
    return sorted(frames, key=lambda frame: frame.name)


def load_dtu(source: Path) -> list[Frame]:
    """Read DTU/IDR projection matrices, optionally with a downloaded subset index."""
    image_root = source / "image"
    index = source / "source_index.json"
    if index.exists():
        names = json.loads(index.read_text())["image_names"]
    else:
        names = sorted(p.name for p in image_root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    frames = []
    masks = {int(path.stem): path for path in (source / "mask").glob("*.png") if path.stem.isdigit()}
    with np.load(source / "cameras.npz", allow_pickle=False) as cameras:
        common_scale = None
        for ordinal, name in enumerate(names):
            index_number = int(Path(name).stem) if Path(name).stem.isdigit() else ordinal
            world = np.asarray(cameras[f"world_mat_{index_number}"], dtype=np.float64)
            scale = np.asarray(cameras.get(f"scale_mat_{index_number}", np.eye(4)), dtype=np.float64)
            if common_scale is None:
                common_scale = scale
            if not np.allclose(scale, common_scale, atol=1e-6):
                raise ValueError("DTU cameras use inconsistent world normalization")
            k, c2w = decompose_projection((world @ scale)[:3, :4])
            path = safe_child(image_root, name)
            if path.exists():
                with Image.open(path) as image:
                    w, h = image.size
            else:
                # Downloaded subsets retain the full source name list for correct split assignment.
                w, h = 0, 0
            mask = masks.get(index_number, source / "mask" / (Path(name).stem + ".png"))
            frames.append(Frame(name, path, w, h, k, c2w, mask))
    if not frames:
        raise ValueError("No DTU cameras")
    for frame in frames:
        validate_frame(frame)
    return sorted(frames, key=lambda frame: frame.name)


def prepare_image(frame: Frame, long_edge: int, background: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize color and alpha, composite once, and map calibrated intrinsics."""
    if long_edge < 1:
        raise ValueError("long_edge must be positive")
    with Image.open(frame.image) as image:
        if frame.width and image.size != (frame.width, frame.height):
            raise ValueError(f"Calibrated/image size mismatch for {frame.name}")
        old_size = image.size
        ratio = min(1.0, long_edge / max(old_size))
        size = tuple(max(1, round(d * ratio)) for d in old_size)
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255
        if frame.mask is not None:
            if not frame.mask.is_file():
                raise ValueError(f"Missing silhouette mask: {frame.mask}")
            with Image.open(frame.mask) as mask_image:
                if mask_image.size != old_size:
                    raise ValueError(f"Mask/image size mismatch for {frame.name}")
                alpha *= np.asarray(mask_image.convert("L"), dtype=np.float32) / 255
        # Composite before LANCZOS, preventing transparent RGB from bleeding into edges.
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32) / 255
        composited = rgb * alpha[..., None] + background * (1-alpha[..., None])
        output = Image.fromarray(np.rint(composited.clip(0, 1)*255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)
        resized_alpha = Image.fromarray(np.rint(alpha*255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS)
        return np.asarray(output).copy(), np.asarray(resized_alpha).copy() / 255, resize_intrinsics(frame.intrinsics, old_size, size)


def project(points: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into zero-based CV pixel coordinates."""
    w2c = np.linalg.inv(frame.camera_to_world)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    image = camera @ frame.intrinsics.T
    return image[:, :2] / np.where(abs(image[:, 2:]) > 1e-12, image[:, 2:], 1e-12), camera[:, 2]


def sample_colors(points: np.ndarray, frames: list[Frame], pixels: list[np.ndarray],
                  alphas: list[np.ndarray]) -> np.ndarray:
    """Average only in-mask, positive-depth training pixels for each seed."""
    colors = np.zeros((len(points), 3), dtype=np.float64)
    counts = np.zeros(len(points))
    for frame, image, mask in zip(frames, pixels, alphas, strict=True):
        xy, depth = project(points, frame)
        rounded = np.rint(xy).astype(np.int64)
        valid = (depth > 0) & (rounded[:, 0] >= 0) & (rounded[:, 0] < image.shape[1]) & (rounded[:, 1] >= 0) & (rounded[:, 1] < image.shape[0])
        ids = np.flatnonzero(valid)
        ids = ids[mask[rounded[ids, 1], rounded[ids, 0]] > 0.5]
        colors[ids] += image[rounded[ids, 1], rounded[ids, 0]] / 255
        counts[ids] += 1
    if np.any(counts == 0):
        raise ValueError("Initialization contains seeds not observed by any training image")
    return colors / counts[:, None]


def triangulate_training(frames: list[Frame], limit: int) -> np.ndarray:
    """Re-triangulate selected training tracks, never reuse all-view point XYZ."""
    if limit < 4:
        raise ValueError("Triangulation seed limit must be at least four")
    if len(frames) < 2:
        raise ValueError("Triangulation requires at least two training cameras")
    centers = np.stack([frame.camera_to_world[:3, 3] for frame in frames])
    if np.linalg.norm(np.ptp(centers, axis=0)) < 1e-8:
        raise ValueError("Training cameras have no translational baseline")
    tracks: dict[int, list[tuple[Frame, np.ndarray]]] = {}
    for frame in frames:
        for point_id, xy in frame.observations.items():
            tracks.setdefault(point_id, []).append((frame, xy))
    points = []
    for point_id in sorted(tracks):
        observations = tracks[point_id]
        if len(observations) < 2:
            continue
        rays = []
        for frame, xy in observations:
            ray = frame.camera_to_world[:3, :3] @ np.linalg.solve(frame.intrinsics, [*xy, 1])
            rays.append(ray/np.linalg.norm(ray))
        # Pure rotations and near-parallel rays leave depth ill-conditioned.
        if np.max(np.linalg.norm(np.cross(rays[0], rays), axis=1)) < 1e-3:
            continue
        rows = []
        for frame, xy in observations:
            p = frame.intrinsics @ np.linalg.inv(frame.camera_to_world)[:3]
            rows.extend((xy[0]*p[2]-p[0], xy[1]*p[2]-p[1]))
        _, _, vh = np.linalg.svd(rows)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-10:
            continue
        point = homogeneous[:3] / homogeneous[3]
        if not np.isfinite(point).all():
            continue
        if all((project(point[None], frame)[1][0] > 0 and
                np.linalg.norm(project(point[None], frame)[0][0] - xy) < 3)
               for frame, xy in observations):
            points.append(point)
    if len(points) < 4:
        raise ValueError("Need at least four valid seeds triangulated from selected training tracks")
    points = np.asarray(points)
    return points[np.linspace(0, len(points)-1, min(limit, len(points))).round().astype(int)]


def visual_hull(frames: list[Frame], alphas: list[np.ndarray], limit: int,
                resolution: int = 48, *, smooth_normals: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Intersect training silhouettes inside DTU's normalized [-1,1]^3 volume."""
    if not 8 <= resolution <= 128:
        raise ValueError("Hull resolution must be in [8,128]")
    axis = np.linspace(-1, 1, resolution)
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    occupied = np.ones(len(points), dtype=bool)
    for frame, mask in zip(frames, alphas, strict=True):
        xy, depth = project(points, frame)
        uv = np.rint(xy).astype(int)
        valid = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
        ids = np.flatnonzero(valid)
        valid[ids] &= mask[uv[ids, 1], uv[ids, 0]] > 0.5
        occupied &= valid
    occupancy = occupied.reshape((resolution,)*3)
    padded = np.pad(occupancy, 1)
    interior = occupancy.copy()
    for dimension in range(3):
        for offset in (0, 2):
            slices = [slice(1, -1)]*3
            slices[dimension] = slice(offset, offset+resolution)
            interior &= padded[tuple(slices)]
    boundary = occupancy & ~interior
    ids = np.flatnonzero(boundary.ravel())
    if len(ids) < 4:
        raise ValueError("Training silhouettes have an empty/degenerate visual hull")
    ids = ids[np.linspace(0, len(ids)-1, min(limit, len(ids))).round().astype(int)]
    normal_field = np.pad(occupancy.astype(float), 1)
    if smooth_normals:
        # Two separable binomial passes reduce grid-normal quantization without
        # changing which training-silhouette voxels seed the geometry.
        for _ in range(2):
            for dimension in range(3):
                normal_field = (np.roll(normal_field, 1, axis=dimension)
                                + 2 * normal_field
                                + np.roll(normal_field, -1, axis=dimension)) * .25
    gradient = np.stack(np.gradient(normal_field), axis=-1)[1:-1, 1:-1, 1:-1]
    normals = -gradient.reshape(-1, 3)[ids]
    zero = np.linalg.norm(normals, axis=1) < 1e-9
    normals[zero] = points[ids][zero] - points[ids].mean(axis=0)
    zero = np.linalg.norm(normals, axis=1) < 1e-9
    normals[zero] = [0, 0, 1]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points[ids], normals, 2/(resolution-1)*1.2


def point_geometry(points: np.ndarray, camera_centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA normals and spacing, with optional SciPy acceleration for large clouds."""
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all()
            or camera_centers.ndim != 2 or camera_centers.shape[1] != 3
            or len(camera_centers) < 1 or not np.isfinite(camera_centers).all()):
        raise ValueError("PCA initialization requires at least four finite points and finite camera centers")
    k = min(12, len(points))
    try:
        from scipy.spatial import cKDTree
        distances, indices = cKDTree(points).query(points, k=k, workers=1)
    except ImportError:
        if len(points) > 5000:
            raise ValueError("Install scipy for COLMAP clouds larger than 5000 seeds") from None
        indices = np.empty((len(points), k), dtype=int)
        distances = np.empty((len(points), k))
        for start in range(0, len(points), 128):
            squared = ((points[start:start+128, None] - points[None])**2).sum(axis=2)
            local = np.argsort(squared, axis=1)[:, :k]
            indices[start:start+128] = local
            distances[start:start+128] = np.sqrt(np.take_along_axis(squared, local, axis=1))
    neighbors = points[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered))
    normals = eigenvectors[:, :, 0]
    planarity = (eigenvalues[:, 1]-eigenvalues[:, 0])/np.maximum(eigenvalues[:, 2], 1e-15)
    unsupported = planarity < .05
    if np.any(unsupported):
        # A line/point has no stable PCA normal. Use a finite facing prior,
        # explicitly distinct from an observed surface normal.
        nearby = ((points[unsupported, None]-camera_centers[None])**2).sum(axis=2).argmin(axis=1)
        fallback = camera_centers[nearby]-points[unsupported]
        fallback[np.linalg.norm(fallback, axis=1) < 1e-12] = [0, 0, 1]
        normals[unsupported] = fallback/np.linalg.norm(fallback, axis=1, keepdims=True)
    toward_camera = camera_centers.mean(axis=0) - points
    normals *= np.where((normals*toward_camera).sum(axis=1) < 0, -1, 1)[:, None]
    extent = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-3)
    scales = np.sqrt(np.mean(distances[:, 1:min(k, 4)]**2, axis=1)).clip(extent*1e-5, extent*.1)
    return normals, scales


def pack_parameters(points: np.ndarray, normals: np.ndarray, scales: np.ndarray | float,
                    colors: np.ndarray) -> np.ndarray:
    """Pack activated V1 inputs into native N×20 little-endian float32 rows."""
    count = len(points)
    scales_array = np.asarray(scales)
    if (np.shape(points) != (count, 3) or np.shape(normals) != (count, 3)
            or np.shape(colors) != (count, 3) or not np.isfinite(scales_array).all()
            or np.any(scales_array <= 0)):
        raise ValueError("Invalid parameter shapes or nonpositive tangent scales")
    output = np.zeros((count, 20), dtype="<f4")
    output[:, :3] = points
    output[:, 3:5] = np.log(np.broadcast_to(scales_array.reshape(-1, 1), (count, 2)))
    output[:, 5:9] = rotations_from_normals(normals)
    output[:, 9:12] = colors
    output[:, 18] = math.log(.1/.9)
    output[:, 19] = math.log(10)
    if count < 1 or not np.isfinite(output).all() or np.any((colors < 0) | (colors > 1)):
        raise ValueError("Invalid initial parameters")
    return output


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_bundle(output: Path, frames: list[Frame], train_names: list[str],
                 pixels: list[np.ndarray], parameters: np.ndarray,
                 background: np.ndarray, seed: int, steps: int, provenance: dict,
                 *, recipe_hint: str | None = None) -> dict:
    """Write a new native bundle; refuse overwriting prior experiment artifacts."""
    if output.exists():
        raise FileExistsError(f"Bundle already exists: {output}")
    if steps < 1:
        raise ValueError("Sampling schedule must contain at least one step")
    if recipe_hint is not None and recipe_hint not in {"standard", "detail"}:
        raise ValueError("recipe_hint must be standard or detail")
    if (parameters.ndim != 2 or parameters.shape[1] != 20 or len(parameters) < 1
            or not np.isfinite(parameters).all()
            or not np.allclose(np.linalg.norm(parameters[:, 5:9], axis=1), 1, atol=1e-5)):
        raise ValueError("Parameters must be finite N×20 rows with unit WXYZ rotations")
    if len(frames) != len(pixels) or len({frame.name for frame in frames}) != len(frames):
        raise ValueError("Bundle needs one image per uniquely named camera")
    if not train_names or len(set(train_names)) != len(train_names) or not set(train_names) < {frame.name for frame in frames}:
        raise ValueError("Bundle needs disjoint, nonempty training and held-out views")
    for frame, image in zip(frames, pixels, strict=True):
        validate_frame(frame)
        if image.shape != (frame.height, frame.width, 3) or image.dtype != np.uint8:
            raise ValueError(f"Image shape/dtype differs from camera: {frame.name}")
    target = output
    output = target.with_name("."+target.name+".partial")
    if output.exists():
        raise FileExistsError(f"Incomplete bundle at {output}; remove it before rerunning")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    entries = []
    for i, (frame, image) in enumerate(zip(frames, pixels, strict=True)):
        relative = f"images/{i:05d}.png"
        Image.fromarray(image).save(output / relative)
        k = frame.intrinsics
        entries.append({"name": frame.name, "image": relative, "width": image.shape[1],
                        "height": image.shape[0], "intrinsics": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                        "camera_to_world": frame.camera_to_world.ravel().tolist(),
                        "split": "train" if frame.name in train_names else "test"})
    parameters.astype("<f4").tofile(output / "parameters.f32")
    train_indices = [i for i, entry in enumerate(entries) if entry["split"] == "train"]
    rng = np.random.default_rng(seed)
    schedule = np.concatenate([rng.permutation(train_indices) for _ in range(math.ceil(steps/len(train_indices)))])[:steps]
    schedule.astype("<u4").tofile(output / "view_order.u32")
    manifest = {"schema_version": 1, "variant": "v1", "parameter_layout": LAYOUT,
                "primitive_count": len(parameters), "parameters_file": "parameters.f32",
                "view_order_file": "view_order.u32", "background": background.tolist(),
                "frames": entries, "provenance": provenance,
                "pixel_coordinates": "zero_based_centers_half_pixel_resize",
                "color_space": "sRGB_encoded_RGB_0_1", "scene_units": "source_normalized_not_assumed_meters",
                "seed": seed}
    if recipe_hint is not None:
        manifest["recipe_hint"] = recipe_hint
    manifest["sha256"] = {str(path.relative_to(output)): sha256(path)
                           for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    output.rename(target)
    return manifest


def prepare(source: Path, output: Path, *, kind: str = "auto", train_count: int = 24,
            test_count: int = 6, long_edge: int = 640, max_points: int = 50000,
            seed: int = 0, steps: int = 30000, hull_resolution: int = 48,
            images: str = "images", background: tuple[float, float, float] = (1, 1, 1),
            colmap_model: str | None = None, quality_seed: str = "legacy",
            seed_long_edge: int | None = None, recipe_hint: str | None = None) -> dict:
    """Prepare the selected known-camera dataset into a bounded native bundle."""
    if not 4 <= max_points <= 50000:
        raise ValueError("max_points must be in [4,50000]")
    if quality_seed not in {"legacy", "visible", "visible_sharp", "calibrated"}:
        raise ValueError("quality_seed must be legacy, visible, visible_sharp, or calibrated")
    if kind == "auto":
        kind = "dtu" if (source / "cameras.npz").is_file() else "colmap"
    if quality_seed == "calibrated" and kind != "dtu":
        raise ValueError("The bounded calibrated SIFT initializer currently requires normalized DTU input")
    if seed_long_edge is not None and (quality_seed != "calibrated" or not 32 <= seed_long_edge <= 2048):
        raise ValueError("seed_long_edge requires calibrated initialization and must be in [32,2048]")
    frames = load_dtu(source) if kind == "dtu" else load_colmap(source, images, colmap_model)
    train_names, test_names = select_views([frame.name for frame in frames], train_count, test_count)
    selected = [frame for frame in frames if frame.name in train_names or frame.name in test_names]
    background_array = np.asarray(background, dtype=np.float64)
    if background_array.shape != (3,) or not np.isfinite(background_array).all() or np.any((background_array < 0) | (background_array > 1)):
        raise ValueError("Background must be three finite values in [0,1]")
    pixels, alphas = [], []
    for frame in selected:
        image, alpha, k = prepare_image(frame, long_edge, background_array)
        frame.observations = {pid: ((xy+.5)*[image.shape[1]/frame.width, image.shape[0]/frame.height]-.5)
                              for pid, xy in frame.observations.items()}
        frame.intrinsics, frame.width, frame.height = k, image.shape[1], image.shape[0]
        pixels.append(image)
        alphas.append(alpha)
    train_ids = [i for i, frame in enumerate(selected) if frame.name in train_names]
    training = [selected[i] for i in train_ids]
    if kind == "dtu" and quality_seed == "calibrated":
        from calibrated_initialization import calibrated_parameters
        seed_frames, seed_pixels, seed_alphas = training, [pixels[i] for i in train_ids], [alphas[i] for i in train_ids]
        if seed_long_edge is not None and seed_long_edge != long_edge:
            seed_frames = [frame for frame in load_dtu(source) if frame.name in train_names]
            seed_pixels, seed_alphas = [], []
            for frame in seed_frames:
                image, alpha, k = prepare_image(frame, seed_long_edge, background_array)
                frame.intrinsics, frame.width, frame.height = k, image.shape[1], image.shape[0]
                seed_pixels.append(image)
                seed_alphas.append(alpha)
        parameters = calibrated_parameters(seed_frames, seed_pixels, seed_alphas, max_points)
        method = "selected_training_SIFT_tracks_3plus_views_known_camera_triangulation"
    elif kind == "dtu":
        points, normals, scales = visual_hull(training, [alphas[i] for i in train_ids], max_points, hull_resolution,
                                              smooth_normals=quality_seed != "legacy")
        method = "training_silhouettes_visual_hull_not_MVS"
    else:
        points = triangulate_training(training, max_points)
        normals, scales = point_geometry(points, np.stack([frame.camera_to_world[:3, 3] for frame in training]))
        method = "retriangulated_selected_training_tracks_PCA_normals_camera_fallback"
    if quality_seed != "calibrated":
        colors = sample_colors(points, training, [pixels[i] for i in train_ids], [alphas[i] for i in train_ids])
        parameters = pack_parameters(points, normals, scales, colors)
    if quality_seed in {"visible", "visible_sharp"}:
        from quality_initialization import visible_parameters
        parameters = visible_parameters(parameters, normals, training,
                                        [pixels[i] for i in train_ids], [alphas[i] for i in train_ids],
                                        sharp=quality_seed == "visible_sharp")
    provenance = {"source": str(source.resolve()), "format": kind, "initialization": method,
                  "source_view_names": [frame.name for frame in frames], "training_views": train_names,
                  "heldout_views": test_names, "split_rule": "every_eighth_sorted_source_before_subsampling",
                  "initialization_uses_heldout_observations": False,
                  "camera_calibration": "provided_dataset_cameras_may_have_all_view_provenance",
                  "source_pixel_convention": "COLMAP_corner_centers_shifted_minus_half" if kind == "colmap" else "DTU_assumed_zero_based_pixel_centers",
                  "mask_policy": "alpha_and_external_mask_composited_before_resize", "hull_resolution": hull_resolution if kind == "dtu" and quality_seed != "calibrated" else None}
    if quality_seed != "legacy":
        provenance["quality_seed"] = quality_seed
        if quality_seed == "calibrated":
            provenance["seed_long_edge"] = seed_long_edge or long_edge
        if quality_seed in {"visible", "visible_sharp"}:
            provenance["initialization"] += "_visible_training_RGB"
        if kind == "dtu" and quality_seed in {"visible", "visible_sharp"}:
            provenance["initialization"] += "_smoothed_hull_normals"
        if quality_seed in {"visible", "visible_sharp"}:
            provenance["visibility"] = "8px_depth_cells_3_voxel_depth_tolerance_top2_facing_views"
    if kind == "dtu":
        with np.load(source / "cameras.npz", allow_pickle=False) as cameras:
            number = int(Path(frames[0].name).stem) if Path(frames[0].name).stem.isdigit() else 0
            provenance["source_coordinates_from_bundle"] = np.asarray(
                cameras.get(f"scale_mat_{number}", np.eye(4))).ravel().tolist()
    else:
        provenance["source_coordinates_from_bundle"] = np.eye(4).ravel().tolist()
        provenance["colmap_model"] = colmap_model or "default_sparse_0_then_sparse_then_scene_root"
    acquisition = source / "source_index.json"
    if acquisition.exists():
        provenance["acquisition"] = json.loads(acquisition.read_text())
    return write_bundle(output, selected, train_names, pixels, parameters, background_array, seed, steps, provenance,
                        recipe_hint=recipe_hint)


def synthetic(output: Path, *, long_edge: int = 64, seed: int = 0, steps: int = 30000,
              recipe_hint: str | None = None) -> dict:
    """Generate an explicitly synthetic known-plane smoke case, not a benchmark."""
    width, height = long_edge, max(8, round(long_edge*.75))
    k = np.array([[width*.85, 0, (width-1)/2], [0, width*.85, (height-1)/2], [0, 0, 1.]])
    frames, pixels = [], []
    y, x = np.mgrid[:height, :width]
    for i in range(48):
        angle = i/48*2*np.pi
        c2w = np.eye(4)
        c2w[:3, 3] = [.22*np.cos(angle), .15*np.sin(angle), -2]
        px = (x-k[0, 2])/k[0, 0]*2+c2w[0, 3]
        py = (y-k[1, 2])/k[1, 1]*2+c2w[1, 3]
        inside = (abs(px) <= .9) & (abs(py) <= .65)
        image = np.ones((height, width, 3))
        color = np.stack((.45+.2*px, .5+.25*py, np.full_like(px, .35)), axis=-1)
        image[inside] = color[inside]
        pixels.append(np.rint(image.clip(0, 1)*255).astype(np.uint8))
        frames.append(Frame(f"synthetic_{i:03d}.png", Path("unused"), width, height, k.copy(), c2w))
    train_names, test_names = select_views([frame.name for frame in frames], 24, 6)
    ids = [i for i, frame in enumerate(frames) if frame.name in train_names+test_names]
    grid = np.stack(np.meshgrid(np.linspace(-.8, .8, 9), np.linspace(-.55, .55, 7)), axis=-1).reshape(-1, 2)
    points = np.column_stack((grid, np.zeros(len(grid))))
    normals = np.tile([0., 0., -1.], (len(grid), 1))
    colors = np.full((len(grid), 3), .5)
    parameters = pack_parameters(points, normals, .16, colors)
    return write_bundle(output, [frames[i] for i in ids], train_names, [pixels[i] for i in ids],
                        parameters, np.ones(3), seed, steps,
                        {"source": "generated_analytic_plane", "synthetic": True,
                         "initialization": "known_synthetic_plane_neutral_color", "initialization_uses_heldout_observations": False},
                        recipe_hint=recipe_hint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("auto", "dtu", "colmap"), default="auto")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--train-views", type=int, default=24)
    parser.add_argument("--test-views", type=int, default=6)
    parser.add_argument("--long-edge", type=int, default=640)
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--hull-resolution", type=int, default=48)
    parser.add_argument("--quality-seed", choices=("legacy", "visible", "visible_sharp", "calibrated"), default="legacy")
    parser.add_argument("--seed-long-edge", type=int, help="Optional calibrated-feature input resolution; optimizer images retain --long-edge")
    parser.add_argument("--recipe-hint", choices=("standard", "detail"), help="Optional native training preset for this new bundle")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--images", default="images")
    parser.add_argument("--colmap-model", help="Model directory relative to scene root, e.g. sparse/1")
    parser.add_argument("--background", choices=("white", "black"), default="white")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.synthetic:
        manifest = synthetic(args.output, long_edge=args.long_edge, seed=args.seed, steps=args.steps,
                             recipe_hint=args.recipe_hint)
    else:
        if args.source is None:
            parser.error("--source is required unless --synthetic is selected")
        manifest = prepare(args.source, args.output, kind=args.format, train_count=args.train_views,
                           test_count=args.test_views, long_edge=args.long_edge, max_points=args.max_points,
                           seed=args.seed, steps=args.steps, hull_resolution=args.hull_resolution,
                           quality_seed=args.quality_seed,
                           seed_long_edge=args.seed_long_edge,
                           recipe_hint=args.recipe_hint,
                           images=args.images, background=(1, 1, 1) if args.background == "white" else (0, 0, 0),
                           colmap_model=args.colmap_model)
    LOGGER.info("Prepared %s: %d frames, %d V1 Hexels", args.output, len(manifest["frames"]), manifest["primitive_count"])


if __name__ == "__main__":
    main()
