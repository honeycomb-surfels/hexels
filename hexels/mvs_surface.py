from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from .geometry_init import canonical_rotations_from_normals


MIN_SOURCE_COUNT = 1
SOURCE_COUNT_CAP = 20
RELATIVE_DEPTH_MARGIN = 0.01
SMOOTH_L1_BETA = 0.01
MAX_CAMERA_DEPTH = 100.0


@dataclass(frozen=True)
class DenseSurfaceSample:
    indices: np.ndarray
    means3d: torch.Tensor
    rotations: torch.Tensor
    base_scales: torch.Tensor
    normal_coherence: torch.Tensor
    visibility_count: torch.Tensor


@dataclass(frozen=True)
class FusedVisibility:
    offsets: np.ndarray
    camera_ids: np.ndarray


@dataclass(frozen=True)
class FusionCameraMapping:
    fusion_names: tuple[str, ...]
    registered_indices: np.ndarray


def read_fused_visibility_counts(path: Path, point_count: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<u4")
    if values.size < 2:
        raise ValueError("Fused visibility file is truncated")
    stored_count = int(values[0]) | (int(values[1]) << 32)
    if stored_count != point_count:
        raise ValueError("Fused visibility point count differs")
    counts = np.empty(point_count, dtype=np.uint32)
    offset = 2
    for index in range(point_count):
        if offset >= values.size:
            raise ValueError("Fused visibility file is truncated")
        count = int(values[offset])
        offset += 1
        if offset + count > values.size:
            raise ValueError("Fused visibility file is truncated")
        counts[index] = count
        offset += count
    if offset != values.size:
        raise ValueError("Fused visibility file has trailing data")
    return counts


def read_fused_visibility(
    path: Path,
    point_count: int,
    camera_count: int,
) -> FusedVisibility:
    """Read COLMAP fused-point visibility while preserving camera IDs."""
    if point_count < 0 or camera_count <= 0:
        raise ValueError("Fused visibility dimensions are invalid")
    values = np.memmap(path, dtype="<u4", mode="r")
    if values.size < 2:
        raise ValueError("Fused visibility file is truncated")
    stored_count = int(values[0]) | (int(values[1]) << 32)
    if stored_count != point_count:
        raise ValueError("Fused visibility point count differs")

    maximum_observations = values.size - 2 - point_count
    if maximum_observations < 0:
        raise ValueError("Fused visibility file is truncated")
    offsets = np.empty(point_count + 1, dtype="<u8")
    camera_ids = np.empty(maximum_observations, dtype="<u4")
    offsets[0] = 0
    value_offset = 2
    observation_offset = 0
    for point_index in range(point_count):
        if value_offset >= values.size:
            raise ValueError("Fused visibility file is truncated")
        count = int(values[value_offset])
        value_offset += 1
        end = value_offset + count
        if end > values.size:
            raise ValueError("Fused visibility file is truncated")
        source_ids = values[value_offset:end]
        camera_ids[observation_offset : observation_offset + count] = source_ids
        observation_offset += count
        offsets[point_index + 1] = observation_offset
        value_offset = end
    if value_offset != values.size:
        raise ValueError("Fused visibility file has trailing data")
    del values
    camera_ids = camera_ids[:observation_offset]
    if bool(np.any(camera_ids >= camera_count)):
        raise ValueError("Fused visibility contains an out-of-range camera ID")
    for start in range(0, point_count, 262_144):
        end = min(start + 262_144, point_count)
        counts = np.diff(offsets[start : end + 1]).astype(np.int64, copy=False)
        first_observation = int(offsets[start])
        last_observation = int(offsets[end])
        point_ids = np.repeat(
            np.arange(end - start, dtype=np.uint64),
            counts,
        )
        keys = (
            point_ids * np.uint64(camera_count)
            + camera_ids[first_observation:last_observation]
        )
        if len(np.unique(keys)) != len(keys):
            raise ValueError("Fused visibility contains duplicate camera IDs")
    return FusedVisibility(
        offsets=offsets,
        camera_ids=camera_ids,
    )


def subset_fused_visibility(
    visibility: FusedVisibility,
    point_indices: np.ndarray,
) -> FusedVisibility:
    """Select fused-point visibility records without changing row order."""
    indices = np.asarray(point_indices, dtype=np.int64)
    point_count = len(visibility.offsets) - 1
    if (
        indices.ndim != 1
        or bool(np.any(indices < 0))
        or bool(np.any(indices >= point_count))
    ):
        raise ValueError("Fused visibility point indices are invalid")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Fused visibility point indices must be unique")
    counts = visibility.offsets[indices + 1] - visibility.offsets[indices]
    offsets = np.empty(len(indices) + 1, dtype="<u8")
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    camera_ids = np.empty(int(offsets[-1]), dtype="<u4")
    for output_index, point_index in enumerate(indices):
        source = slice(
            int(visibility.offsets[point_index]),
            int(visibility.offsets[point_index + 1]),
        )
        target = slice(int(offsets[output_index]), int(offsets[output_index + 1]))
        camera_ids[target] = visibility.camera_ids[source]
    return FusedVisibility(offsets=offsets, camera_ids=camera_ids)


def read_fusion_camera_mapping(
    path: Path,
    registered_names: Sequence[str],
) -> FusionCameraMapping:
    """Map fusion-list camera IDs to registered camera positions."""
    lines = path.read_text().splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Fusion camera list must contain non-empty names")
    fusion_names = tuple(Path(line.strip()).stem for line in lines)
    normalized_registered = tuple(Path(name).stem for name in registered_names)
    if len(set(fusion_names)) != len(fusion_names):
        raise ValueError("Fusion camera names must be unique")
    if (
        not normalized_registered
        or any(not name for name in normalized_registered)
        or len(set(normalized_registered)) != len(normalized_registered)
    ):
        raise ValueError("Registered camera names must be non-empty and unique")
    registered_lookup = {
        name: index for index, name in enumerate(normalized_registered)
    }
    if set(fusion_names) != set(normalized_registered):
        raise ValueError("Fusion and registered camera names differ")
    return FusionCameraMapping(
        fusion_names=fusion_names,
        registered_indices=np.asarray(
            [registered_lookup[name] for name in fusion_names],
            dtype="<u4",
        ),
    )


_COORDINATE_KEY_DTYPE = np.dtype(
    [("x", "<u4"), ("y", "<u4"), ("z", "<u4")]
)


def _coordinate_keys(points: np.ndarray) -> np.ndarray:
    canonical = np.ascontiguousarray(points, dtype="<f4")
    return canonical.view("<u4").reshape(-1, 3).view(_COORDINATE_KEY_DTYPE).reshape(-1)


def match_exact_fused_rows(
    fused_points: np.ndarray,
    row_means: np.ndarray,
    *,
    chunk_size: int = 262_144,
) -> np.ndarray:
    """Recover each row's unique fused-point ID by exact float32 coordinates."""
    points = np.asarray(fused_points)
    means = np.asarray(row_means)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or means.ndim != 2
        or means.shape[1:] != (3,)
        or len(means) == 0
        or chunk_size <= 0
        or not np.isfinite(points).all()
        or not np.isfinite(means).all()
    ):
        raise ValueError("Exact fused-row inputs are invalid")

    target_keys = _coordinate_keys(means)
    target_order = np.argsort(target_keys, kind="stable")
    sorted_targets = target_keys[target_order]
    if bool(np.any(sorted_targets[1:] == sorted_targets[:-1])):
        raise ValueError("Exact fused-row targets contain duplicate coordinates")

    matches = np.full(len(means), -1, dtype=np.int64)
    for start in range(0, len(points), chunk_size):
        point_keys = _coordinate_keys(points[start : start + chunk_size])
        positions = np.searchsorted(sorted_targets, point_keys)
        in_range = positions < len(sorted_targets)
        is_match = np.zeros(len(point_keys), dtype=bool)
        is_match[in_range] = (
            sorted_targets[positions[in_range]] == point_keys[in_range]
        )
        for local_index, position in zip(
            np.flatnonzero(is_match),
            positions[is_match],
        ):
            row = int(target_order[position])
            if matches[row] >= 0:
                raise ValueError("Exact fused-row target has multiple point matches")
            matches[row] = start + int(local_index)
    if bool(np.any(matches < 0)):
        raise ValueError("Exact fused-row target is missing from the fused points")
    return matches


def _voxel_representatives(
    points: np.ndarray,
    indices: np.ndarray,
    visibility: np.ndarray,
    *,
    voxel_size: float,
) -> np.ndarray:
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    representatives: dict[tuple[int, int, int], int] = {}
    for index in np.asarray(indices, dtype=np.int64):
        key = tuple(np.floor(points[index] / voxel_size).astype(np.int64))
        current = representatives.get(key)
        if (
            current is None
            or visibility[index] > visibility[current]
            or (
                visibility[index] == visibility[current]
                and int(index) < current
            )
        ):
            representatives[key] = int(index)
    return np.asarray(sorted(representatives.values()), dtype=np.int64)


def _deterministic_normal_orientation(normals: np.ndarray) -> np.ndarray:
    axes = np.argmax(np.abs(normals), axis=1)
    signs = normals[np.arange(len(normals)), axes]
    return normals * np.where(signs < 0.0, -1.0, 1.0)[:, None]


def sample_dense_surface(
    points: np.ndarray,
    normals: np.ndarray,
    visibility: np.ndarray,
    *,
    target_count: int,
    sampler: str,
    random_seed: int,
    minimum_visibility: int = 5,
    normal_neighbors: int = 16,
    minimum_normal_coherence: float = 0.85,
    voxel_size: float = 0.02,
    scale_neighbors: int = 3,
) -> DenseSurfaceSample:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    visibility = np.asarray(visibility)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or normals.shape != points.shape
        or visibility.shape != (len(points),)
    ):
        raise ValueError("Dense surface inputs have incompatible shapes")
    if (
        target_count <= 0
        or normal_neighbors < 3
        or scale_neighbors < 1
        or sampler not in {"density", "voxel"}
    ):
        raise ValueError("Dense surface sampling settings are invalid")
    lengths = np.linalg.norm(normals, axis=1)
    eligible = np.flatnonzero(
        np.isfinite(points).all(axis=1)
        & np.isfinite(normals).all(axis=1)
        & (lengths > 1e-12)
        & (visibility >= minimum_visibility)
    )
    if sampler == "voxel":
        eligible = _voxel_representatives(
            points,
            eligible,
            visibility,
            voxel_size=voxel_size,
        )
    if len(eligible) < target_count:
        raise ValueError("Dense surface has too few eligible samples")
    rng = np.random.default_rng(random_seed)
    probabilities = visibility[eligible].astype(np.float64)
    probabilities /= probabilities.sum()
    selected = np.sort(
        rng.choice(
            eligible,
            size=target_count,
            replace=False,
            p=probabilities,
        )
    )

    tree = cKDTree(points)
    _, normal_indices = tree.query(
        points[selected],
        k=min(normal_neighbors, len(points)),
        workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "-1")),
    )
    neighbor_normals = normals[normal_indices]
    neighbor_normals /= np.linalg.norm(neighbor_normals, axis=2, keepdims=True)
    orientation = np.einsum(
        "pki,pkj->pij",
        neighbor_normals,
        neighbor_normals,
    ) / neighbor_normals.shape[1]
    values, vectors = np.linalg.eigh(orientation)
    coherence = values[:, -1]
    if np.any(coherence < minimum_normal_coherence):
        raise ValueError("Selected dense surface normals are incoherent")
    estimated_normals = _deterministic_normal_orientation(vectors[:, :, -1])

    distances, _ = tree.query(
        points[selected],
        k=min(scale_neighbors + 1, len(points)),
        workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "-1")),
    )
    if distances.ndim == 1:
        distances = distances[:, None]
    base_scales = distances[:, 1:].mean(axis=1)
    if np.any(~np.isfinite(base_scales)) or np.any(base_scales <= 0.0):
        raise ValueError("Dense surface scales are invalid")
    means = torch.from_numpy(points[selected]).to(torch.float32)
    normal_tensor = torch.from_numpy(estimated_normals).to(torch.float32)
    return DenseSurfaceSample(
        indices=selected,
        means3d=means,
        rotations=canonical_rotations_from_normals(normal_tensor),
        base_scales=torch.from_numpy(base_scales).to(torch.float32),
        normal_coherence=torch.from_numpy(coherence).to(torch.float32),
        visibility_count=torch.from_numpy(visibility[selected].copy()).to(torch.long),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_colmap_header(handle: Any) -> tuple[int, int, int]:
    fields: list[bytes] = []
    field = bytearray()
    while len(fields) < 3:
        value = handle.read(1)
        if not value:
            raise ValueError("COLMAP dense file has an incomplete header")
        if value == b"&":
            fields.append(bytes(field))
            field.clear()
        else:
            field.extend(value)
    width, height, channels = (int(value) for value in fields)
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError("COLMAP dense file has invalid dimensions")
    return width, height, channels


def read_colmap_dense_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        width, height, channels = _read_colmap_header(handle)
        values = np.fromfile(handle, dtype=np.float32)
    expected = width * height * channels
    if values.size != expected:
        raise ValueError(
            f"COLMAP dense array has {values.size} values, expected {expected}"
        )
    array = values.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze(axis=2)


def read_colmap_consistency_counts(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        width, height, channels = _read_colmap_header(handle)
        values = np.fromfile(handle, dtype=np.int32)
    if channels != 1:
        raise ValueError("COLMAP consistency graph must have one channel")
    counts = np.zeros((height, width), dtype=np.uint8)
    offset = 0
    while offset < values.size:
        if offset + 3 > values.size:
            raise ValueError("COLMAP consistency graph has a truncated record")
        x, y, source_count = (int(value) for value in values[offset : offset + 3])
        offset += 3
        if (
            x < 0
            or x >= width
            or y < 0
            or y >= height
            or source_count < 0
            or offset + source_count > values.size
        ):
            raise ValueError("COLMAP consistency graph has an invalid record")
        counts[y, x] = min(source_count, 255)
        offset += source_count
    return counts


def camera_nearest_indices(
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    source_intrinsics: tuple[float, float, float, float],
    target_focal: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    source_height, source_width = source_size
    target_height, target_width = target_size
    source_fx, source_fy, source_cx, source_cy = source_intrinsics
    target_fx, target_fy = target_focal
    if min(
        source_height,
        source_width,
        target_height,
        target_width,
    ) <= 0 or min(source_fx, source_fy, target_fx, target_fy) <= 0.0:
        raise ValueError("Camera target mapping has invalid dimensions or focal length")
    target_x = torch.arange(target_width, dtype=torch.float64)
    target_y = torch.arange(target_height, dtype=torch.float64)
    source_x = (
        (target_x - (target_width - 1) / 2.0) * (source_fx / target_fx)
        + source_cx
    )
    source_y = (
        (target_y - (target_height - 1) / 2.0) * (source_fy / target_fy)
        + source_cy
    )
    source_x = torch.floor(source_x + 0.5).to(torch.long).clamp(0, source_width - 1)
    source_y = torch.floor(source_y + 0.5).to(torch.long).clamp(0, source_height - 1)
    return source_y, source_x


def camera_nearest_gather(
    value: torch.Tensor,
    *,
    target_size: tuple[int, int],
    source_intrinsics: tuple[float, float, float, float],
    target_focal: tuple[float, float],
) -> torch.Tensor:
    if value.ndim != 2:
        raise ValueError("MVS target must have shape (H,W)")
    source_y, source_x = camera_nearest_indices(
        source_size=tuple(value.shape),
        target_size=target_size,
        source_intrinsics=source_intrinsics,
        target_focal=target_focal,
    )
    return value.index_select(0, source_y).index_select(1, source_x)


def prepare_mvs_surface_target(
    depth: np.ndarray,
    source_count: np.ndarray,
    size: tuple[int, int],
    *,
    source_intrinsics: tuple[float, float, float, float],
    target_focal: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if depth.ndim != 2 or source_count.shape != depth.shape:
        raise ValueError("MVS depth and source-count shapes differ")
    depth_tensor = camera_nearest_gather(
        torch.from_numpy(depth.copy()),
        target_size=size,
        source_intrinsics=source_intrinsics,
        target_focal=target_focal,
    )
    count_tensor = camera_nearest_gather(
        torch.from_numpy(source_count.copy()),
        target_size=size,
        source_intrinsics=source_intrinsics,
        target_focal=target_focal,
    )
    valid = (
        torch.isfinite(depth_tensor)
        & (depth_tensor > 0.0)
        & (depth_tensor <= MAX_CAMERA_DEPTH)
        & (count_tensor >= MIN_SOURCE_COUNT)
    )
    depth_tensor = torch.where(valid, depth_tensor, torch.zeros_like(depth_tensor))
    count_tensor = torch.where(
        valid,
        count_tensor.clamp(max=SOURCE_COUNT_CAP),
        torch.zeros_like(count_tensor),
    )
    return depth_tensor.to(torch.float32), count_tensor.to(torch.uint8)


@dataclass(frozen=True)
class MvsSurfaceLoss:
    depth: torch.Tensor
    foreground_spill: torch.Tensor
    fixed_mean_absolute_relative_depth: torch.Tensor
    fixed_foreground_spill_fraction: torch.Tensor
    valid_alpha: torch.Tensor
    fixed_absolute_relative_depth_sum: torch.Tensor
    fixed_foreground_spill_sum: torch.Tensor
    valid_alpha_sum: torch.Tensor
    confidence_sum: torch.Tensor
    alpha_weighted_absolute_relative_depth_sum: torch.Tensor
    alpha_weighted_foreground_spill_sum: torch.Tensor
    alpha_weight_sum: torch.Tensor


def rendered_mvs_surface_loss(
    surf_depth: torch.Tensor,
    surf_alpha: torch.Tensor,
    target_depth: torch.Tensor,
    source_count: torch.Tensor,
) -> MvsSurfaceLoss:
    if (
        surf_depth.ndim != 2
        or surf_alpha.shape != surf_depth.shape
        or target_depth.shape != surf_depth.shape
        or source_count.shape != surf_depth.shape
    ):
        raise ValueError("Rendered and MVS surface tensors must share shape (H,W)")
    valid = (
        torch.isfinite(target_depth)
        & (target_depth > 0.0)
        & (source_count >= MIN_SOURCE_COUNT)
    )
    confidence = (
        source_count.to(dtype=surf_depth.dtype).clamp(max=SOURCE_COUNT_CAP)
        / float(SOURCE_COUNT_CAP)
    ) * valid.to(dtype=surf_depth.dtype)
    alpha = surf_alpha.clamp_min(0.0)
    rendered_depth = surf_depth / alpha.clamp_min(1e-6)
    relative_error = (
        rendered_depth - target_depth
    ) / target_depth.clamp_min(1e-6)
    confidence_sum = confidence.sum()
    confidence_denominator = confidence_sum.clamp_min(1e-6)
    depth_penalty = F.smooth_l1_loss(
        relative_error,
        torch.zeros_like(relative_error),
        reduction="none",
        beta=SMOOTH_L1_BETA,
    )
    foreground = F.relu(-relative_error - RELATIVE_DEPTH_MARGIN)
    spill_penalty = F.smooth_l1_loss(
        foreground,
        torch.zeros_like(foreground),
        reduction="none",
        beta=SMOOTH_L1_BETA,
    )
    foreground_indicator = (
        relative_error.detach() < -RELATIVE_DEPTH_MARGIN
    ).to(alpha.dtype)
    fixed_absolute_relative_depth_sum = (
        confidence * relative_error.abs()
    ).sum()
    fixed_foreground_spill_sum = (confidence * foreground_indicator).sum()
    valid_alpha_sum = (confidence * alpha.detach()).sum()
    alpha_weights = confidence * alpha.detach()
    alpha_weighted_absolute_relative_depth_sum = (
        alpha_weights * relative_error.abs()
    ).sum()
    alpha_weighted_foreground_spill_sum = (
        alpha_weights * foreground_indicator
    ).sum()
    alpha_weight_sum = alpha_weights.sum()
    return MvsSurfaceLoss(
        depth=(confidence * depth_penalty).sum() / confidence_denominator,
        foreground_spill=(confidence * spill_penalty).sum()
        / confidence_denominator,
        fixed_mean_absolute_relative_depth=fixed_absolute_relative_depth_sum
        / confidence_denominator,
        fixed_foreground_spill_fraction=fixed_foreground_spill_sum
        / confidence_denominator,
        valid_alpha=valid_alpha_sum / confidence_denominator,
        fixed_absolute_relative_depth_sum=fixed_absolute_relative_depth_sum,
        fixed_foreground_spill_sum=fixed_foreground_spill_sum,
        valid_alpha_sum=valid_alpha_sum,
        confidence_sum=confidence_sum,
        alpha_weighted_absolute_relative_depth_sum=(
            alpha_weighted_absolute_relative_depth_sum
        ),
        alpha_weighted_foreground_spill_sum=(
            alpha_weighted_foreground_spill_sum
        ),
        alpha_weight_sum=alpha_weight_sum,
    )


@dataclass
class MvsSurfaceTargets:
    camera_names: tuple[str, ...]
    depth: torch.Tensor
    source_count: torch.Tensor
    metadata: dict[str, Any]

    def target(
        self,
        camera_name: str,
        *,
        device: torch.device,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            index = self.camera_names.index(camera_name)
        except ValueError as error:
            raise ValueError(f"MVS surface target is missing camera {camera_name}") from error
        if tuple(self.depth.shape[1:]) != size:
            raise ValueError(
                f"MVS target size {tuple(self.depth.shape[1:])} differs from camera {size}"
            )
        return (
            self.depth[index].to(device=device, non_blocking=True),
            self.source_count[index].to(device=device, non_blocking=True),
        )


def load_mvs_surface_targets(
    path: Path,
    *,
    camera_names: Sequence[str],
    validate_tensors: bool = True,
) -> MvsSurfaceTargets:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("MVS surface sidecar must contain a mapping")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "mvs_surface_targets"
    ):
        raise ValueError("MVS surface sidecar schema differs")
    stored_names = payload.get("camera_names")
    depth = payload.get("depth")
    source_count = payload.get("source_count")
    metadata = payload.get("metadata")
    if (
        not isinstance(stored_names, list)
        or any(not isinstance(name, str) for name in stored_names)
        or not isinstance(depth, torch.Tensor)
        or not isinstance(source_count, torch.Tensor)
        or not isinstance(metadata, Mapping)
    ):
        raise ValueError("MVS surface sidecar fields are invalid")
    expected_names = list(camera_names)
    if stored_names != expected_names:
        raise ValueError("MVS surface camera names or order differ")
    if (
        depth.dtype != torch.float32
        or source_count.dtype != torch.uint8
        or depth.ndim != 3
        or source_count.shape != depth.shape
        or depth.shape[0] != len(stored_names)
    ):
        raise ValueError("MVS surface sidecar tensor shapes or dtypes differ")
    if (
        metadata.get("minimum_source_count") != MIN_SOURCE_COUNT
        or metadata.get("source_count_cap") != SOURCE_COUNT_CAP
        or metadata.get("maximum_camera_depth") != MAX_CAMERA_DEPTH
        or metadata.get("resize_mode") != "camera-intrinsics-nearest"
    ):
        raise ValueError("MVS surface construction settings differ")
    if validate_tensors:
        depth_valid = (depth > 0.0) & (depth <= MAX_CAMERA_DEPTH)
        count_valid = source_count >= MIN_SOURCE_COUNT
        valid_per_camera = (
            (depth_valid & count_valid)
            .reshape(len(stored_names), -1)
            .any(dim=1)
        )
        if (
            not bool(torch.isfinite(depth).all().item())
            or bool((depth < 0.0).any().item())
            or bool((depth > MAX_CAMERA_DEPTH).any().item())
            or bool((depth_valid != count_valid).any().item())
            or bool((source_count > SOURCE_COUNT_CAP).any().item())
            or not bool(valid_per_camera.all().item())
        ):
            raise ValueError("MVS surface sidecar tensors are invalid")
    names_sha256 = hashlib.sha256(
        json.dumps(stored_names, separators=(",", ":")).encode()
    ).hexdigest()
    if metadata.get("camera_names_sha256") != names_sha256:
        raise ValueError("MVS surface camera-name hash differs")
    return MvsSurfaceTargets(
        camera_names=tuple(stored_names),
        depth=depth,
        source_count=source_count,
        metadata={**dict(metadata), "sidecar_sha256": sha256_file(path)},
    )
