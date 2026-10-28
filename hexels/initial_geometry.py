from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch


_FIELDS = ("means3d", "scales", "rotations", "colors")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_initial_geometry(
    path: str | Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    geometry_path = Path(path).resolve()
    if not geometry_path.is_file():
        raise ValueError(f"Initial geometry pack does not exist: {geometry_path}")
    file_sha256 = _sha256(geometry_path)
    payload = torch.load(geometry_path, map_location="cpu", weights_only=True)
    if _sha256(geometry_path) != file_sha256:
        raise RuntimeError("Initial geometry pack changed while it was loaded")
    if not isinstance(payload, Mapping) or set(payload) != set(_FIELDS):
        raise ValueError(f"Initial geometry pack must contain exactly: {', '.join(_FIELDS)}")
    if any(not isinstance(payload[name], torch.Tensor) for name in _FIELDS):
        raise ValueError("Initial geometry pack values must be tensors")

    tensors = {name: payload[name].detach().to(dtype=torch.float32) for name in _FIELDS}
    means3d = tensors["means3d"]
    scales = tensors["scales"]
    rotations = tensors["rotations"]
    colors = tensors["colors"]
    if means3d.ndim != 2 or means3d.shape[1:] != (3,) or means3d.shape[0] == 0:
        raise ValueError("Initial geometry means3d must have shape (P,3) with P > 0")
    point_count = int(means3d.shape[0])
    expected_shapes = {
        "scales": (point_count, 2),
        "rotations": (point_count, 4),
        "colors": (point_count, 3),
    }
    for name, shape in expected_shapes.items():
        if tensors[name].shape != shape:
            raise ValueError(f"Initial geometry {name} must have shape {shape}")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors.values()):
        raise ValueError("Initial geometry tensors must be finite")
    if not bool((scales > 0.0).all().item()):
        raise ValueError("Initial geometry scales must be positive")
    rotation_norms = torch.linalg.vector_norm(rotations, dim=1, keepdim=True)
    if not bool(
        (torch.isfinite(rotation_norms) & (rotation_norms > 1e-12)).all().item()
    ):
        raise ValueError("Initial geometry rotations must be normalizable")
    if not bool(((colors >= 0.0) & (colors <= 1.0)).all().item()):
        raise ValueError("Initial geometry colors must be in [0,1]")
    normalized_rotations = rotations / rotation_norms
    if not bool(torch.isfinite(normalized_rotations).all().item()):
        raise ValueError("Initial geometry rotations must be normalizable")
    tensors["rotations"] = normalized_rotations
    tensors = {name: tensor.to(device=device) for name, tensor in tensors.items()}
    metadata = {
        "path": str(geometry_path),
        "sha256": file_sha256,
        "point_count": point_count,
    }
    return tensors, metadata
