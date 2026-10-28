from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class NormalTrustSidecar:
    reference_means3d: torch.Tensor
    reference_rotations: torch.Tensor
    reliability: torch.Tensor
    metadata: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quaternion_normal_z(rotations: torch.Tensor) -> torch.Tensor:
    rotations = torch.nn.functional.normalize(rotations, dim=-1)
    w, x, y, z = rotations.unbind(dim=-1)
    return torch.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quaternion_from_two_vectors(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    source = torch.nn.functional.normalize(source, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    dot = (source * target).sum(dim=-1, keepdim=True)
    cross = torch.cross(source, target, dim=-1)
    value = torch.cat((1.0 + dot, cross), dim=-1)
    return torch.nn.functional.normalize(value, dim=-1)


def project_normal_trust_cone(
    rotations: torch.Tensor,
    reference_normals: torch.Tensor,
    cone_radians: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project plane normals into sign-invariant cones while retaining tangent twist."""
    if (
        rotations.ndim != 2
        or rotations.shape[1] != 4
        or reference_normals.shape != (len(rotations), 3)
        or cone_radians.shape != (len(rotations),)
    ):
        raise ValueError("Normal trust tensors have incompatible shapes")
    current_rotations = torch.nn.functional.normalize(rotations, dim=-1)
    current_normals = quaternion_normal_z(current_rotations)
    reference_normals = torch.nn.functional.normalize(reference_normals, dim=-1)
    signed_dot = (current_normals * reference_normals).sum(dim=-1)
    aligned_reference = torch.where(
        (signed_dot < 0.0)[:, None],
        -reference_normals,
        reference_normals,
    )
    cosine = signed_dot.abs().clamp(0.0, 1.0)
    before = torch.acos(cosine)
    projected_rows = before > cone_radians
    if not bool(projected_rows.any().item()):
        return current_rotations, projected_rows, before, before

    tangent = current_normals - cosine[:, None] * aligned_reference
    tangent = torch.nn.functional.normalize(tangent, dim=-1)
    target_normals = (
        torch.cos(cone_radians)[:, None] * aligned_reference
        + torch.sin(cone_radians)[:, None] * tangent
    )
    delta = _quaternion_from_two_vectors(current_normals, target_normals)
    projected = torch.nn.functional.normalize(
        _quaternion_multiply(delta, current_rotations),
        dim=-1,
    )
    output = torch.where(projected_rows[:, None], projected, current_rotations)
    after = torch.acos(
        (
            quaternion_normal_z(output) * reference_normals
        ).sum(dim=-1).abs().clamp(0.0, 1.0)
    )
    return output, projected_rows, before, after


def load_normal_trust_sidecar(
    path: Path,
    device: torch.device,
) -> NormalTrustSidecar:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Normal trust sidecar must contain a mapping")
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "normal_trust"
    ):
        raise ValueError("Normal trust sidecar schema differs")
    means = payload.get("reference_means3d")
    rotations = payload.get("reference_rotations")
    reliability = payload.get("reliability")
    if (
        not isinstance(means, torch.Tensor)
        or not isinstance(rotations, torch.Tensor)
        or not isinstance(reliability, torch.Tensor)
        or means.ndim != 2
        or means.shape[1] != 3
        or rotations.shape != (len(means), 4)
        or reliability.shape != (len(means),)
    ):
        raise ValueError("Normal trust sidecar tensor shapes differ")
    if (
        not bool(torch.isfinite(means).all().item())
        or not bool(torch.isfinite(rotations).all().item())
        or not bool(torch.isfinite(reliability).all().item())
        or bool((torch.linalg.vector_norm(rotations, dim=1) <= 1e-12).any().item())
        or bool(((reliability < 0.0) | (reliability > 1.0)).any().item())
    ):
        raise ValueError("Normal trust sidecar tensors are invalid")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Normal trust sidecar metadata is missing")
    return NormalTrustSidecar(
        reference_means3d=means.to(torch.float32),
        reference_rotations=torch.nn.functional.normalize(
            rotations.to(torch.float32),
            dim=1,
        ),
        reliability=reliability.to(torch.float32),
        metadata={**dict(metadata), "sidecar_sha256": sha256_file(path)},
    )
