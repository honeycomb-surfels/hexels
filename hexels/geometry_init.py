from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree


def canonical_rotations_from_normals(normals: torch.Tensor) -> torch.Tensor:
    """Return zero-twist WXYZ rotations that map local +z onto each normal."""
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must have shape (P,3), got {tuple(normals.shape)}")
    if not normals.is_floating_point():
        raise ValueError("normals must have a floating-point dtype")
    if not bool(torch.isfinite(normals).all().item()):
        raise ValueError("normals must be finite")
    source_dtype = normals.dtype
    work = normals.to(torch.float64)
    lengths = torch.linalg.vector_norm(work, dim=1, keepdim=True)
    if not bool((lengths > 1e-12).all().item()):
        raise ValueError("normals must be nonzero")

    unit = work / lengths
    zeros = torch.zeros_like(unit[:, 0])
    rotations = torch.stack(
        [1.0 + unit[:, 2], -unit[:, 1], unit[:, 0], zeros],
        dim=1,
    )
    antiparallel = torch.linalg.vector_norm(rotations, dim=1) <= 1e-7
    if bool(antiparallel.any().item()):
        rotations = rotations.clone()
        rotations[antiparallel] = rotations.new_tensor([0.0, 1.0, 0.0, 0.0])
    return F.normalize(rotations, dim=1).to(source_dtype)


def confidence_gated_pca_rotations(
    points: torch.Tensor, *, neighbors: int = 16, min_planarity: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """Estimate supported planar normals without changing centers or scales."""
    if points.shape[0] <= neighbors or neighbors < 3:
        raise ValueError("PCA requires more points than its neighborhood size")
    coordinates = points.detach().cpu().double().numpy()
    tree = cKDTree(coordinates)
    normals = np.zeros_like(coordinates)
    normals[:, 2] = 1.0
    accepted_count = 0
    for start in range(0, len(coordinates), 8192):
        stop = min(start + 8192, len(coordinates))
        _, indices = tree.query(coordinates[start:stop], k=neighbors + 1, workers=1)
        local = coordinates[indices[:, 1:]]
        centered = local - local.mean(axis=1, keepdims=True)
        covariance = np.einsum("pki,pkj->pij", centered, centered) / neighbors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        planarity = (eigenvalues[:, 1] - eigenvalues[:, 0]) / np.maximum(eigenvalues[:, 2], 1e-12)
        variation = eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-12)
        accepted = (planarity >= min_planarity) & (variation <= 0.05)
        candidates = eigenvectors[:, :, 0]
        largest = np.argmax(np.abs(candidates), axis=1)
        signs = candidates[np.arange(len(candidates)), largest]
        candidates *= np.where(signs < 0, -1.0, 1.0)[:, None]
        normals[start:stop][accepted] = candidates[accepted]
        accepted_count += int(accepted.sum())
    rotations = canonical_rotations_from_normals(torch.as_tensor(normals, device=points.device, dtype=points.dtype))
    return rotations, accepted_count / len(coordinates)


def pca_initial_geometry(
    points: torch.Tensor,
    *,
    scale_k: int,
    pca_k: int,
    scale_multiplier: float,
    max_scale: float,
    query_chunk_size: int = 65_536,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RMS-kNN scales and shortest-arc rotations to local PCA normals."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P,3), got {tuple(points.shape)}")
    if not points.is_floating_point():
        raise ValueError("points must have a floating-point dtype")
    if not bool(torch.isfinite(points).all().item()):
        raise ValueError("points must be finite")
    if scale_k < 1:
        raise ValueError("scale_k must be positive")
    if pca_k < 3:
        raise ValueError("pca_k must be at least 3")
    if points.shape[0] <= max(scale_k, pca_k):
        raise ValueError("point count must exceed scale_k and pca_k")
    if not math.isfinite(scale_multiplier) or scale_multiplier <= 0.0:
        raise ValueError("scale_multiplier must be finite and positive")
    if not math.isfinite(max_scale):
        raise ValueError("max_scale must be finite")
    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")

    points_np = points.detach().to(device="cpu", dtype=torch.float64).numpy()
    tree = cKDTree(points_np)
    rms = np.empty(points.shape[0], dtype=np.float64)
    normals = np.empty((points.shape[0], 3), dtype=np.float64)
    query_k = max(scale_k, pca_k) + 1

    for start in range(0, points.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, points.shape[0])
        distances, indices = tree.query(
            points_np[start:stop],
            k=query_k,
            workers=-1,
        )
        rms[start:stop] = np.sqrt(np.mean(distances[:, 1 : scale_k + 1] ** 2, axis=1))

        neighbors = tree.data[indices[:, 1 : pca_k + 1]]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("pki,pkj->pij", centered, centered) / float(pca_k)
        normals[start:stop] = np.linalg.eigh(covariance)[1][:, :, 0]

    largest_components = np.argmax(np.abs(normals), axis=1)
    normal_signs = normals[np.arange(normals.shape[0]), largest_components]
    normals *= np.where(normal_signs < 0.0, -1.0, 1.0)[:, None]

    rotations = np.stack(
        [
            1.0 + normals[:, 2],
            -normals[:, 1],
            normals[:, 0],
            np.zeros(normals.shape[0], dtype=normals.dtype),
        ],
        axis=1,
    )
    rotations /= np.linalg.norm(rotations, axis=1, keepdims=True)

    scales = torch.from_numpy(rms).to(device=points.device, dtype=points.dtype)
    scales = scales.clamp_min(1e-7**0.5) * float(scale_multiplier)
    if max_scale > 0.0:
        scales = scales.clamp_max(float(max_scale))
    scales = scales[:, None].repeat(1, 2)
    rotations_tensor = torch.from_numpy(rotations).to(
        device=points.device,
        dtype=points.dtype,
    )
    return scales, rotations_tensor
