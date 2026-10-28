"""Colored surfel polygons and conservative, geometry-only boundary contacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContactOptions:
    neighbors: int = 16
    max_angle_degrees: float = 30.0
    max_endpoint_fraction: float = 0.25
    max_plane_fraction: float = 0.1
    max_length_ratio: float = 2.0
    max_vertex_shift_fraction: float = 0.2


def rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Convert normalized WXYZ quaternions to local-to-world rotations."""
    quaternions = np.asarray(quaternions, dtype=np.float64)
    lengths = np.linalg.norm(quaternions, axis=1)
    if np.any(lengths == 0) or not np.isfinite(quaternions).all():
        raise ValueError("Rotations must be finite nonzero WXYZ quaternions")
    real, axis_x, axis_y, axis_z = (quaternions / lengths[:, None]).T
    return np.stack([
        1 - 2 * (axis_y**2 + axis_z**2), 2 * (axis_x * axis_y - real * axis_z),
        2 * (axis_x * axis_z + real * axis_y), 2 * (axis_x * axis_y + real * axis_z),
        1 - 2 * (axis_x**2 + axis_z**2), 2 * (axis_y * axis_z - real * axis_x),
        2 * (axis_x * axis_z - real * axis_y), 2 * (axis_y * axis_z + real * axis_x),
        1 - 2 * (axis_x**2 + axis_y**2),
    ], axis=1).reshape(-1, 3, 3)


def primitive_polygons(
    centers: np.ndarray, scales: np.ndarray, rotations: np.ndarray,
    *, family: str, gaussian_sides: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Hexel SDF-zero boundary or Gaussian half-peak contour, in world coordinates."""
    if family not in {"hexels", "2dgs"} or gaussian_sides < 6:
        raise ValueError("Expected hexels/2dgs and at least six Gaussian sides")
    if centers.shape != (len(scales), 3) or scales.shape[1:] != (2,):
        raise ValueError("Expected centers (P,3) and tangent scales (P,2)")
    if not np.isfinite(centers).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Centers/scales must be finite with positive scales")
    sides = 6 if family == "hexels" else gaussian_sides
    radius = 2 / np.sqrt(3) if family == "hexels" else np.sqrt(2 * np.log(2))
    angles = np.arange(sides) * (2 * np.pi / sides)
    local = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    matrices = rotation_matrices(rotations)
    corners = centers[:, None] + np.einsum("pij,psj->psi", matrices[:, :, :2],
                                          scales[:, None] * local[None])
    return corners, matrices[:, :, 2]


def find_contacts(
    corners: np.ndarray, normals: np.ndarray, options: ContactOptions,
    *, workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Match mutual-best sides; reject stacks, sharp folds and large endpoint gaps."""
    count, sides, _ = corners.shape
    starts = corners.reshape(-1, 3)
    ends = np.roll(corners, -1, axis=1).reshape(-1, 3)
    midpoints = (starts + ends) / 2
    tangents = ends - starts
    lengths = np.linalg.norm(tangents, axis=1)
    directions = tangents / lengths[:, None]
    outward = np.cross(directions, np.repeat(normals, sides, axis=0))
    centers = corners.mean(axis=1)
    tree = cKDTree(midpoints)
    best = np.full(len(starts), -1, dtype=np.int64)
    reverse = np.zeros(len(starts), dtype=bool)
    scores = np.full(len(starts), np.inf)
    cosine = np.cos(np.deg2rad(options.max_angle_degrees))
    neighbors = min(options.neighbors + sides, len(starts))
    for offset in range(0, len(starts), 16384):
        edge_ids = np.arange(offset, min(offset + 16384, len(starts)))
        _, candidates = tree.query(midpoints[edge_ids], k=neighbors, workers=workers)
        owners = edge_ids[:, None] // sides
        other_owners = candidates // sides
        minimum = np.minimum(lengths[edge_ids, None], lengths[candidates])
        maximum = np.maximum(lengths[edge_ids, None], lengths[candidates])
        aligned = np.maximum(
            np.linalg.norm(starts[edge_ids, None] - starts[candidates], axis=2),
            np.linalg.norm(ends[edge_ids, None] - ends[candidates], axis=2))
        flipped = np.maximum(
            np.linalg.norm(starts[edge_ids, None] - ends[candidates], axis=2),
            np.linalg.norm(ends[edge_ids, None] - starts[candidates], axis=2))
        mismatch = np.minimum(aligned, flipped) / minimum
        delta = centers[other_owners] - centers[owners]
        plane_gap = np.maximum(
            np.abs(np.sum(delta * normals[owners], axis=2)),
            np.abs(np.sum(delta * normals[other_owners], axis=2)))
        valid = (owners != other_owners) & (maximum <= options.max_length_ratio * minimum)
        valid &= np.abs(np.sum(normals[owners] * normals[other_owners], axis=2)) >= cosine
        valid &= np.abs(np.sum(directions[edge_ids, None] * directions[candidates], axis=2)) >= cosine
        valid &= np.sum(outward[edge_ids, None] * outward[candidates], axis=2) <= -cosine
        valid &= np.sum(delta * outward[edge_ids, None], axis=2) > 0
        valid &= np.sum(delta * outward[candidates], axis=2) < 0
        valid &= plane_gap <= options.max_plane_fraction * minimum
        valid &= mismatch <= options.max_endpoint_fraction
        ranked = np.where(valid, mismatch, np.inf)
        selected = ranked.argmin(axis=1)
        rows = np.arange(len(edge_ids))
        found = np.isfinite(ranked[rows, selected])
        chosen = edge_ids[found]
        best[chosen] = candidates[rows[found], selected[found]]
        reverse[chosen] = flipped[rows[found], selected[found]] < aligned[rows[found], selected[found]]
        scores[chosen] = ranked[rows[found], selected[found]]
    first = np.flatnonzero(best >= 0)
    first = first[(best[best[first]] == first) & (first < best[first])]
    second = best[first]
    pairs = np.stack([first, second], axis=1)
    next_first = (first // sides) * sides + (first + 1) % sides
    next_second = (second // sides) * sides + (second + 1) % sides
    endpoint_pairs = np.concatenate([
        np.stack([first, np.where(reverse[first], next_second, second)], axis=1),
        np.stack([next_first, np.where(reverse[first], second, next_second)], axis=1),
    ])
    diagnostics = {
        "candidate_contacts": len(pairs),
        "candidate_isolated_primitive_fraction": float(1 - len(np.unique(pairs // sides)) / count),
        "mean_endpoint_mismatch_fraction": float(scores[first].mean()) if len(first) else None,
        "contact_evidence": "geometry_only_no_visibility_test",
    }
    return pairs, endpoint_pairs, diagnostics


def nondegenerate_patches(corners: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Identify patches representable as positive-area triangles in float64 world space."""
    relative = corners - centers[:, None]
    areas = np.linalg.norm(np.cross(relative, np.roll(relative, -1, axis=1)), axis=2)
    return (areas > 0).all(axis=1)


def weld_contacts(
    corners: np.ndarray, colors: np.ndarray, endpoint_pairs: np.ndarray,
    *, max_shift_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Share contact vertices, rejecting entire groups that would move too far."""
    vertices = corners.reshape(-1, 3)
    vertex_colors = np.repeat(colors, corners.shape[1], axis=0)
    graph = coo_matrix((np.ones(len(endpoint_pairs)), endpoint_pairs.T),
                       shape=(len(vertices), len(vertices))).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    counts = np.bincount(labels)
    averages = np.stack([np.bincount(labels, weights=vertices[:, axis]) / counts
                         for axis in range(3)], axis=1)
    lengths = np.linalg.norm(np.roll(corners, -1, axis=1) - corners, axis=2)
    local_size = np.minimum(lengths, np.roll(lengths, 1, axis=1)).ravel()
    displacement = np.linalg.norm(vertices - averages[labels], axis=1)
    rejected = np.zeros(component_count, dtype=bool)
    rejected[labels[displacement > max_shift_fraction * local_size]] = True
    labels = np.where(rejected[labels], component_count + np.arange(len(vertices)), labels)
    _, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse)
    merged = np.stack([np.bincount(inverse, weights=vertices[:, axis]) / counts
                       for axis in range(3)], axis=1)
    merged_colors = np.stack([np.bincount(inverse, weights=vertex_colors[:, axis]) / counts
                              for axis in range(3)], axis=1)
    return merged, merged_colors, inverse.reshape(corners.shape[:2]), {
        "welded_vertices_removed": len(vertices) - len(merged),
        "rejected_vertex_groups": int(rejected.sum()),
        "max_vertex_displacement": float(np.linalg.norm(vertices - merged[inverse], axis=1).max()),
    }


def triangulate_patches(
    vertices: np.ndarray, vertex_colors: np.ndarray, corner_ids: np.ndarray,
    centers: np.ndarray, colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fan-fill each patch for surface evaluation, retaining boundary edges separately."""
    center_ids = np.broadcast_to(np.arange(len(centers))[:, None] + len(vertices), corner_ids.shape)
    next_ids = np.roll(corner_ids, -1, axis=1)
    faces = np.stack([center_ids, corner_ids, next_ids], axis=-1).reshape(-1, 3)
    edges = np.unique(np.sort(np.stack([corner_ids, next_ids], axis=-1).reshape(-1, 2), axis=1), axis=0)
    return (np.concatenate([vertices, centers]), np.concatenate([vertex_colors, colors]), faces, edges)
