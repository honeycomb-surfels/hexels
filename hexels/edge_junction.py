from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .edge_shift import canonical_edge_normals, material_residual_mask
from .refinement import _quat_to_tangents


_VERTEX_EDGE_PAIRS = (
    (0, 2),
    (0, 5),
    (1, 3),
    (1, 4),
    (2, 4),
    (3, 5),
)


@dataclass(frozen=True)
class EdgeJunctionSelection:
    parent_ids: Tensor
    vertex_edge_ids: Tensor
    scores: Tensor
    view_counts: Tensor
    member_counts: Tensor
    stable_member_parent_ids: Tensor
    stable_member_vertex_edge_ids: Tensor
    stable_view_counts: Tensor


def canonical_vertex_edge_pairs(*, device: torch.device | str | None = None) -> Tensor:
    """Return the adjacent signed-edge pair for each canonical Hexel vertex."""
    return torch.tensor(_VERTEX_EDGE_PAIRS, device=device, dtype=torch.long)


def _edge_vertex_ids(*, device: torch.device | str | None = None) -> Tensor:
    pairs = canonical_vertex_edge_pairs(device=device)
    result = torch.empty((6, 2), device=device, dtype=torch.long)
    for edge_id in range(6):
        vertex_ids = torch.nonzero(pairs == edge_id, as_tuple=False)[:, 0]
        result[edge_id] = torch.sort(vertex_ids).values
    return result


def world_hexel_vertices(
    means3d: Tensor,
    log_scales: Tensor,
    rotations: Tensor,
    *,
    scale_modifier: float,
) -> tuple[Tensor, Tensor]:
    """Return world-space vertices and unoriented plane normals."""
    primitive_count = means3d.shape[0]
    if means3d.shape != (primitive_count, 3):
        raise ValueError("means3d must have shape (P,3)")
    if log_scales.shape != (primitive_count, 2):
        raise ValueError("log_scales must have shape (P,2)")
    if rotations.shape != (primitive_count, 4):
        raise ValueError("rotations must have shape (P,4)")
    if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
        raise ValueError("scale_modifier must be finite and positive")
    if not all(bool(torch.isfinite(value).all()) for value in (means3d, log_scales, rotations)):
        raise ValueError("Hexel geometry must be finite")
    if bool((torch.linalg.vector_norm(rotations, dim=1) == 0.0).any()):
        raise ValueError("Rotation quaternions must be nonzero")

    tangent_u, tangent_v = _quat_to_tangents(rotations)
    plane_normals = torch.nn.functional.normalize(
        torch.linalg.cross(tangent_u, tangent_v),
        dim=1,
    )
    edge_normals = canonical_edge_normals(device=means3d.device, dtype=means3d.dtype)
    pairs = canonical_vertex_edge_pairs(device=means3d.device)
    local_vertices = (2.0 / 3.0) * (
        edge_normals[pairs[:, 0]] + edge_normals[pairs[:, 1]]
    )
    scales = torch.exp(log_scales)
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("Hexel scales must be finite")
    offsets = float(scale_modifier) * (
        tangent_u[:, None, :]
        * (scales[:, None, 0, None] * local_vertices[None, :, 0, None])
        + tangent_v[:, None, :]
        * (scales[:, None, 1, None] * local_vertices[None, :, 1, None])
    )
    return means3d[:, None, :] + offsets, plane_normals


def _project_vertices(
    vertices: Tensor,
    world_view_transform: Tensor,
    full_proj_transform: Tensor,
    image_size: Tensor,
    *,
    margin: float,
) -> tuple[Tensor, Tensor, Tensor]:
    ones = torch.ones((*vertices.shape[:-1], 1), dtype=vertices.dtype, device=vertices.device)
    homogeneous = torch.cat((vertices, ones), dim=-1)
    camera = homogeneous @ world_view_transform
    clip = homogeneous @ full_proj_transform
    denominator = clip[:, 3]
    valid = (denominator > 0.0) & (camera[:, 2] > 0.0)
    ndc = clip[:, :2] / denominator[:, None].clamp_min(torch.finfo(vertices.dtype).eps)
    width, height = image_size
    pixels = torch.stack(
        (
            0.5 * (ndc[:, 0] + 1.0) * width - 0.5,
            0.5 * (ndc[:, 1] + 1.0) * height - 0.5,
        ),
        dim=1,
    )
    valid &= (
        (pixels[:, 0] >= -margin)
        & (pixels[:, 0] <= width - 1.0 + margin)
        & (pixels[:, 1] >= -margin)
        & (pixels[:, 1] <= height - 1.0 + margin)
    )
    return pixels, camera[:, 2], valid


def _coherent(
    first: int,
    second: int,
    *,
    depths: Tensor,
    normals: Tensor,
    max_relative_depth: float,
    min_normal_dot: float,
) -> bool:
    depth_scale = max(abs(float(depths[first])), abs(float(depths[second])), 1e-12)
    if abs(float(depths[first] - depths[second])) > max_relative_depth * depth_scale:
        return False
    return abs(float(torch.dot(normals[first], normals[second]))) >= min_normal_dot


def _triangles(
    adjacency: list[set[int]],
    primitive_ids: list[int],
) -> set[tuple[int, int, int]]:
    triangles: set[tuple[int, int, int]] = set()
    for first, neighbors in enumerate(adjacency):
        for second in neighbors:
            if second <= first:
                continue
            for third in neighbors & adjacency[second]:
                if third <= second:
                    continue
                if len({primitive_ids[first], primitive_ids[second], primitive_ids[third]}) == 3:
                    triangles.add((first, second, third))
    return triangles


def select_edge_junctions(
    *,
    persistent_mass: Tensor,
    contribution: Tensor,
    edge_ids: Tensor,
    eligible_mask: Tensor,
    means3d: Tensor,
    log_scales: Tensor,
    rotations: Tensor,
    world_view_transforms: Tensor,
    full_proj_transforms: Tensor,
    image_sizes: Tensor,
    scale_modifier: float,
    material_fraction: float = 0.1,
    pixel_radius: float = 1.5,
    max_relative_depth: float = 0.01,
    normal_max_angle_deg: float = 20.0,
    min_views: int = 3,
    max_junctions: int = 128,
) -> EdgeJunctionSelection:
    """Select recurrent three-Hexel vertex junctions from edge-direction signals.

    Each qualifying vertex must join three distinct, materially erroneous Hexels
    within ``pixel_radius`` in at least ``min_views``. Candidate endpoints must
    also agree in camera depth and world-space plane normal.
    """
    if persistent_mass.ndim != 2 or contribution.shape != persistent_mass.shape:
        raise ValueError("persistent_mass and contribution must have shape (V,P)")
    view_count, primitive_count = persistent_mass.shape
    if edge_ids.shape != (view_count, primitive_count) or edge_ids.dtype != torch.long:
        raise ValueError("edge_ids must be a long tensor with shape (V,P)")
    if eligible_mask.shape != (primitive_count,) or eligible_mask.dtype != torch.bool:
        raise ValueError("eligible_mask must be a bool vector with shape (P,)")
    if world_view_transforms.shape != (view_count, 4, 4):
        raise ValueError("world_view_transforms must have shape (V,4,4)")
    if full_proj_transforms.shape != (view_count, 4, 4):
        raise ValueError("full_proj_transforms must have shape (V,4,4)")
    if image_sizes.shape != (view_count, 2):
        raise ValueError("image_sizes must have shape (V,2)")
    if bool((image_sizes <= 0.0).any()):
        raise ValueError("image_sizes must be positive")
    if bool(((edge_ids < -1) | (edge_ids >= 6)).any()):
        raise ValueError("edge_ids must be in [-1,6)")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (
            persistent_mass,
            contribution,
            world_view_transforms,
            full_proj_transforms,
            image_sizes,
        )
    ):
        raise ValueError("Junction inputs must be finite")
    if bool((persistent_mass < 0.0).any()) or bool((contribution < 0.0).any()):
        raise ValueError("Mass and contribution must be non-negative")
    if (
        not math.isfinite(pixel_radius)
        or not math.isfinite(max_relative_depth)
        or pixel_radius <= 0.0
        or max_relative_depth < 0.0
    ):
        raise ValueError("pixel_radius must be positive and max_relative_depth non-negative")
    if not 0.0 < normal_max_angle_deg < 90.0:
        raise ValueError("normal_max_angle_deg must be in (0,90)")
    if min_views < 2 or max_junctions < 1:
        raise ValueError("min_views must be at least two and max_junctions positive")

    vertices, plane_normals = world_hexel_vertices(
        means3d,
        log_scales,
        rotations,
        scale_modifier=scale_modifier,
    )
    edge_vertices = _edge_vertex_ids(device=edge_ids.device)
    support_counts = torch.zeros((primitive_count, 6), dtype=torch.int64)
    scores = torch.zeros((primitive_count, 6), dtype=torch.float64)
    member_counts = torch.zeros((primitive_count, 6), dtype=torch.int64)
    view_triangles: list[set[tuple[int, int, int]]] = [set() for _ in range(view_count)]
    triangle_view_counts: dict[tuple[int, int, int], int] = {}
    min_normal_dot = math.cos(math.radians(normal_max_angle_deg))
    radius_squared = float(pixel_radius) ** 2

    for view_index in range(view_count):
        material = material_residual_mask(
            persistent_mass[view_index],
            contribution[view_index],
            fraction=material_fraction,
        )
        primitive_ids = torch.nonzero(
            material & eligible_mask & (edge_ids[view_index] >= 0),
            as_tuple=False,
        ).flatten()
        if primitive_ids.numel() < 3:
            continue
        selected_edges = edge_ids[view_index, primitive_ids]
        vertex_ids = edge_vertices[selected_edges].reshape(-1)
        primitive_ids = primitive_ids.repeat_interleave(2)
        selected_vertices = vertices[primitive_ids, vertex_ids]
        pixels, depths, valid = _project_vertices(
            selected_vertices,
            world_view_transforms[view_index],
            full_proj_transforms[view_index],
            image_sizes[view_index],
            margin=float(pixel_radius),
        )
        primitive_ids = primitive_ids[valid]
        vertex_ids = vertex_ids[valid]
        pixels = pixels[valid]
        depths = depths[valid]
        if primitive_ids.numel() < 3:
            continue
        normals = plane_normals[primitive_ids]
        primitive_list = primitive_ids.tolist()
        vertex_list = vertex_ids.tolist()

        buckets: dict[tuple[int, int], list[int]] = {}
        for index, pixel in enumerate(pixels):
            cell = (
                math.floor(float(pixel[0]) / pixel_radius),
                math.floor(float(pixel[1]) / pixel_radius),
            )
            buckets.setdefault(cell, []).append(index)
        adjacency = [set() for _ in range(len(primitive_list))]
        for first, pixel in enumerate(pixels):
            cell_x = math.floor(float(pixel[0]) / pixel_radius)
            cell_y = math.floor(float(pixel[1]) / pixel_radius)
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    for second in buckets.get((cell_x + offset_x, cell_y + offset_y), ()):
                        if second <= first or primitive_list[first] == primitive_list[second]:
                            continue
                        if float(torch.sum((pixel - pixels[second]).square())) > radius_squared:
                            continue
                        if not _coherent(
                            first,
                            second,
                            depths=depths,
                            normals=normals,
                            max_relative_depth=max_relative_depth,
                            min_normal_dot=min_normal_dot,
                        ):
                            continue
                        adjacency[first].add(second)
                        adjacency[second].add(first)

        triangles = _triangles(adjacency, primitive_list)
        if not triangles:
            continue
        triangle_keys = {
            tuple(
                sorted(
                    primitive_list[index] * 6 + vertex_list[index]
                    for index in triangle
                )
            )
            for triangle in triangles
        }
        view_triangles[view_index] = triangle_keys
        for triangle in triangle_keys:
            triangle_view_counts[triangle] = triangle_view_counts.get(triangle, 0) + 1

    recurrent_triangles = {
        triangle
        for triangle, count in triangle_view_counts.items()
        if count >= min_views
    }
    pair_view_counts: dict[tuple[int, int], int] = {}
    for view_index, triangles in enumerate(view_triangles):
        supported_keys: set[int] = set()
        members_by_key: dict[int, set[int]] = {}
        supported_pairs: set[tuple[int, int]] = set()
        for triangle in triangles & recurrent_triangles:
            parent_members = {key // 6 for key in triangle}
            for key in triangle:
                supported_keys.add(key)
                members_by_key.setdefault(key, set()).update(parent_members)
            supported_pairs.update(
                {
                    (triangle[0], triangle[1]),
                    (triangle[0], triangle[2]),
                    (triangle[1], triangle[2]),
                }
            )
        for key in supported_keys:
            parent_id, vertex_id = divmod(key, 6)
            support_counts[parent_id, vertex_id] += 1
            scores[parent_id, vertex_id] += float(persistent_mass[view_index, parent_id])
            coherent_member_count = len(members_by_key[key])
            member_counts[parent_id, vertex_id] = max(
                member_counts[parent_id, vertex_id],
                coherent_member_count,
            )
        for pair in supported_pairs:
            pair_view_counts[pair] = pair_view_counts.get(pair, 0) + 1

    candidate_keys = torch.nonzero(
        support_counts.reshape(-1) >= min_views,
        as_tuple=False,
    ).flatten().tolist()
    candidate_keys.sort(
        key=lambda key: (
            -float(scores.reshape(-1)[key]),
            -int(support_counts.reshape(-1)[key]),
            key,
        )
    )
    selected_keys: list[int] = []
    selected_parents: set[int] = set()
    for key in candidate_keys:
        parent_id = key // 6
        if parent_id in selected_parents:
            continue
        if any(
            pair_view_counts.get((min(key, other), max(key, other)), 0) >= min_views
            for other in selected_keys
        ):
            continue
        selected_keys.append(key)
        selected_parents.add(parent_id)
        if len(selected_keys) == max_junctions:
            break

    keys = torch.tensor(selected_keys, dtype=torch.long)
    parents = torch.div(keys, 6, rounding_mode="floor")
    vertex_ids = torch.remainder(keys, 6)
    stable_triangles = [
        min(
            (triangle for triangle in recurrent_triangles if key in triangle),
            key=lambda triangle: (-triangle_view_counts[triangle], triangle),
        )
        for key in selected_keys
    ]
    stable_keys = torch.tensor(stable_triangles, dtype=torch.long).reshape(-1, 3)
    stable_vertex_ids = torch.remainder(stable_keys, 6)
    return EdgeJunctionSelection(
        parent_ids=parents,
        vertex_edge_ids=canonical_vertex_edge_pairs()[vertex_ids],
        scores=scores.reshape(-1)[keys],
        view_counts=support_counts.reshape(-1)[keys],
        member_counts=member_counts.reshape(-1)[keys],
        stable_member_parent_ids=torch.div(stable_keys, 6, rounding_mode="floor"),
        stable_member_vertex_edge_ids=canonical_vertex_edge_pairs()[stable_vertex_ids],
        stable_view_counts=torch.tensor(
            [triangle_view_counts[triangle] for triangle in stable_triangles],
            dtype=torch.int64,
        ),
    )
