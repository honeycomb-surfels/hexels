from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .edge_junction import canonical_vertex_edge_pairs, world_hexel_vertices
from .edge_shift import canonical_edge_normals


_EDGE_VERTEX_IDS = (
    (0, 1),
    (2, 5),
    (0, 4),
    (2, 3),
    (3, 4),
    (1, 5),
)


@dataclass(frozen=True)
class HexelWireframe:
    vertices: Tensor
    edges: Tensor
    faces: Tensor
    primitive_ids: Tensor
    side_ids: Tensor
    plane_normals: Tensor
    vertex_colors: Tensor | None


def extract_hexel_wireframe(
    means3d: Tensor,
    log_scales: Tensor,
    rotations: Tensor,
    *,
    scale_modifier: float = 1.0,
    active_mask: Tensor | None = None,
    colors: Tensor | None = None,
    color_grads: Tensor | None = None,
) -> HexelWireframe:
    """Convert Hexels to differentiable colored surface cells and line segments."""
    primitive_count = means3d.shape[0]
    if active_mask is None:
        primitive_ids = torch.arange(
            primitive_count,
            device=means3d.device,
            dtype=torch.long,
        )
    else:
        if active_mask.shape != (primitive_count,) or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be a bool vector with shape (P,)")
        primitive_ids = torch.nonzero(active_mask, as_tuple=False).flatten()

    selected_means = means3d.index_select(0, primitive_ids)
    selected_log_scales = log_scales.index_select(0, primitive_ids)
    selected_rotations = rotations.index_select(0, primitive_ids)
    vertices, plane_normals = world_hexel_vertices(
        selected_means,
        selected_log_scales,
        selected_rotations,
        scale_modifier=scale_modifier,
    )

    selected_count = primitive_ids.numel()
    vertices = vertices.reshape(-1, 3)
    local_edges = torch.tensor(
        _EDGE_VERTEX_IDS,
        device=means3d.device,
        dtype=torch.long,
    )
    offsets = 6 * torch.arange(
        selected_count,
        device=means3d.device,
        dtype=torch.long,
    )
    edges = (local_edges[None, :, :] + offsets[:, None, None]).reshape(-1, 2)
    vertex_cycle = torch.tensor(
        [0, 4, 3, 2, 5, 1],
        device=means3d.device,
        dtype=torch.long,
    )
    faces = vertex_cycle[None, :] + offsets[:, None]
    side_ids = torch.arange(6, device=means3d.device, dtype=torch.long).repeat(
        selected_count
    )

    if colors is None:
        if color_grads is not None:
            raise ValueError("color_grads requires colors")
        vertex_colors = None
    else:
        if colors.shape != (primitive_count, 3):
            raise ValueError("colors must have shape (P,3)")
        selected_colors = colors.index_select(0, primitive_ids)
        vertex_colors = selected_colors[:, None, :].expand(-1, 6, -1)
        if color_grads is not None:
            if color_grads.shape != (primitive_count, 2, 3):
                raise ValueError("color_grads must have shape (P,2,3)")
            edge_normals = canonical_edge_normals(
                device=means3d.device,
                dtype=means3d.dtype,
            )
            pairs = canonical_vertex_edge_pairs(device=means3d.device)
            local_vertices = (2.0 / 3.0) * (
                edge_normals[pairs[:, 0]] + edge_normals[pairs[:, 1]]
            )
            selected_grads = color_grads.index_select(0, primitive_ids)
            vertex_colors = vertex_colors + torch.einsum(
                "vd,pdc->pvc",
                local_vertices,
                selected_grads,
            )
        vertex_colors = vertex_colors.reshape(-1, 3)

    return HexelWireframe(
        vertices=vertices,
        edges=edges,
        faces=faces,
        primitive_ids=primitive_ids.repeat_interleave(6),
        side_ids=side_ids,
        plane_normals=plane_normals,
        vertex_colors=vertex_colors,
    )
