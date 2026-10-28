from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


_MATERIAL_FRACTION = 0.1
_MIN_MATERIAL_VIEWS = 8
_AGREEMENT_NUMERATOR = 2
_AGREEMENT_DENOMINATOR = 3


def canonical_edge_normals(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the six signed Hexel edge normals in native tie order.

    The CUDA SDF prefers ``abs(a*u-b*v)`` over ``abs(a*u+b*v)`` on a tie,
    and either slanted family over ``abs(v)``. Positive signs win the final
    zero-valued tie.
    """
    a = 0.5 * math.sqrt(3.0)
    return torch.tensor(
        [
            [a, -0.5],
            [-a, 0.5],
            [a, 0.5],
            [-a, -0.5],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )


def material_residual_mask(
    persistent_mass: Tensor,
    contribution: Tensor,
    *,
    fraction: float = _MATERIAL_FRACTION,
) -> Tensor:
    """Select persistent, materially attributed primitives in one camera."""
    if persistent_mass.ndim != 1 or contribution.shape != persistent_mass.shape:
        raise ValueError("persistent_mass and contribution must be matching vectors")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0,1]")
    if not bool(torch.isfinite(persistent_mass).all()) or bool((persistent_mass < 0.0).any()):
        raise ValueError("persistent_mass must be finite and non-negative")
    if not bool(torch.isfinite(contribution).all()) or bool((contribution < 0.0).any()):
        raise ValueError("contribution must be finite and non-negative")

    visible = contribution > 0.0
    positive_ids = torch.nonzero(visible & (persistent_mass > 0.0), as_tuple=False).flatten()
    material = torch.zeros_like(visible)
    if positive_ids.numel() == 0:
        return material

    density = torch.zeros_like(persistent_mass)
    density[visible] = persistent_mass[visible] / contribution[visible]
    count = max(1, math.ceil(fraction * positive_ids.numel()))
    order = torch.argsort(persistent_mass[positive_ids], descending=True, stable=True)
    top_ids = positive_ids[order[:count]]
    material[top_ids] = density[top_ids] >= torch.median(density[visible])
    return material


def persistent_residual_weights(native: Tensor, filtered: Tensor, target: Tensor) -> Tensor:
    """Return detached RGB residual magnitude that remains after 2x filtering."""
    if native.shape != filtered.shape or native.shape != target.shape:
        raise ValueError("native, filtered, and target must have matching shapes")
    return torch.minimum(
        torch.abs(native.detach() - target.detach()),
        torch.abs(filtered.detach() - target.detach()),
    )


def first_moment_edge_ids(mass: Tensor, moment: Tensor) -> Tensor:
    """Map local residual first moments to the canonical signed Hexel edges."""
    if mass.ndim != 1 or moment.shape != (mass.shape[0], 2):
        raise ValueError("mass and moment must have shapes (P,) and (P,2)")
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0.0).any()):
        raise ValueError("mass must be finite and non-negative")
    if not bool(torch.isfinite(moment).all()):
        raise ValueError("moment must be finite")

    normals = canonical_edge_normals(device=moment.device, dtype=moment.dtype)
    edge_ids = torch.argmax(moment @ normals.T, dim=1)
    return edge_ids.masked_fill(mass <= 0.0, -1)


def residual_first_moments(
    color_gradient: Tensor,
    color_grads_gradient: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recover persistent mass, local centroid, and signed edge from appearance VJPs."""
    if color_gradient.ndim != 2 or color_gradient.shape[1] != 3:
        raise ValueError("color_gradient must have shape (P,3)")
    if color_grads_gradient.shape != (color_gradient.shape[0], 2, 3):
        raise ValueError("color_grads_gradient must have shape (P,2,3)")
    if not bool(torch.isfinite(color_gradient).all()) or not bool(
        torch.isfinite(color_grads_gradient).all()
    ):
        raise ValueError("appearance gradients must be finite")
    if bool((color_gradient < -1e-4).any()):
        raise ValueError("color_gradient contains negative compositing mass")

    mass = color_gradient.clamp_min(0.0).sum(dim=1)
    numerator = color_grads_gradient.sum(dim=2)
    centroid = torch.zeros_like(numerator)
    positive = mass > 0.0
    centroid[positive] = numerator[positive] / mass[positive, None]
    edge_ids = first_moment_edge_ids(mass, numerator)
    return mass, centroid, edge_ids


def update_residual_topk_views(
    top_scores: Tensor,
    top_centroids: Tensor,
    scores: Tensor,
    centroids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Insert one camera's residual evidence while preserving score-centroid pairs."""
    if top_scores.ndim != 2 or top_scores.shape[1] < 1:
        raise ValueError("top_scores must have shape (P,K) with K > 0")
    if top_centroids.shape != (*top_scores.shape, 2):
        raise ValueError("top_centroids must have shape (P,K,2)")
    if scores.shape != (top_scores.shape[0],):
        raise ValueError("scores must have shape (P,)")
    if centroids.shape != (top_scores.shape[0], 2):
        raise ValueError("centroids must have shape (P,2)")
    if not bool(torch.isfinite(top_scores).all()) or bool((top_scores < 0.0).any()):
        raise ValueError("top_scores must be finite and non-negative")
    if not bool(torch.isfinite(scores).all()) or bool((scores < 0.0).any()):
        raise ValueError("scores must be finite and non-negative")
    if not bool(torch.isfinite(top_centroids).all()) or not bool(
        torch.isfinite(centroids).all()
    ):
        raise ValueError("centroids must be finite")

    scores = scores.to(device=top_scores.device, dtype=top_scores.dtype)
    centroids = centroids.to(device=top_centroids.device, dtype=top_centroids.dtype)
    combined_scores = torch.cat([top_scores, scores[:, None]], dim=1)
    combined_centroids = torch.cat([top_centroids, centroids[:, None, :]], dim=1)
    order = torch.argsort(combined_scores, dim=1, descending=True, stable=True)
    order = order[:, : top_scores.shape[1]]
    return (
        torch.gather(combined_scores, 1, order),
        torch.gather(
            combined_centroids,
            1,
            order[:, :, None].expand(-1, -1, 2),
        ),
    )


@dataclass(frozen=True)
class ResidualTopKSummary:
    dominant_edge_ids: Tensor
    view_counts: Tensor
    dominant_edge_vote_counts: Tensor
    consistency: Tensor
    kth_scores: Tensor
    centroids: Tensor


def summarize_residual_topk_views(
    top_scores: Tensor,
    top_centroids: Tensor,
    edge_vote_counts: Tensor,
) -> ResidualTopKSummary:
    """Summarize dominant-edge persistent residual evidence for each primitive."""
    if top_scores.ndim != 3 or top_scores.shape[1] != 6 or top_scores.shape[2] < 1:
        raise ValueError("top_scores must have shape (P,6,K) with K > 0")
    if top_centroids.shape != (*top_scores.shape, 2):
        raise ValueError("top_centroids must have shape (P,6,K,2)")
    if edge_vote_counts.shape != top_scores.shape[:2]:
        raise ValueError("edge_vote_counts must have shape (P,6)")

    dominant_votes, dominant_edges = torch.max(edge_vote_counts, dim=1)
    view_counts = edge_vote_counts.sum(dim=1)
    rows = torch.arange(top_scores.shape[0], device=top_scores.device)
    dominant_scores = top_scores[rows, dominant_edges]
    dominant_centroids = top_centroids[rows, dominant_edges]
    score_sum = dominant_scores.sum(dim=1)
    centroids = torch.sum(
        dominant_scores[:, :, None] * dominant_centroids,
        dim=1,
    ) / score_sum.clamp_min(torch.finfo(top_scores.dtype).tiny)[:, None]
    consistency = dominant_votes.to(dtype=top_scores.dtype) / view_counts.clamp_min(1).to(
        dtype=top_scores.dtype
    )
    return ResidualTopKSummary(
        dominant_edge_ids=dominant_edges,
        view_counts=view_counts,
        dominant_edge_vote_counts=dominant_votes,
        consistency=consistency,
        kth_scores=dominant_scores[:, -1],
        centroids=centroids,
    )


def _quat_tangents(rotations: Tensor) -> tuple[Tensor, Tensor]:
    norms = torch.linalg.vector_norm(rotations, dim=-1, keepdim=True)
    if not bool(torch.isfinite(rotations).all()) or bool((norms <= 0.0).any()):
        raise ValueError("rotations must be finite nonzero WXYZ quaternions")
    w, x, y, z = (rotations / norms).unbind(dim=-1)
    tangent_u = torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dim=-1,
    )
    tangent_v = torch.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        dim=-1,
    )
    return tangent_u, tangent_v


def _field_directions(
    scales: Tensor,
    rotations: Tensor,
    color_grads: Tensor,
    edge_ids: Tensor,
    *,
    scale_modifier: float,
) -> tuple[Tensor, Tensor]:
    leading_shape = scales.shape[:-1]
    if scales.shape[-1:] != (2,):
        raise ValueError("scales must have shape (...,2)")
    if rotations.shape != (*leading_shape, 4):
        raise ValueError("rotations must have shape (...,4)")
    if color_grads.shape != (*leading_shape, 2, 3):
        raise ValueError("color_grads must have shape (...,2,3)")
    if edge_ids.shape != leading_shape or edge_ids.dtype != torch.long:
        raise ValueError("edge_ids must be a long tensor matching the leading shape")
    if bool(((edge_ids < 0) | (edge_ids >= 6)).any()):
        raise ValueError("edge_ids must be in [0,6)")
    if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
        raise ValueError("scale_modifier must be finite and positive")
    if not bool(torch.isfinite(scales).all()) or bool((scales <= 0.0).any()):
        raise ValueError("scales must be finite and positive")
    if not bool(torch.isfinite(color_grads).all()):
        raise ValueError("color_grads must be finite")

    normals = canonical_edge_normals(device=scales.device, dtype=scales.dtype)
    local_direction = normals[edge_ids]
    tangent_u, tangent_v = _quat_tangents(rotations)
    world_direction = float(scale_modifier) * (
        tangent_u * (scales[..., 0] * local_direction[..., 0])[..., None]
        + tangent_v * (scales[..., 1] * local_direction[..., 1])[..., None]
    )
    color_direction = torch.sum(color_grads * local_direction[..., :, None], dim=-2)
    return world_direction, color_direction


def field_preserving_edge_derivatives(
    grad_means: Tensor,
    grad_colors: Tensor,
    scales: Tensor,
    rotations: Tensor,
    color_grads: Tensor,
    edge_ids: Tensor,
    *,
    scale_modifier: float,
) -> Tensor:
    """Return ``d loss / d epsilon`` for a positive field-preserving edge shift."""
    if grad_means.shape[-1:] != (3,) or grad_colors.shape != grad_means.shape:
        raise ValueError("grad_means and grad_colors must be matching (...,3) tensors")
    if grad_means.shape[:-1] != scales.shape[:-1]:
        raise ValueError("gradient and primitive leading shapes must match")
    if not bool(torch.isfinite(grad_means).all()) or not bool(torch.isfinite(grad_colors).all()):
        raise ValueError("loss gradients must be finite")
    world_direction, color_direction = _field_directions(
        scales,
        rotations,
        color_grads,
        edge_ids,
        scale_modifier=scale_modifier,
    )
    return torch.sum(grad_means * world_direction, dim=-1) + torch.sum(
        grad_colors * color_direction,
        dim=-1,
    )


@dataclass(frozen=True)
class EdgeShiftSelection:
    parent_ids: Tensor
    edge_ids: Tensor
    shift_signs: Tensor
    scores: Tensor
    material_view_counts: Tensor
    edge_vote_counts: Tensor
    derivative_agreement_counts: Tensor


def select_edge_shifts(
    persistent_mass: Tensor,
    material_mask: Tensor,
    edge_ids: Tensor,
    directional_derivatives: Tensor,
    eligible_mask: Tensor,
    *,
    num_parents: int,
    min_material_views: int = _MIN_MATERIAL_VIEWS,
    agreement_numerator: int = _AGREEMENT_NUMERATOR,
    agreement_denominator: int = _AGREEMENT_DENOMINATOR,
) -> EdgeShiftSelection:
    """Select exactly ``num_parents`` multiview-consistent edge shifts or fail."""
    if persistent_mass.ndim != 2:
        raise ValueError("persistent_mass must have shape (V,P)")
    shape = persistent_mass.shape
    if material_mask.shape != shape or material_mask.dtype != torch.bool:
        raise ValueError("material_mask must be bool with shape (V,P)")
    if edge_ids.shape != shape or edge_ids.dtype != torch.long:
        raise ValueError("edge_ids must be long with shape (V,P)")
    if directional_derivatives.shape != shape:
        raise ValueError("directional_derivatives must have shape (V,P)")
    if eligible_mask.shape != (shape[1],) or eligible_mask.dtype != torch.bool:
        raise ValueError("eligible_mask must be bool with shape (P,)")
    if not bool(torch.isfinite(persistent_mass).all()) or bool((persistent_mass < 0.0).any()):
        raise ValueError("persistent_mass must be finite and non-negative")
    if not bool(torch.isfinite(directional_derivatives).all()):
        raise ValueError("directional_derivatives must be finite")
    if bool((((edge_ids < 0) | (edge_ids >= 6)) & material_mask).any()):
        raise ValueError("material edge_ids must be in [0,6)")
    if num_parents < 1:
        raise ValueError("num_parents must be positive")
    if min_material_views < 1:
        raise ValueError("min_material_views must be positive")
    if not (0 < agreement_numerator <= agreement_denominator):
        raise ValueError("agreement fraction must be in (0,1]")

    material_counts = material_mask.sum(dim=0)
    edge_counts = torch.stack(
        [((edge_ids == edge) & material_mask).sum(dim=0) for edge in range(6)],
        dim=1,
    )
    edge_agreement_counts, dominant_edges = torch.max(edge_counts, dim=1)
    edge_consistent = (
        edge_agreement_counts * agreement_denominator
        >= material_counts * agreement_numerator
    )

    dominant_edge_views = material_mask & (edge_ids == dominant_edges.unsqueeze(0))
    negative_counts = ((directional_derivatives < 0.0) & dominant_edge_views).sum(dim=0)
    positive_counts = ((directional_derivatives > 0.0) & dominant_edge_views).sum(dim=0)
    positive_wins = positive_counts > negative_counts
    derivative_signs = torch.where(
        positive_wins,
        torch.ones_like(positive_counts),
        -torch.ones_like(negative_counts),
    )
    derivative_agreement_counts = torch.maximum(negative_counts, positive_counts)
    derivative_consistent = (
        derivative_agreement_counts * agreement_denominator
        >= edge_agreement_counts * agreement_numerator
    )

    candidates = (
        eligible_mask
        & (material_counts >= min_material_views)
        & edge_consistent
        & derivative_consistent
    )
    candidate_ids = torch.nonzero(candidates, as_tuple=False).flatten()
    if candidate_ids.numel() < num_parents:
        raise ValueError(
            f"Only {candidate_ids.numel()} parents pass the frozen edge gates; "
            f"{num_parents} required"
        )

    scores = torch.where(dominant_edge_views, persistent_mass, 0.0).to(torch.float64).sum(dim=0)
    order = torch.argsort(scores[candidate_ids], descending=True, stable=True)
    selected = candidate_ids[order[:num_parents]]
    return EdgeShiftSelection(
        parent_ids=selected,
        edge_ids=dominant_edges[selected],
        shift_signs=-derivative_signs[selected],
        scores=scores[selected],
        material_view_counts=material_counts[selected],
        edge_vote_counts=edge_agreement_counts[selected],
        derivative_agreement_counts=derivative_agreement_counts[selected],
    )


def _shifted_fields(
    means: Tensor,
    colors: Tensor,
    scales: Tensor,
    rotations: Tensor,
    color_grads: Tensor,
    edge_ids: Tensor,
    shift_signs: Tensor,
    *,
    shift_amount: float,
    scale_modifier: float,
) -> tuple[Tensor, Tensor]:
    if means.shape[-1:] != (3,) or colors.shape != means.shape:
        raise ValueError("means and colors must be matching (...,3) tensors")
    if shift_signs.shape != means.shape[:-1]:
        raise ValueError("shift_signs must match the primitive leading shape")
    if not bool(torch.isfinite(shift_signs).all()) or bool((torch.abs(shift_signs) != 1).any()):
        raise ValueError("shift_signs must contain only -1 or +1")
    if not math.isfinite(shift_amount) or shift_amount < 0.0:
        raise ValueError("shift_amount must be finite and non-negative")
    world_direction, color_direction = _field_directions(
        scales,
        rotations,
        color_grads,
        edge_ids,
        scale_modifier=scale_modifier,
    )
    signed_amount = shift_signs.to(dtype=means.dtype) * float(shift_amount)
    return (
        means + world_direction * signed_amount[..., None],
        colors + color_direction * signed_amount[..., None],
    )


class PackedEdgeShiftModel:
    """Render-time model wrapper for a reversible field-preserving parent shift."""

    def __init__(
        self,
        model: Any,
        parent_ids: Tensor,
        edge_ids: Tensor,
        shift_signs: Tensor,
        *,
        shift_amount: float,
        scale_modifier: float,
    ) -> None:
        if parent_ids.ndim != 1 or parent_ids.dtype != torch.long:
            raise ValueError("parent_ids must be a long vector")
        if edge_ids.shape != parent_ids.shape or edge_ids.dtype != torch.long:
            raise ValueError("edge_ids must be a long vector matching parent_ids")
        if shift_signs.shape != parent_ids.shape:
            raise ValueError("shift_signs must match parent_ids")
        if parent_ids.numel() != torch.unique(parent_ids).numel() or bool((parent_ids < 0).any()):
            raise ValueError("parent_ids must be unique and non-negative")
        if bool(((edge_ids < 0) | (edge_ids >= 6)).any()):
            raise ValueError("edge_ids must be in [0,6)")
        if not math.isfinite(shift_amount) or shift_amount < 0.0:
            raise ValueError("shift_amount must be finite and non-negative")
        if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
            raise ValueError("scale_modifier must be finite and positive")
        self.model = model
        self.parent_ids = parent_ids.detach().clone()
        self.edge_ids = edge_ids.detach().clone()
        self.shift_signs = shift_signs.detach().clone()
        self.shift_amount = float(shift_amount)
        self.scale_modifier = float(scale_modifier)
        self.num_base = int(model.num_base)
        self.num_children = int(model.num_children)

    def pack_for_rasterizer(self, **kwargs: object) -> dict[str, Tensor]:
        pack = dict(self.model.pack_for_rasterizer(**kwargs))
        base_ids = pack["base_ids"]
        if base_ids.ndim != 1 or base_ids.dtype != torch.long:
            raise ValueError("packed base_ids must be a long vector")
        if base_ids.numel() > 1 and bool((base_ids[1:] <= base_ids[:-1]).any()):
            raise ValueError("packed base_ids must be strictly increasing")

        parent_ids = self.parent_ids.to(device=base_ids.device)
        positions = torch.searchsorted(base_ids, parent_ids)
        in_range = positions < base_ids.numel()
        if not bool(in_range.all()) or not torch.equal(base_ids[positions], parent_ids):
            raise ValueError("Every shifted parent must be an active packed base primitive")
        if self.shift_amount == 0.0:
            return pack
        edge_ids = self.edge_ids.to(device=base_ids.device)
        shift_signs = self.shift_signs.to(device=base_ids.device)
        shifted_means, shifted_colors = _shifted_fields(
            pack["means3d"].index_select(0, positions),
            pack["colors"].index_select(0, positions),
            pack["scales"].index_select(0, positions),
            pack["rotations"].index_select(0, positions),
            pack["color_grads"].index_select(0, positions),
            edge_ids,
            shift_signs,
            shift_amount=self.shift_amount,
            scale_modifier=self.scale_modifier,
        )
        means3d = pack["means3d"].clone(memory_format=torch.preserve_format)
        colors = pack["colors"].clone(memory_format=torch.preserve_format)
        means3d.index_copy_(0, positions, shifted_means)
        colors.index_copy_(0, positions, shifted_colors)
        pack["means3d"] = means3d
        pack["colors"] = colors
        return pack
