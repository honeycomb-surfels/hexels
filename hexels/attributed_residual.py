from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .edge_shift import canonical_edge_normals


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


@dataclass(frozen=True)
class OwnerSupportMasks:
    """Mutually exclusive per-pixel owner-support classifications."""

    unowned_or_very_weak: Tensor
    weak_owned: Tensor
    strong_owned: Tensor
    ambiguous_overlap: Tensor


@dataclass(frozen=True)
class AttributedResidualAggregation:
    """One camera's attributed residual statistics.

    ``mass`` has shape ``(P, 6)``, ``centroid_mass`` has shape
    ``(P, 6, 2)``, and ``camera_vote_mask`` marks the dominant positive bin
    for each primitive.
    """

    mass: Tensor
    centroid_mass: Tensor
    camera_vote_mask: Tensor


@dataclass(frozen=True)
class AttributedEdgeSummary:
    """Camera-complete attributed evidence reduced to one edge per primitive."""

    dominant_edge_ids: Tensor
    view_counts: Tensor
    dominant_edge_vote_counts: Tensor
    consistency: Tensor
    mass: Tensor
    centroids: Tensor


@dataclass(frozen=True)
class OwnerLocalMomentAggregation:
    """One camera's owner-local residual moments in canonical Hexel UV."""

    coverage: Tensor
    energy: Tensor
    robust_weight: Tensor
    first_moment: Tensor
    second_moment: Tensor
    principal_axis: Tensor
    axis_projector: Tensor
    double_angle: Tensor
    axis_valid: Tensor
    side_mass: Tensor


@dataclass(frozen=True)
class OwnerLocalMomentSummary:
    """Accumulated owner-local moments and bounded symmetric birth geometry."""

    coverage: Tensor
    energy: Tensor
    robust_weight: Tensor
    score: Tensor
    mean: Tensor
    covariance: Tensor
    principal_axis: Tensor
    axis_projector: Tensor
    lambda1: Tensor
    lambda2: Tensor
    anisotropy: Tensor
    double_angle_sum: Tensor
    axis_view_count: Tensor
    axis_coherence: Tensor
    material_view_count: Tensor
    side_mass: Tensor
    side_balance: Tensor
    center: Tensor
    delta_raw: Tensor
    delta_max: Tensor
    delta: Tensor
    pair_offsets: Tensor
    axis_valid: Tensor
    eligible: Tensor


def _validate_cpu_tensor(tensor: Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")


def _validate_owner_fields(
    owner_ids: Tensor,
    owner_weights: Tensor,
    alpha: Tensor,
    *,
    num_primitives: int,
) -> None:
    if not isinstance(num_primitives, int) or isinstance(num_primitives, bool):
        raise ValueError("num_primitives must be an integer")
    if num_primitives < 1:
        raise ValueError("num_primitives must be positive")
    if owner_ids.dtype not in _INTEGER_DTYPES:
        raise ValueError("owner_ids must have an integer dtype")
    if not owner_weights.is_floating_point() or not alpha.is_floating_point():
        raise ValueError("owner_weights and alpha must be floating-point tensors")
    if owner_ids.shape != owner_weights.shape or owner_ids.shape != alpha.shape:
        raise ValueError("owner_ids, owner_weights, and alpha must have matching shapes")
    for tensor, name in (
        (owner_ids, "owner_ids"),
        (owner_weights, "owner_weights"),
        (alpha, "alpha"),
    ):
        _validate_cpu_tensor(tensor, name)
    if not bool(torch.isfinite(owner_weights).all()) or not bool(torch.isfinite(alpha).all()):
        raise ValueError("owner_weights and alpha must be finite")
    if bool(((owner_ids < -1) | (owner_ids >= num_primitives)).any()):
        raise ValueError("owner_ids must be -1 or index a primitive")
    if bool(((owner_weights < 0.0) | (owner_weights > 1.0)).any()):
        raise ValueError("owner_weights must be in [0,1]")
    if bool(((alpha < 0.0) | (alpha > 1.0)).any()):
        raise ValueError("alpha must be in [0,1]")


def classify_owner_support(
    owner_ids: Tensor,
    owner_weights: Tensor,
    alpha: Tensor,
    *,
    num_primitives: int,
    very_weak_alpha: float,
    weak_alpha: float,
    min_owner_fraction: float,
) -> OwnerSupportMasks:
    """Classify owner support using caller-specified alpha and dominance gates.

    Pixels without an owner or below ``very_weak_alpha`` are grouped together.
    Remaining owned pixels are ambiguous when ``owner_weight / alpha`` is below
    ``min_owner_fraction``; the rest are split at ``weak_alpha``.
    """
    _validate_owner_fields(
        owner_ids,
        owner_weights,
        alpha,
        num_primitives=num_primitives,
    )
    thresholds = (very_weak_alpha, weak_alpha, min_owner_fraction)
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("support thresholds must be finite")
    if not 0.0 <= very_weak_alpha <= weak_alpha <= 1.0:
        raise ValueError("alpha thresholds must satisfy 0 <= very_weak_alpha <= weak_alpha <= 1")
    if not 0.0 <= min_owner_fraction <= 1.0:
        raise ValueError("min_owner_fraction must be in [0,1]")

    owned = owner_ids >= 0
    unowned_or_very_weak = (~owned) | (alpha < very_weak_alpha)
    owner_fraction = torch.zeros_like(owner_weights)
    positive_alpha = alpha > 0.0
    owner_fraction[positive_alpha] = owner_weights[positive_alpha] / alpha[positive_alpha]
    remaining = owned & ~unowned_or_very_weak
    ambiguous_overlap = remaining & (owner_fraction < min_owner_fraction)
    resolved = remaining & ~ambiguous_overlap
    weak_owned = resolved & (alpha < weak_alpha)
    strong_owned = resolved & ~weak_owned
    return OwnerSupportMasks(
        unowned_or_very_weak=unowned_or_very_weak,
        weak_owned=weak_owned,
        strong_owned=strong_owned,
        ambiguous_overlap=ambiguous_overlap,
    )


def signed_hexel_edge_ids(owner_uv: Tensor) -> Tensor:
    """Return six signed edge IDs for local ``(...,2)`` coordinates.

    ``torch.argmax`` and :func:`canonical_edge_normals` reproduce the native
    signed-edge tie order, including edge zero at the Hexel center.
    """
    if not owner_uv.is_floating_point() or owner_uv.ndim < 1 or owner_uv.shape[-1] != 2:
        raise ValueError("owner_uv must be a floating-point tensor with shape (...,2)")
    _validate_cpu_tensor(owner_uv, "owner_uv")
    if not bool(torch.isfinite(owner_uv).all()):
        raise ValueError("owner_uv must be finite")
    normals = canonical_edge_normals(device=owner_uv.device, dtype=owner_uv.dtype)
    return torch.argmax(owner_uv @ normals.T, dim=-1)


def aggregate_attributed_residual(
    native: Tensor,
    filtered: Tensor,
    target: Tensor,
    owner_ids: Tensor,
    owner_weights: Tensor,
    owner_uv: Tensor,
    alpha: Tensor,
    *,
    num_primitives: int,
    weak_support_power: int,
) -> AttributedResidualAggregation:
    """Aggregate one camera's persistent RGB error into primitive-edge bins.

    Each owned pixel contributes

    ``sum(min(abs(native-target), abs(filtered-target))**2)``
    ``* owner_weight * (1-alpha)**weak_support_power``.

    ``weak_support_power`` is restricted to zero or two. Accumulation is
    performed in float64 on CPU and returns new tensors without retaining an
    autograd graph or changing any input.
    """
    if weak_support_power not in {0, 2} or isinstance(weak_support_power, bool):
        raise ValueError("weak_support_power must be 0 or 2")
    if native.shape != filtered.shape or native.shape != target.shape:
        raise ValueError("native, filtered, and target must have matching shapes")
    if native.ndim != 3 or native.shape[0] != 3:
        raise ValueError("native, filtered, and target must have shape (3,H,W)")
    for tensor, name in (
        (native, "native"),
        (filtered, "filtered"),
        (target, "target"),
    ):
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        _validate_cpu_tensor(tensor, name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")

    pixel_shape = native.shape[1:]
    _validate_owner_fields(
        owner_ids,
        owner_weights,
        alpha,
        num_primitives=num_primitives,
    )
    if owner_ids.shape != pixel_shape:
        raise ValueError("owner maps must match the image spatial shape")
    if owner_uv.shape != (*pixel_shape, 2):
        raise ValueError("owner_uv must have shape (H,W,2)")
    edge_ids = signed_hexel_edge_ids(owner_uv)

    native64 = native.detach().to(dtype=torch.float64)
    filtered64 = filtered.detach().to(dtype=torch.float64)
    target64 = target.detach().to(dtype=torch.float64)
    persistent = torch.minimum(
        torch.abs(native64 - target64),
        torch.abs(filtered64 - target64),
    )
    score = persistent.square().sum(dim=0)
    alpha64 = alpha.detach().to(dtype=torch.float64)
    if weak_support_power == 2:
        score = score * (1.0 - alpha64).square()
    score = score * owner_weights.detach().to(dtype=torch.float64)

    flat_owner = owner_ids.detach().to(dtype=torch.int64).reshape(-1)
    flat_edges = edge_ids.reshape(-1)
    flat_score = score.reshape(-1)
    flat_uv = owner_uv.detach().to(dtype=torch.float64).reshape(-1, 2)
    owned = flat_owner >= 0
    bins = flat_owner[owned] * 6 + flat_edges[owned]

    mass = torch.zeros(num_primitives * 6, dtype=torch.float64)
    centroid_mass = torch.zeros((num_primitives * 6, 2), dtype=torch.float64)
    mass.index_add_(0, bins, flat_score[owned])
    centroid_mass.index_add_(0, bins, flat_score[owned, None] * flat_uv[owned])
    mass = mass.reshape(num_primitives, 6)
    centroid_mass = centroid_mass.reshape(num_primitives, 6, 2)
    dominant_edges = torch.argmax(mass, dim=1)
    rows = torch.arange(num_primitives)
    camera_vote_mask = torch.zeros_like(mass, dtype=torch.bool)
    positive = mass[rows, dominant_edges] > 0.0
    camera_vote_mask[rows[positive], dominant_edges[positive]] = True
    return AttributedResidualAggregation(
        mass=mass,
        centroid_mass=centroid_mass,
        camera_vote_mask=camera_vote_mask,
    )


def summarize_attributed_edge_evidence(
    mass: Tensor,
    centroid_mass: Tensor,
    camera_vote_counts: Tensor,
) -> AttributedEdgeSummary:
    """Summarize accumulated attributed evidence with stable edge ties.

    Camera votes choose the edge. Accumulated mass breaks equal vote counts,
    then the native canonical edge order breaks any remaining tie.
    """
    if mass.ndim != 2 or mass.shape[1] != 6:
        raise ValueError("mass must have shape (P,6)")
    if centroid_mass.shape != (*mass.shape, 2):
        raise ValueError("centroid_mass must have shape (P,6,2)")
    if camera_vote_counts.shape != mass.shape:
        raise ValueError("camera_vote_counts must have shape (P,6)")
    if not mass.is_floating_point() or not centroid_mass.is_floating_point():
        raise ValueError("mass and centroid_mass must be floating-point tensors")
    if camera_vote_counts.dtype not in _INTEGER_DTYPES:
        raise ValueError("camera_vote_counts must have an integer dtype")
    if not (
        mass.device == centroid_mass.device == camera_vote_counts.device
    ):
        raise ValueError("attributed evidence tensors must share a device")
    if not bool(torch.isfinite(mass).all()) or bool((mass < 0.0).any()):
        raise ValueError("mass must be finite and non-negative")
    if not bool(torch.isfinite(centroid_mass).all()):
        raise ValueError("centroid_mass must be finite")
    if bool((camera_vote_counts < 0).any()):
        raise ValueError("camera_vote_counts must be non-negative")

    dominant_votes = camera_vote_counts.max(dim=1).values
    vote_ties = camera_vote_counts == dominant_votes[:, None]
    tied_mass = torch.where(vote_ties, mass, torch.full_like(mass, -1.0))
    dominant_edges = torch.argmax(tied_mass, dim=1)
    rows = torch.arange(mass.shape[0], device=mass.device)
    selected_mass = mass[rows, dominant_edges]
    centroids = centroid_mass[rows, dominant_edges] / selected_mass.clamp_min(
        torch.finfo(mass.dtype).tiny
    )[:, None]
    view_counts = camera_vote_counts.sum(dim=1)
    consistency = dominant_votes.to(mass.dtype) / view_counts.clamp_min(1).to(mass.dtype)
    return AttributedEdgeSummary(
        dominant_edge_ids=dominant_edges,
        view_counts=view_counts,
        dominant_edge_vote_counts=dominant_votes,
        consistency=consistency,
        mass=selected_mass,
        centroids=centroids,
    )


def _owner_local_principal_geometry(
    robust_weight: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    *,
    covariance_epsilon: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    mean = torch.zeros_like(first_moment)
    positive = robust_weight > 0.0
    mean[positive] = first_moment[positive] / robust_weight[positive, None]
    covariance = second_moment / robust_weight.clamp_min(
        torch.finfo(robust_weight.dtype).tiny
    )[:, None, None] - mean[:, :, None] * mean[:, None, :]
    covariance = 0.5 * (covariance + covariance.transpose(1, 2))

    c00 = covariance[:, 0, 0]
    c01 = covariance[:, 0, 1]
    c11 = covariance[:, 1, 1]
    theta = 0.5 * torch.atan2(2.0 * c01, c00 - c11)
    axis = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)
    vertical = torch.abs(axis[:, 0]) <= 8.0 * torch.finfo(axis.dtype).eps
    flip = (axis[:, 0] < 0.0) | (vertical & (axis[:, 1] < 0.0))
    axis = torch.where(flip[:, None], -axis, axis)

    trace = c00 + c11
    discriminant = torch.sqrt(
        torch.clamp(0.25 * (c00 - c11).square() + c01.square(), min=0.0)
    )
    lambda1 = torch.clamp(0.5 * trace + discriminant, min=covariance_epsilon)
    lambda2 = torch.clamp(0.5 * trace - discriminant, min=covariance_epsilon)
    finite = (
        torch.isfinite(mean).all(dim=1)
        & torch.isfinite(covariance).all(dim=(1, 2))
        & torch.isfinite(axis).all(dim=1)
        & torch.isfinite(lambda1)
        & torch.isfinite(lambda2)
    )
    valid = positive & finite
    axis = torch.where(valid[:, None], axis, torch.zeros_like(axis))
    projector = axis[:, :, None] * axis[:, None, :]
    double_angle = torch.stack(
        (axis[:, 0].square() - axis[:, 1].square(), 2.0 * axis[:, 0] * axis[:, 1]),
        dim=1,
    )
    lambda1 = torch.where(valid, lambda1, torch.zeros_like(lambda1))
    lambda2 = torch.where(valid, lambda2, torch.zeros_like(lambda2))
    return mean, covariance, axis, projector, double_angle, lambda1, lambda2, valid


def aggregate_owner_local_moments(
    native: Tensor,
    filtered: Tensor,
    target: Tensor,
    owner_ids: Tensor,
    owner_weights: Tensor,
    owner_uv: Tensor,
    alpha: Tensor,
    *,
    num_primitives: int,
    weak_support_power: int = 0,
) -> OwnerLocalMomentAggregation:
    """Aggregate SAD-style robust residual moments in owner-local coordinates.

    The persistent RGB error is weighted by native ownership. Coordinates
    outside the canonical Hexel are projected radially to its boundary before
    accumulating first and second moments.
    """
    if weak_support_power not in {0, 2} or isinstance(weak_support_power, bool):
        raise ValueError("weak_support_power must be 0 or 2")
    if native.shape != filtered.shape or native.shape != target.shape:
        raise ValueError("native, filtered, and target must have matching shapes")
    if native.ndim != 3 or native.shape[0] != 3:
        raise ValueError("native, filtered, and target must have shape (3,H,W)")
    for tensor, name in (
        (native, "native"),
        (filtered, "filtered"),
        (target, "target"),
    ):
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        _validate_cpu_tensor(tensor, name)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")

    pixel_shape = native.shape[1:]
    _validate_owner_fields(
        owner_ids,
        owner_weights,
        alpha,
        num_primitives=num_primitives,
    )
    if owner_ids.shape != pixel_shape:
        raise ValueError("owner maps must match the image spatial shape")
    if owner_uv.shape != (*pixel_shape, 2):
        raise ValueError("owner_uv must have shape (H,W,2)")
    if not owner_uv.is_floating_point():
        raise ValueError("owner_uv must be a floating-point tensor")
    _validate_cpu_tensor(owner_uv, "owner_uv")
    if not bool(torch.isfinite(owner_uv).all()):
        raise ValueError("owner_uv must be finite")

    native64 = native.detach().to(dtype=torch.float64)
    filtered64 = filtered.detach().to(dtype=torch.float64)
    target64 = target.detach().to(dtype=torch.float64)
    residual = torch.minimum(
        torch.abs(native64 - target64),
        torch.abs(filtered64 - target64),
    ).square().sum(dim=0)
    alpha64 = alpha.detach().to(dtype=torch.float64)
    if weak_support_power == 2:
        residual = residual * (1.0 - alpha64).square()
    ownership = owner_weights.detach().to(dtype=torch.float64)
    robust = ownership * torch.clamp(residual, max=1.0)

    uv = owner_uv.detach().to(dtype=torch.float64)
    normals = canonical_edge_normals(dtype=torch.float64)
    rho = torch.amax(uv @ normals.T, dim=-1)
    projected_uv = uv / torch.clamp(rho, min=1.0)[..., None]

    flat_owner = owner_ids.detach().to(dtype=torch.int64).reshape(-1)
    owned = flat_owner >= 0
    parent_ids = flat_owner[owned]
    flat_ownership = ownership.reshape(-1)[owned]
    flat_residual = residual.reshape(-1)[owned]
    flat_robust = robust.reshape(-1)[owned]
    flat_uv = projected_uv.reshape(-1, 2)[owned]

    coverage = torch.zeros(num_primitives, dtype=torch.float64)
    energy = torch.zeros(num_primitives, dtype=torch.float64)
    robust_weight = torch.zeros(num_primitives, dtype=torch.float64)
    first_moment = torch.zeros((num_primitives, 2), dtype=torch.float64)
    second_moment = torch.zeros((num_primitives, 2, 2), dtype=torch.float64)
    coverage.index_add_(0, parent_ids, flat_ownership)
    energy.index_add_(0, parent_ids, flat_ownership * flat_residual)
    robust_weight.index_add_(0, parent_ids, flat_robust)
    first_moment.index_add_(0, parent_ids, flat_robust[:, None] * flat_uv)
    second_moment.index_add_(
        0,
        parent_ids,
        flat_robust[:, None, None] * flat_uv[:, :, None] * flat_uv[:, None, :],
    )
    statistics = (coverage, energy, robust_weight, first_moment, second_moment)
    if not all(bool(torch.isfinite(value).all()) for value in statistics):
        raise ValueError("owner-local residual statistics must be finite")

    (
        mean,
        _,
        principal_axis,
        axis_projector,
        double_angle,
        _,
        _,
        axis_valid,
    ) = _owner_local_principal_geometry(
        robust_weight,
        first_moment,
        second_moment,
        covariance_epsilon=1e-6,
    )
    centered = flat_uv - mean[parent_ids]
    signed_distance = torch.sum(centered * principal_axis[parent_ids], dim=1)
    side_ids = (signed_distance >= 0.0).to(dtype=torch.int64)
    side_bins = 2 * parent_ids + side_ids
    side_mass = torch.zeros(2 * num_primitives, dtype=torch.float64)
    side_mass.index_add_(0, side_bins, flat_robust)
    side_mass = side_mass.reshape(num_primitives, 2)
    return OwnerLocalMomentAggregation(
        coverage=coverage,
        energy=energy,
        robust_weight=robust_weight,
        first_moment=first_moment,
        second_moment=second_moment,
        principal_axis=principal_axis,
        axis_projector=axis_projector,
        double_angle=double_angle,
        axis_valid=axis_valid,
        side_mass=side_mass,
    )


def summarize_owner_local_moments(
    coverage: Tensor,
    energy: Tensor,
    robust_weight: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    double_angle_sum: Tensor,
    axis_view_count: Tensor,
    material_view_count: Tensor,
    side_mass: Tensor,
    *,
    min_material_views: int = 8,
    min_anisotropy: float = 0.15,
    min_axis_coherence: float = 0.50,
    min_side_balance: float = 0.30,
    min_delta: float = 0.10,
) -> OwnerLocalMomentSummary:
    """Summarize accumulated moments and propose an in-support child pair.

    Pair offsets are ordered negative then positive along the sign-canonical
    principal axis. Eligibility is returned separately from all measurements.
    """
    if coverage.ndim != 1:
        raise ValueError("coverage must have shape (P,)")
    primitive_count = coverage.shape[0]
    vectors = {
        "energy": energy,
        "robust_weight": robust_weight,
    }
    for name, tensor in vectors.items():
        if tensor.shape != coverage.shape:
            raise ValueError(f"{name} must have shape (P,)")
    matrices = {
        "first_moment": (first_moment, (primitive_count, 2)),
        "second_moment": (second_moment, (primitive_count, 2, 2)),
        "double_angle_sum": (double_angle_sum, (primitive_count, 2)),
        "side_mass": (side_mass, (primitive_count, 2)),
    }
    for name, (tensor, shape) in matrices.items():
        if tensor.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    counts = {
        "axis_view_count": axis_view_count,
        "material_view_count": material_view_count,
    }
    for name, tensor in counts.items():
        if tensor.shape != coverage.shape:
            raise ValueError(f"{name} must have shape (P,)")
        if tensor.dtype not in _INTEGER_DTYPES:
            raise ValueError(f"{name} must have an integer dtype")
        if bool((tensor < 0).any()):
            raise ValueError(f"{name} must be non-negative")

    floating = {
        "coverage": coverage,
        "energy": energy,
        "robust_weight": robust_weight,
        **{name: tensor for name, (tensor, _) in matrices.items()},
    }
    devices = {tensor.device for tensor in (*floating.values(), *counts.values())}
    if len(devices) != 1 or next(iter(devices)).type != "cpu":
        raise ValueError("owner-local moment tensors must share the CPU device")
    for name, tensor in floating.items():
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be finite")
    for name in ("coverage", "energy", "robust_weight", "side_mass"):
        if bool((floating[name] < 0.0).any()):
            raise ValueError(f"{name} must be non-negative")

    if not isinstance(min_material_views, int) or isinstance(min_material_views, bool):
        raise ValueError("min_material_views must be an integer")
    if min_material_views < 1:
        raise ValueError("min_material_views must be positive")
    thresholds = (min_anisotropy, min_axis_coherence, min_side_balance, min_delta)
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("moment-summary thresholds must be finite")
    if not 0.0 <= min_anisotropy <= 1.0:
        raise ValueError("min_anisotropy must be in [0,1]")
    if not 0.0 <= min_axis_coherence <= 1.0:
        raise ValueError("min_axis_coherence must be in [0,1]")
    if not 0.0 <= min_side_balance <= 1.0:
        raise ValueError("min_side_balance must be in [0,1]")
    if min_delta < 0.0:
        raise ValueError("min_delta must be non-negative")

    values64 = {
        name: tensor.detach().to(dtype=torch.float64) for name, tensor in floating.items()
    }
    coverage64 = values64["coverage"]
    energy64 = values64["energy"]
    robust64 = values64["robust_weight"]
    first64 = values64["first_moment"]
    second64 = values64["second_moment"]
    angles64 = values64["double_angle_sum"]
    sides64 = values64["side_mass"]
    (
        mean,
        covariance,
        axis,
        projector,
        _,
        lambda1,
        lambda2,
        axis_valid,
    ) = _owner_local_principal_geometry(
        robust64,
        first64,
        second64,
        covariance_epsilon=1e-6,
    )

    epsilon = 1e-12
    score = energy64 / (coverage64 + epsilon).pow(0.7)
    anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + epsilon)
    axis_counts64 = axis_view_count.detach().to(dtype=torch.float64)
    axis_coherence = torch.linalg.vector_norm(angles64, dim=1) / axis_counts64.clamp_min(1.0)
    side_total = sides64.sum(dim=1)
    side_balance = 2.0 * torch.amin(sides64, dim=1) / (side_total + epsilon)

    center = 0.6 * mean
    delta_raw = torch.sqrt(lambda1)
    normals = canonical_edge_normals(dtype=torch.float64)
    support_numerator = 0.85 - center @ normals.T
    support_denominator = torch.abs(axis @ normals.T) + epsilon
    delta_max = torch.amin(support_numerator / support_denominator, dim=1)
    delta = torch.minimum(
        torch.minimum(delta_raw, torch.full_like(delta_raw, 0.40)),
        delta_max,
    )
    pair_offsets = torch.stack(
        (center - delta[:, None] * axis, center + delta[:, None] * axis),
        dim=1,
    )
    eligible = (
        axis_valid
        & (material_view_count >= min_material_views)
        & (anisotropy >= min_anisotropy)
        & (axis_coherence >= min_axis_coherence)
        & (side_balance >= min_side_balance)
        & (delta >= min_delta)
    )
    outputs = (
        score,
        mean,
        covariance,
        axis,
        projector,
        lambda1,
        lambda2,
        anisotropy,
        axis_coherence,
        side_balance,
        center,
        delta_raw,
        delta_max,
        delta,
        pair_offsets,
    )
    if not all(bool(torch.isfinite(value).all()) for value in outputs):
        raise ValueError("owner-local moment summary must be finite")
    return OwnerLocalMomentSummary(
        coverage=coverage64,
        energy=energy64,
        robust_weight=robust64,
        score=score,
        mean=mean,
        covariance=covariance,
        principal_axis=axis,
        axis_projector=projector,
        lambda1=lambda1,
        lambda2=lambda2,
        anisotropy=anisotropy,
        double_angle_sum=angles64,
        axis_view_count=axis_view_count.detach().clone(),
        axis_coherence=axis_coherence,
        material_view_count=material_view_count.detach().clone(),
        side_mass=sides64,
        side_balance=side_balance,
        center=center,
        delta_raw=delta_raw,
        delta_max=delta_max,
        delta=delta,
        pair_offsets=pair_offsets,
        axis_valid=axis_valid,
        eligible=eligible,
    )
