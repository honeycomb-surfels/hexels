from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import torch
from torch import Tensor


_BASE_TENSOR_NAMES = (
    "means3d",
    "log_scales",
    "rotation_quats",
    "colors",
    "color_grads",
    "color_quads",
    "view_grads",
    "specular_ks",
    "specular_log_s",
    "opacity_logits",
    "log_softness",
)


def _quat_to_tangents(quaternions: Tensor) -> tuple[Tensor, Tensor]:
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = quaternions.unbind(dim=-1)
    tangent_x = torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dim=-1,
    )
    tangent_y = torch.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        dim=-1,
    )
    return tangent_x, tangent_y


def _validate_ids(ids: Tensor, name: str, population: int) -> Tensor:
    if ids.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if ids.dtype != torch.long:
        raise ValueError(f"{name} must have dtype torch.long")
    ids_cpu = ids.detach().cpu()
    if ids_cpu.numel() == 0:
        return ids_cpu
    if int(ids_cpu.min()) < 0:
        raise ValueError(f"{name} must be non-negative")
    if int(ids_cpu.max()) >= population:
        raise ValueError(f"{name} must be smaller than the population {population}")
    if torch.unique(ids_cpu).numel() != ids_cpu.numel():
        raise ValueError(f"{name} must be unique")
    return ids_cpu


def _validate_pair_ids(parent_ids: Tensor, victim_ids: Tensor, population: int) -> tuple[Tensor, Tensor]:
    if parent_ids.shape != victim_ids.shape:
        raise ValueError("parent_ids and victim_ids must have matching shapes")
    parents = _validate_ids(parent_ids, "parent_ids", population)
    victims = _validate_ids(victim_ids, "victim_ids", population)
    if torch.isin(parents, victims).any():
        raise ValueError("parent_ids and victim_ids must be disjoint")
    return parents, victims


def _validate_model_state(
    model_state: Mapping[str, Tensor],
    *,
    require_all_active: bool = True,
    require_zero_color_quads: bool = True,
) -> int:
    missing = [name for name in _BASE_TENSOR_NAMES if name not in model_state]
    if missing:
        raise ValueError(f"Model state is missing tensors: {', '.join(missing)}")

    means3d = model_state["means3d"]
    if means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError(f"means3d must have shape (P,3), got {tuple(means3d.shape)}")
    population = int(means3d.shape[0])
    active_mask = model_state.get("active_mask")
    if active_mask is not None:
        if active_mask.dtype != torch.bool or active_mask.shape != (population,):
            raise ValueError(f"active_mask must be bool with shape ({population},)")
        if require_all_active and not bool(active_mask.all().item()):
            raise ValueError("Split-swap refinement requires every base primitive to be active")
    for name in _BASE_TENSOR_NAMES:
        value = model_state[name]
        if value.ndim == 0 or value.shape[0] != population:
            raise ValueError(f"{name} must have first dimension {population}, got {tuple(value.shape)}")
    if model_state["log_scales"].shape != (population, 2):
        raise ValueError("log_scales must have shape (P,2)")
    if model_state["rotation_quats"].shape != (population, 4):
        raise ValueError("rotation_quats must have shape (P,4)")
    if model_state["colors"].shape != (population, 3):
        raise ValueError("colors must have shape (P,3)")
    if model_state["color_grads"].shape != (population, 2, 3):
        raise ValueError("color_grads must have shape (P,2,3)")
    if model_state["color_quads"].shape != (population, 3, 3):
        raise ValueError("color_quads must have shape (P,3,3)")
    if require_zero_color_quads and torch.count_nonzero(model_state["color_quads"]).item() != 0:
        raise ValueError("Symmetric linear split-swap requires zero inactive color_quads")

    for name, value in model_state.items():
        if name.startswith("child_") and (value.ndim == 0 or value.shape[0] != 0):
            raise ValueError("Symmetric split-swap requires an empty child state")
    return population


def _validate_directional_child_parameters(
    *,
    scale_modifier: float,
    scale_ratio: float,
    init_opacity: float,
    extent_cap_fraction: float,
    parent_opacity_fraction: float,
    scene_extent: float | None,
) -> None:
    if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
        raise ValueError("scale_modifier must be finite and positive")
    if not (0.0 < scale_ratio < 1.0):
        raise ValueError("scale_ratio must be in (0,1)")
    if not (0.0 < init_opacity < 1.0):
        raise ValueError("init_opacity must be in (0,1)")
    if not math.isfinite(extent_cap_fraction) or extent_cap_fraction < 0.0:
        raise ValueError("extent_cap_fraction must be finite and non-negative")
    if not math.isfinite(parent_opacity_fraction) or not (0.0 <= parent_opacity_fraction <= 1.0):
        raise ValueError("parent_opacity_fraction must be finite and in [0,1]")
    if extent_cap_fraction > 0.0 and (
        scene_extent is None or not math.isfinite(scene_extent) or scene_extent <= 0.0
    ):
        raise ValueError("scene_extent must be finite and positive when the extent cap is enabled")


def _zero_optimizer_rows(
    optimizer_state: dict[str, object],
    destination_ids: Tensor,
    population: int,
) -> None:
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise ValueError("Optimizer state must contain state and param_groups")

    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise ValueError("Each optimizer parameter group must contain a params list")
        for parameter_id in group["params"]:
            parameter_state = state.get(parameter_id)
            if parameter_state is None:
                continue
            if not isinstance(parameter_state, dict):
                raise ValueError("Optimizer parameter state must be a mapping")
            for value in parameter_state.values():
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == population:
                    value[destination_ids.to(value.device)] = 0


@torch.no_grad()
def symmetric_split_children(
    model_state: Mapping[str, Tensor],
    parent_ids: Tensor,
    *,
    offset_fraction: float,
    scale_ratio: float,
    split_axes: Tensor | None = None,
    scale_modifier: float = 1.0,
    split_opacity: bool = True,
) -> dict[str, Tensor]:
    """Generate a first-order-field-initialized approximate split without mutation.

    Each returned tensor has leading dimension ``2 * len(parent_ids)``. The first
    row of each pair is offset along the negative selected tangent and the second
    row along the positive tangent. The major axis is used by default.
    """
    if offset_fraction < 0.0:
        raise ValueError("offset_fraction must be non-negative")
    if not (0.0 < scale_ratio < 1.0):
        raise ValueError("scale_ratio must be in (0,1)")
    if not math.isfinite(scale_modifier) or scale_modifier <= 0:
        raise ValueError("scale_modifier must be finite and positive")

    population = _validate_model_state(model_state)
    parents_cpu = _validate_ids(parent_ids, "parent_ids", population)
    children: dict[str, Tensor] = {}
    for name in _BASE_TENSOR_NAMES:
        value = model_state[name]
        parents = parents_cpu.to(value.device)
        children[name] = value[parents].repeat_interleave(2, dim=0).detach().clone()

    means3d = model_state["means3d"]
    device = means3d.device
    parents = parents_cpu.to(device)
    log_scales = model_state["log_scales"][parents]
    scales = torch.exp(log_scales)
    if split_axes is None:
        split_axis = (scales[:, 1] > scales[:, 0]).to(torch.long)
    else:
        if split_axes.ndim != 1 or split_axes.shape != parents_cpu.shape:
            raise ValueError("split_axes must match parent_ids")
        if split_axes.dtype != torch.long or bool(((split_axes < 0) | (split_axes > 1)).any()):
            raise ValueError("split_axes must contain tangent-axis indices 0 or 1")
        split_axis = split_axes.to(device=device)
    batch = torch.arange(parents.numel(), device=device)
    split_scale = scales[batch, split_axis]

    tangent_x, tangent_y = _quat_to_tangents(model_state["rotation_quats"][parents])
    split_tangent = torch.where(split_axis[:, None] == 0, tangent_x, tangent_y)
    offset = split_tangent * (split_scale * float(offset_fraction))[:, None]
    child_means = torch.stack(
        [model_state["means3d"][parents] - offset, model_state["means3d"][parents] + offset],
        dim=1,
    ).reshape(-1, 3)
    children["means3d"] = child_means

    axis_mask = torch.nn.functional.one_hot(split_axis, num_classes=2).to(log_scales.dtype)
    child_log_scales = (
        log_scales[:, None, :] + math.log(float(scale_ratio)) * axis_mask[:, None, :]
    ).expand(-1, 2, -1)
    children["log_scales"] = child_log_scales.reshape(-1, 2)

    color_grads = model_state["color_grads"][parents]
    split_color_grad = color_grads[batch, split_axis]
    color_shift = split_color_grad * (float(offset_fraction) / scale_modifier)
    child_colors = torch.stack(
        [model_state["colors"][parents] - color_shift, model_state["colors"][parents] + color_shift],
        dim=1,
    ).reshape(-1, 3)
    children["colors"] = child_colors

    grad_factor = 1.0 + (float(scale_ratio) - 1.0) * axis_mask
    child_color_grads = (color_grads * grad_factor[:, :, None])[:, None].expand(-1, 2, -1, -1)
    children["color_grads"] = child_color_grads.reshape(-1, 2, 3)

    parent_logits = model_state["opacity_logits"][parents]
    half_log_transmittance = -0.5 * torch.nn.functional.softplus(parent_logits)
    child_logits = torch.log(-torch.expm1(half_log_transmittance)) - half_log_transmittance
    if split_opacity:
        children["opacity_logits"] = child_logits.repeat_interleave(2)
    return children


def rank_absolute_candidates(
    signed_scores: Tensor, absolute_scores: Tensor, eligible: Tensor, threshold: float,
) -> Tensor:
    """Change parent ranking without increasing this state's signed-rule birth count."""
    count = int(torch.count_nonzero(eligible & (signed_scores >= threshold)))
    scores = absolute_scores.masked_fill(~eligible, -torch.inf)
    return torch.topk(scores, k=count, sorted=False).indices


def transferred_linear_appearance(
    parent_means: Tensor,
    parent_scales: Tensor,
    parent_rotations: Tensor,
    parent_colors: Tensor,
    parent_gradients: Tensor,
    child_means: Tensor,
    child_scales: Tensor,
    *,
    scale_modifier: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Re-express a coplanar linear field after translating/scaling its footprint."""
    tangent_x, tangent_y = _quat_to_tangents(parent_rotations)
    offset = child_means - parent_means
    local_offset = torch.stack(((offset * tangent_x).sum(-1), (offset * tangent_y).sum(-1)), dim=-1)
    local_offset = local_offset / (parent_scales * scale_modifier)
    colors = parent_colors + (parent_gradients * local_offset[..., None]).sum(dim=-2)
    gradients = parent_gradients * (child_scales / parent_scales)[..., None]
    return colors, gradients


def _directional_child_rows(
    model_state: Mapping[str, Tensor],
    parents_cpu: Tensor,
    local_offsets: Tensor,
    *,
    scale_modifier: float,
    scale_ratio: float,
    init_opacity: float,
    extent_cap_fraction: float,
    parent_opacity_fraction: float,
    scene_extent: float | None,
) -> dict[str, Tensor]:
    children: dict[str, Tensor] = {}
    for name in _BASE_TENSOR_NAMES:
        value = model_state[name]
        children[name] = value[parents_cpu.to(value.device)].detach().clone()

    device = model_state["means3d"].device
    parents = parents_cpu.to(device)
    offsets = local_offsets.to(device=device, dtype=model_state["means3d"].dtype)
    scales = torch.exp(model_state["log_scales"][parents])
    tangent_x, tangent_y = _quat_to_tangents(model_state["rotation_quats"][parents])
    world_offset = float(scale_modifier) * (
        tangent_x * (scales[:, 0] * offsets[:, 0])[:, None]
        + tangent_y * (scales[:, 1] * offsets[:, 1])[:, None]
    )
    effective_offsets = offsets
    effective_scale_ratio: Tensor | None = None
    if extent_cap_fraction > 0.0:
        extent_cap = float(extent_cap_fraction) * float(scene_extent)
        if not math.isfinite(extent_cap):
            raise ValueError("extent_cap_fraction * scene_extent must be finite")
        offset_ratio = (
            extent_cap
            / torch.linalg.vector_norm(world_offset, dim=1).clamp_min(
                torch.finfo(world_offset.dtype).tiny
            )
        ).clamp(max=1.0)
        effective_offsets = offsets * offset_ratio[:, None]
        world_offset = world_offset * offset_ratio[:, None]
        parent_max_scale = float(scale_modifier) * scales.max(dim=1).values
        effective_scale_ratio = torch.minimum(
            torch.full_like(parent_max_scale, float(scale_ratio)),
            extent_cap / parent_max_scale,
        )
    children["means3d"] = model_state["means3d"][parents] + world_offset

    color_grads = model_state["color_grads"][parents]
    color_quads = model_state["color_quads"][parents]
    offset_u = effective_offsets[:, 0, None]
    offset_v = effective_offsets[:, 1, None]
    children["colors"] = (
        model_state["colors"][parents]
        + color_grads[:, 0] * offset_u
        + color_grads[:, 1] * offset_v
        + color_quads[:, 0] * offset_u.square()
        + color_quads[:, 1] * offset_u * offset_v
        + color_quads[:, 2] * offset_v.square()
    )
    centered_color_grads = torch.stack(
        [
            color_grads[:, 0] + 2.0 * color_quads[:, 0] * offset_u + color_quads[:, 1] * offset_v,
            color_grads[:, 1] + color_quads[:, 1] * offset_u + 2.0 * color_quads[:, 2] * offset_v,
        ],
        dim=1,
    )
    if effective_scale_ratio is None:
        children["log_scales"] = model_state["log_scales"][parents] + math.log(float(scale_ratio))
        children["color_grads"] = centered_color_grads * float(scale_ratio)
        children["color_quads"] = color_quads * float(scale_ratio) ** 2
        children["log_softness"] = model_state["log_softness"][parents] + math.log(float(scale_ratio))
    else:
        log_ratio = torch.log(effective_scale_ratio)
        children["log_scales"] = model_state["log_scales"][parents] + log_ratio[:, None]
        children["color_grads"] = centered_color_grads * effective_scale_ratio[:, None, None]
        children["color_quads"] = color_quads * effective_scale_ratio.square()[:, None, None]
        softness_ratio = log_ratio if children["log_softness"].ndim == 1 else log_ratio[:, None]
        children["log_softness"] = model_state["log_softness"][parents] + softness_ratio

    parent_logits = model_state["opacity_logits"][parents]
    if parent_opacity_fraction == 0.0:
        children["opacity_logits"] = torch.full_like(
            parent_logits,
            math.log(float(init_opacity) / (1.0 - float(init_opacity))),
        )
    else:
        log_opacity = -torch.nn.functional.softplus(-parent_logits) + math.log(
            float(parent_opacity_fraction)
        )
        log_opacity.clamp_(max=math.log(float(init_opacity)))
        children["opacity_logits"] = log_opacity - torch.log1p(-torch.exp(log_opacity))
    return children


@torch.no_grad()
def directional_edge_children(
    model_state: Mapping[str, Tensor],
    parent_ids: Tensor,
    *,
    local_offsets: Tensor,
    scale_modifier: float,
    scale_ratio: float,
    init_opacity: float,
    extent_cap_fraction: float = 0.0,
    parent_opacity_fraction: float = 0.0,
    scene_extent: float | None = None,
) -> dict[str, Tensor]:
    """Create faint one-sided children while leaving parent rows unchanged."""
    _validate_directional_child_parameters(
        scale_modifier=scale_modifier,
        scale_ratio=scale_ratio,
        init_opacity=init_opacity,
        extent_cap_fraction=extent_cap_fraction,
        parent_opacity_fraction=parent_opacity_fraction,
        scene_extent=scene_extent,
    )
    population = _validate_model_state(model_state, require_all_active=False)
    parents_cpu = _validate_ids(parent_ids, "parent_ids", population)
    if local_offsets.shape != (parents_cpu.numel(), 2):
        raise ValueError("local_offsets must have shape (len(parent_ids),2)")
    if not bool(torch.isfinite(local_offsets).all()):
        raise ValueError("local_offsets must be finite")
    active_mask = model_state.get("active_mask")
    if active_mask is not None and not bool(active_mask[parents_cpu.to(active_mask.device)].all()):
        raise ValueError("Every directional child parent must be active")
    return _directional_child_rows(
        model_state,
        parents_cpu,
        local_offsets,
        scale_modifier=scale_modifier,
        scale_ratio=scale_ratio,
        init_opacity=init_opacity,
        extent_cap_fraction=extent_cap_fraction,
        parent_opacity_fraction=parent_opacity_fraction,
        scene_extent=scene_extent,
    )


@torch.no_grad()
def covariance_pair_children(
    model_state: Mapping[str, Tensor],
    parent_ids: Tensor,
    *,
    local_offsets: Tensor,
    scale_modifier: float,
    scale_ratio: float = 0.5,
    init_opacity: float = 0.005,
    extent_cap_fraction: float = 0.0,
    parent_opacity_fraction: float = 0.05,
    scene_extent: float | None = None,
) -> dict[str, Tensor]:
    """Create two ordered, field-preserving children per retained parent."""
    _validate_directional_child_parameters(
        scale_modifier=scale_modifier,
        scale_ratio=scale_ratio,
        init_opacity=init_opacity,
        extent_cap_fraction=extent_cap_fraction,
        parent_opacity_fraction=parent_opacity_fraction,
        scene_extent=scene_extent,
    )
    if parent_opacity_fraction == 0.0:
        raise ValueError("parent_opacity_fraction must be positive for covariance-pair children")

    population = _validate_model_state(
        model_state,
        require_all_active=False,
        require_zero_color_quads=False,
    )
    parents_cpu = _validate_ids(parent_ids, "parent_ids", population)
    expected_shape = (parents_cpu.numel(), 2, 2)
    if local_offsets.shape != expected_shape:
        raise ValueError(f"local_offsets must have shape {expected_shape}")
    if not bool(torch.isfinite(local_offsets).all()):
        raise ValueError("local_offsets must be finite")
    active_mask = model_state.get("active_mask")
    if active_mask is not None and not bool(active_mask[parents_cpu.to(active_mask.device)].all()):
        raise ValueError("Every covariance-pair child parent must be active")

    return _directional_child_rows(
        model_state,
        parents_cpu.repeat_interleave(2),
        local_offsets.reshape(-1, 2),
        scale_modifier=scale_modifier,
        scale_ratio=scale_ratio,
        init_opacity=init_opacity,
        extent_cap_fraction=extent_cap_fraction,
        parent_opacity_fraction=parent_opacity_fraction,
        scene_extent=scene_extent,
    )


@torch.no_grad()
def apply_refinement_rows(
    model_state: Mapping[str, Tensor],
    optimizer_state: Mapping[str, object],
    destination_ids: Tensor,
    rows: Mapping[str, Tensor],
    *,
    activate_destinations: bool = False,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Write complete base rows into cloned same-population checkpoint state."""
    population = _validate_model_state(
        model_state,
        require_all_active=not activate_destinations,
    )
    destinations_cpu = _validate_ids(destination_ids, "destination_ids", population)
    if set(rows) != set(_BASE_TENSOR_NAMES):
        raise ValueError("Refinement rows must contain every base model tensor exactly once")

    refined_model = {name: value.detach().clone() for name, value in model_state.items()}
    for name in _BASE_TENSOR_NAMES:
        value = rows[name]
        expected_shape = (destinations_cpu.numel(), *model_state[name].shape[1:])
        if value.shape != expected_shape:
            raise ValueError(f"Refinement rows for {name} must have shape {expected_shape}")
        if value.dtype != model_state[name].dtype or value.device != model_state[name].device:
            raise ValueError(f"Refinement rows for {name} must match the model tensor dtype and device")
        refined_model[name][destinations_cpu.to(value.device)] = value

    if activate_destinations:
        active_mask = model_state.get("active_mask")
        if active_mask is None or bool(active_mask[destinations_cpu.to(active_mask.device)].any()):
            raise ValueError("Activated refinement destinations must be inactive base rows")
        refined_model["active_mask"][destinations_cpu.to(active_mask.device)] = True

    refined_optimizer = copy.deepcopy(dict(optimizer_state))
    _zero_optimizer_rows(refined_optimizer, destinations_cpu, population)
    return refined_model, refined_optimizer


@torch.no_grad()
def symmetric_split_swap(
    model_state: Mapping[str, Tensor],
    optimizer_state: Mapping[str, object],
    parent_ids: Tensor,
    victim_ids: Tensor,
    *,
    offset_fraction: float,
    scale_ratio: float,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Apply a first-order-field-initialized approximate split to cloned state.

    The parent slot receives the negative child and the victim slot receives the
    positive child. Population size, tensor shapes, scalar optimizer state, and
    optimizer parameter-group values are preserved.
    """
    population = _validate_model_state(model_state)
    parents_cpu, victims_cpu = _validate_pair_ids(parent_ids, victim_ids, population)
    children = symmetric_split_children(
        model_state,
        parents_cpu,
        offset_fraction=offset_fraction,
        scale_ratio=scale_ratio,
    )
    destinations_cpu = torch.stack([parents_cpu, victims_cpu], dim=1).reshape(-1)
    return apply_refinement_rows(model_state, optimizer_state, destinations_cpu, children)
