from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from .model import Hexels3DGeometry
from .teacher_continuation import TrainableTeacherHexelModel


BASE_ROW_NAMES = (
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

OptimizerGroupFactory = Callable[
    [TrainableTeacherHexelModel],
    list[dict[str, object]],
]


def _base_population(state: Mapping[str, Tensor]) -> int:
    missing = [name for name in BASE_ROW_NAMES if name not in state]
    if missing:
        raise ValueError(f"State is missing base tensors: {', '.join(missing)}")
    means3d = state["means3d"]
    if means3d.ndim != 2 or means3d.shape[1] != 3:
        raise ValueError("means3d must have shape (P,3)")
    population = int(means3d.shape[0])
    if population < 1:
        raise ValueError("State must contain at least one base primitive")
    for name in BASE_ROW_NAMES:
        value = state[name]
        if value.ndim == 0 or value.shape[0] != population:
            raise ValueError(f"{name} must have leading dimension {population}")
    expected_shapes = {
        "log_scales": (population, 2),
        "rotation_quats": (population, 4),
        "colors": (population, 3),
        "color_grads": (population, 2, 3),
        "color_quads": (population, 3, 3),
        "view_grads": (population, 15, 3),
        "specular_ks": (population, 3),
        "specular_log_s": (population,),
        "opacity_logits": (population,),
        "log_softness": (population,),
    }
    for name, shape in expected_shapes.items():
        if state[name].shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    active_mask = state.get("active_mask")
    if (
        active_mask is None
        or active_mask.dtype != torch.bool
        or active_mask.shape != (population,)
    ):
        raise ValueError(f"active_mask must be bool with shape ({population},)")
    return population


def _require_empty_children(state: Mapping[str, Tensor]) -> None:
    child_parent_idx = state.get("child_parent_idx")
    if child_parent_idx is None or child_parent_idx.ndim != 1:
        raise ValueError("State must contain one-dimensional child_parent_idx")
    if child_parent_idx.numel() != 0:
        raise ValueError("Base population mutation requires an empty child population")


def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def _validate_rows(
    state: Mapping[str, Tensor],
    rows: Mapping[str, Tensor],
) -> int:
    if set(rows) != set(BASE_ROW_NAMES):
        raise ValueError("Rows must contain every base tensor exactly once")
    count = int(rows["means3d"].shape[0])
    for name in BASE_ROW_NAMES:
        value = rows[name]
        reference = state[name]
        expected_shape = (count, *reference.shape[1:])
        if value.shape != expected_shape:
            raise ValueError(f"Rows for {name} must have shape {expected_shape}")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"Rows for {name} must match state dtype and device")
    return count


@torch.no_grad()
def append_base_state(
    state: Mapping[str, Tensor],
    rows: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Append complete base rows to a cloned teacher-Hexel state."""
    _base_population(state)
    _require_empty_children(state)
    count = _validate_rows(state, rows)
    appended = _clone_state(state)
    for name in BASE_ROW_NAMES:
        appended[name] = torch.cat((appended[name], rows[name].detach()), dim=0)
    appended["active_mask"] = torch.cat(
        (
            appended["active_mask"],
            torch.ones(
                count,
                dtype=torch.bool,
                device=appended["active_mask"].device,
            ),
        ),
        dim=0,
    )
    return appended


@torch.no_grad()
def prune_base_state(
    state: Mapping[str, Tensor],
    keep_mask: Tensor,
) -> dict[str, Tensor]:
    """Physically retain selected base rows in a cloned teacher-Hexel state."""
    population = _base_population(state)
    _require_empty_children(state)
    if keep_mask.dtype != torch.bool or keep_mask.shape != (population,):
        raise ValueError(f"keep_mask must be bool with shape ({population},)")
    if not bool(keep_mask.any().item()):
        raise ValueError("Pruning cannot remove every base primitive")

    pruned = _clone_state(state)
    for name in BASE_ROW_NAMES:
        mask = keep_mask.to(device=pruned[name].device)
        pruned[name] = pruned[name][mask]
    active_mask = keep_mask.to(device=pruned["active_mask"].device)
    pruned["active_mask"] = pruned["active_mask"][active_mask]
    if not bool(pruned["active_mask"].any().item()):
        raise ValueError("Pruning cannot remove every active base primitive")
    return pruned


def tangent_axes(quaternions: Tensor) -> tuple[Tensor, Tensor]:
    """Return the local x and y axes of wxyz quaternions."""
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError("quaternions must have shape (P,4)")
    quaternions = torch.nn.functional.normalize(quaternions, dim=-1)
    w, x, y, z = quaternions.unbind(dim=-1)
    tangent_x = torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        dim=-1,
    )
    tangent_y = torch.stack(
        (
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z + w * x),
        ),
        dim=-1,
    )
    return tangent_x, tangent_y


@torch.no_grad()
def symmetric_tangent_split_state(
    state: Mapping[str, Tensor],
    parent_ids: Tensor,
    axis_ids: Tensor,
    offset_fraction: float = 0.25,
) -> dict[str, Tensor]:
    """Replace each parent by its negative half and append its positive half."""
    population = _base_population(state)
    _require_empty_children(state)
    if parent_ids.dtype != torch.long or parent_ids.ndim != 1:
        raise ValueError("parent_ids must be a one-dimensional long tensor")
    if axis_ids.dtype != torch.long or axis_ids.shape != parent_ids.shape:
        raise ValueError("axis_ids must be long and match parent_ids")
    if parent_ids.numel() == 0:
        return _clone_state(state)
    if (
        int(parent_ids.min().item()) < 0
        or int(parent_ids.max().item()) >= population
        or torch.unique(parent_ids).numel() != parent_ids.numel()
    ):
        raise ValueError("parent_ids must contain unique valid base indices")
    if bool(((axis_ids < 0) | (axis_ids > 1)).any().item()):
        raise ValueError("axis_ids must contain only 0 or 1")
    if not math.isfinite(offset_fraction) or offset_fraction < 0.0:
        raise ValueError("offset_fraction must be finite and non-negative")

    device = state["means3d"].device
    parents = parent_ids.to(device=device)
    axes = axis_ids.to(device=device)
    if not bool(state["active_mask"][parents].all().item()):
        raise ValueError("Every split parent must be active")
    if bool(torch.count_nonzero(state["color_quads"][parents]).item()):
        raise ValueError("First-order tangent splitting requires zero parent color_quads")

    rows = {
        name: state[name][parents.to(state[name].device)].detach().clone()
        for name in BASE_ROW_NAMES
    }
    batch = torch.arange(parents.numel(), device=device)
    scales = torch.exp(state["log_scales"][parents])
    selected_scales = scales[batch, axes]
    tangent_x, tangent_y = tangent_axes(state["rotation_quats"][parents])
    tangents = torch.where(axes[:, None] == 0, tangent_x, tangent_y)
    offsets = tangents * (selected_scales * float(offset_fraction))[:, None]

    negative_rows = {name: value.clone() for name, value in rows.items()}
    positive_rows = {name: value.clone() for name, value in rows.items()}
    parent_means = state["means3d"][parents]
    negative_rows["means3d"] = parent_means - offsets
    positive_rows["means3d"] = parent_means + offsets

    axis_mask = torch.nn.functional.one_hot(axes, num_classes=2).to(
        dtype=state["log_scales"].dtype
    )
    child_log_scales = state["log_scales"][parents] + math.log(0.5) * axis_mask
    negative_rows["log_scales"] = child_log_scales
    positive_rows["log_scales"] = child_log_scales.clone()

    color_grads = state["color_grads"][parents]
    color_offsets = color_grads[batch, axes] * float(offset_fraction)
    # Recenter the affine color field on each half. Halving the footprint also
    # halves its normalized-coordinate gradient, preserving the parent's field.
    negative_rows["colors"] = state["colors"][parents] - color_offsets
    positive_rows["colors"] = state["colors"][parents] + color_offsets
    gradient_scale = 1.0 - 0.5 * axis_mask
    child_color_grads = color_grads * gradient_scale[:, :, None]
    negative_rows["color_grads"] = child_color_grads
    positive_rows["color_grads"] = child_color_grads.clone()

    split = _clone_state(state)
    for name in BASE_ROW_NAMES:
        destination = parents.to(split[name].device)
        split[name][destination] = negative_rows[name]
    return append_base_state(split, positive_rows)


def trainable_model_from_state(
    state: Mapping[str, Tensor],
) -> TrainableTeacherHexelModel:
    """Construct and strictly reload a teacher-Hexel model of any base count."""
    _base_population(state)
    reference_softness = state.get("_reference_softness")
    if reference_softness is None or reference_softness.numel() != 1:
        raise ValueError("State must contain scalar _reference_softness")
    geometry = Hexels3DGeometry(
        means3d=state["means3d"].detach().clone(),
        scales=torch.exp(state["log_scales"]).detach().clone(),
        rotations=state["rotation_quats"].detach().clone(),
    )
    model = TrainableTeacherHexelModel(
        geometry=geometry,
        colors_init=state["colors"].detach().clone(),
        opacity_logits=state["opacity_logits"].detach().clone(),
        sh_coefficients=state["view_grads"].detach().clone(),
        softness=float(reference_softness.item()),
    )
    child_parent_idx = state.get("child_parent_idx")
    if child_parent_idx is None or child_parent_idx.ndim != 1:
        raise ValueError("State must contain one-dimensional child_parent_idx")
    model.allocate_children(int(child_parent_idx.numel()))
    model.load_state_dict(dict(state), strict=True)
    return model


def fresh_adam(
    model: TrainableTeacherHexelModel,
    group_factory: OptimizerGroupFactory,
    *,
    eps: float = 1e-15,
) -> torch.optim.Adam:
    """Build a new Adam over the current population using caller-defined groups."""
    return torch.optim.Adam(group_factory(model), eps=eps)
