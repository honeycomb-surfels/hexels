from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .edge_shift import canonical_edge_normals
from .refinement import (
    _BASE_TENSOR_NAMES,
    _quat_to_tangents,
    _validate_ids,
    _validate_model_state,
    _zero_optimizer_rows,
    apply_refinement_rows,
)


_SCHEMA_VERSION = 1
_BURN_IN_FROZEN_NAMES = {
    "means3d",
    "log_scales",
    "rotation_quats",
    "opacity_logits",
    "log_softness",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def edge_boundary_directions(edge_ids: Tensor) -> Tensor:
    """Return canonical edge-midpoint coordinates for signed edge IDs."""
    if edge_ids.ndim != 1 or edge_ids.dtype != torch.long:
        raise ValueError("edge_ids must be a long vector")
    if bool(((edge_ids < 0) | (edge_ids >= 6)).any()):
        raise ValueError("edge_ids must be in [0,6)")
    return canonical_edge_normals(device=edge_ids.device)[edge_ids]


def vertex_boundary_directions(edge_pairs: Tensor) -> Tensor:
    """Return exact canonical vertices for pairs of adjacent signed edges."""
    if edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2 or edge_pairs.dtype != torch.long:
        raise ValueError("edge_pairs must have shape (N,2) and dtype long")
    if bool(((edge_pairs < 0) | (edge_pairs >= 6)).any()):
        raise ValueError("edge_pairs must be in [0,6)")
    normals = canonical_edge_normals(device=edge_pairs.device)
    first = normals[edge_pairs[:, 0]]
    second = normals[edge_pairs[:, 1]]
    adjacent = torch.isclose(torch.sum(first * second, dim=1), torch.full_like(first[:, 0], 0.5))
    if not bool(adjacent.all()):
        raise ValueError("Each vertex requires two adjacent Hexel edges")
    return (2.0 / 3.0) * (first + second)


def _world_offsets(
    log_scales: Tensor,
    rotations: Tensor,
    local_offsets: Tensor,
    *,
    scale_modifier: float,
) -> Tensor:
    if log_scales.shape != local_offsets.shape or log_scales.shape[-1] != 2:
        raise ValueError("log_scales and local_offsets must have shape (N,2)")
    if rotations.shape != (log_scales.shape[0], 4):
        raise ValueError("rotations must have shape (N,4)")
    if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
        raise ValueError("scale_modifier must be finite and positive")
    tangent_u, tangent_v = _quat_to_tangents(rotations)
    scales = torch.exp(log_scales)
    return float(scale_modifier) * (
        tangent_u * (scales[:, 0] * local_offsets[:, 0])[:, None]
        + tangent_v * (scales[:, 1] * local_offsets[:, 1])[:, None]
    )


@torch.no_grad()
def directed_child_rows(
    model_state: Mapping[str, Tensor],
    parent_ids: Tensor,
    boundary_directions: Tensor,
    *,
    scale_modifier: float,
    scale_ratio: float,
    overlap: float,
    opacity: float,
) -> dict[str, Tensor]:
    """Initialize one outward, field-covariant child per selected parent."""
    population = _validate_model_state(model_state, require_all_active=False)
    parents_cpu = _validate_ids(parent_ids, "parent_ids", population)
    count = parents_cpu.numel()
    if boundary_directions.shape != (count, 2):
        raise ValueError("boundary_directions must have shape (N,2)")
    if not bool(torch.isfinite(boundary_directions).all()):
        raise ValueError("boundary_directions must be finite")
    if not (0.0 < scale_ratio < 1.0):
        raise ValueError("scale_ratio must be in (0,1)")
    if not (0.0 < overlap < scale_ratio):
        raise ValueError("overlap must be in (0,scale_ratio)")
    if not (0.0 < opacity < 1.0) or 0.5 * opacity <= 1.0 / 255.0:
        raise ValueError("opacity must remain above the rasterizer cutoff at the contour")

    active_mask = model_state.get("active_mask")
    if active_mask is not None and not bool(active_mask[parents_cpu.to(active_mask.device)].all()):
        raise ValueError("Every edge-transfer parent must be active")
    parent_quads = model_state["color_quads"][parents_cpu.to(model_state["color_quads"].device)]
    parent_specular = model_state["specular_ks"][parents_cpu.to(model_state["specular_ks"].device)]
    if bool((parent_quads != 0.0).any()) or bool((parent_specular != 0.0).any()):
        raise ValueError("Minimal edge transfer requires zero quadratic and specular parent fields")

    rows: dict[str, Tensor] = {}
    for name in _BASE_TENSOR_NAMES:
        value = model_state[name]
        rows[name] = value[parents_cpu.to(value.device)].detach().clone()

    device = rows["means3d"].device
    dtype = rows["means3d"].dtype
    directions = boundary_directions.to(device=device, dtype=dtype)
    beta = 1.0 + float(scale_ratio) - float(overlap)
    local_centers = beta * directions
    rows["means3d"] += _world_offsets(
        rows["log_scales"],
        rows["rotation_quats"],
        local_centers,
        scale_modifier=scale_modifier,
    )
    rows["log_scales"] += math.log(float(scale_ratio))
    rows["colors"] += torch.sum(rows["color_grads"] * local_centers[:, :, None], dim=1)
    rows["color_grads"] *= float(scale_ratio)
    rows["opacity_logits"].fill_(math.log(opacity / (1.0 - opacity)))
    rows["log_softness"] += math.log(float(scale_ratio))
    return rows


@torch.no_grad()
def apply_edge_transfer_birth(
    model_state: Mapping[str, Tensor],
    optimizer_state: Mapping[str, object],
    parent_ids: Tensor,
    child_ids: Tensor,
    boundary_directions: Tensor,
    *,
    scale_modifier: float,
    scale_ratio: float,
    overlap: float,
    opacity: float,
    reset_parent_optimizer: bool = True,
) -> tuple[dict[str, Tensor], dict[str, object]]:
    """Activate inactive base rows as children and reset their Adam rows."""
    population = _validate_model_state(model_state, require_all_active=False)
    parents_cpu = _validate_ids(parent_ids, "parent_ids", population)
    children_cpu = _validate_ids(child_ids, "child_ids", population)
    if parents_cpu.shape != children_cpu.shape:
        raise ValueError("parent_ids and child_ids must have matching shapes")
    if bool(torch.isin(parents_cpu, children_cpu).any()):
        raise ValueError("parent_ids and child_ids must be disjoint")
    active_mask = model_state.get("active_mask")
    if active_mask is None:
        raise ValueError("Edge-transfer birth requires an active_mask")
    if bool(active_mask[children_cpu.to(active_mask.device)].any()):
        raise ValueError("Every edge-transfer child destination must be inactive")

    rows = directed_child_rows(
        model_state,
        parents_cpu,
        boundary_directions,
        scale_modifier=scale_modifier,
        scale_ratio=scale_ratio,
        overlap=overlap,
        opacity=opacity,
    )
    refined_model, refined_optimizer = apply_refinement_rows(
        model_state,
        optimizer_state,
        children_cpu,
        rows,
        activate_destinations=True,
    )
    if reset_parent_optimizer:
        _zero_optimizer_rows(refined_optimizer, parents_cpu, population)
    return refined_model, refined_optimizer


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


class EdgeTransferController:
    """Control trainable rows during an edge-transfer continuation."""

    def __init__(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        manifest: Mapping[str, object],
        *,
        manifest_path: Path,
        checkpoint_sha256: str,
        start_iteration: int,
        target_iteration: int,
        scale_modifier: float,
    ) -> None:
        if manifest.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported edge-transfer manifest schema")
        if manifest.get("kind") != "hexels3d_edge_transfer_training":
            raise ValueError("Invalid edge-transfer manifest kind")
        if manifest.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("Edge-transfer manifest does not match the resume checkpoint")
        if manifest.get("start_iteration") != start_iteration:
            raise ValueError("Edge-transfer manifest start iteration mismatch")
        if manifest.get("target_iteration") != target_iteration:
            raise ValueError("Edge-transfer manifest target iteration mismatch")

        parent_ids = manifest.get("parent_ids")
        child_ids = manifest.get("child_ids")
        edge_ids = manifest.get("edge_ids")
        vertex_edge_ids = manifest.get("vertex_edge_ids")
        if not all(isinstance(value, Tensor) for value in (parent_ids, child_ids)):
            raise ValueError("Edge-transfer parent and child IDs must be tensors")
        if (edge_ids is None) == (vertex_edge_ids is None):
            raise ValueError("Edge transfer requires exactly one boundary representation")
        population = int(model.num_base)
        self.parent_ids = _validate_ids(parent_ids, "parent_ids", population).to(model.means3d.device)
        self.child_ids = _validate_ids(child_ids, "child_ids", population).to(model.means3d.device)
        if self.parent_ids.shape != self.child_ids.shape:
            raise ValueError("Edge-transfer parent and child IDs must match")
        self.edge_ids: Tensor | None = None
        self.vertex_edge_ids: Tensor | None = None
        if isinstance(edge_ids, Tensor):
            self.edge_ids = edge_ids.detach().cpu()
            if self.edge_ids.shape != self.parent_ids.shape:
                raise ValueError("Edge-transfer parent, child, and edge IDs must match")
            if self.edge_ids.dtype != torch.long or bool(
                ((self.edge_ids < 0) | (self.edge_ids >= 6)).any()
            ):
                raise ValueError("Edge-transfer edge IDs must be long values in [0,6)")
            boundary_directions = edge_boundary_directions(self.edge_ids)
        elif isinstance(vertex_edge_ids, Tensor):
            self.vertex_edge_ids = vertex_edge_ids.detach().cpu()
            if self.vertex_edge_ids.shape != (self.parent_ids.numel(), 2):
                raise ValueError("vertex_edge_ids must have shape (N,2)")
            boundary_directions = vertex_boundary_directions(self.vertex_edge_ids)
        else:
            raise ValueError("Edge-transfer boundary IDs must be tensors")
        if bool(torch.isin(self.parent_ids, self.child_ids).any()):
            raise ValueError("Edge-transfer parent and child IDs must be disjoint")
        if not bool(model.active_mask[self.parent_ids].all()) or not bool(
            model.active_mask[self.child_ids].all()
        ):
            raise ValueError("Edge-transfer parent and child rows must be active")

        self.appearance_steps = int(manifest.get("appearance_steps", -1))
        self.anchor_release_steps = int(manifest.get("anchor_release_steps", -1))
        self.retraction_start_step = int(manifest.get("retraction_start_step", -1))
        self.retraction_ramp_steps = int(manifest.get("retraction_ramp_steps", -1))
        self.max_retraction = float(manifest.get("max_retraction", math.nan))
        self.retraction_lr = float(manifest.get("retraction_lr", math.nan))
        self.overlap = float(manifest.get("overlap", math.nan))
        total_steps = target_iteration - start_iteration
        if not (0 <= self.appearance_steps < total_steps):
            raise ValueError("appearance_steps must be in [0,total_steps)")
        if not (1 <= self.anchor_release_steps <= total_steps - self.appearance_steps):
            raise ValueError("anchor_release_steps exceeds the post-appearance schedule")
        if not (self.appearance_steps < self.retraction_start_step < total_steps):
            raise ValueError("retraction_start_step must be after appearance and before completion")
        if not (1 <= self.retraction_ramp_steps <= total_steps - self.retraction_start_step):
            raise ValueError("retraction_ramp_steps exceeds the remaining schedule")
        if not (0.0 < self.overlap < 1.0):
            raise ValueError("overlap must be in (0,1)")
        if not (0.0 <= self.max_retraction <= self.overlap):
            raise ValueError("max_retraction must be in [0,overlap]")
        if not math.isfinite(self.retraction_lr) or self.retraction_lr < 0.0:
            raise ValueError("retraction_lr must be finite and non-negative")
        if self.max_retraction > 0.0 and self.retraction_lr == 0.0:
            raise ValueError("Positive max_retraction requires a positive retraction_lr")
        if not math.isfinite(scale_modifier) or scale_modifier <= 0.0:
            raise ValueError("scale_modifier must be finite and positive")

        local_joint_start_step = manifest.get("local_joint_start_step")
        global_relaxation_start_step = manifest.get("global_relaxation_start_step")
        self.staged_training = (
            local_joint_start_step is not None or global_relaxation_start_step is not None
        )
        if self.staged_training:
            for value, name in (
                (local_joint_start_step, "local_joint_start_step"),
                (global_relaxation_start_step, "global_relaxation_start_step"),
            ):
                if value is not None and (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1 <= value <= total_steps
                ):
                    raise ValueError(f"{name} must be an integer in [1,total_steps]")
            if (
                local_joint_start_step is not None
                and global_relaxation_start_step is not None
                and local_joint_start_step >= global_relaxation_start_step
            ):
                raise ValueError("local_joint_start_step must precede global_relaxation_start_step")
            release_start = min(
                value
                for value in (local_joint_start_step, global_relaxation_start_step)
                if value is not None
            )
            if self.appearance_steps + self.anchor_release_steps > release_start:
                raise ValueError("View anchoring must be released before a broader trainable scope")
            if self.max_retraction != 0.0:
                raise ValueError("Staged edge transfer requires scalar retraction to be disabled")
            neighbor_ids = manifest.get("neighbor_ids", torch.empty(0, dtype=torch.long))
            if not isinstance(neighbor_ids, Tensor):
                raise ValueError("neighbor_ids must be a tensor")
            if neighbor_ids.numel() > 0 and local_joint_start_step is None:
                raise ValueError("neighbor_ids requires a local joint stage")
            self.neighbor_ids = _validate_ids(
                neighbor_ids, "neighbor_ids", population
            ).to(model.means3d.device)
            selected_ids = torch.cat((self.parent_ids, self.child_ids))
            if bool(torch.isin(self.neighbor_ids, selected_ids).any()):
                raise ValueError("neighbor_ids must be disjoint from parent_ids and child_ids")
            if not bool(model.active_mask[self.neighbor_ids].all()):
                raise ValueError("Every edge-transfer neighbor row must be active")
            self.local_joint_start_step = local_joint_start_step
            self.global_relaxation_start_step = global_relaxation_start_step
        else:
            if "neighbor_ids" in manifest:
                raise ValueError("neighbor_ids requires a staged edge-transfer schedule")
            self.neighbor_ids = torch.empty(
                0, device=model.means3d.device, dtype=torch.long
            )
            self.local_joint_start_step = None
            self.global_relaxation_start_step = None

        round_index = manifest.get("round_index")
        parent_edge_transfer_sha256 = manifest.get("parent_edge_transfer_sha256")
        self.has_round_provenance = (
            round_index is not None or parent_edge_transfer_sha256 is not None
        )
        if self.has_round_provenance:
            if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index <= 0:
                raise ValueError("round_index must be a positive integer")
            if not isinstance(parent_edge_transfer_sha256, str) or not parent_edge_transfer_sha256:
                raise ValueError("parent_edge_transfer_sha256 must be a non-empty string")
            self.round_index = round_index
            self.parent_edge_transfer_sha256 = parent_edge_transfer_sha256
        else:
            self.round_index = None
            self.parent_edge_transfer_sha256 = None

        global_train_rotations = manifest.get("global_train_rotations", False)
        if not isinstance(global_train_rotations, bool):
            raise ValueError("global_train_rotations must be a boolean")
        if global_train_rotations and global_relaxation_start_step is None:
            raise ValueError("global_train_rotations requires a global relaxation stage")
        self.global_train_rotations = global_train_rotations

        self.model = model
        self.model_optimizer = optimizer
        self.num_base = int(model.num_base)
        self.num_children = int(model.num_children)
        self.manifest_path = manifest_path
        self.manifest_sha256 = _sha256(manifest_path)
        self.scale_modifier = float(scale_modifier)
        self.current_step = 0
        self.total_steps = total_steps
        self.boundary_directions = boundary_directions.to(
            device=model.means3d.device,
            dtype=model.means3d.dtype,
        )
        self.base_means = model.means3d.detach()[self.parent_ids].clone()
        self.base_log_scales = model.log_scales.detach()[self.parent_ids].clone()
        self.base_colors = model.colors.detach()[self.parent_ids].clone()
        self.base_color_grads = model.color_grads.detach()[self.parent_ids].clone()
        self.base_log_softness = model.log_softness.detach()[self.parent_ids].clone()
        self.world_directions = _world_offsets(
            self.base_log_scales,
            model.rotation_quats.detach()[self.parent_ids],
            self.boundary_directions,
            scale_modifier=self.scale_modifier,
        )
        self.color_directions = torch.sum(
            self.base_color_grads * self.boundary_directions[:, :, None],
            dim=1,
        )
        self.retractions = torch.zeros(
            self.parent_ids.numel(),
            device=model.means3d.device,
            dtype=model.means3d.dtype,
            requires_grad=True,
        )
        self.retraction_optimizer = (
            torch.optim.Adam([self.retractions], lr=self.retraction_lr)
            if self.max_retraction > 0.0
            else None
        )
        self.child_mask = torch.zeros(population, device=model.means3d.device, dtype=torch.bool)
        self.child_mask[self.child_ids] = True
        self.local_mask = self.child_mask.clone()
        self.local_mask[self.parent_ids] = True
        self.local_mask[self.neighbor_ids] = True
        self.frozen_parameters = {
            name: getattr(model, name).detach().clone()
            for name in _BASE_TENSOR_NAMES
        }
        self.frozen_optimizer_rows: dict[tuple[str, str], Tensor] = {}
        for group in optimizer.param_groups:
            name = str(group.get("name", ""))
            parameters = group.get("params", [])
            if name not in _BASE_TENSOR_NAMES or len(parameters) != 1:
                continue
            parameter = parameters[0]
            for state_name, value in optimizer.state.get(parameter, {}).items():
                if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == population:
                    self.frozen_optimizer_rows[(name, state_name)] = value.detach().clone()

    @classmethod
    def from_file(
        cls,
        model: Any,
        optimizer: torch.optim.Optimizer,
        path: str | Path,
        *,
        checkpoint_sha256: str,
        start_iteration: int,
        target_iteration: int,
        scale_modifier: float,
    ) -> EdgeTransferController:
        manifest_path = Path(path).resolve()
        manifest = _mapping(
            torch.load(manifest_path, map_location="cpu", weights_only=True),
            "Edge-transfer manifest",
        )
        return cls(
            model,
            optimizer,
            manifest,
            manifest_path=manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            start_iteration=start_iteration,
            target_iteration=target_iteration,
            scale_modifier=scale_modifier,
        )

    def set_step(self, step: int) -> None:
        self.current_step = int(step)

    def is_complete(self) -> bool:
        return self.current_step >= self.total_steps and self._anchor_release() >= 1.0

    def _anchor_release(self) -> float:
        released_steps = max(0, self.current_step - self.appearance_steps)
        return min(1.0, released_steps / self.anchor_release_steps)

    def _trainable_scope(self, step: int) -> str:
        if not self.staged_training:
            return "child_only"
        if (
            self.global_relaxation_start_step is not None
            and step >= self.global_relaxation_start_step
        ):
            return "global_relaxation"
        if self.local_joint_start_step is not None and step >= self.local_joint_start_step:
            return "local_joint"
        return "child_only"

    def _trainable_mask(self, step: int) -> Tensor:
        scope = self._trainable_scope(step)
        if scope == "global_relaxation":
            return self.model.active_mask
        if scope == "local_joint":
            return self.local_mask
        return self.child_mask

    def _rotation_training(self, step: int) -> bool:
        return self.global_train_rotations and self._trainable_scope(step) == "global_relaxation"

    @staticmethod
    def _positions(base_ids: Tensor, requested_ids: Tensor) -> Tensor:
        positions = torch.searchsorted(base_ids, requested_ids)
        if not bool((positions < base_ids.numel()).all()) or not torch.equal(
            base_ids[positions], requested_ids
        ):
            raise ValueError("Every edge-transfer row must remain active")
        return positions

    def pack_for_rasterizer(self, **kwargs: object) -> dict[str, Tensor]:
        pack = dict(self.model.pack_for_rasterizer(**kwargs))
        release = self._anchor_release()
        if release >= 1.0:
            return pack
        base_ids = pack["base_ids"]
        parent_positions = self._positions(base_ids, self.parent_ids)
        child_positions = self._positions(base_ids, self.child_ids)
        anchors = pack["means3d"].clone(memory_format=torch.preserve_format)
        parent_anchor = self.base_means + release * (
            pack["means3d"].index_select(0, parent_positions) - self.base_means
        )
        child_means = pack["means3d"].index_select(0, child_positions)
        child_anchor = self.base_means + release * (child_means - self.base_means)
        anchors.index_copy_(0, parent_positions, parent_anchor)
        anchors.index_copy_(0, child_positions, child_anchor)
        pack["view_anchor3d"] = anchors
        return pack

    def prepare_gradients(self, step: int) -> None:
        self.set_step(step)
        retraction_active = (
            step > self.retraction_start_step and self.retraction_optimizer is not None
        )
        if retraction_active:
            amount = self.retractions.detach()
            scale = 1.0 - 0.5 * amount
            grad = torch.zeros_like(amount)
            for name, derivative in (
                ("means3d", -0.5 * self.world_directions),
                ("log_scales", -0.5 / scale[:, None].expand(-1, 2)),
                ("colors", -0.5 * self.color_directions),
                ("color_grads", -0.5 * self.base_color_grads),
                (
                    "log_softness",
                    -0.5 / scale.reshape((-1,) + (1,) * (self.base_log_softness.ndim - 1)),
                ),
            ):
                parameter_grad = getattr(self.model, name).grad
                if parameter_grad is not None:
                    selected = parameter_grad.index_select(0, self.parent_ids)
                    grad += (selected * derivative).reshape(amount.numel(), -1).sum(dim=1)
            self.retraction_optimizer.zero_grad(set_to_none=True)
            self.retractions.grad = grad

        appearance_only = step <= self.appearance_steps
        trainable_mask = self._trainable_mask(step)
        rotation_training = self._rotation_training(step)
        for name in _BASE_TENSOR_NAMES:
            parameter = getattr(self.model, name)
            if name == "rotation_quats" and self.staged_training and not rotation_training:
                parameter.grad = None
                continue
            if parameter.grad is None:
                continue
            if name == "rotation_quats" and not rotation_training:
                parameter.grad.zero_()
            elif appearance_only and name in _BURN_IN_FROZEN_NAMES:
                parameter.grad.zero_()
            else:
                parameter.grad[~trainable_mask] = 0

    @torch.no_grad()
    def after_optimizer_step(self, step: int) -> None:
        if step > self.retraction_start_step and self.retraction_optimizer is not None:
            self.retraction_optimizer.step()
            ramp_step = step - self.retraction_start_step
            cap = self.max_retraction * min(1.0, ramp_step / self.retraction_ramp_steps)
            self.retractions.clamp_(0.0, cap)
        frozen_mask = ~self._trainable_mask(step)
        for name, values in self.frozen_parameters.items():
            getattr(self.model, name)[frozen_mask] = values[frozen_mask]
        for group in self.model_optimizer.param_groups:
            name = str(group.get("name", ""))
            parameters = group.get("params", [])
            if len(parameters) != 1:
                continue
            parameter = parameters[0]
            for state_name, value in self.model_optimizer.state.get(parameter, {}).items():
                frozen = self.frozen_optimizer_rows.get((name, state_name))
                if frozen is not None:
                    value[frozen_mask] = frozen[frozen_mask]
        if not self.staged_training:
            amount = self.retractions.detach()
            scale = 1.0 - 0.5 * amount
            self.model.means3d[self.parent_ids] = (
                self.base_means - 0.5 * amount[:, None] * self.world_directions
            )
            self.model.log_scales[self.parent_ids] = self.base_log_scales + torch.log(scale[:, None])
            self.model.colors[self.parent_ids] = (
                self.base_colors - 0.5 * amount[:, None] * self.color_directions
            )
            self.model.color_grads[self.parent_ids] = self.base_color_grads * scale[:, None, None]
            softness_scale = scale.reshape((-1,) + (1,) * (self.base_log_softness.ndim - 1))
            self.model.log_softness[self.parent_ids] = self.base_log_softness + torch.log(
                softness_scale
            )

    def state(self) -> dict[str, object]:
        values = self.retractions.detach().cpu()
        state = {
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "parent_ids": self.parent_ids.detach().cpu().tolist(),
            "child_ids": self.child_ids.detach().cpu().tolist(),
            "retractions": values.tolist(),
            "mean_retraction": float(values.mean()) if values.numel() else 0.0,
            "max_retraction": float(values.max()) if values.numel() else 0.0,
            "positive_retractions": int((values > 0.0).sum()),
            "anchor_release": self._anchor_release(),
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "complete": self.is_complete(),
        }
        if self.edge_ids is not None:
            state["edge_ids"] = self.edge_ids.tolist()
        elif self.vertex_edge_ids is not None:
            state["vertex_edge_ids"] = self.vertex_edge_ids.tolist()
        if self.staged_training:
            state["neighbor_ids"] = self.neighbor_ids.detach().cpu().tolist()
            if self.local_joint_start_step is not None:
                state["local_joint_start_step"] = self.local_joint_start_step
            if self.global_relaxation_start_step is not None:
                state["global_relaxation_start_step"] = self.global_relaxation_start_step
            state["trainable_scope"] = self._trainable_scope(self.current_step)
            state["global_train_rotations"] = self.global_train_rotations
            state["rotation_training"] = self._rotation_training(self.current_step)
        if self.has_round_provenance:
            state["round_index"] = self.round_index
            state["parent_edge_transfer_sha256"] = self.parent_edge_transfer_sha256
        return state
