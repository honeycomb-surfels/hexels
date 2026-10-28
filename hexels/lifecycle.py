from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from .population import BASE_ROW_NAMES


@dataclass(frozen=True)
class MergeThresholds:
    distance_scale_ratio: float = 0.02
    quaternion_cosine: float = 0.9999
    log_scale_delta: float = 0.03
    color_delta: float = 0.01
    field_delta: float = 0.02
    max_pairs: int = 256


def _opacity_to_optical_depth(opacity: Tensor) -> Tensor:
    return -torch.log1p(-opacity.clamp(0.0, 1.0 - torch.finfo(opacity.dtype).eps))


def _optical_depth_to_logits(optical_depth: Tensor) -> Tensor:
    opacity = -torch.expm1(-optical_depth)
    eps = torch.finfo(opacity.dtype).eps
    opacity = opacity.clamp(eps, 1.0 - eps)
    return torch.logit(opacity)


@torch.no_grad()
def redistribute_parent_child_opacity(
    parent_opacity_logits: Tensor,
    child_opacity_logits: Tensor,
    child_parent_positions: Tensor,
    *,
    transfer_fraction: float,
) -> tuple[Tensor, Tensor]:
    """Redistribute peak optical depth, not pixelwise coverage of shifted soft footprints."""
    if parent_opacity_logits.ndim != 1 or child_opacity_logits.ndim != 1:
        raise ValueError("Parent and child opacity logits must be one-dimensional")
    if (
        child_parent_positions.dtype != torch.long
        or child_parent_positions.shape != child_opacity_logits.shape
    ):
        raise ValueError("child_parent_positions must be long and match child logits")
    if not 0.0 <= transfer_fraction <= 1.0:
        raise ValueError("transfer_fraction must be in [0,1]")
    if child_parent_positions.numel() == 0:
        return parent_opacity_logits.detach().clone(), child_opacity_logits.detach().clone()
    if (
        int(child_parent_positions.min()) < 0
        or int(child_parent_positions.max()) >= parent_opacity_logits.numel()
    ):
        raise ValueError("child_parent_positions contains an invalid parent position")

    # Optical depths add under alpha compositing; raw opacities do not.
    # This transfers peak opacity mass, not exact coverage of displaced footprints.
    parent_tau = _opacity_to_optical_depth(torch.sigmoid(parent_opacity_logits))
    learned_child_tau = _opacity_to_optical_depth(torch.sigmoid(child_opacity_logits))
    parent_positions = child_parent_positions.to(device=parent_tau.device)
    learned_child_tau = learned_child_tau.to(device=parent_tau.device)

    child_tau_sum = torch.zeros_like(parent_tau)
    child_tau_sum.index_add_(0, parent_positions, learned_child_tau)
    child_count = torch.zeros_like(parent_tau)
    child_count.index_add_(0, parent_positions, torch.ones_like(learned_child_tau))
    fallback_weight = 1.0 / child_count[parent_positions].clamp_min(1.0)
    learned_weight = learned_child_tau / child_tau_sum[parent_positions].clamp_min(
        torch.finfo(parent_tau.dtype).tiny
    )
    child_weight = torch.where(
        child_tau_sum[parent_positions] > 0.0,
        learned_weight,
        fallback_weight,
    )

    parent_result = _optical_depth_to_logits(parent_tau * (1.0 - transfer_fraction))
    child_result = _optical_depth_to_logits(
        learned_child_tau
        + parent_tau[parent_positions] * transfer_fraction * child_weight
    )
    return parent_result, child_result.to(device=child_opacity_logits.device)


def _normalized_quaternions(rotation_quats: Tensor) -> np.ndarray:
    return torch.nn.functional.normalize(
        rotation_quats.detach().cpu().to(torch.float64),
        dim=1,
    ).numpy()


def _maximum_delta(state: Mapping[str, Tensor], first: int, second: int) -> float:
    names = (
        "color_grads",
        "color_quads",
        "view_grads",
        "specular_ks",
        "specular_log_s",
        "log_softness",
    )
    return max(
        float(
            torch.max(
                torch.abs(
                    state[name][first].detach().cpu()
                    - state[name][second].detach().cpu()
                )
            ).item()
        )
        for name in names
    )


@torch.no_grad()
def compatible_merge_pairs(
    state: Mapping[str, Tensor],
    thresholds: MergeThresholds = MergeThresholds(),
) -> tuple[Tensor, Tensor]:
    """Return deterministic, disjoint pairs eligible for render-validated merging."""
    means = state["means3d"]
    scales = torch.exp(state["log_scales"])
    rotations = state["rotation_quats"]
    colors = state["colors"]
    active_mask = state["active_mask"]
    population = int(means.shape[0])
    if (
        means.shape != (population, 3)
        or scales.shape != (population, 2)
        or rotations.shape != (population, 4)
        or colors.shape != (population, 3)
        or active_mask.dtype != torch.bool
        or active_mask.shape != (population,)
    ):
        raise ValueError("Model state has incompatible base tensor shapes")
    if thresholds.distance_scale_ratio <= 0.0 or thresholds.max_pairs < 1:
        raise ValueError("Merge distance and maximum pair count must be positive")

    active_ids = torch.nonzero(active_mask.detach().cpu(), as_tuple=False).flatten()
    if active_ids.numel() < 2:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty
    active_means = means.detach().cpu()[active_ids].to(torch.float64).numpy()
    distances, neighbors = cKDTree(active_means).query(active_means, k=2, workers=-1)
    quaternions = _normalized_quaternions(rotations)
    scales_cpu = scales.detach().cpu()
    log_scales_cpu = state["log_scales"].detach().cpu()
    colors_cpu = colors.detach().cpu()

    candidates: list[tuple[float, int, int]] = []
    for active_position, neighbor_position in enumerate(neighbors[:, 1]):
        first = int(active_ids[active_position])
        second = int(active_ids[int(neighbor_position)])
        if first == second:
            continue
        keeper, victim = sorted((first, second))
        scale = float(
            torch.minimum(
                scales_cpu[keeper].max(),
                scales_cpu[victim].max(),
            ).item()
        )
        normalized_distance = float(distances[active_position, 1]) / max(scale, 1e-12)
        if normalized_distance > thresholds.distance_scale_ratio:
            continue
        if abs(float(np.dot(quaternions[keeper], quaternions[victim]))) < thresholds.quaternion_cosine:
            continue
        if float(torch.max(torch.abs(log_scales_cpu[keeper] - log_scales_cpu[victim]))) > thresholds.log_scale_delta:
            continue
        if float(torch.max(torch.abs(colors_cpu[keeper] - colors_cpu[victim]))) > thresholds.color_delta:
            continue
        if _maximum_delta(state, keeper, victim) > thresholds.field_delta:
            continue
        candidates.append((normalized_distance, keeper, victim))

    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for _, keeper, victim in sorted(set(candidates)):
        if keeper in used or victim in used:
            continue
        selected.append((keeper, victim))
        used.update((keeper, victim))
        if len(selected) == thresholds.max_pairs:
            break
    if not selected:
        empty = torch.empty((0,), dtype=torch.long)
        return empty, empty
    selected.sort()
    keepers, victims = zip(*selected)
    return torch.tensor(keepers, dtype=torch.long), torch.tensor(victims, dtype=torch.long)


@torch.no_grad()
def merged_rows(
    state: Mapping[str, Tensor],
    keeper_ids: Tensor,
    victim_ids: Tensor,
) -> dict[str, Tensor]:
    """Construct optical-depth-weighted rows for prevalidated merge pairs."""
    if (
        keeper_ids.dtype != torch.long
        or victim_ids.dtype != torch.long
        or keeper_ids.ndim != 1
        or keeper_ids.shape != victim_ids.shape
    ):
        raise ValueError("Merge IDs must be aligned one-dimensional long tensors")
    if keeper_ids.numel() == 0:
        return {name: state[name][:0].detach().clone() for name in state if name != "active_mask"}
    population = int(state["means3d"].shape[0])
    if (
        int(torch.min(torch.cat((keeper_ids, victim_ids)))) < 0
        or int(torch.max(torch.cat((keeper_ids, victim_ids)))) >= population
        or torch.unique(torch.cat((keeper_ids, victim_ids))).numel() != 2 * keeper_ids.numel()
    ):
        raise ValueError("Merge IDs must be valid and disjoint")

    device = state["means3d"].device
    keepers = keeper_ids.to(device=device)
    victims = victim_ids.to(device=device)
    keeper_tau = _opacity_to_optical_depth(torch.sigmoid(state["opacity_logits"][keepers]))
    victim_tau = _opacity_to_optical_depth(torch.sigmoid(state["opacity_logits"][victims]))
    total_tau = keeper_tau + victim_tau
    keeper_weight = keeper_tau / total_tau.clamp_min(torch.finfo(total_tau.dtype).tiny)
    victim_weight = 1.0 - keeper_weight

    rows: dict[str, Tensor] = {}
    for name, value in state.items():
        if name.startswith("child_") or name.startswith("_") or name == "active_mask":
            continue
        first = value[keepers]
        second = value[victims]
        weight_shape = (keeper_weight.shape[0],) + (1,) * (first.ndim - 1)
        weight = keeper_weight.reshape(weight_shape)
        rows[name] = first * weight + second * victim_weight.reshape(weight_shape)
    rows["opacity_logits"] = _optical_depth_to_logits(total_tau)

    first_quaternion = state["rotation_quats"][keepers]
    second_quaternion = state["rotation_quats"][victims]
    sign = torch.where(
        torch.sum(first_quaternion * second_quaternion, dim=1, keepdim=True) < 0.0,
        -torch.ones_like(keeper_weight[:, None]),
        torch.ones_like(keeper_weight[:, None]),
    )
    weight = keeper_weight[:, None]
    rows["rotation_quats"] = torch.nn.functional.normalize(
        first_quaternion * weight + second_quaternion * sign * (1.0 - weight),
        dim=1,
    )
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def compact_active_resume(
    checkpoint_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Physically remove inactive rows while preserving aligned Adam state."""
    checkpoint_path = checkpoint_path.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = json.loads(config_path.read_text())
    model_state = checkpoint["model"]
    optimizer_state = checkpoint["optimizer"]
    active_mask = model_state["active_mask"]
    population = int(active_mask.numel())
    if active_mask.dtype != torch.bool or active_mask.shape != (population,):
        raise ValueError("Checkpoint active mask is invalid")
    if not bool(active_mask.any()) or bool(active_mask.all()):
        raise ValueError("Compaction requires both active and inactive rows")
    if int(model_state["child_parent_idx"].numel()) != 0:
        raise ValueError("Compaction requires an empty learned-child population")

    keep = active_mask.detach().cpu()
    output = copy.deepcopy(checkpoint)
    output_model = output["model"]
    for name in BASE_ROW_NAMES:
        output_model[name] = output_model[name][keep].clone()
    output_model["active_mask"] = torch.ones(
        (int(keep.sum()),),
        dtype=torch.bool,
    )
    output_optimizer = output["optimizer"]
    for parameter_state in output_optimizer["state"].values():
        for name, value in tuple(parameter_state.items()):
            if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == population:
                parameter_state[name] = value[keep].clone()

    geometry = {
        "means3d": output_model["means3d"].to(torch.float32),
        "scales": output_model["log_scales"].exp().to(torch.float32),
        "rotations": torch.nn.functional.normalize(
            output_model["rotation_quats"].to(torch.float32),
            dim=1,
        ),
        "colors": output_model["colors"].to(torch.float32).clamp(0.0, 1.0),
    }
    output_dir.mkdir(parents=True)
    geometry_path = output_dir / "initial_geometry.pt"
    torch.save(geometry, geometry_path)
    geometry_record = {
        "path": str(geometry_path),
        "sha256": _sha256(geometry_path),
        "point_count": int(keep.sum()),
    }
    output["init_geometry"] = geometry_record
    output["resume_compatible"] = True
    output["lineage"] = {
        "kind": "hexels3d_active_compaction",
        "parent_checkpoint": str(checkpoint_path),
        "parent_checkpoint_sha256": _sha256(checkpoint_path),
        "parent_lineage": copy.deepcopy(checkpoint.get("lineage")),
        "capacity_before": population,
        "active_count": int(keep.sum()),
        "removed_count": int((~keep).sum()),
    }

    config["init_geometry_path"] = str(geometry_path)
    config["model_path"] = str(output_dir)
    config["resume_checkpoint"] = None
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n"
    )
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir()
    output_checkpoint = checkpoint_dir / f"ckpt_{int(output['iteration']):06d}.pt"
    torch.save(output, output_checkpoint)
    return {
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": _sha256(output_checkpoint),
        "geometry": geometry_record,
        **output["lineage"],
    }
