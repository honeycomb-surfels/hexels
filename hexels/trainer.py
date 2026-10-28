from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
import torch

from .attributed_residual import (
    OwnerLocalMomentSummary,
    aggregate_attributed_residual,
    aggregate_owner_local_moments,
    summarize_attributed_edge_evidence,
    summarize_owner_local_moments,
)
from .colmap import load_colmap_scene
from .edge_shift import (
    canonical_edge_normals,
    residual_first_moments,
    summarize_residual_topk_views,
    update_residual_topk_views,
)
from .edge_transfer import EdgeTransferController
from .initial_geometry import load_initial_geometry
from .lifecycle import (
    MergeThresholds,
    compatible_merge_pairs,
    merged_rows,
    redistribute_parent_child_opacity,
)
from .mcmc import compute_relocation, init_relocation_buffers, opacity_to_logits
from .model import Hexels3DGeometry, Hexels3DModel, validate_active_mask_state
from .mvs_surface import (
    MvsSurfaceLoss,
    MvsSurfaceTargets,
    load_mvs_surface_targets,
    rendered_mvs_surface_loss,
)
from .normal_trust import (
    load_normal_trust_sidecar,
    project_normal_trust_cone,
    quaternion_normal_z,
)
from .options import Hexels3DOptions, color_model_uses_quadratic_coefficients
from .render import _box_downsample, _scaled_camera
from .refinement import (
    covariance_pair_children,
    directional_edge_children,
    rank_absolute_candidates,
    symmetric_split_children,
    transferred_linear_appearance,
)
from .renderer import Hexels3DRenderer
from .surface_objective import surface_normal_loss
from .train_camera_manifest import select_train_cameras


_RESUME_CONFIG_FIELDS = (
    "images",
    "eval",
    "llffhold",
    "resolution",
    "seed",
    "camera_sampling",
    "train_mode",
    "softness_mode",
    "scale_modifier",
    "color_model",
    "use_view_grads",
    "view_model",
    "sh_progressive_interval",
    "init_normal_mode",
    "spatial_appearance_start",
    "lambda_dssim",
    "white_background",
    "prefilter_edge_on",
    "repulsion_screen_weight",
    "repulsion_screen_target",
    "repulsion_knn_weight",
    "surface_coplanar_weight",
    "surface_normal_smooth_weight",
    "surface_normal_depth_weight",
    "surface_normal_depth_start",
    "surface_normal_depth_alpha_thresh",
    "surface_normal_objective",
    "surface_normal_depth_mode",
    "lr_means3d",
    "lr_means3d_final",
    "lr_means3d_max_steps",
    "lr_means3d_delay_steps",
    "lr_means3d_delay_mult",
    "spatial_lr_scale",
)


def load_camera_sampling_weights(
    path: Path,
    camera_names: Sequence[str],
) -> tuple[list[float], str]:
    value = json.loads(path.read_text())
    weights = value.get("weights") if isinstance(value, Mapping) else None
    if value.get("schema_version") != 1 or not isinstance(weights, Mapping):
        raise ValueError("Camera sampling weights must use schema version 1")
    if set(weights) != set(camera_names):
        raise ValueError("Camera sampling weight names differ from training cameras")
    ordered = []
    for name in camera_names:
        weight = weights[name]
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise ValueError("Camera sampling weights must be positive and finite")
        ordered.append(float(weight))
    return ordered, _sha256_file(path)


@dataclass
class _DensifyShadowEvent:
    parent_ids: torch.Tensor
    child_ids: torch.Tensor
    birth_iteration: int
    steps_completed: int = 0
    birth_parent_distances: Optional[torch.Tensor] = None
    birth_physical_max_scales: Optional[torch.Tensor] = None
    birth_log_opacities: Optional[torch.Tensor] = None
    envelope_projection_steps: int = 0
    envelope_projected_mean_rows: int = 0
    envelope_projected_scale_rows: int = 0
    envelope_projected_opacity_rows: int = 0
    residual_topk_evidence: Optional[dict[str, object]] = None
    attributed_residual_evidence: Optional[dict[str, object]] = None


@dataclass
class _DensifyJointCoplanarEvent:
    parent_ids: torch.Tensor
    child_ids: torch.Tensor
    birth_iteration: int
    configured_steps: int
    steps_completed: int = 0
    first_constrained_iteration: Optional[int] = None
    last_constrained_iteration: Optional[int] = None


class _DensifyShadowViewAnchor:
    def __init__(
        self,
        model: Hexels3DModel,
        event: _DensifyShadowEvent,
        release: float,
    ) -> None:
        self.model = model
        self.event = event
        self.release = float(release)

    def __getattr__(self, name: str) -> object:
        return getattr(self.model, name)

    def pack_for_rasterizer(self, **kwargs: object) -> dict[str, torch.Tensor]:
        pack = dict(self.model.pack_for_rasterizer(**kwargs))
        base_ids = pack["base_ids"]
        parent_positions = torch.searchsorted(base_ids, self.event.parent_ids)
        child_positions = torch.searchsorted(base_ids, self.event.child_ids)
        if not torch.equal(base_ids[parent_positions], self.event.parent_ids):
            raise RuntimeError("Shadow parent rows must remain active during view anchoring")
        if not torch.equal(base_ids[child_positions], self.event.child_ids):
            raise RuntimeError("Shadow child rows must remain active during view anchoring")
        anchors = pack["means3d"].clone(memory_format=torch.preserve_format)
        parent_means = pack["means3d"].index_select(0, parent_positions)
        child_means = pack["means3d"].index_select(0, child_positions)
        child_anchors = parent_means + self.release * (child_means - parent_means)
        anchors.index_copy_(0, child_positions, child_anchors)
        pack["view_anchor3d"] = anchors
        return pack


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(f"{value.dtype}:{tuple(value.shape)}:".encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _split_resume_train_cameras(
    cameras: list[object],
    *,
    holdout_stride: int,
) -> tuple[list[object], list[object]]:
    named: list[tuple[str, object]] = []
    for camera in cameras:
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Every resume training camera must have a non-empty image_name")
        named.append((name, camera))
    named.sort(key=lambda item: item[0])
    names = [name for name, _ in named]
    if len(set(names)) != len(names):
        raise ValueError("Resume training camera image_name values must be unique")
    if holdout_stride < 2:
        raise ValueError("resume_train_holdout_stride must be at least 2")
    train = [camera for index, (_, camera) in enumerate(named) if index % holdout_stride != 0]
    holdout = [camera for index, (_, camera) in enumerate(named) if index % holdout_stride == 0]
    if not train:
        raise ValueError(f"resume_train_holdout_stride={holdout_stride} leaves no training cameras")
    return train, holdout


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_resume_train_camera_manifest(
    cameras: list[object],
    *,
    manifest_path: Path,
) -> tuple[list[object], list[object], list[str], str, str]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ValueError(f"Resume training camera manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("Resume training camera manifest must contain a mapping")
    expected_keys = {"schema_version", "camera_names", "camera_names_sha256"}
    if set(manifest) != expected_keys:
        raise ValueError(
            "Resume training camera manifest must contain exactly: "
            + ", ".join(sorted(expected_keys))
        )
    if manifest["schema_version"] != 1 or isinstance(manifest["schema_version"], bool):
        raise ValueError("Resume training camera manifest schema_version must be 1")
    camera_names = manifest["camera_names"]
    if (
        not isinstance(camera_names, list)
        or not camera_names
        or any(not isinstance(name, str) or not name for name in camera_names)
    ):
        raise ValueError("Resume training camera manifest camera_names must be non-empty strings")
    if camera_names != sorted(camera_names) or len(set(camera_names)) != len(camera_names):
        raise ValueError("Resume training camera manifest camera_names must be sorted and unique")
    camera_names_sha256 = _json_sha256(camera_names)
    if manifest["camera_names_sha256"] != camera_names_sha256:
        raise ValueError("Resume training camera manifest camera_names_sha256 mismatch")

    named_cameras: dict[str, object] = {}
    for camera in cameras:
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Every resume training camera must have a non-empty image_name")
        if name in named_cameras:
            raise ValueError("Resume training camera image_name values must be unique")
        named_cameras[name] = camera
    unknown = sorted(set(camera_names) - set(named_cameras))
    if unknown:
        raise ValueError(
            "Resume training camera manifest contains unknown camera names: "
            + ", ".join(unknown)
        )
    selected = [named_cameras[name] for name in camera_names]
    selected_names = set(camera_names)
    excluded = [named_cameras[name] for name in sorted(set(named_cameras) - selected_names)]
    return (
        selected,
        excluded,
        list(camera_names),
        camera_names_sha256,
        _sha256_file(manifest_path),
    )


def _should_render_progress(
    iteration: int,
    *,
    iterations: int,
    render_every: int,
) -> bool:
    return render_every > 0 and (
        iteration % render_every == 0 or iteration == iterations
    )


def _balanced_densify_shadow_groups(
    parent_ids: torch.Tensor,
    child_ids: torch.Tensor,
    group_count: int,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    if parent_ids.ndim != 1 or child_ids.ndim != 1 or parent_ids.shape != child_ids.shape:
        raise ValueError("Shadow salvage parent_ids and child_ids must be aligned vectors")
    count = int(parent_ids.numel())
    if group_count < 2 or group_count > count:
        raise ValueError("Shadow salvage group count must be in [2, child_count]")
    parents_cpu = parent_ids.detach().to(device="cpu", dtype=torch.long)
    children_cpu = child_ids.detach().to(device="cpu", dtype=torch.long)
    if int(torch.unique(parents_cpu).numel()) != count:
        raise ValueError("Shadow salvage requires one unique parent per child")
    if int(torch.unique(children_cpu).numel()) != count:
        raise ValueError("Shadow salvage child IDs must be unique")
    order = sorted(
        range(count),
        key=lambda index: (int(parents_cpu[index]), int(children_cpu[index])),
    )
    order_tensor = torch.tensor(order, device=parent_ids.device, dtype=torch.long)
    sorted_parents = parent_ids.index_select(0, order_tensor)
    sorted_children = child_ids.index_select(0, order_tensor)
    return [
        (group_id, sorted_parents[group_id::group_count], sorted_children[group_id::group_count])
        for group_id in range(group_count)
    ]


def _compare_densify_shadow_renders(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("Shadow render comparisons require equal non-empty camera records")
    reference_sse = 0.0
    candidate_sse = 0.0
    deltas: list[float] = []
    per_view: list[dict[str, object]] = []
    finite = True
    for reference_record, candidate_record in zip(reference, candidate):
        reference_name = str(reference_record["image_name"])
        if str(candidate_record["image_name"]) != reference_name:
            raise ValueError("Shadow render comparison camera order differs")
        reference_count = int(reference_record["count"])
        if int(candidate_record["count"]) != reference_count or reference_count <= 0:
            raise ValueError("Shadow render comparison target sizes differ")
        reference_view_sse = float(reference_record["sse"])
        candidate_view_sse = float(candidate_record["sse"])
        if not math.isfinite(reference_view_sse) or not math.isfinite(candidate_view_sse):
            finite = False
            break
        reference_sse += reference_view_sse
        candidate_sse += candidate_view_sse
        reference_psnr = -10.0 * math.log10(max(reference_view_sse / reference_count, 1e-12))
        candidate_psnr = -10.0 * math.log10(max(candidate_view_sse / reference_count, 1e-12))
        delta = candidate_psnr - reference_psnr
        deltas.append(delta)
        per_view.append({"image_name": reference_name, "psnr_delta": float(delta)})
    if finite and len(deltas) == len(reference):
        pooled = (reference_sse - candidate_sse) / max(reference_sse, 1e-12)
        mean_delta = sum(deltas) / len(deltas)
        win_fraction = sum(delta > 0.0 for delta in deltas) / len(deltas)
        worst_delta = min(deltas)
        finite = all(
            math.isfinite(value)
            for value in (pooled, mean_delta, win_fraction, worst_delta)
        )
    else:
        pooled = -1e30
        mean_delta = -1e30
        win_fraction = 0.0
        worst_delta = -1e30
        per_view = []
    return {
        "finite": bool(finite),
        "num_training_cameras": len(reference),
        "reference_sse": float(reference_sse),
        "candidate_sse": float(candidate_sse),
        "pooled_sse_reduction": float(pooled),
        "mean_psnr_delta": float(mean_delta),
        "win_fraction": float(win_fraction),
        "worst_psnr_delta": float(worst_delta),
        "per_view": per_view,
    }


def _get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    *,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int,
) -> object:
    """Exponential LR decay matching 2DGS/3DGS-MCMC schedule."""
    import numpy as np

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0.0, 1.0)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max(max_steps, 1), 0.0, 1.0)
        log_lerp = np.exp(np.log(max(lr_init, 1e-30)) * (1.0 - t) + np.log(max(lr_final, 1e-30)) * t)
        return float(delay_rate * log_lerp)

    return helper


def _seed_everything(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coverage_weighted_alpha_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    residual = (prediction.detach() - target).abs().mean(dim=0)
    return (residual * (1.0 - alpha).square()).mean()


def _residual_mass_or_density_mask(
    mass: torch.Tensor,
    contribution: torch.Tensor,
) -> torch.Tensor:
    visible_ids = torch.nonzero((mass > 0.0) & (contribution > 0.0), as_tuple=True)[0]
    material = torch.zeros_like(mass, dtype=torch.bool)
    if visible_ids.numel() == 0:
        return material

    count = max(1, (visible_ids.numel() + 9) // 10)
    density = mass[visible_ids] / contribution[visible_ids]
    mass_order = torch.argsort(mass[visible_ids], descending=True, stable=True)
    density_order = torch.argsort(density, descending=True, stable=True)
    material[visible_ids[mass_order[:count]]] = True
    material[visible_ids[density_order[:count]]] = True
    return material


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def _sh_degree_for_iteration(iteration: int, interval: int) -> int:
    if interval == 0:
        return 3
    return min(iteration // interval, 3)


def _knn_initial_scales(
    points: torch.Tensor,
    *,
    k: int,
    multiplier: float,
    max_scale: float,
) -> torch.Tensor:
    from scipy.spatial import cKDTree

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (P,3), got {tuple(points.shape)}")
    if points.shape[0] <= k:
        raise ValueError(f"Need more than {k} points for kNN initialization")
    points_np = points.detach().cpu().numpy()
    distances, _ = cKDTree(points_np).query(points_np, k=k + 1, workers=-1)
    rms = (distances[:, 1:] ** 2).mean(axis=1) ** 0.5
    scale = torch.from_numpy(rms).to(device=points.device, dtype=points.dtype)
    scale = scale.clamp_min(1e-7**0.5) * float(multiplier)
    if max_scale > 0.0:
        scale = scale.clamp_max(float(max_scale))
    return scale[:, None].repeat(1, 2)


def _save_image(path: Path, image_chw: torch.Tensor) -> None:
    from PIL import Image

    img = image_chw.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def _quat_to_normal_z(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    n = torch.stack([nx, ny, nz], dim=-1)
    return torch.nn.functional.normalize(n, dim=-1)


def _quat_to_tangent_xy(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    tx = torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dim=-1,
    )
    ty = torch.stack(
        [
            2.0 * (x * y - w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + w * x),
        ],
        dim=-1,
    )
    tx = torch.nn.functional.normalize(tx, dim=-1)
    ty = torch.nn.functional.normalize(ty, dim=-1)
    return tx, ty


def _depths_to_points(camera: object, depth_hw: torch.Tensor) -> torch.Tensor:
    if depth_hw.ndim != 2:
        raise ValueError(f"depth_hw must have shape (H,W), got {tuple(depth_hw.shape)}")
    device = depth_hw.device
    dtype = depth_hw.dtype

    W = int(camera.image_width)
    H = int(camera.image_height)
    if depth_hw.shape != (H, W):
        raise ValueError(f"depth_hw shape {tuple(depth_hw.shape)} does not match camera {(H, W)}")

    c2w = torch.linalg.inv(camera.world_view_transform.T)
    ndc2pix = torch.tensor(
        [
            [W / 2.0, 0.0, 0.0, W / 2.0],
            [0.0, H / 2.0, 0.0, H / 2.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    ).T
    projection_matrix = c2w.T @ camera.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T
    intrins_inv_t = torch.linalg.inv(intrins).T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )
    pix = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pix @ intrins_inv_t @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depth_hw.reshape(-1, 1) * rays_d + rays_o
    return points.reshape(H, W, 3)


def _depth_to_normal(camera: object, depth_hw: torch.Tensor) -> torch.Tensor:
    points = _depths_to_points(camera, depth_hw)
    output = torch.zeros_like(points)

    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normal_map = torch.cross(dx, dy, dim=-1)
    normal_map = torch.nn.functional.normalize(normal_map, dim=-1)
    output[1:-1, 1:-1] = normal_map
    return output


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def _quat_from_two_vectors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    dot = (a * b).sum(dim=-1, keepdim=True)
    cross = torch.cross(a, b, dim=-1)

    w = 1.0 + dot
    q = torch.cat([w, cross], dim=-1)

    near_opposite = w.squeeze(-1) < 1e-6
    if bool(near_opposite.any().item()):
        axis_x = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a)
        axis_y = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a)
        use_x = a[:, 0].abs() < 0.9
        axis = torch.where(use_x[:, None], axis_x, axis_y)
        v = torch.nn.functional.normalize(torch.cross(a, axis, dim=-1), dim=-1)
        q = torch.where(near_opposite[:, None], torch.cat([torch.zeros_like(dot), v], dim=-1), q)

    return torch.nn.functional.normalize(q, dim=-1)


class Hexels3DTrainer:
    def __init__(self, opt: Hexels3DOptions) -> None:
        self.opt = opt
        _seed_everything(opt.seed)

        self.model_path = Path(opt.model_path)
        self.model_path.mkdir(parents=True, exist_ok=False)
        (self.model_path / "renders").mkdir(parents=True, exist_ok=True)
        (self.model_path / "checkpoints").mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("hexels")
        self.logger.setLevel(logging.INFO)
        self._setup_logging()
        self.logger.info("Opt: %s", opt)

        opt_dict = opt.as_dict() if hasattr(opt, "as_dict") else vars(opt)
        with open(self.model_path / "config.json", "w") as f:
            json.dump(opt_dict, f, indent=2, default=str)

        self.device = torch.device(opt.data_device)
        self.resume_parameter_policy = str(
            getattr(opt, "resume_parameter_policy", "inherit")
        )
        self.resume_geometry_policy = str(getattr(opt, "resume_geometry_policy", "inherit"))
        resume_scale_lower_geometry_path = getattr(
            opt, "resume_scale_lower_geometry_path", None
        )
        self.resume_scale_lower_geometry_path = (
            Path(resume_scale_lower_geometry_path).resolve()
            if resume_scale_lower_geometry_path is not None
            else None
        )
        resume_scale_upper_geometry_path = getattr(
            opt, "resume_scale_upper_geometry_path", None
        )
        self.resume_scale_upper_geometry_path = (
            Path(resume_scale_upper_geometry_path).resolve()
            if resume_scale_upper_geometry_path is not None
            else None
        )
        self.resume_scale_anchor_weight = float(
            getattr(opt, "resume_scale_anchor_weight", 0.0)
        )
        self.resume_scale_lower_log: torch.Tensor | None = None
        self.resume_scale_upper_log: torch.Tensor | None = None
        self.resume_scale_constraint_metadata: dict[str, object] | None = None
        self.resume_checkpoint_scale_constraint: dict[str, object] | None = None
        resume_normal_trust_path = getattr(opt, "resume_normal_trust_path", None)
        self.resume_normal_trust_path = (
            Path(resume_normal_trust_path).resolve()
            if resume_normal_trust_path is not None
            else None
        )
        self.resume_normal_trust_min_degrees = float(
            getattr(opt, "resume_normal_trust_min_degrees", 5.0)
        )
        self.resume_normal_trust_max_degrees = float(
            getattr(opt, "resume_normal_trust_max_degrees", 25.0)
        )
        self.resume_normal_trust_reference: torch.Tensor | None = None
        self.resume_normal_trust_reliability: torch.Tensor | None = None
        self.resume_normal_trust_cones: torch.Tensor | None = None
        self.resume_normal_trust_metadata: dict[str, object] | None = None
        self.resume_checkpoint_normal_trust: dict[str, object] | None = None
        self.resume_normal_trust_projection_steps = 0
        self.resume_normal_trust_projected_rows = 0
        resume_surface_anchor_path = getattr(opt, "resume_surface_anchor_path", None)
        self.resume_surface_anchor_path = (
            Path(resume_surface_anchor_path).resolve()
            if resume_surface_anchor_path is not None
            else None
        )
        self.resume_surface_anchor_weight = float(
            getattr(opt, "resume_surface_anchor_weight", 0.0)
        )
        self.resume_surface_anchor_reference: torch.Tensor | None = None
        self.resume_surface_anchor_normals: torch.Tensor | None = None
        self.resume_surface_anchor_reliability: torch.Tensor | None = None
        self.resume_surface_anchor_scales: torch.Tensor | None = None
        self.resume_surface_anchor_metadata: dict[str, object] | None = None
        self.resume_checkpoint_surface_anchor: dict[str, object] | None = None
        resume_mvs_surface_path = getattr(opt, "resume_mvs_surface_path", None)
        self.resume_mvs_surface_path = (
            Path(resume_mvs_surface_path).resolve()
            if resume_mvs_surface_path is not None
            else None
        )
        self.resume_mvs_surface_weight = float(
            getattr(opt, "resume_mvs_surface_weight", 0.0)
        )
        self.resume_mvs_surface_targets: MvsSurfaceTargets | None = None
        self.resume_mvs_surface_metadata: dict[str, object] | None = None
        self.resume_checkpoint_mvs_surface: dict[str, object] | None = None
        self.resume_parent_geometry_policy = "inherit"
        self.resume_allow_lambda_dssim_change = bool(
            getattr(opt, "resume_allow_lambda_dssim_change", False)
        )
        self.resume_allow_photometric_loss_change = bool(
            getattr(opt, "resume_allow_photometric_loss_change", False)
        )
        self.resume_allow_camera_sampling_change = bool(
            getattr(opt, "resume_allow_camera_sampling_change", False)
        )
        self.resume_parent_lambda_dssim: Optional[float] = None
        self.resume_parent_photometric_loss: Optional[str] = None
        self.resume_parent_camera_sampling: Optional[str] = None
        self.resume_parent_train_holdout_stride = 0
        self.resume_train_holdout_stride = int(getattr(opt, "resume_train_holdout_stride", 0))
        resume_train_camera_manifest = getattr(opt, "resume_train_camera_manifest", None)
        self.resume_train_camera_manifest_path = (
            Path(resume_train_camera_manifest).resolve()
            if resume_train_camera_manifest is not None
            else None
        )
        self.resume_train_camera_manifest_sha256: Optional[str] = None
        self.resume_train_camera_names: list[str] = []
        self.resume_train_camera_names_sha256: Optional[str] = None
        self.resume_train_camera_trace_names: list[str] = []
        self.train_camera_trace_names: list[str] = []
        self.resume_train_holdout_cameras: list[object] = []
        self.resume_train_holdout_camera_names: list[str] = []
        self.train_camera_selection: dict[str, object] | None = None

        scene = load_colmap_scene(opt)
        self.train_cameras = scene.train_cameras
        self.test_cameras = scene.test_cameras
        train_camera_manifest = getattr(opt, "train_camera_manifest", None)
        if train_camera_manifest is not None:
            self.train_cameras, self.train_camera_selection = select_train_cameras(
                self.train_cameras,
                train_camera_manifest,
            )
            self.logger.info(
                "Training camera manifest: retained=%d sha256=%s",
                len(self.train_cameras),
                self.train_camera_selection["manifest_sha256"],
            )
        if self.resume_train_camera_manifest_path is not None:
            (
                self.train_cameras,
                self.resume_train_holdout_cameras,
                self.resume_train_camera_names,
                self.resume_train_camera_names_sha256,
                self.resume_train_camera_manifest_sha256,
            ) = _select_resume_train_camera_manifest(
                self.train_cameras,
                manifest_path=self.resume_train_camera_manifest_path,
            )
            self.resume_train_holdout_camera_names = [
                str(camera.image_name) for camera in self.resume_train_holdout_cameras
            ]
            self.logger.info(
                "Resume training camera manifest: retained=%d excluded=%d sha256=%s",
                len(self.train_cameras),
                len(self.resume_train_holdout_cameras),
                self.resume_train_camera_manifest_sha256,
            )
        elif self.resume_train_holdout_stride > 0:
            self.train_cameras, self.resume_train_holdout_cameras = _split_resume_train_cameras(
                self.train_cameras,
                holdout_stride=self.resume_train_holdout_stride,
            )
            self.resume_train_holdout_camera_names = [
                str(camera.image_name) for camera in self.resume_train_holdout_cameras
            ]
            self.logger.info(
                "Resume training holdout: stride=%d retained=%d held_out=%d names=%s",
                self.resume_train_holdout_stride,
                len(self.train_cameras),
                len(self.resume_train_holdout_cameras),
                self.resume_train_holdout_camera_names,
            )
        self.densify_shadow_gate_cameras: list[object] = []
        self.densify_shadow_gate_camera_names: set[str] = set()
        self.densify_shadow_fit_cameras = list(self.train_cameras)
        self.densify_shadow_fit_camera_names = {
            str(camera.image_name) for camera in self.train_cameras
        }
        self.densify_shadow_selector_cameras: list[object] = []
        self.densify_shadow_selector_camera_names: set[str] = set()
        shadow_gate_stride = int(getattr(opt, "densify_shadow_gate_stride", 0))
        if shadow_gate_stride > 0:
            fit_cameras, self.densify_shadow_gate_cameras = _split_resume_train_cameras(
                self.train_cameras,
                holdout_stride=shadow_gate_stride,
            )
            self.densify_shadow_fit_camera_names = {
                str(camera.image_name) for camera in fit_cameras
            }
            self.densify_shadow_fit_cameras = fit_cameras
            self.densify_shadow_gate_camera_names = {
                str(camera.image_name) for camera in self.densify_shadow_gate_cameras
            }
            self.logger.info(
                "Densify shadow gate: stride=%d fit=%d held_out=%d names=%s",
                shadow_gate_stride,
                len(fit_cameras),
                len(self.densify_shadow_gate_cameras),
                sorted(self.densify_shadow_gate_camera_names),
            )
        self.densify_shadow_proposal_cameras = list(
            self.densify_shadow_fit_cameras
        )
        self.densify_shadow_proposal_camera_names = set(
            self.densify_shadow_fit_camera_names
        )
        selector_stride = int(
            getattr(opt, "densify_shadow_salvage_selector_stride", 0)
        )
        if selector_stride > 0:
            (
                self.densify_shadow_proposal_cameras,
                self.densify_shadow_selector_cameras,
            ) = _split_resume_train_cameras(
                self.densify_shadow_fit_cameras,
                holdout_stride=selector_stride,
            )
            self.densify_shadow_proposal_camera_names = {
                str(camera.image_name)
                for camera in self.densify_shadow_proposal_cameras
            }
            self.densify_shadow_selector_camera_names = {
                str(camera.image_name)
                for camera in self.densify_shadow_selector_cameras
            }
            self.logger.info(
                "Densify shadow selector: stride=%d proposal=%d selector=%d names=%s",
                selector_stride,
                len(self.densify_shadow_proposal_cameras),
                len(self.densify_shadow_selector_cameras),
                sorted(self.densify_shadow_selector_camera_names),
            )
        self.camera_sampling = str(getattr(opt, "camera_sampling", "random"))
        self._camera_stack: list[object] = []
        self.camera_sampling_weights_path: Path | None = None
        self.camera_sampling_weights_sha256: str | None = None
        self._camera_sampling_weights: list[float] | None = None
        if self.camera_sampling == "weighted":
            weights_path = Path(str(opt.camera_sampling_weights_path)).resolve()
            camera_names = [str(camera.image_name) for camera in self.train_cameras]
            (
                self._camera_sampling_weights,
                self.camera_sampling_weights_sha256,
            ) = load_camera_sampling_weights(weights_path, camera_names)
            self.camera_sampling_weights_path = weights_path
        self.scene_extent = float(scene.cameras_extent)
        self.spatial_lr_scale = float(scene.cameras_extent) if bool(getattr(opt, "spatial_lr_scale", True)) else 1.0
        self.start_iteration = 0
        self.resume_checkpoint_path: Optional[Path] = None
        self.resume_checkpoint_sha256: Optional[str] = None
        self.resume_checkpoint_lineage: dict[str, object] | None = None
        self.resume_checkpoint_schema_version: object | None = None
        self.resume_checkpoint_type: object | None = None
        self.resume_checkpoint_edge_transfer: dict[str, object] | None = None
        self.resume_geometry_lr_scale = (
            float(getattr(opt, "resume_geometry_lr_scale", 1.0))
            if getattr(opt, "resume_checkpoint", None) is not None
            else 1.0
        )
        self.resume_appearance_lr_scale = (
            float(getattr(opt, "resume_appearance_lr_scale", 1.0))
            if getattr(opt, "resume_checkpoint", None) is not None
            else 1.0
        )
        self.lr_decay_milestones = frozenset(
            int(milestone) for milestone in getattr(opt, "lr_decay_milestones", [])
        )
        self.lr_decay_factor = float(getattr(opt, "lr_decay_factor", 0.5))
        self.position_lr_resume_scale = 1.0
        self.geometry_start_iteration = int(getattr(opt, "geometry_start_iteration", 0))
        self.init_geometry_metadata: dict[str, object] | None = None

        self.position_lr_scheduler = None
        lr_means3d_final = float(getattr(opt, "lr_means3d_final", 0.0))
        if lr_means3d_final > 0.0:
            lr_init = float(opt.lr_means3d) * self.spatial_lr_scale
            lr_final = lr_means3d_final * self.spatial_lr_scale
            max_steps = int(getattr(opt, "lr_means3d_max_steps", opt.iterations))
            lr_delay_steps = int(getattr(opt, "lr_means3d_delay_steps", 0))
            lr_delay_mult = float(getattr(opt, "lr_means3d_delay_mult", 1.0))
            self.position_lr_scheduler = _get_expon_lr_func(
                lr_init,
                lr_final,
                lr_delay_steps=lr_delay_steps,
                lr_delay_mult=lr_delay_mult,
                max_steps=max_steps,
            )

        init_geometry_path = getattr(opt, "init_geometry_path", None)
        if init_geometry_path is not None:
            initial_geometry, self.init_geometry_metadata = load_initial_geometry(
                init_geometry_path,
                self.device,
            )
            points = initial_geometry["means3d"]
            scales = initial_geometry["scales"]
            rotations = initial_geometry["rotations"]
            colors = initial_geometry["colors"]
            P = points.shape[0]
            self.logger.info(
                "Init: geometry_pack points=%d sha256=%s",
                int(P),
                self.init_geometry_metadata["sha256"],
            )
        else:
            points = torch.from_numpy(scene.point_cloud.points).to(self.device, dtype=torch.float32)
            colors = torch.from_numpy(scene.point_cloud.colors).to(self.device, dtype=torch.float32)
            total_points = int(points.shape[0])
            if opt.max_points > 0 and points.shape[0] > opt.max_points:
                perm = torch.randperm(points.shape[0], device=self.device)[: opt.max_points]
                points = points[perm]
                colors = colors[perm]

            P = points.shape[0]
            if opt.init_scale_mode == "knn":
                max_scale = float(opt.init_knn_max_scale_frac) * float(scene.cameras_extent)
                scales = _knn_initial_scales(
                    points,
                    k=int(opt.init_knn_k),
                    multiplier=float(opt.init_knn_scale_multiplier),
                    max_scale=max_scale,
                )
                quantiles = torch.quantile(scales[:, 0], torch.tensor([0.05, 0.5, 0.95], device=self.device))
                self.logger.info(
                    "Init: points=%d/%d scale_mode=knn k=%d multiplier=%.4g cap=%.6g scale_p05/50/95=%.6g/%.6g/%.6g",
                    int(P),
                    total_points,
                    int(opt.init_knn_k),
                    float(opt.init_knn_scale_multiplier),
                    max_scale,
                    float(quantiles[0].item()),
                    float(quantiles[1].item()),
                    float(quantiles[2].item()),
                )
            else:
                init_scale = float(opt.init_scale_frac) * float(scene.cameras_extent)
                if opt.resolution in (1, 2, 4, 8):
                    init_scale *= float(opt.resolution)
                scales = torch.full((P, 2), init_scale, device=self.device, dtype=torch.float32)
                self.logger.info(
                    "Init: points=%d/%d scale_mode=uniform init_scale=%.6g (frac=%.6g extent=%.6g res=%s)",
                    int(P),
                    total_points,
                    init_scale,
                    float(opt.init_scale_frac),
                    float(scene.cameras_extent),
                    str(opt.resolution),
                )
            rotations = torch.zeros((P, 4), device=self.device, dtype=torch.float32)
            rotations[:, 0] = 1.0  # identity quaternion (w, x, y, z)
            if opt.init_normal_mode == "pca_confident":
                from .geometry_init import confidence_gated_pca_rotations

                rotations, accepted_fraction = confidence_gated_pca_rotations(points)
                self.logger.info("Confidence-gated PCA frames: accepted_fraction=%.4f", accepted_fraction)
        self.logger.info(
            "Init appearance: opacity=%.3g softness=%.3g scale_modifier=%.3g",
            float(opt.init_opacity),
            float(opt.softness),
            float(opt.scale_modifier),
        )
        if self.position_lr_scheduler is not None:
            lr_init = float(opt.lr_means3d) * self.spatial_lr_scale
            lr_final = float(getattr(opt, "lr_means3d_final", 0.0)) * self.spatial_lr_scale
            self.logger.info(
                "Position LR schedule: init=%.6g final=%.6g (spatial_lr_scale=%.4g) delay_steps=%d delay_mult=%.4g max_steps=%d",
                lr_init, lr_final, self.spatial_lr_scale,
                int(getattr(opt, "lr_means3d_delay_steps", 0)),
                float(getattr(opt, "lr_means3d_delay_mult", 1.0)),
                int(getattr(opt, "lr_means3d_max_steps", opt.iterations)),
            )
        else:
            effective_lr = float(opt.lr_means3d) * self.spatial_lr_scale
            self.logger.info(
                "Position LR: flat=%.6g (spatial_lr_scale=%.4g)",
                effective_lr, self.spatial_lr_scale,
            )
        view_model = str(getattr(opt, "view_model", "linear"))
        geometry = Hexels3DGeometry(means3d=points, scales=scales, rotations=rotations)
        self.model = Hexels3DModel(
            geometry=geometry,
            colors_init=colors,
            init_opacity=opt.init_opacity,
            init_softness=opt.softness,
            softness_mode=str(opt.softness_mode),
            view_basis_dim=15 if view_model == "sh3" else 3,
        ).to(self.device)
        self.renderer = Hexels3DRenderer(opt)

        self.bg_color = torch.tensor(
            [1.0, 1.0, 1.0] if opt.white_background else [0.0, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        )

        optimize_geometry = opt.train_mode in {"geom", "full"}
        optimize_softness = opt.train_mode == "full"
        self.optimize_geometry = optimize_geometry
        self.optimize_softness = optimize_softness
        optimize_color_grads = str(getattr(opt, "color_model", "linear")) != "flat"
        optimize_color_quads = color_model_uses_quadratic_coefficients(
            str(getattr(opt, "color_model", "linear"))
        )
        optimize_view_grads = view_model in {"linear", "sh3"} and bool(getattr(opt, "use_view_grads", True))
        optimize_specular = view_model == "lobe"
        self.optimize_color_grads = optimize_color_grads
        self.optimize_color_quads = optimize_color_quads
        self.optimize_view_grads = optimize_view_grads
        self.optimize_specular = optimize_specular
        self.view_model = view_model
        self.sh_progressive_interval = int(getattr(opt, "sh_progressive_interval", 0))
        self.active_sh_degree: Optional[int] = (
            None if view_model == "sh3" and self.sh_progressive_interval > 0 else 3
        )
        self.logger.info(
            "Train mode: %s (geometry=%s softness=%s)",
            opt.train_mode,
            str(optimize_geometry),
            str(optimize_softness),
        )
        self.logger.info(
            "Appearance: color_model=%s view_model=%s view_grads=%s",
            str(getattr(opt, "color_model", "linear")),
            view_model,
            str(optimize_view_grads),
        )

        self.model.means3d.requires_grad_(optimize_geometry)
        self.model.log_scales.requires_grad_(optimize_geometry)
        self.model.rotation_quats.requires_grad_(optimize_geometry)
        self.model.color_grads.requires_grad_(optimize_color_grads)
        self.model.color_quads.requires_grad_(optimize_color_quads)
        self.model.view_grads.requires_grad_(optimize_view_grads)
        self.model.specular_ks.requires_grad_(optimize_specular)
        self.model.specular_log_s.requires_grad_(optimize_specular)
        self.model.child_color_grads.requires_grad_(optimize_color_grads)
        self.model.child_color_quads.requires_grad_(optimize_color_quads)
        self.model.child_view_grads.requires_grad_(optimize_view_grads)
        self.model.child_specular_ks.requires_grad_(optimize_specular)
        self.model.child_specular_log_s.requires_grad_(optimize_specular)
        self.model.child_rotvec.requires_grad_(bool(optimize_geometry and opt.enable_children and opt.child_rotation_delta))
        self.model.log_softness.requires_grad_(optimize_softness)
        self.model.child_log_softness.requires_grad_(optimize_softness)
        self.optimizer = self._build_optimizer()

        self.mcmc_binoms = None
        self.contrib_ema = None
        self.mcmc_mode = str(getattr(opt, "mcmc_mode", "contrib"))
        if opt.enable_mcmc:
            self.mcmc_binoms = init_relocation_buffers(self.device, torch.float32)
            if self.mcmc_mode == "contrib":
                self.contrib_ema = torch.zeros((P,), device=self.device, dtype=torch.float32)
            grow = int(opt.mcmc_cap_max) > 0
            self.logger.info(
                "MCMC enabled: mode=%s interval=%d dead_threshold=%.4g cap_max=%d grow_rate=%.4g ema_decay=%.3g noise_lr=%.3g spawn_jitter=%.3g normal_noise=%.3g window=[%d,%s] (growth=%s)",
                str(self.mcmc_mode),
                int(opt.mcmc_interval),
                float(opt.mcmc_dead_threshold),
                int(opt.mcmc_cap_max),
                float(opt.mcmc_grow_rate),
                float(opt.mcmc_ema_decay),
                float(getattr(opt, "mcmc_noise_lr", 0.0)),
                float(getattr(opt, "mcmc_spawn_jitter", 0.0)),
                float(getattr(opt, "mcmc_normal_noise_frac", 0.0)),
                int(getattr(opt, "mcmc_from_iter", 0)),
                "end" if int(getattr(opt, "mcmc_until_iter", -1)) == -1 else str(int(getattr(opt, "mcmc_until_iter", -1))),
                "on" if grow else "off",
            )

        self.enable_densify = bool(getattr(opt, "enable_densify", False))
        self.densify_grad_accum: Optional[torch.Tensor] = None
        self.densify_abs_grad_accum: Optional[torch.Tensor] = None
        self.densify_score_audit: list[dict[str, object]] = []
        self.densify_denom: Optional[torch.Tensor] = None
        self.densify_max_radii2d: Optional[torch.Tensor] = None
        self.densify_local_grad_abs_accum: Optional[torch.Tensor] = None
        self.densify_local_grad_signed_accum: Optional[torch.Tensor] = None
        self.densify_residual_mass_accum: Optional[torch.Tensor] = None
        self.densify_centroid_mass_accum: Optional[torch.Tensor] = None
        self.densify_direction_vote_accum: Optional[torch.Tensor] = None
        self.densify_moment_vote_count: Optional[torch.Tensor] = None
        self.densify_mv_edge_top_scores: Optional[torch.Tensor] = None
        self.densify_mv_edge_top_centroids: Optional[torch.Tensor] = None
        self.densify_mv_edge_vote_counts: Optional[torch.Tensor] = None
        self.densify_attributed_edge_mass: Optional[torch.Tensor] = None
        self.densify_attributed_edge_centroid_mass: Optional[torch.Tensor] = None
        self.densify_attributed_edge_vote_counts: Optional[torch.Tensor] = None
        self.densify_covariance_coverage: Optional[torch.Tensor] = None
        self.densify_covariance_energy: Optional[torch.Tensor] = None
        self.densify_covariance_robust_weight: Optional[torch.Tensor] = None
        self.densify_covariance_first_moment: Optional[torch.Tensor] = None
        self.densify_covariance_second_moment: Optional[torch.Tensor] = None
        self.densify_covariance_double_angle: Optional[torch.Tensor] = None
        self.densify_covariance_side_mass: Optional[torch.Tensor] = None
        self.densify_covariance_material_view_count: Optional[torch.Tensor] = None
        self.densify_covariance_axis_view_count: Optional[torch.Tensor] = None
        self.densify_moment_camera_names: set[str] = set()
        self.densify_contrib_accum: Optional[torch.Tensor] = None
        self.densify_contrib_camera_names: set[str] = set()
        self.densify_birth_iteration: Optional[torch.Tensor] = None
        self.densify_has_spawned_child: Optional[torch.Tensor] = None
        self.densify_shadow_child_mask: Optional[torch.Tensor] = None
        self.densify_shadow_attempted_parent_mask: Optional[torch.Tensor] = None
        self.densify_shadow_event: Optional[_DensifyShadowEvent] = None
        self.densify_joint_coplanar_event: Optional[_DensifyJointCoplanarEvent] = None
        self.densify_joint_coplanar_last_completed: Optional[dict[str, object]] = None
        self.densify_shadow_frozen_step_ids: Optional[torch.Tensor] = None
        self.densify_shadow_counts = {
            "proposed_events": 0,
            "accepted_events": 0,
            "rejected_events": 0,
            "proposed_children": 0,
            "accepted_children": 0,
            "rejected_children": 0,
        }
        self.densify_shadow_last_decision: Optional[dict[str, object]] = None
        self.densify_shadow_decisions: list[dict[str, object]] = []
        self.densify_merge_decisions: list[dict[str, object]] = []
        self.densify_prune_audit: list[dict[str, object]] = []
        self.densify_peak_population = int(P)
        if self.enable_densify:
            densify_cap_max = int(getattr(opt, "densify_cap_max", 0))
            self.densify_grad_accum = torch.zeros((P, 1), device=self.device, dtype=torch.float32)
            if bool(getattr(opt, "densify_absgrad_rank", False)):
                self.densify_abs_grad_accum = torch.zeros_like(self.densify_grad_accum)
            self.densify_denom = torch.zeros((P, 1), device=self.device, dtype=torch.float32)
            self.densify_max_radii2d = torch.zeros((P,), device=self.device, dtype=torch.float32)
            if str(getattr(opt, "densify_action", "mixed")) in {"bisplit", "edge_child"}:
                direction_signal = str(
                    getattr(opt, "densify_direction_signal", "position")
                )
                edge_child = str(getattr(opt, "densify_action", "mixed")) == "edge_child"
                if not edge_child or direction_signal == "position":
                    self.densify_local_grad_abs_accum = torch.zeros((P, 2), device=self.device)
                    self.densify_local_grad_signed_accum = torch.zeros((P, 2), device=self.device)
                elif direction_signal == "residual_moment":
                    self.densify_residual_mass_accum = torch.zeros((P,), device=self.device)
                    self.densify_centroid_mass_accum = torch.zeros((P, 2), device=self.device)
                    self.densify_direction_vote_accum = torch.zeros((P, 2), device=self.device)
                    self.densify_moment_vote_count = torch.zeros((P,), device=self.device)
                    residual_topk = int(getattr(opt, "densify_residual_topk_views", 0))
                    if residual_topk > 0:
                        self.densify_mv_edge_top_scores = torch.zeros(
                            (P, 6, residual_topk),
                            device=self.device,
                            dtype=torch.float64,
                        )
                        self.densify_mv_edge_top_centroids = torch.zeros(
                            (P, 6, residual_topk, 2),
                            device=self.device,
                            dtype=torch.float64,
                        )
                        self.densify_mv_edge_vote_counts = torch.zeros(
                            (P, 6),
                            device=self.device,
                            dtype=torch.int32,
                        )
                elif direction_signal == "attributed_residual":
                    self.densify_attributed_edge_mass = torch.zeros(
                        (P, 6), device=self.device, dtype=torch.float64
                    )
                    self.densify_attributed_edge_centroid_mass = torch.zeros(
                        (P, 6, 2), device=self.device, dtype=torch.float64
                    )
                    self.densify_attributed_edge_vote_counts = torch.zeros(
                        (P, 6), device=self.device, dtype=torch.int32
                    )
                elif direction_signal == "attributed_covariance":
                    self.densify_covariance_coverage = torch.zeros(
                        (P,), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_energy = torch.zeros(
                        (P,), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_robust_weight = torch.zeros(
                        (P,), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_first_moment = torch.zeros(
                        (P, 2), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_second_moment = torch.zeros(
                        (P, 2, 2), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_double_angle = torch.zeros(
                        (P, 2), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_side_mass = torch.zeros(
                        (P, 2), device=self.device, dtype=torch.float64
                    )
                    self.densify_covariance_material_view_count = torch.zeros(
                        (P,), device=self.device, dtype=torch.int32
                    )
                    self.densify_covariance_axis_view_count = torch.zeros(
                        (P,), device=self.device, dtype=torch.int32
                    )
                else:
                    raise ValueError(f"Unknown densify direction signal: {direction_signal}")
                self.densify_contrib_accum = torch.zeros((P,), device=self.device)
                self.densify_birth_iteration = torch.zeros((P,), device=self.device, dtype=torch.long)
                if str(getattr(opt, "densify_action", "mixed")) == "edge_child":
                    self.densify_has_spawned_child = torch.zeros((P,), device=self.device, dtype=torch.bool)
                    if int(getattr(opt, "densify_shadow_steps", 0)) > 0:
                        self.densify_shadow_child_mask = torch.zeros(
                            (P,), device=self.device, dtype=torch.bool
                        )
                        self.densify_shadow_attempted_parent_mask = torch.zeros(
                            (P,), device=self.device, dtype=torch.bool
                        )
            if bool(getattr(opt, "densify_coordinated_prune", False)) and self.densify_birth_iteration is None:
                self.densify_birth_iteration = torch.zeros((P,), device=self.device, dtype=torch.long)
            self.logger.info(
                "Densify enabled: impl=%s action=%s interval=%d from=%d until=%d grad_thr=%.4g max_add=%d cap_max=%d min_denom=%d clone_jit=%.3g clone_op=%.3g split_jit=%.3g op_split=%s opacity_cull=%.3g reset_int=%d percent_dense=%.3g max_screen=%d",
                str(getattr(opt, "densify_impl", "2dgs")),
                str(getattr(opt, "densify_action", "mixed")),
                int(getattr(opt, "densification_interval", 100)),
                int(getattr(opt, "densify_from_iter", 500)),
                int(getattr(opt, "densify_until_iter", 15_000)),
                float(getattr(opt, "densify_grad_threshold", 0.0002)),
                int(getattr(opt, "densify_max_add", 0)),
                densify_cap_max,
                int(getattr(opt, "densify_min_denom", 0)),
                float(getattr(opt, "densify_clone_jitter", 0.0)),
                float(getattr(opt, "densify_clone_init_opacity", 0.0)),
                float(getattr(opt, "densify_split_jitter", 1.0)),
                str(bool(getattr(opt, "densify_opacity_split", True))),
                float(getattr(opt, "opacity_cull", 0.05)),
                int(getattr(opt, "opacity_reset_interval", 3000)),
                float(getattr(opt, "percent_dense", 0.01)),
                int(getattr(opt, "densify_max_screen_size", 20)),
            )
            if self.densify_birth_iteration is not None:
                self.logger.info(
                    "Directional density: split_offset=%.3g split_ratio=%.3g child_offset=%.3g child_ratio=%.3g child_softness_ratio=%.3g child_reorient=%s child_opacity=%.3g child_extent_cap=%.3g child_parent_opacity=%.3g consistency=%.3g signal=%s moment_interval=%d moment_radius=%.3g repeat_parent=%s coverage_weight=%.3g newborn_grace=%d prune_interval=%d prune_full_sweep=%s prune_contrib=%.3g",
                    float(getattr(opt, "densify_split_offset_fraction", 0.25)),
                    float(getattr(opt, "densify_split_scale_ratio", 0.75)),
                    float(getattr(opt, "densify_child_offset_fraction", 0.75)),
                    float(getattr(opt, "densify_child_scale_ratio", 0.5)),
                    float(getattr(opt, "densify_child_softness_ratio", 1.0)),
                    str(bool(getattr(opt, "densify_child_reorient", False))),
                    float(getattr(opt, "densify_child_init_opacity", 0.01)),
                    float(getattr(opt, "densify_child_extent_cap_fraction", 0.0)),
                    float(getattr(opt, "densify_child_parent_opacity_fraction", 0.0)),
                    float(getattr(opt, "densify_direction_consistency", 0.25)),
                    str(getattr(opt, "densify_direction_signal", "position")),
                    int(getattr(opt, "densify_moment_interval", 10)),
                    float(getattr(opt, "densify_moment_min_radius", 0.25)),
                    str(bool(getattr(opt, "densify_repeat_parent", False))),
                    float(getattr(opt, "densify_coverage_weight", 0.0)),
                    int(getattr(opt, "densify_newborn_grace", 500)),
                    int(getattr(opt, "densify_prune_interval", 1000)),
                    str(bool(getattr(opt, "densify_prune_full_sweep", False))),
                    float(getattr(opt, "densify_prune_contrib_threshold", 0.0)),
                )
            if self.densify_shadow_child_mask is not None:
                salvage_groups = int(getattr(opt, "densify_shadow_salvage_groups", 0))
                if salvage_groups > 0:
                    self.logger.info(
                        "Shadow density salvage: groups=%d selector=bidirectional_group_prefix_v1 order=%s",
                        salvage_groups,
                        str(getattr(opt, "densify_shadow_salvage_order", "score")),
                    )
                    prefix_policy = str(
                        getattr(opt, "densify_shadow_salvage_prefix_policy", "best_pooled")
                    )
                    if prefix_policy != "best_pooled":
                        self.logger.info(
                            "Shadow density salvage prefix policy: %s",
                            prefix_policy,
                        )
                    selector_stride = int(
                        getattr(opt, "densify_shadow_salvage_selector_stride", 0)
                    )
                    if selector_stride > 0:
                        self.logger.info(
                            "Shadow density nested selector: stride=%d defer_inner_gate=%s",
                            selector_stride,
                            str(bool(getattr(opt, "densify_shadow_defer_inner_gate", False))),
                        )
                    if bool(
                        getattr(
                            opt,
                            "densify_shadow_salvage_allow_partial_cohort",
                            False,
                        )
                    ):
                        self.logger.info("Shadow density salvage partial cohorts: enabled")
                self.logger.info(
                    "Shadow density: steps=%d permanent_residual=%s gate_stride=%d parent_view_anchor=%s pooled=%.4g mean=%.4g wins=%.4g worst=%.4g append_only=true",
                    int(getattr(opt, "densify_shadow_steps", 0)),
                    str(bool(getattr(opt, "densify_shadow_permanent_residual", True))),
                    int(getattr(opt, "densify_shadow_gate_stride", 0)),
                    str(bool(getattr(opt, "densify_shadow_parent_view_anchor", False))),
                    float(getattr(opt, "densify_shadow_min_pooled_sse_reduction", 0.005)),
                    float(getattr(opt, "densify_shadow_min_mean_psnr_delta", 0.02)),
                    float(getattr(opt, "densify_shadow_min_win_fraction", 0.6)),
                    float(getattr(opt, "densify_shadow_min_worst_psnr_delta", -0.15)),
                )
                if bool(getattr(opt, "densify_shadow_enforce_birth_envelope", False)):
                    self.logger.info(
                        "Shadow birth envelope: extent_cap=%.4g parent_scale_ratio=%.4g parent_opacity_fraction=%.4g",
                        float(getattr(opt, "densify_child_extent_cap_fraction", 0.0)),
                        float(getattr(opt, "densify_child_scale_ratio", 0.5)),
                        float(getattr(opt, "densify_child_parent_opacity_fraction", 0.0)),
                    )

        if opt.enable_children:
            self.logger.info(
                "Children enabled: interval=%d parents_per_interval=%d offset_frac=%.4g scale_ratio=%.4g cap_max=%d max_layers=%d sides=%s detach_geometry=%s rot_delta=%s rot_delta_w=%.4g",
                int(opt.child_interval),
                int(opt.child_parents_per_interval),
                float(opt.child_offset_frac),
                float(opt.child_scale_ratio),
                int(opt.child_cap_max),
                int(getattr(opt, "child_max_layers_per_parent", 1)),
                str(getattr(opt, "child_sides", "both")),
                str(bool(getattr(opt, "children_detach_geometry", True))),
                str(bool(getattr(opt, "child_rotation_delta", False))),
                float(getattr(opt, "child_rotation_delta_weight", 0.0)),
            )

        self.repulsion_knn_idx: Optional[torch.Tensor] = None
        self.repulsion_knn_P: int = 0
        self.surface_knn_idx: Optional[torch.Tensor] = None
        self.surface_knn_P: int = 0

        self.wandb_run = None
        if opt.wandb:
            import wandb

            self.wandb_run = wandb.init(
                project=opt.wandb_project,
                entity=opt.wandb_entity,
                tags=opt.wandb_tags,
                name=self.model_path.name,
                config=opt_dict,
                dir=str(self.model_path),
            )

        from utils.loss_utils import l1_loss, ssim

        self.l1_loss = l1_loss
        self.ssim = ssim
        self.metric_ssim = ssim
        if opt.use_fused_ssim:
            from fused_ssim import fused_ssim

            self.ssim = lambda prediction, target: fused_ssim(
                prediction.unsqueeze(0), target.unsqueeze(0), padding="same"
            )
        if opt.resident_training_targets:
            for camera in self.train_cameras:
                camera.original_image = camera.original_image.to(self.device)
            self.logger.info("Training targets resident on %s", self.device)
        self.test_iterations = {int(i) for i in getattr(opt, "test_iterations", []) if int(i) > 0}
        self.test_train_views = int(getattr(opt, "test_train_views", 10))
        self.wandb_test_images = int(getattr(opt, "wandb_test_images", 0))
        resume_checkpoint = getattr(opt, "resume_checkpoint", None)
        if resume_checkpoint is not None:
            self._load_resume_checkpoint(Path(resume_checkpoint))
            self._configure_resume_scale_constraint()
            self._configure_resume_normal_trust()
            self._configure_resume_surface_anchor()
            self._configure_resume_mvs_surface()
        if any(milestone <= self.start_iteration for milestone in self.lr_decay_milestones):
            raise ValueError("Learning-rate decay milestones must follow the resume iteration")
        if self.enable_densify:
            densify_cap_max = int(getattr(opt, "densify_cap_max", 0))
            if densify_cap_max > 0 and self.model.num_base + self.model.num_children > densify_cap_max:
                raise ValueError(
                    f"Initial active primitive count {self.model.num_active_base} exceeds "
                    f"--densify_cap_max {densify_cap_max}"
                )
        if self.densify_shadow_child_mask is not None and self.model.num_active_base != self.model.num_base:
            raise ValueError("Shadow densification requires every allocated base row to be active")
        self.edge_transfer_controller: EdgeTransferController | None = None
        edge_transfer_manifest = getattr(opt, "edge_transfer_manifest", None)
        if edge_transfer_manifest is not None:
            if self.resume_checkpoint_sha256 is None:
                raise ValueError("Edge transfer requires a loaded resume checkpoint")
            self.edge_transfer_controller = EdgeTransferController.from_file(
                self.model,
                self.optimizer,
                edge_transfer_manifest,
                checkpoint_sha256=self.resume_checkpoint_sha256,
                start_iteration=self.start_iteration,
                target_iteration=int(opt.iterations),
                scale_modifier=float(opt.scale_modifier),
            )
            self.logger.info(
                "Edge transfer: parents=%d appearance_steps=%d max_retraction=%.4g lr=%.4g",
                int(self.edge_transfer_controller.parent_ids.numel()),
                self.edge_transfer_controller.appearance_steps,
                self.edge_transfer_controller.max_retraction,
                self.edge_transfer_controller.retraction_lr,
            )
        self._replay_camera_sampler(self.start_iteration)

    def _setup_logging(self) -> None:
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
        fh = logging.FileHandler(self.model_path / "log.txt")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def _optimizer_group_specs(self) -> list[tuple[str, float]]:
        specs: list[tuple[str, float]] = [
            ("colors", self.opt.lr_colors),
            ("opacity_logits", self.opt.lr_opacity),
        ]
        if self.optimize_color_grads:
            specs.append(("color_grads", self.opt.lr_color_grads))
        if self.optimize_color_quads:
            quad_lr = getattr(self.opt, "lr_color_quads", None)
            specs.append(
                ("color_quads", self.opt.lr_color_grads if quad_lr is None else quad_lr)
            )
        if self.optimize_view_grads:
            view_lr = self.opt.lr_sh_coeffs if self.view_model == "sh3" else self.opt.lr_view_grads
            specs.append(("view_grads", view_lr))
        if bool(getattr(self, "optimize_specular", False)):
            specs.append(("specular_ks", float(getattr(self.opt, "lr_specular_ks", self.opt.lr_view_grads))))
            specs.append(("specular_log_s", float(getattr(self.opt, "lr_specular_s", self.opt.lr_view_grads))))
        if self.opt.enable_children:
            specs.append(("child_colors", self.opt.lr_colors))
            specs.append(("child_opacity_logits", self.opt.lr_opacity))
            if bool(getattr(self.opt, "child_rotation_delta", False)):
                specs.append(("child_rotvec", self.opt.lr_rotations))
            if self.optimize_color_grads:
                specs.append(("child_color_grads", self.opt.lr_color_grads))
            if self.optimize_color_quads:
                specs.append(
                    (
                        "child_color_quads",
                        self.opt.lr_color_grads if quad_lr is None else quad_lr,
                    )
                )
            if self.optimize_view_grads:
                specs.append(("child_view_grads", view_lr))
            if bool(getattr(self, "optimize_specular", False)):
                specs.append(("child_specular_ks", float(getattr(self.opt, "lr_specular_ks", self.opt.lr_view_grads))))
                specs.append(("child_specular_log_s", float(getattr(self.opt, "lr_specular_s", self.opt.lr_view_grads))))
        if self.optimize_geometry:
            means3d_lr = self.opt.lr_means3d * getattr(self, "spatial_lr_scale", 1.0)
            specs = [
                ("means3d", means3d_lr),
                ("log_scales", self.opt.lr_scales),
                ("rotation_quats", self.opt.lr_rotations),
            ] + specs
        if self.optimize_softness:
            specs.append(("log_softness", self.opt.lr_softness))
            if self.opt.enable_children:
                specs.append(("child_log_softness", self.opt.lr_softness))
        return specs

    def _base_optimizer_param_names(self) -> tuple[str, ...]:
        names_list: list[str] = ["colors", "opacity_logits"]
        if self.optimize_color_grads:
            names_list.append("color_grads")
        if self.optimize_color_quads:
            names_list.append("color_quads")
        if self.optimize_view_grads:
            names_list.append("view_grads")
        if bool(getattr(self, "optimize_specular", False)):
            names_list.extend(["specular_ks", "specular_log_s"])
        names = tuple(names_list)
        if self.optimize_geometry:
            names = ("means3d", "log_scales", "rotation_quats") + names
        if self.optimize_softness:
            names = names + ("log_softness",)
        return names

    def _build_optimizer(self) -> torch.optim.Adam:
        groups = []
        for name, lr in self._optimizer_group_specs():
            groups.append({"name": name, "params": [getattr(self.model, name)], "lr": float(lr)})
        return torch.optim.Adam(groups)

    def _optimizer_lr_map(self) -> dict[str, float]:
        return {str(group["name"]): float(group["lr"]) for group in self.optimizer.param_groups}

    def _freeze_geometry_gradients(self, iteration: int) -> None:
        if iteration < int(getattr(getattr(self, "opt", None), "spatial_appearance_start", 0)):
            self.model.color_grads.grad = None
        if iteration < int(getattr(self, "geometry_start_iteration", 0)):
            for name in ("means3d", "log_scales", "rotation_quats"):
                getattr(self.model, name).grad = None
            return
        policy = str(getattr(self, "resume_geometry_policy", "inherit"))
        if policy == "inherit":
            return
        self.model.means3d.grad = None
        self.model.rotation_quats.grad = None
        if policy == "frozen":
            self.model.log_scales.grad = None
            return
        if policy != "bounded_isotropic_shrink":
            raise ValueError(f"Unknown resume geometry policy: {policy}")
        gradient = self.model.log_scales.grad
        if gradient is not None:
            shared = gradient.sum(dim=1, keepdim=True)
            gradient.copy_(shared.expand_as(gradient))

    def _resume_trainable_group_names(self) -> set[str]:
        policy = str(getattr(self, "resume_parameter_policy", "inherit"))
        if policy == "inherit":
            return {
                str(group["name"]) for group in self.optimizer.param_groups
            }
        if policy not in {"appearance_only", "view_only"}:
            raise ValueError(f"Unknown resume parameter policy: {policy}")
        if policy == "appearance_only":
            return {
                str(group["name"])
                for group in self.optimizer.param_groups
                if str(group["name"])
                not in {"means3d", "log_scales", "rotation_quats"}
            }
        if policy == "view_only" and "view_grads" not in {
            str(group["name"]) for group in self.optimizer.param_groups
        }:
            raise ValueError("view_only requires a view_grads optimizer group")
        return {"view_grads"}

    def _freeze_resume_parameter_gradients(self) -> None:
        allowed = self._resume_trainable_group_names()
        for group in self.optimizer.param_groups:
            if str(group["name"]) not in allowed:
                group["params"][0].grad = None

    def _configure_resume_scale_constraint(self) -> None:
        if self.resume_geometry_policy != "bounded_isotropic_shrink":
            return
        lower_path = self.resume_scale_lower_geometry_path
        upper_path = self.resume_scale_upper_geometry_path
        if lower_path is None or upper_path is None:
            raise ValueError("bounded_isotropic_shrink requires lower and upper geometry packs")
        lower_geometry, lower_metadata = load_initial_geometry(lower_path, self.device)
        upper_geometry, upper_metadata = load_initial_geometry(upper_path, self.device)
        population = int(self.model.num_base)
        checkpoint_rotations = torch.nn.functional.normalize(
            self.model.rotation_quats.detach(), dim=1
        )
        for name, geometry in (("lower", lower_geometry), ("upper", upper_geometry)):
            if geometry["scales"].shape != (population, 2):
                raise ValueError(
                    f"Resume scale {name} geometry population differs from checkpoint"
                )
            if not torch.equal(geometry["means3d"], self.model.means3d.detach()):
                raise ValueError(f"Resume scale {name} geometry means differ from checkpoint")
            rotations = torch.nn.functional.normalize(geometry["rotations"], dim=1)
            if not torch.allclose(
                rotations, checkpoint_rotations, rtol=1e-6, atol=1e-7
            ):
                raise ValueError(f"Resume scale {name} geometry rotations differ from checkpoint")

        lower_log = torch.log(lower_geometry["scales"])
        upper_log = torch.log(upper_geometry["scales"])
        lower_isotropy = (lower_log[:, 0] - lower_log[:, 1]).abs().max()
        upper_isotropy = (upper_log[:, 0] - upper_log[:, 1]).abs().max()
        if float(lower_isotropy.item()) > 1e-6 or float(upper_isotropy.item()) > 1e-6:
            raise ValueError("Bounded resume scale geometry must be isotropic")
        lower_shared = lower_log.mean(dim=1, keepdim=True)
        upper_shared = upper_log.mean(dim=1, keepdim=True)
        if bool((lower_shared > upper_shared).any().item()):
            raise ValueError("Resume scale lower bound exceeds checkpoint scale")
        flexible = (upper_shared - lower_shared) > 1e-7
        if not bool(flexible.any().item()):
            raise ValueError("Resume scale bounds contain no flexible rows")
        checkpoint_shared = self.model.log_scales.detach().mean(dim=1, keepdim=True)
        checkpoint_isotropy = (
            self.model.log_scales[:, 0] - self.model.log_scales[:, 1]
        ).abs().max()
        if float(checkpoint_isotropy.item()) > 1e-6:
            raise ValueError("Resume checkpoint scales are not isotropic")
        if bool(
            (
                (checkpoint_shared < lower_shared - 2e-6)
                | (checkpoint_shared > upper_shared + 2e-6)
            )
            .any()
            .item()
        ):
            raise ValueError("Resume checkpoint scales fall outside the configured bounds")
        if self.resume_parent_geometry_policy == "inherit" and not torch.allclose(
            checkpoint_shared, upper_shared, rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Initial bounded shrink resume must start at the upper geometry pack")
        self.resume_scale_lower_log = lower_shared
        self.resume_scale_upper_log = upper_shared
        self.resume_scale_constraint_metadata = {
            "lower_geometry": lower_metadata,
            "upper_geometry": upper_metadata,
            "lower_scales_sha256": _tensor_sha256(lower_geometry["scales"]),
            "upper_scales_sha256": _tensor_sha256(upper_geometry["scales"]),
            "anchor_weight": self.resume_scale_anchor_weight,
            "flexible_rows": int(flexible.sum().item()),
            "locked_rows": int(population - flexible.sum().item()),
        }
        previous = self.resume_checkpoint_scale_constraint
        if previous is not None:
            for name, expected in self.resume_scale_constraint_metadata.items():
                if previous.get(name) != expected:
                    raise ValueError(
                        f"Resume checkpoint scale constraint {name} differs from configured bounds"
                    )

    def _resume_scale_anchor_loss(self) -> torch.Tensor | None:
        lower = getattr(self, "resume_scale_lower_log", None)
        upper = getattr(self, "resume_scale_upper_log", None)
        if lower is None or upper is None:
            return None
        shared = self.model.log_scales.mean(dim=1, keepdim=True)
        width = upper - lower
        flexible = width > 1e-7
        shrink = (upper[flexible] - shared[flexible]) / width[flexible]
        return shrink.square().mean()

    @torch.no_grad()
    def _project_resume_isotropic_scales(self) -> None:
        lower = getattr(self, "resume_scale_lower_log", None)
        upper = getattr(self, "resume_scale_upper_log", None)
        if lower is None or upper is None:
            return
        parameter = self.model.log_scales
        shared = parameter.mean(dim=1, keepdim=True)
        projected = torch.maximum(torch.minimum(shared, upper), lower)
        projected_rows = (shared != projected).squeeze(1)
        parameter.copy_(projected.expand_as(parameter))
        if not bool(projected_rows.any().item()):
            return
        state = self.optimizer.state.get(parameter, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value[projected_rows] = 0.0

    def _configure_resume_normal_trust(self) -> None:
        if self.resume_normal_trust_path is None:
            if self.resume_checkpoint_normal_trust is not None:
                raise ValueError("Resume normal trust constraint cannot be removed")
            return
        sidecar = load_normal_trust_sidecar(
            self.resume_normal_trust_path,
            self.device,
        )
        population = int(self.model.num_base)
        if sidecar.reference_means3d.shape != (population, 3):
            raise ValueError("Resume normal trust population differs")
        reference_normals = quaternion_normal_z(sidecar.reference_rotations)
        if self.resume_checkpoint_normal_trust is None:
            current_normals = quaternion_normal_z(self.model.rotations.detach())
            if (
                not torch.equal(
                    sidecar.reference_means3d,
                    self.model.means3d.detach(),
                )
                or not torch.allclose(
                    reference_normals,
                    current_normals,
                    rtol=1e-6,
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    "Resume normal trust reference differs from checkpoint geometry"
                )
        reliability = sidecar.reliability
        cone_degrees = self.resume_normal_trust_max_degrees - reliability * (
            self.resume_normal_trust_max_degrees
            - self.resume_normal_trust_min_degrees
        )
        metadata = {
            "path": str(self.resume_normal_trust_path),
            "sidecar_sha256": sidecar.metadata["sidecar_sha256"],
            "reference_geometry_sha256": sidecar.metadata.get(
                "reference_geometry_sha256"
            ),
            "selected_indices_sha256": sidecar.metadata.get(
                "selected_indices_sha256"
            ),
            "min_degrees": self.resume_normal_trust_min_degrees,
            "max_degrees": self.resume_normal_trust_max_degrees,
        }
        if self.resume_checkpoint_normal_trust is not None:
            for name, expected in metadata.items():
                if self.resume_checkpoint_normal_trust.get(name) != expected:
                    raise ValueError(
                        f"Resume normal trust {name} differs from its checkpoint"
                    )
        self.resume_normal_trust_reference = reference_normals
        self.resume_normal_trust_reliability = reliability
        self.resume_normal_trust_cones = torch.deg2rad(cone_degrees)
        self.resume_normal_trust_metadata = metadata

    @torch.no_grad()
    def _project_resume_normal_trust(self, iteration: int) -> None:
        reference = getattr(self, "resume_normal_trust_reference", None)
        cones = getattr(self, "resume_normal_trust_cones", None)
        if (
            reference is None
            or cones is None
            or iteration < self.geometry_start_iteration
        ):
            return
        projected, rows, _, _ = project_normal_trust_cone(
            self.model.rotation_quats,
            reference,
            cones,
        )
        if not bool(rows.any().item()):
            return
        row_ids = torch.nonzero(rows, as_tuple=False).squeeze(1)
        self.model.rotation_quats.index_copy_(
            0,
            row_ids,
            projected.index_select(0, row_ids),
        )
        state = self.optimizer.state.get(self.model.rotation_quats, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(name)
            if isinstance(value, torch.Tensor) and value.shape == self.model.rotation_quats.shape:
                value[rows] = 0.0
        self.resume_normal_trust_projection_steps += 1
        self.resume_normal_trust_projected_rows += int(rows.sum().item())

    def _configure_resume_surface_anchor(self) -> None:
        if self.resume_surface_anchor_path is None:
            if self.resume_checkpoint_surface_anchor is not None:
                raise ValueError("Resume surface anchor cannot be removed")
            return
        sidecar = load_normal_trust_sidecar(
            self.resume_surface_anchor_path,
            self.device,
        )
        population = int(self.model.num_base)
        if sidecar.reference_means3d.shape != (population, 3):
            raise ValueError("Resume surface anchor population differs")
        reference_normals = quaternion_normal_z(sidecar.reference_rotations)
        if self.resume_checkpoint_surface_anchor is None:
            if (
                not torch.equal(
                    sidecar.reference_means3d,
                    self.model.means3d.detach(),
                )
                or not torch.allclose(
                    reference_normals,
                    quaternion_normal_z(self.model.rotations.detach()),
                    rtol=1e-6,
                    atol=1e-7,
                )
            ):
                raise ValueError(
                    "Resume surface anchor reference differs from checkpoint geometry"
                )
        reference_scales = self.model.log_scales.detach().mean(dim=1).exp()
        if float(sidecar.reliability.sum().item()) <= 0.0:
            raise ValueError("Resume surface anchor reliability has zero mass")
        metadata = {
            "path": str(self.resume_surface_anchor_path),
            "sidecar_sha256": sidecar.metadata["sidecar_sha256"],
            "reference_geometry_sha256": sidecar.metadata.get(
                "reference_geometry_sha256"
            ),
            "selected_indices_sha256": sidecar.metadata.get(
                "selected_indices_sha256"
            ),
            "reference_scales_sha256": _tensor_sha256(reference_scales),
            "weight": self.resume_surface_anchor_weight,
        }
        if self.resume_checkpoint_surface_anchor is not None:
            for name, expected in metadata.items():
                if self.resume_checkpoint_surface_anchor.get(name) != expected:
                    raise ValueError(
                        f"Resume surface anchor {name} differs from its checkpoint"
                    )
        self.resume_surface_anchor_reference = sidecar.reference_means3d
        self.resume_surface_anchor_normals = reference_normals
        self.resume_surface_anchor_reliability = sidecar.reliability
        self.resume_surface_anchor_scales = reference_scales.clamp_min(1e-6)
        self.resume_surface_anchor_metadata = metadata

    def _resume_surface_anchor_loss(self) -> torch.Tensor | None:
        reference = getattr(self, "resume_surface_anchor_reference", None)
        normals = getattr(self, "resume_surface_anchor_normals", None)
        reliability = getattr(self, "resume_surface_anchor_reliability", None)
        scales = getattr(self, "resume_surface_anchor_scales", None)
        if (
            reference is None
            or normals is None
            or reliability is None
            or scales is None
        ):
            return None
        displacement = (
            (self.model.means3d - reference) * normals
        ).sum(dim=-1)
        normalized = displacement / scales
        penalty = torch.nn.functional.smooth_l1_loss(
            normalized,
            torch.zeros_like(normalized),
            reduction="none",
            beta=1.0,
        )
        return (reliability * penalty).sum() / reliability.sum()

    def _configure_resume_mvs_surface(self) -> None:
        if self.resume_mvs_surface_path is None:
            if self.resume_checkpoint_mvs_surface is not None:
                raise ValueError("Resume MVS surface constraint cannot be removed")
            return
        camera_names = [str(camera.image_name) for camera in self.train_cameras]
        targets = load_mvs_surface_targets(
            self.resume_mvs_surface_path,
            camera_names=camera_names,
            validate_tensors=self.resume_mvs_surface_weight > 0.0,
        )
        metadata = {
            "path": str(self.resume_mvs_surface_path),
            "sidecar_sha256": targets.metadata["sidecar_sha256"],
            "camera_names_sha256": targets.metadata["camera_names_sha256"],
            "camera_count": len(targets.camera_names),
            "target_shape": list(targets.depth.shape),
            "minimum_source_count": targets.metadata["minimum_source_count"],
            "source_count_cap": targets.metadata["source_count_cap"],
            "maximum_camera_depth": targets.metadata["maximum_camera_depth"],
            "resize_mode": targets.metadata["resize_mode"],
            "weight": self.resume_mvs_surface_weight,
            "loss": (
                "sum(confidence*SmoothL1(e,beta=.01))/sum(confidence)"
                "+sum(confidence*SmoothL1(relu(-e-.01),beta=.01))"
                "/sum(confidence); e=(surf_depth/clamp(surf_alpha,1e-6)"
                "-mvs_depth)/mvs_depth"
            ),
        }
        if self.resume_checkpoint_mvs_surface is not None:
            for name, expected in metadata.items():
                if self.resume_checkpoint_mvs_surface.get(name) != expected:
                    raise ValueError(
                        f"Resume MVS surface {name} differs from its checkpoint"
                    )
        self.resume_mvs_surface_targets = targets
        self.resume_mvs_surface_metadata = metadata

    def _resume_mvs_surface_loss(
        self,
        camera_name: str,
        render_pkg: Mapping[str, torch.Tensor],
    ) -> MvsSurfaceLoss:
        targets = self.resume_mvs_surface_targets
        if targets is None:
            raise RuntimeError("MVS surface targets are not configured")
        surf_depth = render_pkg["surf_depth"]
        surf_alpha = render_pkg["surf_alpha"]
        target_depth, source_count = targets.target(
            camera_name,
            device=self.device,
            size=tuple(surf_depth.shape),
        )
        return rendered_mvs_surface_loss(
            surf_depth,
            surf_alpha,
            target_depth,
            source_count,
        )

    def _load_resume_checkpoint(self, path: Path) -> None:
        self.resume_parameter_policy = str(
            getattr(self, "resume_parameter_policy", "inherit")
        )
        self.resume_geometry_policy = str(
            getattr(self, "resume_geometry_policy", "inherit")
        )
        constrained_geometry = self.resume_geometry_policy != "inherit"
        constrained_parameters = self.resume_parameter_policy != "inherit"
        constrained_resume = constrained_geometry or constrained_parameters
        incompatible = [
            name
            for name, enabled in (
                ("enable_mcmc", bool(getattr(self.opt, "enable_mcmc", False))),
                (
                    "enable_densify",
                    bool(getattr(self.opt, "enable_densify", False))
                    and (
                        constrained_resume
                        or not (
                            str(getattr(self.opt, "densify_impl", "2dgs")) == "hexels_safe"
                            and str(getattr(self.opt, "densify_action", "mixed")) == "edge_child"
                        )
                    ),
                ),
                ("enable_children", bool(getattr(self.opt, "enable_children", False))),
                (
                    "edge_transfer_manifest",
                    constrained_resume
                    and getattr(self.opt, "edge_transfer_manifest", None) is not None,
                ),
                ("random_background", bool(getattr(self.opt, "random_background", False))),
                ("repulsion_knn_weight", float(getattr(self.opt, "repulsion_knn_weight", 0.0)) > 0.0),
                ("surface_coplanar_weight", float(getattr(self.opt, "surface_coplanar_weight", 0.0)) > 0.0),
                (
                    "surface_normal_smooth_weight",
                    float(getattr(self.opt, "surface_normal_smooth_weight", 0.0)) > 0.0,
                ),
            )
            if enabled
        ]
        if incompatible:
            raise ValueError(
                "Resume supports fixed-population deterministic training only; "
                f"disable {', '.join(incompatible)}"
            )
        if self.resume_geometry_lr_scale <= 0.0:
            raise ValueError("resume_geometry_lr_scale must be > 0")
        if self.resume_appearance_lr_scale <= 0.0:
            raise ValueError("resume_appearance_lr_scale must be > 0")
        if self.resume_train_holdout_stride > 0 and not bool(
            getattr(self.opt, "skip_postprocess", False)
        ):
            raise ValueError(
                "resume_train_holdout_stride requires skip_postprocess=True to keep the official test split blind"
            )
        if self.resume_train_holdout_stride > 0 and getattr(self.opt, "test_iterations", []):
            raise ValueError(
                "resume_train_holdout_stride requires empty test_iterations to keep the official test split blind"
            )
        if getattr(self, "resume_train_camera_manifest_path", None) is not None:
            if self.resume_train_holdout_stride > 0:
                raise ValueError(
                    "resume_train_camera_manifest is incompatible with resume_train_holdout_stride"
                )
            if not bool(getattr(self.opt, "skip_postprocess", False)):
                raise ValueError(
                    "resume_train_camera_manifest requires skip_postprocess=True to keep excluded views blind"
                )
            if getattr(self.opt, "test_iterations", []):
                raise ValueError(
                    "resume_train_camera_manifest requires empty test_iterations to keep excluded views blind"
                )
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Resume checkpoint does not exist: {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Resume checkpoint must contain a mapping: {path}")
        if checkpoint.get("resume_compatible") is False:
            raise ValueError("Resume checkpoint was produced by incompatible training options")
        checkpoint_lineage = checkpoint.get("lineage")
        if checkpoint_lineage is not None and not isinstance(checkpoint_lineage, Mapping):
            raise ValueError("Resume checkpoint lineage must be a mapping")
        checkpoint_edge_transfer = checkpoint.get("edge_transfer")
        if checkpoint_edge_transfer is not None and not isinstance(checkpoint_edge_transfer, Mapping):
            raise ValueError("Resume checkpoint edge_transfer must be a mapping")
        checkpoint_scale_constraint = checkpoint.get("resume_scale_constraint")
        if checkpoint_scale_constraint is not None and not isinstance(
            checkpoint_scale_constraint, Mapping
        ):
            raise ValueError("Resume checkpoint scale constraint must be a mapping")
        checkpoint_normal_trust = checkpoint.get("resume_normal_trust")
        if checkpoint_normal_trust is not None and not isinstance(
            checkpoint_normal_trust,
            Mapping,
        ):
            raise ValueError("Resume checkpoint normal trust must be a mapping")
        checkpoint_surface_anchor = checkpoint.get("resume_surface_anchor")
        if checkpoint_surface_anchor is not None and not isinstance(
            checkpoint_surface_anchor,
            Mapping,
        ):
            raise ValueError("Resume checkpoint surface anchor must be a mapping")
        checkpoint_mvs_surface = checkpoint.get("resume_mvs_surface")
        if checkpoint_mvs_surface is not None and not isinstance(
            checkpoint_mvs_surface,
            Mapping,
        ):
            raise ValueError("Resume checkpoint MVS surface must be a mapping")
        if checkpoint.get("init_geometry") != getattr(self, "init_geometry_metadata", None):
            raise ValueError("Resume checkpoint initial geometry path or SHA-256 does not match")
        source_config_path = path.parent.parent / "config.json"
        if not source_config_path.is_file():
            raise ValueError(f"Resume checkpoint is missing its source config: {source_config_path}")
        source_config = json.loads(source_config_path.read_text())
        if not isinstance(source_config, Mapping):
            raise ValueError(f"Resume source config must contain a mapping: {source_config_path}")
        source_config.setdefault("init_normal_mode", "identity")
        source_config.setdefault("spatial_appearance_start", 0)
        source_config.setdefault("surface_normal_objective", "legacy")
        source_config.setdefault("surface_normal_depth_mode", "expected")
        required_config_fields = {"source_path", *_RESUME_CONFIG_FIELDS}
        missing_config_fields = sorted(required_config_fields - set(source_config))
        if missing_config_fields:
            raise ValueError(
                "Resume source config is missing compatibility fields: "
                + ", ".join(missing_config_fields)
            )
        if checkpoint.get("resume_compatible") is None:
            source_incompatible = [
                name
                for name in (
                    "enable_mcmc",
                    "enable_densify",
                    "enable_children",
                    "random_background",
                    "repulsion_knn_weight",
                    "surface_coplanar_weight",
                    "surface_normal_smooth_weight",
                )
                if bool(source_config.get(name, False))
            ]
            if source_incompatible:
                raise ValueError(
                    "Resume checkpoint source used incompatible options: "
                    + ", ".join(source_incompatible)
                )
        source_geometry_policy = str(
            source_config.get("resume_geometry_policy", "inherit")
        )
        source_parameter_policy = str(
            source_config.get("resume_parameter_policy", "inherit")
        )
        checkpoint_parameter_policy = checkpoint.get(
            "resume_parameter_policy"
        )
        if (
            checkpoint_parameter_policy is not None
            and checkpoint_parameter_policy != source_parameter_policy
        ):
            raise ValueError(
                "Resume checkpoint parameter policy differs from its source config"
            )
        if (
            source_parameter_policy != "inherit"
            and self.resume_parameter_policy != source_parameter_policy
        ):
            raise ValueError(
                "resume_parameter_policy is sticky after a constrained continuation"
            )
        self.resume_parent_geometry_policy = source_geometry_policy
        checkpoint_geometry_policy = checkpoint.get("resume_geometry_policy")
        if (
            checkpoint_geometry_policy is not None
            and checkpoint_geometry_policy != source_geometry_policy
        ):
            raise ValueError("Resume checkpoint geometry policy differs from its source config")
        if source_geometry_policy != "inherit":
            if self.resume_geometry_policy != source_geometry_policy:
                raise ValueError(
                    "resume_geometry_policy is sticky after a constrained continuation"
                )
            if source_geometry_policy == "bounded_isotropic_shrink":
                for name in (
                    "resume_scale_lower_geometry_path",
                    "resume_scale_upper_geometry_path",
                ):
                    source_value = source_config.get(name)
                    current_value = getattr(self.opt, name)
                    if source_value is None or current_value is None or Path(
                        str(source_value)
                    ).resolve() != Path(str(current_value)).resolve():
                        raise ValueError(f"{name} is sticky across bounded resume")
                if float(source_config.get("resume_scale_anchor_weight", 0.0)) != float(
                    self.resume_scale_anchor_weight
                ):
                    raise ValueError("resume_scale_anchor_weight is sticky across bounded resume")
                if checkpoint_scale_constraint is None:
                    raise ValueError(
                        "Bounded parent checkpoint is missing scale constraint metadata"
                    )
        source_normal_trust_path = source_config.get("resume_normal_trust_path")
        if source_normal_trust_path is not None:
            if (
                self.resume_normal_trust_path is None
                or Path(str(source_normal_trust_path)).resolve()
                != self.resume_normal_trust_path
                or float(
                    source_config.get("resume_normal_trust_min_degrees", 5.0)
                )
                != self.resume_normal_trust_min_degrees
                or float(
                    source_config.get("resume_normal_trust_max_degrees", 25.0)
                )
                != self.resume_normal_trust_max_degrees
            ):
                raise ValueError("Resume normal trust configuration is sticky")
        source_surface_anchor_path = source_config.get("resume_surface_anchor_path")
        if source_surface_anchor_path is not None:
            if (
                self.resume_surface_anchor_path is None
                or Path(str(source_surface_anchor_path)).resolve()
                != self.resume_surface_anchor_path
                or float(
                    source_config.get(
                        "resume_surface_anchor_weight",
                        0.0,
                    )
                )
                != self.resume_surface_anchor_weight
            ):
                raise ValueError("Resume surface anchor configuration is sticky")
        source_mvs_surface_path = source_config.get("resume_mvs_surface_path")
        if source_mvs_surface_path is not None:
            if (
                self.resume_mvs_surface_path is None
                or Path(str(source_mvs_surface_path)).resolve()
                != self.resume_mvs_surface_path
                or float(source_config.get("resume_mvs_surface_weight", 0.0))
                != self.resume_mvs_surface_weight
                or checkpoint_mvs_surface is None
            ):
                raise ValueError("Resume MVS surface configuration is sticky")
        elif checkpoint_mvs_surface is not None:
            raise ValueError(
                "Resume checkpoint MVS surface metadata has no source configuration"
            )
        source_uses_resume_manifest = (
            source_config.get("resume_train_camera_manifest") is not None
        )
        source_camera_manifest = source_config.get("resume_train_camera_manifest")
        if source_camera_manifest is None:
            source_camera_manifest = source_config.get("train_camera_manifest")
        if constrained_geometry:
            if (
                source_camera_manifest is None
                or self.resume_train_camera_manifest_path is None
                or Path(str(source_camera_manifest)).resolve()
                != self.resume_train_camera_manifest_path
            ):
                raise ValueError(
                    "Constrained resume requires the same training camera manifest as its parent"
                )
            source_summary_path = path.parent.parent / "summary.json"
            if not source_summary_path.is_file():
                raise ValueError("Constrained resume parent is missing summary.json")
            source_summary = json.loads(source_summary_path.read_text())
            if not isinstance(source_summary, Mapping):
                raise ValueError("Constrained resume parent summary must be a mapping")
            if source_uses_resume_manifest:
                source_resume = source_summary.get("resume")
                if not isinstance(source_resume, Mapping):
                    raise ValueError(
                        "Constrained resume parent summary is missing resume metadata"
                    )
                expected_manifest_sha256 = source_resume.get(
                    "train_camera_manifest_sha256"
                )
                expected_names_sha256 = source_resume.get("train_camera_names_sha256")
            else:
                source_selection = source_summary.get("train_camera_selection")
                if not isinstance(source_selection, Mapping):
                    raise ValueError(
                        "Constrained resume parent summary is missing training selection metadata"
                    )
                expected_manifest_sha256 = source_selection.get("manifest_sha256")
                expected_names_sha256 = source_selection.get("camera_names_sha256")
            manifest = json.loads(self.resume_train_camera_manifest_path.read_text())
            if not isinstance(manifest, Mapping):
                raise ValueError("Constrained resume camera manifest must be a mapping")
            if (
                _sha256_file(self.resume_train_camera_manifest_path)
                != expected_manifest_sha256
                or manifest.get("camera_names_sha256") != expected_names_sha256
            ):
                raise ValueError(
                    "Constrained resume camera manifest content differs from its parent"
                )
        source_path = source_config["source_path"]
        if Path(str(source_path)).resolve() != Path(self.opt.source_path).resolve():
            raise ValueError(
                "Resume checkpoint source_path does not match the current source_path: "
                f"checkpoint={source_path!r}, current={self.opt.source_path!r}"
            )
        self.resume_parent_lambda_dssim = float(source_config["lambda_dssim"])
        self.resume_parent_photometric_loss = str(
            source_config.get("photometric_loss", "l1")
        )
        if (
            self.resume_parent_photometric_loss != self.opt.photometric_loss
            and not self.resume_allow_photometric_loss_change
        ):
            raise ValueError(
                "Resume checkpoint photometric_loss mismatch: "
                f"checkpoint={self.resume_parent_photometric_loss!r}, "
                f"current={self.opt.photometric_loss!r}"
            )
        self.resume_parent_camera_sampling = str(source_config["camera_sampling"])
        self.resume_parent_train_holdout_stride = int(
            source_config.get("resume_train_holdout_stride", 0)
        )
        source_geometry_start = int(source_config.get("geometry_start_iteration", 0))
        current_geometry_start = int(getattr(self, "geometry_start_iteration", 0))
        if source_geometry_start != current_geometry_start:
            raise ValueError(
                "geometry_start_iteration is sticky across resume: "
                f"checkpoint={source_geometry_start}, current={current_geometry_start}"
            )
        if (
            self.resume_parent_train_holdout_stride > 0
            and self.resume_train_holdout_stride != self.resume_parent_train_holdout_stride
        ):
            raise ValueError(
                "Resume holdout stride is sticky once enabled: "
                f"checkpoint={self.resume_parent_train_holdout_stride}, "
                f"current={self.resume_train_holdout_stride}"
            )
        mismatches = [
            f"{name}: checkpoint={source_config[name]!r}, current={getattr(self.opt, name)!r}"
            for name in _RESUME_CONFIG_FIELDS
            if source_config[name] != getattr(self.opt, name)
            and not (name == "lambda_dssim" and self.resume_allow_lambda_dssim_change)
            and not (
                name == "camera_sampling" and self.resume_allow_camera_sampling_change
            )
        ]
        if mismatches:
            raise ValueError("Resume checkpoint config mismatch: " + "; ".join(mismatches))

        iteration = checkpoint.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError(f"Resume checkpoint has invalid iteration: {iteration!r}")
        if int(self.opt.iterations) <= iteration:
            raise ValueError(
                f"--iterations must exceed checkpoint iteration {iteration}; got {self.opt.iterations}"
            )
        if (
            self.resume_geometry_policy == "bounded_isotropic_shrink"
            and int(self.opt.iterations) < int(self.geometry_start_iteration)
        ):
            raise ValueError(
                "bounded_isotropic_shrink requires an active geometry optimization window"
            )

        checkpoint_model_state = checkpoint.get("model")
        if not isinstance(checkpoint_model_state, Mapping):
            raise ValueError("Resume checkpoint is missing a model state mapping")
        expected_state = self.model.state_dict()
        model_state = dict(checkpoint_model_state)
        if "active_mask" in expected_state and "active_mask" not in model_state:
            model_state["active_mask"] = expected_state["active_mask"].clone()
        missing_keys = sorted(set(expected_state) - set(model_state))
        unexpected_keys = sorted(set(model_state) - set(expected_state))
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Resume checkpoint model keys do not match the current model: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        shape_mismatches: list[str] = []
        for name, value in expected_state.items():
            checkpoint_value = model_state[name]
            if not isinstance(checkpoint_value, torch.Tensor):
                shape_mismatches.append(f"{name}: checkpoint value is not a tensor")
            elif checkpoint_value.shape != value.shape:
                shape_mismatches.append(
                    f"{name}: checkpoint={tuple(checkpoint_value.shape)} current={tuple(value.shape)}"
                )
        if shape_mismatches:
            raise ValueError(
                "Resume checkpoint is not fixed-population/shape-compatible: "
                + "; ".join(shape_mismatches)
            )
        validate_active_mask_state(model_state, self.model.num_base)

        optimizer_state = checkpoint.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("Resume checkpoint is missing an optimizer state mapping")
        checkpoint_groups = optimizer_state.get("param_groups")
        if not isinstance(checkpoint_groups, list):
            raise ValueError("Resume checkpoint optimizer has invalid param_groups")
        if not all(isinstance(group, Mapping) for group in checkpoint_groups):
            raise ValueError("Resume checkpoint optimizer groups must be mappings")
        current_names = [str(group["name"]) for group in self.optimizer.param_groups]
        checkpoint_names = [str(group.get("name", "")) for group in checkpoint_groups]
        if checkpoint_names != current_names:
            raise ValueError(
                "Resume checkpoint optimizer groups do not match in order: "
                f"checkpoint={checkpoint_names}, current={current_names}"
            )
        if (
            self.resume_geometry_policy == "bounded_isotropic_shrink"
            and "log_scales" not in current_names
        ):
            raise ValueError("bounded_isotropic_shrink requires a log_scales optimizer group")
        for name, checkpoint_group, current_group in zip(
            current_names, checkpoint_groups, self.optimizer.param_groups
        ):
            checkpoint_params = checkpoint_group.get("params")
            if not isinstance(checkpoint_params, list) or len(checkpoint_params) != 1:
                raise ValueError(f"Resume checkpoint optimizer group {name!r} must contain one parameter")
            if len(current_group["params"]) != 1:
                raise ValueError(f"Current optimizer group {name!r} must contain one parameter")

        checkpoint_optimizer_values = optimizer_state.get("state")
        if not isinstance(checkpoint_optimizer_values, Mapping):
            raise ValueError("Resume checkpoint optimizer has invalid state")
        checkpoint_param_ids = [group["params"][0] for group in checkpoint_groups]
        if len(set(checkpoint_param_ids)) != len(checkpoint_param_ids):
            raise ValueError("Resume checkpoint optimizer parameter identifiers are not unique")
        unexpected_state_ids = set(checkpoint_optimizer_values) - set(checkpoint_param_ids)
        if unexpected_state_ids:
            raise ValueError(
                f"Resume checkpoint optimizer has unreferenced state entries: {sorted(unexpected_state_ids)}"
            )
        for name, checkpoint_group, current_group in zip(
            current_names, checkpoint_groups, self.optimizer.param_groups
        ):
            checkpoint_param = checkpoint_group["params"][0]
            parameter = current_group["params"][0]
            parameter_state = checkpoint_optimizer_values.get(checkpoint_param, {})
            if not isinstance(parameter_state, Mapping):
                raise ValueError(f"Resume checkpoint optimizer state for {name!r} is invalid")
            for state_name, state_value in parameter_state.items():
                if (
                    isinstance(state_value, torch.Tensor)
                    and state_value.ndim > 0
                    and state_value.shape != parameter.shape
                ):
                    raise ValueError(
                        f"Resume checkpoint optimizer state {name}.{state_name} has shape "
                        f"{tuple(state_value.shape)}, expected {tuple(parameter.shape)}"
                    )

        self.model.load_state_dict(model_state, strict=True)
        self.optimizer.load_state_dict(optimizer_state)
        if (
            self.resume_geometry_policy == "bounded_isotropic_shrink"
            and source_geometry_policy == "inherit"
        ):
            self.optimizer.state.get(self.model.log_scales, {}).clear()
        elif self.resume_geometry_policy == "bounded_isotropic_shrink":
            scale_state = self.optimizer.state.get(self.model.log_scales, {})
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = scale_state.get(name)
                if (
                    isinstance(value, torch.Tensor)
                    and value.shape == self.model.log_scales.shape
                    and not torch.equal(value[:, 0], value[:, 1])
                ):
                    raise ValueError(
                        f"Bounded resume scale optimizer state {name} is not isotropic"
                    )
        checkpoint_lrs = self._optimizer_lr_map()
        geometry_names = {"means3d", "log_scales", "rotation_quats"}
        if self.position_lr_scheduler is not None and "means3d" in checkpoint_lrs:
            scheduled_lr = float(self.position_lr_scheduler(iteration))
            if scheduled_lr <= 0.0:
                raise ValueError(f"Position LR schedule is nonpositive at resume iteration {iteration}")
            self.position_lr_resume_scale = (
                checkpoint_lrs["means3d"] * self.resume_geometry_lr_scale / scheduled_lr
            )
        for group in self.optimizer.param_groups:
            scale = (
                self.resume_geometry_lr_scale
                if group["name"] in geometry_names
                else self.resume_appearance_lr_scale
            )
            group["lr"] = float(group["lr"]) * scale

        self.start_iteration = iteration
        self.resume_checkpoint_path = path.resolve()
        self.resume_checkpoint_sha256 = _sha256_file(path)
        self.resume_checkpoint_lineage = (
            copy.deepcopy(dict(checkpoint_lineage))
            if checkpoint_lineage is not None
            else None
        )
        self.resume_checkpoint_schema_version = copy.deepcopy(checkpoint.get("schema_version"))
        self.resume_checkpoint_type = copy.deepcopy(checkpoint.get("checkpoint_type"))
        self.resume_checkpoint_edge_transfer = (
            copy.deepcopy(dict(checkpoint_edge_transfer))
            if checkpoint_edge_transfer is not None
            else None
        )
        self.resume_checkpoint_scale_constraint = (
            copy.deepcopy(dict(checkpoint_scale_constraint))
            if checkpoint_scale_constraint is not None
            else None
        )
        self.resume_checkpoint_normal_trust = (
            copy.deepcopy(dict(checkpoint_normal_trust))
            if checkpoint_normal_trust is not None
            else None
        )
        self.resume_checkpoint_surface_anchor = (
            copy.deepcopy(dict(checkpoint_surface_anchor))
            if checkpoint_surface_anchor is not None
            else None
        )
        self.resume_checkpoint_mvs_surface = (
            copy.deepcopy(dict(checkpoint_mvs_surface))
            if checkpoint_mvs_surface is not None
            else None
        )
        self.logger.info(
            "Resumed training from %s at iteration %d; geometry_lr_scale=%.6g "
            "appearance_lr_scale=%.6g position_lr_resume_scale=%.6g "
            "lambda_dssim(parent=%.6g current=%.6g change_allowed=%s) "
            "photometric_loss(parent=%s current=%s change_allowed=%s) "
            "camera_sampling(parent=%s current=%s change_allowed=%s) checkpoint_lrs=%s",
            path,
            iteration,
            self.resume_geometry_lr_scale,
            self.resume_appearance_lr_scale,
            self.position_lr_resume_scale,
            self.resume_parent_lambda_dssim,
            float(self.opt.lambda_dssim),
            self.resume_allow_lambda_dssim_change,
            self.resume_parent_photometric_loss,
            self.opt.photometric_loss,
            self.resume_allow_photometric_loss_change,
            self.resume_parent_camera_sampling,
            self.camera_sampling,
            self.resume_allow_camera_sampling_change,
            checkpoint_lrs,
        )

    def _resume_metadata(self) -> dict[str, object]:
        return {
            "checkpoint": (
                str(self.resume_checkpoint_path)
                if self.resume_checkpoint_path is not None
                else None
            ),
            "checkpoint_sha256": getattr(self, "resume_checkpoint_sha256", None),
            "start_iteration": int(self.start_iteration),
            "target_iteration": int(self.opt.iterations),
            "iterations_run": int(self.opt.iterations) - int(self.start_iteration),
            "geometry_lr_scale": float(self.resume_geometry_lr_scale),
            "appearance_lr_scale": float(self.resume_appearance_lr_scale),
            "position_lr_resume_scale": float(self.position_lr_resume_scale),
            "allow_lambda_dssim_change": self.resume_allow_lambda_dssim_change,
            "parent_lambda_dssim": self.resume_parent_lambda_dssim,
            "current_lambda_dssim": float(self.opt.lambda_dssim),
            "allow_photometric_loss_change": (
                self.resume_allow_photometric_loss_change
            ),
            "parent_photometric_loss": self.resume_parent_photometric_loss,
            "current_photometric_loss": self.opt.photometric_loss,
            "allow_camera_sampling_change": self.resume_allow_camera_sampling_change,
            "parent_camera_sampling": self.resume_parent_camera_sampling,
            "current_camera_sampling": self.camera_sampling,
            "camera_sampling_weights_path": (
                str(self.camera_sampling_weights_path)
                if getattr(self, "camera_sampling_weights_path", None)
                is not None
                else None
            ),
            "camera_sampling_weights_sha256": getattr(
                self,
                "camera_sampling_weights_sha256",
                None,
            ),
            "train_holdout_stride": self.resume_train_holdout_stride,
            "parent_train_holdout_stride": self.resume_parent_train_holdout_stride,
            "train_holdout_camera_names": self.resume_train_holdout_camera_names,
            "train_camera_manifest": (
                str(self.resume_train_camera_manifest_path)
                if self.resume_train_camera_manifest_path is not None
                else None
            ),
            "train_camera_manifest_sha256": self.resume_train_camera_manifest_sha256,
            "train_camera_count": len(self.resume_train_camera_names),
            "train_camera_names": self.resume_train_camera_names,
            "train_camera_names_sha256": self.resume_train_camera_names_sha256,
            "train_camera_trace_count": len(self.resume_train_camera_trace_names),
            "train_camera_trace_sha256": _json_sha256(self.resume_train_camera_trace_names),
            "geometry_policy": str(getattr(self, "resume_geometry_policy", "inherit")),
            "parameter_policy": str(
                getattr(self, "resume_parameter_policy", "inherit")
            ),
            "scale_constraint": self._resume_scale_constraint_summary(),
            "normal_trust": self._resume_normal_trust_summary(),
            "surface_anchor": self._resume_surface_anchor_summary(),
            "mvs_surface": self._resume_mvs_surface_summary(),
        }

    def _resume_mvs_surface_summary(self) -> dict[str, object] | None:
        metadata = getattr(self, "resume_mvs_surface_metadata", None)
        return None if metadata is None else dict(metadata)

    @torch.no_grad()
    def _resume_scale_constraint_summary(self) -> dict[str, object] | None:
        lower = getattr(self, "resume_scale_lower_log", None)
        upper = getattr(self, "resume_scale_upper_log", None)
        metadata = getattr(self, "resume_scale_constraint_metadata", None)
        if lower is None or upper is None or metadata is None:
            return None
        shared = self.model.log_scales.detach().mean(dim=1, keepdim=True)
        width = upper - lower
        flexible = width > 1e-7
        shrink = (upper[flexible] - shared[flexible]) / width[flexible]
        quantiles = torch.quantile(
            shrink.to(dtype=torch.float64),
            torch.tensor(
                [0.0, 0.25, 0.5, 0.75, 1.0],
                device=shrink.device,
                dtype=torch.float64,
            ),
        )
        tolerance = 1e-6
        lower_rows = flexible & ((shared - lower).abs() <= tolerance)
        upper_rows = (
            flexible
            & ~lower_rows
            & ((upper - shared).abs() <= tolerance)
        )
        interior_rows = flexible & ~lower_rows & ~upper_rows
        flexible_count = int(flexible.sum().item())
        return {
            **metadata,
            "maximum_axis_log_difference": float(
                (self.model.log_scales[:, 0] - self.model.log_scales[:, 1])
                .abs()
                .max()
                .item()
            ),
            "maximum_lower_violation": float((lower - shared).clamp_min(0.0).max().item()),
            "maximum_upper_violation": float((shared - upper).clamp_min(0.0).max().item()),
            "lower_fraction": float(lower_rows.sum().item() / flexible_count),
            "interior_fraction": float(interior_rows.sum().item() / flexible_count),
            "upper_fraction": float(upper_rows.sum().item() / flexible_count),
            "locked_fraction": float((~flexible).to(torch.float64).mean().item()),
            "shrink_fraction_quantiles": [float(value) for value in quantiles.tolist()],
            "anchor_loss": float(shrink.square().mean().item()),
        }

    @torch.no_grad()
    def _resume_normal_trust_summary(self) -> dict[str, object] | None:
        metadata = getattr(self, "resume_normal_trust_metadata", None)
        reference = getattr(self, "resume_normal_trust_reference", None)
        reliability = getattr(self, "resume_normal_trust_reliability", None)
        if metadata is None or reference is None or reliability is None:
            return None
        drift = torch.rad2deg(
            torch.acos(
                (
                    quaternion_normal_z(self.model.rotations.detach()) * reference
                )
                .sum(dim=-1)
                .abs()
                .clamp(0.0, 1.0)
            )
        )
        order = torch.argsort(reliability)
        quartiles = []
        for index, rows in enumerate(torch.tensor_split(order, 4)):
            values = drift[rows]
            quartiles.append(
                {
                    "reliability_quartile": index,
                    "rows": int(len(rows)),
                    "mean_degrees": float(values.mean().item()),
                    "p95_degrees": float(torch.quantile(values, 0.95).item()),
                    "max_degrees": float(values.max().item()),
                }
            )
        return {
            **metadata,
            "projection_steps": int(
                getattr(self, "resume_normal_trust_projection_steps", 0)
            ),
            "projected_rows": int(
                getattr(self, "resume_normal_trust_projected_rows", 0)
            ),
            "mean_drift_degrees": float(drift.mean().item()),
            "p95_drift_degrees": float(torch.quantile(drift, 0.95).item()),
            "max_drift_degrees": float(drift.max().item()),
            "drift_by_reliability": quartiles,
        }

    @torch.no_grad()
    def _resume_surface_anchor_summary(self) -> dict[str, object] | None:
        metadata = getattr(self, "resume_surface_anchor_metadata", None)
        reference = getattr(self, "resume_surface_anchor_reference", None)
        normals = getattr(self, "resume_surface_anchor_normals", None)
        reliability = getattr(self, "resume_surface_anchor_reliability", None)
        scales = getattr(self, "resume_surface_anchor_scales", None)
        if (
            metadata is None
            or reference is None
            or normals is None
            or reliability is None
            or scales is None
        ):
            return None
        normalized = (
            (self.model.means3d.detach() - reference) * normals
        ).sum(dim=-1) / scales
        absolute_normalized = normalized.abs()
        order = torch.argsort(reliability)
        quartiles = []
        for index, rows in enumerate(torch.tensor_split(order, 4)):
            values = absolute_normalized[rows]
            quartiles.append(
                {
                    "reliability_quartile": index,
                    "rows": int(len(rows)),
                    "mean_absolute_normalized_displacement": float(
                        values.mean().item()
                    ),
                    "p95_absolute_normalized_displacement": float(
                        torch.quantile(values, 0.95).item()
                    ),
                    "max_absolute_normalized_displacement": float(
                        values.max().item()
                    ),
                }
            )
        weighted_mean = (
            reliability * absolute_normalized
        ).sum() / reliability.sum()
        return {
            **metadata,
            "loss": float(self._resume_surface_anchor_loss().item()),
            "weighted_mean_absolute_normalized_displacement": float(
                weighted_mean.item()
            ),
            "p95_absolute_normalized_displacement": float(
                torch.quantile(absolute_normalized, 0.95).item()
            ),
            "max_absolute_normalized_displacement": float(
                absolute_normalized.max().item()
            ),
            "drift_by_reliability": quartiles,
        }

    def _densify_shadow_summary(self) -> dict[str, object]:
        child_mask = self.densify_shadow_child_mask
        if child_mask is None:
            raise RuntimeError("Shadow summary requested while shadow densification is disabled")
        attempted_mask = self.densify_shadow_attempted_parent_mask
        if attempted_mask is None:
            raise RuntimeError("Shadow attempted-parent state is not initialized")
        event = self.densify_shadow_event
        pending = None
        if event is not None:
            pending = {
                "birth_iteration": int(event.birth_iteration),
                "steps_completed": int(event.steps_completed),
                "parent_count": int(event.parent_ids.numel()),
                "child_count": int(event.child_ids.numel()),
            }
            if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
                pending["birth_envelope"] = self._densify_shadow_birth_envelope_telemetry(
                    event
                )
        pending_events = int(event is not None)
        pending_children = 0 if event is None else int(event.child_ids.numel())
        decided_events = (
            self.densify_shadow_counts["accepted_events"]
            + self.densify_shadow_counts["rejected_events"]
        )
        decided_children = (
            self.densify_shadow_counts["accepted_children"]
            + self.densify_shadow_counts["rejected_children"]
        )
        if self.densify_shadow_counts["proposed_events"] != decided_events + pending_events:
            raise RuntimeError("Shadow event counters are inconsistent")
        if self.densify_shadow_counts["proposed_children"] != decided_children + pending_children:
            raise RuntimeError("Shadow child counters are inconsistent")
        if int(child_mask.sum().item()) != self.densify_shadow_counts["accepted_children"] + pending_children:
            raise RuntimeError("Shadow child lineage and counters are inconsistent")
        summary = {
            "schema_version": 1,
            "append_only": True,
            "steps": int(self.opt.densify_shadow_steps),
            "permanent_residual": bool(self.opt.densify_shadow_permanent_residual),
            "steady_training_passes": (
                2 if bool(self.opt.densify_shadow_permanent_residual) else 1
            ),
            "probation_training_passes": (
                3 if bool(self.opt.densify_shadow_permanent_residual) else 2
            ),
            "thresholds": {
                "min_pooled_sse_reduction": float(
                    self.opt.densify_shadow_min_pooled_sse_reduction
                ),
                "min_mean_psnr_delta": float(self.opt.densify_shadow_min_mean_psnr_delta),
                "min_win_fraction": float(self.opt.densify_shadow_min_win_fraction),
                "min_worst_psnr_delta": float(self.opt.densify_shadow_min_worst_psnr_delta),
            },
            "counts": dict(self.densify_shadow_counts),
            "adaptive_rows": int(child_mask.sum().item()),
            "attempted_parent_rows": int(attempted_mask.sum().item()),
            "pending": pending,
            "last_decision": copy.deepcopy(self.densify_shadow_last_decision),
            "decisions": copy.deepcopy(self.densify_shadow_decisions),
        }
        gate_stride = int(getattr(self.opt, "densify_shadow_gate_stride", 0))
        if gate_stride > 0:
            summary["gate"] = {
                "stride": gate_stride,
                "fit_camera_count": len(self.densify_shadow_fit_camera_names),
                "decision_camera_count": len(self.densify_shadow_gate_camera_names),
                "decision_camera_names": sorted(self.densify_shadow_gate_camera_names),
            }
        salvage_groups = int(getattr(self.opt, "densify_shadow_salvage_groups", 0))
        if salvage_groups > 0:
            summary["salvage"] = {
                "selector": "bidirectional_group_prefix_v1",
                "groups": salvage_groups,
            }
            salvage_order = str(
                getattr(self.opt, "densify_shadow_salvage_order", "score")
            )
            if salvage_order != "score":
                summary["salvage"]["ordering"] = salvage_order
            prefix_policy = str(
                getattr(self.opt, "densify_shadow_salvage_prefix_policy", "best_pooled")
            )
            if prefix_policy != "best_pooled":
                summary["salvage"]["prefix_policy"] = prefix_policy
            selector_stride = int(
                getattr(self.opt, "densify_shadow_salvage_selector_stride", 0)
            )
            if selector_stride > 0:
                proposal_names = sorted(self.densify_shadow_proposal_camera_names)
                selector_names = sorted(self.densify_shadow_selector_camera_names)
                summary["salvage"]["nested_selector"] = {
                    "stride": selector_stride,
                    "proposal_camera_count": len(proposal_names),
                    "proposal_camera_names_sha256": _json_sha256(proposal_names),
                    "selector_camera_count": len(selector_names),
                    "selector_camera_names_sha256": _json_sha256(selector_names),
                    "defer_inner_gate": bool(
                        getattr(self.opt, "densify_shadow_defer_inner_gate", False)
                    ),
                }
            if bool(
                getattr(
                    self.opt,
                    "densify_shadow_salvage_allow_partial_cohort",
                    False,
                )
            ):
                summary["salvage"]["allow_partial_cohort"] = True
        if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
            summary["birth_envelope"] = {
                "enabled": True,
                "scene_physical_max_scale": float(
                    self.scene_extent * self.opt.densify_child_extent_cap_fraction
                ),
                "parent_scale_ratio": float(self.opt.densify_child_scale_ratio),
                "parent_opacity_fraction": float(
                    self.opt.densify_child_parent_opacity_fraction
                ),
            }
        return summary

    def _densify_shadow_checkpoint_audit(self) -> dict[str, object]:
        summary = self._densify_shadow_summary()
        event = self.densify_shadow_event
        pending = None
        if event is not None:
            pending = {
                "birth_iteration": int(event.birth_iteration),
                "steps_completed": int(event.steps_completed),
                "parent_ids": event.parent_ids.detach().clone(),
                "child_ids": event.child_ids.detach().clone(),
            }
            if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
                if (
                    event.birth_parent_distances is None
                    or event.birth_physical_max_scales is None
                    or event.birth_log_opacities is None
                ):
                    raise RuntimeError("Pending shadow birth-envelope bounds are missing")
                pending["birth_envelope"] = {
                    "birth_parent_distances": event.birth_parent_distances.detach().clone(),
                    "birth_physical_max_scales": (
                        event.birth_physical_max_scales.detach().clone()
                    ),
                    "birth_log_opacities": event.birth_log_opacities.detach().clone(),
                    "projection_steps": int(event.envelope_projection_steps),
                    "projected_mean_rows": int(event.envelope_projected_mean_rows),
                    "projected_scale_rows": int(event.envelope_projected_scale_rows),
                    "projected_opacity_rows": int(event.envelope_projected_opacity_rows),
                }
        summary["child_mask"] = self.densify_shadow_child_mask.detach().clone()
        summary["attempted_parent_mask"] = (
            self.densify_shadow_attempted_parent_mask.detach().clone()
        )
        summary["pending"] = pending
        return summary

    def _update_position_lr(self, iteration: int) -> None:
        if self.position_lr_scheduler is None:
            return
        lr = self.position_lr_scheduler(iteration) * self.position_lr_resume_scale
        for group in self.optimizer.param_groups:
            if group.get("name") == "means3d":
                group["lr"] = lr
                return

    def _apply_lr_decay_milestone(self, iteration: int) -> None:
        if iteration not in self.lr_decay_milestones:
            return
        self.position_lr_resume_scale *= self.lr_decay_factor
        for group in self.optimizer.param_groups:
            if group.get("name") != "means3d" or self.position_lr_scheduler is None:
                group["lr"] = float(group["lr"]) * self.lr_decay_factor
        self.logger.info(
            "Learning-rate decay: iteration=%d factor=%.6g cumulative_position_scale=%.6g",
            iteration,
            self.lr_decay_factor,
            self.position_lr_resume_scale,
        )

    def _update_sh_degree(self, iteration: int) -> int:
        if self.view_model != "sh3":
            return 3
        degree = _sh_degree_for_iteration(iteration, self.sh_progressive_interval)
        if degree != self.active_sh_degree:
            self.logger.info("SH active degree: %d at iteration %d", degree, iteration)
            self.active_sh_degree = degree
        return degree

    def _zero_adam_state(self, param: torch.Tensor, idxs: torch.Tensor) -> None:
        state = self.optimizer.state.get(param)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq"):
            buf = state.get(key)
            if isinstance(buf, torch.Tensor):
                buf[idxs] = 0

    def _rebuild_optimizer_with_state(self, old_optimizer: torch.optim.Optimizer) -> None:
        new_optimizer = self._build_optimizer()
        for old_group, new_group in zip(old_optimizer.param_groups, new_optimizer.param_groups):
            for key, value in old_group.items():
                if key != "params":
                    new_group[key] = value
            old_param = old_group["params"][0]
            new_param = new_group["params"][0]
            old_state = old_optimizer.state.get(old_param)
            if not old_state:
                continue
            new_state: dict[str, object] = {}
            for k, v in old_state.items():
                if isinstance(v, torch.Tensor) and v.shape == old_param.shape:
                    if v.ndim == 0:
                        new_v = v.clone()
                    else:
                        new_v = torch.zeros_like(new_param)
                        n = min(int(v.shape[0]), int(new_v.shape[0]))
                        if n > 0:
                            new_v[:n] = v[:n]
                    new_state[k] = new_v
                else:
                    new_state[k] = v
            new_optimizer.state[new_param] = new_state
        self.optimizer = new_optimizer

    def _mcmc_update_ema(self, contrib: torch.Tensor) -> None:
        if self.contrib_ema is None:
            return
        decay = float(self.opt.mcmc_ema_decay)
        if self.contrib_ema.shape != contrib.shape:
            raise ValueError(f"contrib_ema shape {tuple(self.contrib_ema.shape)} != contrib shape {tuple(contrib.shape)}")
        self.contrib_ema.mul_(decay).add_(contrib, alpha=(1.0 - decay))

    def _maybe_update_repulsion_knn(self, iteration: int) -> None:
        if float(getattr(self.opt, "repulsion_knn_weight", 0.0)) <= 0.0:
            return

        P = int(self.model.num_base)
        k = int(getattr(self.opt, "repulsion_knn_k", 8))
        interval = int(getattr(self.opt, "repulsion_knn_interval", 500))
        if P <= 1:
            self.repulsion_knn_idx = None
            self.repulsion_knn_P = P
            return
        if k >= P:
            k = P - 1

        needs_update = (
            self.repulsion_knn_idx is None
            or self.repulsion_knn_P != P
            or iteration == 1
            or (iteration % interval == 0)
        )
        if not needs_update:
            return

        from scipy.spatial import cKDTree

        pts = self.model.means3d.detach().float().cpu().numpy()
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k + 1)
        idx = idx[:, 1:]
        self.repulsion_knn_idx = torch.from_numpy(idx).to(device=self.device, dtype=torch.long)
        self.repulsion_knn_P = P

    def _repulsion_knn_loss(self) -> torch.Tensor:
        idx = self.repulsion_knn_idx
        if idx is None:
            return torch.zeros((), device=self.device, dtype=torch.float32)

        means = self.model.means3d
        scales = self.model.scales
        normals = _quat_to_normal_z(self.model.rotations)

        nbr_means = means[idx]
        delta = nbr_means - means[:, None, :]
        proj = (delta * normals[:, None, :]).sum(dim=-1, keepdim=True) * normals[:, None, :]
        delta_t = delta - proj
        dist = torch.linalg.norm(delta_t, dim=-1)

        radius = float(getattr(self.opt, "repulsion_knn_radius_frac", 1.0)) * 0.5 * (scales[:, 0] + scales[:, 1])
        excess = (radius[:, None] - dist).clamp_min(0.0)
        return (excess * excess).mean()

    def _maybe_update_surface_knn(self, iteration: int) -> None:
        need_knn = (
            float(getattr(self.opt, "surface_coplanar_weight", 0.0)) > 0.0
            or float(getattr(self.opt, "surface_normal_smooth_weight", 0.0)) > 0.0
            or bool(getattr(self.opt, "mcmc_reorient", False))
            or bool(getattr(self.opt, "densify_child_reorient", False))
        )
        if not need_knn:
            return

        P = int(self.model.num_base)
        k = int(getattr(self.opt, "surface_knn_k", 8))
        interval = int(getattr(self.opt, "surface_knn_interval", 500))
        if P <= 1:
            self.surface_knn_idx = None
            self.surface_knn_P = P
            return
        if k >= P:
            k = P - 1

        needs_update = (
            self.surface_knn_idx is None
            or self.surface_knn_P != P
            or iteration == 1
            or (iteration % interval == 0)
        )
        if not needs_update:
            return

        from scipy.spatial import cKDTree

        pts = self.model.means3d.detach().float().cpu().numpy()
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k + 1)
        idx = idx[:, 1:]
        self.surface_knn_idx = torch.from_numpy(idx).to(device=self.device, dtype=torch.long)
        self.surface_knn_P = P

    def _surface_knn_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.surface_knn_idx
        if idx is None:
            z = torch.zeros((), device=self.device, dtype=torch.float32)
            return z, z

        means = self.model.means3d
        normals = _quat_to_normal_z(self.model.rotations)

        nbr_means = means[idx]
        delta = nbr_means - means[:, None, :]
        proj_dist = (delta * normals[:, None, :]).sum(dim=-1)
        coplanar_loss = (proj_dist * proj_dist).mean()

        nbr_normals = normals[idx]
        dot = (normals[:, None, :] * nbr_normals).sum(dim=-1).clamp(-1.0, 1.0)
        normal_smooth_loss = (1.0 - dot).mean()
        return coplanar_loss, normal_smooth_loss

    def _estimate_pca_normals(self, src_idx: torch.Tensor) -> torch.Tensor:
        idx = self.surface_knn_idx
        if idx is None:
            raise ValueError("surface_knn_idx is not initialized")
        if src_idx.numel() == 0:
            return torch.empty((0, 3), device=self.device, dtype=torch.float32)

        means = self.model.means3d.detach()
        nbr_idx = idx[src_idx]
        nbr_pts = means[nbr_idx]
        centered = nbr_pts - nbr_pts.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered
        evals, evecs = torch.linalg.eigh(cov)
        normal = evecs[:, :, 0]

        cur = _quat_to_normal_z(self.model.rotations.detach()[src_idx])
        flip = (normal * cur).sum(dim=-1) < 0
        normal = torch.where(flip[:, None], -normal, normal)
        return torch.nn.functional.normalize(normal, dim=-1)

    def _estimate_pca_planarity(self, src_idx: torch.Tensor) -> torch.Tensor:
        idx = self.surface_knn_idx
        if idx is None:
            raise ValueError("surface_knn_idx is not initialized")
        if src_idx.numel() == 0:
            return torch.empty(
                (0,),
                device=self.device,
                dtype=torch.float32,
            )
        means = self.model.means3d.detach()
        neighbors = means[idx[src_idx]]
        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        eigenvalues = torch.linalg.eigvalsh(
            centered.transpose(1, 2) @ centered
        )
        denominator = eigenvalues[:, 2].clamp_min(
            torch.finfo(eigenvalues.dtype).eps
        )
        return (
            (eigenvalues[:, 1] - eigenvalues[:, 0]) / denominator
        ).clamp(0.0, 1.0)

    def _reorient_densify_children(
        self,
        children: dict[str, torch.Tensor],
        parent_ids: torch.Tensor,
    ) -> None:
        if not bool(getattr(self.opt, "densify_child_reorient", False)):
            return
        target_normals = self._estimate_pca_normals(parent_ids)
        old_rotations = torch.nn.functional.normalize(
            children["rotation_quats"],
            dim=-1,
        )
        old_normals = _quat_to_normal_z(old_rotations)
        minimum_planarity = float(
            getattr(
                self.opt,
                "densify_child_reorient_min_planarity",
                0.0,
            )
        )
        maximum_degrees = float(
            getattr(
                self.opt,
                "densify_child_reorient_max_degrees",
                180.0,
            )
        )
        accepted = torch.ones(
            parent_ids.shape,
            device=parent_ids.device,
            dtype=torch.bool,
        )
        if minimum_planarity > 0.0:
            accepted &= (
                self._estimate_pca_planarity(parent_ids)
                >= minimum_planarity
            )
        if maximum_degrees < 180.0:
            cosine_limit = math.cos(math.radians(maximum_degrees))
            alignment_cosine = (old_normals * target_normals).sum(
                dim=-1
            )
            accepted &= alignment_cosine >= cosine_limit
        target_normals = torch.where(
            accepted[:, None],
            target_normals,
            old_normals,
        )
        alignment = _quat_from_two_vectors(old_normals, target_normals)
        new_rotations = _quat_mul_wxyz(alignment, old_rotations)

        old_tx, old_ty = _quat_to_tangent_xy(old_rotations)
        new_tx, new_ty = _quat_to_tangent_xy(new_rotations)
        scales = children["log_scales"].exp()
        gradients = children["color_grads"]
        world_gradients = (
            old_tx[:, :, None]
            * (gradients[:, 0] / scales[:, 0, None])[:, None, :]
            + old_ty[:, :, None]
            * (gradients[:, 1] / scales[:, 1, None])[:, None, :]
        )
        children["color_grads"] = torch.stack(
            (
                torch.einsum("ni,nic->nc", new_tx, world_gradients)
                * scales[:, 0, None],
                torch.einsum("ni,nic->nc", new_ty, world_gradients)
                * scales[:, 1, None],
            ),
            dim=1,
        )
        children["rotation_quats"] = new_rotations

    def _append_primitives(
        self,
        new_tensors: dict[str, torch.Tensor],
        *,
        birth_iteration: int = 0,
    ) -> None:
        old_optimizer = self.optimizer
        old_P = int(self.model.opacity_logits.shape[0])
        next_population = old_P + int(new_tensors["means3d"].shape[0]) + self.model.num_children
        cap = int(getattr(self.opt, "densify_cap_max", 0))
        if cap > 0 and next_population > cap:
            raise ValueError(f"Primitive append would exceed budget: {next_population} > {cap}")
        self.densify_peak_population = max(getattr(self, "densify_peak_population", old_P), next_population)

        req_geom = bool(self.model.means3d.requires_grad)
        req_soft = bool(self.model.log_softness.requires_grad)
        req_colors = bool(self.model.colors.requires_grad)
        req_color_grads = bool(self.model.color_grads.requires_grad)
        req_color_quads = bool(self.model.color_quads.requires_grad)
        req_view_grads = bool(self.model.view_grads.requires_grad)
        req_specular = bool(self.model.specular_ks.requires_grad)
        req_opacity = bool(self.model.opacity_logits.requires_grad)

        with torch.no_grad():
            self.model.means3d = torch.nn.Parameter(
                torch.cat([self.model.means3d.data, new_tensors["means3d"]], dim=0),
                requires_grad=req_geom,
            )
            self.model.log_scales = torch.nn.Parameter(
                torch.cat([self.model.log_scales.data, new_tensors["log_scales"]], dim=0),
                requires_grad=req_geom,
            )
            self.model.rotation_quats = torch.nn.Parameter(
                torch.cat([self.model.rotation_quats.data, new_tensors["rotation_quats"]], dim=0),
                requires_grad=req_geom,
            )
            self.model.colors = torch.nn.Parameter(
                torch.cat([self.model.colors.data, new_tensors["colors"]], dim=0),
                requires_grad=req_colors,
            )
            self.model.color_grads = torch.nn.Parameter(
                torch.cat([self.model.color_grads.data, new_tensors["color_grads"]], dim=0),
                requires_grad=req_color_grads,
            )
            self.model.color_quads = torch.nn.Parameter(
                torch.cat([self.model.color_quads.data, new_tensors["color_quads"]], dim=0),
                requires_grad=req_color_quads,
            )
            self.model.view_grads = torch.nn.Parameter(
                torch.cat([self.model.view_grads.data, new_tensors["view_grads"]], dim=0),
                requires_grad=req_view_grads,
            )
            self.model.specular_ks = torch.nn.Parameter(
                torch.cat([self.model.specular_ks.data, new_tensors["specular_ks"]], dim=0),
                requires_grad=req_specular,
            )
            self.model.specular_log_s = torch.nn.Parameter(
                torch.cat([self.model.specular_log_s.data, new_tensors["specular_log_s"]], dim=0),
                requires_grad=req_specular,
            )
            self.model.opacity_logits = torch.nn.Parameter(
                torch.cat([self.model.opacity_logits.data, new_tensors["opacity_logits"]], dim=0),
                requires_grad=req_opacity,
            )
            self.model.log_softness = torch.nn.Parameter(
                torch.cat([self.model.log_softness.data, new_tensors["log_softness"]], dim=0),
                requires_grad=req_soft,
            )
            new_P = int(self.model.opacity_logits.shape[0])
            self.model.active_mask = torch.cat(
                [
                    self.model.active_mask,
                    torch.ones(new_P - old_P, device=self.device, dtype=torch.bool),
                ]
            )

        if self.contrib_ema is not None:
            new_P = int(self.model.opacity_logits.shape[0])
            extra = torch.zeros((new_P - old_P,), device=self.device, dtype=self.contrib_ema.dtype)
            self.contrib_ema = torch.cat([self.contrib_ema, extra], dim=0)

        if (
            self.enable_densify
            and self.densify_grad_accum is not None
            and self.densify_denom is not None
            and self.densify_max_radii2d is not None
        ):
            new_P = int(self.model.opacity_logits.shape[0])
            extra = torch.zeros((new_P - old_P, 1), device=self.device, dtype=self.densify_grad_accum.dtype)
            self.densify_grad_accum = torch.cat([self.densify_grad_accum, extra], dim=0)
            if getattr(self, "densify_abs_grad_accum", None) is not None:
                self.densify_abs_grad_accum = torch.cat([self.densify_abs_grad_accum, extra], dim=0)
            self.densify_denom = torch.cat([self.densify_denom, extra], dim=0)
            extra_r = torch.zeros((new_P - old_P,), device=self.device, dtype=self.densify_max_radii2d.dtype)
            self.densify_max_radii2d = torch.cat([self.densify_max_radii2d, extra_r], dim=0)

        added = new_P - old_P
        if self.densify_local_grad_abs_accum is not None:
            extra = torch.zeros((added, 2), device=self.device, dtype=self.densify_local_grad_abs_accum.dtype)
            self.densify_local_grad_abs_accum = torch.cat([self.densify_local_grad_abs_accum, extra], dim=0)
        if self.densify_local_grad_signed_accum is not None:
            extra = torch.zeros((added, 2), device=self.device, dtype=self.densify_local_grad_signed_accum.dtype)
            self.densify_local_grad_signed_accum = torch.cat([self.densify_local_grad_signed_accum, extra], dim=0)
        if self.densify_residual_mass_accum is not None:
            extra = torch.zeros((added,), device=self.device, dtype=self.densify_residual_mass_accum.dtype)
            self.densify_residual_mass_accum = torch.cat([self.densify_residual_mass_accum, extra], dim=0)
        if self.densify_centroid_mass_accum is not None:
            extra = torch.zeros((added, 2), device=self.device, dtype=self.densify_centroid_mass_accum.dtype)
            self.densify_centroid_mass_accum = torch.cat([self.densify_centroid_mass_accum, extra], dim=0)
        if self.densify_direction_vote_accum is not None:
            extra = torch.zeros((added, 2), device=self.device, dtype=self.densify_direction_vote_accum.dtype)
            self.densify_direction_vote_accum = torch.cat([self.densify_direction_vote_accum, extra], dim=0)
        if self.densify_moment_vote_count is not None:
            extra = torch.zeros((added,), device=self.device, dtype=self.densify_moment_vote_count.dtype)
            self.densify_moment_vote_count = torch.cat([self.densify_moment_vote_count, extra], dim=0)
        if self.densify_mv_edge_top_scores is not None:
            extra = torch.zeros(
                (added, *self.densify_mv_edge_top_scores.shape[1:]),
                device=self.device,
                dtype=self.densify_mv_edge_top_scores.dtype,
            )
            self.densify_mv_edge_top_scores = torch.cat(
                [self.densify_mv_edge_top_scores, extra], dim=0
            )
        if self.densify_mv_edge_top_centroids is not None:
            extra = torch.zeros(
                (added, *self.densify_mv_edge_top_centroids.shape[1:]),
                device=self.device,
                dtype=self.densify_mv_edge_top_centroids.dtype,
            )
            self.densify_mv_edge_top_centroids = torch.cat(
                [self.densify_mv_edge_top_centroids, extra], dim=0
            )
        if self.densify_mv_edge_vote_counts is not None:
            extra = torch.zeros(
                (added, *self.densify_mv_edge_vote_counts.shape[1:]),
                device=self.device,
                dtype=self.densify_mv_edge_vote_counts.dtype,
            )
            self.densify_mv_edge_vote_counts = torch.cat(
                [self.densify_mv_edge_vote_counts, extra], dim=0
            )
        if self.densify_attributed_edge_mass is not None:
            extra = torch.zeros(
                (added, 6),
                device=self.device,
                dtype=self.densify_attributed_edge_mass.dtype,
            )
            self.densify_attributed_edge_mass = torch.cat(
                [self.densify_attributed_edge_mass, extra], dim=0
            )
        if self.densify_attributed_edge_centroid_mass is not None:
            extra = torch.zeros(
                (added, 6, 2),
                device=self.device,
                dtype=self.densify_attributed_edge_centroid_mass.dtype,
            )
            self.densify_attributed_edge_centroid_mass = torch.cat(
                [self.densify_attributed_edge_centroid_mass, extra], dim=0
            )
        if self.densify_attributed_edge_vote_counts is not None:
            extra = torch.zeros(
                (added, 6),
                device=self.device,
                dtype=self.densify_attributed_edge_vote_counts.dtype,
            )
            self.densify_attributed_edge_vote_counts = torch.cat(
                [self.densify_attributed_edge_vote_counts, extra], dim=0
            )
        for name in (
            "densify_covariance_coverage",
            "densify_covariance_energy",
            "densify_covariance_robust_weight",
            "densify_covariance_first_moment",
            "densify_covariance_second_moment",
            "densify_covariance_double_angle",
            "densify_covariance_side_mass",
            "densify_covariance_material_view_count",
            "densify_covariance_axis_view_count",
        ):
            buffer = getattr(self, name)
            if buffer is not None:
                extra = torch.zeros(
                    (added, *buffer.shape[1:]),
                    device=self.device,
                    dtype=buffer.dtype,
                )
                setattr(self, name, torch.cat([buffer, extra], dim=0))
        if self.densify_contrib_accum is not None:
            extra = torch.zeros((added,), device=self.device, dtype=self.densify_contrib_accum.dtype)
            self.densify_contrib_accum = torch.cat([self.densify_contrib_accum, extra], dim=0)
        if self.densify_birth_iteration is not None:
            extra = torch.full(
                (added,),
                int(birth_iteration),
                device=self.device,
                dtype=self.densify_birth_iteration.dtype,
            )
            self.densify_birth_iteration = torch.cat([self.densify_birth_iteration, extra], dim=0)
        if self.densify_has_spawned_child is not None:
            extra = torch.zeros((added,), device=self.device, dtype=torch.bool)
            self.densify_has_spawned_child = torch.cat([self.densify_has_spawned_child, extra], dim=0)
        if getattr(self, "densify_shadow_child_mask", None) is not None:
            extra = torch.zeros((added,), device=self.device, dtype=torch.bool)
            self.densify_shadow_child_mask = torch.cat(
                [self.densify_shadow_child_mask, extra], dim=0
            )
        if getattr(self, "densify_shadow_attempted_parent_mask", None) is not None:
            extra = torch.zeros((added,), device=self.device, dtype=torch.bool)
            self.densify_shadow_attempted_parent_mask = torch.cat(
                [self.densify_shadow_attempted_parent_mask, extra], dim=0
            )

        self._rebuild_optimizer_with_state(old_optimizer)
        self._restart_prune_observation()

    @torch.no_grad()
    def _replace_primitive_rows(
        self,
        primitive_ids: torch.Tensor,
        rows: dict[str, torch.Tensor],
        *,
        birth_iteration: int,
    ) -> None:
        primitive_ids = primitive_ids.to(device=self.device, dtype=torch.long)
        for name, values in rows.items():
            parameter = getattr(self.model, name)
            parameter.data[primitive_ids] = values
            if parameter.grad is not None:
                parameter.grad[primitive_ids] = 0

        population = int(self.model.num_base)
        for group in self.optimizer.param_groups:
            parameter = group["params"][0]
            state = self.optimizer.state.get(parameter)
            if not state:
                continue
            for value in state.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == population:
                    value[primitive_ids] = 0

        if self.densify_grad_accum is not None:
            self.densify_grad_accum[primitive_ids] = 0
        if getattr(self, "densify_abs_grad_accum", None) is not None:
            self.densify_abs_grad_accum[primitive_ids] = 0
        if self.densify_denom is not None:
            self.densify_denom[primitive_ids] = 0
        if self.densify_max_radii2d is not None:
            self.densify_max_radii2d[primitive_ids] = 0
        if self.densify_local_grad_abs_accum is not None:
            self.densify_local_grad_abs_accum[primitive_ids] = 0
        if self.densify_local_grad_signed_accum is not None:
            self.densify_local_grad_signed_accum[primitive_ids] = 0
        if self.densify_residual_mass_accum is not None:
            self.densify_residual_mass_accum[primitive_ids] = 0
        if self.densify_centroid_mass_accum is not None:
            self.densify_centroid_mass_accum[primitive_ids] = 0
        if self.densify_direction_vote_accum is not None:
            self.densify_direction_vote_accum[primitive_ids] = 0
        if self.densify_moment_vote_count is not None:
            self.densify_moment_vote_count[primitive_ids] = 0
        if self.densify_mv_edge_top_scores is not None:
            self.densify_mv_edge_top_scores[primitive_ids] = 0
        if self.densify_mv_edge_top_centroids is not None:
            self.densify_mv_edge_top_centroids[primitive_ids] = 0
        if self.densify_mv_edge_vote_counts is not None:
            self.densify_mv_edge_vote_counts[primitive_ids] = 0
        if self.densify_attributed_edge_mass is not None:
            self.densify_attributed_edge_mass[primitive_ids] = 0
        if self.densify_attributed_edge_centroid_mass is not None:
            self.densify_attributed_edge_centroid_mass[primitive_ids] = 0
        if self.densify_attributed_edge_vote_counts is not None:
            self.densify_attributed_edge_vote_counts[primitive_ids] = 0
        for name in (
            "densify_covariance_coverage",
            "densify_covariance_energy",
            "densify_covariance_robust_weight",
            "densify_covariance_first_moment",
            "densify_covariance_second_moment",
            "densify_covariance_double_angle",
            "densify_covariance_side_mass",
            "densify_covariance_material_view_count",
            "densify_covariance_axis_view_count",
        ):
            buffer = getattr(self, name)
            if buffer is not None:
                buffer[primitive_ids] = 0
        if self.densify_contrib_accum is not None:
            self.densify_contrib_accum[primitive_ids] = 0
        if self.densify_birth_iteration is not None:
            self.densify_birth_iteration[primitive_ids] = int(birth_iteration)
        if self.densify_has_spawned_child is not None:
            self.densify_has_spawned_child[primitive_ids] = False
        if getattr(self, "densify_shadow_child_mask", None) is not None:
            self.densify_shadow_child_mask[primitive_ids] = False
        if getattr(self, "densify_shadow_attempted_parent_mask", None) is not None:
            self.densify_shadow_attempted_parent_mask[primitive_ids] = False
        self._restart_prune_observation()

    def _restart_prune_observation(self) -> None:
        if self.densify_contrib_accum is not None:
            self.densify_contrib_accum.zero_()
        self.densify_contrib_camera_names.clear()

    def _reset_densify_stats(self) -> None:
        if not self.enable_densify:
            return
        P = int(self.model.num_base)
        self.densify_grad_accum = torch.zeros((P, 1), device=self.device, dtype=torch.float32)
        if getattr(self, "densify_abs_grad_accum", None) is not None:
            self.densify_abs_grad_accum = torch.zeros_like(self.densify_grad_accum)
        self.densify_denom = torch.zeros((P, 1), device=self.device, dtype=torch.float32)
        self.densify_max_radii2d = torch.zeros((P,), device=self.device, dtype=torch.float32)
        if self.densify_local_grad_abs_accum is not None:
            self.densify_local_grad_abs_accum.zero_()
        if self.densify_local_grad_signed_accum is not None:
            self.densify_local_grad_signed_accum.zero_()
        if self.densify_residual_mass_accum is not None:
            self.densify_residual_mass_accum.zero_()
        if self.densify_centroid_mass_accum is not None:
            self.densify_centroid_mass_accum.zero_()
        if self.densify_direction_vote_accum is not None:
            self.densify_direction_vote_accum.zero_()
        if self.densify_moment_vote_count is not None:
            self.densify_moment_vote_count.zero_()
        if self.densify_mv_edge_top_scores is not None:
            self.densify_mv_edge_top_scores.zero_()
        if self.densify_mv_edge_top_centroids is not None:
            self.densify_mv_edge_top_centroids.zero_()
        if self.densify_mv_edge_vote_counts is not None:
            self.densify_mv_edge_vote_counts.zero_()
        if self.densify_attributed_edge_mass is not None:
            self.densify_attributed_edge_mass.zero_()
        if self.densify_attributed_edge_centroid_mass is not None:
            self.densify_attributed_edge_centroid_mass.zero_()
        if self.densify_attributed_edge_vote_counts is not None:
            self.densify_attributed_edge_vote_counts.zero_()
        for name in (
            "densify_covariance_coverage",
            "densify_covariance_energy",
            "densify_covariance_robust_weight",
            "densify_covariance_first_moment",
            "densify_covariance_second_moment",
            "densify_covariance_double_angle",
            "densify_covariance_side_mass",
            "densify_covariance_material_view_count",
            "densify_covariance_axis_view_count",
        ):
            buffer = getattr(self, name)
            if buffer is not None:
                buffer.zero_()
        self.densify_moment_camera_names.clear()

    @torch.no_grad()
    def _prune_primitives(self, prune_mask: torch.Tensor) -> None:
        if getattr(self, "densify_joint_coplanar_event", None) is not None:
            raise RuntimeError(
                "Physical pruning is blocked while joint coplanar probation is pending"
            )
        prune_mask = prune_mask.to(device=self.device, dtype=torch.bool)
        keep_mask = ~prune_mask
        if int(keep_mask.sum().item()) == int(prune_mask.numel()):
            return

        old_optimizer = self.optimizer
        old_P = int(self.model.num_base)
        new_P = int(keep_mask.sum().item())
        if new_P <= 0:
            raise ValueError("Pruning removed all base primitives")

        req_geom = bool(self.model.means3d.requires_grad)
        req_soft = bool(self.model.log_softness.requires_grad)
        req_colors = bool(self.model.colors.requires_grad)
        req_color_grads = bool(self.model.color_grads.requires_grad)
        req_color_quads = bool(self.model.color_quads.requires_grad)
        req_view_grads = bool(self.model.view_grads.requires_grad)
        req_specular = bool(self.model.specular_ks.requires_grad)
        req_opacity = bool(self.model.opacity_logits.requires_grad)

        with torch.no_grad():
            self.model.means3d = torch.nn.Parameter(self.model.means3d.data[keep_mask], requires_grad=req_geom)
            self.model.log_scales = torch.nn.Parameter(self.model.log_scales.data[keep_mask], requires_grad=req_geom)
            self.model.rotation_quats = torch.nn.Parameter(
                self.model.rotation_quats.data[keep_mask], requires_grad=req_geom
            )
            self.model.colors = torch.nn.Parameter(self.model.colors.data[keep_mask], requires_grad=req_colors)
            self.model.color_grads = torch.nn.Parameter(
                self.model.color_grads.data[keep_mask], requires_grad=req_color_grads
            )
            self.model.color_quads = torch.nn.Parameter(
                self.model.color_quads.data[keep_mask], requires_grad=req_color_quads
            )
            self.model.view_grads = torch.nn.Parameter(self.model.view_grads.data[keep_mask], requires_grad=req_view_grads)
            self.model.specular_ks = torch.nn.Parameter(self.model.specular_ks.data[keep_mask], requires_grad=req_specular)
            self.model.specular_log_s = torch.nn.Parameter(
                self.model.specular_log_s.data[keep_mask], requires_grad=req_specular
            )
            self.model.opacity_logits = torch.nn.Parameter(
                self.model.opacity_logits.data[keep_mask], requires_grad=req_opacity
            )
            self.model.log_softness = torch.nn.Parameter(self.model.log_softness.data[keep_mask], requires_grad=req_soft)
            self.model.active_mask = self.model.active_mask[keep_mask]

        if self.contrib_ema is not None:
            self.contrib_ema = self.contrib_ema[keep_mask]

        if self.densify_grad_accum is not None:
            self.densify_grad_accum = self.densify_grad_accum[keep_mask]
        if getattr(self, "densify_abs_grad_accum", None) is not None:
            self.densify_abs_grad_accum = self.densify_abs_grad_accum[keep_mask]
        if self.densify_denom is not None:
            self.densify_denom = self.densify_denom[keep_mask]
        if self.densify_max_radii2d is not None:
            self.densify_max_radii2d = self.densify_max_radii2d[keep_mask]
        if self.densify_local_grad_abs_accum is not None:
            self.densify_local_grad_abs_accum = self.densify_local_grad_abs_accum[keep_mask]
        if self.densify_local_grad_signed_accum is not None:
            self.densify_local_grad_signed_accum = self.densify_local_grad_signed_accum[keep_mask]
        if self.densify_residual_mass_accum is not None:
            self.densify_residual_mass_accum = self.densify_residual_mass_accum[keep_mask]
        if self.densify_centroid_mass_accum is not None:
            self.densify_centroid_mass_accum = self.densify_centroid_mass_accum[keep_mask]
        if self.densify_direction_vote_accum is not None:
            self.densify_direction_vote_accum = self.densify_direction_vote_accum[keep_mask]
        if self.densify_moment_vote_count is not None:
            self.densify_moment_vote_count = self.densify_moment_vote_count[keep_mask]
        if self.densify_mv_edge_top_scores is not None:
            self.densify_mv_edge_top_scores = self.densify_mv_edge_top_scores[keep_mask]
        if self.densify_mv_edge_top_centroids is not None:
            self.densify_mv_edge_top_centroids = self.densify_mv_edge_top_centroids[
                keep_mask
            ]
        if self.densify_mv_edge_vote_counts is not None:
            self.densify_mv_edge_vote_counts = self.densify_mv_edge_vote_counts[keep_mask]
        if self.densify_attributed_edge_mass is not None:
            self.densify_attributed_edge_mass = self.densify_attributed_edge_mass[keep_mask]
        if self.densify_attributed_edge_centroid_mass is not None:
            self.densify_attributed_edge_centroid_mass = (
                self.densify_attributed_edge_centroid_mass[keep_mask]
            )
        if self.densify_attributed_edge_vote_counts is not None:
            self.densify_attributed_edge_vote_counts = (
                self.densify_attributed_edge_vote_counts[keep_mask]
            )
        for name in (
            "densify_covariance_coverage",
            "densify_covariance_energy",
            "densify_covariance_robust_weight",
            "densify_covariance_first_moment",
            "densify_covariance_second_moment",
            "densify_covariance_double_angle",
            "densify_covariance_side_mass",
            "densify_covariance_material_view_count",
            "densify_covariance_axis_view_count",
        ):
            buffer = getattr(self, name)
            if buffer is not None:
                setattr(self, name, buffer[keep_mask])
        if self.densify_contrib_accum is not None:
            self.densify_contrib_accum = self.densify_contrib_accum[keep_mask]
        if self.densify_birth_iteration is not None:
            self.densify_birth_iteration = self.densify_birth_iteration[keep_mask]
        if self.densify_has_spawned_child is not None:
            self.densify_has_spawned_child = self.densify_has_spawned_child[keep_mask]
        if getattr(self, "densify_shadow_child_mask", None) is not None:
            self.densify_shadow_child_mask = self.densify_shadow_child_mask[keep_mask]
        if getattr(self, "densify_shadow_attempted_parent_mask", None) is not None:
            self.densify_shadow_attempted_parent_mask = self.densify_shadow_attempted_parent_mask[
                keep_mask
            ]

        new_optimizer = self._build_optimizer()
        for old_group, new_group in zip(old_optimizer.param_groups, new_optimizer.param_groups):
            if old_group.get("name") != new_group.get("name"):
                raise ValueError(
                    f"Optimizer group mismatch: {old_group.get('name')} != {new_group.get('name')}"
                )
            for key, value in old_group.items():
                if key != "params":
                    new_group[key] = value
            old_param = old_group["params"][0]
            new_param = new_group["params"][0]
            old_state = old_optimizer.state.get(old_param)
            if not old_state:
                continue
            new_state: dict[str, object] = {}
            for k, v in old_state.items():
                if isinstance(v, torch.Tensor) and v.shape == old_param.shape and v.ndim > 0 and v.shape[0] == old_P:
                    new_state[k] = v[keep_mask].clone()
                elif isinstance(v, torch.Tensor) and v.ndim == 0:
                    new_state[k] = v.clone()
                else:
                    new_state[k] = v
            new_optimizer.state[new_param] = new_state
        self.optimizer = new_optimizer

        self.repulsion_knn_idx = None
        self.repulsion_knn_P = 0
        self.surface_knn_idx = None
        self.surface_knn_P = 0
        self._restart_prune_observation()

        self.logger.info("Densify: pruned %d primitives (P=%d)", int(old_P - new_P), int(new_P))

    @torch.no_grad()
    def _reset_opacity(self) -> None:
        new_opacity = torch.minimum(self.model.opacities.detach(), torch.full_like(self.model.opacity_logits, 0.01))
        self.model.opacity_logits.data.copy_(opacity_to_logits(new_opacity))
        self.model.opacity_logits.grad = None
        state = self.optimizer.state.get(self.model.opacity_logits)
        if state:
            for key in ("exp_avg", "exp_avg_sq"):
                buf = state.get(key)
                if isinstance(buf, torch.Tensor):
                    buf.zero_()

    @torch.no_grad()
    def _densify_add_stats(self, viewspace_points: torch.Tensor, visibility_filter: torch.Tensor) -> None:
        if self.densify_grad_accum is None or self.densify_denom is None:
            return
        grad = viewspace_points.grad
        if grad is None:
            return
        base_P = int(self.model.num_base)
        mag = torch.linalg.norm(grad.detach()[:base_P, :2], dim=-1, keepdim=True)
        self.densify_grad_accum[visibility_filter] += mag[visibility_filter]
        if bool(getattr(self.opt, "densify_absgrad_rank", False)):
            self.densify_abs_grad_accum[visibility_filter] += grad.detach()[:base_P, 3:4][visibility_filter]
        self.densify_denom[visibility_filter] += 1.0
        if self.densify_local_grad_abs_accum is None or self.densify_local_grad_signed_accum is None:
            return
        world_grad = self.model.means3d.grad
        if world_grad is None:
            return
        tangent_x, tangent_y = _quat_to_tangent_xy(self.model.rotations.detach())
        local_grad = torch.stack(
            [
                (world_grad.detach() * tangent_x).sum(dim=-1),
                (world_grad.detach() * tangent_y).sum(dim=-1),
            ],
            dim=-1,
        )
        self.densify_local_grad_abs_accum[visibility_filter] += local_grad[visibility_filter].abs()
        self.densify_local_grad_signed_accum[visibility_filter] += local_grad[visibility_filter]

    @torch.no_grad()
    def _densify_add_contrib(self, contribution: torch.Tensor, *, camera_name: str) -> None:
        if self.densify_contrib_accum is None:
            return
        self.densify_contrib_accum += contribution.detach()[: self.model.num_base]
        self.densify_contrib_camera_names.add(camera_name)

    def _densify_add_moment_stats(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        contribution: torch.Tensor,
        *,
        camera_name: str,
        filtered_prediction: Optional[torch.Tensor] = None,
    ) -> None:
        if (
            self.densify_residual_mass_accum is None
            or self.densify_centroid_mass_accum is None
            or self.densify_direction_vote_accum is None
            or self.densify_moment_vote_count is None
        ):
            return
        if camera_name in self._densify_shadow_withheld_camera_names():
            return
        if camera_name in self.densify_moment_camera_names:
            return
        self.densify_moment_camera_names.add(camera_name)

        pixel_count = prediction.shape[-2] * prediction.shape[-1]
        if filtered_prediction is None:
            residual = (prediction.detach() - target.detach()).square()
        else:
            residual = torch.minimum(
                torch.abs(prediction.detach() - target.detach()),
                torch.abs(filtered_prediction.detach() - target.detach()),
            ).square()
        residual = residual / (3.0 * pixel_count)
        objective = torch.sum(prediction * residual)
        color_gradient, color_grads_gradient = torch.autograd.grad(
            objective,
            (self.model.colors, self.model.color_grads),
            retain_graph=True,
        )
        mass, centroid, _ = residual_first_moments(
            color_gradient,
            color_grads_gradient,
        )
        base_contribution = contribution.detach()[: self.model.num_base]
        material = _residual_mass_or_density_mask(mass, base_contribution)
        min_radius = float(getattr(self.opt, "densify_moment_min_radius", 0.25))
        edge_normals = canonical_edge_normals(device=centroid.device, dtype=centroid.dtype)
        radius = torch.max(centroid @ edge_normals.T, dim=1).values
        material &= radius >= min_radius
        parent_ids = torch.nonzero(material, as_tuple=True)[0]
        if parent_ids.numel() == 0:
            return
        with torch.no_grad():
            selected_centroids = centroid[parent_ids]
            selected_mass = mass[parent_ids]
            residual_topk = int(getattr(self.opt, "densify_residual_topk_views", 0))
            if residual_topk > 0:
                if (
                    self.densify_mv_edge_top_scores is None
                    or self.densify_mv_edge_top_centroids is None
                    or self.densify_mv_edge_vote_counts is None
                ):
                    raise RuntimeError("Multi-view residual top-K buffers are not initialized")
                selected_edges = torch.argmax(
                    selected_centroids @ edge_normals.T,
                    dim=1,
                )
                top_scores = self.densify_mv_edge_top_scores[parent_ids, selected_edges]
                top_centroids = self.densify_mv_edge_top_centroids[
                    parent_ids, selected_edges
                ]
                top_scores, top_centroids = update_residual_topk_views(
                    top_scores,
                    top_centroids,
                    selected_mass,
                    selected_centroids,
                )
                self.densify_mv_edge_top_scores[parent_ids, selected_edges] = top_scores
                self.densify_mv_edge_top_centroids[
                    parent_ids, selected_edges
                ] = top_centroids
                self.densify_mv_edge_vote_counts[parent_ids, selected_edges] += 1
                return
            directions = torch.nn.functional.normalize(selected_centroids, dim=1)
            self.densify_residual_mass_accum[parent_ids] += selected_mass
            self.densify_centroid_mass_accum[parent_ids] += selected_mass[:, None] * selected_centroids
            self.densify_direction_vote_accum[parent_ids] += directions
            self.densify_moment_vote_count[parent_ids] += 1.0

    @torch.no_grad()
    def _densify_add_attributed_stats(
        self,
        prediction: torch.Tensor,
        filtered_prediction: torch.Tensor,
        target: torch.Tensor,
        contribution: torch.Tensor,
        owner: Mapping[str, torch.Tensor],
        *,
        camera_name: str,
    ) -> None:
        if (
            self.densify_attributed_edge_mass is None
            or self.densify_attributed_edge_centroid_mass is None
            or self.densify_attributed_edge_vote_counts is None
        ):
            return
        if camera_name in self._densify_shadow_withheld_camera_names():
            return
        if camera_name in self.densify_moment_camera_names:
            return
        if int(getattr(self.model, "num_children", 0)) != 0:
            raise RuntimeError("Attributed densification does not support learned child owners")
        if not torch.equal(owner["image"], prediction.detach()):
            raise RuntimeError("Owner render image differs from the training render")

        owner_ids = owner["owner_id"]
        if bool((owner_ids >= self.model.num_base).any()):
            raise RuntimeError("Owner map contains a non-base primitive ID")
        aggregation = aggregate_attributed_residual(
            prediction.detach().cpu(),
            filtered_prediction.detach().cpu(),
            target.detach().cpu(),
            owner_ids.detach().cpu(),
            owner["owner_weight"].detach().cpu(),
            owner["owner_uv"].detach().permute(1, 2, 0).contiguous().cpu(),
            owner["alpha"].detach().cpu(),
            num_primitives=int(self.model.num_base),
            weak_support_power=int(
                getattr(self.opt, "densify_attributed_weak_power", 0)
            ),
        )
        total_mass = aggregation.mass.sum(dim=1)
        material = _residual_mass_or_density_mask(
            total_mass,
            contribution.detach()[: self.model.num_base].to(
                device="cpu", dtype=torch.float64
            ),
        )
        parent_ids = torch.nonzero(material, as_tuple=True)[0]
        if parent_ids.numel() > 0:
            edge_ids = torch.argmax(
                aggregation.camera_vote_mask[parent_ids].to(torch.int8), dim=1
            )
            selected_mass = aggregation.mass[parent_ids, edge_ids]
            positive = selected_mass > 0.0
            parent_ids = parent_ids[positive]
            edge_ids = edge_ids[positive]
            selected_mass = selected_mass[positive]
            selected_centroid_mass = aggregation.centroid_mass[
                parent_ids, edge_ids
            ]
            gpu_parents = parent_ids.to(device=self.device)
            gpu_edges = edge_ids.to(device=self.device)
            self.densify_attributed_edge_mass[gpu_parents, gpu_edges] += selected_mass.to(
                device=self.device
            )
            self.densify_attributed_edge_centroid_mass[
                gpu_parents, gpu_edges
            ] += selected_centroid_mass.to(device=self.device)
            self.densify_attributed_edge_vote_counts[gpu_parents, gpu_edges] += 1
        self.densify_moment_camera_names.add(camera_name)

    @torch.no_grad()
    def _densify_add_covariance_stats(
        self,
        prediction: torch.Tensor,
        filtered_prediction: torch.Tensor,
        target: torch.Tensor,
        contribution: torch.Tensor,
        owner: Mapping[str, torch.Tensor],
        *,
        camera_name: str,
    ) -> None:
        buffers = (
            self.densify_covariance_coverage,
            self.densify_covariance_energy,
            self.densify_covariance_robust_weight,
            self.densify_covariance_first_moment,
            self.densify_covariance_second_moment,
            self.densify_covariance_double_angle,
            self.densify_covariance_side_mass,
            self.densify_covariance_material_view_count,
            self.densify_covariance_axis_view_count,
        )
        if any(buffer is None for buffer in buffers):
            return
        if camera_name in self._densify_shadow_withheld_camera_names():
            return
        if camera_name in self.densify_moment_camera_names:
            return
        if int(getattr(self.model, "num_children", 0)) != 0:
            raise RuntimeError("Attributed densification does not support learned child owners")
        if not torch.equal(owner["image"], prediction.detach()):
            raise RuntimeError("Owner render image differs from the training render")

        owner_ids = owner["owner_id"]
        if bool((owner_ids >= self.model.num_base).any()):
            raise RuntimeError("Owner map contains a non-base primitive ID")
        aggregation = aggregate_owner_local_moments(
            prediction.detach().cpu(),
            filtered_prediction.detach().cpu(),
            target.detach().cpu(),
            owner_ids.detach().cpu(),
            owner["owner_weight"].detach().cpu(),
            owner["owner_uv"].detach().permute(1, 2, 0).contiguous().cpu(),
            owner["alpha"].detach().cpu(),
            num_primitives=int(self.model.num_base),
            weak_support_power=int(
                getattr(self.opt, "densify_attributed_weak_power", 0)
            ),
        )
        material = _residual_mass_or_density_mask(
            aggregation.energy,
            contribution.detach()[: self.model.num_base].to(
                device="cpu", dtype=torch.float64
            ),
        )
        parent_ids = torch.nonzero(material, as_tuple=True)[0]
        if parent_ids.numel() > 0:
            gpu_parents = parent_ids.to(device=self.device)
            self.densify_covariance_coverage[gpu_parents] += aggregation.coverage[
                parent_ids
            ].to(device=self.device)
            self.densify_covariance_energy[gpu_parents] += aggregation.energy[
                parent_ids
            ].to(device=self.device)
            self.densify_covariance_robust_weight[
                gpu_parents
            ] += aggregation.robust_weight[parent_ids].to(device=self.device)
            self.densify_covariance_first_moment[
                gpu_parents
            ] += aggregation.first_moment[parent_ids].to(device=self.device)
            self.densify_covariance_second_moment[
                gpu_parents
            ] += aggregation.second_moment[parent_ids].to(device=self.device)
            self.densify_covariance_double_angle[
                gpu_parents
            ] += aggregation.double_angle[parent_ids].to(device=self.device)
            self.densify_covariance_side_mass[
                gpu_parents
            ] += aggregation.side_mass[parent_ids].to(device=self.device)
            self.densify_covariance_material_view_count[gpu_parents] += 1
            axis_parents = parent_ids[aggregation.axis_valid[parent_ids]]
            if axis_parents.numel() > 0:
                self.densify_covariance_axis_view_count[
                    axis_parents.to(device=self.device)
                ] += 1
        self.densify_moment_camera_names.add(camera_name)

    def _densify_covariance_summary(self) -> OwnerLocalMomentSummary:
        buffers = (
            self.densify_covariance_coverage,
            self.densify_covariance_energy,
            self.densify_covariance_robust_weight,
            self.densify_covariance_first_moment,
            self.densify_covariance_second_moment,
            self.densify_covariance_double_angle,
            self.densify_covariance_axis_view_count,
            self.densify_covariance_material_view_count,
            self.densify_covariance_side_mass,
        )
        if any(buffer is None for buffer in buffers):
            raise RuntimeError("Attributed covariance buffers are not initialized")
        return summarize_owner_local_moments(
            *(buffer.detach().cpu() for buffer in buffers),
            min_material_views=int(
                getattr(self.opt, "densify_covariance_min_views", 8)
            ),
            min_anisotropy=float(
                getattr(self.opt, "densify_covariance_min_anisotropy", 0.15)
            ),
            min_axis_coherence=float(
                getattr(self.opt, "densify_covariance_min_axis_coherence", 0.50)
            ),
            min_side_balance=float(
                getattr(self.opt, "densify_covariance_min_side_balance", 0.30)
            ),
            min_delta=float(
                getattr(self.opt, "densify_covariance_min_delta", 0.10)
            ),
        )

    @torch.no_grad()
    def _densify_and_prune(self, *, size_threshold: int, iteration: int = 0) -> None:
        if getattr(self, "densify_shadow_event", None) is not None:
            raise RuntimeError("Densification mutation is blocked while a shadow event is pending")
        if getattr(self, "densify_joint_coplanar_event", None) is not None:
            raise RuntimeError(
                "Densification mutation is blocked while joint coplanar probation is pending"
            )
        impl = str(getattr(self.opt, "densify_impl", "2dgs"))
        if impl == "2dgs":
            self._densify_and_prune_2dgs(size_threshold=size_threshold)
            return
        self._densify_and_prune_hexels_safe(size_threshold=size_threshold, iteration=iteration)

    @torch.no_grad()
    def _start_densify_joint_coplanar_event(
        self,
        parent_ids: torch.Tensor,
        child_ids: torch.Tensor,
        *,
        iteration: int,
    ) -> None:
        if int(getattr(self.opt, "densify_joint_coplanar_steps", 0)) <= 0:
            return
        if getattr(self, "densify_shadow_event", None) is not None:
            raise RuntimeError("Joint coplanar probation cannot overlap shadow probation")
        if getattr(self, "densify_joint_coplanar_event", None) is not None:
            raise RuntimeError("Only one joint coplanar densification event may be pending")

        parents = parent_ids.to(device=self.device, dtype=torch.long).detach().clone()
        children = child_ids.to(device=self.device, dtype=torch.long).detach().clone()
        if parents.ndim != 1 or children.ndim != 1 or parents.shape != children.shape:
            raise ValueError("Joint coplanar parent and child IDs must be aligned vectors")
        if parents.numel() == 0:
            raise ValueError("Joint coplanar probation requires a positive parent-child mapping")
        population = int(self.model.num_base)
        if (
            int(parents.min().item()) < 0
            or int(children.min().item()) < 0
            or int(parents.max().item()) >= population
            or int(children.max().item()) >= population
        ):
            raise ValueError("Joint coplanar parent and child IDs are out of range")
        if int(torch.unique(children).numel()) != int(children.numel()):
            raise ValueError("Joint coplanar child IDs must be unique")
        if bool(torch.isin(parents, children).any().item()):
            raise ValueError("Joint coplanar parent and child rows must be distinct")
        if not bool(torch.all(self.model.active_mask[parents]).item()) or not bool(
            torch.all(self.model.active_mask[children]).item()
        ):
            raise ValueError("Joint coplanar parent and child rows must be active")
        self.densify_joint_coplanar_event = _DensifyJointCoplanarEvent(
            parent_ids=parents,
            child_ids=children,
            birth_iteration=int(iteration),
            configured_steps=int(self.opt.densify_joint_coplanar_steps),
        )

    @staticmethod
    def _densify_joint_coplanar_event_audit(
        event: _DensifyJointCoplanarEvent,
        *,
        completed: bool,
    ) -> dict[str, object]:
        audit: dict[str, object] = {
            "birth_iteration": int(event.birth_iteration),
            "configured_steps": int(event.configured_steps),
            "steps_completed": int(event.steps_completed),
            "first_constrained_iteration": event.first_constrained_iteration,
            "last_constrained_iteration": event.last_constrained_iteration,
            "mapping_count": int(event.child_ids.numel()),
            "unique_parent_count": int(torch.unique(event.parent_ids).numel()),
            "parent_ids": event.parent_ids.detach().cpu().tolist(),
            "child_ids": event.child_ids.detach().cpu().tolist(),
        }
        if completed:
            expected_first = int(event.birth_iteration + 1)
            expected_last = int(event.birth_iteration + event.configured_steps)
            exact = (
                event.steps_completed == event.configured_steps
                and event.first_constrained_iteration == expected_first
                and event.last_constrained_iteration == expected_last
            )
            audit.update(
                {
                    "completion_iteration": event.last_constrained_iteration,
                    "expected_first_constrained_iteration": expected_first,
                    "expected_last_constrained_iteration": expected_last,
                    "exact_constrained_step_count": exact,
                }
            )
        return audit

    def _densify_joint_coplanar_audit(self) -> dict[str, object]:
        event = getattr(self, "densify_joint_coplanar_event", None)
        pending = None
        if event is not None:
            pending = self._densify_joint_coplanar_event_audit(
                event,
                completed=False,
            )
        return {
            "schema_version": 2,
            "configured_steps": int(
                getattr(self.opt, "densify_joint_coplanar_steps", 0)
            ),
            "pending": pending,
            "last_completed": copy.deepcopy(
                getattr(self, "densify_joint_coplanar_last_completed", None)
            ),
        }

    def _densify_shadow_gradient_ids(self) -> torch.Tensor:
        event = self.densify_shadow_event
        if bool(getattr(self.opt, "densify_shadow_permanent_residual", True)):
            if self.densify_shadow_child_mask is None:
                return torch.empty((0,), device=self.device, dtype=torch.long)
            return torch.nonzero(self.densify_shadow_child_mask, as_tuple=True)[0]
        if event is None:
            return torch.empty((0,), device=self.device, dtype=torch.long)
        return event.child_ids

    @torch.no_grad()
    def _constrain_coplanar_child_gradients(
        self,
        parents: torch.Tensor,
        children: torch.Tensor,
    ) -> None:
        means_grad = self.model.means3d.grad
        if means_grad is not None:
            parent_rotations = torch.nn.functional.normalize(
                self.model.rotation_quats.detach()[parents], dim=1
            )
            normals = _quat_to_normal_z(parent_rotations)
            child_grad = means_grad[children]
            means_grad[children] = child_grad - (
                child_grad * normals
            ).sum(dim=1, keepdim=True) * normals
        if self.model.rotation_quats.grad is not None:
            self.model.rotation_quats.grad[children] = 0
        if self.model.view_grads.grad is not None:
            self.model.view_grads.grad[children] = 0

    @torch.no_grad()
    def _project_coplanar_child_rows(
        self,
        parents: torch.Tensor,
        children: torch.Tensor,
    ) -> None:
        parent_rotations = torch.nn.functional.normalize(
            self.model.rotation_quats.detach()[parents], dim=1
        )
        normals = _quat_to_normal_z(parent_rotations)
        parent_means = self.model.means3d[parents]
        offsets = self.model.means3d[children] - parent_means
        projected_means = parent_means + offsets - (
            offsets * normals
        ).sum(dim=1, keepdim=True) * normals
        self.model.means3d.index_copy_(0, children, projected_means)
        self.model.rotation_quats.index_copy_(
            0,
            children,
            self.model.rotation_quats.index_select(0, parents),
        )
        self.model.view_grads.index_copy_(
            0,
            children,
            self.model.view_grads.index_select(0, parents),
        )

    @torch.no_grad()
    def _constrain_densify_shadow_child_gradients(self) -> None:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        self._constrain_coplanar_child_gradients(event.parent_ids, event.child_ids)

    @torch.no_grad()
    def _project_densify_shadow_child_probation(self) -> None:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        self._project_coplanar_child_rows(event.parent_ids, event.child_ids)

    @torch.no_grad()
    def _constrain_densify_joint_coplanar_gradients(self) -> None:
        event = self.densify_joint_coplanar_event
        if event is None:
            raise RuntimeError("No joint coplanar event is pending")
        self._constrain_coplanar_child_gradients(event.parent_ids, event.child_ids)

    @torch.no_grad()
    def _project_densify_joint_coplanar_rows(self) -> None:
        event = self.densify_joint_coplanar_event
        if event is None:
            raise RuntimeError("No joint coplanar event is pending")
        self._project_coplanar_child_rows(event.parent_ids, event.child_ids)

    @torch.no_grad()
    def _finish_densify_joint_coplanar_event(self) -> None:
        event = self.densify_joint_coplanar_event
        if event is None:
            raise RuntimeError("No joint coplanar event is pending")
        completed = self._densify_joint_coplanar_event_audit(event, completed=True)
        if not bool(completed["exact_constrained_step_count"]):
            raise RuntimeError("Joint coplanar probation step accounting is inconsistent")
        self._zero_adam_state(self.model.means3d, event.child_ids)
        self.densify_joint_coplanar_last_completed = completed
        self.densify_joint_coplanar_event = None

    def _densify_shadow_withheld_camera_names(self) -> set[str]:
        return set(getattr(self, "densify_shadow_gate_camera_names", set())) | set(
            getattr(self, "densify_shadow_selector_camera_names", set())
        )

    def _densify_shadow_control_backward(
        self,
        camera: object,
        target: torch.Tensor,
        background: torch.Tensor,
        *,
        sh_degree: int,
    ) -> torch.Tensor:
        self.densify_shadow_frozen_step_ids = None
        adaptive_ids = self._densify_shadow_gradient_ids()
        if adaptive_ids.numel() == 0:
            raise RuntimeError("Shadow control backward requires adaptive rows")
        if not bool(torch.all(self.model.active_mask[adaptive_ids]).item()):
            raise RuntimeError("Shadow adaptive rows must be active during proposal optimization")

        population = int(self.model.num_base)
        row_parameters = [
            group["params"][0]
            for group in self.optimizer.param_groups
            if group["params"][0].ndim > 0
            and group["params"][0].shape[0] == population
        ]

        def capture_gradients(ids: torch.Tensor) -> list[Optional[torch.Tensor]]:
            return [
                None if parameter.grad is None else parameter.grad[ids].detach().clone()
                for parameter in row_parameters
            ]

        def photometric_backward(inactive_ids: torch.Tensor) -> torch.Tensor:
            self.optimizer.zero_grad(set_to_none=True)
            self.model.active_mask[inactive_ids] = False
            prediction = self.renderer.render(
                camera,
                self._render_model(),
                background,
                sh_degree=sh_degree,
            )["image"]
            l1 = self.l1_loss(prediction, target)
            photometric = (
                l1
                if self.opt.photometric_loss == "l1"
                else torch.nn.functional.mse_loss(prediction, target)
            )
            dssim = 1.0 - self.ssim(prediction, target)
            objective = (
                (1.0 - self.opt.lambda_dssim) * photometric
                + self.opt.lambda_dssim * dssim
            )
            objective.backward()
            self.model.active_mask[inactive_ids] = True
            return objective

        event = self.densify_shadow_event
        permanent = bool(getattr(self.opt, "densify_shadow_permanent_residual", True))
        held_out = str(getattr(camera, "image_name", "")) in (
            self._densify_shadow_withheld_camera_names()
        )
        proposal_ids = event.child_ids if permanent and event is not None else adaptive_ids
        proposal_gradients = capture_gradients(proposal_ids)

        accepted_ids = adaptive_ids[:0]
        accepted_gradients: list[Optional[torch.Tensor]] = []
        if permanent and event is not None:
            accepted_mask = self.densify_shadow_child_mask.clone()
            accepted_mask[event.child_ids] = False
            accepted_ids = torch.nonzero(accepted_mask, as_tuple=True)[0]
            if accepted_ids.numel() > 0:
                photometric_backward(event.child_ids)
                accepted_gradients = capture_gradients(accepted_ids)

        control_ids = adaptive_ids if permanent else event.child_ids
        control_loss = photometric_backward(control_ids)

        for parameter, saved in zip(row_parameters, proposal_gradients):
            if parameter.grad is None:
                if saved is None:
                    continue
                parameter.grad = torch.zeros_like(parameter)
            if saved is None:
                parameter.grad[proposal_ids] = 0
            else:
                parameter.grad[proposal_ids] = saved
        for parameter, saved in zip(row_parameters, accepted_gradients):
            if parameter.grad is None:
                if saved is None:
                    continue
                parameter.grad = torch.zeros_like(parameter)
            if saved is None:
                parameter.grad[accepted_ids] = 0
            else:
                parameter.grad[accepted_ids] = saved
        if held_out:
            for parameter in row_parameters:
                if parameter.grad is not None:
                    parameter.grad[adaptive_ids] = 0
            self.densify_shadow_frozen_step_ids = adaptive_ids.detach().clone()
        if (
            event is not None
            and str(
                getattr(self.opt, "densify_shadow_child_probation", "unrestricted")
            )
            == "coplanar_spatial"
        ):
            self._constrain_densify_shadow_child_gradients()
        return control_loss

    def _densify_shadow_metrics_pass(self, metrics: Mapping[str, object]) -> bool:
        return bool(
            metrics["finite"]
            and float(metrics["pooled_sse_reduction"])
            >= float(self.opt.densify_shadow_min_pooled_sse_reduction)
            and float(metrics["mean_psnr_delta"])
            >= float(self.opt.densify_shadow_min_mean_psnr_delta)
            and float(metrics["win_fraction"])
            >= float(self.opt.densify_shadow_min_win_fraction)
            and float(metrics["worst_psnr_delta"])
            >= float(self.opt.densify_shadow_min_worst_psnr_delta)
        )

    @torch.no_grad()
    def _project_densify_shadow_birth_envelope(self) -> None:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        if (
            event.birth_parent_distances is None
            or event.birth_physical_max_scales is None
            or event.birth_log_opacities is None
        ):
            raise RuntimeError("Pending shadow birth-envelope bounds are missing")
        child_count = int(event.child_ids.numel())
        if (
            event.parent_ids.shape != event.child_ids.shape
            or event.birth_parent_distances.shape != (child_count,)
            or event.birth_physical_max_scales.shape != (child_count,)
            or event.birth_log_opacities.shape != (child_count,)
        ):
            raise RuntimeError("Pending shadow birth-envelope bounds are misaligned")

        parents = event.parent_ids
        children = event.child_ids
        offsets = self.model.means3d[children] - self.model.means3d[parents]
        distances = torch.linalg.vector_norm(offsets, dim=1)
        mean_mask = distances > event.birth_parent_distances
        if bool(mean_mask.any().item()):
            projected_ids = children[mean_mask]
            ratios = event.birth_parent_distances[mean_mask] / distances[mean_mask]
            self.model.means3d.data[projected_ids] = (
                self.model.means3d[parents[mean_mask]] + offsets[mean_mask] * ratios[:, None]
            )
            self._zero_adam_state(self.model.means3d, projected_ids)

        scale_modifier = float(getattr(self.opt, "scale_modifier", 1.0))
        parent_physical_max = (
            scale_modifier * self.model.scales[parents].max(dim=1).values
        )
        scene_physical_max = float(
            self.scene_extent * self.opt.densify_child_extent_cap_fraction
        )
        scale_bounds = torch.minimum(
            event.birth_physical_max_scales,
            parent_physical_max * float(self.opt.densify_child_scale_ratio),
        ).clamp(max=scene_physical_max)
        child_physical_max = (
            scale_modifier * self.model.scales[children].max(dim=1).values
        )
        scale_mask = child_physical_max > scale_bounds
        if bool(scale_mask.any().item()):
            projected_ids = children[scale_mask]
            log_ratio = torch.log(
                scale_bounds[scale_mask] / child_physical_max[scale_mask]
            )
            self.model.log_scales.data[projected_ids] += log_ratio[:, None]
            self._zero_adam_state(self.model.log_scales, projected_ids)

        parent_log_opacity = -torch.nn.functional.softplus(
            -self.model.opacity_logits[parents]
        )
        opacity_bounds = torch.minimum(
            event.birth_log_opacities,
            parent_log_opacity
            + math.log(float(self.opt.densify_child_parent_opacity_fraction)),
        )
        child_log_opacity = -torch.nn.functional.softplus(
            -self.model.opacity_logits[children]
        )
        opacity_mask = child_log_opacity > opacity_bounds
        if bool(opacity_mask.any().item()):
            projected_ids = children[opacity_mask]
            bound = opacity_bounds[opacity_mask]
            self.model.opacity_logits.data[projected_ids] = bound - torch.log1p(
                -torch.exp(bound)
            )
            self._zero_adam_state(self.model.opacity_logits, projected_ids)

        event.envelope_projection_steps += 1
        event.envelope_projected_mean_rows += int(mean_mask.sum().item())
        event.envelope_projected_scale_rows += int(scale_mask.sum().item())
        event.envelope_projected_opacity_rows += int(opacity_mask.sum().item())

    @torch.no_grad()
    def _densify_shadow_birth_envelope_telemetry(
        self,
        event: _DensifyShadowEvent,
    ) -> dict[str, object]:
        if (
            event.birth_parent_distances is None
            or event.birth_physical_max_scales is None
            or event.birth_log_opacities is None
        ):
            raise RuntimeError("Pending shadow birth-envelope bounds are missing")

        def quantiles(values: torch.Tensor) -> dict[str, float]:
            values = values.detach().to(device="cpu", dtype=torch.float64)
            points = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], dtype=values.dtype)
            result = torch.quantile(values, points).tolist()
            return dict(zip(("min", "p05", "median", "p95", "max"), result))

        parents = event.parent_ids
        scale_modifier = float(getattr(self.opt, "scale_modifier", 1.0))
        parent_physical_max = (
            scale_modifier * self.model.scales[parents].max(dim=1).values
        )
        scene_physical_max = float(
            self.scene_extent * self.opt.densify_child_extent_cap_fraction
        )
        scale_bounds = torch.minimum(
            event.birth_physical_max_scales,
            parent_physical_max * float(self.opt.densify_child_scale_ratio),
        ).clamp(max=scene_physical_max)
        parent_log_opacity = -torch.nn.functional.softplus(
            -self.model.opacity_logits[parents]
        )
        opacity_bounds = torch.minimum(
            event.birth_log_opacities,
            parent_log_opacity
            + math.log(float(self.opt.densify_child_parent_opacity_fraction)),
        )
        return {
            "birth_bounds": {
                "parent_distance": quantiles(event.birth_parent_distances),
                "physical_max_scale": quantiles(event.birth_physical_max_scales),
                "opacity": quantiles(torch.exp(event.birth_log_opacities)),
            },
            "effective_bounds": {
                "parent_distance": quantiles(event.birth_parent_distances),
                "physical_max_scale": quantiles(scale_bounds),
                "opacity": quantiles(torch.exp(opacity_bounds)),
            },
            "projection_counts": {
                "steps": int(event.envelope_projection_steps),
                "means3d_rows": int(event.envelope_projected_mean_rows),
                "log_scales_rows": int(event.envelope_projected_scale_rows),
                "opacity_logits_rows": int(event.envelope_projected_opacity_rows),
            },
        }

    @torch.no_grad()
    def _densify_shadow_child_state_telemetry(
        self,
        parent_ids: torch.Tensor,
        child_ids: torch.Tensor,
    ) -> dict[str, object]:
        def quantiles(values: torch.Tensor) -> dict[str, float]:
            values = values.detach().to(device="cpu", dtype=torch.float64)
            points = torch.tensor(
                [0.0, 0.05, 0.5, 0.95, 1.0],
                dtype=values.dtype,
            )
            result = torch.quantile(values, points).tolist()
            return dict(zip(("min", "p05", "median", "p95", "max"), result))

        children = child_ids
        parents = parent_ids
        child_scales = self.model.scales[children].max(dim=1).values
        parent_scales = self.model.scales[parents].max(dim=1).values
        child_opacities = self.model.opacities[children]
        parent_opacities = self.model.opacities[parents]
        offsets = torch.linalg.vector_norm(
            self.model.means3d[children] - self.model.means3d[parents],
            dim=1,
        )
        digest = hashlib.sha256()
        for name in self._base_optimizer_param_names():
            value = getattr(self.model, name).detach()[children].contiguous().cpu()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(json.dumps(list(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
        colors = self.model.colors.detach()[children]
        return {
            "child_parameter_rows_sha256": digest.hexdigest(),
            "opacity": quantiles(child_opacities),
            "max_scale": quantiles(child_scales),
            "world_offset": quantiles(offsets),
            "max_scale_parent_ratio": quantiles(
                child_scales / parent_scales.clamp_min(1e-12)
            ),
            "opacity_parent_ratio": quantiles(
                child_opacities / parent_opacities.clamp_min(1e-12)
            ),
            "colors_outside_unit_interval": int(
                ((colors < 0.0) | (colors > 1.0)).any(dim=1).sum().item()
            ),
            "parents_below_child_opacity": int(
                (parent_opacities < child_opacities).sum().item()
            ),
        }

    @torch.no_grad()
    def _render_densify_shadow_subset(
        self,
        cameras: Sequence[object],
        child_ids: torch.Tensor,
    ) -> list[dict[str, object]]:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        if event.steps_completed < int(self.opt.densify_shadow_steps):
            raise RuntimeError("Shadow salvage requires completed probation")
        child_ids = child_ids.to(device=self.device, dtype=torch.long)
        if child_ids.numel() != torch.unique(child_ids).numel() or (
            child_ids.numel() > 0
            and not bool(torch.isin(child_ids, event.child_ids).all().item())
        ):
            raise ValueError("Shadow salvage subset contains invalid child IDs")
        pending_state = self.model.active_mask[event.child_ids].clone()
        if not bool(pending_state.all().item()):
            raise RuntimeError("Pending shadow children must be active before salvage scoring")
        self.model.active_mask[event.child_ids] = False
        if child_ids.numel() > 0:
            self.model.active_mask[child_ids] = True
        records: list[dict[str, object]] = []
        for camera in sorted(cameras, key=lambda value: str(value.image_name)):
            target = camera.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            prediction = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            error = (prediction - target).to(torch.float64)
            records.append(
                {
                    "image_name": str(camera.image_name),
                    "sse": float(error.square().sum().item()),
                    "count": int(target.numel()),
                }
            )
        self.model.active_mask[event.child_ids] = pending_state
        return records

    @torch.no_grad()
    def _select_densify_shadow_salvage(
        self,
    ) -> tuple[dict[str, object], torch.Tensor]:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        group_count = int(self.opt.densify_shadow_salvage_groups)
        expected_child_count = int(self.opt.densify_max_add)
        child_count = int(event.child_ids.numel())
        allow_partial = bool(
            getattr(
                self.opt,
                "densify_shadow_salvage_allow_partial_cohort",
                False,
            )
        )
        if allow_partial:
            if not (group_count <= child_count <= expected_child_count):
                raise RuntimeError(
                    "Shadow salvage partial proposal must contain between "
                    f"{group_count} and {expected_child_count} children, got {child_count}"
                )
        else:
            if expected_child_count <= 0 or child_count != expected_child_count:
                raise RuntimeError(
                    "Shadow salvage proposal must contain exactly densify_max_add children: "
                    f"expected {expected_child_count}, got {child_count}"
                )
            if expected_child_count % group_count != 0:
                raise RuntimeError("Shadow salvage group count must divide densify_max_add")
        groups = _balanced_densify_shadow_groups(
            event.parent_ids,
            event.child_ids,
            group_count,
        )
        group_sizes = [int(child_ids.numel()) for _, _, child_ids in groups]
        if allow_partial:
            if max(group_sizes) - min(group_sizes) > 1 or sum(group_sizes) != child_count:
                raise RuntimeError("Shadow salvage partial groups are not balanced")
        else:
            expected_group_size = expected_child_count // group_count
            if any(size != expected_group_size for size in group_sizes):
                raise RuntimeError(
                    f"Shadow salvage requires {group_count} groups of "
                    f"{expected_group_size} children"
                )
        fit_cameras = list(getattr(self, "densify_shadow_fit_cameras", []))
        if not fit_cameras:
            fit_names = getattr(self, "densify_shadow_fit_camera_names", set())
            fit_cameras = [
                camera for camera in self.train_cameras if str(camera.image_name) in fit_names
            ]
        fit_cameras.sort(key=lambda camera: str(camera.image_name))
        fit_names = [str(camera.image_name) for camera in fit_cameras]
        if not fit_names or set(fit_names) != self.densify_shadow_fit_camera_names:
            raise RuntimeError("Shadow salvage fit-camera partition is incomplete")

        selector_stride = int(
            getattr(self.opt, "densify_shadow_salvage_selector_stride", 0)
        )
        proposal_cameras = fit_cameras
        selector_cameras: list[object] = []
        if selector_stride > 0:
            proposal_cameras = list(self.densify_shadow_proposal_cameras)
            proposal_cameras.sort(key=lambda camera: str(camera.image_name))
            proposal_names = [str(camera.image_name) for camera in proposal_cameras]
            if (
                not proposal_names
                or set(proposal_names) != self.densify_shadow_proposal_camera_names
            ):
                raise RuntimeError("Shadow salvage proposal-camera partition is incomplete")
            selector_cameras = list(self.densify_shadow_selector_cameras)
            selector_cameras.sort(key=lambda camera: str(camera.image_name))
            selector_names = [str(camera.image_name) for camera in selector_cameras]
            if (
                not selector_names
                or set(selector_names) != self.densify_shadow_selector_camera_names
            ):
                raise RuntimeError("Shadow salvage selector-camera partition is incomplete")
            if set(proposal_names) | set(selector_names) != set(fit_names):
                raise RuntimeError("Shadow salvage nested camera partition is incomplete")
            if set(proposal_names) & set(selector_names):
                raise RuntimeError("Shadow salvage nested camera partition overlaps")

        empty = event.child_ids[:0]
        base = self._render_densify_shadow_subset(proposal_cameras, empty)
        full = self._render_densify_shadow_subset(proposal_cameras, event.child_ids)
        full_vs_base = _compare_densify_shadow_renders(base, full)
        group_records: list[dict[str, object]] = []
        child_ids_by_group: dict[int, torch.Tensor] = {}
        for group_id, parent_ids, child_ids in groups:
            child_ids_by_group[group_id] = child_ids
            alone = self._render_densify_shadow_subset(proposal_cameras, child_ids)
            without_ids = event.child_ids[~torch.isin(event.child_ids, child_ids)]
            without = self._render_densify_shadow_subset(proposal_cameras, without_ids)
            addition = _compare_densify_shadow_renders(base, alone)
            conditional = _compare_densify_shadow_renders(without, full)
            keep_score = (
                float(conditional["reference_sse"]) - float(conditional["candidate_sse"])
            ) / max(float(conditional["candidate_sse"]), 1e-12)
            add_score = float(addition["pooled_sse_reduction"])
            finite = bool(addition["finite"] and conditional["finite"] and math.isfinite(keep_score))
            score = min(add_score, keep_score) if finite else -1e30
            group_records.append(
                {
                    "group_id": group_id,
                    "parent_ids": parent_ids.detach().cpu().tolist(),
                    "child_ids": child_ids.detach().cpu().tolist(),
                    "add_score": float(add_score),
                    "keep_score": float(keep_score),
                    "score": float(score),
                    "eligible": bool(finite and add_score > 0.0 and keep_score > 0.0),
                    "addition": addition,
                    "conditional_keep": conditional,
                    "child_state": self._densify_shadow_child_state_telemetry(
                        parent_ids,
                        child_ids,
                    ),
                }
            )

        eligible = [record for record in group_records if bool(record["eligible"])]
        salvage_order = str(
            getattr(self.opt, "densify_shadow_salvage_order", "score")
        )
        if salvage_order == "safe_worst":
            eligible.sort(
                key=lambda record: (
                    not self._densify_shadow_metrics_pass(record["addition"]),
                    -float(record["addition"]["worst_psnr_delta"]),
                    -float(record["score"]),
                    int(record["group_id"]),
                )
            )
        else:
            eligible.sort(
                key=lambda record: (-float(record["score"]), int(record["group_id"]))
            )
        prefix_records: list[dict[str, object]] = []
        prefix_ids = event.child_ids[:0]
        prefix_groups: list[int] = []
        prefix_policy = str(
            getattr(self.opt, "densify_shadow_salvage_prefix_policy", "best_pooled")
        )
        prefix_cameras = selector_cameras if selector_stride > 0 else proposal_cameras
        prefix_base = (
            self._render_densify_shadow_subset(prefix_cameras, empty)
            if selector_stride > 0
            else base
        )
        for record in eligible:
            group_id = int(record["group_id"])
            prefix_groups.append(group_id)
            prefix_ids = torch.cat([prefix_ids, child_ids_by_group[group_id]])
            candidate = self._render_densify_shadow_subset(prefix_cameras, prefix_ids)
            metrics = _compare_densify_shadow_renders(prefix_base, candidate)
            qualifies = self._densify_shadow_metrics_pass(metrics)
            prefix_records.append(
                {
                    "group_ids": list(prefix_groups),
                    "child_ids": prefix_ids.detach().cpu().tolist(),
                    "child_count": int(prefix_ids.numel()),
                    "qualifies": qualifies,
                    "metrics": metrics,
                }
            )
            if prefix_policy == "first_qualifying" and qualifies:
                break

        qualifying = [record for record in prefix_records if bool(record["qualifies"])]
        chosen: dict[str, object] | None = None
        chosen_ids = event.child_ids[:0]
        if qualifying:
            def prefix_key(record: dict[str, object]) -> tuple[object, ...]:
                metrics = record["metrics"]
                group_ids = record["group_ids"]
                if not isinstance(metrics, Mapping) or not isinstance(group_ids, list):
                    raise RuntimeError("Shadow salvage prefix record is invalid")
                return (
                    -float(metrics["pooled_sse_reduction"]),
                    -float(metrics["mean_psnr_delta"]),
                    int(record["child_count"]),
                    tuple(int(value) for value in group_ids),
                )

            chosen = (
                qualifying[0]
                if prefix_policy == "first_qualifying"
                else min(qualifying, key=prefix_key)
            )
            chosen_ids = torch.tensor(
                chosen["child_ids"],
                device=self.device,
                dtype=torch.long,
            )
            chosen_positions = torch.isin(event.child_ids, chosen_ids)
            chosen_parent_ids = event.parent_ids[chosen_positions].detach().cpu().tolist()
            chosen_child_ids = event.child_ids[chosen_positions].detach().cpu().tolist()
            chosen["parent_ids"] = chosen_parent_ids
            chosen["parent_ids_sha256"] = _json_sha256(chosen_parent_ids)
            chosen["child_ids"] = chosen_child_ids
            chosen["child_ids_sha256"] = _json_sha256(chosen_child_ids)

        parent_ids = event.parent_ids.detach().cpu().tolist()
        child_ids = event.child_ids.detach().cpu().tolist()
        selection = {
            "schema_version": 1,
            "kind": "densify_shadow_fit_only_group_salvage_selection",
            "selector": "bidirectional_group_prefix_v1",
            "gate_metrics_computed": False,
            "birth_iteration": int(event.birth_iteration),
            "probation_steps": int(event.steps_completed),
            "fit_camera_names": fit_names,
            "fit_camera_names_sha256": _json_sha256(fit_names),
            "parent_ids": parent_ids,
            "parent_ids_sha256": _json_sha256(parent_ids),
            "child_ids": child_ids,
            "child_ids_sha256": _json_sha256(child_ids),
            "group_count": group_count,
            "group_records": group_records,
            "full_vs_base": full_vs_base,
            "prefix_records": prefix_records,
            "chosen_prefix": chosen,
            "child_state": self._densify_shadow_child_state_telemetry(
                event.parent_ids,
                event.child_ids,
            ),
        }
        if event.residual_topk_evidence is not None:
            selection["residual_topk_evidence"] = copy.deepcopy(
                event.residual_topk_evidence
            )
        if event.attributed_residual_evidence is not None:
            selection["attributed_residual_evidence"] = copy.deepcopy(
                event.attributed_residual_evidence
            )
        if salvage_order != "score":
            selection["ordering"] = salvage_order
        if prefix_policy != "best_pooled":
            selection["prefix_policy"] = prefix_policy
        if selector_stride > 0:
            proposal_names = [str(camera.image_name) for camera in proposal_cameras]
            selector_names = [str(camera.image_name) for camera in selector_cameras]
            selection["nested_selector"] = {
                "stride": selector_stride,
                "proposal_camera_names": proposal_names,
                "proposal_camera_names_sha256": _json_sha256(proposal_names),
                "selector_camera_names": selector_names,
                "selector_camera_names_sha256": _json_sha256(selector_names),
                "prefix_policy": "first_qualifying",
            }
        if allow_partial:
            selection["partial_cohort"] = {
                "enabled": True,
                "maximum_child_count": expected_child_count,
                "actual_child_count": child_count,
                "group_sizes": group_sizes,
            }
        return selection, chosen_ids

    def _write_densify_shadow_salvage_selection(
        self,
        selection: dict[str, object],
        *,
        iteration: int,
    ) -> tuple[Path, str]:
        source_names = (
            "hexels/trainer.py",
            "hexels/options.py",
            "hexels/refinement.py",
            "hexels/renderer.py",
            "hexels/edge_shift.py",
            "hexels/attributed_residual.py",
        )
        root = Path(__file__).resolve().parents[1]
        selection["decision_iteration"] = int(iteration)
        if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
            event = self.densify_shadow_event
            if event is None:
                raise RuntimeError("No shadow event is pending")
            selection["birth_envelope"] = self._densify_shadow_birth_envelope_telemetry(
                event
            )
        resume_checkpoint_path = getattr(self, "resume_checkpoint_path", None)
        selection["parent_checkpoint"] = (
            str(resume_checkpoint_path) if resume_checkpoint_path is not None else None
        )
        selection["parent_checkpoint_sha256"] = getattr(
            self,
            "resume_checkpoint_sha256",
            None,
        )
        selection["source_sha256"] = {
            name: _sha256_file(root / name) for name in source_names
        }
        selection["thresholds"] = {
            "min_pooled_sse_reduction": float(
                self.opt.densify_shadow_min_pooled_sse_reduction
            ),
            "min_mean_psnr_delta": float(self.opt.densify_shadow_min_mean_psnr_delta),
            "min_win_fraction": float(self.opt.densify_shadow_min_win_fraction),
            "min_worst_psnr_delta": float(self.opt.densify_shadow_min_worst_psnr_delta),
        }
        path = self.model_path / f"densify_shadow_salvage_selection_{iteration:06d}.json"
        payload = (json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return path, _sha256_file(path)

    @torch.no_grad()
    def _evaluate_densify_shadow_subset(self, child_ids: torch.Tensor) -> dict[str, object]:
        gate_cameras = list(self.densify_shadow_gate_cameras)
        if not gate_cameras:
            raise RuntimeError("Shadow salvage requires reserved gate cameras")
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        base = self._render_densify_shadow_subset(gate_cameras, event.child_ids[:0])
        candidate = self._render_densify_shadow_subset(gate_cameras, child_ids)
        metrics = _compare_densify_shadow_renders(base, candidate)
        per_view = metrics.pop("per_view")
        if not isinstance(per_view, list):
            raise RuntimeError("Shadow gate per-view metrics are invalid")
        metrics["accepted"] = self._densify_shadow_metrics_pass(metrics)
        metrics["decision_camera_names"] = [str(record["image_name"]) for record in base]
        metrics["worst_views"] = sorted(
            per_view,
            key=lambda record: float(record["psnr_delta"]),
        )[:5]
        return metrics

    @torch.no_grad()
    def _finish_densify_shadow_salvage_event(self, *, iteration: int) -> None:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        was_training = self.model.training
        self.model.eval()
        selection, selected_ids = self._select_densify_shadow_salvage()
        sidecar_path, sidecar_sha256 = self._write_densify_shadow_salvage_selection(
            selection,
            iteration=iteration,
        )
        defer_inner_gate = bool(
            getattr(self.opt, "densify_shadow_defer_inner_gate", False)
        )
        if selected_ids.numel() > 0:
            if defer_inner_gate:
                chosen = selection.get("chosen_prefix")
                if not isinstance(chosen, Mapping) or not isinstance(
                    chosen.get("metrics"), Mapping
                ):
                    raise RuntimeError("Deferred shadow decision requires selector metrics")
                decision = copy.deepcopy(dict(chosen["metrics"]))
                per_view = decision.pop("per_view")
                if not isinstance(per_view, list):
                    raise RuntimeError("Shadow selector per-view metrics are invalid")
                decision["accepted"] = self._densify_shadow_metrics_pass(decision)
                decision["decision_camera_names"] = sorted(
                    self.densify_shadow_selector_camera_names
                )
                decision["worst_views"] = sorted(
                    per_view,
                    key=lambda record: float(record["psnr_delta"]),
                )[:5]
                decision["selector_evaluated"] = True
                decision["inner_gate_deferred"] = True
                gate_evaluated = False
            else:
                decision = self._evaluate_densify_shadow_subset(selected_ids)
                gate_evaluated = True
        else:
            decision = {
                "finite": True,
                "accepted": False,
                "num_training_cameras": 0,
                "reference_sse": 0.0,
                "candidate_sse": 0.0,
                "pooled_sse_reduction": -1e30,
                "mean_psnr_delta": -1e30,
                "win_fraction": 0.0,
                "worst_psnr_delta": -1e30,
            }
            gate_evaluated = False
        self.model.train(was_training)

        population = int(self.model.num_base)
        child_count = int(event.child_ids.numel())
        base_population = population - child_count
        expected_ids = torch.arange(
            base_population,
            population,
            device=self.device,
            dtype=torch.long,
        )
        if not torch.equal(event.child_ids, expected_ids):
            raise RuntimeError("Shadow salvage requires a contiguous appended tail cohort")
        selected_positions = torch.isin(event.child_ids, selected_ids)
        retained_child_ids = event.child_ids[selected_positions]
        retained_parent_ids = event.parent_ids[selected_positions]
        retained_count = int(retained_child_ids.numel()) if bool(decision["accepted"]) else 0
        selected_parent_ids = event.parent_ids[selected_positions].detach().cpu().tolist()
        selected_child_ids = event.child_ids[selected_positions].detach().cpu().tolist()
        decision.update(
            {
                "birth_iteration": int(event.birth_iteration),
                "decision_iteration": int(iteration),
                "probation_steps": int(event.steps_completed),
                "parent_count": int(event.parent_ids.numel()),
                "child_count": child_count,
                "gate_evaluated": gate_evaluated,
                "selected_parent_count": len(selected_parent_ids),
                "selected_child_count": len(selected_child_ids),
                "selected_parent_ids": selected_parent_ids,
                "selected_child_ids": selected_child_ids,
                "selected_parent_ids_sha256": _json_sha256(selected_parent_ids),
                "selected_child_ids_sha256": _json_sha256(selected_child_ids),
                "retained_parent_count": retained_count,
                "retained_child_count": retained_count,
                "dropped_child_count": child_count - retained_count,
                "selection_sidecar": str(sidecar_path.relative_to(self.model_path)),
                "selection_sidecar_sha256": sidecar_sha256,
            }
        )
        if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
            decision["birth_envelope"] = self._densify_shadow_birth_envelope_telemetry(
                event
            )

        if bool(decision["accepted"]):
            if retained_count == 0:
                raise RuntimeError("Accepted shadow salvage must retain at least one child")
            dropped_child_ids = event.child_ids[~selected_positions]
            self.model.active_mask[dropped_child_ids] = False
            transfer = self._try_densify_parent_child_transfer(
                event,
                retained_parent_ids,
                retained_child_ids,
                selected_positions,
            )
            self.model.active_mask[dropped_child_ids] = True
            decision["parent_child_transfer"] = transfer
            if self.densify_has_spawned_child is not None:
                self.densify_has_spawned_child[retained_parent_ids] = True
            prune_mask = torch.zeros((population,), device=self.device, dtype=torch.bool)
            prune_mask[event.child_ids[~selected_positions]] = True
            decision["retained_parent_ids"] = retained_parent_ids.detach().cpu().tolist()
            decision["retained_child_ids_before_compaction"] = (
                retained_child_ids.detach().cpu().tolist()
            )
            decision["retained_child_ids_after_compaction"] = list(
                range(base_population, base_population + retained_count)
            )
            self.densify_shadow_event = None
            if bool(prune_mask.any().item()):
                self._prune_primitives(prune_mask)
            self.densify_shadow_counts["accepted_events"] += 1
            self.densify_shadow_counts["accepted_children"] += retained_count
            self.densify_shadow_counts["rejected_children"] += child_count - retained_count
            outcome = "accepted"
        else:
            prune_mask = torch.zeros((population,), device=self.device, dtype=torch.bool)
            prune_mask[event.child_ids] = True
            self.densify_shadow_event = None
            self._prune_primitives(prune_mask)
            self.densify_shadow_counts["rejected_events"] += 1
            self.densify_shadow_counts["rejected_children"] += child_count
            outcome = "rejected"
        self.densify_shadow_last_decision = decision
        self.densify_shadow_decisions.append(copy.deepcopy(decision))
        self.logger.info(
            "Densify shadow salvage: %s retained=%d/%d gate=%s pooled_sse=%+.4f mean=%+.4f wins=%.3f worst=%+.4f",
            outcome,
            retained_count,
            child_count,
            str(gate_evaluated),
            float(decision["pooled_sse_reduction"]),
            float(decision["mean_psnr_delta"]),
            float(decision["win_fraction"]),
            float(decision["worst_psnr_delta"]),
        )

    @torch.no_grad()
    def _evaluate_densify_shadow_event(self) -> dict[str, object]:
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        if not self.train_cameras:
            raise RuntimeError("Shadow decision requires optimizer-training cameras")
        if not bool(torch.all(self.model.active_mask[event.child_ids]).item()):
            raise RuntimeError("Pending shadow children must be active at decision time")

        was_training = self.model.training
        self.model.eval()
        full_sse = 0.0
        without_sse = 0.0
        per_view_deltas: list[float] = []
        finite = True
        gate_cameras = getattr(self, "densify_shadow_gate_cameras", [])
        decision_cameras = gate_cameras or self.train_cameras
        per_view_records: list[dict[str, object]] = []
        for camera in decision_cameras:
            target = camera.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            full = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            self.model.active_mask[event.child_ids] = False
            without = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            self.model.active_mask[event.child_ids] = True
            if not bool(torch.isfinite(full).all().item() and torch.isfinite(without).all().item()):
                finite = False
                break
            full_error = (full - target).to(torch.float64)
            without_error = (without - target).to(torch.float64)
            full_view_sse = float(full_error.square().sum().item())
            without_view_sse = float(without_error.square().sum().item())
            count = int(target.numel())
            full_sse += full_view_sse
            without_sse += without_view_sse
            full_psnr = -10.0 * math.log10(max(full_view_sse / count, 1e-12))
            without_psnr = -10.0 * math.log10(max(without_view_sse / count, 1e-12))
            delta = full_psnr - without_psnr
            per_view_deltas.append(delta)
            per_view_records.append(
                {"image_name": str(camera.image_name), "psnr_delta": float(delta)}
            )
        self.model.train(was_training)

        if finite and len(per_view_deltas) == len(decision_cameras):
            pooled_sse_reduction = (without_sse - full_sse) / max(without_sse, 1e-12)
            mean_delta = sum(per_view_deltas) / len(per_view_deltas)
            win_fraction = sum(delta > 0.0 for delta in per_view_deltas) / len(per_view_deltas)
            worst_delta = min(per_view_deltas)
            finite = all(
                math.isfinite(value)
                for value in (pooled_sse_reduction, mean_delta, win_fraction, worst_delta)
            )
        else:
            pooled_sse_reduction = -1e30
            mean_delta = -1e30
            win_fraction = 0.0
            worst_delta = -1e30

        accepted = bool(
            finite
            and pooled_sse_reduction
            >= float(self.opt.densify_shadow_min_pooled_sse_reduction)
            and mean_delta >= float(self.opt.densify_shadow_min_mean_psnr_delta)
            and win_fraction >= float(self.opt.densify_shadow_min_win_fraction)
            and worst_delta >= float(self.opt.densify_shadow_min_worst_psnr_delta)
        )
        result: dict[str, object] = {
            "finite": bool(finite),
            "accepted": accepted,
            "num_training_cameras": len(decision_cameras),
            "pooled_sse_reduction": float(pooled_sse_reduction),
            "mean_psnr_delta": float(mean_delta),
            "win_fraction": float(win_fraction),
            "worst_psnr_delta": float(worst_delta),
        }
        if gate_cameras:
            result["decision_camera_names"] = [
                str(camera.image_name) for camera in decision_cameras
            ]
            result["worst_views"] = sorted(
                per_view_records,
                key=lambda record: float(record["psnr_delta"]),
            )[:5]
        return result

    @torch.no_grad()
    def _finish_densify_shadow_event(self, *, iteration: int) -> None:
        if int(getattr(self.opt, "densify_shadow_salvage_groups", 0)) > 0:
            self._finish_densify_shadow_salvage_event(iteration=iteration)
            return
        event = self.densify_shadow_event
        if event is None:
            raise RuntimeError("No shadow event is pending")
        decision = self._evaluate_densify_shadow_event()
        child_count = int(event.child_ids.numel())
        decision.update(
            {
                "birth_iteration": int(event.birth_iteration),
                "decision_iteration": int(iteration),
                "probation_steps": int(event.steps_completed),
                "parent_count": int(event.parent_ids.numel()),
                "child_count": child_count,
            }
        )
        if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
            decision["birth_envelope"] = self._densify_shadow_birth_envelope_telemetry(
                event
            )
        if bool(decision["accepted"]):
            decision["parent_child_transfer"] = (
                self._try_densify_parent_child_transfer(
                    event,
                    event.parent_ids,
                    event.child_ids,
                )
            )
            if self.densify_has_spawned_child is not None:
                self.densify_has_spawned_child[event.parent_ids] = True
            self.densify_shadow_counts["accepted_events"] += 1
            self.densify_shadow_counts["accepted_children"] += child_count
            outcome = "accepted"
            self.densify_shadow_event = None
        else:
            population = int(self.model.num_base)
            expected_ids = torch.arange(
                population - child_count,
                population,
                device=self.device,
                dtype=torch.long,
            )
            if not torch.equal(event.child_ids, expected_ids):
                raise RuntimeError("Shadow rollback requires a contiguous appended tail cohort")
            if self.densify_shadow_child_mask is None or not bool(
                torch.all(self.densify_shadow_child_mask[event.child_ids]).item()
            ):
                raise RuntimeError("Shadow rollback child lineage is inconsistent")
            prune_mask = torch.zeros((population,), device=self.device, dtype=torch.bool)
            prune_mask[event.child_ids] = True
            self.densify_shadow_event = None
            self._prune_primitives(prune_mask)
            self.densify_shadow_counts["rejected_events"] += 1
            self.densify_shadow_counts["rejected_children"] += child_count
            outcome = "rejected"
        self.densify_shadow_last_decision = decision
        self.densify_shadow_decisions.append(copy.deepcopy(decision))
        self.logger.info(
            "Densify shadow: %s children=%d pooled_sse=%+.4f mean=%+.4f wins=%.3f worst=%+.4f",
            outcome,
            child_count,
            float(decision["pooled_sse_reduction"]),
            float(decision["mean_psnr_delta"]),
            float(decision["win_fraction"]),
            float(decision["worst_psnr_delta"]),
        )

    @torch.no_grad()
    def _commit_densify_parent_child_transfer(
        self,
        event: _DensifyShadowEvent,
        parent_ids: torch.Tensor,
        child_ids: torch.Tensor,
        selected_positions: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        fraction = float(
            getattr(self.opt, "densify_accept_parent_transfer_fraction", 0.0)
        )
        if fraction == 0.0:
            return {"fraction": 0.0, "applied": False}
        if selected_positions is None:
            expected_parent_ids = event.parent_ids
            expected_child_ids = event.child_ids
        else:
            expected_parent_ids = event.parent_ids[selected_positions]
            expected_child_ids = event.child_ids[selected_positions]
        if (
            parent_ids.ndim != 1
            or parent_ids.shape != child_ids.shape
            or not torch.equal(parent_ids, expected_parent_ids)
            or not torch.equal(child_ids, expected_child_ids)
        ):
            raise RuntimeError("Accepted parent-child transfer lineage is inconsistent")

        parent_logits, child_logits = redistribute_parent_child_opacity(
            self.model.opacity_logits[parent_ids],
            self.model.opacity_logits[child_ids],
            torch.arange(parent_ids.numel(), device=self.device, dtype=torch.long),
            transfer_fraction=fraction,
        )
        self.model.opacity_logits[parent_ids] = parent_logits
        self.model.opacity_logits[child_ids] = child_logits
        changed_ids = torch.cat((parent_ids, child_ids))
        optimizer_state = self.optimizer.state.get(self.model.opacity_logits)
        if optimizer_state:
            for value in optimizer_state.values():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == self.model.num_base
                ):
                    value[changed_ids] = 0
        return {
            "fraction": fraction,
            "applied": True,
            "parent_count": int(parent_ids.numel()),
            "child_count": int(child_ids.numel()),
        }

    @torch.no_grad()
    def _try_densify_parent_child_transfer(
        self,
        event: _DensifyShadowEvent,
        parent_ids: torch.Tensor,
        child_ids: torch.Tensor,
        selected_positions: Optional[torch.Tensor] = None,
    ) -> dict[str, object]:
        fraction = float(
            getattr(self.opt, "densify_accept_parent_transfer_fraction", 0.0)
        )
        if fraction == 0.0:
            return {"fraction": 0.0, "applied": False}
        cameras = list(self.densify_shadow_gate_cameras) or list(self.train_cameras)
        was_training = self.model.training
        self.model.eval()
        baseline_sse = []
        pixel_counts = []
        for camera in cameras:
            target = camera.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            image = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            baseline_sse.append(float((image - target).to(torch.float64).square().sum()))
            pixel_counts.append(int(target.numel()))

        changed_ids = torch.cat((parent_ids, child_ids))
        original_logits = self.model.opacity_logits[changed_ids].detach().clone()
        optimizer_state = self.optimizer.state.get(self.model.opacity_logits, {})
        original_optimizer_rows = {
            name: value[changed_ids].detach().clone()
            for name, value in optimizer_state.items()
            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == self.model.num_base
            )
        }
        result = self._commit_densify_parent_child_transfer(
            event,
            parent_ids,
            child_ids,
            selected_positions,
        )
        candidate_sse = []
        for camera in cameras:
            target = camera.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            image = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            candidate_sse.append(float((image - target).to(torch.float64).square().sum()))
        self.model.train(was_training)

        psnr_deltas = [
            -10.0 * math.log10(max(candidate / count, 1e-12))
            + 10.0 * math.log10(max(baseline / count, 1e-12))
            for baseline, candidate, count in zip(
                baseline_sse,
                candidate_sse,
                pixel_counts,
            )
        ]
        pooled_reduction = (
            sum(baseline_sse) - sum(candidate_sse)
        ) / max(sum(baseline_sse), 1e-12)
        mean_delta = sum(psnr_deltas) / len(psnr_deltas)
        worst_delta = min(psnr_deltas)
        accepted = (
            pooled_reduction
            >= float(self.opt.densify_accept_transfer_min_pooled_sse_reduction)
            and mean_delta
            >= float(self.opt.densify_accept_transfer_min_mean_psnr_delta)
            and worst_delta
            >= float(self.opt.densify_accept_transfer_min_worst_psnr_delta)
        )
        result.update(
            {
                "accepted": bool(accepted),
                "camera_count": len(cameras),
                "pooled_sse_reduction": float(pooled_reduction),
                "mean_psnr_delta": float(mean_delta),
                "worst_psnr_delta": float(worst_delta),
            }
        )
        if not accepted:
            self.model.opacity_logits[changed_ids] = original_logits
            for name, rows in original_optimizer_rows.items():
                optimizer_state[name][changed_ids] = rows
            result["applied"] = False
        return result

    def _step_optimizer_and_density(
        self,
        *,
        iteration: int,
        run_densify: bool,
        size_threshold: int,
        reset_opacity: bool,
    ) -> None:
        shadow_pending = getattr(self, "densify_shadow_event", None) is not None
        joint_pending = getattr(self, "densify_joint_coplanar_event", None) is not None
        if shadow_pending and joint_pending:
            raise RuntimeError("Shadow and joint coplanar probation cannot overlap")
        if shadow_pending and (run_densify or reset_opacity):
            raise RuntimeError("Topology mutation is blocked during shadow probation")
        if joint_pending and (run_densify or reset_opacity):
            raise RuntimeError("Topology mutation is blocked during joint coplanar probation")
        if joint_pending:
            event = self.densify_joint_coplanar_event
            if event is None:
                raise RuntimeError(
                    "Joint coplanar event disappeared before probation constraints"
                )
            expected_iteration = event.birth_iteration + event.steps_completed + 1
            if iteration != expected_iteration:
                raise RuntimeError(
                    "Joint coplanar probation iterations must be contiguous: "
                    f"expected {expected_iteration}, got {iteration}"
                )
            self._constrain_densify_joint_coplanar_gradients()
        frozen_ids = getattr(self, "densify_shadow_frozen_step_ids", None)
        self._freeze_geometry_gradients(iteration)
        self._freeze_resume_parameter_gradients()
        frozen_rows: list[
            tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
        ] = []
        if frozen_ids is not None:
            population = int(self.model.num_base)
            for group in self.optimizer.param_groups:
                parameter = group["params"][0]
                if parameter.ndim == 0 or parameter.shape[0] != population:
                    continue
                state = self.optimizer.state.get(parameter, {})
                state_rows = {
                    key: value.index_select(0, frozen_ids).detach().clone()
                    for key, value in state.items()
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == population
                }
                frozen_rows.append(
                    (
                        parameter,
                        parameter.detach().index_select(0, frozen_ids).clone(),
                        state_rows,
                    )
                )
        self.optimizer.step()
        if frozen_ids is not None:
            with torch.no_grad():
                for parameter, parameter_rows, state_rows in frozen_rows:
                    parameter.index_copy_(0, frozen_ids, parameter_rows)
                    state = self.optimizer.state[parameter]
                    for key, rows in state_rows.items():
                        state[key].index_copy_(0, frozen_ids, rows)
            self.densify_shadow_frozen_step_ids = None
        self._project_resume_isotropic_scales()
        self._project_resume_normal_trust(iteration)
        if shadow_pending:
            event = self.densify_shadow_event
            if event is None:
                raise RuntimeError("Shadow event disappeared before probation accounting")
            if (
                str(
                    getattr(
                        self.opt,
                        "densify_shadow_child_probation",
                        "unrestricted",
                    )
                )
                == "coplanar_spatial"
            ):
                self._project_densify_shadow_child_probation()
            if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
                self._project_densify_shadow_birth_envelope()
            event.steps_completed += 1
            if event.steps_completed >= int(self.opt.densify_shadow_steps):
                self._finish_densify_shadow_event(iteration=iteration)
        if joint_pending:
            event = self.densify_joint_coplanar_event
            if event is None:
                raise RuntimeError(
                    "Joint coplanar event disappeared before probation accounting"
                )
            self._project_densify_joint_coplanar_rows()
            if event.first_constrained_iteration is None:
                event.first_constrained_iteration = int(iteration)
            event.last_constrained_iteration = int(iteration)
            event.steps_completed += 1
            if event.steps_completed >= event.configured_steps:
                self._finish_densify_joint_coplanar_event()
        if run_densify:
            self._densify_and_prune(size_threshold=size_threshold, iteration=iteration)
        if reset_opacity:
            self._reset_opacity()

    def _transfer_densified_appearance(self, rows: dict[str, torch.Tensor], parents: torch.Tensor) -> None:
        if not bool(getattr(self.opt, "densify_preserve_linear_field", False)):
            return
        rows["colors"], rows["color_grads"] = transferred_linear_appearance(
            self.model.means3d.detach()[parents], self.model.scales.detach()[parents],
            self.model.rotations.detach()[parents], self.model.colors.detach()[parents],
            self.model.color_grads.detach()[parents], rows["means3d"], torch.exp(rows["log_scales"]),
            scale_modifier=float(self.opt.scale_modifier),
        )

    @torch.no_grad()
    def _densify_and_prune_2dgs(self, *, size_threshold: int) -> None:
        if (
            self.densify_grad_accum is None
            or self.densify_denom is None
            or self.densify_max_radii2d is None
        ):
            return

        grads = self.densify_grad_accum / self.densify_denom
        grads = grads.nan_to_num(0.0).squeeze(-1)

        grad_thr = float(getattr(self.opt, "densify_grad_threshold", 0.0002))
        extent = float(getattr(self, "scene_extent", 1.0))
        percent_dense = float(getattr(self.opt, "percent_dense", 0.01))
        scale_max = self.model.scales.detach().max(dim=1).values

        selected = grads >= grad_thr
        clone_mask = selected & (scale_max <= percent_dense * extent)
        split_mask = selected & (scale_max > percent_dense * extent)

        if clone_mask.any():
            idx = clone_mask.nonzero(as_tuple=True)[0].to(dtype=torch.long)
            new_tensors = {
                "means3d": self.model.means3d.detach()[idx].clone(),
                "log_scales": self.model.log_scales.data[idx].clone(),
                "rotation_quats": self.model.rotation_quats.data[idx].clone(),
                "colors": self.model.colors.data[idx].clone(),
                "color_grads": self.model.color_grads.data[idx].clone(),
                "color_quads": self.model.color_quads.data[idx].clone(),
                "view_grads": self.model.view_grads.data[idx].clone(),
                "specular_ks": self.model.specular_ks.data[idx].clone(),
                "specular_log_s": self.model.specular_log_s.data[idx].clone(),
                "opacity_logits": self.model.opacity_logits.data[idx].clone(),
                "log_softness": self.model.log_softness.data[idx].clone(),
            }
            self._transfer_densified_appearance(new_tensors, idx)
            self._append_primitives(new_tensors)

        if split_mask.any():
            idx = split_mask.nonzero(as_tuple=True)[0].to(dtype=torch.long)
            N = 2
            eps = torch.finfo(self.model.opacity_logits.dtype).eps

            scales = self.model.scales.detach()[idx]
            means = self.model.means3d.detach()[idx]
            rots = self.model.rotations.detach()[idx]
            tx, ty = _quat_to_tangent_xy(rots)

            std = scales.repeat_interleave(N, dim=0)
            means_rep = means.repeat_interleave(N, dim=0)
            tx_rep = tx.repeat_interleave(N, dim=0)
            ty_rep = ty.repeat_interleave(N, dim=0)
            duv = torch.randn((std.shape[0], 2), device=self.device, dtype=means.dtype) * std
            offset = tx_rep * duv[:, 0:1] + ty_rep * duv[:, 1:2]
            new_means = means_rep + offset

            new_scales = std / (0.8 * float(N))
            new_log_scales = torch.log(new_scales.clamp_min(eps))

            rep = lambda x: x[idx].repeat_interleave(N, dim=0).clone()
            new_tensors = {
                "means3d": new_means,
                "log_scales": new_log_scales,
                "rotation_quats": rep(self.model.rotation_quats.data),
                "colors": rep(self.model.colors.data),
                "color_grads": rep(self.model.color_grads.data),
                "color_quads": rep(self.model.color_quads.data),
                "view_grads": rep(self.model.view_grads.data),
                "specular_ks": rep(self.model.specular_ks.data),
                "specular_log_s": rep(self.model.specular_log_s.data),
                "opacity_logits": rep(self.model.opacity_logits.data),
                "log_softness": rep(self.model.log_softness.data),
            }
            self._transfer_densified_appearance(new_tensors, idx.repeat_interleave(N))
            self._append_primitives(new_tensors)

            prune_split = torch.zeros((int(self.model.num_base),), device=self.device, dtype=torch.bool)
            prune_split[idx] = True
            self._prune_primitives(prune_split)

        prune_mask = (self.model.opacities.detach() < float(getattr(self.opt, "opacity_cull", 0.05))).to(torch.bool)
        if size_threshold:
            big_points_vs = self.densify_max_radii2d > float(size_threshold)
            big_points_ws = self.model.scales.detach().max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if prune_mask.any():
            self._prune_primitives(prune_mask)

        self._reset_densify_stats()

    @torch.no_grad()
    def _bisplit_primitives(self, parent_ids: torch.Tensor, *, iteration: int) -> None:
        split_axes = None
        if self.densify_local_grad_abs_accum is not None:
            split_axes = self.densify_local_grad_abs_accum[parent_ids].argmax(dim=1).to(dtype=torch.long)
        children = symmetric_split_children(
            self.model.state_dict(),
            parent_ids,
            offset_fraction=float(getattr(self.opt, "densify_split_offset_fraction", 0.25)),
            scale_ratio=float(getattr(self.opt, "densify_split_scale_ratio", 0.6)),
            split_axes=split_axes,
            scale_modifier=float(getattr(self.opt, "scale_modifier", 1.0)),
            split_opacity=bool(getattr(self.opt, "densify_opacity_split", True)),
        )
        negative = {name: values[0::2] for name, values in children.items()}
        positive = {name: values[1::2] for name, values in children.items()}
        self._replace_primitive_rows(parent_ids, negative, birth_iteration=iteration)
        self._append_primitives(positive, birth_iteration=iteration)

    @torch.no_grad()
    def _activate_or_append_primitives(
        self,
        new_tensors: dict[str, torch.Tensor],
        *,
        birth_iteration: int,
    ) -> torch.Tensor:
        count = int(new_tensors["means3d"].shape[0])
        old_population = int(self.model.num_base)
        inactive_ids = torch.nonzero(~self.model.active_mask, as_tuple=True)[0]
        reused = min(count, int(inactive_ids.numel()))
        child_ids = torch.cat(
            (
                inactive_ids[:reused],
                torch.arange(
                    old_population,
                    old_population + count - reused,
                    device=self.device,
                    dtype=torch.long,
                ),
            )
        )
        if reused > 0:
            destination_ids = inactive_ids[:reused]
            rows = {name: values[:reused] for name, values in new_tensors.items()}
            self._replace_primitive_rows(
                destination_ids,
                rows,
                birth_iteration=birth_iteration,
            )
            self.model.active_mask[destination_ids] = True
        if reused < count:
            rows = {name: values[reused:] for name, values in new_tensors.items()}
            self._append_primitives(rows, birth_iteration=birth_iteration)
        return child_ids

    @torch.no_grad()
    def _deactivate_primitives(self, deactivate_mask: torch.Tensor) -> None:
        if getattr(self, "densify_shadow_child_mask", None) is not None:
            raise RuntimeError("Shadow densification is append-only and does not support inactive rows")
        if getattr(self, "densify_joint_coplanar_event", None) is not None:
            raise RuntimeError(
                "Deactivation is blocked while joint coplanar probation is pending"
            )
        deactivate_mask = deactivate_mask.to(device=self.device, dtype=torch.bool)
        deactivate_mask &= self.model.active_mask
        count = int(deactivate_mask.sum().item())
        if count == 0:
            return
        if count == self.model.num_active_base:
            raise ValueError("Pruning deactivated every base primitive")
        primitive_ids = torch.nonzero(deactivate_mask, as_tuple=True)[0]
        population = int(self.model.num_base)
        for group in self.optimizer.param_groups:
            parameter = group["params"][0]
            state = self.optimizer.state.get(parameter)
            if not state:
                continue
            for value in state.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == population:
                    value[primitive_ids] = 0
        self.model.active_mask[primitive_ids] = False
        for buffer in (
            self.densify_grad_accum,
            self.densify_denom,
            self.densify_max_radii2d,
            self.densify_local_grad_abs_accum,
            self.densify_local_grad_signed_accum,
            self.densify_residual_mass_accum,
            self.densify_centroid_mass_accum,
            self.densify_direction_vote_accum,
            self.densify_moment_vote_count,
            self.densify_mv_edge_top_scores,
            self.densify_mv_edge_top_centroids,
            self.densify_mv_edge_vote_counts,
            self.densify_attributed_edge_mass,
            self.densify_attributed_edge_centroid_mass,
            self.densify_attributed_edge_vote_counts,
            self.densify_covariance_coverage,
            self.densify_covariance_energy,
            self.densify_covariance_robust_weight,
            self.densify_covariance_first_moment,
            self.densify_covariance_second_moment,
            self.densify_covariance_double_angle,
            self.densify_covariance_side_mass,
            self.densify_covariance_material_view_count,
            self.densify_covariance_axis_view_count,
            self.densify_contrib_accum,
        ):
            if buffer is not None:
                buffer[primitive_ids] = 0
        self.repulsion_knn_idx = None
        self.repulsion_knn_P = 0
        self.surface_knn_idx = None
        self.surface_knn_P = 0
        self._restart_prune_observation()
        self.logger.info(
            "Densify: deactivated %d primitives (active=%d capacity=%d)",
            count,
            self.model.num_active_base,
            self.model.num_base,
        )

    @torch.no_grad()
    def _covariance_child_primitives(
        self,
        parent_ids: torch.Tensor,
        *,
        iteration: int,
        eligible_candidate_count: int,
    ) -> None:
        if (
            int(getattr(self.opt, "densify_joint_coplanar_steps", 0)) > 0
            and self.densify_joint_coplanar_event is not None
        ):
            raise RuntimeError("Only one joint coplanar densification event may be pending")
        summary = self._densify_covariance_summary()
        parents_cpu = parent_ids.detach().to(device="cpu", dtype=torch.long)
        children_per_parent = int(
            getattr(self.opt, "densify_covariance_children_per_parent", 2)
        )
        centers = summary.center[parents_cpu]
        pair_offsets = summary.pair_offsets[parents_cpu]
        if children_per_parent == 1:
            child_offsets = centers
            child_parent_ids = parent_ids
            children = directional_edge_children(
                self.model.state_dict(),
                parent_ids,
                local_offsets=child_offsets,
                scale_modifier=float(getattr(self.opt, "scale_modifier", 1.0)),
                scale_ratio=float(getattr(self.opt, "densify_child_scale_ratio", 0.5)),
                init_opacity=float(getattr(self.opt, "densify_child_init_opacity", 0.01)),
                extent_cap_fraction=float(
                    getattr(self.opt, "densify_child_extent_cap_fraction", 0.0)
                ),
                parent_opacity_fraction=float(
                    getattr(self.opt, "densify_child_parent_opacity_fraction", 0.0)
                ),
                scene_extent=float(self.scene_extent),
            )
        else:
            child_offsets = pair_offsets.reshape(-1, 2)
            child_parent_ids = parent_ids.repeat_interleave(2)
            children = covariance_pair_children(
                self.model.state_dict(),
                parent_ids,
                local_offsets=pair_offsets,
                scale_modifier=float(getattr(self.opt, "scale_modifier", 1.0)),
                scale_ratio=float(getattr(self.opt, "densify_child_scale_ratio", 0.5)),
                init_opacity=float(getattr(self.opt, "densify_child_init_opacity", 0.01)),
                extent_cap_fraction=float(
                    getattr(self.opt, "densify_child_extent_cap_fraction", 0.0)
                ),
                parent_opacity_fraction=float(
                    getattr(self.opt, "densify_child_parent_opacity_fraction", 0.0)
                ),
                scene_extent=float(self.scene_extent),
            )
        self._reorient_densify_children(children, child_parent_ids)
        children["log_softness"].add_(
            math.log(float(getattr(self.opt, "densify_child_softness_ratio", 1.0)))
        )

        old_population = int(self.model.num_base)
        active_before = int(self.model.num_active_base)
        inactive_ids = torch.nonzero(~self.model.active_mask, as_tuple=True)[0]
        child_count = int(child_parent_ids.numel())
        reused = min(child_count, int(inactive_ids.numel()))
        child_ids = self._activate_or_append_primitives(
            children,
            birth_iteration=iteration,
        )
        if self.densify_has_spawned_child is not None:
            self.densify_has_spawned_child[parent_ids] = True
        self._start_densify_joint_coplanar_event(
            child_parent_ids,
            child_ids,
            iteration=iteration,
        )

        selected = {
            "coverage": summary.coverage[parents_cpu].tolist(),
            "energy": summary.energy[parents_cpu].tolist(),
            "robust_weight": summary.robust_weight[parents_cpu].tolist(),
            "score": summary.score[parents_cpu].tolist(),
            "mean": summary.mean[parents_cpu].tolist(),
            "covariance": summary.covariance[parents_cpu].tolist(),
            "principal_axis": summary.principal_axis[parents_cpu].tolist(),
            "lambda1": summary.lambda1[parents_cpu].tolist(),
            "lambda2": summary.lambda2[parents_cpu].tolist(),
            "anisotropy": summary.anisotropy[parents_cpu].tolist(),
            "axis_view_count": summary.axis_view_count[parents_cpu].tolist(),
            "axis_coherence": summary.axis_coherence[parents_cpu].tolist(),
            "material_view_count": summary.material_view_count[parents_cpu].tolist(),
            "side_mass": summary.side_mass[parents_cpu].tolist(),
            "side_balance": summary.side_balance[parents_cpu].tolist(),
            "center": centers.tolist(),
            "delta": summary.delta[parents_cpu].tolist(),
            "pair_offsets": pair_offsets.tolist(),
        }
        selection = {
            "parent_ids": parents_cpu.tolist(),
            "child_parent_ids": child_parent_ids.detach().cpu().tolist(),
            "child_ids": child_ids.detach().cpu().tolist(),
            "child_local_offsets": child_offsets.tolist(),
        }
        source_paths = {
            "hexels/trainer.py": Path(__file__),
            "hexels/options.py": Path(__file__).with_name("options.py"),
            "hexels/attributed_residual.py": Path(__file__).with_name(
                "attributed_residual.py"
            ),
            "hexels/edge_shift.py": Path(__file__).with_name("edge_shift.py"),
            "hexels/refinement.py": Path(__file__).with_name("refinement.py"),
        }
        record = {
            "schema": "hexels3d_covariance_birth_v1",
            "iteration": int(iteration),
            "direction_signal": "attributed_covariance",
            "children_per_parent": children_per_parent,
            "weak_support_power": int(
                getattr(self.opt, "densify_attributed_weak_power", 0)
            ),
            "score_alpha": 0.7,
            "thresholds": {
                "min_views": int(getattr(self.opt, "densify_covariance_min_views", 8)),
                "min_anisotropy": float(
                    getattr(self.opt, "densify_covariance_min_anisotropy", 0.15)
                ),
                "min_axis_coherence": float(
                    getattr(self.opt, "densify_covariance_min_axis_coherence", 0.50)
                ),
                "min_side_balance": float(
                    getattr(self.opt, "densify_covariance_min_side_balance", 0.30)
                ),
                "min_delta": float(
                    getattr(self.opt, "densify_covariance_min_delta", 0.10)
                ),
            },
            "observed_camera_count": len(self.densify_moment_camera_names),
            "observed_camera_names": sorted(self.densify_moment_camera_names),
            "eligible_candidate_count": int(eligible_candidate_count),
            "selected_parent_count": int(parent_ids.numel()),
            "actual_child_count": child_count,
            "population": {
                "active_before": active_before,
                "active_after": int(self.model.num_active_base),
                "capacity_before": old_population,
                "capacity_after": int(self.model.num_base),
                "reused_inactive_rows": reused,
                "appended_rows": child_count - reused,
            },
            **selection,
            "selected_evidence": selected,
            "selection_sha256": _json_sha256(selection),
            "evidence_sha256": _json_sha256(selected),
            "source_hashes": {
                name: _sha256_file(path) for name, path in source_paths.items()
            },
        }
        sidecar = Path(str(self.opt.model_path)) / (
            f"densify_covariance_birth_{iteration:06d}.json"
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    @torch.no_grad()
    def _edge_child_primitives(self, parent_ids: torch.Tensor, *, iteration: int) -> None:
        if (
            int(getattr(self.opt, "densify_joint_coplanar_steps", 0)) > 0
            and self.densify_joint_coplanar_event is not None
        ):
            raise RuntimeError("Only one joint coplanar densification event may be pending")
        direction_signal = str(getattr(self.opt, "densify_direction_signal", "position"))
        topk_evidence: dict[str, object] | None = None
        attributed_evidence: dict[str, object] | None = None
        if direction_signal == "residual_moment":
            residual_topk = int(getattr(self.opt, "densify_residual_topk_views", 0))
            if residual_topk > 0:
                if (
                    self.densify_mv_edge_top_scores is None
                    or self.densify_mv_edge_top_centroids is None
                    or self.densify_mv_edge_vote_counts is None
                ):
                    raise RuntimeError("Multi-view residual top-K buffers are not initialized")
                summary = summarize_residual_topk_views(
                    self.densify_mv_edge_top_scores,
                    self.densify_mv_edge_top_centroids,
                    self.densify_mv_edge_vote_counts,
                )
                local_offsets = summary.centroids[parent_ids]
                selected_radius = torch.max(
                    local_offsets
                    @ canonical_edge_normals(
                        device=local_offsets.device,
                        dtype=local_offsets.dtype,
                    ).T,
                    dim=1,
                ).values
                topk_evidence = {
                    "k": residual_topk,
                    "observed_camera_count": len(self.densify_moment_camera_names),
                    "dominant_edge_ids": summary.dominant_edge_ids[parent_ids]
                    .detach()
                    .cpu()
                    .tolist(),
                    "view_counts": summary.view_counts[parent_ids].detach().cpu().tolist(),
                    "dominant_edge_vote_counts": summary.dominant_edge_vote_counts[
                        parent_ids
                    ]
                    .detach()
                    .cpu()
                    .tolist(),
                    "consistency": summary.consistency[parent_ids]
                    .detach()
                    .cpu()
                    .tolist(),
                    "kth_scores": summary.kth_scores[parent_ids]
                    .detach()
                    .cpu()
                    .tolist(),
                    "centroids": local_offsets.detach().cpu().tolist(),
                    "canonical_radii": selected_radius.detach().cpu().tolist(),
                }
            else:
                if (
                    self.densify_residual_mass_accum is None
                    or self.densify_centroid_mass_accum is None
                ):
                    raise RuntimeError("Residual-moment centroid buffers are not initialized")
                local_offsets = self.densify_centroid_mass_accum[
                    parent_ids
                ] / self.densify_residual_mass_accum[parent_ids].clamp_min(1e-12)[:, None]
            edge_normals = canonical_edge_normals(
                device=local_offsets.device,
                dtype=local_offsets.dtype,
            )
            radius = torch.max(local_offsets @ edge_normals.T, dim=1).values
            local_offsets = local_offsets / torch.maximum(radius, torch.ones_like(radius))[:, None]
        elif direction_signal == "attributed_residual":
            if (
                self.densify_attributed_edge_mass is None
                or self.densify_attributed_edge_centroid_mass is None
                or self.densify_attributed_edge_vote_counts is None
            ):
                raise RuntimeError("Attributed residual buffers are not initialized")
            summary = summarize_attributed_edge_evidence(
                self.densify_attributed_edge_mass,
                self.densify_attributed_edge_centroid_mass,
                self.densify_attributed_edge_vote_counts,
            )
            local_offsets = summary.centroids[parent_ids]
            edge_normals = canonical_edge_normals(
                device=local_offsets.device,
                dtype=local_offsets.dtype,
            )
            selected_radius = torch.max(
                local_offsets @ edge_normals.T,
                dim=1,
            ).values
            attributed_evidence = {
                "weak_support_power": int(
                    getattr(self.opt, "densify_attributed_weak_power", 0)
                ),
                "observed_camera_count": len(self.densify_moment_camera_names),
                "dominant_edge_ids": summary.dominant_edge_ids[parent_ids]
                .detach()
                .cpu()
                .tolist(),
                "view_counts": summary.view_counts[parent_ids].detach().cpu().tolist(),
                "dominant_edge_vote_counts": summary.dominant_edge_vote_counts[
                    parent_ids
                ]
                .detach()
                .cpu()
                .tolist(),
                "consistency": summary.consistency[parent_ids]
                .detach()
                .cpu()
                .tolist(),
                "mass": summary.mass[parent_ids].detach().cpu().tolist(),
                "centroids": local_offsets.detach().cpu().tolist(),
                "canonical_radii": selected_radius.detach().cpu().tolist(),
            }
            local_offsets = local_offsets / torch.maximum(
                selected_radius,
                torch.ones_like(selected_radius),
            )[:, None]
        else:
            if self.densify_local_grad_abs_accum is None or self.densify_local_grad_signed_accum is None:
                raise RuntimeError("Directional child buffers are not initialized")
            split_axes = self.densify_local_grad_abs_accum[parent_ids].argmax(dim=1).to(dtype=torch.long)
            batch = torch.arange(parent_ids.numel(), device=self.device)
            signed_gradient = self.densify_local_grad_signed_accum[parent_ids][batch, split_axes]
            signs = torch.where(
                signed_gradient >= 0.0,
                -torch.ones_like(signed_gradient),
                torch.ones_like(signed_gradient),
            )
            local_offsets = torch.nn.functional.one_hot(split_axes, num_classes=2).to(
                dtype=self.model.means3d.dtype
            )
            local_offsets *= signs[:, None]
        local_offsets *= float(getattr(self.opt, "densify_child_offset_fraction", 0.75))
        children = directional_edge_children(
            self.model.state_dict(),
            parent_ids,
            local_offsets=local_offsets,
            scale_modifier=float(getattr(self.opt, "scale_modifier", 1.0)),
            scale_ratio=float(getattr(self.opt, "densify_child_scale_ratio", 0.5)),
            init_opacity=float(getattr(self.opt, "densify_child_init_opacity", 0.01)),
            extent_cap_fraction=float(
                getattr(self.opt, "densify_child_extent_cap_fraction", 0.0)
            ),
            parent_opacity_fraction=float(
                getattr(self.opt, "densify_child_parent_opacity_fraction", 0.0)
            ),
            scene_extent=float(self.scene_extent),
        )
        self._reorient_densify_children(children, parent_ids)
        children["log_softness"].add_(
            math.log(float(getattr(self.opt, "densify_child_softness_ratio", 1.0)))
        )
        shadow_steps = int(getattr(self.opt, "densify_shadow_steps", 0))
        if shadow_steps > 0:
            if self.densify_shadow_event is not None:
                raise RuntimeError("Only one shadow densification event may be pending")
            if self.densify_shadow_child_mask is None:
                raise RuntimeError("Shadow child state is not initialized")
            if self.densify_shadow_attempted_parent_mask is None:
                raise RuntimeError("Shadow attempt state is not initialized")
            if self.model.num_active_base != self.model.num_base:
                raise RuntimeError("Shadow densification requires append-only active capacity")
            old_population = int(self.model.num_base)
            self._append_primitives(children, birth_iteration=iteration)
            child_ids = torch.arange(
                old_population,
                int(self.model.num_base),
                device=self.device,
                dtype=torch.long,
            )
            self.densify_shadow_child_mask[child_ids] = True
            self.densify_shadow_attempted_parent_mask[parent_ids] = True
            event = _DensifyShadowEvent(
                parent_ids=parent_ids.detach().clone(),
                child_ids=child_ids,
                birth_iteration=int(iteration),
                residual_topk_evidence=topk_evidence,
                attributed_residual_evidence=attributed_evidence,
            )
            if bool(getattr(self.opt, "densify_shadow_enforce_birth_envelope", False)):
                scale_modifier = float(getattr(self.opt, "scale_modifier", 1.0))
                event.birth_parent_distances = torch.linalg.vector_norm(
                    self.model.means3d[child_ids] - self.model.means3d[parent_ids],
                    dim=1,
                ).detach().clone()
                event.birth_physical_max_scales = (
                    scale_modifier * self.model.scales[child_ids].max(dim=1).values
                ).detach().clone()
                event.birth_log_opacities = (
                    -torch.nn.functional.softplus(-self.model.opacity_logits[child_ids])
                ).detach().clone()
            self.densify_shadow_event = event
            self.densify_shadow_counts["proposed_events"] += 1
            self.densify_shadow_counts["proposed_children"] += int(child_ids.numel())
        else:
            child_ids = self._activate_or_append_primitives(
                children,
                birth_iteration=iteration,
            )
            if self.densify_has_spawned_child is not None:
                self.densify_has_spawned_child[parent_ids] = True
            self._start_densify_joint_coplanar_event(
                parent_ids,
                child_ids,
                iteration=iteration,
            )

    @torch.no_grad()
    def _prune_coordinated_primitives(self, *, iteration: int) -> None:
        """Measure current all-view contribution before recycling a bounded population."""
        interval = int(self.opt.densify_prune_interval)
        if iteration <= 0 or iteration % interval != 0:
            return
        started = time.perf_counter()
        population = self.model.num_base
        scores = torch.zeros(population, device=self.device)
        max_view_threshold = float(getattr(self.opt, "densify_prune_max_view_threshold", -1.0))
        maxima = torch.zeros_like(scores) if max_view_threshold >= 0 else None
        for camera in self.train_cameras:
            result = self.renderer.render(camera, self._render_model(), self.bg_color, return_contrib=True)
            pixels = camera.image_height * camera.image_width
            contribution = result["contrib"][:population] / pixels
            scores += contribution
            if maxima is not None:
                torch.maximum(maxima, contribution, out=maxima)
        scores /= len(self.train_cameras)
        eligible = (iteration - self.densify_birth_iteration) >= int(self.opt.densify_newborn_grace)
        eligible &= scores <= float(self.opt.densify_prune_mean_threshold)
        protected = 0
        if maxima is not None:
            protected = int(torch.count_nonzero(eligible & (maxima > max_view_threshold)))
            eligible &= maxima <= max_view_threshold
        candidates = torch.nonzero(eligible, as_tuple=True)[0]
        limit = min(int(candidates.numel()), int(population * self.opt.densify_prune_max_fraction))
        if limit:
            selected = candidates[torch.argsort(scores[candidates])[:limit]]
            prune = torch.zeros(population, device=self.device, dtype=torch.bool)
            prune[selected] = True
            self._prune_primitives(prune)
        record = dict(iteration=iteration, before=population, pruned=limit,
                      eligible=int(candidates.numel()), views=len(self.train_cameras),
                      max_view_protected=protected,
                      seconds=time.perf_counter() - started)
        self.densify_prune_audit.append(record)
        self.logger.info("Coordinated prune: %s", record)

    @torch.no_grad()
    def _prune_directional_primitives(self, *, iteration: int) -> None:
        prune_interval = int(getattr(self.opt, "densify_prune_interval", 1000))
        if prune_interval <= 0 or iteration <= 0 or iteration % prune_interval != 0:
            return
        if self.densify_contrib_accum is None or self.densify_birth_iteration is None:
            raise RuntimeError("Directional pruning buffers are not initialized")
        expected_cameras = {str(camera.image_name) for camera in self.train_cameras}
        if bool(getattr(self.opt, "densify_prune_full_sweep", False)):
            self._collect_full_prune_contribution()
        if self.densify_contrib_camera_names != expected_cameras:
            self.logger.info(
                "Densify: prune deferred until a complete camera sweep (%d/%d)",
                len(self.densify_contrib_camera_names),
                len(expected_cameras),
            )
            return

        age = iteration - self.densify_birth_iteration
        grace = int(getattr(self.opt, "densify_newborn_grace", 500))
        threshold = float(getattr(self.opt, "densify_prune_contrib_threshold", 0.0))
        prune_mask = self.model.active_mask & (age >= grace) & (self.densify_contrib_accum <= threshold)
        if bool(prune_mask.any().item()):
            if str(getattr(self.opt, "densify_action", "mixed")) == "edge_child":
                self._deactivate_primitives(prune_mask)
            else:
                self._prune_primitives(prune_mask)
        self._restart_prune_observation()

    @torch.no_grad()
    def _merge_compatible_primitives(self, *, iteration: int) -> dict[str, object]:
        interval = int(getattr(self.opt, "densify_merge_interval", 0))
        if interval <= 0 or iteration <= 0 or iteration % interval != 0:
            return {"attempted": False}
        if self.densify_shadow_event is not None or self.densify_joint_coplanar_event is not None:
            raise RuntimeError("Merging is blocked during child probation")

        keepers_cpu, victims_cpu = compatible_merge_pairs(
            self.model.state_dict(),
            MergeThresholds(
                distance_scale_ratio=float(
                    self.opt.densify_merge_distance_scale_ratio
                ),
                max_pairs=int(self.opt.densify_merge_max_pairs),
            ),
        )
        if keepers_cpu.numel() == 0:
            return {"attempted": True, "candidate_pairs": 0, "accepted": False}
        keepers = keepers_cpu.to(device=self.device)
        victims = victims_cpu.to(device=self.device)
        rows = merged_rows(self.model.state_dict(), keepers, victims)
        original_rows = {
            name: getattr(self.model, name)[keepers].detach().clone()
            for name in rows
        }
        original_active = self.model.active_mask[victims].detach().clone()
        cameras = list(self.densify_shadow_gate_cameras) or list(self.train_cameras)
        was_training = self.model.training
        self.model.eval()

        baseline_sse: list[float] = []
        pixel_counts: list[int] = []
        for camera in cameras:
            target = camera.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            image = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            baseline_sse.append(float((image - target).to(torch.float64).square().sum()))
            pixel_counts.append(int(target.numel()))

        for name, values in rows.items():
            getattr(self.model, name)[keepers] = values
        self.model.active_mask[victims] = False
        candidate_sse: list[float] = []
        for camera in cameras:
            target = camera.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            image = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
            )["image"].clamp(0.0, 1.0)
            candidate_sse.append(float((image - target).to(torch.float64).square().sum()))

        for name, values in original_rows.items():
            getattr(self.model, name)[keepers] = values
        self.model.active_mask[victims] = original_active
        self.model.train(was_training)

        psnr_deltas = [
            -10.0 * math.log10(max(candidate / count, 1e-12))
            + 10.0 * math.log10(max(baseline / count, 1e-12))
            for baseline, candidate, count in zip(
                baseline_sse,
                candidate_sse,
                pixel_counts,
            )
        ]
        pooled_reduction = (
            sum(baseline_sse) - sum(candidate_sse)
        ) / max(sum(baseline_sse), 1e-12)
        mean_delta = sum(psnr_deltas) / len(psnr_deltas)
        worst_delta = min(psnr_deltas)
        accepted = (
            pooled_reduction
            >= float(self.opt.densify_merge_min_pooled_sse_reduction)
            and mean_delta >= float(self.opt.densify_merge_min_mean_psnr_delta)
            and worst_delta >= float(self.opt.densify_merge_min_worst_psnr_delta)
        )
        decision: dict[str, object] = {
            "attempted": True,
            "candidate_pairs": int(keepers.numel()),
            "accepted": bool(accepted),
            "camera_count": len(cameras),
            "pooled_sse_reduction": float(pooled_reduction),
            "mean_psnr_delta": float(mean_delta),
            "worst_psnr_delta": float(worst_delta),
        }
        if accepted:
            self._replace_primitive_rows(
                keepers,
                rows,
                birth_iteration=iteration,
            )
            prune_mask = torch.zeros(
                (self.model.num_base,),
                device=self.device,
                dtype=torch.bool,
            )
            prune_mask[victims] = True
            self._prune_primitives(prune_mask)
            decision["merged_pairs"] = int(keepers.numel())
        else:
            decision["merged_pairs"] = 0
        self.logger.info(
            "Densify merge: accepted=%s pairs=%d pooled_sse=%+.6f mean=%+.6f worst=%+.6f",
            str(bool(accepted)),
            int(keepers.numel()),
            float(pooled_reduction),
            float(mean_delta),
            float(worst_delta),
        )
        return decision

    @torch.no_grad()
    def _collect_full_prune_contribution(self) -> None:
        if self.densify_contrib_accum is None:
            raise RuntimeError("Directional contribution buffer is not initialized")
        self._restart_prune_observation()
        was_training = self.model.training
        self.model.eval()
        for camera in self.train_cameras:
            result = self.renderer.render(
                camera,
                self._render_model(),
                self.bg_color,
                return_contrib=True,
            )
            self._densify_add_contrib(
                result["contrib"],
                camera_name=str(camera.image_name),
            )
        self.model.train(was_training)

    @torch.no_grad()
    def _densify_and_prune_hexels_safe(self, *, size_threshold: int, iteration: int = 0) -> None:
        if getattr(self, "densify_shadow_event", None) is not None:
            raise RuntimeError("Densification mutation is blocked while a shadow event is pending")
        if getattr(self, "densify_joint_coplanar_event", None) is not None:
            raise RuntimeError(
                "Densification mutation is blocked while joint coplanar probation is pending"
            )
        if (
            self.densify_grad_accum is None
            or self.densify_denom is None
            or self.densify_max_radii2d is None
        ):
            return
        direction_signal = str(getattr(self.opt, "densify_direction_signal", "position"))
        residual_topk = int(getattr(self.opt, "densify_residual_topk_views", 0))
        persistent_filter = bool(
            getattr(self.opt, "densify_residual_persistent_filter", False)
        )
        camera_complete_evidence = (
            residual_topk > 0
            or direction_signal in {"attributed_residual", "attributed_covariance"}
            or (direction_signal == "residual_moment" and persistent_filter)
        )
        if camera_complete_evidence:
            expected_camera_names = getattr(
                self,
                "densify_shadow_proposal_camera_names",
                None,
            )
            if expected_camera_names is None:
                expected_camera_names = {
                    str(camera.image_name) for camera in self.train_cameras
                }
            expected_camera_names = set(expected_camera_names)
            if self.densify_moment_camera_names != expected_camera_names:
                observed = len(
                    self.densify_moment_camera_names & expected_camera_names
                )
                self.logger.info(
                    "Densify: residual evidence deferred until a complete proposal-camera "
                    "sweep (%d/%d)",
                    observed,
                    len(expected_camera_names),
                )
                return
        shadow_steps = int(getattr(self.opt, "densify_shadow_steps", 0))
        joint_steps = int(getattr(self.opt, "densify_joint_coplanar_steps", 0))
        probation_steps = max(shadow_steps, joint_steps)
        if probation_steps > 0 and iteration + probation_steps > int(self.opt.iterations):
            self._reset_densify_stats()
            return

        action = str(getattr(self.opt, "densify_action", "mixed"))
        coordinated = bool(getattr(self.opt, "densify_coordinated_prune", False))
        if coordinated:
            self._prune_coordinated_primitives(iteration=iteration)
        elif action in {"bisplit", "edge_child"}:
            self._prune_directional_primitives(iteration=iteration)

        grads = self.densify_grad_accum / self.densify_denom.clamp_min(1.0)
        grads = grads.nan_to_num(0.0).squeeze(-1)

        grad_thr = float(getattr(self.opt, "densify_grad_threshold", 0.0002))
        extent = float(getattr(self, "scene_extent", 1.0))
        percent_dense = float(getattr(self.opt, "percent_dense", 0.01))
        scale_max = self.model.scales.detach().max(dim=1).values

        min_denom = int(getattr(self.opt, "densify_min_denom", 0))
        denom = self.densify_denom.squeeze(-1)
        if direction_signal == "attributed_covariance":
            eligible = self.model.active_mask.clone()
        else:
            eligible = self.model.active_mask & (denom >= min_denom)
        eligible &= self.densify_max_radii2d >= float(getattr(self.opt, "densify_min_projected_radius", 0.0))
        selection_scores = grads
        if action in {"bisplit", "edge_child"} or coordinated:
            if self.densify_birth_iteration is None:
                raise RuntimeError("Directional birth buffer is not initialized")
            grace = int(getattr(self.opt, "densify_newborn_grace", 500))
            eligible &= (iteration - self.densify_birth_iteration) >= grace
        if action == "edge_child":
            if self.densify_has_spawned_child is None:
                raise RuntimeError("Edge-child lineage buffer is not initialized")
            if getattr(self, "densify_shadow_attempted_parent_mask", None) is not None:
                eligible &= ~self.densify_shadow_attempted_parent_mask
            elif not bool(getattr(self.opt, "densify_repeat_parent", False)):
                eligible &= ~self.densify_has_spawned_child
            direction_signal = str(getattr(self.opt, "densify_direction_signal", "position"))
            if direction_signal == "residual_moment":
                if residual_topk > 0:
                    if (
                        self.densify_mv_edge_top_scores is None
                        or self.densify_mv_edge_top_centroids is None
                        or self.densify_mv_edge_vote_counts is None
                    ):
                        raise RuntimeError(
                            "Multi-view residual top-K buffers are not initialized"
                        )
                    summary = summarize_residual_topk_views(
                        self.densify_mv_edge_top_scores,
                        self.densify_mv_edge_top_centroids,
                        self.densify_mv_edge_vote_counts,
                    )
                    mean_centroid = summary.centroids
                    edge_normals = canonical_edge_normals(
                        device=mean_centroid.device,
                        dtype=mean_centroid.dtype,
                    )
                    radius = torch.max(mean_centroid @ edge_normals.T, dim=1).values
                    consistency = summary.consistency
                    eligible &= summary.view_counts >= min_denom
                    eligible &= summary.dominant_edge_vote_counts >= residual_topk
                    eligible &= summary.kth_scores > 0.0
                    eligible &= radius >= float(
                        getattr(self.opt, "densify_moment_min_radius", 0.25)
                    )
                    selection_scores = (
                        summary.kth_scores * consistency * radius.clamp(0.0, 1.0)
                    )
                else:
                    if (
                        self.densify_residual_mass_accum is None
                        or self.densify_centroid_mass_accum is None
                        or self.densify_direction_vote_accum is None
                        or self.densify_moment_vote_count is None
                    ):
                        raise RuntimeError("Residual-moment centroid buffers are not initialized")
                    total_votes = self.densify_moment_vote_count
                    consistency = torch.linalg.vector_norm(
                        self.densify_direction_vote_accum,
                        dim=1,
                    ) / total_votes.clamp_min(1.0)
                    mean_centroid = self.densify_centroid_mass_accum / self.densify_residual_mass_accum.clamp_min(
                        1e-12
                    )[:, None]
                    edge_normals = canonical_edge_normals(
                        device=mean_centroid.device,
                        dtype=mean_centroid.dtype,
                    )
                    radius = torch.max(mean_centroid @ edge_normals.T, dim=1).values
                    direction_alignment = torch.sum(
                        mean_centroid * self.densify_direction_vote_accum,
                        dim=1,
                    )
                    eligible &= total_votes >= max(1, min_denom)
                    eligible &= self.densify_residual_mass_accum > 0.0
                    eligible &= radius >= float(getattr(self.opt, "densify_moment_min_radius", 0.25))
                    eligible &= direction_alignment > 0.0
                    selection_scores = self.densify_residual_mass_accum * consistency * radius.clamp(0.0, 1.0)
            elif direction_signal == "attributed_residual":
                if (
                    self.densify_attributed_edge_mass is None
                    or self.densify_attributed_edge_centroid_mass is None
                    or self.densify_attributed_edge_vote_counts is None
                ):
                    raise RuntimeError("Attributed residual buffers are not initialized")
                summary = summarize_attributed_edge_evidence(
                    self.densify_attributed_edge_mass,
                    self.densify_attributed_edge_centroid_mass,
                    self.densify_attributed_edge_vote_counts,
                )
                mean_centroid = summary.centroids
                edge_normals = canonical_edge_normals(
                    device=mean_centroid.device,
                    dtype=mean_centroid.dtype,
                )
                radius = torch.max(mean_centroid @ edge_normals.T, dim=1).values
                consistency = summary.consistency
                eligible &= summary.view_counts >= max(1, min_denom)
                eligible &= summary.mass > 0.0
                eligible &= radius >= float(
                    getattr(self.opt, "densify_moment_min_radius", 0.25)
                )
                selection_scores = (
                    summary.mass * consistency * radius.clamp(0.0, 1.0)
                )
            elif direction_signal == "attributed_covariance":
                covariance_summary = self._densify_covariance_summary()
                children_per_parent = int(
                    getattr(self.opt, "densify_covariance_children_per_parent", 2)
                )
                if children_per_parent == 1:
                    covariance_eligible = (
                        (covariance_summary.robust_weight > 0.0)
                        & (
                            covariance_summary.material_view_count
                            >= int(getattr(self.opt, "densify_covariance_min_views", 8))
                        )
                        & (covariance_summary.score > 0.0)
                    )
                else:
                    covariance_eligible = covariance_summary.eligible
                eligible &= covariance_eligible.to(device=self.device)
                selection_scores = covariance_summary.score.to(device=self.device)
            else:
                if self.densify_local_grad_abs_accum is None or self.densify_local_grad_signed_accum is None:
                    raise RuntimeError("Directional child buffers are not initialized")
                split_axes = self.densify_local_grad_abs_accum.argmax(dim=1)
                batch = torch.arange(split_axes.numel(), device=self.device)
                dominant_abs = self.densify_local_grad_abs_accum[batch, split_axes]
                dominant_signed = self.densify_local_grad_signed_accum[batch, split_axes]
                consistency = dominant_signed.abs() / dominant_abs.clamp_min(1e-12)
                eligible &= dominant_abs > 0.0
            if direction_signal != "attributed_covariance":
                eligible &= consistency >= float(
                    getattr(self.opt, "densify_direction_consistency", 0.25)
                )
        if direction_signal == "attributed_covariance":
            candidates = eligible
        else:
            candidates = eligible & (grads >= grad_thr)
        cand_idx = candidates.nonzero(as_tuple=True)[0]
        if bool(getattr(self.opt, "densify_absgrad_rank", False)):
            absolute_scores = (self.densify_abs_grad_accum / self.densify_denom.clamp_min(1.0)).squeeze(-1)
            cand_idx = rank_absolute_candidates(grads, absolute_scores, eligible, grad_thr)
            record = dict(iteration=iteration, eligible=int(eligible.sum()),
                          candidates=int(cand_idx.numel()), signed_overlap=int(candidates[cand_idx].sum()),
                          signed_sum=float(grads[eligible].sum()), absolute_sum=float(absolute_scores[eligible].sum()))
            self.densify_score_audit.append(record)
            self.logger.info("Densify score: %s", record)
            selection_scores = absolute_scores
        num_candidates = int(cand_idx.numel())
        num_before = int(self.model.num_active_base)

        max_add = int(getattr(self.opt, "densify_max_add", 0))
        selection_limit = int(cand_idx.numel())
        children_per_selected = (
            int(getattr(self.opt, "densify_covariance_children_per_parent", 2))
            if direction_signal == "attributed_covariance"
            else 1
        )
        if max_add > 0:
            selection_limit = min(selection_limit, max_add // children_per_selected)
        cap_max = int(getattr(self.opt, "densify_cap_max", 0))
        if cap_max > 0:
            # Ordinary edge children refill inactive rows before allocating new
            # storage. Shadow proposals append rows and must count every slot.
            occupied = self.model.num_base + self.model.num_children
            if action == "edge_child" and shadow_steps == 0:
                occupied = self.model.num_active_base + self.model.num_children
            selection_limit = min(
                selection_limit,
                max(0, cap_max - occupied) // children_per_selected,
            )
        if selection_limit == 0:
            cand_idx = cand_idx[:0]
        elif cand_idx.numel() > selection_limit:
            if camera_complete_evidence:
                order = torch.argsort(
                    selection_scores[cand_idx],
                    descending=True,
                    stable=True,
                )
                cand_idx = cand_idx[order[:selection_limit]]
            else:
                topk = torch.topk(
                    selection_scores[cand_idx],
                    k=selection_limit,
                    largest=True,
                ).indices
                cand_idx = cand_idx[topk]

        salvage_groups = int(getattr(self.opt, "densify_shadow_salvage_groups", 0))
        if (
            shadow_steps > 0
            and salvage_groups > 0
            and bool(
                getattr(
                    self.opt,
                    "densify_shadow_salvage_allow_partial_cohort",
                    False,
                )
            )
            and int(cand_idx.numel()) < salvage_groups
        ):
            self.logger.info(
                "Densify: skipped partial shadow proposal with %d/%d required children",
                int(cand_idx.numel()),
                salvage_groups,
            )
            self._reset_densify_stats()
            return

        if action == "clone":
            clone_idx = cand_idx
            split_idx = cand_idx[:0]
        elif action in {"bisplit", "edge_child"}:
            clone_idx = cand_idx[:0]
            split_idx = cand_idx
        else:
            split_thr = percent_dense * extent
            clone_idx = cand_idx[scale_max[cand_idx] <= split_thr]
            split_idx = cand_idx[scale_max[cand_idx] > split_thr]
        num_selected = int(cand_idx.numel())
        num_clones = int(clone_idx.numel())
        num_splits = int(split_idx.numel())

        opacity_split = bool(getattr(self.opt, "densify_opacity_split", True))
        eps = torch.finfo(self.model.opacity_logits.dtype).eps

        if clone_idx.numel() > 0:
            clone_idx = clone_idx.to(dtype=torch.long)
            means = self.model.means3d.detach()[clone_idx]
            new_means = means.clone()
            jitter = float(getattr(self.opt, "densify_clone_jitter", 0.0))
            if jitter > 0.0:
                scales = self.model.scales.detach()[clone_idx]
                rots = self.model.rotations.detach()[clone_idx]
                tx, ty = _quat_to_tangent_xy(rots)
                duv = (
                    torch.randn((clone_idx.numel(), 2), device=self.device, dtype=means.dtype)
                    * scales
                    * jitter
                )
                offset = tx * duv[:, 0:1] + ty * duv[:, 1:2]
                new_means = new_means + offset

            clone_opacity_logits = self.model.opacity_logits.data[clone_idx].clone()
            clone_init_opacity = float(getattr(self.opt, "densify_clone_init_opacity", 0.0))
            if clone_init_opacity > 0.0:
                old_opacity = self.model.opacities.detach()[clone_idx]
                init_opacity = torch.minimum(old_opacity, torch.full_like(old_opacity, clone_init_opacity))
                init_opacity = init_opacity.clamp(eps, 1.0 - eps)
                clone_opacity_logits = opacity_to_logits(init_opacity)

            new_tensors = {
                "means3d": new_means,
                "log_scales": self.model.log_scales.data[clone_idx].clone(),
                "rotation_quats": self.model.rotation_quats.data[clone_idx].clone(),
                "colors": self.model.colors.data[clone_idx].clone(),
                "color_grads": self.model.color_grads.data[clone_idx].clone(),
                "color_quads": self.model.color_quads.data[clone_idx].clone(),
                "view_grads": self.model.view_grads.data[clone_idx].clone(),
                "specular_ks": self.model.specular_ks.data[clone_idx].clone(),
                "specular_log_s": self.model.specular_log_s.data[clone_idx].clone(),
                "opacity_logits": clone_opacity_logits.clone(),
                "log_softness": self.model.log_softness.data[clone_idx].clone(),
            }
            self._transfer_densified_appearance(new_tensors, clone_idx)
            self._append_primitives(new_tensors, birth_iteration=iteration)

        if split_idx.numel() > 0:
            split_idx = split_idx.to(dtype=torch.long)
            if action == "bisplit":
                self._bisplit_primitives(split_idx, iteration=iteration)
            elif action == "edge_child":
                if direction_signal == "attributed_covariance":
                    self._covariance_child_primitives(
                        split_idx,
                        iteration=iteration,
                        eligible_candidate_count=num_candidates,
                    )
                else:
                    self._edge_child_primitives(split_idx, iteration=iteration)
            else:
                N = 2
                scales = self.model.scales.detach()[split_idx]
                means = self.model.means3d.detach()[split_idx]
                rots = self.model.rotations.detach()[split_idx]
                tx, ty = _quat_to_tangent_xy(rots)

                std = scales.repeat_interleave(N, dim=0)
                means_rep = means.repeat_interleave(N, dim=0)
                tx_rep = tx.repeat_interleave(N, dim=0)
                ty_rep = ty.repeat_interleave(N, dim=0)
                jitter = float(getattr(self.opt, "densify_split_jitter", 1.0))
                duv = (
                    torch.randn((std.shape[0], 2), device=self.device, dtype=means.dtype)
                    * std
                    * jitter
                )
                offset = tx_rep * duv[:, 0:1] + ty_rep * duv[:, 1:2]
                new_means = means_rep + offset
                new_scales = std / (0.8 * float(N))
                new_log_scales = torch.log(new_scales.clamp_min(eps))

                new_opacity_logits = self.model.opacity_logits.data[split_idx].repeat_interleave(N, dim=0).clone()
                if opacity_split:
                    old_opacity = self.model.opacities.detach()[split_idx]
                    new_opacity = 1.0 - torch.pow((1.0 - old_opacity).clamp_min(eps), 1.0 / float(N))
                    new_opacity = new_opacity.clamp(eps, 1.0 - eps)
                    new_opacity_logits = opacity_to_logits(new_opacity).repeat_interleave(N, dim=0)

                new_tensors = {
                    "means3d": new_means,
                    "log_scales": new_log_scales,
                    "rotation_quats": self.model.rotation_quats.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "colors": self.model.colors.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "color_grads": self.model.color_grads.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "color_quads": self.model.color_quads.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "view_grads": self.model.view_grads.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "specular_ks": self.model.specular_ks.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "specular_log_s": self.model.specular_log_s.data[split_idx].repeat_interleave(N, dim=0).clone(),
                    "opacity_logits": new_opacity_logits.clone(),
                    "log_softness": self.model.log_softness.data[split_idx].repeat_interleave(N, dim=0).clone(),
                }
                self._transfer_densified_appearance(new_tensors, split_idx.repeat_interleave(N))
                if cap_max > 0:
                    first = {name: values[0::2] for name, values in new_tensors.items()}
                    second = {name: values[1::2] for name, values in new_tensors.items()}
                    self._replace_primitive_rows(split_idx, first, birth_iteration=iteration)
                    self._append_primitives(second, birth_iteration=iteration)
                else:
                    self._append_primitives(new_tensors, birth_iteration=iteration)
                    prune_mask = torch.zeros((int(self.model.num_base),), device=self.device, dtype=torch.bool)
                    prune_mask[split_idx] = True
                    self._prune_primitives(prune_mask)

        if action not in {"bisplit", "edge_child"}:
            prune_mask = (self.model.opacities.detach() < float(getattr(self.opt, "opacity_cull", 0.05))).to(torch.bool)
            if size_threshold > 0:
                big_vs = self.densify_max_radii2d > float(size_threshold)
                big_ws = self.model.scales.detach().max(dim=1).values > (0.1 * extent)
                prune_mask = prune_mask | big_vs | big_ws
            if bool(prune_mask.any().item()):
                self._prune_primitives(prune_mask)

        num_after = int(self.model.num_active_base)
        if cap_max > 0 and num_after > cap_max:
            raise RuntimeError(f"Densification exceeded active cap: active={num_after} cap={cap_max}")
        self.logger.info(
            "Densify: candidates=%d selected=%d clones=%d splits=%d active=%d->%d capacity=%d cap=%d",
            num_candidates,
            num_selected,
            num_clones,
            num_splits,
            num_before,
            num_after,
            self.model.num_base,
            cap_max,
        )
        self._reset_densify_stats()

    def _append_children(self, parent_idx: torch.Tensor, *, layer: torch.Tensor | None = None) -> None:
        if parent_idx.numel() == 0:
            return
        sides = str(getattr(self.opt, "child_sides", "both"))
        per_parent = 2 if sides == "both" else 1
        if int(self.opt.child_cap_max) > 0 and (
            self.model.num_children + per_parent * int(parent_idx.numel()) > int(self.opt.child_cap_max)
        ):
            return
        old_optimizer = self.optimizer
        old_C = int(self.model.num_children)
        self.model.append_children(parent_idx, sides=sides, layer=layer)
        self.model.child_color_grads.requires_grad_(self.optimize_color_grads)
        self.model.child_color_quads.requires_grad_(self.optimize_color_quads)
        self.model.child_view_grads.requires_grad_(self.optimize_view_grads)
        self.model.child_specular_ks.requires_grad_(bool(getattr(self, "optimize_specular", False)))
        self.model.child_specular_log_s.requires_grad_(bool(getattr(self, "optimize_specular", False)))
        self.model.child_rotvec.requires_grad_(bool(getattr(self.opt, "child_rotation_delta", False)))
        self._rebuild_optimizer_with_state(old_optimizer)
        new_C = int(self.model.num_children)
        self.logger.info(
            "Children: added %d (parents=%d sides=%s) (C=%d)",
            int(new_C - old_C),
            int(parent_idx.numel()),
            sides,
            int(new_C),
        )

    def _parents_with_children_mask(self, P: int) -> torch.Tensor:
        if self.model.num_children < 1:
            return torch.zeros((P,), device=self.device, dtype=torch.bool)
        mask = torch.zeros((P,), device=self.device, dtype=torch.bool)
        mask[self.model.child_parent_idx] = True
        return mask

    def _parent_max_child_layer(self, P: int) -> torch.Tensor:
        if self.model.num_children < 1:
            return torch.zeros((P,), device=self.device, dtype=torch.long)
        out = torch.zeros((P,), device=self.device, dtype=torch.long)
        out.scatter_reduce_(
            0,
            self.model.child_parent_idx,
            self.model.child_layer,
            reduce="amax",
            include_self=True,
        )
        return out

    def _child_spawn_step(self, mean_grad_norm: torch.Tensor) -> None:
        if not self.opt.enable_children:
            return
        if int(self.opt.child_cap_max) == 0:
            return
        if self.model.num_children >= int(self.opt.child_cap_max):
            return

        P = int(self.model.num_base)
        max_layers = int(getattr(self.opt, "child_max_layers_per_parent", 1))
        parent_layer: torch.Tensor | None
        if max_layers <= 1:
            parent_layer = None
            eligible = ~self._parents_with_children_mask(P)
        else:
            parent_layer = self._parent_max_child_layer(P)
            eligible = parent_layer < max_layers
        num_eligible = int(eligible.sum().item())
        if num_eligible < 1:
            return

        remaining_children = int(self.opt.child_cap_max) - int(self.model.num_children)
        sides = str(getattr(self.opt, "child_sides", "both"))
        per_parent = 2 if sides == "both" else 1
        max_parents = min(int(self.opt.child_parents_per_interval), remaining_children // per_parent, num_eligible)
        if max_parents < 1:
            return

        scores = mean_grad_norm.detach().float().nan_to_num(0.0)
        scores = torch.where(eligible, scores, torch.full_like(scores, float("-inf")))
        parent_idx = torch.topk(scores, k=max_parents, largest=True).indices.to(dtype=torch.long)
        layer = None if parent_layer is None else (parent_layer[parent_idx] + 1)
        self._append_children(parent_idx, layer=layer)

    @torch.no_grad()
    def _mcmc_noise_step(self, iteration: int) -> None:
        if not bool(getattr(self.opt, "enable_mcmc", False)):
            return
        if not self.optimize_geometry:
            return
        mcmc_from = int(getattr(self.opt, "mcmc_from_iter", 0))
        mcmc_until = int(getattr(self.opt, "mcmc_until_iter", -1))
        if iteration < mcmc_from or (mcmc_until != -1 and iteration > mcmc_until):
            return
        noise_lr = float(getattr(self.opt, "mcmc_noise_lr", 0.0))
        if noise_lr <= 0.0:
            return

        P = int(self.model.num_base)
        if P <= 0:
            return

        opacity = self.model.opacities.detach()
        gate = torch.sigmoid(100.0 * ((1.0 - opacity) - 0.995))
        position_lr = next(
            (float(group["lr"]) for group in self.optimizer.param_groups if group.get("name") == "means3d"),
            0.0,
        )
        step_scale = noise_lr * position_lr
        if step_scale == 0.0:
            return

        scales = self.model.scales.detach()
        rots = self.model.rotations.detach()
        tx, ty = _quat_to_tangent_xy(rots)

        noise_uv = torch.randn((P, 2), device=self.device, dtype=self.model.means3d.dtype)
        du = (scales[:, 0] * scales[:, 0]) * noise_uv[:, 0]
        dv = (scales[:, 1] * scales[:, 1]) * noise_uv[:, 1]
        delta = tx * du[:, None] + ty * dv[:, None]

        normal_frac = float(getattr(self.opt, "mcmc_normal_noise_frac", 0.0))
        if normal_frac > 0.0:
            nz = _quat_to_normal_z(rots)
            noise_n = torch.randn((P,), device=self.device, dtype=self.model.means3d.dtype)
            avg_s = 0.5 * (scales[:, 0] + scales[:, 1])
            dn = (avg_s * avg_s) * noise_n * normal_frac
            delta = delta + nz * dn[:, None]

        self.model.means3d.data.add_(delta * (gate * step_scale)[:, None])

    def _mcmc_step(self) -> None:
        if self.mcmc_binoms is None:
            return

        with torch.no_grad():
            reorient = bool(getattr(self.opt, "mcmc_reorient", False))
            mode = str(getattr(self.opt, "mcmc_mode", self.mcmc_mode))
            if mode == "contrib":
                if self.contrib_ema is None:
                    return
                util = self.contrib_ema
                mean_util = util.mean()
                if float(mean_util.item()) <= 0.0:
                    return
                dead_mask = util < (float(self.opt.mcmc_dead_threshold) * mean_util)
            elif mode == "opacity":
                opacity = self.model.opacities.detach()
                util = opacity
                dead_mask = opacity <= float(self.opt.mcmc_dead_threshold)
            else:
                raise ValueError(f"Unsupported --mcmc_mode {mode!r}")

            if self.opt.enable_children:
                dead_mask = dead_mask & (~self._parents_with_children_mask(int(util.shape[0])))
            dead_idx = dead_mask.nonzero(as_tuple=True)[0]
            if dead_idx.numel() > 0:
                alive_idx = (~dead_mask).nonzero(as_tuple=True)[0]
                if alive_idx.numel() > 0:
                    probs = util[alive_idx]
                    if probs.sum() <= 0:
                        probs = torch.ones_like(probs)
                    probs = probs / (probs.sum() + torch.finfo(probs.dtype).eps)
                    sampled_src = alive_idx[torch.multinomial(probs, dead_idx.numel(), replacement=True)]
                    counts = torch.bincount(sampled_src, minlength=util.shape[0])
                    N = counts[sampled_src] + 1

                    new_opacity, new_scales = compute_relocation(
                        opacity_old=self.model.opacities[sampled_src].detach(),
                        scales_old=self.model.scales[sampled_src].detach(),
                        N=N,
                        binoms=self.mcmc_binoms,
                    )
                    new_opacity = new_opacity.clamp(0.005, 1.0 - torch.finfo(new_opacity.dtype).eps)
                    new_log_scales = torch.log(new_scales.clamp_min(torch.finfo(new_scales.dtype).eps))
                    new_opacity_logits = opacity_to_logits(new_opacity)

                    for tname in (
                        "means3d",
                        "rotation_quats",
                        "colors",
                        "color_grads",
                        "color_quads",
                        "view_grads",
                        "specular_ks",
                        "specular_log_s",
                        "log_softness",
                    ):
                        getattr(self.model, tname).data[dead_idx] = getattr(self.model, tname).data[sampled_src]

                    if reorient and self.surface_knn_idx is not None:
                        tgt = self._estimate_pca_normals(sampled_src)
                        q_old = torch.nn.functional.normalize(self.model.rotation_quats.data[dead_idx], dim=-1)
                        n_old = _quat_to_normal_z(q_old)
                        q_align = _quat_from_two_vectors(n_old, tgt)
                        self.model.rotation_quats.data[dead_idx] = _quat_mul_wxyz(q_align, q_old)

                    self.model.log_scales.data[dead_idx] = new_log_scales
                    self.model.opacity_logits.data[dead_idx] = new_opacity_logits
                    self.model.log_scales.data[sampled_src] = new_log_scales
                    self.model.opacity_logits.data[sampled_src] = new_opacity_logits

                    spawn_jitter = float(getattr(self.opt, "mcmc_spawn_jitter", 0.0))
                    if spawn_jitter > 0.0:
                        q = torch.nn.functional.normalize(self.model.rotation_quats.data[dead_idx], dim=-1)
                        tx_j, ty_j = _quat_to_tangent_xy(q)
                        nz_j = _quat_to_normal_z(q)
                        s = self.model.scales.detach()[dead_idx]
                        noise3 = torch.randn((dead_idx.numel(), 3), device=self.device, dtype=self.model.means3d.dtype)
                        offset = (tx_j * noise3[:, 0:1] * s[:, 0:1]
                                  + ty_j * noise3[:, 1:2] * s[:, 1:2]
                                  + nz_j * noise3[:, 2:3] * 0.5 * (s[:, 0:1] + s[:, 1:2]))
                        self.model.means3d.data[dead_idx] += offset * spawn_jitter

                    reset_idx = torch.unique(torch.cat([dead_idx, sampled_src], dim=0))
                    for pname in self._base_optimizer_param_names():
                        self._zero_adam_state(getattr(self.model, pname), reset_idx)

                    if self.contrib_ema is not None:
                        self.contrib_ema[dead_idx] = 0.0
                    self.logger.info("MCMC: relocated %d primitives", int(dead_idx.numel()))

            cap_max = int(self.opt.mcmc_cap_max)
            if cap_max > 0:
                P = int(self.model.opacity_logits.shape[0])
                target = min(cap_max, int((1.0 + float(self.opt.mcmc_grow_rate)) * P))
                num_add = max(0, target - P)
                if num_add > 0:
                    if mode == "opacity":
                        probs = self.model.opacities.detach()
                    else:
                        probs = util
                    if probs.sum() <= 0:
                        probs = torch.ones_like(probs)
                    probs = probs / (probs.sum() + torch.finfo(probs.dtype).eps)
                    sampled_src = torch.multinomial(probs, num_add, replacement=True)
                    counts = torch.bincount(sampled_src, minlength=P)
                    N = counts[sampled_src] + 1

                    new_opacity, new_scales = compute_relocation(
                        opacity_old=self.model.opacities[sampled_src].detach(),
                        scales_old=self.model.scales[sampled_src].detach(),
                        N=N,
                        binoms=self.mcmc_binoms,
                    )
                    new_opacity = new_opacity.clamp(0.005, 1.0 - torch.finfo(new_opacity.dtype).eps)
                    new_log_scales = torch.log(new_scales.clamp_min(torch.finfo(new_scales.dtype).eps))
                    new_opacity_logits = opacity_to_logits(new_opacity)

                    self.model.log_scales.data[sampled_src] = new_log_scales
                    self.model.opacity_logits.data[sampled_src] = new_opacity_logits

                    for pname in self._base_optimizer_param_names():
                        self._zero_adam_state(getattr(self.model, pname), sampled_src)

                    new_tensors = {
                        "means3d": self.model.means3d.data[sampled_src].clone(),
                        "log_scales": new_log_scales.clone(),
                        "rotation_quats": self.model.rotation_quats.data[sampled_src].clone(),
                        "colors": self.model.colors.data[sampled_src].clone(),
                        "color_grads": self.model.color_grads.data[sampled_src].clone(),
                        "color_quads": self.model.color_quads.data[sampled_src].clone(),
                        "view_grads": self.model.view_grads.data[sampled_src].clone(),
                        "specular_ks": self.model.specular_ks.data[sampled_src].clone(),
                        "specular_log_s": self.model.specular_log_s.data[sampled_src].clone(),
                        "opacity_logits": new_opacity_logits.clone(),
                        "log_softness": self.model.log_softness.data[sampled_src].clone(),
                    }
                    spawn_jitter = float(getattr(self.opt, "mcmc_spawn_jitter", 0.0))
                    if spawn_jitter > 0.0:
                        q = torch.nn.functional.normalize(new_tensors["rotation_quats"], dim=-1)
                        tx_j, ty_j = _quat_to_tangent_xy(q)
                        nz_j = _quat_to_normal_z(q)
                        s = torch.exp(new_tensors["log_scales"])
                        noise3 = torch.randn((num_add, 3), device=self.device, dtype=new_tensors["means3d"].dtype)
                        offset = (tx_j * noise3[:, 0:1] * s[:, 0:1]
                                  + ty_j * noise3[:, 1:2] * s[:, 1:2]
                                  + nz_j * noise3[:, 2:3] * 0.5 * (s[:, 0:1] + s[:, 1:2]))
                        new_tensors["means3d"] = new_tensors["means3d"] + offset * spawn_jitter
                    if reorient and self.surface_knn_idx is not None:
                        tgt = self._estimate_pca_normals(sampled_src)
                        q_old = torch.nn.functional.normalize(new_tensors["rotation_quats"], dim=-1)
                        n_old = _quat_to_normal_z(q_old)
                        q_align = _quat_from_two_vectors(n_old, tgt)
                        new_tensors["rotation_quats"] = _quat_mul_wxyz(q_align, q_old)
                    self._append_primitives(new_tensors)
                    self.logger.info("MCMC: added %d primitives (P=%d)", int(num_add), int(self.model.opacity_logits.shape[0]))

    def _cpu_mem_gb(self) -> float:
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3

    def _gpu_mem_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        mem_free, mem_total = torch.cuda.mem_get_info()
        return float(mem_total - mem_free) / 1024**3

    def _background(self) -> torch.Tensor:
        if self.opt.random_background:
            return (torch.rand((3,), device=self.device) * 0.1).to(torch.float32)
        return self.bg_color

    def _render_model(self) -> object:
        edge_transfer = getattr(self, "edge_transfer_controller", None)
        if edge_transfer is not None:
            return edge_transfer
        event = getattr(self, "densify_shadow_event", None)
        if event is None or not bool(
            getattr(self.opt, "densify_shadow_parent_view_anchor", False)
        ):
            return self.model
        if not bool(torch.all(self.model.active_mask[event.child_ids]).item()):
            return self.model
        steps = int(self.opt.densify_shadow_steps)
        release = max(0.0, min(1.0, 2.0 * float(event.steps_completed) / steps - 1.0))
        if release >= 1.0:
            return self.model
        return _DensifyShadowViewAnchor(self.model, event, release)

    def _select_eval_train_cameras(self) -> list[object]:
        cams: list[object] = list(self.train_cameras)
        n = int(getattr(self.opt, "test_train_views", 10))
        if n >= len(cams):
            return cams
        step = max(1, len(cams) // n)
        idxs = list(range(0, len(cams), step))[:n]
        return [cams[i] for i in idxs]

    def _progress_render_camera(self) -> object:
        if self.resume_train_holdout_cameras:
            return self.train_cameras[0]
        if self.test_cameras:
            return self.test_cameras[0]
        return self.train_cameras[0]

    def _next_train_camera(self) -> object:
        if self.camera_sampling == "random":
            return random.choice(self.train_cameras)
        if self.camera_sampling == "weighted":
            if self._camera_sampling_weights is None:
                raise RuntimeError("Weighted camera sampling has no weights")
            return random.choices(
                self.train_cameras,
                weights=self._camera_sampling_weights,
                k=1,
            )[0]
        if not self._camera_stack:
            self._camera_stack = list(self.train_cameras)
        return self._camera_stack.pop(random.randrange(len(self._camera_stack)))

    def _record_resume_train_camera(self, camera: object) -> None:
        if self.resume_train_camera_manifest_path is None:
            return
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or name not in set(self.resume_train_camera_names):
            raise RuntimeError("Selected camera is outside the resume training camera manifest")
        self.resume_train_camera_trace_names.append(name)

    def _record_train_camera(self, camera: object) -> None:
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str):
            raise RuntimeError("Selected training camera has no image name")
        self.train_camera_trace_names.append(name)
        self._record_resume_train_camera(camera)

    def _replay_camera_sampler(self, completed_iterations: int) -> None:
        random.seed(int(self.opt.seed))
        self._camera_stack = []
        for _ in range(completed_iterations):
            self._next_train_camera()

    @torch.no_grad()
    def _eval_psnr_ssim(self, cams: list[object]) -> tuple[float, float]:
        if not cams:
            return 0.0, 0.0
        self.model.eval()
        psnrs: list[torch.Tensor] = []
        ssims: list[torch.Tensor] = []
        for cam in cams:
            pred = self.renderer.render(cam, self._render_model(), self.bg_color)["image"].clamp(0.0, 1.0)
            gt = cam.original_image.to(self.device)[0:3].clamp(0.0, 1.0)
            psnrs.append(_psnr(pred, gt))
            ssims.append(self.metric_ssim(pred, gt))
        mean_psnr = torch.stack(psnrs).mean().item()
        mean_ssim = torch.stack(ssims).mean().item()
        return float(mean_psnr), float(mean_ssim)

    @torch.no_grad()
    def _wandb_log_test_samples(self, iteration: int) -> None:
        if self.wandb_run is None:
            return
        n = int(getattr(self.opt, "wandb_test_images", 0))
        if n <= 0:
            return
        if not self.test_cameras:
            return

        import wandb

        rng = random.Random(int(self.opt.seed) + int(iteration))
        idxs = rng.sample(range(len(self.test_cameras)), k=min(n, len(self.test_cameras)))
        pred_imgs = []
        gt_imgs = []
        depth_imgs = []
        for idx in idxs:
            cam = self.test_cameras[idx]
            render_pkg = self.renderer.render(
                cam,
                self._render_model(),
                self.bg_color,
                return_surface=True,
            )
            pred = render_pkg["image"].clamp(0.0, 1.0)
            gt = cam.original_image.to(self.device)[0:3].clamp(0.0, 1.0)

            surf_depth = render_pkg["surf_depth"]
            surf_alpha = render_pkg["surf_alpha"]
            depth = surf_depth / surf_alpha.clamp_min(1e-6)

            pred_u8 = pred.mul(255).byte().permute(1, 2, 0).cpu().numpy()
            gt_u8 = gt.mul(255).byte().permute(1, 2, 0).cpu().numpy()
            depth_hw = depth.nan_to_num(0.0)
            valid = torch.isfinite(depth_hw) & (depth_hw > 0.0)
            if valid.any():
                d = depth_hw[valid]
                lo = torch.quantile(d, 0.02)
                hi = torch.quantile(d, 0.98)
                depth_norm = ((depth_hw - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
            else:
                depth_norm = torch.zeros_like(depth_hw)
            depth_u8 = depth_norm.mul(255).byte().cpu().numpy()
            depth_u8 = depth_u8[..., None].repeat(3, axis=2)

            pred_imgs.append(wandb.Image(pred_u8))
            gt_imgs.append(wandb.Image(gt_u8))
            depth_imgs.append(wandb.Image(depth_u8))

        self.wandb_run.log(
            {
                "eval/test_pred_rgb": pred_imgs,
                "eval/test_gt_rgb": gt_imgs,
                "eval/test_pred_depth": depth_imgs,
            },
            step=int(iteration),
        )

    def train(self) -> dict[str, object]:
        start_t = time.time()
        peak_cpu_gb = self._cpu_mem_gb()
        peak_gpu_used_gb = self._gpu_mem_gb()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for it in range(self.start_iteration + 1, self.opt.iterations + 1):
            self.model.train()
            shadow_pending = self.densify_shadow_event is not None
            joint_coplanar_pending = self.densify_joint_coplanar_event is not None
            child_probation_pending = shadow_pending or joint_coplanar_pending
            edge_transfer_step = it - self.start_iteration
            if self.edge_transfer_controller is not None:
                self.edge_transfer_controller.set_step(edge_transfer_step)
            self._apply_lr_decay_milestone(it)
            self._update_position_lr(it)
            sh_degree = self._update_sh_degree(it)

            cam = self._next_train_camera()
            self._record_train_camera(cam)
            gt = cam.original_image.to(self.device)
            bg = self._background()

            enable_mcmc = bool(getattr(self.opt, "enable_mcmc", False))
            mcmc_interval = int(getattr(self.opt, "mcmc_interval", 1))
            mcmc_from = int(getattr(self.opt, "mcmc_from_iter", 0))
            mcmc_until = int(getattr(self.opt, "mcmc_until_iter", -1))
            in_mcmc_window = it >= mcmc_from and (mcmc_until == -1 or it <= mcmc_until)
            need_mcmc_step = enable_mcmc and in_mcmc_window and (it % mcmc_interval == 0)
            densify_until = int(getattr(self.opt, "densify_until_iter", 15_000))
            densify_from = int(getattr(self.opt, "densify_from_iter", 500))
            in_densify_window = self.enable_densify and it < densify_until
            directional_density = str(getattr(self.opt, "densify_action", "mixed")) in {
                "bisplit",
                "edge_child",
            }
            prune_interval = int(getattr(self.opt, "densify_prune_interval", 1000))
            newborn_grace = int(getattr(self.opt, "densify_newborn_grace", 500))
            prune_maintenance_until = densify_until + newborn_grace + prune_interval
            need_densify_prune_contrib = (
                self.enable_densify
                and not child_probation_pending
                and directional_density
                and prune_interval > 0
                and it > densify_from
                and it <= prune_maintenance_until
            )
            densify_direction_signal = str(
                getattr(self.opt, "densify_direction_signal", "position")
            )
            need_densify_moment = (
                in_densify_window
                and not child_probation_pending
                and it > densify_from
                and str(cam.image_name)
                in getattr(
                    self,
                    "densify_shadow_proposal_camera_names",
                    self.densify_shadow_fit_camera_names,
                )
                and str(getattr(self.opt, "densify_action", "mixed")) == "edge_child"
                and densify_direction_signal
                in {
                    "residual_moment",
                    "attributed_residual",
                    "attributed_covariance",
                }
                and it % int(getattr(self.opt, "densify_moment_interval", 10)) == 0
            )
            need_new_densify_evidence = (
                need_densify_moment
                and str(cam.image_name) not in self.densify_moment_camera_names
            )
            need_contrib = (
                (need_mcmc_step and self.mcmc_mode == "contrib")
                or need_densify_prune_contrib
                or need_densify_moment
            )
            repulsion_weight = float(getattr(self.opt, "repulsion_screen_weight", 0.0))
            need_overdraw_for_loss = repulsion_weight > 0.0
            track_overdraw = bool(getattr(self.opt, "track_overdraw", False))
            need_overdraw_metrics = track_overdraw and (it % int(self.opt.log_every) == 0 or it == 1)
            need_overdraw = need_overdraw_for_loss or need_overdraw_metrics
            surface_coplanar_weight = float(getattr(self.opt, "surface_coplanar_weight", 0.0))
            surface_normal_smooth_weight = float(getattr(self.opt, "surface_normal_smooth_weight", 0.0))
            surface_normal_depth_weight = float(getattr(self.opt, "surface_normal_depth_weight", 0.0))
            surface_normal_depth_start = int(getattr(self.opt, "surface_normal_depth_start", 0))
            need_normal_surface = surface_normal_depth_weight > 0.0 and it >= surface_normal_depth_start
            ray_distortion_target = float(
                getattr(self.opt, "ray_distortion_weight", 0.0)
            )
            ray_distortion_start = int(
                getattr(self.opt, "ray_distortion_start", 0)
            )
            ray_distortion_ramp_steps = int(
                getattr(self.opt, "ray_distortion_ramp_steps", 0)
            )
            ray_distortion_weight = 0.0
            if ray_distortion_target > 0.0 and it >= ray_distortion_start:
                if ray_distortion_ramp_steps > 0:
                    ray_distortion_weight = ray_distortion_target * min(
                        1.0,
                        float(it - ray_distortion_start + 1)
                        / float(ray_distortion_ramp_steps),
                    )
                else:
                    ray_distortion_weight = ray_distortion_target
            need_ray_distortion = ray_distortion_weight > 0.0
            mvs_surface_weight = float(
                getattr(self, "resume_mvs_surface_weight", 0.0)
            )
            need_mvs_surface = (
                self.resume_mvs_surface_targets is not None
                and mvs_surface_weight > 0.0
            )
            coverage_weight = float(getattr(self.opt, "densify_coverage_weight", 0.0))
            need_surface = (
                need_normal_surface
                or need_mvs_surface
                or need_ray_distortion
                or (in_densify_window and coverage_weight > 0.0)
            )
            need_surface_knn = (
                surface_coplanar_weight > 0.0
                or surface_normal_smooth_weight > 0.0
                or bool(getattr(self.opt, "mcmc_reorient", False))
                or bool(getattr(self.opt, "densify_child_reorient", False))
            )
            if need_surface_knn:
                self._maybe_update_surface_knn(it)
            need_children = bool(self.opt.enable_children and (it % int(self.opt.child_interval) == 0))
            need_viewspace_points = need_children or (
                self.enable_densify
                and not child_probation_pending
                and it > densify_from
                and it < int(getattr(self.opt, "densify_until_iter", 15_000))
            )
            render_pkg = self.renderer.render(
                cam,
                self._render_model(),
                bg,
                return_contrib=need_contrib,
                return_overdraw=need_overdraw,
                return_overdraw_hits=need_overdraw_metrics,
                return_surface=need_surface,
                return_distortion=need_ray_distortion,
                return_median_depth=(need_normal_surface and self.opt.surface_normal_depth_mode == "median"),
                return_viewspace_points=need_viewspace_points,
                sh_degree=sh_degree,
            )
            pred = render_pkg["image"]
            if self.contrib_ema is not None and need_contrib:
                contrib = render_pkg["contrib"].detach()[: self.model.num_base]
                self._mcmc_update_ema(contrib)
            camera_name = str(cam.image_name)
            if need_densify_prune_contrib:
                self._densify_add_contrib(render_pkg["contrib"], camera_name=camera_name)
            if need_new_densify_evidence:
                persistent_filter = bool(
                    getattr(self.opt, "densify_residual_persistent_filter", False)
                )
                filtered_prediction = None
                if densify_direction_signal in {
                    "attributed_residual",
                    "attributed_covariance",
                } or persistent_filter:
                    with torch.no_grad():
                        filtered_prediction = _box_downsample(
                            self.renderer.render(
                                _scaled_camera(cam, 2),
                                self._render_model(),
                                bg,
                                sh_degree=sh_degree,
                            )["image"],
                            2,
                        )
                if densify_direction_signal == "residual_moment":
                    self._densify_add_moment_stats(
                        pred,
                        gt,
                        render_pkg["contrib"],
                        camera_name=camera_name,
                        filtered_prediction=filtered_prediction,
                    )
                else:
                    if filtered_prediction is None:
                        raise RuntimeError("Attributed evidence requires a filtered render")
                    owner = self.renderer.render_owner(
                        cam,
                        self._render_model(),
                        bg,
                        sh_degree=sh_degree,
                    )
                    if densify_direction_signal == "attributed_residual":
                        self._densify_add_attributed_stats(
                            pred,
                            filtered_prediction,
                            gt[0:3],
                            render_pkg["contrib"],
                            owner,
                            camera_name=camera_name,
                        )
                    else:
                        self._densify_add_covariance_stats(
                            pred,
                            filtered_prediction,
                            gt[0:3],
                            render_pkg["contrib"],
                            owner,
                            camera_name=camera_name,
                        )

            l1 = self.l1_loss(pred, gt)
            photometric = (
                l1
                if self.opt.photometric_loss == "l1"
                else torch.nn.functional.mse_loss(pred, gt)
            )
            dssim = 1.0 - self.ssim(pred, gt)
            loss = (
                (1.0 - self.opt.lambda_dssim) * photometric
                + self.opt.lambda_dssim * dssim
            )
            coverage_loss = None
            if in_densify_window and coverage_weight > 0.0:
                coverage_loss = _coverage_weighted_alpha_loss(pred, gt, render_pkg["surf_alpha"])
                loss = loss + coverage_weight * coverage_loss
            repulsion_screen = None
            if repulsion_weight > 0.0:
                overdraw = render_pkg["overdraw"]
                target = float(getattr(self.opt, "repulsion_screen_target", 1.0))
                excess = (overdraw - target).clamp_min(0.0)
                repulsion_screen = (excess * excess).mean()
                loss = loss + repulsion_weight * repulsion_screen
            repulsion_knn = None
            if float(getattr(self.opt, "repulsion_knn_weight", 0.0)) > 0.0:
                self._maybe_update_repulsion_knn(it)
                repulsion_knn = self._repulsion_knn_loss()
                loss = loss + float(self.opt.repulsion_knn_weight) * repulsion_knn
            surface_coplanar = None
            surface_normal_smooth = None
            if surface_coplanar_weight > 0.0 or surface_normal_smooth_weight > 0.0:
                coplanar_loss, normal_smooth_loss = self._surface_knn_losses()
                if surface_coplanar_weight > 0.0:
                    surface_coplanar = coplanar_loss
                    loss = loss + surface_coplanar_weight * coplanar_loss
                if surface_normal_smooth_weight > 0.0:
                    surface_normal_smooth = normal_smooth_loss
                    loss = loss + surface_normal_smooth_weight * normal_smooth_loss
            surface_normal_depth = None
            if need_normal_surface:
                surf_depth = render_pkg["surf_depth"]
                surf_alpha = render_pkg["surf_alpha"]
                surf_normals = render_pkg["surf_normals"]

                if self.opt.surface_normal_depth_mode == "median":
                    depth = render_pkg["surf_median_depth"]
                else:
                    denominator = surf_alpha if self.opt.surface_normal_objective == "2dgs" else surf_alpha.detach()
                    depth = surf_depth / denominator.clamp_min(1e-6)
                n_depth = _depth_to_normal(cam, depth)
                surface_normal_depth = surface_normal_loss(
                    n_depth, surf_normals, surf_alpha,
                    objective=self.opt.surface_normal_objective,
                    alpha_threshold=self.opt.surface_normal_depth_alpha_thresh,
                )
                loss = loss + surface_normal_depth_weight * surface_normal_depth
            ray_distortion = None
            if need_ray_distortion:
                ray_distortion = render_pkg["surf_distortion"].mean()
                if not bool(torch.isfinite(ray_distortion).item()):
                    raise RuntimeError("Ray-distortion loss is nonfinite")
                loss = loss + ray_distortion_weight * ray_distortion
            mvs_surface = None
            if need_mvs_surface:
                mvs_surface = self._resume_mvs_surface_loss(
                    camera_name,
                    render_pkg,
                )
                mvs_terms = torch.stack(
                    (mvs_surface.depth, mvs_surface.foreground_spill)
                )
                if not bool(torch.isfinite(mvs_terms).all().item()):
                    raise RuntimeError("Resume MVS surface loss is nonfinite")
                loss = loss + mvs_surface_weight * mvs_terms.sum()

            child_rot_delta = None
            if self.opt.enable_children and bool(getattr(self.opt, "child_rotation_delta", False)) and self.model.num_children > 0:
                w = float(getattr(self.opt, "child_rotation_delta_weight", 0.0))
                if w > 0.0:
                    child_rot_delta = (self.model.child_rotvec * self.model.child_rotvec).sum(dim=1).mean()
                    loss = loss + w * child_rot_delta

            resume_scale_anchor = self._resume_scale_anchor_loss()
            resume_scale_anchor_weight = float(
                getattr(self, "resume_scale_anchor_weight", 0.0)
            )
            if resume_scale_anchor is not None and resume_scale_anchor_weight > 0.0:
                loss = loss + resume_scale_anchor_weight * resume_scale_anchor
            resume_surface_anchor = self._resume_surface_anchor_loss()
            resume_surface_anchor_weight = float(
                getattr(self, "resume_surface_anchor_weight", 0.0)
            )
            if (
                resume_surface_anchor is not None
                and resume_surface_anchor_weight > 0.0
            ):
                loss = (
                    loss
                    + resume_surface_anchor_weight * resume_surface_anchor
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self._densify_shadow_gradient_ids().numel() > 0:
                self._densify_shadow_control_backward(
                    cam,
                    gt,
                    bg,
                    sh_degree=sh_degree,
                )
            if self.edge_transfer_controller is not None:
                self.edge_transfer_controller.prepare_gradients(edge_transfer_step)
            mean_grad_norm = None
            if need_children:
                vsp = render_pkg.get("viewspace_points")
                if vsp is not None and vsp.grad is not None:
                    mean_grad_norm = torch.linalg.norm(
                        vsp.grad.detach()[: self.model.num_base], dim=-1
                    )
                else:
                    mean_grad_norm = torch.zeros((self.model.num_base,), device=self.device, dtype=torch.float32)

            run_densify = False
            size_threshold = 0
            reset_opacity = False
            if (
                self.enable_densify
                and not child_probation_pending
                and it > densify_from
                and it < int(getattr(self.opt, "densify_until_iter", 15_000))
            ):
                radii_base = render_pkg["radii"].detach()[: self.model.num_base]
                visibility_filter = radii_base > 0
                if self.densify_max_radii2d is not None:
                    self.densify_max_radii2d[visibility_filter] = torch.maximum(
                        self.densify_max_radii2d[visibility_filter], radii_base[visibility_filter]
                    )
                vsp = render_pkg.get("viewspace_points")
                if vsp is not None and camera_name in getattr(
                    self,
                    "densify_shadow_proposal_camera_names",
                    self.densify_shadow_fit_camera_names,
                ):
                    self._densify_add_stats(vsp, visibility_filter)

                densify_interval = int(getattr(self.opt, "densification_interval", 100))
                if it > densify_from and (it % densify_interval == 0):
                    reset_int = int(getattr(self.opt, "opacity_reset_interval", 3000))
                    size_threshold = (
                        int(getattr(self.opt, "densify_max_screen_size", 20))
                        if reset_int > 0 and it > reset_int
                        else 0
                    )
                    run_densify = True

                reset_int = int(getattr(self.opt, "opacity_reset_interval", 3000))
                if reset_int > 0 and (it % reset_int == 0):
                    reset_opacity = True
            self._step_optimizer_and_density(
                iteration=it,
                run_densify=run_densify,
                size_threshold=size_threshold,
                reset_opacity=reset_opacity,
            )
            if (
                self.enable_densify
                and not child_probation_pending
                and directional_density
                and not run_densify
                and not bool(getattr(self.opt, "densify_coordinated_prune", False))
                and densify_from <= it <= prune_maintenance_until
                and prune_interval > 0
                and it % prune_interval == 0
            ):
                self._prune_directional_primitives(iteration=it)
            if (
                self.enable_densify
                and bool(getattr(self.opt, "densify_coordinated_prune", False))
                and not child_probation_pending
                and not run_densify
                and densify_from <= it <= prune_maintenance_until
            ):
                self._prune_coordinated_primitives(iteration=it)
            if (
                self.enable_densify
                and not child_probation_pending
                and directional_density
                and not run_densify
                and int(getattr(self.opt, "densify_merge_interval", 0)) > 0
                and it % int(self.opt.densify_merge_interval) == 0
            ):
                merge_decision = self._merge_compatible_primitives(iteration=it)
                if bool(merge_decision["attempted"]):
                    merge_decision["iteration"] = int(it)
                    self.densify_merge_decisions.append(merge_decision)
            if self.edge_transfer_controller is not None:
                self.edge_transfer_controller.after_optimizer_step(edge_transfer_step)

            if mean_grad_norm is not None:
                self._child_spawn_step(mean_grad_norm)

            if need_mcmc_step:
                self._mcmc_step()
            self._mcmc_noise_step(it)

            if it % self.opt.memory_poll_interval == 0 or it == self.opt.iterations:
                peak_cpu_gb = max(peak_cpu_gb, self._cpu_mem_gb())
                peak_gpu_used_gb = max(peak_gpu_used_gb, self._gpu_mem_gb())
            if self.wandb_run is not None:
                log_dict: dict[str, float | int] = {
                    "loss": float(loss.item()),
                    "l1": float(l1.item()),
                }
                if bool(getattr(self.opt, "enable_mcmc", False)) or bool(getattr(self.opt, "enable_densify", False)):
                    log_dict["num_base"] = int(self.model.num_base)
                    log_dict["num_active_base"] = int(self.model.num_active_base)
                self.wandb_run.log(log_dict, step=int(it))

            if it in self.test_iterations:
                train_psnr, train_ssim = self._eval_psnr_ssim(self._select_eval_train_cameras())
                test_psnr, test_ssim = self._eval_psnr_ssim(list(self.test_cameras))
                self.logger.info(
                    "Eval: iter=%d train_psnr=%.2f train_ssim=%.4f test_psnr=%.2f test_ssim=%.4f",
                    int(it),
                    float(train_psnr),
                    float(train_ssim),
                    float(test_psnr),
                    float(test_ssim),
                )
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {
                            "eval/train_psnr": float(train_psnr),
                            "eval/train_ssim": float(train_ssim),
                            "eval/test_psnr": float(test_psnr),
                            "eval/test_ssim": float(test_ssim),
                        },
                        step=int(it),
                    )
                    self._wandb_log_test_samples(it)

            if it % self.opt.log_every == 0 or it == 1:
                gpu_mem = self._gpu_mem_gb()
                overdraw_alpha_mean = None
                overdraw_hits_mean = None
                if "overdraw" in render_pkg:
                    overdraw_alpha_mean = float(render_pkg["overdraw"].mean().item())
                if "overdraw_hits" in render_pkg:
                    overdraw_hits_mean = float(render_pkg["overdraw_hits"].mean().item())

                msg = "iter=%d loss=%.6f l1=%.6f gpu=%.2fGB"
                msg_args = [
                    it,
                    loss.item(),
                    l1.item(),
                    gpu_mem,
                ]
                if overdraw_alpha_mean is not None:
                    msg += " ov_a=%.3f"
                    msg_args.append(overdraw_alpha_mean)
                if overdraw_hits_mean is not None:
                    msg += " ov_h=%.3f"
                    msg_args.append(overdraw_hits_mean)
                if surface_coplanar is not None:
                    msg += " cop=%.4g"
                    msg_args.append(float(surface_coplanar.item()))
                if surface_normal_smooth is not None:
                    msg += " ns=%.4g"
                    msg_args.append(float(surface_normal_smooth.item()))
                if surface_normal_depth is not None:
                    msg += " nd=%.4g"
                    msg_args.append(float(surface_normal_depth.item()))
                if ray_distortion is not None:
                    msg += " dist=%.4g"
                    msg_args.append(float(ray_distortion.item()))
                if resume_surface_anchor is not None:
                    msg += " sa=%.4g"
                    msg_args.append(float(resume_surface_anchor.item()))
                if mvs_surface is not None:
                    msg += " md=%.4g ms=%.4g"
                    msg_args.extend(
                        (
                            float(mvs_surface.depth.item()),
                            float(mvs_surface.foreground_spill.item()),
                        )
                    )
                if child_rot_delta is not None:
                    msg += " cr=%.4g"
                    msg_args.append(float(child_rot_delta.item()))
                if coverage_loss is not None:
                    msg += " cov=%.4g"
                    msg_args.append(float(coverage_loss.item()))
                self.logger.info(msg, *msg_args)
                if self.wandb_run is not None:
                    log_dict = {
                        "loss": loss.item(),
                        "l1": l1.item(),
                        "gpu_gb": gpu_mem,
                    }
                    if repulsion_screen is not None:
                        log_dict["repulsion_screen"] = repulsion_screen.item()
                    if overdraw_alpha_mean is not None:
                        log_dict["overdraw_mean"] = overdraw_alpha_mean
                    if overdraw_hits_mean is not None:
                        log_dict["overdraw_hits_mean"] = overdraw_hits_mean
                    if repulsion_knn is not None:
                        log_dict["repulsion_knn"] = repulsion_knn.item()
                    if surface_coplanar is not None:
                        log_dict["surface_coplanar"] = surface_coplanar.item()
                    if surface_normal_smooth is not None:
                        log_dict["surface_normal_smooth"] = surface_normal_smooth.item()
                    if surface_normal_depth is not None:
                        log_dict["surface_normal_depth"] = surface_normal_depth.item()
                    if ray_distortion is not None:
                        log_dict["ray_distortion"] = ray_distortion.item()
                        log_dict["ray_distortion_weight"] = ray_distortion_weight
                    if resume_surface_anchor is not None:
                        log_dict["surface_anchor"] = resume_surface_anchor.item()
                    if mvs_surface is not None:
                        log_dict["mvs_surface_depth"] = mvs_surface.depth.item()
                        log_dict["mvs_surface_spill"] = (
                            mvs_surface.foreground_spill.item()
                        )
                        log_dict["mvs_surface_abs_relative_depth"] = (
                            mvs_surface.fixed_mean_absolute_relative_depth.item()
                        )
                        log_dict["mvs_surface_spill_fraction"] = (
                            mvs_surface.fixed_foreground_spill_fraction.item()
                        )
                        log_dict["mvs_surface_valid_alpha"] = (
                            mvs_surface.valid_alpha.item()
                        )
                    if child_rot_delta is not None:
                        log_dict["child_rot_delta"] = child_rot_delta.item()
                    if coverage_loss is not None:
                        log_dict["densify_coverage"] = coverage_loss.item()
                    self.wandb_run.log(log_dict, step=it)

            if _should_render_progress(
                it,
                iterations=int(self.opt.iterations),
                render_every=int(self.opt.render_every),
            ):
                with torch.no_grad():
                    eval_cam = self._progress_render_camera()
                    eval_bg = self.bg_color
                    eval_pred = self.renderer.render(
                        eval_cam,
                        self._render_model(),
                        eval_bg,
                    )["image"]
                    _save_image(self.model_path / "renders" / f"iter_{it:06d}.png", eval_pred)

            if it % self.opt.save_every == 0 or it == self.opt.iterations:
                self._save_checkpoint(it)

        self.logger.info("Training finished in %.2f min.", (time.time() - start_t) / 60.0)
        if self.wandb_run is not None:
            self.wandb_run.finish()
        summary = {
            "train_seconds": float(time.time() - start_t),
            "peak_cpu_gb": float(peak_cpu_gb),
            "peak_gpu_used_gb": float(peak_gpu_used_gb),
            "memory_poll_interval": self.opt.memory_poll_interval,
            "num_base": int(self.model.num_base),
            "num_active_base": int(self.model.num_active_base),
            "num_children": int(self.model.num_children),
            "num_primitives": int(self.model.num_base + self.model.num_children),
            "num_active_primitives": int(self.model.num_active_base + self.model.num_children),
            "model_parameter_count": int(sum(parameter.numel() for parameter in self.model.parameters())),
            "trainable_parameter_count": int(
                sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
            ),
            "effective_optimizer_parameter_count": int(
                sum(
                    group["params"][0].numel()
                    for group in self.optimizer.param_groups
                    if str(group["name"])
                    in self._resume_trainable_group_names()
                )
            ),
            "model_state_bytes": int(
                sum(value.numel() * value.element_size() for value in self.model.state_dict().values())
            ),
            "resume": self._resume_metadata(),
            "train_camera_trace_count": len(self.train_camera_trace_names),
            "train_camera_trace_sha256": _json_sha256(self.train_camera_trace_names),
            "final_learning_rates": self._optimizer_lr_map(),
        }
        if torch.cuda.is_available():
            summary["peak_gpu_allocated_gb"] = float(torch.cuda.max_memory_allocated() / 1024**3)
            summary["peak_gpu_reserved_gb"] = float(
                torch.cuda.max_memory_reserved() / 1024**3
            )
        if self.train_camera_selection is not None:
            summary["train_camera_selection"] = self.train_camera_selection
        init_geometry_metadata = getattr(self, "init_geometry_metadata", None)
        geometry_start_iteration = int(getattr(self, "geometry_start_iteration", 0))
        if init_geometry_metadata is not None:
            summary["initial_geometry"] = {
                **init_geometry_metadata,
                "geometry_start_iteration": geometry_start_iteration,
            }
        elif geometry_start_iteration > 0:
            summary["geometry_start_iteration"] = geometry_start_iteration
        edge_transfer_controller = getattr(self, "edge_transfer_controller", None)
        if edge_transfer_controller is not None:
            summary["edge_transfer"] = edge_transfer_controller.state()
        if getattr(self, "densify_has_spawned_child", None) is not None:
            summary["densification"] = {
                "repeat_parent": bool(getattr(self.opt, "densify_repeat_parent", False)),
                "rows_that_spawned": int(self.densify_has_spawned_child.sum().item()),
            }
        if self.enable_densify:
            summary["population_budget"] = dict(
                limit=int(getattr(self.opt, "densify_cap_max", 0)),
                peak=int(self.densify_peak_population),
                pruning=self.densify_prune_audit,
                gradient_scores=self.densify_score_audit,
            )
        if getattr(self, "densify_shadow_child_mask", None) is not None:
            summary["densification_shadow"] = self._densify_shadow_summary()
        if getattr(self, "densify_merge_decisions", []):
            summary["densification_merges"] = copy.deepcopy(
                self.densify_merge_decisions
            )
        joint_coplanar_audit = self._densify_joint_coplanar_audit()
        if (
            joint_coplanar_audit["pending"] is not None
            or joint_coplanar_audit["last_completed"] is not None
        ):
            summary["densification_joint_coplanar"] = joint_coplanar_audit
        (self.model_path / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _save_checkpoint(self, iteration: int) -> None:
        path = self.model_path / "checkpoints" / f"ckpt_{iteration:06d}.pt"
        checkpoint: dict[str, object] = {
            "iteration": iteration,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "resume_geometry_policy": str(
                getattr(self, "resume_geometry_policy", "inherit")
            ),
            "resume_parameter_policy": str(
                getattr(self, "resume_parameter_policy", "inherit")
            ),
            "resume_compatible": not any(
                (
                    bool(getattr(self.opt, "enable_mcmc", False)),
                    bool(getattr(self.opt, "enable_densify", False)),
                    bool(getattr(self.opt, "enable_children", False)),
                    bool(getattr(self.opt, "random_background", False)),
                    float(getattr(self.opt, "repulsion_knn_weight", 0.0)) > 0.0,
                    float(getattr(self.opt, "surface_coplanar_weight", 0.0)) > 0.0,
                    float(getattr(self.opt, "surface_normal_smooth_weight", 0.0)) > 0.0,
                )
            ),
        }
        scale_constraint = self._resume_scale_constraint_summary()
        if scale_constraint is not None:
            checkpoint["resume_scale_constraint"] = scale_constraint
        normal_trust = self._resume_normal_trust_summary()
        if normal_trust is not None:
            checkpoint["resume_normal_trust"] = normal_trust
        surface_anchor = self._resume_surface_anchor_summary()
        if surface_anchor is not None:
            checkpoint["resume_surface_anchor"] = surface_anchor
        mvs_surface = self._resume_mvs_surface_summary()
        if mvs_surface is not None:
            checkpoint["resume_mvs_surface"] = mvs_surface
        init_geometry_metadata = getattr(self, "init_geometry_metadata", None)
        if init_geometry_metadata is not None:
            checkpoint["init_geometry"] = init_geometry_metadata
        edge_transfer_controller = getattr(self, "edge_transfer_controller", None)
        if edge_transfer_controller is not None:
            checkpoint["edge_transfer"] = edge_transfer_controller.state()
            checkpoint["resume_compatible"] = edge_transfer_controller.is_complete()
            checkpoint["checkpoint_type"] = (
                "hexels3d_edge_transfer_complete"
                if edge_transfer_controller.is_complete()
                else "hexels3d_edge_transfer_incomplete"
            )
        elif self.resume_checkpoint_edge_transfer is not None:
            checkpoint["edge_transfer"] = copy.deepcopy(self.resume_checkpoint_edge_transfer)
        if getattr(self, "densify_has_spawned_child", None) is not None:
            checkpoint["densification_audit"] = {
                "schema_version": 1,
                "has_spawned_child": self.densify_has_spawned_child.detach().clone(),
            }
        if getattr(self, "densify_shadow_child_mask", None) is not None:
            checkpoint["densification_shadow_audit"] = self._densify_shadow_checkpoint_audit()
        if getattr(self, "densify_merge_decisions", []):
            checkpoint["densification_merge_audit"] = copy.deepcopy(
                self.densify_merge_decisions
            )
        joint_coplanar_audit = self._densify_joint_coplanar_audit()
        if (
            joint_coplanar_audit["pending"] is not None
            or joint_coplanar_audit["last_completed"] is not None
        ):
            checkpoint["densification_joint_coplanar_audit"] = joint_coplanar_audit
        parent_lineage = getattr(self, "resume_checkpoint_lineage", None)
        parent_schema_version = getattr(self, "resume_checkpoint_schema_version", None)
        parent_checkpoint_type = getattr(self, "resume_checkpoint_type", None)
        if parent_schema_version is not None:
            checkpoint["schema_version"] = parent_schema_version
        if parent_checkpoint_type is not None and edge_transfer_controller is None:
            checkpoint["checkpoint_type"] = parent_checkpoint_type
        resume_checkpoint_path = getattr(self, "resume_checkpoint_path", None)
        if resume_checkpoint_path is not None:
            if parent_lineage is None:
                parent_lineage = {
                    "kind": "hexels3d_resume_root",
                    "parent_checkpoint": str(resume_checkpoint_path),
                    "parent_checkpoint_sha256": self.resume_checkpoint_sha256,
                }
            if edge_transfer_controller is None:
                lineage = copy.deepcopy(parent_lineage)
            else:
                lineage = {
                    "kind": checkpoint["checkpoint_type"],
                    "status": "ungated",
                    "parent_lineage": copy.deepcopy(parent_lineage),
                    "edge_transfer": checkpoint["edge_transfer"],
                }
            continuations = lineage.setdefault("continuations", [])
            if not isinstance(continuations, list):
                raise ValueError("Resume checkpoint lineage continuations must be a list")
            continuations.append(
                {
                    "parent_checkpoint": str(resume_checkpoint_path),
                    "parent_checkpoint_sha256": self.resume_checkpoint_sha256,
                    "start_iteration": int(self.start_iteration),
                    "saved_iteration": int(iteration),
                    "geometry_policy": str(
                        getattr(self, "resume_geometry_policy", "inherit")
                    ),
                    "scale_constraint": scale_constraint,
                    "normal_trust": normal_trust,
                    "surface_anchor": surface_anchor,
                    "mvs_surface": mvs_surface,
                }
            )
            checkpoint["lineage"] = lineage
        torch.save(checkpoint, path)
        self.logger.info("Saved checkpoint: %s", path)
