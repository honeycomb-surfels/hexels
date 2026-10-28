from __future__ import annotations

import copy
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tap import Tap

from .colmap import load_colmap_scene
from .compact_checkpoint import (
    METADATA_KEY,
    expand_compact_appearance_state,
    infer_appearance_layout,
    validate_appearance_state,
    validate_deployment_checkpoint,
)
from .model import Hexels3DGeometry, Hexels3DModel, validate_active_mask_state
from .renderer import Hexels3DRenderer


_CONFIG_RENDER_FIELDS = {
    "source_path",
    "images",
    "eval",
    "llffhold",
    "resolution",
    "softness",
    "softness_mode",
    "scale_modifier",
    "color_model",
    "use_view_grads",
    "view_model",
    "prefilter_edge_on",
    "child_offset_frac",
    "child_scale_ratio",
    "child_rotation_delta",
    "children_detach_geometry",
    "white_background",
    "random_background",
    "debug",
}


class RenderOptions(Tap):
    checkpoint: str = ""
    model_path: Optional[str] = None
    out_dir: Optional[str] = None
    split: str = "test"  # "train" | "test"
    max_views: int = 0
    camera_stride: int = 1
    camera_offset: int = 0
    camera_manifest: Optional[str] = None
    skip_test: bool = False
    skip_path: bool = False
    traj_frames: int = 300
    traj_fps: float = 30.0
    supersample: int = 1
    sh_degree: int = 3

    # Loaded from config.json if present, unless given on the command line
    source_path: str = ""
    images: Optional[str] = None
    eval: bool = True
    llffhold: int = 8
    resolution: int = -1
    data_device: str = "cuda"
    softness: float = 10.0
    softness_mode: str = "iso"
    scale_modifier: float = 1.0
    color_model: str = "linear"
    use_view_grads: Optional[bool] = True
    view_model: str = "linear"
    prefilter_edge_on: bool = False
    child_offset_frac: float = 0.5
    child_scale_ratio: float = 0.5
    child_rotation_delta: bool = False
    children_detach_geometry: Optional[bool] = True
    white_background: bool = False
    random_background: bool = False
    debug: bool = False

    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        known_only: bool = False,
        legacy_config_parsing: bool = False,
    ) -> "RenderOptions":
        """Remember the flags given on the command line so they take precedence over config.json."""
        command_line = sys.argv[1:] if args is None else args
        self._explicit_flags = {item[2:].split("=", 1)[0] for item in command_line if item.startswith("--")}
        return super().parse_args(args, known_only, legacy_config_parsing)

    def process_args(self) -> None:
        if not self.checkpoint:
            raise ValueError("--checkpoint is required")

        ckpt_path = Path(self.checkpoint).resolve()
        if self.model_path is None:
            self.model_path = str(ckpt_path.parents[1])

        cfg_path = Path(self.model_path) / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            explicit = getattr(self, "_explicit_flags", set())
            for name in _CONFIG_RENDER_FIELDS:
                if name in cfg and name not in explicit:
                    setattr(self, name, cfg[name])

        if not self.source_path:
            raise ValueError("--source_path is required (or provide a checkpoint under a model dir with config.json)")
        if self.split not in {"train", "test"}:
            raise ValueError("--split must be one of: train, test")
        if self.view_model not in {"none", "linear", "sh3", "lobe"}:
            raise ValueError("--view_model must be one of: none, linear, sh3, lobe")
        if self.supersample < 1:
            raise ValueError("--supersample must be at least 1")
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("--sh_degree must be between 0 and 3")
        if self.camera_stride < 1:
            raise ValueError("--camera_stride must be at least 1")
        if self.camera_offset < 0 or self.camera_offset >= self.camera_stride:
            raise ValueError("--camera_offset must satisfy 0 <= offset < camera_stride")
        if (self.camera_stride != 1 or self.camera_offset != 0) and self.out_dir is None:
            raise ValueError("--out_dir is required when selecting a camera subset")
        if self.camera_manifest is not None and self.out_dir is None:
            raise ValueError("--out_dir is required with --camera_manifest")
        if self.sh_degree != 3:
            if self.view_model != "sh3":
                raise ValueError("--sh_degree below 3 requires --view_model sh3")
            if self.out_dir is None:
                raise ValueError("--out_dir is required when --sh_degree is below 3")
            if not self.skip_path:
                raise ValueError("--skip_path is required when --sh_degree is below 3")


def _save_image(path: Path, image_chw: torch.Tensor) -> None:
    from PIL import Image

    img = image_chw.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def _save_depth_tiff(path: Path, depth_hw: torch.Tensor) -> None:
    from PIL import Image

    depth = depth_hw.detach().float().cpu().numpy()
    Image.fromarray(np.nan_to_num(depth).astype(np.float32)).save(path)


def _select_split_cameras(scene: object, split: str) -> list[object]:
    cams = scene.test_cameras if split == "test" else scene.train_cameras
    if not cams:
        raise ValueError(
            f"No {split} cameras are available; train with --eval for a held-out test split "
            "or render explicitly with --split train"
        )
    return list(cams)


def _select_camera_subset(
    cameras: list[object],
    *,
    camera_stride: int,
    camera_offset: int,
) -> list[object]:
    if camera_stride == 1 and camera_offset == 0:
        return list(cameras)

    named: list[tuple[str, object]] = []
    for camera in cameras:
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Every selected render camera must have a non-empty image_name")
        named.append((name, camera))
    named.sort(key=lambda item: item[0])
    names = [name for name, _ in named]
    if len(set(names)) != len(names):
        raise ValueError("Selected render camera image_name values must be unique")
    selected = [camera for _, camera in named][camera_offset::camera_stride]
    if not selected:
        raise ValueError("Camera stride and offset select no render cameras")
    return selected


def _select_camera_manifest(
    cameras: list[object],
    manifest_path: str,
) -> list[object]:
    from .train_camera_manifest import select_train_cameras

    selected, _ = select_train_cameras(cameras, manifest_path)
    return selected


def _checkpoint_view_basis_dim(state: dict[str, torch.Tensor], view_model: str) -> int:
    view_grads = state.get("view_grads")
    if view_grads is None or view_grads.ndim != 3 or view_grads.shape[2] != 3:
        raise ValueError("Checkpoint view_grads must have shape (P,K,3)")
    basis_dim = int(view_grads.shape[1])
    expected = 15 if view_model == "sh3" else 3
    if basis_dim != expected:
        raise ValueError(
            f"Checkpoint view_grads basis dimension {basis_dim} is incompatible with view_model={view_model!r}; "
            f"expected {expected}"
        )
    return basis_dim


def _render_output_name(iteration: int, supersample: int) -> str:
    return _supersample_output_dir(Path(f"ours_{iteration}"), supersample).name


def _supersample_output_dir(out_dir: Path, supersample: int) -> Path:
    suffix = f"_ss{supersample}"
    if supersample == 1 or out_dir.name.endswith(suffix):
        return out_dir
    return out_dir.with_name(f"{out_dir.name}{suffix}")


def _split_render_output_dir(
    opt: RenderOptions,
    *,
    model_path: Path,
    iteration: int,
) -> Path:
    canonical = _supersample_output_dir(
        model_path / opt.split / f"ours_{iteration}",
        opt.supersample,
    )
    if opt.out_dir is None:
        return canonical

    out_dir = _supersample_output_dir(Path(opt.out_dir), opt.supersample)
    diagnostic = (
        opt.camera_stride != 1
        or opt.camera_offset != 0
        or opt.camera_manifest is not None
        or opt.sh_degree != 3
    )
    if diagnostic and out_dir.resolve() == canonical.resolve():
        raise ValueError("Camera-subset and SH-degree diagnostics cannot use the canonical render directory")
    return out_dir


def _scaled_camera(camera: object, supersample: int) -> object:
    if supersample == 1:
        return camera
    scaled = copy.copy(camera)
    scaled.image_width = int(camera.image_width) * supersample
    scaled.image_height = int(camera.image_height) * supersample
    return scaled


def _box_downsample(value: torch.Tensor, factor: int) -> torch.Tensor:
    if value.ndim not in {2, 3}:
        raise ValueError(f"Expected an HW or CHW tensor, got shape {tuple(value.shape)}")
    if factor < 1:
        raise ValueError(f"Downsampling factor must be at least 1, got {factor}")
    if factor == 1:
        return value
    height, width = value.shape[-2:]
    if height % factor != 0 or width % factor != 0:
        raise ValueError(f"Tensor shape {tuple(value.shape)} is not divisible by factor {factor}")
    batched = value[None, None] if value.ndim == 2 else value[None]
    pooled = F.avg_pool2d(batched, kernel_size=factor, stride=factor)
    return pooled[0, 0] if value.ndim == 2 else pooled[0]


def _prepare_render_model_state(
    checkpoint: Mapping[str, object],
    model: Hexels3DModel,
    opt: RenderOptions,
) -> dict[str, object]:
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint is missing its model state")
    layout = infer_appearance_layout(
        {
            "color_model": opt.color_model,
            "view_model": opt.view_model,
            "use_view_grads": opt.use_view_grads,
        }
    )
    if METADATA_KEY in checkpoint:
        if layout is None:
            raise ValueError("Compact checkpoint requires a V1, V2, or V3 render configuration")
        validate_deployment_checkpoint(checkpoint, layout)
        return expand_compact_appearance_state(state, model.state_dict(), layout)
    if layout is not None:
        validate_appearance_state(state, layout, inactive_storage="full")
    return dict(state)


@torch.no_grad()
def _render_views(
    *,
    cams: list[object],
    renderer: Hexels3DRenderer,
    model: Hexels3DModel,
    bg_color: torch.Tensor,
    out_dir: Path,
    save_gt: bool,
    supersample: int = 1,
    sh_degree: int = 3,
) -> None:
    renders_dir = out_dir / "renders"
    gt_dir = out_dir / "gt"
    vis_dir = out_dir / "vis"
    generated_dirs = [renders_dir, vis_dir]
    if save_gt:
        generated_dirs.append(gt_dir)
    for directory in generated_dirs:
        if directory.exists():
            shutil.rmtree(directory)
    for metrics_file in (out_dir / "results.json", out_dir / "per_view.json"):
        metrics_file.unlink(missing_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if save_gt:
        gt_dir.mkdir(parents=True, exist_ok=True)

    for idx, cam in enumerate(cams):
        render_cam = _scaled_camera(cam, supersample)
        pkg = renderer.render(render_cam, model, bg_color, return_surface=True, sh_degree=sh_degree)
        pred = _box_downsample(pkg["image"], supersample)
        surf_depth = _box_downsample(pkg["surf_depth"], supersample)
        surf_alpha = _box_downsample(pkg["surf_alpha"], supersample)
        depth = surf_depth / surf_alpha.clamp_min(1e-6)
        zfar = float(getattr(cam, "zfar", 100.0))
        depth = torch.where(surf_alpha > 0.0, depth, torch.full_like(depth, zfar))

        _save_image(renders_dir / f"{idx:05d}.png", pred)
        _save_depth_tiff(vis_dir / f"depth_{idx:05d}.tiff", depth)
        if save_gt:
            gt = cam.original_image[0:3]
            _save_image(gt_dir / f"{idx:05d}.png", gt)


def _load_render_components(
    opt: RenderOptions,
) -> tuple[int, Hexels3DModel, object, Hexels3DRenderer, torch.Tensor]:
    device = torch.device(opt.data_device)

    ckpt = torch.load(opt.checkpoint, map_location=device, weights_only=True)
    iteration = int(ckpt.get("iteration", -1))
    state = ckpt["model"]

    means3d = state["means3d"].to(device)
    scales = torch.exp(state["log_scales"].to(device))
    rotations = state["rotation_quats"].to(device)
    colors_init = state["colors"].to(device)
    view_basis_dim = _checkpoint_view_basis_dim(state, opt.view_model)

    geometry = Hexels3DGeometry(means3d=means3d, scales=scales, rotations=rotations)
    model = Hexels3DModel(
        geometry=geometry,
        colors_init=colors_init,
        init_opacity=0.1,
        init_softness=opt.softness,
        softness_mode=str(opt.softness_mode),
        view_basis_dim=view_basis_dim,
    ).to(device)
    model.allocate_children(int(state.get("child_parent_idx", torch.empty(0)).numel()))
    state = _prepare_render_model_state(ckpt, model, opt)
    validate_active_mask_state(state, model.num_base)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected keys in checkpoint: {incompatible.unexpected_keys}")
    missing = set(incompatible.missing_keys)
    allowed_missing = {"active_mask", "child_rotvec", "child_layer"}
    if missing - allowed_missing:
        raise ValueError(f"Missing keys in checkpoint: {sorted(missing)}")
    if model.child_parent_idx.numel() > 0 and model.child_layer.numel() == 0:
        model.child_layer = torch.ones_like(model.child_parent_idx, dtype=torch.long)
    model.eval()

    scene = load_colmap_scene(opt)
    renderer = Hexels3DRenderer(opt)

    bg_color = torch.tensor(
        [1.0, 1.0, 1.0] if opt.white_background else [0.0, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )
    return iteration, model, scene, renderer, bg_color


def _run(opt: RenderOptions) -> None:
    iteration, model, scene, renderer, bg_color = _load_render_components(opt)

    model_path = Path(opt.model_path) if opt.model_path is not None else Path(opt.checkpoint).resolve().parents[1]
    out_dir = _split_render_output_dir(opt, model_path=model_path, iteration=iteration)
    if not opt.skip_test:
        cams = _select_split_cameras(scene, opt.split)
        if opt.camera_manifest is not None:
            cams = _select_camera_manifest(cams, opt.camera_manifest)
        cams = _select_camera_subset(
            cams,
            camera_stride=opt.camera_stride,
            camera_offset=opt.camera_offset,
        )
        if opt.max_views > 0:
            cams = cams[: opt.max_views]
        _render_views(
            cams=cams,
            renderer=renderer,
            model=model,
            bg_color=bg_color,
            out_dir=out_dir,
            save_gt=True,
            supersample=opt.supersample,
            sh_degree=opt.sh_degree,
        )

    if not opt.skip_path:
        from utils.render_utils import create_videos, generate_path

        traj_dir = model_path / "traj" / _render_output_name(iteration, opt.supersample)
        cam_traj = generate_path(scene.train_cameras, n_frames=int(opt.traj_frames))
        _render_views(
            cams=cam_traj,
            renderer=renderer,
            model=model,
            bg_color=bg_color,
            out_dir=traj_dir,
            save_gt=False,
            supersample=opt.supersample,
            sh_degree=opt.sh_degree,
        )
        create_videos(
            base_dir=str(traj_dir),
            input_dir=str(traj_dir),
            out_name="render_traj",
            num_frames=int(opt.traj_frames),
            fps=float(opt.traj_fps),
            codec="h264",
            encoded_format="yuv420p",
        )


def run_render(
    *,
    checkpoint: str,
    model_path: str | None = None,
    split: str = "test",
    max_views: int = 0,
    traj_frames: int = 300,
    traj_fps: float = 30.0,
    skip_test: bool = False,
    skip_path: bool = False,
    supersample: int = 1,
) -> None:
    opt = RenderOptions()
    opt.checkpoint = checkpoint
    opt.model_path = model_path
    opt.split = split
    opt.max_views = max_views
    opt.traj_frames = traj_frames
    opt.traj_fps = traj_fps
    opt.skip_test = skip_test
    opt.skip_path = skip_path
    opt.supersample = supersample
    opt.process_args()
    _run(opt)


def main() -> None:
    opt = RenderOptions().parse_args()
    _run(opt)


if __name__ == "__main__":
    main()
