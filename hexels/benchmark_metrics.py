from __future__ import annotations

import ast
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WARMUP_PASSES = 2
TIMED_PASSES = 10
TWO_DGS_PIPELINE = {"compute_cov3D_python":False,"convert_SHs_python":False,"depth_ratio":0,"debug":False}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must name a path")
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def normalize_cuda_device(device: torch.device) -> torch.device:
    if device.type != "cuda" or device.index not in {None, 0}:
        raise ValueError("Evaluation requires logical CUDA device 0")
    return torch.device("cuda", 0)


def parse_namespace_config(text: str) -> dict[str, object]:
    """Parse the small argparse Namespace written by 2DGS without using eval."""
    expression = ast.parse(text, mode="eval").body
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "Namespace"
        or expression.args
    ):
        raise ValueError("2DGS cfg_args must contain one keyword-only Namespace call")
    values: dict[str, object] = {}
    for keyword in expression.keywords:
        if keyword.arg is None or keyword.arg in values:
            raise ValueError("2DGS cfg_args contains unpacking or duplicate fields")
        try:
            values[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError) as error:
            raise ValueError(f"2DGS cfg_args field {keyword.arg} is not literal") from error
    return values


def _camera_name(camera: object) -> str:
    name = getattr(camera, "image_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Every camera must have a non-empty image_name")
    return name


def camera_geometry(camera: object) -> dict[str, object]:
    return {
        "name": _camera_name(camera),
        "width": int(getattr(camera, "image_width")),
        "height": int(getattr(camera, "image_height")),
        "fov_x": float(getattr(camera, "FoVx")),
        "fov_y": float(getattr(camera, "FoVy")),
        "world_view_transform": camera.world_view_transform.detach().cpu().tolist(),
        "full_proj_transform": camera.full_proj_transform.detach().cpu().tolist(),
        "camera_center": camera.camera_center.detach().cpu().tolist(),
    }


def _camera_target(camera: object) -> torch.Tensor:
    target = getattr(camera, "original_image", None)
    if not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] < 3:
        raise ValueError(f"Camera {_camera_name(camera)} has an invalid target tensor")
    if target.dtype != torch.float32:
        raise ValueError(f"Camera {_camera_name(camera)} target is not float32")
    target = target[:3].detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(target).all()):
        raise ValueError(f"Camera {_camera_name(camera)} target is not finite")
    return target


def camera_contract(cameras: Sequence[object], *, include_names: bool) -> dict[str, object]:
    if not cameras:
        raise ValueError("Camera contract cannot be empty")
    names = [_camera_name(camera) for camera in cameras]
    if len(set(names)) != len(names):
        raise ValueError("Camera names must be unique")
    geometry = [camera_geometry(camera) for camera in cameras]
    targets = [
        [name, _tensor_sha256(_camera_target(camera))]
        for name, camera in zip(names, cameras)
    ]
    result: dict[str, object] = {
        "count": len(cameras),
        "ordered_names_sha256": _json_sha256(names),
        "ordered_geometry_sha256": _json_sha256(geometry),
        "ordered_target_tensors_sha256": _json_sha256(targets),
    }
    if include_names:
        result["ordered_names"] = names
    return result


def validate_camera_sets(
    hexels_cameras: Sequence[object],
    two_dgs_cameras: Sequence[object],
    expected: Mapping[str, Any],
    *,
    include_names: bool,
) -> dict[str, object]:
    if len(hexels_cameras) != len(two_dgs_cameras):
        raise ValueError("Hexels and 2DGS camera counts differ")
    for index, (hexels_camera, two_dgs_camera) in enumerate(
        zip(hexels_cameras, two_dgs_cameras)
    ):
        if camera_geometry(hexels_camera) != camera_geometry(two_dgs_camera):
            raise ValueError(f"Hexels and 2DGS camera geometry differs at index {index}")
        if not torch.equal(_camera_target(hexels_camera), _camera_target(two_dgs_camera)):
            raise ValueError(f"Hexels and 2DGS target tensor differs at index {index}")
    observed = camera_contract(hexels_cameras, include_names=include_names)
    if dict(expected) != observed:
        raise ValueError("Live camera contract differs from the frozen protocol")
    return observed


def _ply_vertex_count(path: Path) -> int:
    count: int | None = None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("2DGS artifact is not a PLY file")
        for _ in range(10_000):
            line = handle.readline()
            if not line:
                raise ValueError("PLY header has no end_header marker")
            fields = line.strip().split()
            if len(fields) == 3 and fields[:2] == [b"element", b"vertex"]:
                if count is not None:
                    raise ValueError("PLY header contains multiple vertex elements")
                try:
                    count = int(fields[2])
                except ValueError as error:
                    raise ValueError("PLY vertex count is not an integer") from error
            if fields == [b"end_header"]:
                break
        else:
            raise ValueError("PLY header is unreasonably long")
    if count is None or count < 1:
        raise ValueError("PLY header has no positive vertex count")
    return count


def _add_2dgs_to_path() -> None:
    path = str(REPO_ROOT / "2d-gaussian-splatting")
    if path not in sys.path:
        sys.path.insert(0, path)


def validate_two_dgs_camera_archive(
    path: Path,
    test_camera_infos: Sequence[object],
    train_camera_infos: Sequence[object],
) -> None:
    from utils.camera_utils import camera_to_JSON

    observed = json.loads(path.read_text())
    if not isinstance(observed, list):
        raise ValueError("2DGS cameras.json must contain a list")
    ordered_infos = [*test_camera_infos, *train_camera_infos]
    expected = [camera_to_JSON(index, camera) for index, camera in enumerate(ordered_infos)]
    validate_two_dgs_camera_records(observed, expected)


def validate_two_dgs_camera_records(observed: object, expected: object) -> None:
    if not isinstance(observed, list) or observed != expected:
        raise ValueError("2DGS cameras.json differs from the live ordered camera archive")


def _load_independent_cameras(
    dataset: Mapping[str, Any],
    hexels_config: Mapping[str, Any],
    two_dgs_config: Mapping[str, Any],
    two_dgs_cameras_path: Path | None,
) -> tuple[list[object], list[object], list[object], list[object]]:
    from hexels.colmap import load_colmap_scene

    dataset_path = _resolve_path(dataset["path"], "dataset.path")
    hexels_options = SimpleNamespace(**dict(hexels_config))
    hexels_options.source_path = str(dataset_path)
    hexels_options.data_device = "cpu"
    hexels_scene = load_colmap_scene(hexels_options)

    _add_2dgs_to_path()
    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos

    scene_info = readColmapSceneInfo(
        str(dataset_path),
        two_dgs_config.get("images"),
        bool(two_dgs_config["eval"]),
        llffhold=int(dataset["llffhold"]),
    )
    if two_dgs_cameras_path is not None:
        validate_two_dgs_camera_archive(
            two_dgs_cameras_path, scene_info.test_cameras, scene_info.train_cameras,
        )
    camera_args = SimpleNamespace(
        resolution=int(two_dgs_config["resolution"]),
        data_device="cpu",
    )
    two_dgs_train = cameraList_from_camInfos(scene_info.train_cameras, 1.0, camera_args)
    two_dgs_eval = cameraList_from_camInfos(scene_info.test_cameras, 1.0, camera_args)
    return (
        list(hexels_scene.train_cameras),
        list(hexels_scene.test_cameras),
        list(two_dgs_train),
        list(two_dgs_eval),
    )


def validate_camera_device(cameras: Sequence[object], device: torch.device) -> None:
    expected = normalize_cuda_device(device)
    for camera in cameras:
        for field in (
            "world_view_transform",
            "projection_matrix",
            "full_proj_transform",
            "camera_center",
        ):
            value = getattr(camera, field, None)
            if not isinstance(value, torch.Tensor) or value.device != expected:
                raise ValueError(f"Camera {_camera_name(camera)} {field} is not on logical cuda:0")


def _load_hexels_model(
    checkpoint_path: Path,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Any, torch.Tensor]:
    from hexels.model import Hexels3DGeometry, Hexels3DModel, validate_active_mask_state
    from hexels.render import _checkpoint_view_basis_dim, _prepare_render_model_state
    from hexels.renderer import Hexels3DRenderer

    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location=device, weights_only=True),
        "Hexels checkpoint",
    )
    state = _mapping(checkpoint["model"], "Hexels checkpoint model")
    geometry = Hexels3DGeometry(
        means3d=state["means3d"].to(device),
        scales=torch.exp(state["log_scales"].to(device)),
        rotations=state["rotation_quats"].to(device),
    )
    model = Hexels3DModel(
        geometry=geometry,
        colors_init=state["colors"].to(device),
        init_opacity=0.1,
        init_softness=float(config["softness"]),
        softness_mode=str(config["softness_mode"]),
        view_basis_dim=_checkpoint_view_basis_dim(dict(state), str(config["view_model"])),
    ).to(device)
    children = state.get("child_parent_idx", torch.empty(0))
    model.allocate_children(int(children.numel()))
    prepared_state = _prepare_render_model_state(checkpoint, model, SimpleNamespace(**dict(config)))
    validate_active_mask_state(prepared_state, model.num_base)
    incompatible = model.load_state_dict(prepared_state, strict=False)
    allowed_missing = {"active_mask", "child_rotvec", "child_layer"}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) - allowed_missing:
        raise ValueError(f"Hexels checkpoint/model mismatch: {incompatible}")
    model.eval()
    renderer = Hexels3DRenderer(SimpleNamespace(**dict(config)))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if config["white_background"] else [0.0, 0.0, 0.0],
        device=device,
        dtype=torch.float32,
    )
    return model, renderer, background


def _load_two_dgs_model(
    ply_path: Path,
    config: Mapping[str, Any],
    expected_primitives: int,
) -> tuple[Any, Any, Any, torch.Tensor]:
    _add_2dgs_to_path()
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel

    model = GaussianModel(int(config["sh_degree"]))
    model.load_ply(str(ply_path))
    if int(model.get_xyz.shape[0]) != expected_primitives or model.active_sh_degree != int(config["sh_degree"]):
        raise ValueError("Loaded 2DGS model differs from the frozen PLY contract")
    pipeline = SimpleNamespace(**TWO_DGS_PIPELINE)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if config["white_background"] else [0.0, 0.0, 0.0],
        device="cuda",
        dtype=torch.float32,
    )
    return model, render, pipeline, background


def float_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    ssim_fn: Any,
    lpips_fn: Any,
) -> dict[str, object]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[0] != 3:
        raise ValueError("Prediction and target must be matching 3-channel CHW tensors")
    if prediction.dtype != torch.float32 or target.dtype != torch.float32:
        raise ValueError("Prediction and target must both be float32")
    if prediction.device != target.device:
        raise ValueError("Prediction and target must be on the same device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("Prediction and target must be finite")
    raw_sha256 = _tensor_sha256(prediction)
    clipped_element_count = int(((prediction < 0.0) | (prediction > 1.0)).sum().item())
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    prediction_cpu = prediction.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    target_cpu = target.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    difference = prediction_cpu.astype(np.float64) - target_cpu.astype(np.float64)
    sse = float(np.sum(np.square(difference), dtype=np.float64))
    if not math.isfinite(sse) or sse <= 0.0:
        raise ValueError("Direct-float RGB SSE must be positive and finite")
    values = {
        "rgb_sse": sse,
        "psnr_db": -10.0 * math.log10(sse / prediction_cpu.size),
        "ssim": float(ssim_fn(prediction, target).item()),
        "lpips_vgg": float(
            lpips_fn(prediction[None] * 2.0 - 1.0, target[None] * 2.0 - 1.0).item()
        ),
        "raw_float_sha256": raw_sha256,
        "metric_float_sha256": _tensor_sha256(prediction),
        "clipped_element_count": clipped_element_count,
    }
    if not all(math.isfinite(float(values[key])) for key in ("psnr_db", "ssim", "lpips_vgg")):
        raise ValueError("PSNR, SSIM, or LPIPS is not finite")
    return values


def _artifact_record(path: Path) -> dict[str, int | str]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _model_tensor_bytes(state: Mapping[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("No metric rows")
    psnr = sorted(float(row["psnr_db"]) for row in rows)
    tail_count = math.ceil(0.1 * len(rows))
    elements = sum(int(row["elements"]) for row in rows)
    sse = sum(float(row["rgb_sse"]) for row in rows)
    return {
        "view_count": len(rows),
        "mean_psnr_db": statistics.fmean(psnr),
        "pooled_psnr_db": -10.0 * math.log10(sse / elements),
        "mean_ssim": statistics.fmean(float(row["ssim"]) for row in rows),
        "mean_lpips_vgg": statistics.fmean(
            float(row["lpips_vgg"]) for row in rows
        ),
        "worst_psnr_db": psnr[0],
        "bottom_decile_psnr_db": statistics.fmean(psnr[:tail_count]),
    }


@torch.inference_mode()
def _benchmark(
    render: Callable[[Any], torch.Tensor],
    cameras: Sequence[Any],
    device: torch.device,
) -> dict[str, float | int]:
    for _ in range(WARMUP_PASSES):
        for camera in cameras:
            render(camera)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(TIMED_PASSES):
        for camera in cameras:
            render(camera)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    frames = TIMED_PASSES * len(cameras)
    return {
        "warmup_passes": WARMUP_PASSES,
        "timed_passes": TIMED_PASSES,
        "frames": frames,
        "seconds": seconds,
        "fps": frames / seconds,
        "latency_ms": 1000.0 * seconds / frames,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


@torch.inference_mode()
def _metrics(
    render: Callable[[Any], torch.Tensor],
    cameras: Sequence[Any],
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    _add_2dgs_to_path()
    from utils.loss_utils import ssim

    import lpips

    lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
    rows: list[dict[str, Any]] = []
    for camera in cameras:
        prediction = render(camera)
        target = camera.original_image[:3].to(device)
        values = float_metrics(prediction, target, ssim, lpips_model)
        rows.append(
            {
                "image_name": str(camera.image_name),
                "elements": int(target.numel()),
                "rgb_sse": values["rgb_sse"],
                "psnr_db": values["psnr_db"],
                "ssim": values["ssim"],
                "lpips_vgg": values["lpips_vgg"],
                "clipped_element_count": values["clipped_element_count"],
                "raw_float_sha256": values["raw_float_sha256"],
            }
        )
    return _aggregate(rows), rows
