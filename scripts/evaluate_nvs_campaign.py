"""Shared direct-float metrics, render-only timing and fly-around export for NVS."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hexels import benchmark_metrics as quality
from scripts.nvs_trajectory import make_camera, orbit


ROOT = Path(__file__).resolve().parents[1]


def load_reference(settings_path: Path) -> tuple[list, dict, dict]:
    from scripts.evaluate_paper_nvs import _load_cameras

    settings = json.loads(settings_path.read_text())
    return _load_cameras(Path(settings["source_path"]), settings["llffhold"], None, None,
                         torch.device("cuda"), settings_path, with_geometry=True)


def prepare_cameras(settings_path: Path, output: Path, video_options: dict) -> None:
    from hexels.colmap import load_colmap_scene
    from types import SimpleNamespace

    if output.exists():
        raise FileExistsError(output)
    settings = json.loads(settings_path.read_text())
    scene = load_colmap_scene(SimpleNamespace(**settings))
    payload = {"settings": settings,
               "training": quality.camera_contract(scene.train_cameras, include_names=False),
               "test": quality.camera_contract(scene.test_cameras, include_names=True),
               "trajectory": orbit([quality.camera_geometry(camera) for camera in scene.train_cameras], video_options)}
    payload["geometry"] = {split: [quality.camera_geometry(camera) for camera in cameras]
                           for split, cameras in (("training", scene.train_cameras), ("test", scene.test_cameras))}
    if "/dtu/" in settings["source_path"]:
        from hexel_viewer.viewer_model import colmap_viewpoints
        from scripts.nvs_object_trajectory import object_orbit

        heldout = set(payload["test"]["ordered_names"])
        poses = [pose for pose in colmap_viewpoints(Path(settings["source_path"])) if Path(pose["name"]).stem not in heldout]
        payload["trajectory"] = object_orbit(poses, payload["trajectory"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def load_model(job: dict) -> tuple[object, dict, list, dict, dict | None]:
    """Load cameras and the model without writing outputs, so a failed check leaves nothing behind."""
    if job["method"] not in {"hexels", "2dgs"}:
        from scripts.nvs_external import load_model as load_external
        return (*load_external(job), None)
    from scripts.evaluate_paper_nvs import _load_hexels, _load_two_dgs

    cameras, contract, geometry = load_reference(Path(job["camera_settings"]))
    path = Path(job["model_path"])
    expected = json.loads(Path(job["camera_settings"]).read_text())
    actual = (json.loads((path / "config.json").read_text()) if job["method"] == "hexels"
              else quality.parse_namespace_config((path / "cfg_args").read_text()))
    for key in ("eval", "images", "resolution", "llffhold"):
        if actual.get(key, 8 if key == "llffhold" else None) != expected[key]:
            raise ValueError(f"Training camera configuration differs: {key}")
    if Path(actual["source_path"]).resolve() != Path(expected["source_path"]).resolve():
        raise ValueError("Training source differs from canonical input")
    if job["method"] == "hexels":
        summary_path = Path(job.get("summary_path", path / "summary.json"))
        render, artifact = _load_hexels(path, job["iteration"], [summary_path],
                                       job["bytes_per_primitive"], torch.device("cuda"))
        summary = json.loads(summary_path.read_text())
        artifact["peak_primitives"] = summary.get("population_budget", {}).get("peak", artifact["primitives"])
    else:
        render, artifact = _load_two_dgs(path, job["iteration"])
        artifact["deployment_files"] = [artifact["point_cloud"], artifact["config"]]
        artifact["serialized_model_bytes"] = sum(record["bytes"] for record in artifact["deployment_files"])
    return render, artifact, cameras, contract, geometry


def write_deployment(job: dict, artifact: dict) -> None:
    """Export the compact Hexel deployment state once all checks pass; a decoded archive already is one."""
    if job["method"] != "hexels":
        return
    from hexels.compact_checkpoint import export_compact_checkpoint

    path = Path(job["model_path"])
    checkpoint = path / f"checkpoints/ckpt_{job['iteration']:06d}.pt"
    if "deployment" in torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True):
        deployment = checkpoint
    else:
        deployment = Path(job["result"]).parent / "deployment.pt"
        export_compact_checkpoint(checkpoint, deployment, active_only=True)
    artifact["deployment_files"] = [quality._artifact_record(deployment), quality._artifact_record(path / "config.json")]
    artifact["serialized_model_bytes"] = sum(record["bytes"] for record in artifact["deployment_files"])


@torch.inference_mode()
def benchmark(render: object, cameras: list, options: dict) -> dict:
    for _ in range(options["warmup_passes"]):
        for camera in cameras:
            render(camera)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    trials = []
    for _ in range(options["timing_trials"]):
        started = time.perf_counter()
        for _ in range(options["timed_passes"]):
            for camera in cameras:
                render(camera)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        frames = len(cameras) * options["timed_passes"]
        trials.append({"seconds": seconds, "frames": frames, "fps": frames / seconds})
    return {"trials": trials, "median_fps": statistics.median(trial["fps"] for trial in trials),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "scope": "resident model, render-only synchronized wall time; no readback, encoding or metrics"}


def rgb_array(prediction: torch.Tensor) -> np.ndarray:
    return (prediction.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


@torch.inference_mode()
def export_video(render: object, trajectory: dict, output: Path) -> dict:
    import imageio_ffmpeg

    first = trajectory["frames"][0]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-n", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{first['width']}x{first['height']}", "-r", str(trajectory["fps"]),
               "-i", "pipe:0", "-an", "-c:v", trajectory["codec"], "-crf", str(trajectory["crf"]),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    started = time.perf_counter()
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        for frame in trajectory["frames"]:
            encoder.stdin.write(rgb_array(render(make_camera(frame, torch.device("cuda")))).tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"Video encoding failed: {output}")
    return {"seconds": time.perf_counter() - started, "frames": len(trajectory["frames"]),
            "fps": trajectory["fps"], "width": first["width"], "height": first["height"],
            "trajectory_sha256": quality._json_sha256(trajectory), "artifact": quality._artifact_record(output)}


@torch.no_grad()
def evaluate(job: dict) -> None:
    import lpips

    output = Path(job["result"])
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    render, artifact, cameras, contract, geometry = load_model(job)
    canonical = json.loads(Path(job["camera_contract"]).read_text())
    from scripts.nvs_camera_validation import validate as validate_cameras
    from scripts.nvs_budgets import validate as validate_budget

    if geometry is None:
        geometry_path = output.parent / "native_camera_geometry.json"
        geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else None
    camera_validation = validate_cameras(canonical, contract, geometry)
    if job.get("capacity_reference"):
        validate_budget(artifact, job["capacity_reference"])
    write_deployment(job, artifact)
    timing = benchmark(render, cameras, job["measurement"])
    specification = importlib.util.spec_from_file_location("reference_ssim", ROOT / "2d-gaussian-splatting/utils/loss_utils.py")
    reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reference)
    perceptual = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
    rows = []
    render_dir = output.parent / "renders"
    render_dir.mkdir(parents=True, exist_ok=False)
    metric_start = time.perf_counter()
    for index, camera in enumerate(cameras):
        prediction = render(camera)
        target = camera.original_image[:3].cuda()
        values = quality.float_metrics(prediction, target, reference.ssim, perceptual)
        rows.append({"image_name": str(camera.image_name), "elements": target.numel(), **values})
        Image.fromarray(rgb_array(prediction)).save(render_dir / f"{index:05d}.png")
    metric_seconds = time.perf_counter() - metric_start
    del perceptual
    video = export_video(render, canonical["trajectory"], output.parent / "flyaround.mp4")
    result = {"complete": True, "id": job["id"], "camera_contract": contract, "artifact": artifact,
              "camera_validation": camera_validation, "capacity_reference": job.get("capacity_reference"),
              "benchmark": timing, "aggregate": quality._aggregate(rows), "per_view": rows,
              "metric_and_png_seconds": metric_seconds, "video": video,
              "evaluation_seconds": time.perf_counter() - started,
              "hardware": {"name": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda},
              "provenance": {"evaluator": quality._artifact_record(Path(__file__)),
                             "metrics": quality._artifact_record(ROOT / "hexels/benchmark_metrics.py"),
                             "camera_contract": quality._artifact_record(Path(job["camera_contract"]))}}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["cameras", "evaluate"])
    parser.add_argument("--job", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nvs.json")
    parser.add_argument("--model-path", type=Path,
                        help="Evaluate another model directory, such as a decoded .hexcodec archive, under the job's cameras and protocol")
    parser.add_argument("--result", type=Path, help="Fresh metrics path, required with --model-path")
    args = parser.parse_args()
    if args.action == "cameras":
        prepare_cameras(args.settings, args.output, json.loads(args.config.read_text())["video"])
    else:
        job = json.loads(args.job.read_text())
        if args.model_path is not None:
            if args.result is None:
                raise ValueError("--model-path needs a fresh --result path")
            # A decoded archive has no training summary of its own; it shares the job's training run.
            summary = args.model_path / "summary.json"
            job.update(summary_path=str(summary if summary.is_file() else Path(job["model_path"]) / "summary.json"),
                       model_path=str(args.model_path), result=str(args.result))
        evaluate(job)


if __name__ == "__main__":
    main()
