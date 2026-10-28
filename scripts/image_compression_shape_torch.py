"""Four primitive controls sharing a pure-PyTorch normalized image renderer.

Weights are evaluated globally before top-ten selection, without CUDA tile
bounds or support cutoffs. Each pixel chunk backpropagates before the next
chunk is constructed; Adam updates once after the full image gradient.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def initial_fields(target: torch.Tensor, count: int, seed: int,
                   init_scale: float = 5.0) -> dict[str, torch.Tensor]:
    """Return deterministic CPU fields from uniform distinct pixel centers.

    The returned zero gradients can be omitted for flat RGB, leaving eight
    scalars per primitive. Keeping them gives fourteen scalars for linear RGB.
    """
    target = target.detach().cpu().float()
    height, width = target.shape[1:]
    if not 0 < count <= height * width:
        raise ValueError("Primitive count must fit the distinct pixel initialization")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(height * width, generator=generator)[:count]
    x, y = indices.remainder(width), indices.div(width, rounding_mode="floor")
    positions = (torch.stack((x, y), dim=-1).float() + 0.5) / torch.tensor([width, height])
    return {
        "positions": positions,
        "log_scales": torch.full((count, 2), math.log(init_scale), dtype=torch.float32),
        "rotations": torch.zeros(count, 1, dtype=torch.float32),
        "features": target[:, y, x].T.contiguous(),
        "color_gradients": torch.zeros(count, 2, 3, dtype=torch.float32),
    }


def pixel_coordinates(height: int, width: int, like: torch.Tensor) -> torch.Tensor:
    """Return flattened integer pixel coordinates in (x,y) order."""
    y, x = torch.meshgrid(torch.arange(height, device=like.device, dtype=like.dtype),
                          torch.arange(width, device=like.device, dtype=like.dtype), indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def render_pixels(values: dict[str, torch.Tensor], method: str, height: int,
                  width: int, pixels: torch.Tensor, *, topk: int = 10,
                  softness: float = 10.0) -> torch.Tensor:
    """Render RGB at selected pixels through a common top-K normalized blend."""
    centers = values["positions"] * values["positions"].new_tensor([width, height])
    delta = centers[None] - pixels[:, None]
    dx, dy = delta.unbind(-1)
    angle = values["rotations"][:, 0]
    cosine, sine = angle.cos(), angle.sin()
    scales = values["log_scales"].exp()
    if method in {"gaussian", "imagegs"}:
        u = (dx * cosine - dy * sine) / scales[:, 0]
        v = (dx * sine + dy * cosine) / scales[:, 1]
        weights = torch.exp(-0.5 * (u.square() + v.square()))
    elif method == "hexel":
        scales = scales + 1e-8
        u = (dx * cosine + dy * sine) / scales[:, 0]
        v = (-dx * sine + dy * cosine) / scales[:, 1]
        distance = torch.stack((v.abs(), (math.sqrt(3) / 2 * u + v / 2).abs(),
                                (math.sqrt(3) / 2 * u - v / 2).abs()), dim=-1).amax(-1) - 1
        slope = F.softplus(values["softness_logits"]) if "softness_logits" in values else softness
        weights = torch.sigmoid(-slope * distance)
    else:
        raise ValueError(f"Unknown primitive: {method}")
    if "opacity_logits" in values:
        weights = weights * values["opacity_logits"].sigmoid()
    selected, indices = weights.topk(min(topk, weights.shape[1]), dim=-1)
    colors = values["features"][indices]
    if "color_gradients" in values:
        selected_delta = delta.gather(1, indices[..., None].expand(-1, -1, 2))
        colors = colors + 0.1 * (
            selected_delta[..., None] * values["color_gradients"][indices]).sum(-2)
    normalized = selected / (selected.sum(-1, keepdim=True) + 1e-4)
    return (normalized[..., None] * colors).sum(1)


def render_image(values: dict[str, torch.Tensor], method: str, height: int,
                 width: int, *, pixel_chunk: int = 4096, topk: int = 10,
                 softness: float = 10.0) -> torch.Tensor:
    """Render a CHW image; use backward_mse for memory-bounded training."""
    pixels = pixel_coordinates(height, width, values["positions"])
    rows = [render_pixels(values, method, height, width, chunk, topk=topk,
                          softness=softness) for chunk in pixels.split(pixel_chunk)]
    return torch.cat(rows).reshape(height, width, 3).permute(2, 0, 1).contiguous()


def backward_mse(values: dict[str, torch.Tensor], method: str,
                 target: torch.Tensor, *, pixel_chunk: int = 4096,
                 topk: int = 10, softness: float = 10.0,
                 pixels: torch.Tensor | None = None) -> None:
    """Accumulate exact full-image MSE gradients without retaining chunk graphs."""
    height, width = target.shape[1:]
    if pixels is None:
        pixels = pixel_coordinates(height, width, values["positions"])
    truth = target.permute(1, 2, 0).reshape(-1, 3)
    for start in range(0, height * width, pixel_chunk):
        end = start + pixel_chunk
        prediction = render_pixels(values, method, height, width, pixels[start:end],
                                   topk=topk, softness=softness)
        ((prediction - truth[start:end]).square().sum() / target.numel()).backward()


@torch.no_grad()
def basic_metrics(image: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Report clamped PSNR/SSIM and raw MSE for the fitted objective.

    SSIM uses pytorch_msssim's valid 11x11 Gaussian window, rather than the
    custom fused-SSIM extension used by the historical compression benchmark.
    """
    from pytorch_msssim import ssim

    prediction = image.clamp(0, 1)
    mse = (prediction.double() - target.double()).square().mean()
    raw_mse = (image.double() - target.double()).square().mean()
    return {"psnr_db": float(-10 * mse.log10()),
            "ssim": float(ssim(prediction[None], target[None], data_range=1,
                               size_average=True)),
            "raw_mse": float(raw_mse), "raw_psnr_db": float(-10 * raw_mse.log10())}


@torch.no_grad()
def gradient_rmse(image: torch.Tensor, target: torch.Tensor) -> float:
    """Measure RGB forward-difference error across horizontal/vertical neighbors."""
    error = image.clamp(0, 1) - target.clamp(0, 1)
    differences = torch.cat((error.diff(dim=1).flatten(), error.diff(dim=2).flatten()))
    return float(differences.square().mean().sqrt())


def train(job: dict, protocol: dict, output: Path) -> dict:
    """Fit an immutable fixed-population MSE control on its Slurm GPU."""
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Shape fitting requires a Slurm GPU allocation")
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(protocol.get("image_root", ROOT / "data/imagegs/images")) / job["image"]
    resolution = int(job.get("resolution", protocol.get("resolution", 512)))
    loaded = Image.open(source).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    target_cpu = torch.from_numpy(np.array(loaded)).permute(2, 0, 1).float() / 255
    target = target_cpu.to("cuda")
    method = job["method"]
    linear = job.get("appearance", "linear" if "linear" in job["arm"] else "flat") == "linear"
    initial = initial_fields(target_cpu, job["count"], protocol["seed"],
                             protocol.get("init_scale", 5.0))
    if not linear:
        del initial["color_gradients"]
    if job.get("learn_opacity", False):
        initial["opacity_logits"] = torch.full((job["count"],), protocol["opacity_initial_logit"])
    if job.get("learn_softness", False):
        initial["softness_logits"] = torch.full((job["count"],), math.log(math.expm1(protocol["softness"])))
    initial_hashes = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                      for name, value in initial.items()}
    values = {name: torch.nn.Parameter(value.to("cuda")) for name, value in initial.items()}
    learning_rates = protocol.get("optimizer", {"positions": 5e-4, "log_scales": 0.01,
                                                "rotations": 2e-3, "features": 5e-3,
                                                "color_gradients": 1e-3})
    optimizer = torch.optim.Adam([{"params": value, "lr": learning_rates[name]}
                                  for name, value in values.items()])
    max_steps = int(job.get("max_steps", protocol["max_steps"]))
    chunk = int(protocol.get("pixel_chunk", 4096))
    topk, softness = int(protocol.get("topk", 10)), float(protocol.get("softness", 10.0))
    height, width = target.shape[1:]
    pixels = pixel_coordinates(height, width, values["positions"])
    def render() -> torch.Tensor:
        return render_image(values, method, height, width, pixel_chunk=chunk,
                            topk=topk, softness=softness)

    (output / "settings.json").write_text(json.dumps({"job": job, "protocol": protocol}, indent=2) + "\n")
    history, checkpoints = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - started
    training_seconds = 0.0
    with torch.no_grad():
        image = render()
        initial_quality = basic_metrics(image, target)
    torch.cuda.synchronize()
    interval = perf_counter()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        backward_mse(values, method, target, pixel_chunk=chunk, topk=topk,
                     softness=softness, pixels=pixels)
        optimizer.step()
        if step % protocol.get("eval_steps", 100) != 0 and step != max_steps:
            continue
        torch.cuda.synchronize()
        training_seconds += perf_counter() - interval
        with torch.no_grad():
            image = render()
            row = {"step": step, "training_seconds": training_seconds,
                   "primitives": job["count"], **basic_metrics(image, target)}
        history.append(row)
        (output / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
        logging.info("%s step=%d PSNR=%.4f training=%.2fs", job["id"], step,
                     row["psnr_db"], training_seconds)
        if step in protocol.get("checkpoints", [max_steps]) or step == max_steps:
            path = output / f"render_parameters_{step:06d}.pt"
            layout = "learned_weights" if job.get("learn_opacity", False) else ("linear14" if linear else "flat8")
            torch.save({"step": step, "method": method, "layout": layout,
                        "height": height, "width": width,
                        "parameters": {name: value.detach().cpu() for name, value in values.items()}}, path)
            checkpoints.append({"step": step, "primitives": job["count"], "file": path.name,
                                "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **row})
        torch.cuda.synchronize()
        interval = perf_counter()
    from lpips import LPIPS

    with torch.no_grad():
        perceptual = LPIPS(net="alex").to(target.device).eval()
        final = dict(history[-1], lpips_alex=float(perceptual(
            image.clamp(0, 1)[None] * 2 - 1, target[None] * 2 - 1)))
        final["gradient_rmse"] = gradient_rmse(image, target)
        for _ in range(2):
            render()
        durations = []
        for _ in range(10):
            torch.cuda.synchronize()
            begin = perf_counter()
            render()
            torch.cuda.synchronize()
            durations.append(perf_counter() - begin)
    final.update(decode_median_ms=median(durations) * 1000, median_fps=1 / median(durations))
    preview = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(preview).save(output / "render.png")
    scalars = sum(value.numel() for value in values.values())
    artifact = checkpoints[-1]
    final.update(file=artifact["file"], sha256=artifact["sha256"], bytes=artifact["bytes"],
                 bpp=artifact["bytes"] * 8 / (height * width),
                 fp32_parameter_bytes=scalars * 4,
                 payload_bpp=scalars * 32 / (height * width),
                 storage_format="serialized float32 checkpoint")
    result = {"complete": True, "job": job, "smoke": protocol.get("smoke", False),
              "final_step": max_steps, "training_seconds": training_seconds,
              "setup_seconds": setup_seconds,
              "total_wall_seconds": perf_counter() - started, "gpu": torch.cuda.get_device_name(),
              "torch": torch.__version__, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "parameter_count": scalars, "fp32_parameter_bytes": scalars * 4,
              "initial": initial_quality,
              "initial_parameters_sha256": initial_hashes,
              "initialization": "uniform distinct pixel centers; target RGB; zero linear gradients",
              "ssim_definition": "pytorch_msssim, valid 11x11 Gaussian window, data_range=1",
              "gradient_rmse_definition": "RGB forward differences over all horizontal/vertical neighbors",
              "rendering": "global top10, eps=1e-4; no tile support cutoff; optional learned opacity/softness",
              "history": history, "checkpoints": checkpoints, "final": final}
    if job.get("learn_opacity", False):
        with torch.no_grad():
            learned = {"opacity": values["opacity_logits"].sigmoid()}
            if job.get("learn_softness", False):
                learned["softness"] = F.softplus(values["softness_logits"])
            result["learned_weights"] = {name: {"min": float(value.min()), "median": float(value.median()),
                                                "max": float(value.max())} for name, value in learned.items()}
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
