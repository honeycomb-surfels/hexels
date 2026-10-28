"""Isolated GPU experiments for stable 2D compression and flat-color controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import resource
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from scripts import image_compression_training as legacy
from scripts.image_compression_compact_codec import decode, encode

ROOT = legacy.ROOT


class StableImageGS(legacy.GaussianSplatting2D):
    """Share native fitting while representing positive scales in log space."""

    def __init__(self, args: argparse.Namespace, *, method: str, rich: bool,
                 linear: bool = False, exact_gaussian_gradients: bool = False) -> None:
        self.method = method
        self.rich = rich
        self.linear = linear
        self.exact_gaussian_gradients = exact_gaussian_gradients
        self.layout = "dense15" if rich else "linear14" if linear else "flat8"
        super().__init__(args)
        if rich or linear:
            self.color_gradients = torch.nn.Parameter(torch.zeros(
                self.num_gaussians, 2, 3, device=self.device))
        if rich:
            self.softness = torch.nn.Parameter(torch.full(
                (self.num_gaussians,), 10.0, device=self.device))
        if rich or linear:
            self._add_appearance_groups()

    def _init_pos_scale_feat(self, args: argparse.Namespace) -> None:
        super()._init_pos_scale_feat(args)
        with torch.no_grad():
            self.scale.fill_(math.log(self.init_scale))

    def _log_compression_rate(self) -> None:
        count = 15 if self.rich else 14 if self.linear else 8
        self.num_bytes = 1024 + self.total_num_gaussians * count * 2
        self.worklog.info(f"{self.method} {count}-scalar uint16 budget: {self.num_bytes} bytes")

    def _add_appearance_groups(self) -> None:
        self.optimizer.add_param_group({"params": self.color_gradients, "lr": 1e-3})
        if self.rich:
            self.optimizer.add_param_group({"params": self.softness, "lr": 1e-3})

    def _get_scale(self, upsample_ratio: float | None = None) -> torch.Tensor:
        scales = self.scale.exp()
        return scales if upsample_ratio is None else scales * upsample_ratio

    def _add_gaussians(self, add_num: int, plot_gaussians: bool = False) -> None:
        old_count = self.num_gaussians
        super()._add_gaussians(add_num, plot_gaussians)
        added = self.num_gaussians - old_count
        if not added:
            return
        with torch.no_grad():
            self.scale[old_count:].fill_(math.log(self.init_scale))
        if self.rich or self.linear:
            self.color_gradients = torch.nn.Parameter(torch.cat((
                self.color_gradients.detach(),
                torch.zeros(added, 2, 3, device=self.device))))
        if self.rich:
            self.softness = torch.nn.Parameter(torch.cat((
                self.softness.detach(), torch.full((added,), 10.0, device=self.device))))
        if self.rich or self.linear:
            self._add_appearance_groups()

    def forward(self, img_h: int, img_w: int, tile_bounds: tuple,
                upsample_ratio: float | None = None, benchmark: bool = False):
        if upsample_ratio not in (None, 1.0) or benchmark:
            raise ValueError("Use the experiment's explicit decoder timing path")
        values = storage_parameters(self)
        if self.quantize:
            values = quantize_parameters(values)
        image = render_image(values, self.method, self.layout, img_h, img_w,
                             exact_gaussian_gradients=self.exact_gaussian_gradients)
        return image, 0.0


def storage_parameters(model: StableImageGS) -> dict[str, torch.Tensor]:
    """Return fields before quantization; scales remain logarithmic."""
    values = {"positions": model.xy, "log_scales": model.scale,
              "rotations": model.rot, "features": model.feat}
    if model.rich or model.linear:
        values["color_gradients"] = model.color_gradients
    if model.rich:
        values["softness"] = F.softplus(model.softness)
    return values


def quantize_parameters(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Use native uniform uint16 STE on stored fields, including log scales."""
    return {name: legacy.ste_quantize(value, 16) for name, value in values.items()}


def render_image(values: dict[str, torch.Tensor], method: str, layout: str,
                 height: int, width: int, *, exact_gaussian_gradients: bool = False) -> torch.Tensor:
    """Render decoded fields using existing kernels and their historical APIs."""
    scales = values["log_scales"].exp()
    features = values["features"]
    if layout == "linear14":
        centers = (values["positions"] - 0.5) * features.new_tensor([width, height])
        gradients = values["color_gradients"]
        intercept = features + 0.1 * (centers[..., None] * gradients).sum(1)
        features = torch.cat((intercept, gradients.flatten(1)), dim=1)
    if method == "imagegs":
        tiles = ((width + 15) // 16, (height + 15) // 16, 1)
        xy, radii, conics, hits = legacy.project_gaussians_2d_scale_rot(
            values["positions"], scales, values["rotations"], height, width, tiles)
        if exact_gaussian_gradients and torch.is_grad_enabled():
            angle = values["rotations"][:, 0]
            cosine, sine = angle.cos(), angle.sin()
            inv_x, inv_y = scales.reciprocal().square().unbind(-1)
            analytic = torch.stack((cosine.square() * inv_x + sine.square() * inv_y,
                                    cosine * sine * (inv_y - inv_x),
                                    sine.square() * inv_x + cosine.square() * inv_y), dim=-1)
            # Retain native forward values and differentiate the exact inverse covariance.
            conics = conics.detach() + (analytic - analytic.detach())
            xy = values["positions"] * scales.new_tensor([width, height])
        image = legacy.rasterize_gaussians_sum(
            xy, radii, conics, hits, features, height, width, 16, 16, True)
    else:
        image = legacy.rasterize_hexels_sum(
            xys=values["positions"] * values["positions"].new_tensor([width, height]),
            scales=scales, rotations=values["rotations"], colors=features,
            color_grads=None if layout in {"flat8", "linear14"} else values["color_gradients"],
            softness=10.0 if layout in {"flat8", "linear14"} else values["softness"],
            img_height=height, img_width=width, color_grad_scale=0.1)
    if layout == "linear14":
        # Linearity of the normalized blend gives the same RGB field via nine channels.
        x = torch.arange(width, device=image.device, dtype=image.dtype) - width / 2
        y = torch.arange(height, device=image.device, dtype=image.dtype) - height / 2
        image = image[..., :3] - 0.1 * (
            x[None, :, None] * image[..., 3:6] + y[:, None, None] * image[..., 6:9])
    return image.permute(2, 0, 1).contiguous()


def quantized_codes(values: dict[str, torch.Tensor]) -> dict[str, tuple]:
    """Preserve CUDA rounding when writing the CPU-only codec."""
    result = {}
    for name, value in values.items():
        minimum, maximum = float(value.min()), float(value.max())
        step = max((maximum - minimum) / 65535, 1e-8)
        codes = torch.round((value - minimum) / step).clamp(0, 65535)
        result[name] = (codes.detach().cpu().numpy().astype(np.uint16), minimum, step)
    return result


@torch.no_grad()
def quality_and_timing(values: dict[str, torch.Tensor], method: str, layout: str,
                       target: torch.Tensor, image: torch.Tensor) -> dict:
    """Common perceptual metrics and timing with decoded parameters resident."""
    from lpips import LPIPS
    from pytorch_msssim import ms_ssim
    from utils.flip import LDRFLIPLoss

    prediction, truth = image.clamp(0, 1)[None], target[None]
    perceptual = LPIPS(net="alex").to(target.device).eval()
    result = {
        "lpips_alex": float(perceptual(prediction * 2 - 1, truth * 2 - 1)),
        "ms_ssim": float(ms_ssim(prediction, truth, data_range=1, size_average=True)),
        "flip": float(LDRFLIPLoss().to(target.device)(prediction, truth)),
    }
    height, width = target.shape[1:]
    for _ in range(10):
        render_image(values, method, layout, height, width)
    trials = []
    for _ in range(5):
        torch.cuda.synchronize()
        started = perf_counter()
        for _ in range(20):
            render_image(values, method, layout, height, width)
        torch.cuda.synchronize()
        trials.append((perf_counter() - started) / 20)
    return dict(result, render_seconds_trials=trials, median_fps=1 / float(np.median(trials)))


@torch.no_grad()
def export_evaluate(model: StableImageGS, output: Path, step: int, *, final: bool) -> dict:
    """Require the actual stored representation to reproduce the training render."""
    values = storage_parameters(model)
    arrays = {name: value.detach().cpu().numpy() for name, value in values.items()}
    layout = model.layout
    path = output / f"step_{step:06d}_16bit.hxc"
    encode(path, arrays, method=model.method, width=model.img_w, height=model.img_h,
           layout=layout, quantized=quantized_codes(values))
    header, decoded = decode(path)
    decoded_values = {name: torch.from_numpy(value).to(model.device)
                      for name, value in decoded.items()}
    image = render_image(decoded_values, model.method, layout, model.img_h, model.img_w)
    expected = render_image(quantize_parameters(values), model.method, layout,
                            model.img_h, model.img_w)
    torch.testing.assert_close(image, expected, rtol=2e-4, atol=2e-5)
    record = {
        "step": step, "bits": 16, "file": path.name, "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "primitives": model.num_gaussians, "layout": header["layout"],
        "roundtrip_max_abs_error": float((image - expected).abs().max()),
        "bpp": path.stat().st_size * 8 / (model.img_h * model.img_w),
        **legacy.basic_metrics(image, model.gt_images),
    }
    if final:
        record.update(quality_and_timing(decoded_values, model.method, layout,
                                        model.gt_images, image))
        pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(output / "decoded.png")
    return record


@torch.no_grad()
def diagnostics(model: legacy.GaussianSplatting2D, method: str, stable: bool) -> dict:
    """Record the scale distribution and float/QAT quality outside training timing."""
    for name, value in model.named_parameters():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite parameter: {name}")
    scales = model._get_scale()
    if not torch.isfinite(scales).all():
        raise FloatingPointError("Nonfinite rendered scales")
    values = storage_parameters(model) if stable else legacy.render_parameters(model, method)
    stored_scales = values["log_scales" if stable else "scales"]
    quantized = quantize_parameters(values)
    if stable:
        layout = model.layout
        raw_image = render_image(values, method, layout, model.img_h, model.img_w)
        qat_image = render_image(quantized, method, layout, model.img_h, model.img_w)
        decoded_scales = quantized["log_scales"].exp()
    else:
        raw_image = legacy.render_parameters_image(values, method, model.img_h, model.img_w)
        qat_image = legacy.render_parameters_image(quantized, method, model.img_h, model.img_w)
        decoded_scales = quantized["scales"]
    return {
        "float_psnr_db": legacy.basic_metrics(raw_image, model.gt_images)["psnr_db"],
        "qat_psnr_db": legacy.basic_metrics(qat_image, model.gt_images)["psnr_db"],
        "scale_min_median_max": torch.quantile(scales.flatten(), scales.new_tensor([0, .5, 1])).tolist(),
        "decoded_scale_min_median_max": torch.quantile(
            decoded_scales.flatten(), scales.new_tensor([0, .5, 1])).tolist(),
        "scale_quantization_space": "log" if stable else "pixels",
        "scale_quantization_step": max(float(stored_scales.max() - stored_scales.min()) / 65535, 1e-8),
        "raw_inverse_min_abs": None if stable else float(model.scale.abs().min()),
        "raw_inverse_nearest_zero": None if stable else float(
            model.scale.flatten()[model.scale.abs().argmin()]),
        "scale_max_abs_gradient": None if model.scale.grad is None else float(model.scale.grad.abs().max()),
    }


def train(job: dict, protocol: dict, output: Path) -> dict:
    """Fit a fresh isolated control or diagnostic run under a frozen protocol."""
    if not torch.cuda.is_available():
        raise RuntimeError("Stable compression experiments require a Slurm GPU allocation")
    output = (output if output.is_absolute() else ROOT / output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    arm = job["arm"]
    stable = job.get("scale_mode", "reciprocal" if arm.startswith("original_") else "log") == "log"
    method = job.get("method", "imagegs" if arm.startswith("gaussian") else "hexel")
    linear = "linear" in arm
    rich = method == "hexel" and "flat" not in arm and not linear
    default_steps = int(protocol["max_steps"])
    if stable and method == "hexel":
        default_steps = min(default_steps, 5100)
    max_steps = int(job.get("max_steps", default_steps))
    args_dict = yaml.safe_load((ROOT / "image-gs/cfgs/default.yaml").read_text())
    args_dict.update(
        seed=protocol["seed"], device="cuda:0", input_path=job["image"],
        data_root=str(ROOT / "data/imagegs/images"), log_dir=str(output / "native"),
        num_gaussians=job["count"], max_steps=max_steps,
        quantize=job.get("quantize", arm != "original_noqat"),
        disable_lr_schedule=True, save_image_format="png",
        eval_steps=protocol["eval_steps"], add_steps=protocol["add_steps"], post_min_steps=0,
        scale_lr=0.01 if stable else 0.002,
    )
    allocation = job.get("allocation", "grow")
    if allocation != "grow":
        args_dict.update(disable_prog_optim=True, num_gaussians=job.get("initial_count", job["count"]))
    if "loss" in job:
        mse = job["loss"] == "mse"
        args_dict.update(l1_loss_ratio=0.0 if mse else 1.0, l2_loss_ratio=1.0 if mse else 0.0,
                         ssim_loss_ratio=0.1 if job["loss"] == "l1_ssim" else 0.0)
    (output / "settings.json").write_text(json.dumps(args_dict, indent=2) + "\n")
    kernel_started = perf_counter()
    if job.get("prune_schedule"):
        from scripts.image_compression_allocation import load_prune_scores

        # Build the scorer before any timing; a first JIT build would otherwise count as training.
        load_prune_scores()
    kernel_setup_seconds = perf_counter() - kernel_started
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    args = argparse.Namespace(**args_dict)
    if stable:
        model = StableImageGS(args, method=method, rich=rich, linear=linear,
                              exact_gaussian_gradients=job.get("exact_gaussian_gradients", False))
    else:
        model = legacy.HexelImageGS(args) if method == "hexel" else legacy.GaussianSplatting2D(args)
    model.total_num_gaussians = job["count"]
    if job.get("initialization") == "uniform_shared":
        from scripts.image_compression_shape_torch import initial_fields

        fields = initial_fields(model.gt_images.detach().cpu(), job["count"], protocol["seed"],
                                init_scale=model.init_scale)
        with torch.no_grad():
            for parameter, field in ((model.xy, "positions"), (model.scale, "log_scales"),
                                     (model.rot, "rotations"), (model.feat, "features")):
                parameter.copy_(fields[field].to(model.device))
    initial_hashes = None
    if job.get("record_initialization") or job.get("initialization") == "uniform_shared":
        fields = storage_parameters(model) if stable else legacy.render_parameters(model, method)
        initial_hashes = {name: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                          for name, value in fields.items()}
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - started
    rows, checkpoints = [], []
    pruning = []
    training_seconds, best_psnr = 0.0, -math.inf
    stopped_reason = None
    interval_start = perf_counter()
    for step in range(1, max_steps + 1):
        model.step = step
        if step == protocol["anneal_step"]:
            for group in model.optimizer.param_groups:
                group["lr"] *= 0.1
        model.optimizer.zero_grad(set_to_none=True)
        image, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
        model._get_total_loss(image)
        model.total_loss.backward()
        model.optimizer.step()
        with torch.no_grad():
            if (not model.disable_prog_optim and step % model.add_steps == 0
                    and model.num_gaussians < model.total_num_gaussians):
                model._add_gaussians(model.max_add_num)
            prune_count = job.get("prune_schedule", {}).get(str(step))
            if prune_count is not None:
                from scripts.image_compression_allocation import prune_model

                pruning.append(prune_model(model, prune_count))
            if step % protocol["eval_steps"] != 0 and step != max_steps:
                continue
            torch.cuda.synchronize()
            training_seconds += perf_counter() - interval_start
            image, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
            row = {"step": step, "training_seconds": training_seconds,
                   "wall_seconds": perf_counter() - started, "primitives": model.num_gaussians,
                   **legacy.basic_metrics(image, model.gt_images),
                   **diagnostics(model, method, stable)}
            rows.append(row)
            with (output / "history.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            logging.info("%s step=%d PSNR=%.4f float=%.4f training=%.2fs", job["id"], step,
                         row["psnr_db"], row["float_psnr_db"], training_seconds)
            drop = job.get("stop_on_collapse_db")
            if drop is not None and step > 2100 and row["psnr_db"] < best_psnr - drop:
                stopped_reason = f"Diagnostic PSNR dropped more than {drop} dB from sampled best at step {step}"
                torch.save({"step": step, "state_dict": model.state_dict(),
                            "optimizer": model.optimizer.state_dict(), "diagnostics": row},
                           output / "collapse_snapshot.pt")
            if model.num_gaussians == model.total_num_gaussians:
                best_psnr = max(best_psnr, row["psnr_db"])
            final = step == max_steps or stopped_reason is not None
            if step in protocol["checkpoints"] or final:
                if stable:
                    checkpoints.append(export_evaluate(model, output, step, final=final))
                else:
                    checkpoints.extend(legacy.export_evaluate(model, method, output, step, final=final))
            if step == 5100 or final:
                fields = storage_parameters(model) if stable else legacy.render_parameters(model, method)
                torch.save({"step": step, "method": method,
                            "layout": model.layout if stable else "dense15" if rich else "flat8",
                            "scale_mode": "log" if stable else "reciprocal",
                            "height": model.img_h, "width": model.img_w,
                            "parameters": {key: value.detach().cpu() for key, value in fields.items()}},
                           output / f"render_parameters_{step:06d}.pt")
            if final:
                break
            torch.cuda.synchronize()
            interval_start = perf_counter()
    if model.num_gaussians != job["count"]:
        raise RuntimeError("Training ended before reaching the requested population")
    result = {
        "complete": True, "job": job, "smoke": protocol.get("smoke", False),
        "diagnostic": arm.startswith("original_") or bool(job.get("diagnostic", False)),
        "stopped_early_reason": stopped_reason, "final_step": step,
        "setup_seconds": setup_seconds, "kernel_setup_seconds": kernel_setup_seconds,
        "training_seconds": training_seconds,
        "total_wall_seconds": perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "checkpoints": checkpoints, "history": rows,
    }
    if "allocation" in job:
        result["pruning"] = pruning
    if initial_hashes is not None:
        result["initial_parameters_sha256"] = initial_hashes
    if stable and not model.quantize:
        with torch.no_grad():
            image, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
            result["float_final"] = {
                **legacy.basic_metrics(image, model.gt_images),
                **quality_and_timing(storage_parameters(model), method, model.layout, model.gt_images, image),
            }
            pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(output / "float.png")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
