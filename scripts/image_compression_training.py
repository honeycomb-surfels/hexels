"""GPU-only Image-GS/Hexel fitting and standalone representation evaluation."""

from __future__ import annotations

import argparse
from contextlib import chdir
import hashlib
import json
import logging
import resource
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Bind the installed hexel-kernel `gsplat` before exposing the upstream checkout.
# `image-gs/gsplat` has no `__init__.py`, so a later import would resolve it as
# an empty namespace package ahead of an editable kernel install.
from gsplat import project_gaussians_2d_scale_rot, rasterize_gaussians_sum

from scripts.image_compression_codec import decode, encode, stored_bytes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "image-gs"))

# Upstream loads its font during import using a relative path. Restore the
# caller's directory before resolving any user-supplied input or output paths.
with chdir(ROOT / "image-gs"):
    from model import GaussianSplatting2D
    from utils.quantization_utils import ste_quantize
from fused_ssim import fused_ssim
from gsplat_hexels import rasterize_hexels_sum


class HexelImageGS(GaussianSplatting2D):
    """Reuse native initialization, loss and progressive allocation for Hexels."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.color_gradients = torch.nn.Parameter(torch.zeros(
            self.num_gaussians, 2, 3, device=self.device))
        self.softness = torch.nn.Parameter(torch.full(
            (self.num_gaussians,), 10.0, device=self.device))
        self._add_appearance_groups()

    def _add_appearance_groups(self) -> None:
        self.optimizer.add_param_group({"params": self.color_gradients, "lr": 1e-3})
        self.optimizer.add_param_group({"params": self.softness, "lr": 1e-3})

    def _log_compression_rate(self) -> None:
        self.num_bytes = stored_bytes("hexel", self.total_num_gaussians)
        self.worklog.info(f"Hexel uint16 file budget including header: {self.num_bytes} bytes")

    def forward(self, img_h: int, img_w: int, tile_bounds: tuple,
                upsample_ratio: float | None = None, benchmark: bool = False):
        if upsample_ratio not in (None, 1.0) or benchmark:
            raise ValueError("Use the campaign's separate decoder timing path")
        parameters = render_parameters(self, "hexel")
        if self.quantize:
            parameters = {key: ste_quantize(value, 16) for key, value in parameters.items()}
        return render_parameters_image(parameters, "hexel", img_h, img_w), 0.0

    def _add_gaussians(self, add_num: int, plot_gaussians: bool = False) -> None:
        old_count = self.num_gaussians
        # Upstream resets Adam when adding primitives; apply that policy to both methods.
        super()._add_gaussians(add_num, plot_gaussians)
        added = self.num_gaussians - old_count
        if added:
            self.color_gradients = torch.nn.Parameter(torch.cat((
                self.color_gradients.detach(),
                torch.zeros(added, 2, 3, device=self.device))))
            self.softness = torch.nn.Parameter(torch.cat((
                self.softness.detach(), torch.full((added,), 10.0, device=self.device))))
            self._add_appearance_groups()


def render_parameters(model: GaussianSplatting2D, method: str) -> dict[str, torch.Tensor]:
    """Extract only the parameters the decoder needs, in render coordinates."""
    values = {
        "positions": model.xy, "scales": model._get_scale(),
        "rotations": model.rot, "features": model.feat,
    }
    if method == "hexel":
        values.update(color_gradients=model.color_gradients,
                      softness=F.softplus(model.softness))
    return values


def render_parameters_image(values: dict[str, torch.Tensor], method: str,
                            height: int, width: int) -> torch.Tensor:
    """Decode an image directly from stored fields, without a training model."""
    if method == "hexel":
        wh = values["positions"].new_tensor([width, height])
        image = rasterize_hexels_sum(
            xys=values["positions"] * wh, scales=values["scales"],
            rotations=values["rotations"], colors=values["features"],
            color_grads=values["color_gradients"], softness=values["softness"],
            img_height=height, img_width=width, color_grad_scale=0.1)
    else:
        tiles = ((width + 15) // 16, (height + 15) // 16, 1)
        xy, radii, conics, hits = project_gaussians_2d_scale_rot(
            values["positions"], values["scales"], values["rotations"],
            height, width, tiles)
        image = rasterize_gaussians_sum(xy, radii, conics, hits,
                                       values["features"], height, width, 16, 16, True)
    return image.permute(2, 0, 1).contiguous()


def basic_metrics(image: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute direct-float RGB metrics after one common [0,1] clamp."""
    image = image.clamp(0, 1)
    mse = (image.double() - target.double()).square().mean()
    return {"psnr_db": float(-10 * mse.log10()),
            "ssim": float(fused_ssim(image[None], target[None]))}


@torch.no_grad()
def export_evaluate(model: GaussianSplatting2D, method: str, output: Path,
                    step: int, *, final: bool) -> list[dict]:
    """Measure decoded files, including quantization error and container overhead."""
    tensors = render_parameters(model, method)
    arrays = {name: value.detach().cpu().numpy() for name, value in tensors.items()}
    target = model.gt_images
    raw_image = render_parameters_image(tensors, method, model.img_h, model.img_w)
    qat_values = {key: ste_quantize(value, 16) for key, value in tensors.items()}
    qat_image = render_parameters_image(qat_values, method, model.img_h, model.img_w)
    quantized = {}
    for name, value in tensors.items():
        minimum, maximum = float(value.min()), float(value.max())
        scale = max((maximum - minimum) / 65535, 1e-8)
        # Preserve CUDA's rounding decisions at quantization-bin boundaries.
        codes = torch.round((value - minimum) / scale).clamp(0, 65535)
        quantized[name] = (codes.cpu().numpy().astype(np.uint16), minimum, scale)
    records = []
    for bits in (32, 16):
        path = output / f"step_{step:06d}_{bits}bit.hxi"
        encode(path, arrays, method=method, width=model.img_w, height=model.img_h,
               bits=bits, quantized=quantized if bits == 16 else None)
        header, decoded = decode(path)
        values = {name: torch.from_numpy(value).to(model.device) for name, value in decoded.items()}
        image = render_parameters_image(values, method, header["height"], header["width"])
        expected = raw_image if bits == 32 else qat_image
        max_error = float((image - expected).abs().max())
        torch.testing.assert_close(image, expected, rtol=2e-4, atol=2e-5)
        size = path.stat().st_size
        record = {
            "step": step, "bits": bits, "file": path.name, "bytes": size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "primitives": model.num_gaussians, "roundtrip_max_abs_error": max_error,
            "bpp": size * 8 / (model.img_h * model.img_w),
            "raw_rgb8_compression_ratio": model.img_h * model.img_w * 3 / size,
            **basic_metrics(image, target),
        }
        if final and bits == 16:
            from lpips import LPIPS
            from pytorch_msssim import ms_ssim
            from utils.flip import LDRFLIPLoss

            prediction = image.clamp(0, 1)[None]
            truth = target[None]
            lpips_metric = LPIPS(net="alex").to(model.device).eval()
            record["lpips_alex"] = float(lpips_metric(prediction * 2 - 1, truth * 2 - 1))
            record["ms_ssim"] = float(ms_ssim(prediction, truth, data_range=1, size_average=True))
            record["flip"] = float(LDRFLIPLoss().to(model.device)(prediction, truth))
            pixels = (prediction[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(pixels).save(output / "decoded.png")
            for _ in range(10):
                render_parameters_image(values, method, model.img_h, model.img_w)
            trials = []
            for _ in range(5):
                torch.cuda.synchronize()
                start = perf_counter()
                for _ in range(20):
                    render_parameters_image(values, method, model.img_h, model.img_w)
                torch.cuda.synchronize()
                trials.append((perf_counter() - start) / 20)
            record["render_seconds_trials"] = trials
            record["median_fps"] = 1 / float(np.median(trials))
        records.append(record)
    return records


def train(job: dict, protocol: dict, output: Path) -> dict:
    """Fit with native Image-GS growth and shared, explicitly recorded schedules."""
    if not torch.cuda.is_available():
        raise RuntimeError("Image compression training requires a Slurm GPU allocation")
    output.mkdir(parents=True, exist_ok=False)
    args_dict = yaml.safe_load((ROOT / "image-gs/cfgs/default.yaml").read_text())
    args_dict.update(
        seed=protocol["seed"], device="cuda:0", input_path=job["image"],
        data_root=str(ROOT / "data/imagegs/images"), log_dir=str(output / "native"),
        num_gaussians=job["count"], max_steps=protocol["max_steps"],
        quantize=True, disable_lr_schedule=True, save_image_format="png",
        eval_steps=protocol["eval_steps"], add_steps=protocol["add_steps"],
        post_min_steps=0,
    )
    args = argparse.Namespace(**args_dict)
    (output / "settings.json").write_text(json.dumps(args_dict, indent=2) + "\n")
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    model_class = GaussianSplatting2D if job["method"] == "imagegs" else HexelImageGS
    model = model_class(args)
    torch.cuda.synchronize()
    setup_seconds = perf_counter() - started
    rows, checkpoints = [], []
    training_seconds = 0.0
    interval_start = perf_counter()
    for step in range(1, protocol["max_steps"] + 1):
        model.step = step
        if step == protocol["anneal_step"]:
            for group in model.optimizer.param_groups:
                group["lr"] *= 0.1
        model.optimizer.zero_grad(set_to_none=True)
        prediction, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
        model._get_total_loss(prediction)
        model.total_loss.backward()
        model.optimizer.step()
        with torch.no_grad():
            if step % model.add_steps == 0 and model.num_gaussians < model.total_num_gaussians:
                model._add_gaussians(model.max_add_num)
            if step % protocol["eval_steps"] == 0 or step == protocol["max_steps"]:
                torch.cuda.synchronize()
                training_seconds += perf_counter() - interval_start
                image, _ = model.forward(model.img_h, model.img_w, model.tile_bounds)
                row = {"step": step, "training_seconds": training_seconds,
                       "wall_seconds": perf_counter() - started, "primitives": model.num_gaussians,
                       **basic_metrics(image, model.gt_images)}
                rows.append(row)
                logging.info("%s step=%d PSNR=%.4f training=%.2fs", job["id"], step,
                             row["psnr_db"], training_seconds)
                with (output / "history.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step in protocol["checkpoints"]:
                    checkpoints.extend(export_evaluate(
                        model, job["method"], output, step,
                        final=step == protocol["max_steps"]))
                torch.cuda.synchronize()
                interval_start = perf_counter()
    if model.num_gaussians != job["count"]:
        raise RuntimeError("Training ended before reaching the requested population")
    metrics = {
        "complete": True, "job": job, "smoke": protocol["smoke"],
        "setup_seconds": setup_seconds, "training_seconds": training_seconds,
        "total_wall_seconds": perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "checkpoints": checkpoints, "history": rows,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return metrics
