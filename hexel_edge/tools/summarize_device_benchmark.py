"""Validate and plot locally exported iOS quality/thermal/battery reports.

Quality PNG metrics are display-space checks. Prefer the app's raw-float PSNR
for scientific comparisons. Battery estimates include the whole device/screen.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

LOGGER = logging.getLogger(__name__)
THERMAL_STATES = ("nominal", "fair", "serious", "critical")


def optional_number(value: Any) -> float | None:
    """Read a finite scalar, preserving missing sensor values as unavailable."""
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Telemetry must not contain nonfinite numbers")
    return result


def validate_samples(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Validate CSV samples with nondecreasing elapsed time and optimizer step."""
    parsed: list[dict[str, Any]] = []
    previous_time, previous_step = -1.0, -1
    for row in rows:
        elapsed = optional_number(row.get("elapsed_seconds"))
        step = optional_number(row.get("step"))
        if elapsed is None or elapsed < previous_time or elapsed < 0:
            raise ValueError("Telemetry elapsed time must be present and nondecreasing")
        if step is None or step < previous_step or step < 0 or not step.is_integer():
            raise ValueError("Telemetry steps must be nonnegative, integral and nondecreasing")
        state = row.get("thermal_state", "").lower()
        if state not in THERMAL_STATES:
            raise ValueError(f"Unknown thermal state: {state}")
        battery = optional_number(row.get("battery_fraction"))
        if battery is not None and not 0 <= battery <= 1:
            raise ValueError("Battery fraction must be unavailable or within [0,1]")
        sample: dict[str, Any] = dict(row)
        sample.update(elapsed_seconds=elapsed, step=int(step), thermal_state=state,
                      battery_fraction=battery)
        for key in ("footprint_bytes", "gpu_bytes", "available_memory_bytes", "step_seconds"):
            value = optional_number(row.get(key))
            if value is not None and value < 0:
                raise ValueError(f"Negative telemetry value: {key}")
            sample[key] = value
        parsed.append(sample)
        previous_time, previous_step = elapsed, int(step)
    return parsed


def summarize_samples(samples: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    """Summarize sampled thermal dwell and conservatively validate discharge."""
    thermal_seconds = {state: 0.0 for state in THERMAL_STATES}
    for left, right in zip(samples, samples[1:]):
        thermal_seconds[left["thermal_state"]] += right["elapsed_seconds"] - left["elapsed_seconds"]
    reasons = list(report.get("battery_invalid_reasons", []))
    if not report.get("battery_measurement_valid", False):
        reasons.append("App did not certify an unplugged battery interval")
    if len(samples) < 2:
        reasons.append("At least two telemetry samples are required")
    if any(sample.get("battery_state", "").lower() != "unplugged" for sample in samples):
        reasons.append("Measured telemetry contains charging or unknown battery state")
    if any(sample.get("battery_fraction") is None for sample in samples):
        reasons.append("A measured battery sample is unavailable")
    if any(left.get("battery_fraction") is not None and right.get("battery_fraction") is not None
           and right["battery_fraction"] > left["battery_fraction"] + 1e-6
           for left, right in zip(samples, samples[1:])):
        reasons.append("Charge increased during the measured interval")
    first = samples[0] if samples else {}
    last = samples[-1] if samples else {}
    start, end = first.get("battery_fraction"), last.get("battery_fraction")
    seconds = last.get("elapsed_seconds", 0) - first.get("elapsed_seconds", 0)
    if start is None or end is None:
        reasons.append("Battery endpoints are unavailable")
    elif end > start + 1e-6:
        reasons.append("Charge increased during the measured interval")
    if seconds <= 0:
        reasons.append("Measured interval has no positive duration")
    valid = not reasons

    def peak(key: str) -> float | None:
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        return max(values) if values else None

    drop = 100 * (start - end) if valid else None
    target = report.get("target_step")
    budget_observation = None
    if isinstance(target, int):
        for index, current in enumerate(samples):
            if current["step"] >= target:
                previous = samples[max(0, index - 1)]
                budget_observation = {
                    "target_step": target,
                    "observed_step": current["step"],
                    "elapsed_seconds_range": [previous["elapsed_seconds"], current["elapsed_seconds"]],
                    "battery_drop_percentage_points_at_observation":
                    100 * (start - current["battery_fraction"]) if valid else None,
                    "scope": "First telemetry sample at/after requested updates; the stress test may continue for its duration target",
                }
                break
    return {
        "sample_count": len(samples),
        "sampled_duration_seconds": seconds,
        "sampled_thermal_seconds": thermal_seconds,
        "peak_sampled_process_footprint_bytes": peak("footprint_bytes"),
        "peak_sampled_metal_allocated_bytes": peak("gpu_bytes"),
        "battery_measurement_valid": valid,
        "battery_invalid_reasons": sorted(set(reasons)),
        "battery_drop_percentage_points": drop,
        "estimated_percentage_points_per_hour": drop * 3600 / seconds
        if valid and seconds >= 600 and drop is not None and drop >= 1 else None,
        "rate_estimate_note": "Hourly extrapolation requires at least ten minutes and one observed percentage point; zero observed change is not zero energy use",
        "battery_scope": "Coarse whole-device/screen discharge, not isolated training energy or watts",
        "thermal_scope": "Sampled categorical system pressure, not chip temperature in Celsius",
        "requested_budget_observation": budget_observation,
    }


def display_image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    """Compute RGB PSNR and Gaussian-window SSIM for matching HWC uint8 images."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 3:
        raise ValueError("Images must have matching HWC RGB shapes")
    if min(prediction.shape[:2]) < 11 or prediction.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError("Metrics require uint8 RGB images at least 11 pixels on each axis")
    x, y = prediction.astype(np.float64) / 255, target.astype(np.float64) / 255
    ux = gaussian_filter(x, sigma=(1.5, 1.5, 0), truncate=3.5)
    uy = gaussian_filter(y, sigma=(1.5, 1.5, 0), truncate=3.5)
    vx = gaussian_filter(x*x, sigma=(1.5, 1.5, 0), truncate=3.5) - ux*ux
    vy = gaussian_filter(y*y, sigma=(1.5, 1.5, 0), truncate=3.5) - uy*uy
    covariance = gaussian_filter(x*y, sigma=(1.5, 1.5, 0), truncate=3.5) - ux*uy
    ssim = ((2*ux*uy + .01**2) * (2*covariance + .03**2)
            / ((ux*ux + uy*uy + .01**2) * (vx + vy + .03**2)))
    mse = float(np.mean((x-y)**2))
    return {"display_rgb_mse": mse,
            "display_psnr_db": -10 * math.log10(mse) if mse else None,
            "display_ssim": float(np.mean(ssim[5:-5, 5:-5]))}


def evaluate_exported_images(directory: Path) -> list[dict[str, Any]]:
    """Evaluate exported pairs without cropping/replacing any saved render."""
    evaluations = []
    for folder in sorted(directory.glob("eval_*")):
        if not folder.is_dir() or not folder.name[5:].isdigit():
            continue
        frames = []
        for render in sorted(folder.glob("render_*.png")):
            target = folder / render.name.replace("render_", "target_", 1)
            if not target.is_file():
                raise ValueError(f"Missing target for {render.name}")
            with Image.open(render) as image:
                prediction = np.asarray(image.convert("RGB"))
            with Image.open(target) as image:
                truth = np.asarray(image.convert("RGB"))
            frames.append({"render": str(render.relative_to(directory)),
                           "target": str(target.relative_to(directory)),
                           **display_image_metrics(prediction, truth)})
        if frames:
            evaluations.append({"step": int(folder.name[5:]), "frames": frames,
                                "mean_display_ssim": float(np.mean([f["display_ssim"] for f in frames]))})
    return evaluations


def plot_report(samples: list[dict[str, Any]], report: dict[str, Any], output: Path) -> None:
    """Save independently labeled quality, thermal, memory and battery traces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), constrained_layout=True)
    times = np.array([sample["elapsed_seconds"] for sample in samples]) / 60
    evaluations = report.get("evaluations", [])
    axes[0, 0].plot([e["elapsed_seconds"]/60 for e in evaluations],
                    [e["psnr_db"] for e in evaluations], "o-", color="#2463a6")
    axes[0, 0].set(title="Held-out quality · raw RGB", ylabel="Pooled PSNR (dB)")
    axes[0, 1].step(times, [THERMAL_STATES.index(s["thermal_state"]) for s in samples],
                    where="post", color="#ae641d")
    axes[0, 1].set(title="System thermal pressure", yticks=range(4), yticklabels=THERMAL_STATES)
    for key, label, color in (("footprint_bytes", "Process footprint", "#2463a6"),
                               ("gpu_bytes", "Metal allocations (overlap)", "#627f3b")):
        axes[1, 0].plot(times, [np.nan if s.get(key) is None else s[key]/2**20 for s in samples],
                        label=label, color=color)
    axes[1, 0].set(title="Sampled memory", ylabel="MiB")
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].plot(times, [np.nan if s["battery_fraction"] is None else 100*s["battery_fraction"]
                           for s in samples], color="#627f3b")
    axes[1, 1].set(title="Whole-device battery charge", ylabel="Battery (%)", ylim=(0, 100))
    for ax in axes.flat:
        ax.set_xlabel("Measured elapsed minutes")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Hexels on-device quality and resource test", fontsize=17, fontweight="bold")
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Locally copied Reports/<run-id> directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = json.loads((args.directory / "report.json").read_text())
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported device report schema")
    with (args.directory / "telemetry.csv").open(newline="") as handle:
        samples = validate_samples(list(csv.DictReader(handle)))
    summary = {"schema_version": 1, "run_status": report.get("status"),
               **summarize_samples(samples, report),
               "exported_display_metrics": evaluate_exported_images(args.directory),
               "metrics_scope": "App report contains raw-float quality; PNG SSIM is encoded-sRGB display-space"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_report(samples, report, args.output / "quality_resources.png")
    LOGGER.info("Analyzed %d samples; battery interval valid: %s", len(samples), summary["battery_measurement_valid"])


if __name__ == "__main__":
    main()
