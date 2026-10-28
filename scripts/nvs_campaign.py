"""Expand the approved NVS matrix without importing any GPU dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_minimum_3d_benchmark import MIPNERF360_OUTDOOR


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/nvs.json"


def expand(config: dict) -> list[dict]:
    """Enumerate native and supported capacity arms, including approved repeats."""
    arms = [f"hexels_{variant}_{regime}" for variant in config["hexels"]["variants"]
            for regime in config["hexels"]["regimes"]]
    arms += [f"{method}_native" for method in config["native_baselines"]]
    arms += [f"{method}_{regime}" for method in config["controlled_baselines"]
             for regime in config.get("controlled_regimes_by_method", {}).get(method, config["controlled_regimes"])]
    runs = []
    for seed, datasets, selected in [(config["seed"], config["datasets"], arms), *[
        (seed, config["repeats"]["datasets"], config["repeats"]["arms"])
        for seed in config["repeats"]["seeds"]
    ]]:
        for dataset, scenes in datasets.items():
            for scene in scenes:
                for arm in selected:
                    if arm not in arms:
                        raise ValueError(f"Unknown repeated arm: {arm}")
                    methods = ["hexels", *(set(config["native_baselines"]) | set(config["controlled_baselines"]))]
                    method = next(method for method in sorted(methods, key=len, reverse=True)
                                  if arm.startswith(f"{method}_"))
                    budget_anchor = None
                    if "grown_v3" in arm:
                        budget_anchor = f"{dataset}/{scene}/seed0/hexels_v3_mixed_weak500"
                    elif "_sparse" in arm:
                        budget_anchor = f"{dataset}/{scene}/seed0/hexels_v1_fixed"
                    runs.append({"id": f"{dataset}/{scene}/seed{seed}/{arm}",
                                 "dataset": dataset, "scene": scene, "seed": seed,
                                 "arm": arm, "method": method, "budget_anchor": budget_anchor})
    if len({run["id"] for run in runs}) != len(runs):
        raise ValueError("Campaign contains duplicate experiments")
    if len(runs) != config["planned_training_runs"]["total"]:
        raise ValueError("Expanded campaign differs from approved run count")
    return runs


def cameras(config: dict, dataset: str, scene: str) -> dict:
    images, resolution = "images", 2
    if dataset == "mipnerf360":
        images, resolution = ("images_4" if scene in MIPNERF360_OUTDOOR else "images_2"), 1
    elif dataset == "tanks_and_temples":
        images, resolution = "images_2", 1
    return {"source_path": str((ROOT / config["dataset_roots"][dataset] / scene).resolve()),
            "images": images, "resolution": resolution, "eval": True, "llffhold": 8,
            "data_device": "cpu"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    payload = {"protocol": config, "executable": False,
               "status": "requires_per_method_runtime_and_camera_gates", "runs": expand(config)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
