"""Train one scene with the paper's fixed or mixed-growth Hexel recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scripts.prepare_nvs_smoke import training_command
from scripts.run_reproduction import set_option

ROOT = Path(__file__).resolve().parents[1]
# The paper's all-view surface objective: 2DGS-style normal consistency on median depth and ray distortion.
SURFACE_LOSSES = {
    "--surface_normal_objective": "2dgs",
    "--surface_normal_depth_mode": "median",
    "--surface_normal_depth_weight": 0.05,
    "--surface_normal_depth_start": 7000,
    "--ray_distortion_weight": 1000,
    "--ray_distortion_start": 3000,
    "--ray_distortion_ramp_steps": 1000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", choices=("dtu", "mipnerf360", "tanks_and_temples"), required=True)
    parser.add_argument("--variant", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--growth", action="store_true")
    parser.add_argument("--all-views", action="store_true", help="Train geometry using every registered view")
    parser.add_argument("--init-geometry", type=Path, help="Prepared MVS initialization pack")
    parser.add_argument("--surface", action="store_true",
                        help="Add the paper's surface losses (normal consistency on median depth and ray distortion)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--images")
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--prepare", action="store_true", help="Print the resolved command without training")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Additional Hexels3DOptions after --")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    config = json.loads((ROOT / "configs/nvs.json").read_text())
    config["dataset_roots"][args.dataset] = str(source.parent)
    job = {
        "dataset": args.dataset, "scene": source.name, "method": "hexels",
        "arm": f"hexels_{args.variant}_{'mixed_weak500' if args.growth else 'fixed'}",
        "model_path": str(output), "iteration": args.iterations, "seed": args.seed,
    }
    command, cwd = training_command(job, config, False)
    if args.all_views:
        command.remove("--eval")
        if args.dataset == "tanks_and_temples":
            # Surface scenes use the GOF layout, which stores the full-size images under images/.
            set_option(command, "--images", "images")
            set_option(command, "--resolution", 2)
    if args.surface:
        for option, value in SURFACE_LOSSES.items():
            set_option(command, option, value)
    if args.init_geometry is not None:
        set_option(command, "--init_geometry_path", args.init_geometry.resolve())
    if args.images is not None:
        set_option(command, "--images", args.images)
    if args.resolution is not None:
        set_option(command, "--resolution", args.resolution)
    command.extend(args.extra[1:] if args.extra[:1] == ["--"] else args.extra)
    if args.prepare:
        print(json.dumps({"command": command, "cwd": str(cwd)}, indent=2))
    else:
        subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
