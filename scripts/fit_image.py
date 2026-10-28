"""Fit one image with the paper Image-GS or flat/rich Hexel recipe."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def recipe(image: Path, variant: str, allocation: str, count: int,
           initial_count: int, steps: int, loss: str, seed: int) -> tuple[dict, dict]:
    """Build an explicit image experiment without historical output dependencies."""
    if count < 1 or steps < 1:
        raise ValueError("Primitive count and training steps must be positive")
    if variant == "imagegs" and allocation == "prune":
        raise ValueError("Pruning scores Hexel footprints and the paper has no Image-GS pruning arm; "
                         "use --allocation grow or fixed with --variant imagegs")
    if allocation == "grow" and steps < 2000:
        raise ValueError("Growth needs four 500-step additions; use at least 2000 steps or --allocation fixed")
    if allocation == "prune" and (initial_count <= count or steps < 3000):
        raise ValueError("Pruning requires an initial population above the target and at least 3000 steps")
    method = "imagegs" if variant == "imagegs" else "hexel"
    arm = f"{method}_{variant}_{allocation}_{loss}"
    job = {
        "id": f"{image.stem}/{arm}", "image": str(image.resolve()),
        "method": method, "arm": arm, "count": count,
        "appearance": variant, "allocation": allocation, "loss": loss,
        "initial_count": initial_count if allocation == "prune" else count,
        "max_steps": steps, "prune_schedule": {},
    }
    if allocation == "prune":
        # Fifty small deletions reach the target at step 3000. Row selection
        # keeps each survivor's Adam history instead of restarting its optimizer.
        job["prune_schedule"] = {
            str(step): initial_count - (initial_count - count) * (index + 1) // 50
            for index, step in enumerate(range(550, 3001, 50))
        }
    protocol = {
        "seed": seed, "smoke": False, "max_steps": steps, "eval_steps": 100,
        "add_steps": 500, "anneal_step": 5001,
        "checkpoints": sorted({step for step in ((1000, 2000, 5000, steps) if variant == "imagegs"
                                               else (2000, 3000, 4000, 5000, steps)) if step <= steps}),
    }
    return job, protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("imagegs", "flat", "rich"), default="rich")
    parser.add_argument("--allocation", choices=("grow", "fixed", "prune"))
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--initial-count", type=int, default=20000)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--loss", choices=("l1", "l1_ssim", "mse"), default="l1_ssim")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    allocation = args.allocation or ("grow" if args.variant == "imagegs" else "prune")
    steps = args.steps if args.steps is not None else (10000 if args.variant == "imagegs" else 5100)
    job, protocol = recipe(args.image, args.variant, allocation, args.count,
                           args.initial_count, steps, args.loss, args.seed)
    if args.variant == "imagegs" and allocation == "grow":
        if args.loss != "l1_ssim":
            raise ValueError("Native Image-GS uses L1+SSIM; use the fixed-population Gaussian control for loss ablations")
        from scripts.image_compression_training import train
    else:
        from scripts.image_compression_stable import train

    logging.basicConfig(level=logging.INFO)
    output = args.output.resolve()
    result = train(job, protocol, output)
    if args.variant != "imagegs":
        from scripts.image_compression_precision import export_paper_profile

        # The paper's packed file: balanced profile for flat, mixed14 for rich.
        result["mixed_precision"] = export_paper_profile(output / f"render_parameters_{steps:06d}.pt", output)
        logging.info("Wrote %s (%d bytes)", result["mixed_precision"]["file"], result["mixed_precision"]["bytes"])
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
