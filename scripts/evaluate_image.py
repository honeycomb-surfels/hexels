"""Evaluate raw FP32 or decoded stored image fields with the shared metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    from lpips import LPIPS
    import torch
    from scripts.decode_image import render_representation
    from scripts.image_compression_training import basic_metrics

    pixels = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32).copy() / 255
    target = torch.from_numpy(pixels).cuda().permute(2, 0, 1).contiguous()
    with torch.no_grad():
        prediction = render_representation(args.model).clamp(0, 1)
        metrics = basic_metrics(prediction, target)
        perceptual = LPIPS(net="alex").cuda().eval()
        metrics["lpips_alex"] = float(perceptual(prediction[None] * 2 - 1, target[None] * 2 - 1))
    metrics.update(model=str(args.model), image=str(args.image),
                   condition="raw_fp32" if args.model.suffix == ".pt" else "decoded_stored_fields",
                   stored_file_bytes=args.model.stat().st_size)
    with args.output.open("x") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
