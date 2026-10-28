"""Decode a stored flat/rich Hexel or Image-GS image representation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_representation(path: Path) -> tuple[dict, dict, bool]:
    """Keep raw FP32 parameters and stored quantized fields as distinct inputs."""
    from scripts import image_compression_codec as native
    from scripts import image_compression_compact_codec as compact
    from scripts import image_compression_precision as precision

    if path.suffix == ".pt":
        header = torch.load(path, map_location="cpu", weights_only=True)
        return header, header["parameters"], "scales" in header["parameters"]
    with path.open("rb") as stream:
        magic = stream.read(8)
    if magic == precision.MAGIC:
        header, arrays = precision.decode(path)
        header.update(method="hexel", layout="dense15" if header["rich"] else "flat8")
        return header, {name: torch.from_numpy(value) for name, value in arrays.items()}, False
    legacy = magic == native.MAGIC
    header, arrays = (native if legacy else compact).decode(path)
    return header, {name: torch.from_numpy(value) for name, value in arrays.items()}, legacy


def render_representation(path: Path) -> torch.Tensor:
    """Render either file format through its matching historical parameterization."""
    from scripts.image_compression_training import render_parameters_image
    from scripts.image_compression_stable import render_image

    header, arrays, legacy = load_representation(path)
    values = {name: value.cuda() for name, value in arrays.items()}
    return (render_parameters_image(values, header["method"], header["height"], header["width"])
            if legacy else render_image(values, header["method"], header["layout"], header["height"], header["width"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    with torch.no_grad():
        image = render_representation(args.model)
    pixels = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(pixels).save(args.output)


if __name__ == "__main__":
    main()
