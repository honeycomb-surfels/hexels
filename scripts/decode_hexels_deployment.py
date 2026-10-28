"""Restore a lossless or appearance-PQ archive to the ordinary inference layout."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import zipfile

import torch

from scripts import hexels_linear_codec, hexels_sh_codec


def restore(archive: Path, output: Path) -> Path:
    """Decode on CPU without overwriting a model or claiming training resumability."""
    with zipfile.ZipFile(archive) as packed:
        format_name = json.loads(packed.read("manifest.json"))["format"]
    codecs = {codec.FORMAT: codec for codec in (hexels_linear_codec, hexels_sh_codec)}
    if format_name not in codecs:
        raise ValueError(f"Unsupported deployment format: {format_name}")
    checkpoint, config = codecs[format_name].decode(archive)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_bytes(config)
    (output / "checkpoints").mkdir()
    destination = output / "checkpoints" / f"ckpt_{checkpoint['iteration']:06d}.pt"
    torch.save(checkpoint, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    logging.info("Restored inference checkpoint: %s", restore(args.archive, args.output))


if __name__ == "__main__":
    main()
