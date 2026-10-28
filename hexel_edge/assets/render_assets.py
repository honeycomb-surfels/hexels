#!/usr/bin/env python3
"""Export original SVG artwork and verify the bundled public DTU figures.

Install Python dependencies with:
    python -m pip install -r hexel_edge/assets/requirements.txt
CairoSVG also requires native Cairo (for example, ``brew install cairo`` on macOS).
Run from any directory; this script needs only the files in this assets folder.
SVG files are the editable source artwork. Benchmark PNGs are verified against
their recorded public-export SHA256 values and are never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import cairosvg
from PIL import Image

ASSETS = Path(__file__).resolve().parent
EXPORTS = {
    "hero": (1600, 720),
    "logo-wordmark": (1152, 280),
    "bee-icon": (1024, 1024),
    "app-icon": (1024, 1024),
}
PUBLIC_IMAGES = {
    "ipad-dtu-app.png": "7727082dc819c34eb92a01d1bdaf29c51488dd435d682ede4e433c49cfd38d16",
    "ipad-dtu-comparison.png": "1cb450ad93505da48b79d9d77125c74a4f0f30c008ecf8326fb78bdfa5e8f90f",
    "ipad-dtu-resources.png": "c253b71bb59aaf728cdc92214831f9050c651ea8864fa983ee6927ab2ae1a7db",
}


def sha256(path: Path) -> str:
    """Return a file's byte-level SHA256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Render exact-size artwork and save relative-path public provenance."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    artwork = []
    for name, size in EXPORTS.items():
        source, output = ASSETS / f"{name}.svg", ASSETS / f"{name}.png"
        cairosvg.svg2png(url=str(source), write_to=str(output), output_width=size[0], output_height=size[1])
        with Image.open(output) as image:
            if image.size != size:
                raise ValueError(f"Wrong dimensions for {output.name}: {image.size}")
        artwork.append({"source": source.name, "source_sha256": sha256(source), "png": output.name,
                        "png_sha256": sha256(output), "dimensions": list(size)})
        logging.info("Rendered %s at %s × %s", output.name, *size)

    figures = []
    for name, digest in PUBLIC_IMAGES.items():
        canonical = ASSETS / name
        if sha256(canonical) != digest:
            raise ValueError(f"Bundled public asset differs from its recorded SHA256: {name}")
        with Image.open(canonical) as image:
            size = list(image.size)
        figure = {"file": name, "source": name, "sha256": digest, "dimensions": size,
                  "byte_identical": name != "ipad-dtu-resources.png"}
        if name == "ipad-dtu-resources.png":
            figure.update(original_export_sha256="0efc0ea96aba922fe9b7451eb85a84e0232507a60c0b4e33529477875a52d3c9",
                          presentation_change="Memory panel title simplified; all pixels outside the title remain unchanged.")
        figures.append(figure)
        logging.info("Verified canonical public DTU asset hash: %s", name)

    provenance = {
        "schema_version": 1,
        "brand": "Hexel Edge",
        "status": "IN DEVELOPMENT",
        "artwork": "Original code-native SVG artwork created for Hexel Edge; no stock logo or AI bitmap.",
        "palette": {"amber": "#FFC45C", "charcoal": "#171D20", "cream": "#FFF4DC"},
        "exports": artwork,
        "public_dtu_images": figures,
        "image_scope": "Only public DTU scan24 benchmark imagery; no private captures or desktop screenshots.",
        "benchmark_evidence": "../results/README.md",
        "benchmark_measurements": "../results/dtu_scan24_ipad.json",
        "benchmark_quality": "../results/dtu_scan24_quality.csv",
        "benchmark_telemetry": "../results/dtu_scan24_telemetry.csv",
        "comparison_scope": "Different training recipes/view counts/budgets; development-validation views reused for selection.",
        "resource_scope": "One M2 iPad run; memory is sampled, battery is coarse whole-device charge, thermal states are categorical.",
        "render_command": "python hexel_edge/assets/render_assets.py",
    }
    (ASSETS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
