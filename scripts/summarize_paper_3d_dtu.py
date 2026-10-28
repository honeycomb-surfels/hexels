from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any

from scripts.prepare_minimum_3d_benchmark import (
    DTU,
    HEXELS_COMPACT_BYTES_PER_PRIMITIVE,
    ROOT,
    TWO_DGS_BYTES_PER_PRIMITIVE,
)
from scripts.run_paper_3d_benchmark import PAPER_METHODS


LOGGER = logging.getLogger("paper_3d_dtu_summary")


def model_path(old_root: Path, current_root: Path, method: str, scene: str) -> Path:
    root = current_root if method == "2dgs_equal_v3_compact_memory" else old_root
    path = root / method / "dtu" / scene
    return path / "30k" if method.startswith("hexels_") else path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _artifact(model: Path, method: str) -> dict[str, float]:
    if method.startswith("hexels_"):
        variant = method.removeprefix("hexels_")
        summary = _read_json(model / "summary.json")
        primitives = float(
            summary.get("num_active_primitives", summary["num_primitives"])
        )
        training_seconds = sum(
            float(_read_json(model.parent / f"{stage}k/summary.json")["train_seconds"])
            for stage in (15, 20, 25, 30)
        )
        return {
            "primitives": primitives,
            "compact_learned_bytes": primitives
            * HEXELS_COMPACT_BYTES_PER_PRIMITIVE[variant],
            "resident_model_tensor_bytes": float(summary["model_state_bytes"]),
            "training_seconds": training_seconds,
        }
    telemetry = _read_json(model / "training_telemetry.json")
    primitives = float(telemetry["final_points"])
    return {
        "primitives": primitives,
        "compact_learned_bytes": primitives * TWO_DGS_BYTES_PER_PRIMITIVE,
        "resident_model_tensor_bytes": float(telemetry["model_state_bytes"]),
        "training_seconds": float(telemetry["wall_time_seconds"]),
    }


def aggregate_rows(rows: list[dict[str, float]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("Cannot aggregate an empty result list")
    fields = (
        "surface_accuracy",
        "surface_completeness",
        "surface_overall",
        "primitives",
        "compact_learned_bytes",
        "resident_model_tensor_bytes",
        "training_seconds",
    )
    return {
        "scenes": len(rows),
        **{
            field: statistics.fmean(float(row[field]) for row in rows)
            for field in fields
        },
    }


def summarize(
    old_root: Path,
    current_root: Path,
    *,
    scenes: tuple[str, ...] = DTU,
    methods: tuple[str, ...] = PAPER_METHODS,
) -> dict[str, Any]:
    per_scene: dict[str, dict[str, dict[str, float]]] = {}
    aggregates = {}
    for method in methods:
        rows = []
        for scene in scenes:
            model = model_path(old_root, current_root, method, scene)
            metrics = _read_json(model / "dtu_eval/results.json")
            row = {
                "surface_accuracy": float(metrics["mean_d2s"]),
                "surface_completeness": float(metrics["mean_s2d"]),
                "surface_overall": float(metrics["overall"]),
                **_artifact(model, method),
            }
            per_scene.setdefault(scene, {})[method] = row
            rows.append(row)
        aggregates[method] = aggregate_rows(rows)
    return {
        "schema_version": 1,
        "protocol": "paper_3d_dtu_official_geometry_v1",
        "scene_count": len(scenes),
        "method_count": len(methods),
        "scenes": list(scenes),
        "methods": list(methods),
        "aggregates": aggregates,
        "per_scene": per_scene,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True,
                        help="Completed staged-protocol root holding the Hexel and native 2DGS DTU runs")
    parser.add_argument("--current-root", type=Path, required=True,
                        help="Completed paper-protocol root holding the 2dgs_equal_v3_compact_memory DTU runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.old_root.resolve(), args.current_root.resolve())
    _write_json(args.output.resolve(), result)
    LOGGER.info("Wrote %d-scene DTU summary: %s", result["scene_count"], args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
