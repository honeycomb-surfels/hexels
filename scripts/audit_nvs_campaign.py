"""Fail-closed endpoint audit for shared NVS smoke and production workers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.nvs_camera_validation import validate as validate_cameras
from scripts.nvs_budgets import validate as validate_budget


def audit(root: Path) -> dict:
    manifest_path = root / "manifests/jobs.json"
    if not manifest_path.exists():
        manifest_path = root / "manifests/smoke.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for entry in manifest["jobs"]:
        result_path = Path(entry["result"])
        run = result_path.parent
        job = json.loads((run / "job.json").read_text())
        issues = []
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if not result.get("complete"):
            issues.append("missing_or_incomplete_result")
        else:
            if result["id"] != job["id"]:
                issues.append("experiment_identity")
            if not all(math.isfinite(result["aggregate"][metric]) for metric in ("mean_psnr_db", "mean_ssim", "mean_lpips_vgg")):
                issues.append("nonfinite_quality")
            primitives = result["artifact"]["primitives"]
            if primitives <= 0 or (job["method"] == "hexels" and primitives > 2000000):
                issues.append("population")
            canonical = json.loads(Path(job["camera_contract"]).read_text())
            geometry_path = run / "native_camera_geometry.json"
            geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else None
            try:
                validate_cameras(canonical, result["camera_contract"], geometry)
                if job.get("capacity_reference"):
                    validate_budget(result["artifact"], job["capacity_reference"])
            except ValueError as error:
                issues.append(str(error))
            if len(list((run / "renders").glob("*.png"))) != len(result["per_view"]):
                issues.append("render_coverage")
            if result["video"]["frames"] != len(canonical["trajectory"]["frames"]):
                issues.append("video_coverage")
            if result["benchmark"]["median_fps"] <= 0:
                issues.append("fps")
        for phase in ("training", "evaluation"):
            path = Path(job["training_resource"]) if phase == "training" and "training_resource" in job else run / f"resources_{phase}.json"
            resource = json.loads(path.read_text()) if path.exists() else {}
            if not resource.get("complete") or resource.get("peak_gpu_bytes", 0) <= 0:
                issues.append(f"resource_{phase}")
            else:
                measurement = job["measurement"]
                expected_interval = (measurement["resource_sample_seconds"] if phase == "training"
                                     else measurement.get("evaluation_resource_sample_seconds", 0.2))
                if resource["sample_interval_seconds"] != expected_interval:
                    issues.append(f"resource_{phase}_interval")
                policy = measurement.get("training_memory_policy")
                if phase == "training" and policy and (
                        resource.get("training_memory_policy") != policy
                        or resource.get("training_method") != job["method"]):
                    issues.append("resource_training_policy")
        row = {"id": job["id"], "method": job["method"], "dataset": job["dataset"],
               "complete": not issues, "issues": issues, "smoke": job["smoke"]}
        if result.get("complete"):
            row.update(result["aggregate"])
            row.update(primitives=result["artifact"]["primitives"],
                       serialized_model_bytes=result["artifact"]["serialized_model_bytes"],
                       fps=result["benchmark"]["median_fps"])
        rows.append(row)
    return {"complete": bool(rows) and all(row["complete"] for row in rows),
            "completed": sum(row["complete"] for row in rows), "expected": len(rows), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as handle:
            handle.write(text)
    print(text)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
