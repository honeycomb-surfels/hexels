"""Stage native and sparse-budget NVS experiments without releasing ungated jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

from PIL import Image

from scripts import nvs_budgets
from scripts.nvs_campaign import ROOT, cameras, expand
from scripts.prepare_nvs_smoke import PYTHON, training_command, wrapped
from scripts.run_paper_3d_benchmark import _write_manifest
from scripts.setup_nvs_baselines import environment_prefix


def inspect_input(settings: dict) -> dict:
    """Validate registered image files and estimate resident float32 image storage."""
    specification = importlib.util.spec_from_file_location("input_colmap", ROOT / "2d-gaussian-splatting/scene/colmap_loader.py")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    source = Path(settings["source_path"])
    intrinsic = loader.read_intrinsics_binary(source / "sparse/0/cameras.bin")
    poses = loader.read_extrinsics_binary(source / "sparse/0/images.bin")
    pixels, raw_pixels, files = 0, 0, []
    for pose in sorted(poses.values(), key=lambda item: item.name):
        if intrinsic[pose.camera_id].model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Non-pinhole camera in {source}")
        path = source / settings["images"] / Path(pose.name).name
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        raw_pixels += width * height
        pixels += round(width / settings["resolution"]) * round(height / settings["resolution"])
        files.append({"name": path.name, "bytes": path.stat().st_size, "width": width, "height": height})
    with (source / "sparse/0/points3D.bin").open("rb") as handle:
        population, = struct.unpack("<Q", handle.read(8))
    if len(files) < 9 or population < 1:
        raise ValueError(f"Insufficient COLMAP input in {source}")
    hashes = {name: hashlib.sha256((source / "sparse/0" / name).read_bytes()).hexdigest()
              for name in ("cameras.bin", "images.bin", "points3D.bin")}
    return {"images": files, "image_count": len(files), "rgb_float_bytes": pixels * 12,
            "raw_rgba_bytes": raw_pixels * 4, "sfm_points": population, "colmap_sha256": hashes}


def prepare(output: Path, config: dict, anchors: list[Path], smoke: bool, scenes: list[str],
            identifiers: set[str] | None = None, runs_root: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(output)
    reuse_two_dgs = config["planned_training_runs"].get("reused_native_2dgs", 0) > 0
    selected = [run for run in expand(config) if not (reuse_two_dgs and run["arm"] == "2dgs_native")]
    if identifiers is not None:
        unknown = identifiers - {run["id"] for run in selected}
        if unknown:
            raise ValueError(f"Unknown campaign identities: {sorted(unknown)}")
        selected = [run for run in selected if run["id"] in identifiers]
    elif smoke:
        selected = [run for run in selected if run["scene"] in scenes and run["budget_anchor"]
                    and (run["arm"].endswith("count_sparse") or run["arm"].endswith("bytes_sparse_v1"))]
    elif len(selected) != config["planned_training_runs"]["new_training_runs"]:
        raise ValueError("Expanded jobs differ from the approved new-training count")
    budgets = {run["id"]: nvs_budgets.resolve(run, anchors) for run in selected if run["budget_anchor"]}
    scene_keys = sorted({(run["dataset"], run["scene"]) for run in selected})
    inputs = {f"{dataset}/{scene}": inspect_input(cameras(config, dataset, scene)) for dataset, scene in scene_keys}
    _write_manifest(output / "protocol.json", config | {"smoke": smoke})
    _write_manifest(output / "input_audit.json", inputs)
    camera_jobs = []
    for dataset, scene in scene_keys:
        settings = output / "cameras" / dataset / f"{scene}_settings.json"
        contract = settings.with_name(f"{scene}_contract.json")
        _write_manifest(settings, cameras(config, dataset, scene))
        camera_jobs.append({"dataset": dataset, "scene": scene, "method": "cameras", "status": "ready",
            "cwd": str(ROOT), "commands": [[str(PYTHON), "-m", "scripts.evaluate_nvs_campaign", "cameras",
                "--settings", str(settings), "--output", str(contract), "--config", str(output / "protocol.json")]],
            "command_stages": ["canonical_cameras_and_trajectory"]})
    jobs = []
    for run in selected:
        directory = (runs_root if runs_root is not None else output / "runs") / run["id"]
        method = run["method"]
        iteration = 3200 if smoke else 32000 if method == "bbsplat" else 18000 if method == "milo" and run["dataset"] != "dtu" else 30000
        job = {**run, "iteration": iteration, "smoke": smoke, "model_path": str(directory / "model"),
               "result": str(directory / "metrics.json"), "measurement": config["measurement"],
               "camera_settings": str(output / "cameras" / run["dataset"] / f"{run['scene']}_settings.json"),
               "camera_contract": str(output / "cameras" / run["dataset"] / f"{run['scene']}_contract.json")}
        if method == "hexels":
            job["bytes_per_primitive"] = {"v1": 80, "v2": 116, "v3": 260}[run["arm"].split("_")[1]]
        command, cwd = training_command(job, config, smoke)
        if run["budget_anchor"]:
            job["capacity_reference"] = budgets[run["id"]]
            nvs_budgets.apply(command, job, job["capacity_reference"])
        specification = directory / "job.json"
        _write_manifest(specification, job)
        interpreter = PYTHON if method in {"hexels", "2dgs"} else environment_prefix(method) / "bin/python"
        evaluate = [str(interpreter), "-m", "scripts.evaluate_nvs_campaign", "evaluate", "--job", str(specification)]
        provenance = None if smoke else output / "provenance/provenance.json"
        jobs.append({"id": run["id"], "dataset": run["dataset"], "scene": run["scene"], "method": run["arm"],
                     "status": "ready", "cwd": str(ROOT), "result": job["result"],
                     "commands": [wrapped(command, cwd, directory / "resources_training.json", provenance),
                                  wrapped(evaluate, ROOT, directory / "resources_evaluation.json", provenance)],
                     "command_stages": ["train", "evaluate_and_video"]})
    _write_manifest(output / "manifests/cameras.json", {"jobs": camera_jobs, "job_count": len(camera_jobs)})
    _write_manifest(output / "manifests/jobs.json", {"jobs": jobs, "job_count": len(jobs)})
    _write_manifest(output / "release.json", {"released": [], "held": len(jobs),
                    "reason": "awaiting native/capacity gates, measured resources and frozen provenance"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/capacity.json")
    parser.add_argument("--anchors", nargs="+", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scenes", nargs="+", default=["garden", "room", "scan65", "Truck"])
    args = parser.parse_args()
    prepare(args.output.resolve(), json.loads(args.config.read_text()), [path.resolve() for path in args.anchors], args.smoke, args.scenes)


if __name__ == "__main__":
    main()
