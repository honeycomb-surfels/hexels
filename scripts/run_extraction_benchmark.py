"""Benchmark colored primitive/contact meshes and TSDF on matched all-view checkpoints."""

from __future__ import annotations

import sys

import argparse
import copy
import hashlib
import json
import logging
import math
import shlex
import statistics
import subprocess
from pathlib import Path
from typing import Any

from scripts.run_minimum_3d_benchmark import job_name, load_jobs
from scripts.run_paper_3d_benchmark import (
    DEFAULT_CONFIG, ROOT, _write_manifest, load_config, prepare_training_manifest,
)
from scripts.run_reproduction import DEFAULT_GRES, set_option

LOGGER = logging.getLogger(__name__)
METHODS = ["hexels_v1", "2dgs_native", "2dgs_equal_primitives"]
SCENES = ["scan24", "scan37", "scan63", "scan65", "scan83", "scan106"]
MODES = ["raw", "contact", "tsdf"]
SOURCES = ["hexels/contact_mesh.py", "scripts/extract_contact_mesh.py",
           "scripts/extract_hexels_dtu_mesh.py", "2d-gaussian-splatting/render.py",
           "2d-gaussian-splatting/utils/mesh_utils.py", "scripts/run_extraction_benchmark.py",
           "scripts/prepare_minimum_3d_benchmark.py", "submodules/hexel-kernel/RELEASE_MANIFEST.json",
           "baselines/2d-gaussian-splatting/adaptations.patch"]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze read-only checkpoints, identical evaluator commands and fresh output paths."""
    if args.repeats < 1 or set(args.datasets) - {"dtu", "tanks_and_temples"}:
        raise ValueError("Geometry datasets and positive repeat counts are required")
    config = load_config(args.config)
    output = args.output_root.resolve()
    source_root = args.checkpoint_root.resolve()
    template = prepare_training_manifest(
        output_root=output / "unused_training_template", manifest_path=output / "template.json",
        config=config, datasets=set(args.datasets), scenes=set(args.scenes),
    )
    jobs = []
    for original in template["jobs"]:
        if original["method"] not in args.methods:
            continue
        if original["status"] != "ready" or original["task"] != "geometry":
            raise ValueError(f"Missing geometry inputs or non-geometry protocol: {original}")
        method, dataset, scene = (original[key] for key in ("method", "dataset", "scene"))
        family = "hexels" if method.startswith("hexels_") else "2dgs"
        model = source_root / method / dataset / scene
        if family == "hexels":
            model = model / "30k"
        checkpoint = model / ("checkpoints/ckpt_030000.pt" if family == "hexels"
                              else "point_cloud/iteration_30000/point_cloud.ply")
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        if family == "hexels":
            if json.loads((model / "config.json").read_text()).get("eval", False):
                raise ValueError(f"Expected an all-view checkpoint: {model}")
        elif "eval=True" in (model / "cfg_args").read_text():
            raise ValueError(f"Expected an all-view checkpoint: {model}")
        destination = output / "exports" / method / dataset / scene
        commands, stages = [], []
        for repeat in range(args.repeats):
            ordering = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
            for mode in ordering:
                target = destination / f"trial_{repeat}" / mode
                if mode == "tsdf":
                    command = copy.deepcopy(original["commands"][-2])
                    if family == "hexels":
                        set_option(command, "--checkpoint", checkpoint)
                        set_option(command, "--model_path", model)
                        set_option(command, "--depth_mode", "median")
                        set_option(command, "--preview_views", 0)
                    else:
                        command[1] = str(ROOT / "2d-gaussian-splatting/render.py")
                        set_option(command, "-m", model)
                        if "--skip_train" not in command:
                            command.append("--skip_train")
                    set_option(command, "--output_model_path", target)
                else:
                    command = [config["python"], "-m", "scripts.extract_contact_mesh", "--family", family,
                               "--model-path", str(model), "--output-dir", str(target), "--mode", mode]
                commands.append(command)
                stages.append(f"extract_{mode}_{repeat}")
        for mode in MODES:
            target = destination / "trial_0" / mode
            mesh = target / ("train/ours_30000/fuse_post.ply" if mode == "tsdf" else "mesh.ply")
            command = copy.deepcopy(original["commands"][-1])
            mesh_option, output_option = (("--input_mesh", "--output_dir") if dataset == "dtu"
                                          else ("--ply-path", "--out-dir"))
            set_option(command, mesh_option, mesh)
            set_option(command, output_option, target / f"{dataset}_eval")
            commands.append(command)
            stages.append(f"evaluate_{mode}")
        jobs.append({"dataset": dataset, "scene": scene, "method": method, "status": "ready",
                     "family": family, "cwd": str(ROOT), "commands": commands, "command_stages": stages,
                     "source_model": str(model), "checkpoint": str(checkpoint),
                     "checkpoint_sha256": file_hash(checkpoint), "destination": str(destination)})
    expected = {(dataset, scene, method) for dataset in args.datasets
                for scene in args.scenes if any(job["dataset"] == dataset and job["scene"] == scene for job in jobs)
                for method in args.methods}
    actual = {(job["dataset"], job["scene"], job["method"]) for job in jobs}
    if not jobs or expected != actual or {job["scene"] for job in jobs} != set(args.scenes):
        raise ValueError("Incomplete requested scene/method coverage")
    manifest = {"protocol": "colored_contact_extraction_v1", "repeats": args.repeats,
                "modes": MODES, "smoke": args.smoke, "jobs": jobs,
                "source_hashes": {name: file_hash(ROOT / name) for name in SOURCES},
                "quality": "same pinned evaluator on filled patches or postprocessed TSDF; no NVS",
                "timing": "fresh processes, rotated mode order; load, construction, writes and evaluator separate",
                "contact_rules": {"neighbors": 16, "max_angle_degrees": 30, "max_endpoint_fraction": 0.25,
                                  "max_plane_fraction": 0.1, "max_length_ratio": 2,
                                  "max_vertex_shift_fraction": 0.2, "opacity_threshold": 0.1,
                                  "gaussian_sides": 12, "visibility_test": False}}
    _write_manifest(output / "manifest.json", manifest)
    LOGGER.info("Prepared %d matched tasks (%d repeats), smoke=%s", len(jobs), args.repeats, args.smoke)
    return manifest


def verify(output: Path) -> None:
    """Refuse source or checkpoint drift after the manifest was frozen."""
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Source drift: {name}; prepare a fresh run")
    for job in manifest["jobs"]:
        if file_hash(Path(job["checkpoint"])) != job["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint drift: {job['checkpoint']}")


def submit(output: Path, gate: str | None, qos: str | None = None,
           partition: str | None = None, gres: str = DEFAULT_GRES) -> dict[str, str]:
    """Use the existing generic batch worker, without a concurrency cap."""
    verify(output)
    record = output / "submission.json"
    if record.exists():
        return json.loads(record.read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    logs = output / "slurm"
    logs.mkdir(exist_ok=True)
    large = any(job["dataset"] == "tanks_and_temples" for job in manifest["jobs"])
    command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
               *([f"--qos={qos}"] if qos else []), f"--gres={gres}",
               f"--cpus-per-task={32 if large else 8}", f"--mem={128000 if large else 32000}M",
               "--time=12:00:00", "--no-requeue", f"--chdir={ROOT}", "--job-name=contact_mesh",
               f"--array=0-{len(manifest['jobs']) - 1}", f"--output={logs}/%x_%A_%a.log"]
    if gate:
        command.append(f"--dependency=afterok:{gate}")
    command.extend([str(ROOT / "scripts/run_benchmark_array.sh"), str(output / "manifest.json"),
                    str(output / "runner")])
    job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
    result = {"array": job_id}
    record.write_text(json.dumps(result, indent=2) + "\n")
    summary_command = [str(Path(sys.executable)), "-m", "scripts.run_extraction_benchmark",
                       "summarize", "--output-root", str(output)]
    result["summary"] = subprocess.check_output([
        "sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
        *([f"--qos={qos}"] if qos else []), "--cpus-per-task=1", "--mem=4000M",
        "--time=00:15:00", f"--chdir={ROOT}", f"--dependency=afterany:{job_id}",
        "--job-name=contact_summary", f"--output={logs}/summary_%j.log", "--wrap", shlex.join(summary_command),
    ], text=True).strip().split(";")[0]
    record.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("Submitted %s", result)
    return result


def summarize(output: Path) -> dict[str, Any]:
    """Only aggregate complete, identical scene cohorts; preserve every failure."""
    manifest = json.loads((output / "manifest.json").read_text())
    rows = []
    for job in manifest["jobs"]:
        status_path = output / "runner/status" / f"{job_name(job)}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending"}
        for mode in MODES:
            destination = Path(job["destination"])
            target = destination / "trial_0" / mode
            metrics = {}
            if job["dataset"] == "dtu":
                metric_path = target / "dtu_eval/results.json"
                if metric_path.exists():
                    metrics = json.loads(metric_path.read_text())
            else:
                metric_path = target / "tanks_and_temples_eval" / f"{job['scene']}.prf_tau_plotstr.txt"
                if metric_path.exists():
                    values = list(map(float, metric_path.read_text().split()))
                    metrics = dict(zip(("precision", "recall", "fscore", "threshold", "stretch"), values, strict=True))
            trials = []
            for repeat in range(manifest["repeats"]):
                directory = destination / f"trial_{repeat}" / mode
                timing_path = directory / ("train/ours_30000/extraction.json" if mode == "tsdf" else "extraction.json")
                if timing_path.exists():
                    timing = json.loads(timing_path.read_text())
                    if mode == "tsdf" and job["family"] == "hexels":
                        timing["extraction_seconds"] = timing["mesh_seconds"]
                    trials.append(timing)
            quality_keys = ("mean_d2s", "mean_s2d", "overall") if job["dataset"] == "dtu" else ("precision", "recall", "fscore")
            complete = (status["status"] == "completed" and len(trials) == manifest["repeats"]
                        and all(isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
                                for key in quality_keys))
            row = {key: job[key] for key in ("dataset", "scene", "method", "family")}
            row.update(mode=mode, status=status["status"], complete=complete, metrics=metrics, trials=trials,
                       destination=str(target), command_timings=status.get("command_timings", []))
            if trials:
                for key in ("extraction_seconds", "load_seconds", "total_seconds", "write_seconds"):
                    row[f"median_{key}"] = statistics.median(trial[key] for trial in trials)
                row["min_extraction_seconds"] = min(trial["extraction_seconds"] for trial in trials)
                row["max_extraction_seconds"] = max(trial["extraction_seconds"] for trial in trials)
            rows.append(row)
    cohorts = {}
    for dataset in sorted({job["dataset"] for job in manifest["jobs"]}):
        expected = {job["scene"] for job in manifest["jobs"] if job["dataset"] == dataset}
        for method in sorted({job["method"] for job in manifest["jobs"]}):
            for mode in MODES:
                selected = [row for row in rows if (row["dataset"], row["method"], row["mode"]) == (dataset, method, mode)]
                complete = {row["scene"] for row in selected if row["complete"]} == expected
                cohort = {"complete": complete, "completed": sum(row["complete"] for row in selected),
                          "expected": len(expected)}
                if complete:
                    metric = "overall" if dataset == "dtu" else "fscore"
                    cohort[metric] = statistics.fmean(row["metrics"][metric] for row in selected)
                    cohort["extraction_seconds"] = statistics.fmean(row["median_extraction_seconds"] for row in selected)
                    cohort["total_seconds"] = statistics.fmean(row["median_total_seconds"] for row in selected)
                cohorts[f"{dataset}/{method}/{mode}"] = cohort
    result = {"smoke": manifest["smoke"], "cohorts": cohorts, "rows": rows}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    LOGGER.info("Cohorts: %s", cohorts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "summarize", "verify", "gate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "outputs/surface/geometry/training")
    parser.add_argument("--datasets", nargs="+", default=["dtu"])
    parser.add_argument("--scenes", nargs="+", default=SCENES)
    parser.add_argument("--methods", nargs="+", choices=METHODS + ["hexels_v2", "hexels_v3"], default=METHODS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate-job")
    parser.add_argument("--qos", help="Slurm QoS; omit to use the account default")
    parser.add_argument("--partition", help="Slurm partition; omit to use the scheduler default")
    parser.add_argument("--gres", default=DEFAULT_GRES, help="Slurm GPU request per task, e.g. gpu:a5000:1 (default: %(default)s)")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "submit":
        submit(args.output_root, args.gate_job, args.qos, args.partition, args.gres)
    elif args.action == "verify":
        verify(args.output_root)
    else:
        result = summarize(args.output_root)
        if args.action == "gate" and not all(row["complete"] for row in result["rows"]):
            raise ValueError("Incomplete or failed extraction/quality validation")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
