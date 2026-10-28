"""Prepare, submit and summarize shared-config surface ablations without scene scripts."""

from __future__ import annotations

import sys

import argparse
import copy
import csv
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
    DEFAULT_CONFIG, ROOT, _model_path, _option, _write_manifest,
    load_config, prepare_training_manifest,
)
from scripts.run_reproduction import DEFAULT_GRES, camera_settings, evaluation_command, set_option

LOGGER = logging.getLogger(__name__)
DEFAULT_PROTOCOL = ROOT / "configs/surface.json"


def prepare(output: Path, config: dict[str, Any], protocol: dict[str, Any], smoke: bool) -> list[Path]:
    """Freeze independent ablations; short smoke stages exercise resume and every option."""
    output = output.resolve()
    config = copy.deepcopy(config)
    config["seed"] = protocol["seed"]
    dataset = protocol["dataset"]
    _write_manifest(output / "protocol.json", protocol | {"smoke": smoke})
    templates = prepare_training_manifest(
        output_root=output / "template", manifest_path=output / "template.json",
        datasets={dataset}, scenes=set(protocol["scenes"]), config=config,
    )
    templates = [job for job in templates["jobs"] if job["method"] == f"hexels_{protocol['variant']}"]
    if any(job["status"] != "ready" for job in templates):
        raise ValueError("Missing benchmark inputs: " + str(templates))
    jobs_by_stage: dict[str, list[dict[str, Any]]] = {"extraction": []}
    baseline_root = (ROOT / protocol["baseline_root"]).resolve()
    for template in templates:
        scene = template["scene"]
        camera_config = output / "cameras" / f"{dataset}_{scene}.json"
        _write_manifest(camera_config, camera_settings(config, dataset, scene, "geometry"))
        for arm, settings in protocol["extraction"].items():
            job = copy.deepcopy(template)
            model = baseline_root / "geometry/training" / template["method"] / dataset / scene / "30k"
            checkpoint = model / "checkpoints/ckpt_030000.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            destination = output / "extraction" / arm / dataset / scene
            commands = copy.deepcopy(job["commands"][4:])
            set_option(commands[0], "--checkpoint", checkpoint)
            set_option(commands[0], "--model_path", model)
            set_option(commands[0], "--output_model_path", destination)
            for option, value in settings.items():
                set_option(commands[0], f"--{option}", value)
            set_option(commands[0], "--preview_views", 3)
            set_option(commands[1], "--input_mesh", destination / "train/ours_30000/fuse_post.ply")
            set_option(commands[1], "--output_dir", destination / "dtu_eval")
            job.update(method=f"{template['method']}_{arm}", arm=arm, stage="extraction",
                       model_path=str(destination), iteration=30000, seed=protocol["seed"],
                       commands=commands, command_stages=["mesh_extraction", "geometry_evaluation"])
            jobs_by_stage["extraction"].append(job)
        for arm, settings in protocol["training"].items():
            job = copy.deepcopy(template)
            stage = settings["stage"]
            method = f"{template['method']}_{arm}"
            old_base = _model_path(job).parent
            new_base = output / stage / "training" / method / dataset / scene
            commands = [[str(value).replace(str(old_base), str(new_base)) for value in command]
                        for command in job["commands"]]
            initial_root = protocol.get("initial_geometry_root")
            if initial_root is not None:
                geometry = (ROOT / initial_root / scene / "initial_geometry.pt").resolve()
                if not geometry.is_file():
                    raise FileNotFoundError(geometry)
                for command in commands:
                    if "hexels.train" in command:
                        set_option(command, "--init_geometry_path", geometry)
            iterations = [40, 80, 120, 160] if smoke else [15000, 20000, 25000, 30000]
            if smoke:
                for command in commands:
                    for index, value in enumerate(command):
                        for previous, current in zip((15000, 20000, 25000, 30000), iterations):
                            value = value.replace(f"/{previous // 1000}k", f"/step_{current}")
                            value = value.replace(f"ckpt_{previous:06d}", f"ckpt_{current:06d}")
                            value = value.replace(f"ours_{previous}", f"ours_{current}")
                        command[index] = value
            for iteration, command in zip(iterations, commands[:4]):
                set_option(command, "--iterations", iteration)
                set_option(command, "--save_every", iteration)
                set_option(command, "--render_every", 0)
                for option, value in settings["options"].items():
                    if smoke and option.endswith("_start"):
                        value = 20
                    if smoke and option in {"ray_distortion_ramp_steps", "lr_means3d_max_steps"}:
                        value = 40 if option.endswith("ramp_steps") else iterations[-1]
                    set_option(command, f"--{option}", value)
            set_option(commands[4], "--preview_views", 3)
            job["commands"] = commands
            model = _model_path(job)
            telemetry = output / stage / "metrics" / f"{dataset}_{scene}_{arm}.json"
            job["method"] = method
            commands.append(evaluation_command(
                job, model, camera_config, telemetry, config["python"], iterations, benchmark_only=True,
            ))
            if smoke:
                summary_indices = [index + 1 for index, value in enumerate(commands[-1]) if value == "--summary"]
                for index, iteration in zip(summary_indices, iterations):
                    commands[-1][index] = str(new_base / f"step_{iteration}/summary.json")
                commands = commands[:4] + commands[-1:]
            command_stages = [f"train_{iteration}" for iteration in iterations]
            if not smoke:
                command_stages.extend(["mesh_extraction", "geometry_evaluation"])
            job.update(arm=arm, stage=stage, model_path=str(model), iteration=iterations[-1],
                       seed=protocol["seed"], telemetry=str(telemetry), commands=commands,
                       command_stages=command_stages + ["render_benchmark"])
            jobs_by_stage.setdefault(stage, []).append(job)
    paths = []
    for stage, jobs in jobs_by_stage.items():
        path = output / stage / "manifests/jobs.json"
        _write_manifest(path, {"protocol": protocol["protocol"], "smoke": smoke, "jobs": jobs})
        paths.append(path)
    return paths


def submit(output: Path, stages: list[str], gate: str, qos: str | None = None,
           partition: str | None = None, gres: str = DEFAULT_GRES) -> dict[str, str]:
    """Submit uncapped GPU arrays after an explicit successful validation job."""
    record = output / "submission.json"
    submitted = json.loads(record.read_text()) if record.exists() else {}
    logs = output / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        if stage in submitted:
            continue
        manifest = output / stage / "manifests/jobs.json"
        jobs = load_jobs(manifest)
        command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
                   *([f"--qos={qos}"] if qos else []), f"--gres={gres}",
                   "--cpus-per-task=8", "--mem=32000M", "--time=24:00:00", "--no-requeue",
                   f"--chdir={ROOT}", f"--array=0-{len(jobs) - 1}", f"--dependency=afterok:{gate}",
                   f"--job-name=surface_{stage}", f"--output={logs}/%x_%A_%a.log",
                   str(ROOT / "scripts/run_benchmark_array.sh"), str(manifest),
                   str(output / stage / "runner")]
        submitted[stage] = subprocess.check_output(command, text=True).strip().split(";")[0]
        record.write_text(json.dumps(submitted, indent=2) + "\n")
        LOGGER.info("Submitted %s: %s (%d tasks, no cap)", stage, submitted[stage], len(jobs))
    summary_record = output / f"summary_submission_{len(submitted)}.json"
    if submitted and not summary_record.exists():
        command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
                   *([f"--qos={qos}"] if qos else []), "--cpus-per-task=1",
                   "--mem=4000M", "--time=00:15:00", f"--chdir={ROOT}", "--job-name=surface_summary",
                   f"--dependency=afterany:{':'.join(submitted.values())}",
                   f"--output={logs}/summary_%j.log", "--wrap",
                   shlex.join([str(Path(sys.executable)), "-m", "scripts.run_surface_ablation",
                               "summarize", "--output-root", str(output)])]
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        summary_record.write_text(json.dumps({"job_id": job_id, "command": command}, indent=2) + "\n")
    return submitted


def summarize(output: Path) -> dict[str, Any]:
    """Keep missing/failed rows visible; compare only complete identical scene cohorts."""
    protocol = json.loads((output / "protocol.json").read_text())
    rows = []
    for manifest in sorted(output.glob("*/manifests/jobs.json")):
        for job in json.loads(manifest.read_text())["jobs"]:
            model = Path(job["model_path"])
            status_file = manifest.parents[1] / "runner/status" / f"{job_name(job)}.json"
            status = json.loads(status_file.read_text()) if status_file.exists() else {"status": "pending"}
            result_file = model / "dtu_eval/results.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else {}
            row = {key: job[key] for key in ("scene", "stage", "arm", "seed")}
            row.update(status=status["status"], **result)
            row["complete"] = status["status"] == "completed" and all(
                isinstance(result.get(key), (int, float)) and math.isfinite(result[key])
                for key in ("mean_d2s", "mean_s2d", "overall"))
            extraction_file = model / f"train/ours_{job['iteration']}/extraction.json"
            if extraction_file.exists():
                row.update(json.loads(extraction_file.read_text()))
            row["command_seconds"] = status.get("successful_command_seconds")
            row["geometry_evaluation_seconds"] = next((item["seconds"] for item in status.get("command_timings", [])
                                                       if item["stage"] == "geometry_evaluation"), None)
            if job.get("telemetry") and Path(job["telemetry"]).exists():
                metrics = json.loads(Path(job["telemetry"]).read_text())
                artifact = metrics.get("artifact", {})
                row.update(primitives=artifact.get("primitives"), fps=metrics.get("benchmark", {}).get("fps"),
                           train_seconds=artifact.get("training", {}).get("optimization_seconds"),
                           compact_bytes=artifact.get("compact_learned_bytes"))
            rows.append(row)
    cohorts = {}
    scenes = set(protocol["scenes"])
    for arm in dict.fromkeys(row["arm"] for row in rows):
        selected = [row for row in rows if row["arm"] == arm]
        complete = {row["scene"] for row in selected if row["complete"]} == scenes
        cohorts[arm] = {"complete": complete, "completed": sum(row["complete"] for row in selected),
                        "expected": len(scenes)}
        if complete:
            cohorts[arm]["overall_mm"] = statistics.fmean(row["overall"] for row in selected)
            control = "expected" if selected[0]["stage"] == "extraction" else "baseline"
            controls = {row["scene"]: row for row in rows if row["arm"] == control and row["complete"]}
            if set(controls) == scenes:
                baseline_mean = statistics.fmean(row["overall"] for row in controls.values())
                reduction = 1 - cohorts[arm]["overall_mm"] / baseline_mean
                wins = sum(row["overall"] < controls[row["scene"]]["overall"] for row in selected)
                cohorts[arm].update(reduction_fraction=reduction, scene_wins=wins)
    baseline = ROOT / protocol["baseline_root"] / "results.json"
    historical = [row for row in json.loads(baseline.read_text())
                  if row["group"] == "geometry" and row["dataset"] == protocol["dataset"]
                  and row["scene"] in scenes and row["method"] in
                  {f"hexels_{protocol['variant']}", "2dgs_native", "2dgs_equal_primitives"}]
    summary = {"smoke": protocol["smoke"], "scenes": protocol["scenes"], "cohorts": cohorts,
               "reference_same_scene_rows": historical, "rows": rows}
    (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Cohorts: %s", cohorts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "summarize", "gate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--stages", nargs="+", default=["extraction", "objectives"])
    parser.add_argument("--gate-job")
    parser.add_argument("--qos", help="Slurm QoS; omit to use the account default")
    parser.add_argument("--partition", help="Slurm partition; omit to use the scheduler default")
    parser.add_argument("--gres", default=DEFAULT_GRES, help="Slurm GPU request per task, e.g. gpu:a5000:1 (default: %(default)s)")
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.action == "prepare":
        protocol = json.loads(args.protocol.read_text())
        if args.scenes:
            protocol["scenes"] = args.scenes
        prepare(output, load_config(args.config), protocol, args.smoke)
    elif args.action == "submit":
        if not args.gate_job:
            parser.error("submit requires --gate-job for the batch validation dependency")
        submit(output, args.stages, args.gate_job, args.qos, args.partition, args.gres)
    else:
        summary = summarize(output)
        if args.action == "gate":
            for manifest in output.glob("*/manifests/jobs.json"):
                for job in load_jobs(manifest):
                    path = manifest.parents[1] / "runner/status" / f"{job_name(job)}.json"
                    if not path.exists() or json.loads(path.read_text())["status"] != "completed":
                        raise ValueError(f"Validation incomplete or failed: {job_name(job)}")
                    if job.get("telemetry"):
                        telemetry = json.loads(Path(job["telemetry"]).read_text())
                        if not telemetry["complete"] or not math.isfinite(telemetry["benchmark"]["fps"]):
                            raise ValueError(f"Invalid render telemetry: {job_name(job)}")
                    elif not summary["cohorts"][job["arm"]]["complete"]:
                        raise ValueError(f"Invalid surface results: {job_name(job)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
