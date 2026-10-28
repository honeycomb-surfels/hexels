from __future__ import annotations

import sys

import argparse
import copy
import csv
import json
import logging
import shlex
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from scripts.prepare_minimum_3d_benchmark import HEXELS_COMPACT_BYTES_PER_PRIMITIVE, MIPNERF360_OUTDOOR
from scripts.run_paper_3d_benchmark import (
    DEFAULT_CONFIG, ROOT, _model_path, _option, _write_manifest, load_config, prepare_training_manifest,
    require_executable,
)
from scripts.run_minimum_3d_benchmark import job_name

LOGGER = logging.getLogger(__name__)
DEFAULT_GRES = "gpu:1"
COLMAP_REMEDY = ("Create it from the repository root with `CONDA_OVERRIDE_CUDA=12.1 CONDA_CHANNEL_PRIORITY=flexible "
                 "conda env create -p .envs/colmap37 -f environment-colmap.yml`, or pass --colmap with an existing binary.")


def set_option(command: list[str], option: str, value: object) -> None:
    if option in command:
        command[command.index(option) + 1] = str(value)
    else:
        command.extend([option, str(value)])


def camera_settings(config: dict[str, Any], dataset: str, scene: str, task: str) -> dict[str, Any]:
    settings = config["datasets"][dataset]
    images = "images"
    if dataset == "mipnerf360":
        images = "images_4" if scene in MIPNERF360_OUTDOOR else "images_2"
    return {"source_path": str(Path(settings["root"]) / scene), "images": images,
            "resolution": settings["resolution"], "eval": task == "nvs", "llffhold": 8, "data_device": "cpu"}


def evaluation_command(
    job: dict[str, Any], model: Path, camera_config: Path, output: Path,
    python: str, stages: list[int], *, benchmark_only: bool = False,
) -> list[str]:
    hexels = job["method"].startswith("hexels_")
    command = [python, str(ROOT / "scripts/evaluate_paper_nvs.py"),
        "--dataset", job["dataset"], "--scene", job["scene"], "--source", job["source"],
        "--method", job["method"], "--model-kind", "hexels" if hexels else "2dgs",
        "--model-path", str(model), "--iteration", str(stages[-1]),
        "--camera-config", str(camera_config), "--output", str(output)]
    if hexels:
        variant = job["method"].split("_")[1]
        command.extend(["--compact-bytes-per-primitive", str(HEXELS_COMPACT_BYTES_PER_PRIMITIVE[variant])])
        for stage in stages:
            directory = model if len(stages) == 1 else model.parent / f"{stage // 1000}k"
            command.extend(["--summary", str(directory / "summary.json")])
    if benchmark_only:
        command.extend(["--split", "train", "--benchmark-only"])
    return command


def prepare(output: Path, config: dict[str, Any], scenes: set[str], smoke_iterations: int) -> list[Path]:
    """Prepare self-contained scene jobs; no cross-method checkpoint dependencies."""
    paths = []
    for group in ("nvs", "geometry"):
        selected = copy.deepcopy(config)
        datasets = set(selected["datasets"])
        if group == "geometry":
            datasets &= {"dtu", "tanks_and_temples"}
        if not datasets:
            continue
        for dataset in datasets:
            selected["datasets"][dataset]["task"] = group
        group_scenes = scenes
        if scenes and group == "geometry":
            from scripts.run_paper_3d_benchmark import DATASET_SCENES
            group_scenes = scenes & {scene for dataset in datasets for scene in DATASET_SCENES[dataset]}
            if not group_scenes:
                continue
        base = output / group
        manifest = prepare_training_manifest(output_root=base / "training",
            manifest_path=base / "manifests/source.json", datasets=datasets,
            scenes=group_scenes, config=selected)
        if any(job["status"] != "ready" for job in manifest["jobs"]):
            raise ValueError(f"Missing inputs in {base / 'manifests/source.json'}")
        for job in manifest["jobs"]:
            hexels = job["method"].startswith("hexels_")
            count = 4 if hexels else 1
            model = _model_path(job)
            if group == "nvs":
                job["commands"] = job["commands"][:count]
            if hexels:
                for command in job["commands"][:count]:
                    if "--skip_postprocess" not in command:
                        command.append("--skip_postprocess")
                    set_option(command, "--render_every", 0)
            else:
                set_option(job["commands"][0], "--port", 0)
                set_option(job["commands"][0], "--ip", "127.0.0.1")
            stages = [15_000, 20_000, 25_000, 30_000] if hexels else [30_000]
            camera_config = base / "cameras" / job["dataset"] / f"{job['scene']}.json"
            _write_manifest(camera_config, camera_settings(config, job["dataset"], job["scene"], group))
            result = base / "evaluation" / job["method"] / job["dataset"] / f"{job['scene']}.json"
            evaluation = evaluation_command(job, model, camera_config, result, config["python"], stages,
                benchmark_only=group == "geometry")
            job["command_stages"] = [f"train_{stage}" for stage in stages]
            if group == "geometry":
                job["command_stages"].extend(["mesh_extraction", "geometry_evaluation"])
            job["commands"].append(evaluation)
            job["command_stages"].append("render_benchmark" if group == "geometry" else "nvs_evaluation")
            if smoke_iterations:
                replacements = dict(zip([15_000, 20_000, 25_000, 30_000],
                    [smoke_iterations // 4, smoke_iterations // 2, 3 * smoke_iterations // 4, smoke_iterations]))
                for command in job["commands"]:
                    for index, value in enumerate(command):
                        for original, shortened in replacements.items():
                            value = value.replace(f"ckpt_{original:06d}", f"ckpt_{shortened:06d}")
                            value = value.replace(f"ours_{original}", f"ours_{shortened}")
                        command[index] = value
                    for option in ("--iterations", "--iteration", "--save_every", "--save_iterations", "--position_lr_max_steps"):
                        if option in command:
                            current = int(_option(command, option))
                            set_option(command, option, replacements.get(current, current))
                job["command_stages"] = [
                    f"train_{replacements[int(stage.removeprefix('train_'))]}" if stage.startswith("train_") else stage
                    for stage in job["command_stages"]
                ]
            job["result"] = str(result)
            job["model_path"] = str(model)
            job["seed"] = config["seed"]
            job["group"] = group
        manifest["protocol"] = "a5000_generic_reproduction_v1"
        manifest["smoke_iterations"] = smoke_iterations
        path = base / "manifests/jobs.json"
        _write_manifest(path, manifest)
        paths.append(path)
    return paths


def prepare_mvs(output: Path, config: dict[str, Any], scenes: set[str], colmap: Path) -> Path:
    """One shared MVS recipe, retaining the original candidate-pool sampling contract."""
    from scripts.run_paper_3d_benchmark import DATASET_SCENES
    if any(not scenes or scene in scenes for dataset in {"mipnerf360"} & config["datasets"].keys()
           for scene in DATASET_SCENES[dataset]):
        require_executable(colmap, "Mip-NeRF 360 MVS preprocessing", COLMAP_REMEDY)
    jobs = []
    for dataset in {"mipnerf360"} & config["datasets"].keys():
        for scene in DATASET_SCENES[dataset]:
            if scenes and scene not in scenes:
                continue
            base = output / "mvs" / dataset / scene
            settings = camera_settings(config, dataset, scene, "nvs")
            cameras = base / "cameras.json"
            _write_manifest(cameras, settings)
            job = {"dataset": dataset, "scene": scene, "source": settings["source_path"],
                "method": "hexels_v3_mvs", "task": "nvs", "group": "mvs", "seed": config["seed"],
                "status": "ready", "cwd": str(ROOT), "commands": [], "command_stages": [], "results": []}
            job["commands"].append([config["python"], str(ROOT / "scripts/prepare_mvs_geometry.py"),
                "--source", settings["source_path"], "--images", settings["images"],
                "--output", str(base / "initialization"), "--colmap", str(colmap), "--seed", str(config["seed"])])
            job["command_stages"].append("mvs_preprocessing")
            for iteration in (30_000, 33_000):
                model = base / f"{iteration // 1000}k"
                train = [config["python"], "-m", "hexels.train", "--source_path", settings["source_path"],
                    "--model_path", str(model), "--seed", str(config["seed"]), "--iterations", str(iteration),
                    "--camera_sampling", "random", "--init_geometry_path", str(base / "initialization/initial_geometry.pt"),
                    "--spatial_lr_scale", "False", "--lr_means3d_final", "0.0", "--lambda_dssim", "0.05",
                    "--color_model", "linear", "--view_model", "sh3", "--use_view_grads", "True",
                    "--save_every", str(iteration), "--render_every", "0", "--images", settings["images"],
                    "--eval", "--llffhold", "8", "--skip_postprocess", "--quality_safe_retry",
                    "--lr_sh_coeffs", "0.0005", "--sh_progressive_interval", "0"]
                if iteration == 33_000:
                    train.extend(["--resume_checkpoint", str(base / "30k/checkpoints/ckpt_030000.pt"),
                        "--resume_geometry_lr_scale", "1.0", "--resume_appearance_lr_scale", "1.0",
                        "--resume_parameter_policy", "appearance_only"])
                job["commands"].append(train)
                job["command_stages"].append(f"train_{iteration}")
                result = model / "evaluation.json"
                evaluate = evaluation_command(job, model, cameras, result, config["python"], [iteration])
                if iteration == 33_000:
                    evaluate.extend(["--summary", str(base / "30k/summary.json")])
                evaluate.extend(["--preprocessing-artifact", str(base / "initialization/mvs_artifact.json")])
                job["commands"].append(evaluate)
                job["command_stages"].append(f"nvs_evaluation_{iteration}")
                job["results"].append(str(result))
            jobs.append(job)
    path = output / "mvs/manifests/jobs.json"
    _write_manifest(path, {"protocol": "a5000_mvs_v1", "job_count": len(jobs), "jobs": jobs})
    return path


def job_resources(group: str, dataset: str) -> tuple[int, int]:
    """CPU and host-memory defaults; T&T surface fusion is substantially larger."""
    return (32, 128000) if group == "geometry" and dataset == "tanks_and_temples" else (8, 32000)


def require_completed(manifest: Path) -> None:
    """Release a production group only after every named validation task succeeds."""
    for job in json.loads(manifest.read_text())["jobs"]:
        status_path = manifest.parents[1] / "runner/status" / f"{job_name(job)}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get("status") != "completed":
            raise ValueError(f"Validation is not complete: {job_name(job)} ({status.get('status', 'missing')})")


def submit(paths: list[Path], output: Path, qos: str | None = None, groups: set[str] | None = None,
           partition: str | None = None, gres: str = DEFAULT_GRES) -> dict[str, str]:
    """Submit unthrottled arrays; Slurm enforces user and resource limits."""
    record = output / "submission.json"
    logs = output / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    submissions = json.loads(record.read_text()) if record.exists() else {}
    for path in paths:
        manifest = json.loads(path.read_text())
        if not manifest["jobs"]:
            continue
        group = path.parents[1].name
        if group in submissions:
            LOGGER.info("Already submitted %s: %s", group, submissions[group])
            continue
        batches = {group: list(range(len(manifest["jobs"])))}
        if group == "geometry":
            batches = {f"geometry_{dataset}": [index for index, job in enumerate(manifest["jobs"])
                if job["dataset"] == dataset] for dataset in sorted({job["dataset"] for job in manifest["jobs"]})}
        for name, indices in batches.items():
            if name in submissions or (groups and name not in groups):
                continue
            cpus, memory_mb = job_resources(group, manifest["jobs"][indices[0]].get("dataset", ""))
            array = (f"{indices[0]}-{indices[-1]}" if indices == list(range(indices[0], indices[-1] + 1))
                else ",".join(map(str, indices)))
            command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
                *([f"--qos={qos}"] if qos else []), f"--gres={gres}",
                f"--cpus-per-task={cpus}", f"--mem={memory_mb}M", "--time=24:00:00", f"--chdir={ROOT}",
                f"--job-name=hexels_{name}", f"--array={array}",
                f"--export=ALL,HEXELS_WORKER_COUNT={len(manifest['jobs'])}",
                f"--output={logs}/%x_%A_%a.log", str(ROOT / "scripts/run_benchmark_array.sh"),
                str(path), str(path.parents[1] / "runner")]
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            submissions[name] = job_id
            record.write_text(json.dumps(submissions, indent=2) + "\n")
            LOGGER.info("Submitted %s: %s (%s jobs)", name, job_id, len(indices))
    summary_record = output / f"summary_submission_{len(submissions)}.json"
    if submissions and not summary_record.exists():
        command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
            *([f"--qos={qos}"] if qos else []),
            "--cpus-per-task=1", "--mem=4G", "--time=00:15:00", f"--chdir={ROOT}",
            "--job-name=hexels_summary", f"--dependency=afterany:{':'.join(submissions.values())}",
            f"--output={logs}/summary_%j.log", "--wrap",
            shlex.join([str(Path(sys.executable)), "-m", "scripts.run_reproduction",
                "summarize", "--output-root", str(output)])]
        job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
        summary_record.write_text(json.dumps({"job_id": job_id, "command": command}, indent=2) + "\n")
    return submissions


def retry_failed(paths: list[Path], output: Path, qos: str | None, partition: str | None = None,
                 gres: str = DEFAULT_GRES) -> dict[str, str]:
    """Queue failed command prefixes once per attempt; leave running jobs untouched."""
    record = output / "retry_submissions.json"
    retries = json.loads(record.read_text()) if record.exists() else {}
    for path in paths:
        group = path.parents[1]
        for job in json.loads(path.read_text())["jobs"]:
            name = job_name(job)
            status_path = group / "runner/status" / f"{name}.json"
            if not status_path.exists():
                continue
            status = json.loads(status_path.read_text())
            key = f"{group.name}/{name}/attempt{status['attempt']}"
            if status["status"] != "failed" or key in retries:
                continue
            cpus, memory_mb = job_resources(group.name, job["dataset"])
            command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []),
                *([f"--qos={qos}"] if qos else []), f"--gres={gres}",
                f"--cpus-per-task={cpus}", f"--mem={memory_mb}M", "--time=24:00:00", f"--chdir={ROOT}",
                "--job-name=hexels_retry", f"--output={output}/slurm/retry_%j.log",
                str(ROOT / "scripts/run_benchmark_array.sh"), str(path), str(group / "runner"), name]
            job_id = subprocess.check_output(command, text=True).strip().split(";")[0]
            retries[key] = job_id
            record.write_text(json.dumps(retries, indent=2) + "\n")
            LOGGER.info("Retry %s: %s", key, job_id)
    return retries


def summarize(output: Path) -> None:
    """Report completed and missing jobs explicitly, without fabricating partial averages."""
    records = [output / name for name in ("submission.json", "retry_submissions.json")]
    job_ids = [job_id for path in records if path.exists() for job_id in json.loads(path.read_text()).values()]
    if job_ids and shutil.which("sacct"):
        accounting = subprocess.check_output(["sacct", "-j", ",".join(job_ids), "--parsable2",
            "--format=JobIDRaw,State,ElapsedRaw,ExitCode,MaxRSS,AllocTRES,NodeList"], text=True)
        (output / "slurm_accounting.psv").write_text(accounting)
    rows = []
    for manifest_path in sorted(output.glob("*/manifests/jobs.json")):
        group = manifest_path.parents[1]
        for job in json.loads(manifest_path.read_text())["jobs"]:
            status_path = group / "runner/status" / f"{job_name(job)}.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending"}
            result_paths = job.get("results", [job.get("result")])
            for result_path in result_paths:
                result = json.loads(Path(result_path).read_text()) if result_path and Path(result_path).exists() else {}
                artifact = result.get("artifact", {})
                training = artifact.get("training", {})
                timings = status.get("command_timings", [])
                stage_seconds = {timing["stage"]: timing["seconds"] for timing in timings
                    if timing["returncode"] == 0 and "stage" in timing}
                row = {"group": job["group"], "dataset": job["dataset"], "scene": job["scene"],
                    "method": job["method"], "seed": job["seed"], "status": status["status"],
                    "result_complete": result.get("complete", False),
                    "iteration": artifact.get("iteration"), "primitives": artifact.get("primitives"),
                    "compact_bytes": artifact.get("compact_learned_bytes"),
                    "resident_bytes": artifact.get("resident_model_tensor_bytes"),
                    "train_seconds": training.get("optimization_seconds"),
                    "peak_train_allocated_gb": training.get("peak_gpu_allocated_gb"),
                    "peak_train_used_or_reserved_gb": training.get("peak_gpu_used_gb"),
                    "preprocessing_seconds": artifact.get("preprocessing", {}).get("seconds"),
                    "evaluation_seconds": result.get("evaluation_seconds"), "fps": result.get("benchmark", {}).get("fps"),
                    "peak_render_bytes": result.get("benchmark", {}).get("peak_allocated_bytes"),
                    "command_seconds": status.get("successful_command_seconds"),
                    "attempted_command_seconds": status.get("attempted_command_seconds"),
                    "mesh_extraction_seconds": stage_seconds.get("mesh_extraction"),
                    "geometry_evaluation_seconds": stage_seconds.get("geometry_evaluation"),
                    "hostname": status.get("hostname"), "gpu": result.get("hardware", {}).get("device_name"),
                    "result_path": result_path, "slurm_job_id": status.get("slurm", {}).get("SLURM_JOB_ID")}
                if row["train_seconds"] is not None and row["preprocessing_seconds"] is not None:
                    row["train_plus_preprocessing_seconds"] = row["train_seconds"] + row["preprocessing_seconds"]
                row.update(result.get("aggregate", {}))
                if job["group"] == "geometry":
                    model = Path(job["model_path"])
                    if job["dataset"] == "dtu" and (model / "dtu_eval/results.json").exists():
                        row.update(json.loads((model / "dtu_eval/results.json").read_text()))
                    path = model / "tnt_eval" / f"{job['scene']}.prf_tau_plotstr.txt"
                    if job["dataset"] == "tanks_and_temples" and path.exists():
                        precision, recall, fscore, threshold, stretch = map(float, path.read_text().split())
                        row.update(precision=precision, recall=recall, f1=fscore, threshold=threshold)
                rows.append(row)
    groups = {}
    for row in rows:
        endpoint = Path(row["result_path"]).parent.name if row["group"] == "mvs" else "30k"
        key = f"{row['group']}/{row['dataset']}/{row['method']}/{endpoint}"
        groups.setdefault(key, []).append(row)
    aggregates = {}
    for key, selected in groups.items():
        completed = [row for row in selected if row["status"] == "completed" and row["result_complete"]]
        entry = {"expected_scenes": len(selected), "completed_scenes": len(completed), "means": {}}
        if len(completed) == len(selected):
            for field in selected[0]:
                values = [row.get(field) for row in selected]
                if all(isinstance(value, (float, int)) and not isinstance(value, bool) for value in values):
                    entry["means"][field] = statistics.fmean(values)
        aggregates[key] = entry
    (output / "aggregate.json").write_text(json.dumps(aggregates, indent=2, allow_nan=False) + "\n")
    (output / "results.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d result rows", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic multi-dataset A5000 reproduction runner.")
    parser.add_argument("action", choices=("prepare", "submit", "retry", "summarize"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--smoke-iterations", type=int, default=0)
    parser.add_argument("--colmap", type=Path, default=ROOT / ".envs/colmap37/bin/colmap")
    parser.add_argument("--qos", help="Slurm QoS; omit to use the account default")
    parser.add_argument("--partition", help="Slurm partition; omit to use the scheduler default")
    parser.add_argument("--gres", default=DEFAULT_GRES, help="Slurm GPU request per task, e.g. gpu:a5000:1 (default: %(default)s)")
    parser.add_argument("--groups", nargs="*", choices=("nvs", "geometry_dtu", "geometry_tanks_and_temples", "mvs"))
    parser.add_argument("--require-completed", type=Path, help="Validation manifest that must have all tasks completed before submission.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = args.output_root.resolve()
    if args.action == "summarize":
        summarize(output)
        return
    if args.action == "prepare":
        config = load_config(args.config)
        paths = prepare(output, config, set(args.scenes), args.smoke_iterations)
        if not args.smoke_iterations:
            paths.append(prepare_mvs(output, config, set(args.scenes), args.colmap))
        _write_manifest(output / "manifest_index.json", {"manifests": [str(path) for path in paths]})
        for path in paths:
            LOGGER.info("Prepared %s", path)
    else:
        paths = [Path(path) for path in json.loads((output / "manifest_index.json").read_text())["manifests"]]
        if args.action == "retry":
            retry_failed(paths, output, args.qos, args.partition, args.gres)
        else:
            if args.require_completed is not None:
                require_completed(args.require_completed.resolve())
            submit(paths, output, args.qos, set(args.groups or []), args.partition, args.gres)


if __name__ == "__main__":
    main()
