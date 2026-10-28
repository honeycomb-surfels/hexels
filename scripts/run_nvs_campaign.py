"""Prepare and submit independently gated cohorts from the approved NVS matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.audit_nvs_campaign import audit
from scripts.nvs_campaign import CONFIG, ROOT, cameras, expand
from scripts.prepare_nvs_smoke import PYTHON, training_command, wrapped
from scripts.run_paper_3d_benchmark import _write_manifest
from scripts.run_reproduction import DEFAULT_GRES
from scripts.setup_nvs_baselines import environment_prefix
from scripts.measure_command import verify_provenance


def approved_methods(gates: list[Path], datasets: set[str]) -> set[str]:
    evidence = {}
    for root in gates:
        result = audit(root)
        for row in result["rows"]:
            evidence.setdefault(row["method"], []).append(row)
    return {method for method, rows in evidence.items()
            if all(row["complete"] for row in rows) and {row["dataset"] for row in rows} >= datasets}


def prepare(output: Path, config: dict, methods: list[str], gates: list[Path]) -> None:
    """Prepare native runs; capacity controls are prepared by scripts.prepare_nvs_budget_sweep."""
    if output.exists():
        raise FileExistsError(output)
    permitted = approved_methods(gates, set(config["datasets"]))
    if set(methods) - permitted:
        raise ValueError(f"Missing complete three-dataset runtime gates: {sorted(set(methods) - permitted)}")
    selected = [run for run in expand(config) if run["method"] in methods and not run["budget_anchor"]]
    if not selected:
        raise ValueError("No selected native experiments; prepare capacity controls with scripts.prepare_nvs_budget_sweep")
    _write_manifest(output / "protocol.json", config)
    _write_manifest(output / "gates.json", {str(root): audit(root) for root in gates})
    camera_jobs = []
    for dataset, scenes in config["datasets"].items():
        for scene in scenes:
            settings = output / "cameras" / dataset / f"{scene}_settings.json"
            contract = settings.with_name(f"{scene}_contract.json")
            _write_manifest(settings, cameras(config, dataset, scene))
            camera_jobs.append({"dataset": dataset, "scene": scene, "method": "cameras", "status": "ready",
                "cwd": str(ROOT), "commands": [[str(PYTHON), "-m", "scripts.evaluate_nvs_campaign", "cameras",
                "--settings", str(settings), "--output", str(contract), "--config", str(output / "protocol.json")]],
                "command_stages": ["canonical_cameras_and_trajectory"]})
    jobs = []
    for run in selected:
        directory = output / "runs" / run["id"]
        method = run["method"]
        iteration = 32000 if method == "bbsplat" else 18000 if method == "milo" and run["dataset"] != "dtu" else 30000
        job = {**run, "iteration": iteration, "smoke": False, "model_path": str(directory / "model"),
               "result": str(directory / "metrics.json"), "measurement": config["measurement"],
               "camera_settings": str(output / "cameras" / run["dataset"] / f"{run['scene']}_settings.json"),
               "camera_contract": str(output / "cameras" / run["dataset"] / f"{run['scene']}_contract.json")}
        if method == "hexels":
            job["bytes_per_primitive"] = {"v1": 80, "v2": 116, "v3": 260}[run["arm"].split("_")[1]]
        command, cwd = training_command(job, config, False)
        specification = directory / "job.json"
        _write_manifest(specification, job)
        interpreter = PYTHON if method in {"hexels", "2dgs"} else environment_prefix(method) / "bin/python"
        evaluate = [str(interpreter), "-m", "scripts.evaluate_nvs_campaign", "evaluate", "--job", str(specification)]
        jobs.append({"dataset": run["dataset"], "scene": run["scene"], "method": f"{run['arm']}_seed{run['seed']}",
                     "status": "ready", "cwd": str(ROOT), "result": job["result"],
                     "commands": [wrapped(command, cwd, directory / "resources_training.json", output / "provenance/provenance.json"),
                                  wrapped(evaluate, ROOT, directory / "resources_evaluation.json", output / "provenance/provenance.json")],
                     "command_stages": ["train", "evaluate_and_video"]})
    _write_manifest(output / "manifests/cameras.json", {"jobs": camera_jobs, "job_count": len(camera_jobs)})
    _write_manifest(output / "manifests/jobs.json", {"jobs": jobs, "job_count": len(jobs)})


def submit(output: Path, qos: str | None = None, partition: str | None = None, gres: str = DEFAULT_GRES) -> None:
    """Submit all tasks in a validated cohort without an artificial concurrency cap."""
    if list(output.glob("submission*.json")):
        raise FileExistsError(output / "submission.json")
    verify_provenance(output / "provenance/provenance.json")
    submissions = []
    for group in ("cameras", "jobs"):
        manifest = output / "manifests" / f"{group}.json"
        payload = json.loads(manifest.read_text())
        command = ["sbatch", "--parsable", *([f"--partition={partition}"] if partition else []), f"--gres={gres}",
                   *([f"--qos={qos}"] if qos else []),
                   f"--array=0-{payload['job_count'] - 1}", f"--job-name=nvs_{group}",
                   f"--output={output}/{group}_%A_%a.log"]
        if submissions:
            command += [f"--dependency=afterok:{submissions[0]['job_id']}", "--kill-on-invalid-dep=yes"]
        command += [str(ROOT / "scripts/run_benchmark_array.sh"), str(manifest), str(output / "runner" / group)]
        identifier = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
        submissions.append({"group": group, "job_id": identifier, "tasks": payload["job_count"], "command": command})
        _write_manifest(output / f"submission_{group}.json", submissions[-1])
    _write_manifest(output / "submission.json", {"jobs": submissions})


def capture(output: Path) -> None:
    """Archive all source, but gate workers on their actual runtime dependencies."""
    from scripts.capture_reproduction_provenance import capture as capture_source

    destination = output / "provenance"
    capture_source(destination)
    path = destination / "provenance.json"
    record = json.loads(path.read_text())
    runtime_scripts = {"scripts/measure_command.py", "scripts/evaluate_nvs_campaign.py",
        "scripts/evaluate_paper_nvs.py", "scripts/nvs_trajectory.py",
        "scripts/run_minimum_3d_benchmark.py", "scripts/run_benchmark_array.sh",
        "scripts/nvs_camera_validation.py", "scripts/nvs_budgets.py", "scripts/nvs_external.py",
        "scripts/nvs_object_trajectory.py", "scripts/setup_nvs_baselines.py"}
    entries = json.loads((output / "manifests/jobs.json").read_text())["jobs"]
    methods = {json.loads((Path(entry["result"]).parent / "job.json").read_text())["method"] for entry in entries}
    from scripts.nvs_external_provenance import capture as capture_external
    runtime_scripts.update(capture_external(destination, methods, record))
    record["runtime_paths"] = [relative for relative in record["source_sha256"]
        if relative in runtime_scripts or relative.startswith(("hexels/", "submodules/hexel-kernel/", "submodules/hexel-viewer/", "image-gs/", "2d-gaussian-splatting/"))]
    path.write_text(json.dumps(record, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "capture", "submit"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--methods", nargs="+", default=["hexels", "2dgs"])
    parser.add_argument("--gates", nargs="+", type=Path, default=[])
    parser.add_argument("--qos", help="Slurm QoS; omit to use the account default")
    parser.add_argument("--partition", help="Slurm partition; omit to use the scheduler default")
    parser.add_argument("--gres", default=DEFAULT_GRES, help="Slurm GPU request per task, e.g. gpu:a5000:1 (default: %(default)s)")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve(), json.loads(args.config.read_text()), args.methods,
                [root.resolve() for root in args.gates])
    elif args.action == "capture":
        capture(args.output.resolve())
    else:
        submit(args.output.resolve(), args.qos, args.partition, args.gres)


if __name__ == "__main__":
    main()
