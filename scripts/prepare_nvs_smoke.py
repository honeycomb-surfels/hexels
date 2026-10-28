"""Prepare scene-independent NVS end-to-end gates in a fresh output root."""

from __future__ import annotations

import sys

import argparse
import json
from pathlib import Path

from scripts.nvs_campaign import CONFIG, ROOT, cameras
from scripts.measure_command import DEFAULT_SAMPLE_INTERVAL_SECONDS
from scripts.prepare_minimum_3d_benchmark import SceneSpec, _hexels_stage_commands, _two_dgs_commands
from scripts.run_paper_3d_benchmark import _write_manifest
from scripts.run_reproduction import set_option
from scripts.setup_nvs_baselines import environment_prefix


PYTHON = Path(sys.executable)


def training_command(job: dict, config: dict, smoke: bool) -> tuple[list[str], Path]:
    """Reuse the accepted training schedule, never cap a native baseline implicitly."""
    settings = cameras(config, job["dataset"], job["scene"])
    scene = SceneSpec(job["dataset"], job["scene"], Path(settings["source_path"]), "nvs",
                      settings["images"], True, settings["resolution"])
    if job["method"] == "2dgs":
        command = _two_dgs_commands(scene, method="2dgs_native", output_root=Path(job["model_path"]).parent,
            primitive_cap=None, port=0, dtu_official_root=None, dtu_dataset_root=ROOT / "data/dtu",
            python_bin=PYTHON, seed=job["seed"])[0]
        for name, value in {"-m": job["model_path"], "--iterations": job["iteration"],
                            "--save_iterations": job["iteration"], "--data_device": "cuda"}.items():
            set_option(command, name, value)
        return command, ROOT / "2d-gaussian-splatting"
    if job["method"] != "hexels":
        from scripts.nvs_external import training_command as external_command
        return external_command(job, settings, smoke)
    variant = job["arm"].split("_")[1]
    command = _hexels_stage_commands(scene, variant=variant, output_root=Path(job["model_path"]).parent,
                                     python_bin=PYTHON, seed=job["seed"])[0]
    values = {"model_path": job["model_path"], "iterations": job["iteration"],
              "save_every": job["iteration"], "render_every": 0, "lr_decay_factor": 0.5}
    for name, value in values.items():
        set_option(command, f"--{name}", value)
    command += ["--lr_decay_milestones", *map(str, [101, 141, 171] if smoke else [15001, 20001, 25001])]
    recipe = json.loads((ROOT / config["hexels"]["recipe"]).read_text())
    options = recipe["common_options"].copy()
    options["densify_cap_max"] = config["hexels"]["population_ceiling"]
    if smoke:
        options.update(densify_from_iter=20, densify_until_iter=100, densification_interval=20,
                       densify_min_denom=1, densify_newborn_grace=20, densify_prune_interval=40,
                       densify_grad_threshold=0.00001, densify_cap_max=200000)
    if job["arm"].endswith("mixed_weak500"):
        options["densify_prune_mean_threshold"] = 1e-7
        command.extend(["--enable_densify", "--densify_coordinated_prune"])
    for name, value in options.items():
        set_option(command, f"--{name}", value)
    return command, ROOT


def wrapped(command: list[str], cwd: Path, output: Path, provenance: Path | None = None,
            *, training_method: str | None = None,
            interval: float = DEFAULT_SAMPLE_INTERVAL_SECONDS) -> list[str]:
    options = ["--provenance", str(provenance)] if provenance else []
    if training_method is not None:
        options.extend(["--training-method", training_method])
    return [str(PYTHON), str(ROOT / "scripts/measure_command.py"), "--output", str(output),
            "--cwd", str(cwd), "--interval", str(interval), *options, "--", *command]


def prepare(output: Path, config: dict, scenes: list[str], methods: list[str],
            phase_complete: bool = False) -> None:
    if output.exists():
        raise FileExistsError(output)
    _write_manifest(output / "protocol.json", config | {"smoke": True})
    selected = [(dataset, scene) for dataset, names in config["datasets"].items()
                for scene in names if scene in scenes]
    if {scene for _, scene in selected} != set(scenes):
        raise ValueError("Unknown smoke scene")
    camera_jobs, jobs = [], []
    for dataset, scene in selected:
        settings = output / "cameras" / dataset / f"{scene}_settings.json"
        contract = settings.with_name(f"{scene}_contract.json")
        _write_manifest(settings, cameras(config, dataset, scene))
        camera_jobs.append({"dataset": dataset, "scene": scene, "method": "cameras", "status": "ready",
            "cwd": str(ROOT), "commands": [[str(PYTHON), "-m", "scripts.evaluate_nvs_campaign", "cameras",
                "--settings", str(settings), "--output", str(contract), "--config", str(output / "protocol.json")]],
            "command_stages": ["canonical_cameras_and_trajectory"]})
        for arm in methods:
            method = "hexels" if arm.startswith("hexels_") else arm.removesuffix("_native")
            identifier = f"{dataset}/{scene}/seed0/{arm}"
            run = output / "runs" / identifier
            iteration = 200 if method == "hexels" else 3200
            if phase_complete and method in {"mesh_splatting", "milo"}:
                iteration = 18000 if method == "milo" and dataset != "dtu" else 30000
            job = {"id": identifier, "method": method, "arm": arm, "dataset": dataset,
                   "scene": scene, "seed": 0, "iteration": iteration, "smoke": True,
                   "model_path": str(run / "model"), "result": str(run / "metrics.json"),
                   "camera_settings": str(settings), "camera_contract": str(contract),
                   "measurement": config["measurement"]}
            if method == "milo" and dataset != "dtu":
                activation = run / "activation_config"
                activation.parent.mkdir(parents=True, exist_ok=True)
                native = (ROOT / "MILo/milo/configs/fast").read_text().strip()
                activation.write_text(native.replace("iterations=18_000", f"iterations={iteration}") + "\n")
                job["activation_config"] = str(activation)
            job["phase_complete_gate"] = phase_complete
            if method == "hexels":
                job["bytes_per_primitive"] = {"v1": 80, "v2": 116, "v3": 260}[arm.split("_")[1]]
            command, cwd = training_command(job, config, True)
            job_path = run / "job.json"
            _write_manifest(job_path, job)
            interpreter = str(PYTHON) if method in {"hexels", "2dgs"} else str(environment_prefix(method) / "bin/python")
            evaluate = [interpreter, "-m", "scripts.evaluate_nvs_campaign", "evaluate", "--job", str(job_path)]
            jobs.append({"dataset": dataset, "scene": scene, "method": arm, "status": "ready", "cwd": str(ROOT),
                         "commands": [wrapped(command, cwd, run / "resources_training.json", training_method=method,
                                              interval=config["measurement"]["resource_sample_seconds"]),
                                      wrapped(evaluate, ROOT, run / "resources_evaluation.json",
                                              interval=config["measurement"].get("evaluation_resource_sample_seconds", 0.2))],
                         "command_stages": ["train", "evaluate_and_video"], "result": job["result"]})
    _write_manifest(output / "manifests/cameras.json", {"jobs": camera_jobs, "job_count": len(camera_jobs)})
    _write_manifest(output / "manifests/smoke.json", {"jobs": jobs, "job_count": len(jobs)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["garden", "scan65", "Truck"])
    parser.add_argument("--methods", nargs="+", default=["hexels_v1_fixed", "hexels_v2_mixed_weak500", "hexels_v3_mixed_weak500", "2dgs_native"])
    parser.add_argument("--phase-complete", action="store_true")
    args = parser.parse_args()
    prepare(args.output.resolve(), json.loads(args.config.read_text()), args.scenes, args.methods, args.phase_complete)


if __name__ == "__main__":
    main()
