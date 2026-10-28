from __future__ import annotations

import ast
import hashlib
import json
import struct
from pathlib import Path

import pytest

from scripts.run_reproduction import prepare, prepare_mvs, require_completed, retry_failed, submit, summarize
from scripts.run_paper_3d_benchmark import DEFAULT_CONFIG, ROOT, load_config
from scripts.prepare_minimum_3d_benchmark import image_holdout_split


def test_nvs_smoke_is_self_contained_and_shortens_checkpoints(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config["datasets"] = {"mipnerf360": {"root": str(tmp_path / "data"), "resolution": -1}}
    sparse = tmp_path / "data/bicycle/sparse/0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 100))
    paths = prepare(tmp_path / "run", config, {"bicycle"}, 2000)
    assert len(paths) == 1
    jobs = json.loads(paths[0].read_text())["jobs"]
    assert len(jobs) == 8
    for job in jobs:
        command = job["commands"][-1]
        assert "--camera-config" in command
        assert "--reference-hexel-path" not in command
        assert command[command.index("--iteration") + 1] == "2000"
        assert len(job["command_stages"]) == len(job["commands"])
        if job["method"].startswith("hexels_"):
            assert "--skip_postprocess" in job["commands"][3]
            assert "ckpt_000500.pt" in " ".join(job["commands"][1])
            assert job["command_stages"][:4] == ["train_500", "train_1000", "train_1500", "train_2000"]
        else:
            train = job["commands"][0]
            assert train[train.index("--port") + 1] == "0"


def test_submission_has_no_concurrency_cap(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "nvs/manifests/jobs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"jobs": [{}] * 8}))
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        return "12345\n"

    monkeypatch.setattr("subprocess.check_output", check_output)
    assert submit([path], tmp_path) == {"nvs": "12345"}
    assert "--array=0-7" in commands[0]
    assert "--gres=gpu:1" in commands[0]
    assert not any(argument.startswith(("--partition=", "--qos=")) for argument in commands[0])
    assert "--mem=32000M" in commands[0]
    assert submit([path], tmp_path) == {"nvs": "12345"}
    assert len(commands) == 2
    assert "--dependency=afterany:12345" in commands[1]


def test_slurm_partition_and_gpu_request_reach_every_submission(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_extraction_benchmark, run_nvs_campaign, run_surface_ablation

    reproduction = tmp_path / "reproduction/nvs/manifests/jobs.json"
    reproduction.parent.mkdir(parents=True)
    reproduction.write_text(json.dumps({"jobs": [{"dataset": "dtu", "scene": "scan24", "method": "hexels_v1"}]}))
    status = tmp_path / "reproduction/nvs/runner/status/dtu__scan24__hexels_v1.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "failed", "attempt": 1}))
    surface = tmp_path / "surface/objectives/manifests/jobs.json"
    surface.parent.mkdir(parents=True)
    surface.write_text(json.dumps({"jobs": [{"status": "ready"}]}))
    checkpoint = tmp_path / "ckpt_030000.pt"
    checkpoint.write_bytes(b"model")
    (tmp_path / "extraction").mkdir()
    (tmp_path / "extraction/manifest.json").write_text(json.dumps({"source_hashes": {}, "jobs": [{
        "dataset": "dtu", "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(b"model").hexdigest()}]}))
    campaign = tmp_path / "campaign"
    (campaign / "provenance").mkdir(parents=True)
    (campaign / "provenance/provenance.json").write_text(json.dumps({"source_sha256": {}, "extension_sha256": {}}))
    (campaign / "manifests").mkdir()
    for group in ("cameras", "jobs"):
        (campaign / "manifests" / f"{group}.json").write_text(json.dumps({"jobs": [{}], "job_count": 1}))
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        return "12345\n"

    monkeypatch.setattr("subprocess.check_output", check_output)
    slurm = {"partition": "gpu-a100", "gres": "gpu:a100:1"}
    submit([reproduction], tmp_path / "reproduction", **slurm)
    retry_failed([reproduction], tmp_path / "reproduction", None, **slurm)
    run_surface_ablation.submit(tmp_path / "surface", ["objectives"], "999", **slurm)
    run_extraction_benchmark.submit(tmp_path / "extraction", None, **slurm)
    run_nvs_campaign.submit(campaign, **slurm)
    assert len(commands) == 9
    gpu_tasks = 0
    for command in commands:
        gpu_task = any(argument.endswith("run_benchmark_array.sh") for argument in command)
        gpu_tasks += gpu_task
        assert "--partition=gpu-a100" in command
        assert [argument for argument in command if argument.startswith("--gres=")] == (
            ["--gres=gpu:a100:1"] if gpu_task else [])
        assert not any(argument.startswith("--qos=") for argument in command)
    assert gpu_tasks == 6


def test_mvs_preparation_requires_colmap_before_writing_jobs(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    config["datasets"] = {"mipnerf360": {"root": str(tmp_path / "data"), "resolution": -1}}
    with pytest.raises(ValueError, match="conda env create -p .envs/colmap37 -f environment-colmap.yml"):
        prepare_mvs(tmp_path / "run", config, {"bicycle"}, tmp_path / "missing/colmap")
    assert not (tmp_path / "run").exists()
    assert json.loads(prepare_mvs(tmp_path / "dtu_only", config, {"scan24"}, tmp_path / "missing/colmap").read_text())["jobs"] == []
    colmap = tmp_path / "bin/colmap"
    colmap.parent.mkdir()
    colmap.write_text("#!/bin/sh\n")
    colmap.chmod(0o755)
    job, = json.loads(prepare_mvs(tmp_path / "run", config, {"bicycle"}, colmap).read_text())["jobs"]
    assert job["commands"][0][job["commands"][0].index("--colmap") + 1] == str(colmap)


def test_nvs_evaluation_runs_as_a_module_from_the_repository_root(tmp_path: Path, monkeypatch) -> None:
    """External environments do not install this repository; `-m` from its root makes it importable."""
    from scripts import prepare_nvs_budget_sweep, prepare_nvs_smoke, run_nvs_campaign

    config = json.loads((ROOT / "configs/nvs.json").read_text())
    prepare_nvs_smoke.prepare(tmp_path / "smoke", config, ["garden"], ["hexels_v1_fixed", "3dgs_native"])
    jobs = json.loads((tmp_path / "smoke/manifests/smoke.json").read_text())["jobs"]
    config["datasets"], config["planned_training_runs"] = {"mipnerf360": ["garden"]}, {"total": 14}
    monkeypatch.setattr(run_nvs_campaign, "approved_methods", lambda gates, datasets: {"hexels", "2dgs", "3dgs"})
    run_nvs_campaign.prepare(tmp_path / "campaign", config, ["hexels", "3dgs"], [])
    campaign = json.loads((tmp_path / "campaign/manifests/jobs.json").read_text())["jobs"]
    assert len(campaign) == 7
    capacity = json.loads((ROOT / "configs/capacity.json").read_text())
    monkeypatch.setattr(prepare_nvs_budget_sweep, "inspect_input", lambda settings: {})
    monkeypatch.setattr(prepare_nvs_budget_sweep.nvs_budgets, "resolve",
                        lambda run, anchors: {"primitive_ceiling": 1000, "active_learned_byte_budget": 236000})
    prepare_nvs_budget_sweep.prepare(tmp_path / "sweep", capacity, [], False, [],
                                     identifiers={"mipnerf360/garden/seed0/3dgs_bytes_sparse_v1"})
    sweep = json.loads((tmp_path / "sweep/manifests/jobs.json").read_text())["jobs"]
    assert (len(jobs), len(sweep)) == (2, 1)
    for job in jobs + campaign + sweep:
        evaluate = job["commands"][1]
        assert evaluate[evaluate.index("--cwd") + 1] == str(ROOT)
        assert evaluate[evaluate.index("--") + 2:evaluate.index("--") + 5] == ["-m", "scripts.evaluate_nvs_campaign", "evaluate"]


def test_nvs_campaign_leaves_capacity_controls_to_the_budget_sweep(tmp_path: Path, monkeypatch) -> None:
    from scripts import run_nvs_campaign

    capacity = json.loads((ROOT / "configs/capacity.json").read_text())
    monkeypatch.setattr(run_nvs_campaign, "approved_methods", lambda gates, datasets: {"2dgs", "triangle"})
    with pytest.raises(ValueError, match="prepare_nvs_budget_sweep"):
        run_nvs_campaign.prepare(tmp_path / "capacity", capacity, ["2dgs", "triangle"], [])
    assert not (tmp_path / "capacity").exists()


def test_external_trainers_reject_seeds_they_cannot_honor() -> None:
    from scripts.nvs_external import training_command

    settings = {"source_path": "/data/mipnerf360/garden", "images": "images_4", "resolution": 1}
    job = {"method": "3dgs", "dataset": "mipnerf360", "scene": "garden", "model_path": "/runs/model",
           "iteration": 30000, "seed": 0}
    command, _ = training_command(job, settings, False)
    assert "--seed" not in command
    with pytest.raises(ValueError, match="seed 1"):
        training_command(job | {"seed": 1}, settings, False)


def test_baseline_builds_respect_user_cuda_architectures(tmp_path: Path, monkeypatch) -> None:
    from scripts import setup_nvs_baselines

    default: dict[str, str] = {}
    setup_nvs_baselines.cuda_architectures(default)
    assert default == {"TORCH_CUDA_ARCH_LIST": "8.6", "TCNN_CUDA_ARCHITECTURES": "86", "CMAKE_CUDA_ARCHITECTURES": "86"}
    derived = {"TORCH_CUDA_ARCH_LIST": "8.0;8.6+PTX"}
    setup_nvs_baselines.cuda_architectures(derived)
    assert derived["TCNN_CUDA_ARCHITECTURES"] == derived["CMAKE_CUDA_ARCHITECTURES"] == "80;86"
    explicit = {"TORCH_CUDA_ARCH_LIST": "8.9", "TCNN_CUDA_ARCHITECTURES": "89", "CMAKE_CUDA_ARCHITECTURES": "89"}
    setup_nvs_baselines.cuda_architectures(explicit)
    assert explicit == {"TORCH_CUDA_ARCH_LIST": "8.9", "TCNN_CUDA_ARCHITECTURES": "89", "CMAKE_CUDA_ARCHITECTURES": "89"}

    def no_commands(*args, **kwargs):
        raise AssertionError("install ran a command before checking CUDA_HOME")

    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setattr("subprocess.check_output", no_commands)
    monkeypatch.setattr("subprocess.run", no_commands)
    with pytest.raises(ValueError, match="CUDA_HOME"):
        setup_nvs_baselines.install("3dgs", tmp_path, "conda")


def test_hidden_resource_files_do_not_change_holdout(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for index in range(16):
        (images / f"{index:04d}.png").touch()
        (images / f"._{index:04d}.png").touch()
    result = image_holdout_split(tmp_path, 8)
    assert result["total_views"] == 16
    assert result["test_camera_names"] == ["0000", "0008"]


def test_new_shared_modules_do_not_import_scene_specific_scripts() -> None:
    for relative in ("hexels/benchmark_metrics.py", "scripts/evaluate_paper_nvs.py", "scripts/prepare_mvs_geometry.py"):
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "scripts"
                assert not (node.module or "").startswith("scripts.run_mipnerf")


def test_summary_keeps_missing_results_and_omits_partial_means(tmp_path: Path) -> None:
    manifest = tmp_path / "nvs/manifests/jobs.json"
    manifest.parent.mkdir(parents=True)
    jobs = [{"dataset": "dtu", "scene": scene, "method": "hexels_v1", "seed": 0,
        "group": "nvs", "result": str(tmp_path / f"{scene}.json")} for scene in ("scan24", "scan37")]
    manifest.write_text(json.dumps({"jobs": jobs}))
    summarize(tmp_path)
    rows = json.loads((tmp_path / "results.json").read_text())
    assert len(rows) == 2
    assert all(row["status"] == "pending" and not row["result_complete"] for row in rows)
    aggregate = json.loads((tmp_path / "aggregate.json").read_text())["nvs/dtu/hexels_v1/30k"]
    assert aggregate == {"expected_scenes": 2, "completed_scenes": 0, "means": {}}


def test_failed_attempt_is_queued_once(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "nvs/manifests/jobs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"jobs": [{"dataset": "dtu", "scene": "scan24", "method": "hexels_v1"}]}))
    status = tmp_path / "nvs/runner/status/dtu__scan24__hexels_v1.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "failed", "attempt": 1}))
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        return "12456\n"

    monkeypatch.setattr("subprocess.check_output", check_output)
    first = retry_failed([path], tmp_path, "low")
    assert retry_failed([path], tmp_path, "low") == first
    assert len(commands) == 1
    assert commands[0][-1] == "dtu__scan24__hexels_v1"


def test_geometry_submission_keeps_original_manifest_indices(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "geometry/manifests/jobs.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"jobs": [{"dataset": dataset} for dataset in ("dtu", "dtu", "tanks_and_temples")]}))
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        return "12345\n"

    monkeypatch.setattr("subprocess.check_output", check_output)
    submit([path], tmp_path, groups={"geometry_dtu"})
    assert "--array=0-1" in commands[0]
    assert "--export=ALL,HEXELS_WORKER_COUNT=3" in commands[0]
    assert "--mem=32000M" in commands[0]
    assert len(commands) == 2


def test_production_gate_requires_every_validation_task(tmp_path: Path) -> None:
    manifest = tmp_path / "geometry/manifests/jobs.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"jobs": [{"dataset": "dtu", "scene": "scan24", "method": "hexels_v1"}]}))
    with pytest.raises(ValueError, match="Validation is not complete"):
        require_completed(manifest)
    status = tmp_path / "geometry/runner/status/dtu__scan24__hexels_v1.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"status": "completed"}))
    require_completed(manifest)
