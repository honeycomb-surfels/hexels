from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import pytest

from scripts.prepare_minimum_3d_benchmark import (
    DTU,
    MIPNERF360_INDOOR,
    MIPNERF360_OUTDOOR,
    SceneSpec,
    _hexels_stage_commands,
    hexel_training_defaults,
)
from scripts.run_paper_3d_benchmark import (
    DEFAULT_CONFIG,
    PAPER_METHODS,
    ROOT,
    _write_manifest,
    build_evaluation_manifest,
    geometry_data_paths,
    load_config,
    prepare_training_manifest,
    summarize_suite,
)


def _config(tmp_path: Path) -> dict[str, object]:
    config = load_config(DEFAULT_CONFIG)
    config["python"] = sys.executable
    config["datasets"]["tanks_and_temples"]["evaluation_python"] = sys.executable
    config["seed"] = 17
    scenes_by_dataset = {
        "mipnerf360": (*MIPNERF360_OUTDOOR, *MIPNERF360_INDOOR),
        "tanks_and_temples": config["datasets"]["tanks_and_temples"]["scenes"],
        "dtu": DTU,
    }
    for dataset, scenes in scenes_by_dataset.items():
        root = tmp_path / dataset
        config["datasets"][dataset]["root"] = str(root)
        for scene in scenes:
            sparse = root / scene / "sparse/0"
            sparse.mkdir(parents=True)
            (sparse / "points3D.bin").write_bytes(struct.pack("<Q", 100))
    for dataset in ("dtu", "tanks_and_temples"):
        official_root = tmp_path / f"official_{dataset}"
        config["datasets"][dataset]["official_root"] = str(official_root)
        for scene in scenes_by_dataset[dataset]:
            for path in geometry_data_paths(dataset, scene, tmp_path / dataset / scene, official_root):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.name == "mask":
                    path.mkdir()
                else:
                    path.write_bytes(b"fixture")
    return config


def _prepare(tmp_path: Path, config: dict[str, object], **kwargs: object) -> dict[str, object]:
    return prepare_training_manifest(
        output_root=tmp_path / "outputs/training",
        manifest_path=tmp_path / "outputs/manifests/train.json",
        datasets=set(config["datasets"]),
        config=config,
        **kwargs,
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_full_suite_has_30_scenes_and_separate_nvs_evaluation(tmp_path: Path) -> None:
    training = _prepare(tmp_path, _config(tmp_path))
    evaluation = build_evaluation_manifest(training, output_root=tmp_path / "evaluation")

    assert training["scene_count"] == 30
    assert training["job_count"] == 30 * len(PAPER_METHODS)
    assert evaluation["scene_count"] == 9
    assert evaluation["job_count"] == 9 * len(PAPER_METHODS)
    assert set(evaluation["datasets"]) == {"mipnerf360"}
    assert "Church" not in training["datasets"]["tanks_and_temples"]
    assert "deep_blending" not in training["datasets"]
    assert all(job["status"] == "ready" for job in training["jobs"])
    assert all(job["commands"][0][0] == sys.executable for job in evaluation["jobs"])
    assert training["kernel_release_manifest_sha256"] == hashlib.sha256(
        (ROOT / "submodules/hexel-kernel/RELEASE_MANIFEST.json").read_bytes()
    ).hexdigest()


def test_shared_dataset_defaults_and_interpreter_reach_every_stage(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["datasets"]["tanks_and_temples"]["hexels"] = {"init_knn_scale_multiplier": 0.3}
    training = _prepare(tmp_path, config)
    for job in training["jobs"]:
        assert all(command[0] == sys.executable for command in job["commands"])
        assert _option(job["commands"][0], "--seed") == "17"
        if job["dataset"] == "tanks_and_temples":
            assert _option(job["commands"][0], "--resolution") == "2"
            if job["method"].startswith("hexels_"):
                for command in job["commands"][:4]:
                    assert _option(command, "--init_knn_scale_multiplier") == "0.3"
        if job["dataset"] == "mipnerf360":
            images = "images_4" if job["scene"] in MIPNERF360_OUTDOOR else "images_2"
            assert _option(job["commands"][0], "--images") == images
        if job["dataset"] == "dtu":
            assert job["task"] == "geometry"
            assert "--eval" not in job["commands"][0]
            assert "--DTU" in job["commands"][-1]


def test_dtu_only_does_not_create_nvs_jobs(tmp_path: Path) -> None:
    training = _prepare(tmp_path, _config(tmp_path), scenes={"scan24"})
    evaluation = build_evaluation_manifest(training, output_root=tmp_path / "evaluation")
    assert training["scene_count"] == 1
    assert evaluation["jobs"] == []
    assert evaluation["scene_count"] == 0


def test_tnt_uses_all_views_and_original_geometry_evaluator(tmp_path: Path) -> None:
    training = _prepare(tmp_path, _config(tmp_path), scenes={"Barn", "Courthouse"})
    assert build_evaluation_manifest(training, output_root=tmp_path / "evaluation")["jobs"] == []
    for job in training["jobs"]:
        assert job["task"] == "geometry"
        assert all("--eval" not in command for command in job["commands"])
        mesh, evaluate = job["commands"][-2:]
        if job["method"].startswith("2dgs_"):
            assert "--skip_train" in mesh and "--skip_test" in mesh
        else:
            assert _option(mesh, "--depth_mode") == "expected"
        large = job["scene"] == "Courthouse"
        assert _option(mesh, "--voxel_size") == ("0.006" if large else "0.004")
        assert _option(mesh, "--sdf_trunc") == ("0.024" if large else "0.016")
        assert _option(mesh, "--depth_trunc") == ("4.5" if large else "3.0")
        assert evaluate[1].endswith("scripts/eval_tnt/run.py")
        assert _option(evaluate, "--dataset-dir").endswith(f"official_tanks_and_temples/{job['scene']}")
        assert _option(evaluate, "--traj-path") == str(Path(job["source"]) / f"{job['scene']}_COLMAP_SfM.log")
        assert _option(evaluate, "--ply-path").endswith("train/ours_30000/fuse_post.ply")
        assert "--DTU" not in evaluate


def test_tnt_uses_separate_evaluation_interpreter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = tmp_path / "env/tnt/bin/python"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("#!/bin/sh\n")
    evaluator.chmod(0o755)
    config["datasets"]["tanks_and_temples"]["evaluation_python"] = str(evaluator)
    training = _prepare(tmp_path, config, scenes={"Barn"})
    for job in training["jobs"]:
        assert all(command[0] == sys.executable for command in job["commands"][:-1])
        assert job["commands"][-1][0] == str(evaluator)


def test_tnt_geometry_requires_the_pinned_evaluator_environment(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["datasets"]["tanks_and_temples"]["evaluation_python"] = str(tmp_path / "missing/bin/python")
    with pytest.raises(ValueError, match="conda env create -p .envs/tnt-eval -f environment-tnt-eval.yml"):
        _prepare(tmp_path, config, scenes={"Barn"})
    assert not (tmp_path / "outputs/manifests/train.json").exists()
    assert _prepare(tmp_path, config, scenes={"scan24"})["job_count"] == len(PAPER_METHODS)


def test_staged_protocol_keeps_expected_depth_and_surface_arms_stay_paired(tmp_path: Path) -> None:
    from scripts.run_surface_ablation import prepare as prepare_surface

    protocol = json.loads((ROOT / "configs/surface.json").read_text())
    protocol.update(scenes=["scan24"], baseline_root=str(tmp_path / "baseline"))
    checkpoint = tmp_path / "baseline/geometry/training/hexels_v3/dtu/scan24/30k/checkpoints/ckpt_030000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    paths = prepare_surface(tmp_path / "surface", _config(tmp_path), protocol, smoke=False)
    modes = {}
    for job in (job for path in paths for job in json.loads(path.read_text())["jobs"]):
        extraction = job["commands"][0 if job["stage"] == "extraction" else 4]
        assert extraction.count("--depth_mode") == 1
        modes[job["arm"]] = _option(extraction, "--depth_mode")
    assert modes == {"expected": "expected", "median": "median", "baseline": "expected",
                     "normal": "expected", "distortion": "expected", "normal_distortion": "expected"}


def test_shared_hexel_training_settings_come_only_from_the_train_config(tmp_path: Path) -> None:
    shared = json.loads(DEFAULT_CONFIG.read_text())["hexels"]
    assert hexel_training_defaults() == shared
    scene = SceneSpec("dtu", "scan24", tmp_path, "nvs", None, True, 2)
    command = _hexels_stage_commands(scene, variant="v1", output_root=tmp_path)[0]
    assert all(_option(command, f"--{name}") == str(value) for name, value in shared.items())
    config = json.loads(DEFAULT_CONFIG.read_text())
    del config["hexels"]["lr_means3d_final"]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="lr_means3d_final"):
        hexel_training_defaults(path)


def test_default_tanks_and_temples_root_matches_the_dataset_guide(tmp_path: Path) -> None:
    training = prepare_training_manifest(
        output_root=tmp_path / "training", manifest_path=tmp_path / "train.json",
        datasets={"tanks_and_temples"}, scenes={"Barn"},
    )
    assert {job["source"] for job in training["jobs"]} == {
        str((ROOT / "data/tanks_and_temples").resolve() / "Barn")
    }


@pytest.mark.parametrize("suffix", [".ply", ".json", "_trans.txt", "_COLMAP_SfM.log"])
def test_tnt_missing_reference_blocks_geometry(tmp_path: Path, suffix: str) -> None:
    config = _config(tmp_path)
    missing = Path(config["datasets"]["tanks_and_temples"]["official_root"]) / "Barn" / f"Barn{suffix}"
    missing.unlink()
    training = _prepare(tmp_path, config, scenes={"Barn"})
    assert all(job["status"] == "blocked_missing_geometry_data" for job in training["jobs"])
    assert all(str(missing) in job["data_error"] for job in training["jobs"])


def test_tnt_summary_requires_all_results_and_reports_percentages(tmp_path: Path) -> None:
    training = _prepare(tmp_path, _config(tmp_path), scenes={"Barn"})
    evaluation = build_evaluation_manifest(training, output_root=tmp_path / "evaluation")
    for job in training["jobs"][:-1]:
        path = Path(_option(job["commands"][-1], "--out-dir")) / "Barn.prf_tau_plotstr.txt"
        path.parent.mkdir(parents=True)
        path.write_text("0.6 0.6 0.6 0.01 5")
    with pytest.raises(FileNotFoundError):
        summarize_suite(training, evaluation, tmp_path / "outputs")
    path = Path(_option(training["jobs"][-1]["commands"][-1], "--out-dir")) / "Barn.prf_tau_plotstr.txt"
    path.parent.mkdir(parents=True)
    path.write_text("0.8 0.8 0.8 0.01 5")
    summary = summarize_suite(training, evaluation, tmp_path / "outputs")
    assert set(summary) == {"protocol", "tnt_geometry"}
    assert summary["tnt_geometry"]["aggregates"]["hexels_v1"]["f1_percent"] == 60.0
    assert summary["tnt_geometry"]["aggregates"][PAPER_METHODS[-1]]["f1_percent"] == 80.0


def test_missing_sparse_data_keeps_blocked_evaluation_rows(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["datasets"]["mipnerf360"]["root"] = str(tmp_path / "missing")
    training = _prepare(tmp_path, config, scenes={"bicycle"})
    evaluation = build_evaluation_manifest(training, output_root=tmp_path / "evaluation")
    assert evaluation["job_count"] == len(PAPER_METHODS)
    assert all(job["status"] != "ready" for job in evaluation["jobs"])
    assert any(not job["commands"] for job in evaluation["jobs"])
    with pytest.raises(ValueError, match="blocked training"):
        summarize_suite(training, evaluation, tmp_path)


def test_missing_official_dtu_data_blocks_geometry_jobs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["datasets"]["dtu"]["official_root"] = str(tmp_path / "missing")
    training = _prepare(tmp_path, config)
    assert all(
        job["status"] == "blocked_missing_dtu_official_root"
        for job in training["jobs"]
        if job["dataset"] == "dtu"
    )


def test_unknown_scene_fails_instead_of_producing_an_empty_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown scenes"):
        _prepare(tmp_path, _config(tmp_path), scenes={"typo"})


@pytest.mark.parametrize("setting", ["lr_typo", "enable_densify", "max_points"])
def test_config_rejects_typos_and_population_changes(tmp_path: Path, setting: str) -> None:
    config = json.loads(DEFAULT_CONFIG.read_text())
    config["hexels"][setting] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unknown shared Hexel"):
        load_config(path)


def test_manifest_is_stable_and_refuses_protocol_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = _prepare(tmp_path, config)
    assert _prepare(tmp_path, config) == first
    config["seed"] = 18
    with pytest.raises(ValueError, match="fresh output root"):
        _prepare(tmp_path, config)


def test_preparation_interpreter_does_not_change_runtime_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    first = _prepare(tmp_path, config)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert _prepare(tmp_path, config) == first
    assert "open3d_available" not in first["runtime_preflight"]


def test_suite_requires_every_selected_nvs_result(tmp_path: Path) -> None:
    training = _prepare(tmp_path, _config(tmp_path), scenes={"bicycle"})
    evaluation = build_evaluation_manifest(training, output_root=tmp_path / "evaluation")
    with pytest.raises(FileNotFoundError):
        summarize_suite(training, evaluation, tmp_path)
    assert not (tmp_path / "summary.json").exists()


def test_suite_aggregates_only_selected_dtu_scans(tmp_path: Path) -> None:
    training = _prepare(tmp_path, _config(tmp_path), scenes={"scan24"})
    for method in PAPER_METHODS:
        model = tmp_path / "outputs/training" / method / "dtu/scan24"
        if method.startswith("hexels_"):
            for stage in (15, 20, 25, 30):
                _write_manifest(model / f"{stage}k/summary.json", {
                    "num_primitives": 100, "model_state_bytes": 20000, "train_seconds": 1.0,
                })
            model = model / "30k"
        else:
            _write_manifest(model / "training_telemetry.json", {
                "final_points": 100, "model_state_bytes": 23200, "wall_time_seconds": 4.0,
            })
        _write_manifest(model / "dtu_eval/results.json", {
            "mean_d2s": 0.5, "mean_s2d": 0.7, "overall": 0.6,
        })
    evaluation = build_evaluation_manifest(training, output_root=tmp_path / "evaluation")
    summary = summarize_suite(training, evaluation, tmp_path / "outputs")
    assert "nvs" not in summary
    assert summary["dtu_geometry"]["scene_count"] == 1
    assert summary["dtu_geometry"]["aggregates"]["hexels_v2"]["surface_overall"] == 0.6


def test_slurm_worker_uses_assigned_gpu_not_array_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_minimum_3d_benchmark as runner

    calls = []

    def worker(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(runner, "run_worker", worker)
    monkeypatch.setattr(sys, "argv", [
        "runner", "--worker-index", "7", "--worker-count", "12", "--gpu", "GPU-assigned",
    ])
    with pytest.raises(SystemExit) as result:
        runner.main()
    assert result.value.code == 0
    assert calls[0]["worker_index"] == 7
    assert calls[0]["worker_count"] == 12
    assert calls[0]["gpu"] == "GPU-assigned"
