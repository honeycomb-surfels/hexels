from __future__ import annotations

import sys

import argparse
import hashlib
import json
import logging
import math
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scripts.prepare_minimum_3d_benchmark import (
    HEXELS_COMPACT_BYTES_PER_PRIMITIVE,
    HEXEL_TRAINING_KEYS,
    DTU,
    DEEP_BLENDING,
    MIPNERF360_INDOOR,
    MIPNERF360_OUTDOOR,
    TANKS_AND_TEMPLES,
    ROOT,
    TWO_DGS_ROOT,
    _filter_manifest,
    build_manifest,
)
from scripts.run_minimum_3d_benchmark import run_worker


DEFAULT_OUTPUT_ROOT = ROOT / "benchmark_outputs/paper_3d"
DEFAULT_CONFIG = ROOT / "configs/train.json"
DATASET_SCENES = {
    "tanks_and_temples": TANKS_AND_TEMPLES,
    "mipnerf360": (*MIPNERF360_OUTDOOR, *MIPNERF360_INDOOR),
    "dtu": DTU,
    "deep_blending": DEEP_BLENDING,
}
PAPER_METHODS = (
    "hexels_v1",
    "hexels_v2",
    "hexels_v3",
    "2dgs_native",
    "2dgs_equal_primitives",
    "2dgs_equal_v1_compact_memory",
    "2dgs_equal_v2_compact_memory",
    "2dgs_equal_v3_compact_memory",
)
LOGGER = logging.getLogger("paper_3d_benchmark")
TNT_EVALUATION_REMEDY = (
    "Create it from the repository root with `conda env create -p .envs/tnt-eval -f environment-tnt-eval.yml`, "
    "or set tanks_and_temples.evaluation_python in the config to an existing interpreter."
)


def require_executable(path: str | Path, purpose: str, remedy: str) -> None:
    """Fail at preparation, not after hours of training, when a pinned tool is missing."""
    if not (Path(path).is_file() and os.access(path, os.X_OK)):
        raise ValueError(f"{purpose} needs {path}, which is missing or not executable. {remedy}")


def load_config(path: Path) -> dict[str, Any]:
    """Load shared dataset settings; paths are relative to the repository root."""
    config = json.loads(path.read_text())
    unknown = config.keys() - {"python", "seed", "hexels", "datasets"}
    if unknown:
        raise ValueError(f"Unknown benchmark settings: {sorted(unknown)}")
    if not config["datasets"] or config["datasets"].keys() - DATASET_SCENES.keys():
        raise ValueError(f"Config datasets must be a nonempty subset of {sorted(DATASET_SCENES)}")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    for dataset, settings in config["datasets"].items():
        unknown = settings.keys() - {
            "root", "official_root", "resolution", "hexels", "scenes", "task", "evaluation_python",
        }
        if unknown:
            raise ValueError(f"Unknown {dataset} settings: {sorted(unknown)}")
        if "scenes" in settings and (
            not settings["scenes"] or set(settings["scenes"]) - set(DATASET_SCENES[dataset])
        ):
            raise ValueError(f"Invalid scene selection for {dataset}: {settings['scenes']}")
        if settings.get("task", "nvs") not in {"nvs", "geometry"}:
            raise ValueError(f"Invalid task for {dataset}: {settings['task']}")
        if settings.get("task") == "geometry" and dataset not in {"dtu", "tanks_and_temples"}:
            raise ValueError(f"No geometry evaluation is configured for {dataset}")
        if "evaluation_python" in settings:
            if dataset != "tanks_and_temples":
                raise ValueError("evaluation_python is only supported for the legacy T&T evaluator")
            settings["evaluation_python"] = str(
                (ROOT / Path(settings["evaluation_python"]).expanduser()).resolve()
            )
        resolution = settings["resolution"]
        if resolution not in {-1, 1, 2, 4, 8}:
            raise ValueError(f"Unsupported resolution for {dataset}: {resolution}")
        training = config.get("hexels", {}) | settings.get("hexels", {})
        unknown = training.keys() - HEXEL_TRAINING_KEYS
        if unknown:
            raise ValueError(f"Unknown shared Hexel training settings: {sorted(unknown)}")
        settings["root"] = str((ROOT / Path(settings["root"]).expanduser()).resolve())
        if settings.get("official_root") is not None:
            settings["official_root"] = str(
                (ROOT / Path(settings["official_root"]).expanduser()).resolve()
            )
    config["python"] = str((ROOT / Path(config["python"]).expanduser()).resolve())
    return config


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"Manifest changed: {path}. Use a fresh output root for a new protocol.")
    _write_json(path, manifest)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def geometry_data_paths(dataset: str, scene: str, source: Path, official: Path) -> list[Path]:
    """List required inputs to the pinned geometry evaluators, not just root directories."""
    if dataset == "tanks_and_temples":
        return [source / f"{scene}_COLMAP_SfM.log"] + [
            official / scene / f"{scene}{suffix}"
            for suffix in (".ply", ".json", "_trans.txt", "_COLMAP_SfM.log")
        ]
    scan = int(scene.removeprefix("scan"))
    return [
        source / "cameras.npz", source / "mask",
        official / f"ObsMask/ObsMask{scan}_10.mat",
        official / f"ObsMask/Plane{scan}.mat",
        official / f"Points/stl/stl{scan:03}_total.ply",
    ]


def prepare_training_manifest(
    *,
    output_root: Path,
    manifest_path: Path,
    datasets: set[str],
    scenes: set[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = config["datasets"] if config is not None else {}
    if config is not None and datasets - settings.keys():
        raise ValueError(f"Datasets missing from config: {sorted(datasets - settings.keys())}")
    official_root = settings.get("dtu", {}).get("official_root")
    roots = {
        "mipnerf360": ROOT / "data/mipnerf360",
        "dtu": ROOT / "data/dtu",
        "deep_blending": ROOT / "data/deep_blending",
        "tanks_and_temples": ROOT / "data/tanks_and_temples",
    }
    roots = {
        dataset: Path(settings.get(dataset, {}).get("root", default)).resolve()
        for dataset, default in roots.items()
    }
    python_bin = Path(config["python"]) if config is not None else Path(sys.executable)
    manifest = build_manifest(
        mipnerf360_root=roots["mipnerf360"],
        dtu_root=roots["dtu"],
        deep_blending_root=roots["deep_blending"],
        tanks_and_temples_root=roots["tanks_and_temples"],
        output_root=output_root.resolve(),
        dtu_official_root=Path(official_root) if official_root is not None else None,
        dtu_nvs=settings.get("dtu", {}).get("task") == "nvs",
        python_bin=python_bin,
        seed=config["seed"] if config is not None else 0,
        dataset_defaults=settings,
        hexel_defaults=config.get("hexels", {}) if config is not None else None,
    )
    manifest = _filter_manifest(
        manifest,
        datasets=datasets,
        scenes=scenes or set(),
        methods=set(PAPER_METHODS),
    )
    selected_scenes = {job["scene"] for job in manifest["jobs"]}
    if not manifest["jobs"] or (scenes and scenes - selected_scenes):
        raise ValueError("Scene selection is empty or contains unknown scenes for these datasets")
    manifest["protocol"] = (
        "hexels_shared_dataset_defaults_v2" if config is not None else "hexels_paper_3d_nvs_v1"
    )
    if config is not None:
        manifest["config"] = config
        del manifest["runtime_preflight"]["open3d_available"]
        manifest["kernel_release_manifest_sha256"] = hashlib.sha256(
            (ROOT / "submodules/hexel-kernel/RELEASE_MANIFEST.json").read_bytes()
        ).hexdigest()
    for job in manifest["jobs"]:
        if job["task"] != "geometry":
            continue
        dataset = job["dataset"]
        evaluation_root = settings.get(dataset, {}).get("official_root")
        if evaluation_root is None or not Path(evaluation_root).is_dir():
            job["status"] = f"blocked_missing_{dataset}_official_root"
            job["data_error"] = f"{dataset} official evaluation root is missing: {evaluation_root}"
        else:
            missing = [str(path) for path in geometry_data_paths(
                dataset, job["scene"], Path(job["source"]), Path(evaluation_root),
            ) if not path.exists()]
            if missing:
                job["status"] = "blocked_missing_geometry_data"
                job["data_error"] = f"Missing geometry inputs: {', '.join(missing)}"
    tnt_evaluator = str(TWO_DGS_ROOT / "scripts/eval_tnt/run.py")
    for interpreter in sorted({command[0] for job in manifest["jobs"] for command in job["commands"]
                               if command[1:2] == [tnt_evaluator]}):
        require_executable(interpreter, "T&T geometry evaluation", TNT_EVALUATION_REMEDY)
    _write_manifest(manifest_path, manifest)
    return manifest


def _model_path(job: dict[str, Any]) -> Path:
    command = job["commands"][3] if job["method"].startswith("hexels_") else job["commands"][0]
    option = "--model_path" if job["method"].startswith("hexels_") else "-m"
    return Path(_option(command, option)).resolve()


def build_evaluation_manifest(
    training_manifest: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    jobs_by_key = {
        (job["dataset"], job["scene"], job["method"]): job
        for job in training_manifest["jobs"]
    }
    evaluation_jobs = []
    for job in training_manifest["jobs"]:
        if job.get("task", "nvs") != "nvs":
            continue
        dataset = str(job["dataset"])
        scene = str(job["scene"])
        method = str(job["method"])
        evaluation_job = {
            "dataset": dataset,
            "scene": scene,
            "task": "nvs_evaluation",
            "method": method,
            "status": job.get("status", "ready"),
            "cwd": str(ROOT),
            "commands": [],
        }
        evaluation_jobs.append(evaluation_job)
        if not job["commands"]:
            continue
        model_path = _model_path(job)
        reference_hexel = _model_path(jobs_by_key[(dataset, scene, "hexels_v2")])
        reference_2dgs = _model_path(jobs_by_key[(dataset, scene, "2dgs_native")])
        command = [
            job["commands"][0][0],
            str(ROOT / "scripts/evaluate_paper_nvs.py"),
            "--dataset",
            dataset,
            "--scene",
            scene,
            "--source",
            str(Path(job["source"]).resolve()),
            "--method",
            method,
            "--model-kind",
            "hexels" if method.startswith("hexels_") else "2dgs",
            "--model-path",
            str(model_path),
            "--reference-hexel-path",
            str(reference_hexel),
            "--reference-2dgs-path",
            str(reference_2dgs),
            "--output",
            str((output_root / method / dataset / f"{scene}.json").resolve()),
        ]
        if method.startswith("hexels_"):
            variant = method.removeprefix("hexels_")
            for stage in (15, 20, 25, 30):
                command.extend(
                    [
                        "--summary",
                        str(model_path.parent / f"{stage}k/summary.json"),
                    ]
                )
            command.extend(
                [
                    "--compact-bytes-per-primitive",
                    str(HEXELS_COMPACT_BYTES_PER_PRIMITIVE[variant]),
                ]
            )
        evaluation_job["commands"] = [command]
    return _filter_manifest(
        {
            "protocol": "hexels_paper_3d_nvs_evaluation_v1",
            "scene_count": training_manifest["scene_count"],
            "method_count": training_manifest["method_count"],
            "job_count": len(evaluation_jobs),
            "datasets": training_manifest["datasets"],
            "methods": training_manifest["methods"],
            "jobs": evaluation_jobs,
        },
        datasets=set(), scenes=set(), methods=set(),
    )


def run_parallel(manifest_path: Path, state_root: Path, gpus: list[str]) -> None:
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(
                run_worker,
                manifest_path=manifest_path,
                state_root=state_root,
                gpu=gpu,
                worker_index=index,
                worker_count=len(gpus),
            )
            for index, gpu in enumerate(gpus)
        ]
        return_codes = [future.result() for future in futures]
    if any(return_codes):
        raise RuntimeError(f"Benchmark workers failed: {return_codes}")


def status(state_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in (state_root / "status").glob("*.json"):
        value = json.loads(path.read_text())["status"]
        counts[value] = counts.get(value, 0) + 1
    return counts


def summarize(
    evaluation_root: Path,
    output_path: Path,
    *,
    result_paths: list[Path] | None = None,
) -> dict[str, Any]:
    results = [
        json.loads(path.read_text())
        for path in (result_paths if result_paths is not None else evaluation_root.rglob("*.json"))
        if path.resolve() != output_path.resolve()
    ]
    if not results or any(result.get("complete") is not True for result in results):
        raise ValueError("Evaluation results are missing or incomplete")
    datasets = sorted({str(result["dataset"]) for result in results})
    methods = sorted({str(result["method"]) for result in results})
    aggregates: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        aggregates[dataset] = {}
        for method in methods:
            rows = [
                result
                for result in results
                if result["dataset"] == dataset and result["method"] == method
            ]
            if not rows:
                continue
            aggregates[dataset][method] = {
                "scenes": len(rows),
                "PSNR": statistics.fmean(
                    float(row["aggregate"]["mean_psnr_db"]) for row in rows
                ),
                "SSIM": statistics.fmean(
                    float(row["aggregate"]["mean_ssim"]) for row in rows
                ),
                "LPIPS": statistics.fmean(
                    float(row["aggregate"]["mean_lpips_vgg"]) for row in rows
                ),
                "primitives": statistics.fmean(
                    float(row["artifact"]["primitives"]) for row in rows
                ),
                "compact_learned_bytes": statistics.fmean(
                    float(row["artifact"]["compact_learned_bytes"]) for row in rows
                ),
                "resident_model_tensor_bytes": statistics.fmean(
                    float(row["artifact"]["resident_model_tensor_bytes"]) for row in rows
                ),
                "training_seconds": statistics.fmean(
                    float(row["artifact"]["training"]["optimization_seconds"])
                    for row in rows
                ),
                "preprocessing_seconds": statistics.fmean(
                    float(row["artifact"].get("preprocessing", {}).get("seconds", 0.0))
                    for row in rows
                ),
                "preprocessing_timing_complete": all(
                    bool(
                        row["artifact"]
                        .get("preprocessing", {})
                        .get("timing_complete", True)
                    )
                    for row in rows
                ),
                "end_to_end_seconds": statistics.fmean(
                    float(row["artifact"]["training"]["optimization_seconds"])
                    + float(row["artifact"]["training"].get("postprocess_seconds", 0.0))
                    + float(row["artifact"].get("preprocessing", {}).get("seconds", 0.0))
                    for row in rows
                ),
                "peak_training_gpu_gb": statistics.fmean(
                    max(
                        float(row["artifact"]["training"]["peak_gpu_used_gb"]),
                        float(row["artifact"]["training"]["peak_gpu_allocated_gb"]),
                    )
                    for row in rows
                ),
                "fps": statistics.fmean(float(row["benchmark"]["fps"]) for row in rows),
                "latency_ms": statistics.fmean(
                    float(row["benchmark"]["latency_ms"]) for row in rows
                ),
                "peak_render_bytes": statistics.fmean(
                    float(row["benchmark"]["peak_allocated_bytes"]) for row in rows
                ),
            }
    summary = {
        "schema_version": 1,
        "scene_count": len({(result["dataset"], result["scene"]) for result in results}),
        "method_count": len(methods),
        "datasets": datasets,
        "methods": methods,
        "aggregates": aggregates,
    }
    _write_json(output_path, summary)
    return summary


def summarize_suite(
    training: dict[str, Any],
    evaluation: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Aggregate manifest-selected NVS, DTU distance and T&T F-score results separately."""
    if any(job["status"] != "ready" for job in training["jobs"]):
        raise ValueError("Cannot summarize a blocked training manifest")
    summary: dict[str, Any] = {"protocol": training["protocol"]}
    if evaluation["jobs"]:
        paths = [Path(_option(job["commands"][0], "--output")) for job in evaluation["jobs"]]
        summary["nvs"] = summarize(
            output_root / "evaluation", output_root / "summary_nvs.json", result_paths=paths
        )
    dtu_jobs = [job for job in training["jobs"] if job["task"] == "geometry" and job["dataset"] == "dtu"]
    if dtu_jobs:
        from scripts.summarize_paper_3d_dtu import summarize as summarize_dtu

        summary["dtu_geometry"] = summarize_dtu(
            output_root / "training",
            output_root / "training",
            scenes=tuple(dict.fromkeys(job["scene"] for job in dtu_jobs)),
            methods=tuple(training["methods"]),
        )
        _write_json(output_root / "summary_dtu.json", summary["dtu_geometry"])
    tnt_jobs = [job for job in training["jobs"] if job["task"] == "geometry" and job["dataset"] == "tanks_and_temples"]
    if tnt_jobs:
        rows = []
        for job in tnt_jobs:
            path = _model_path(job) / "tnt_eval" / f"{job['scene']}.prf_tau_plotstr.txt"
            precision, recall, fscore, threshold, plot_stretch = map(float, path.read_text().split())
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (precision, recall, fscore)):
                raise ValueError(f"Invalid T&T precision/recall/F-score: {path}")
            if not all(math.isfinite(value) and value > 0 for value in (threshold, plot_stretch)):
                raise ValueError(f"Invalid T&T threshold or plot stretch: {path}")
            rows.append({
                "scene": job["scene"], "method": job["method"],
                "precision_percent": 100 * precision, "recall_percent": 100 * recall,
                "f1_percent": 100 * fscore, "threshold": threshold,
            })
        aggregates = {
            method: {
                metric: statistics.mean(row[metric] for row in rows if row["method"] == method)
                for metric in ("precision_percent", "recall_percent", "f1_percent")
            }
            for method in training["methods"]
        }
        summary["tnt_geometry"] = {
            "scene_count": len({job["scene"] for job in tnt_jobs}),
            "rows": rows, "aggregates": aggregates,
        }
        _write_json(output_root / "summary_tnt.json", summary["tnt_geometry"])
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "run-all", "status", "summarize"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-bin", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASET_SCENES),
    )
    parser.add_argument("--scenes", nargs="+", default=[])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest_root = (
        args.manifest_root.resolve() if args.manifest_root is not None else output_root / "manifests"
    )
    train_manifest_path = manifest_root / "train.json"
    evaluation_manifest_path = manifest_root / "evaluate.json"
    evaluation_root = output_root / "evaluation"
    summary_path = output_root / "summary.json"

    if args.action in {"prepare", "run-all"}:
        config = load_config(args.config)
        if args.python_bin is not None:
            config["python"] = str(args.python_bin.expanduser().resolve())
        training_manifest = prepare_training_manifest(
            output_root=output_root / "training",
            manifest_path=train_manifest_path,
            datasets=set(args.datasets or config["datasets"]),
            scenes=set(args.scenes),
            config=config,
        )
        evaluation_manifest = build_evaluation_manifest(
            training_manifest,
            output_root=evaluation_root,
        )
        _write_manifest(evaluation_manifest_path, evaluation_manifest)
        LOGGER.info(
            "Prepared %d training and %d evaluation jobs",
            training_manifest["job_count"],
            evaluation_manifest["job_count"],
        )
        if args.action == "prepare":
            blocked = [job for job in training_manifest["jobs"] if job["status"] != "ready"]
            LOGGER.info(
                "Dry run only: %d jobs blocked by missing data; Python: %s",
                len(blocked), config["python"],
            )
            return

    if args.action == "run-all":
        gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError("Provide a non-empty list of distinct GPU IDs")
        if not os.access(config["python"], os.X_OK):
            raise ValueError(f"Configured Python is not executable: {config['python']}")
        run_parallel(train_manifest_path, output_root / "runner/train", gpus)
        if evaluation_manifest["jobs"]:
            run_parallel(evaluation_manifest_path, output_root / "runner/evaluate", gpus)
        summarize_suite(training_manifest, evaluation_manifest, output_root)
        LOGGER.info("Completed paper benchmark: %s", summary_path)
    elif args.action == "status":
        print(
            json.dumps(
                {
                    "train": status(output_root / "runner/train"),
                    "evaluate": status(output_root / "runner/evaluate"),
                },
                indent=2,
            )
        )
    elif args.action == "summarize":
        summarize_suite(
            json.loads(train_manifest_path.read_text()),
            json.loads(evaluation_manifest_path.read_text()),
            output_root,
        )
        LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
