from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from hexels import benchmark_metrics as quality


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _training_cost(summary_paths: Sequence[Path]) -> dict[str, float]:
    summaries = [_read_json(path) for path in summary_paths]
    return {
        "optimization_seconds": sum(float(value["train_seconds"]) for value in summaries),
        "postprocess_seconds": sum(float(value.get("postprocess_seconds", 0.0)) for value in summaries),
        "peak_gpu_used_gb": max(float(value.get("peak_gpu_used_gb", 0.0)) for value in summaries),
        "peak_gpu_allocated_gb": max(
            float(value.get("peak_gpu_allocated_gb", 0.0)) for value in summaries
        ),
    }


def _preprocessing_cost(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"seconds": 0.0, "timing_complete": True, "artifact": None}
    payload = _read_json(path)
    timings = payload.get("timings_seconds", {})
    if not isinstance(timings, Mapping):
        raise ValueError(f"Invalid preprocessing timings: {path}")
    return {
        "seconds": sum(float(value) for value in timings.values()),
        "timing_complete": not payload.get("reused_stages"),
        "artifact": quality._artifact_record(path),
    }


def _load_cameras(
    source: Path,
    llffhold: int,
    reference_hexel_path: Path | None,
    reference_2dgs_path: Path | None,
    device: torch.device,
    camera_config: Path | None = None,
    split: str = "test",
    with_geometry: bool = False,
) -> tuple[list[Any], dict[str, object]] | tuple[list[Any], dict[str, object], dict[str, list]]:
    if camera_config is None:
        hexel_config = _read_json(reference_hexel_path / "config.json")
        two_dgs_config = quality.parse_namespace_config((reference_2dgs_path / "cfg_args").read_text())
    else:
        hexel_config = _read_json(camera_config)
        two_dgs_config = dict(hexel_config)
    hexel_train, hexel_test, two_dgs_train, two_dgs_test = (
        quality._load_independent_cameras(
            {"path": str(source), "llffhold": llffhold},
            hexel_config,
            two_dgs_config,
            reference_2dgs_path / "cameras.json" if reference_2dgs_path is not None else None,
        )
    )
    training_contract = quality.camera_contract(hexel_train, include_names=False)
    test_contract = quality.camera_contract(hexel_test, include_names=True) if hexel_test else None
    quality.validate_camera_sets(
        hexel_train,
        two_dgs_train,
        training_contract,
        include_names=False,
    )
    if hexel_test:
        quality.validate_camera_sets(hexel_test, two_dgs_test, test_contract, include_names=True)
    selected = hexel_test if split == "test" else hexel_train
    quality.validate_camera_device(selected, device)
    contract = {"training": training_contract, "test": test_contract}
    if with_geometry:
        geometry = {"training": [quality.camera_geometry(camera) for camera in hexel_train],
                    "test": [quality.camera_geometry(camera) for camera in hexel_test]}
        return selected, contract, geometry
    return selected, contract


def _load_hexels(
    model_path: Path,
    iteration: int,
    summaries: Sequence[Path],
    compact_bytes_per_primitive: int,
    device: torch.device,
) -> tuple[Callable[[Any], torch.Tensor], dict[str, Any]]:
    config_path = model_path / "config.json"
    checkpoint_path = model_path / "checkpoints" / f"ckpt_{iteration:06d}.pt"
    config = _read_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("iteration") != iteration:
        raise ValueError(f"Invalid Hexel checkpoint: {checkpoint_path}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping):
        raise ValueError(f"Missing Hexel model state: {checkpoint_path}")
    model, renderer, background = quality._load_hexels_model(
        checkpoint_path,
        config,
        device,
    )

    def render(camera: Any) -> torch.Tensor:
        return renderer.render(camera, model, background)["image"]

    primitives = model.num_active_base + model.num_children
    cost = _training_cost(summaries)
    artifact = {
        "iteration": iteration,
        "primitives": primitives,
        "base_primitives": model.num_active_base,
        "child_primitives": model.num_children,
        "resident_model_tensor_bytes": quality._model_tensor_bytes(state),
        "compact_learned_bytes": primitives * compact_bytes_per_primitive,
        "compact_bytes_per_primitive": compact_bytes_per_primitive,
        "training": cost,
        "training_timing_scope": "trainer loop, including checkpoint writes and in-loop reporting",
        "checkpoint": quality._artifact_record(checkpoint_path),
        "config": quality._artifact_record(config_path),
        "summaries": [quality._artifact_record(path) for path in summaries],
    }
    return render, artifact


def _load_two_dgs(
    model_path: Path,
    iteration: int,
) -> tuple[Callable[[Any], torch.Tensor], dict[str, Any]]:
    config_path = model_path / "cfg_args"
    telemetry_path = model_path / "training_telemetry.json"
    point_cloud_path = model_path / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    config = quality.parse_namespace_config(config_path.read_text())
    telemetry = _read_json(telemetry_path)
    primitives = quality._ply_vertex_count(point_cloud_path)
    model, render_2dgs, pipeline, background = quality._load_two_dgs_model(
        point_cloud_path,
        config,
        primitives,
    )

    def render(camera: Any) -> torch.Tensor:
        return render_2dgs(camera, model, pipeline, background)["render"]

    state = {
        name: getattr(model, name)
        for name in (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
        )
    }
    artifact = {
        "iteration": iteration,
        "primitives": primitives,
        "resident_model_tensor_bytes": quality._model_tensor_bytes(state),
        "compact_learned_bytes": quality._model_tensor_bytes(state),
        "compact_bytes_per_primitive": quality._model_tensor_bytes(state) // primitives,
        "training": {
            "optimization_seconds": float(telemetry["wall_time_seconds"]),
            "postprocess_seconds": 0.0,
            "peak_gpu_used_gb": float(telemetry.get("peak_cuda_reserved_bytes", 0.0))
            / 1024**3,
            "peak_gpu_allocated_gb": float(
                telemetry.get("peak_cuda_allocated_bytes", 0.0)
            )
            / 1024**3,
        },
        "training_timing_scope": "scene/model initialization and trainer loop, including saves and evaluations",
        "point_cloud": quality._artifact_record(point_cloud_path),
        "config": quality._artifact_record(config_path),
        "telemetry": quality._artifact_record(telemetry_path),
    }
    return render, artifact


def evaluate(
    *,
    dataset: str,
    scene: str,
    source: Path,
    method: str,
    model_kind: str,
    model_path: Path,
    iteration: int,
    summaries: Sequence[Path],
    compact_bytes_per_primitive: int,
    reference_hexel_path: Path | None,
    reference_2dgs_path: Path | None,
    preprocessing_artifact: Path | None,
    llffhold: int,
    output: Path,
    device: torch.device,
    camera_config: Path | None = None,
    split: str = "test",
    benchmark_only: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    if camera_config is not None:
        expected = _read_json(camera_config)
        actual = (_read_json(model_path / "config.json") if model_kind == "hexels"
                  else quality.parse_namespace_config((model_path / "cfg_args").read_text()))
        for name in ("eval", "images", "resolution"):
            value = actual.get(name)
            if name == "images":
                value = value or "images"
            if value != expected[name]:
                raise ValueError(f"Model camera setting differs: {name}: {value} != {expected[name]}")
        if Path(actual["source_path"]).resolve() != source.resolve():
            raise ValueError("Model source differs from the evaluation source")
        if Path(expected["source_path"]).resolve() != source.resolve():
            raise ValueError("Camera manifest source differs from the evaluation source")
        if int(actual.get("llffhold", 8)) != llffhold or int(expected["llffhold"]) != llffhold:
            raise ValueError("Model or camera manifest holdout differs from the evaluation holdout")
    if output.exists():
        raise FileExistsError(output)
    cameras, camera_contract = _load_cameras(
        source,
        llffhold,
        reference_hexel_path,
        reference_2dgs_path,
        device,
        camera_config,
        split,
    )
    if model_kind == "hexels":
        render, artifact = _load_hexels(
            model_path,
            iteration,
            summaries,
            compact_bytes_per_primitive,
            device,
        )
    elif model_kind == "2dgs":
        render, artifact = _load_two_dgs(model_path, iteration)
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    artifact["preprocessing"] = _preprocessing_cost(preprocessing_artifact)

    benchmark = quality._benchmark(render, cameras, device)
    metric_started = time.perf_counter()
    aggregate, rows = ({}, []) if benchmark_only else quality._metrics(render, cameras, device)
    metric_seconds = time.perf_counter() - metric_started
    result = {
        "schema_version": 1,
        "complete": True,
        "dataset": dataset,
        "scene": scene,
        "method": method,
        "model_kind": model_kind,
        "split": {"eval": split == "test", "name": split, "llffhold": llffhold, "views": len(cameras)},
        "evaluation_seconds": time.perf_counter() - started,
        "metric_seconds": metric_seconds,
        "camera_contract": camera_contract,
        "artifact": artifact,
        "aggregate": aggregate,
        "benchmark": benchmark,
        "hardware": {
            "device_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "per_view": rows,
        "provenance": {
            "evaluator": quality._artifact_record(Path(__file__).resolve()),
            "source_images_bin": quality._artifact_record(source / "sparse/0/images.bin"),
            "metric_implementation": quality._artifact_record(
                ROOT / "hexels/benchmark_metrics.py"
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model-kind", choices=("hexels", "2dgs"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument("--compact-bytes-per-primitive", type=int, default=0)
    parser.add_argument("--reference-hexel-path", type=Path)
    parser.add_argument("--reference-2dgs-path", type=Path)
    parser.add_argument("--camera-config", type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--preprocessing-artifact", type=Path)
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model_kind == "hexels" and (
        not args.summary or args.compact_bytes_per_primitive < 1
    ):
        raise ValueError("Hexel evaluation requires summaries and compact bytes")
    result = evaluate(
        dataset=args.dataset,
        scene=args.scene,
        source=args.source.resolve(),
        method=args.method,
        model_kind=args.model_kind,
        model_path=args.model_path.resolve(),
        iteration=args.iteration,
        summaries=[path.resolve() for path in args.summary],
        compact_bytes_per_primitive=args.compact_bytes_per_primitive,
        reference_hexel_path=args.reference_hexel_path.resolve() if args.reference_hexel_path else None,
        reference_2dgs_path=args.reference_2dgs_path.resolve() if args.reference_2dgs_path else None,
        preprocessing_artifact=(
            args.preprocessing_artifact.resolve()
            if args.preprocessing_artifact is not None
            else None
        ),
        llffhold=args.llffhold,
        output=args.output.resolve(),
        device=torch.device("cuda"),
        camera_config=args.camera_config,
        split=args.split,
        benchmark_only=args.benchmark_only,
    )
    print(json.dumps({"dataset": args.dataset, "scene": args.scene, **result["aggregate"]}))


if __name__ == "__main__":
    main()
