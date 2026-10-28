from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree

from hexels.mvs_input import prepare_proposal_mvs_input, read_points3d_binary
from hexels.mvs_surface import read_fused_visibility_counts, sample_dense_surface
from hexels.benchmark_metrics import _artifact_record


def _read_fused(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(path)["vertex"].data
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"]))
    normals = np.column_stack((vertex["nx"], vertex["ny"], vertex["nz"]))
    colors = np.column_stack((vertex["red"], vertex["green"], vertex["blue"]))
    return points, normals, colors


def _knn_scales(points: np.ndarray, multiplier: float) -> torch.Tensor:
    distances, _ = cKDTree(points.astype(np.float64, copy=False)).query(
        points,
        k=4,
        workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    )
    rms = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))
    scales = np.minimum(multiplier * rms, 0.25).astype(np.float32)
    return torch.from_numpy(scales)[:, None].repeat(1, 2)


def _pack_payload(
    *,
    mode: str,
    target_count: int,
    scale_multiplier: float,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
    dense_points: np.ndarray,
    dense_colors: np.ndarray,
    dense_rotations: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if mode == "mvs":
        dense_count = target_count
        points = dense_points[:dense_count]
        colors = dense_colors[:dense_count]
        rotations = dense_rotations[:dense_count]
    elif mode == "hybrid":
        dense_count = target_count - len(sparse_points)
        if dense_count <= 0:
            raise ValueError("Hybrid target must exceed the sparse point count")
        selected_dense = dense_points[:dense_count]
        selected_rotations = dense_rotations[:dense_count]
        nearest_dense = cKDTree(selected_dense.astype(np.float64)).query(
            sparse_points.astype(np.float64),
            k=1,
            workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
        )[1]
        points = np.concatenate((sparse_points, selected_dense), axis=0)
        colors = np.concatenate((sparse_colors, dense_colors[:dense_count]), axis=0)
        rotations = torch.cat(
            (selected_rotations[torch.from_numpy(nearest_dense)], selected_rotations),
            dim=0,
        )
    else:
        raise ValueError(f"Unknown pack mode: {mode}")
    if len(points) != target_count or len(dense_points) < dense_count:
        raise ValueError("Dense surface pool is smaller than the requested pack")
    return {
        "means3d": torch.from_numpy(points.astype(np.float32, copy=True)),
        "scales": _knn_scales(points, scale_multiplier),
        "rotations": rotations.clone(),
        "colors": torch.from_numpy(colors.astype(np.float32, copy=True) / 255.0),
    }


def training_camera_names(cameras: list, smoke_test: bool = False, all_views: bool = False) -> list[str]:
    names = [Path(camera.name).stem for index, camera in enumerate(cameras) if all_views or index % 8 != 0]
    return names[:12] if smoke_test else names


def prepare(source: Path, output: Path, images: str, colmap: Path, seed: int, smoke_test: bool = False,
            *, all_views: bool = False, recipe: str = "hybrid160_s045", max_image_size: int = 1600) -> None:
    """Prepare shared MVS packs under an explicit training-view and population contract."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2d-gaussian-splatting"))
    from scene.colmap_loader import read_extrinsics_binary

    output.mkdir(parents=True, exist_ok=True)
    cameras = sorted(read_extrinsics_binary(source / "sparse/0/images.bin").values(), key=lambda camera: camera.name)
    names = training_camera_names(cameras, smoke_test, all_views)
    contract = {"source": str(source), "images": images, "seed": seed, "all_views": all_views,
                "recipe": recipe, "max_image_size": max_image_size, "smoke": smoke_test, "camera_names": names}
    contract_path = output / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("MVS contract changed; use a fresh output directory")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    timings = {}
    started = time.perf_counter()
    inputs = output / "input"
    input_record = output / "input_preparation.json"
    if not inputs.exists():
        prepare_proposal_mvs_input(source_path=source, camera_names=names, output_path=inputs, image_directory=images)
        input_record.write_text(json.dumps({"seconds": time.perf_counter() - started}) + "\n")
    timings["input_preparation"] = json.loads(input_record.read_text())["seconds"]
    workspace = inputs / "workspace"
    fused = workspace / "fused.ply"
    commands = {
        "image_undistorter": [str(colmap), "image_undistorter", "--image_path", str(inputs / "input_images"),
            "--input_path", str(inputs / "input_model"), "--output_path", str(workspace),
            "--output_type", "COLMAP", "--max_image_size", "320" if smoke_test else str(max_image_size), "--copy_policy", "soft-link",
            "--num_patch_match_src_images", "20"],
        "patch_match_stereo": [str(colmap), "patch_match_stereo", "--workspace_path", str(workspace),
            "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.write_consistency_graph", "1", "--PatchMatchStereo.cache_size", "8",
            "--PatchMatchStereo.gpu_index", "0"],
        "stereo_fusion": [str(colmap), "stereo_fusion", "--workspace_path", str(workspace),
            "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", str(fused),
            "--StereoFusion.min_num_pixels", "5", "--StereoFusion.max_reproj_error", "2",
            "--StereoFusion.max_depth_error", "0.01", "--StereoFusion.max_normal_error", "10",
            "--StereoFusion.num_threads", os.environ.get("SLURM_CPUS_PER_TASK", "8")],
    }
    for stage, command in commands.items():
        record_path = output / f"{stage}.json"
        if record_path.exists():
            timings[stage] = json.loads(record_path.read_text())["seconds"]
            continue
        started = time.perf_counter()
        with (output / f"{stage}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        timings[stage] = time.perf_counter() - started
        record_path.write_text(json.dumps({"command": command, "seconds": timings[stage]}) + "\n")
    if smoke_test:
        points, normals, colors = _read_fused(fused)
        visibility = read_fused_visibility_counts(fused.with_suffix(".ply.vis"), len(points))
        sampled = sample_dense_surface(points, normals, visibility, target_count=100,
            sampler="density", random_seed=20260801 + seed, minimum_visibility=5,
            normal_neighbors=16, minimum_normal_coherence=0.0, scale_neighbors=3)
        payload = _pack_payload(mode="mvs", target_count=100, scale_multiplier=0.45,
            sparse_points=np.empty((0, 3)), sparse_colors=np.empty((0, 3)),
            dense_points=points[sampled.indices], dense_colors=colors[sampled.indices],
            dense_rotations=sampled.rotations)
        torch.save(payload, output / "smoke_geometry.pt")
        (output / "smoke_result.json").write_text(json.dumps({"smoke_only": True,
            "fused_points": len(points), "pack_points": 100, "timings_seconds": timings}) + "\n")
        return
    started = time.perf_counter()
    sparse = read_points3d_binary(source / "sparse/0/points3D.bin")
    sparse_rows = [sparse[index] for index in sorted(sparse)]
    sparse_points = np.stack([point.xyz for point in sparse_rows])
    sparse_colors = np.stack([point.rgb for point in sparse_rows])
    equal_sparse = recipe == "equal_sparse"
    sampling_seed = seed if equal_sparse else 20260801 + seed
    target_count = len(sparse_points) if equal_sparse else max(160_000, math.ceil(1.25 * len(sparse_points)))
    pool_count = target_count if equal_sparse else max(160_000, max(224_000, math.ceil(1.25 * len(sparse_points))) - len(sparse_points))
    points, normals, colors = _read_fused(fused)
    visibility = read_fused_visibility_counts(fused.with_suffix(".ply.vis"), len(points))
    eligible = np.count_nonzero(np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
        & (np.linalg.norm(normals, axis=1) > 1e-12) & (visibility >= 5))
    candidates = sample_dense_surface(points, normals, visibility,
        target_count=min(eligible, max(pool_count * 2, pool_count + 100_000)),
        sampler="density", random_seed=sampling_seed, minimum_visibility=5,
        normal_neighbors=16, minimum_normal_coherence=0.0, scale_neighbors=3)
    coherent = torch.nonzero(candidates.normal_coherence >= 0.85, as_tuple=False).flatten().numpy()
    if len(coherent) < pool_count:
        raise ValueError(f"Insufficient coherent MVS samples: {len(coherent)} < {pool_count}")
    rng = np.random.default_rng(sampling_seed)
    selected = coherent[rng.permutation(len(coherent))[:pool_count]]
    fused_rows = candidates.indices[selected]
    payload = _pack_payload(mode="mvs" if equal_sparse else "hybrid", target_count=target_count,
        scale_multiplier=0.25 if equal_sparse else 0.45,
        sparse_points=sparse_points, sparse_colors=sparse_colors, dense_points=points[fused_rows],
        dense_colors=colors[fused_rows], dense_rotations=candidates.rotations[torch.from_numpy(selected)])
    torch.save(payload, output / "initial_geometry.pt")
    timings["surface_pack"] = time.perf_counter() - started
    artifact = {"uses_2dgs_inputs": False, "seed": seed, "sampling_seed": sampling_seed,
        "all_views": all_views, "training_camera_names": names,
        "test_camera_names": [] if all_views else [Path(camera.name).stem for camera in cameras[::8]],
        "timings_seconds": timings, "commands": commands, "target_count": target_count,
        "colmap_version": subprocess.check_output([str(colmap), "-h"], text=True),
        "recipe": recipe, "fused_ply": _artifact_record(fused),
        "initial_geometry": _artifact_record(output / "initial_geometry.pt")}
    (output / "mvs_artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-only MVS initialization for any COLMAP scene.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--colmap", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-views", action="store_true")
    parser.add_argument("--recipe", choices=("hybrid160_s045", "equal_sparse"), default="hybrid160_s045")
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument("--smoke-test", action="store_true", help="Validate 12 training views at 320px; not a benchmark result.")
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve(), args.images, args.colmap.resolve(), args.seed, args.smoke_test,
            all_views=args.all_views, recipe=args.recipe, max_image_size=args.max_image_size)
