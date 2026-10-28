"""Native external-baseline adapters, isolated to one method per process."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

from scripts.setup_nvs_baselines import METHODS, ROOT, environment_prefix


def training_command(job: dict, settings: dict, smoke: bool) -> tuple[list[str], Path]:
    method = job["method"]
    if job["seed"] != 0:
        # None of the pinned external trainers exposes a seed option; each safe_state() seeds 0.
        raise ValueError(f"{method} cannot honor seed {job['seed']}: its trainer has no seed option and always seeds 0")
    repository = ROOT / METHODS[method][0]
    cwd = repository / "milo" if method == "milo" else repository
    python = environment_prefix(method) / "bin/python"
    entrypoint = "train_regular_densification.py" if method == "milo" and job["dataset"] == "dtu" else "train.py"
    command = [str(python), entrypoint, "-s", settings["source_path"], "-m", job["model_path"],
               "--images", settings["images"], "--resolution", str(settings["resolution"]),
               "--eval", "--data_device", "cuda", "--iterations", str(job["iteration"]),
               "--save_iterations", str(job["iteration"]), "--test_iterations", "-1", "--quiet"]
    outdoor = job["dataset"] == "tanks_and_temples" or (job["dataset"] == "mipnerf360" and job["scene"] in {"bicycle", "flowers", "garden", "stump", "treehill"})
    if method in {"3dgs", "nexels"}:
        command.append("--disable_viewer")
        if job["dataset"] == "dtu":
            command.append("--preserve_rgb_background")
    elif method in {"3d_convex", "triangle"}:
        if outdoor:
            command.append("--outdoor")
        if method == "triangle" and job["dataset"] == "dtu":
            command += ["--max_shapes", "500000", "--depth_ratio", "1.0",
                        "--lambda_normals", "0.0028", "--lambda_dist", "0.014",
                        "--iteration_mesh", "25000", "--densify_until_iter", "25000",
                        "--lambda_opacity", "0.0044", "--importance_threshold", "0.027",
                        "--lr_triangles_points_init", "0.0015", "--lambda_size", "0.0", "--no_dome"]
    elif method == "mesh_splatting" and not outdoor:
        command.append("--indoor")
    if method == "bbsplat":
        cap = 60000 if job["dataset"] == "dtu" else 300000 if outdoor else 160000
        command += ["--cap_max", str(cap), "--max_read_points", str(cap if job["dataset"] == "dtu" else cap - 10000)]
        if job["dataset"] == "dtu":
            command += ["--lambda_normal", "0.05", "--lambda_dist", "100"]
        else:
            command.append("--add_sky_box")
    if method == "nexels":
        command += ["--minres", "16", "--maxres", "256" if job["dataset"] == "mipnerf360" else "1024",
                    "--log_hash_table_size", "20", "--cap_max_init", "200000", "--cap_max_final", "400000", "--no_texture_clamp"]
    if method == "milo":
        command += ["--rasterizer", "radegs", "--imp_metric", "outdoor" if outdoor else "indoor",
                    "--log_interval", "200", "--ip", "127.0.0.1", "--port", "0"]
        if job["dataset"] == "dtu":
            command += ["--mesh_config", "default_dtu", "--decoupled_appearance"]
        elif smoke:
            command += ["--config_path", job["activation_config"]]
        elif job["iteration"] != 18000:
            raise ValueError("Native MILo fast training ends at 18K, not 30K")
    return command, cwd


def tensor_bytes(value: object) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def float_scalars(value: object) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() if value.is_floating_point() else 0
    if isinstance(value, dict):
        return sum(float_scalars(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(float_scalars(item) for item in value)
    return 0


def model_artifact(method: str, path: Path, iteration: int, model: object) -> dict:
    import numpy as np
    import torch
    from plyfile import PlyData
    from hexels import benchmark_metrics as quality

    folder = path / "point_cloud" / f"iteration_{iteration}"
    files = [path / "cfg_args"]
    diagnostics = {}
    if method in {"3d_convex", "triangle", "mesh_splatting"}:
        state_path = folder / "point_cloud_state_dict.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        keys = {
            "3d_convex": ["convex_points", "delta", "sigma", "features_dc", "features_rest", "opacity"],
            "triangle": ["triangles_points", "sigma", "features_dc", "features_rest", "opacity"],
            "mesh_splatting": ["triangles_points", "_triangle_indices", "vertex_weight", "sigma", "features_dc", "features_rest"],
        }[method]
        learned_bytes = sum(tensor_bytes(state[key]) for key in keys)
        scalar_count = sum(float_scalars(state[key]) for key in keys)
        count_key = "_triangle_indices" if method == "mesh_splatting" else "convex_points" if method == "3d_convex" else "triangles_points"
        primitives = len(state[count_key])
        files.append(state_path)
        if method == "mesh_splatting" and (path / "initialization_budget_audit.json").exists():
            from utils.memory_control import topology_summary

            audit_path = path / "initialization_budget_audit.json"
            diagnostics = {"initialization_budget_audit": json.loads(audit_path.read_text()),
                           "topology": topology_summary(state["triangles_points"].detach().numpy(), state["_triangle_indices"].detach().numpy()),
                           "control_evidence": [quality._artifact_record(audit_path),
                                                quality._artifact_record(path / "initialization_indices.npy")]}
        if method != "mesh_splatting":
            files.append(folder / "hyperparameters.pt")
    else:
        ply_path = folder / "point_cloud.ply"
        vertices = PlyData.read(str(ply_path))["vertex"].data
        primitives = len(vertices)
        learned_bytes = sum(vertices[name].nbytes for name in vertices.dtype.names if name not in {"nx", "ny", "nz"})
        scalar_count = sum(vertices[name].size for name in vertices.dtype.names
                           if name not in {"nx", "ny", "nz"} and np.issubdtype(vertices[name].dtype, np.floating))
        files.append(ply_path)
        if method == "bbsplat":
            for field in ("texture_alpha", "texture_color"):
                texture = folder / f"{field}.npz"
                with np.load(texture) as arrays:
                    learned_bytes += sum(array.size * 4 for array in arrays.values())
                    scalar_count += sum(array.size for array in arrays.values())
                files.append(texture)
        if method == "nexels":
            tensor_path = path / "tensor" / f"iteration_{iteration}" / "tensor.pth"
            tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
            learned_bytes += tensor_bytes(tensors)
            scalar_count += float_scalars(tensors)
            files.append(tensor_path)
    records = [quality._artifact_record(file) for file in files]
    return {"iteration": iteration, "primitives": primitives, "compact_learned_bytes": learned_bytes, **diagnostics,
            "model_float_scalars": scalar_count,
            "parameter_count_scope": "deployment floating-point geometry/appearance state; excludes optimizer, includes persistent float buffers",
            "serialized_model_bytes": sum(record["bytes"] for record in records),
            "deployment_files": records,
            "state_accounting": "native learned tensors, plus texture/grid/topology; serialized loader artifacts separately"}


def load_model(job: dict) -> tuple[object, dict, list, dict]:
    import torch
    from hexels import benchmark_metrics as quality

    for filename, expected in job.get("reused_training", {}).get("sha256", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Reused training artifact changed: {filename}")

    method = job["method"]
    repository = ROOT / METHODS[method][0]
    module_root = repository / "milo" if method == "milo" else repository
    sys.path.insert(0, str(module_root))
    os.chdir(module_root)
    from arguments import ModelParams, PipelineParams
    from scene import Scene

    path = Path(job["model_path"])
    parser = argparse.ArgumentParser()
    model_params, pipeline_params = ModelParams(parser), PipelineParams(parser)
    nexel_params = None
    if method == "nexels":
        from arguments import NexelParams
        nexel_params = NexelParams(parser)
    actual = quality.parse_namespace_config((path / "cfg_args").read_text())
    arguments = argparse.Namespace(**{**vars(parser.parse_args([])), **actual})
    expected = json.loads(Path(job["camera_settings"]).read_text())
    for key in ("images", "resolution", "eval", "llffhold"):
        if getattr(arguments, key, 8 if key == "llffhold" else None) != expected[key]:
            raise ValueError(f"Native camera configuration differs: {method}/{key}")
    if Path(arguments.source_path).resolve() != Path(expected["source_path"]).resolve():
        raise ValueError("Native source differs from canonical input")
    arguments.model_path, arguments.data_device = str(path), "cpu"
    dataset, pipeline = model_params.extract(arguments), pipeline_params.extract(arguments)
    background = torch.tensor([1.0 if dataset.white_background else 0.0] * 3, device="cuda")
    scene_options = {"load_iteration": job["iteration"], "shuffle": False}
    if method == "3d_convex":
        from scene import ConvexModel
        from convex_renderer import render
        model = ConvexModel(dataset.sh_degree)
        scene = Scene(dataset, model, None, None, None, None, None, None, **scene_options)
    elif method in {"triangle", "mesh_splatting"}:
        from scene import TriangleModel
        from triangle_renderer import render
        model = TriangleModel(dataset.sh_degree)
        if method == "triangle":
            scene = Scene(dataset, model, None, None, None, None, **scene_options)
        else:
            model.scaling = 4
            scene = Scene(dataset, model, None, None, **scene_options)
    elif method == "nexels":
        from nexel_renderer import NexelModel, render
        model = NexelModel(dataset.sh_degree, dataset.texture_sh_degree, nexel_params.extract(arguments))
        scene = Scene(dataset, model, **scene_options)
    else:
        from scene import GaussianModel
        if method == "bbsplat" and hasattr(dataset, "texture_size"):
            model = GaussianModel(dataset.sh_degree, texture_preproc=True, texture_size=dataset.texture_size)
        else:
            model = GaussianModel(dataset.sh_degree, texture_preproc=True) if method == "bbsplat" else GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, model, **scene_options)
        if method == "milo":
            from gaussian_renderer.radegs import render_radegs as render
        else:
            from gaussian_renderer import render

    def render_rgb(camera: object) -> torch.Tensor:
        if method == "nexels":
            if not hasattr(camera, "k_matrix"):
                from utils.graphics_utils import getIntrinsicsMatrix
                camera.k_matrix = getIntrinsicsMatrix(camera.FoVx, camera.FoVy, camera.image_width, camera.image_height).cuda()
            return render(job["iteration"], False, camera, model, pipeline, background,
                          compute_geometric=False, use_vis_features=False, override_settings=arguments.override_settings)["render"]
        if method == "bbsplat":
            return render(camera, model, pipeline, background, additional_return=True)["render"]
        return render(camera, model, pipeline, background)["render"]

    def canonical_contract(cameras: list, include_names: bool) -> dict:
        normalized = []
        for camera in cameras:
            proxy = copy.copy(camera)
            proxy.image_name = Path(camera.image_name).name.split(".")[0]
            normalized.append(proxy)
        return quality.camera_contract(normalized, include_names=include_names)

    contract = {"training": canonical_contract(scene.getTrainCameras(), False),
                "test": canonical_contract(scene.getTestCameras(), True)}
    if job.get("camera_geometry_output") or job.get("camera_contract"):
        geometry = {}
        for split, cameras in (("training", scene.getTrainCameras()), ("test", scene.getTestCameras())):
            geometry[split] = []
            for camera in cameras:
                proxy = copy.copy(camera)
                proxy.image_name = Path(camera.image_name).name.split(".")[0]
                geometry[split].append(quality.camera_geometry(proxy))
        geometry_path = Path(job.get("camera_geometry_output", str(Path(job["result"]).parent / "native_camera_geometry.json")))
        geometry_path.write_text(json.dumps(geometry, indent=2) + "\n")
    diagnostic = Path(job["result"]).parent / "native_camera_contract.json"
    diagnostic.write_text(json.dumps(contract, indent=2) + "\n")
    return render_rgb, model_artifact(method, path, job["iteration"], model), scene.getTestCameras(), contract
