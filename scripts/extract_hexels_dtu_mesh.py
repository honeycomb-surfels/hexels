from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hexels.render import RenderOptions, _load_render_components, _save_image


LOGGER = logging.getLogger("hexels_mesh")


def _camera_parameters(camera: object) -> o3d.camera.PinholeCameraParameters:
    width = int(camera.image_width)
    height = int(camera.image_height)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=0.5 * width / math.tan(0.5 * float(camera.FoVx)),
        fy=0.5 * height / math.tan(0.5 * float(camera.FoVy)),
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
    )
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = intrinsic
    parameters.extrinsic = np.asarray(camera.world_view_transform.T.detach().cpu())
    return parameters


def _post_process_mesh(
    mesh: o3d.geometry.TriangleMesh,
    *,
    clusters_to_keep: int,
) -> o3d.geometry.TriangleMesh:
    triangle_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    cluster_ids = np.asarray(triangle_clusters)
    sizes = np.asarray(cluster_sizes)
    threshold = max(int(np.sort(sizes)[-clusters_to_keep]), 50)
    mesh.remove_triangles_by_mask(sizes[cluster_ids] < threshold)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


@torch.no_grad()
def extract_mesh(
    *,
    checkpoint: Path,
    model_path: Path | None,
    output_model_path: Path | None,
    voxel_size: float,
    sdf_trunc: float,
    depth_trunc: float,
    clusters_to_keep: int,
    depth_mode: str = "median",
    alpha_threshold: float = 0.0,
    preview_views: int = 0,
) -> tuple[Path, Path]:
    if depth_mode not in {"expected", "median"} or not 0.0 <= alpha_threshold < 1.0:
        raise ValueError("Invalid depth mode or alpha threshold")
    started = time.perf_counter()
    opt = RenderOptions()
    opt.checkpoint = str(checkpoint)
    opt.model_path = None if model_path is None else str(model_path)
    opt.split = "train"
    opt.skip_test = True
    opt.skip_path = True
    opt.process_args()
    if opt.eval:
        LOGGER.warning("This checkpoint was trained with held-out views; official surface evaluation uses all-view checkpoints")
    iteration, model, scene, renderer, background = _load_render_components(opt)
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    destination = output_model_path or (
        Path(opt.model_path) if opt.model_path is not None else checkpoint.resolve().parents[1]
    )
    output_dir = destination / "train" / f"ours_{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    valid_pixels = 0
    total_pixels = 0
    render_seconds = 0.0
    fusion_seconds = 0.0
    preview_seconds = 0.0

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    cameras = list(scene.train_cameras)
    for index, camera in enumerate(cameras):
        render_started = time.perf_counter()
        package = renderer.render(
            camera, model, background, return_surface=True,
            return_median_depth=depth_mode == "median",
        )
        alpha = package["surf_alpha"]
        depth = (package["surf_median_depth"] if depth_mode == "median"
                 else package["surf_depth"] / alpha.clamp_min(1e-6))
        depth = torch.where(alpha > alpha_threshold, depth, torch.zeros_like(depth))
        if camera.gt_alpha_mask is not None:
            depth = torch.where(camera.gt_alpha_mask < 0.5, torch.zeros_like(depth), depth)

        color_hwc = (
            package["image"]
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        depth_hw = depth.squeeze(0).float().contiguous().cpu().numpy()
        render_seconds += time.perf_counter() - render_started
        valid_pixels += int(np.count_nonzero((depth_hw > 0) & (depth_hw < depth_trunc)))
        total_pixels += depth_hw.size
        if index < preview_views:
            preview_started = time.perf_counter()
            preview_dir = output_dir / "previews"
            preview_dir.mkdir(exist_ok=True)
            _save_image(preview_dir / f"{index:03d}_rgb.png", package["image"])
            _save_image(preview_dir / f"{index:03d}_gt.png", camera.original_image[:3])
            normals = torch.nn.functional.normalize(package["surf_normals"], dim=0)
            _save_image(preview_dir / f"{index:03d}_normal.png", normals * 0.5 + 0.5)
            np.save(preview_dir / f"{index:03d}_depth.npy", depth_hw)
            preview_seconds += time.perf_counter() - preview_started
        fusion_started = time.perf_counter()
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_hwc),
            o3d.geometry.Image(depth_hw),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        parameters = _camera_parameters(camera)
        volume.integrate(rgbd, parameters.intrinsic, parameters.extrinsic)
        fusion_seconds += time.perf_counter() - fusion_started
        if (index + 1) % 10 == 0 or index + 1 == len(cameras):
            LOGGER.info("Integrated %d/%d cameras", index + 1, len(cameras))

    fusion_started = time.perf_counter()
    mesh = volume.extract_triangle_mesh()
    fusion_seconds += time.perf_counter() - fusion_started
    raw_path = output_dir / "fuse.ply"
    post_path = output_dir / "fuse_post.ply"
    write_started = time.perf_counter()
    o3d.io.write_triangle_mesh(str(raw_path), mesh)
    write_seconds = time.perf_counter() - write_started
    post_started = time.perf_counter()
    mesh = _post_process_mesh(mesh, clusters_to_keep=clusters_to_keep)
    post_seconds = time.perf_counter() - post_started
    write_started = time.perf_counter()
    o3d.io.write_triangle_mesh(str(post_path), mesh)
    write_seconds += time.perf_counter() - write_started
    torch.cuda.synchronize()
    finished = time.perf_counter()
    telemetry = {
        "checkpoint": str(checkpoint.resolve()), "iteration": iteration,
        "depth_mode": depth_mode, "alpha_threshold": alpha_threshold,
        "voxel_size": voxel_size, "sdf_trunc": sdf_trunc, "depth_trunc": depth_trunc,
        "views": len(cameras), "valid_depth_fraction": valid_pixels / total_pixels,
        "extraction_seconds": time.perf_counter() - started,
        "load_seconds": loaded - started, "render_seconds": render_seconds,
        "fusion_polygonization_seconds": fusion_seconds, "postprocess_seconds": post_seconds,
        "write_seconds": write_seconds, "preview_seconds": preview_seconds,
        "mesh_seconds": finished - loaded - preview_seconds,
        "total_seconds": finished - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "vertices": len(mesh.vertices), "triangles": len(mesh.triangles),
        "gpu": torch.cuda.get_device_name(),
    }
    (output_dir / "extraction.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    return raw_path, post_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse Hexel surface depth into a DTU or T&T mesh.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--output_model_path", type=Path)
    parser.add_argument("--voxel_size", type=float, default=0.004,
                        help="TSDF voxel size; use 0.006 for T&T Courthouse and Meetingroom")
    parser.add_argument("--sdf_trunc", type=float, default=0.016,
                        help="TSDF truncation; use 0.024 for T&T Courthouse and Meetingroom")
    parser.add_argument("--depth_trunc", type=float, default=3.0,
                        help="Maximum fused depth; use 4.5 for T&T Courthouse and Meetingroom")
    parser.add_argument("--num_cluster", type=int, default=1)
    parser.add_argument("--depth_mode", choices=("expected", "median"), default="median",
                        help="median matches the paper's surface results; expected is the older staged protocol")
    parser.add_argument("--alpha_threshold", type=float, default=0.0)
    parser.add_argument("--preview_views", type=int, default=0)
    args = parser.parse_args()
    raw_path, post_path = extract_mesh(
        checkpoint=args.checkpoint,
        model_path=args.model_path,
        output_model_path=args.output_model_path,
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        clusters_to_keep=args.num_cluster,
        depth_mode=args.depth_mode,
        alpha_threshold=args.alpha_threshold,
        preview_views=args.preview_views,
    )
    LOGGER.info("Wrote raw mesh %s and filtered mesh %s", raw_path, post_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
