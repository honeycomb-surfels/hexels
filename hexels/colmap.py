from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .options import Hexels3DOptions


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_repo_deps_to_path() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "2d-gaussian-splatting"))


@dataclass(frozen=True)
class ColmapScene:
    point_cloud: Any
    train_cameras: list[Any]
    test_cameras: list[Any]
    cameras_extent: float


def load_colmap_scene(opt: Hexels3DOptions) -> ColmapScene:
    _add_repo_deps_to_path()

    from scene.dataset_readers import readColmapSceneInfo
    from utils.camera_utils import cameraList_from_camInfos

    scene_info = readColmapSceneInfo(opt.source_path, opt.images, opt.eval, llffhold=opt.llffhold)
    if scene_info.point_cloud is None:
        raise RuntimeError(f"Failed to load point cloud from {scene_info.ply_path}")

    args = SimpleNamespace(resolution=opt.resolution, data_device=opt.data_device)
    train_cams = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
    test_cams = cameraList_from_camInfos(scene_info.test_cameras, 1.0, args)
    cameras_extent = float(scene_info.nerf_normalization["radius"])
    return ColmapScene(
        point_cloud=scene_info.point_cloud,
        train_cameras=train_cams,
        test_cameras=test_cams,
        cameras_extent=cameras_extent,
    )
