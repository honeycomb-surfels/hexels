"""Deterministic camera-only orbits shared by all NVS representations."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError("Cannot infer an orbit from degenerate camera axes")
    return vector / length


def orbit(records: list[dict], options: dict) -> dict:
    """Fit an upright ellipse to training poses, without using predictions or GT RGB."""
    transforms = np.asarray([record["world_view_transform"] for record in records])
    camera_to_world = np.linalg.inv(transforms.transpose(0, 2, 1))
    positions = camera_to_world[:, :3, 3]
    directions = camera_to_world[:, :3, 2]
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    focus = np.linalg.lstsq(projectors.sum(0), np.einsum("nij,nj->i", projectors, positions), rcond=None)[0]
    up = normalize(-camera_to_world[:, :3, 1].mean(0))
    offset = positions - positions.mean(0)
    planar = offset - np.outer(offset @ up, up)
    _, _, axes = np.linalg.svd(planar, full_matrices=False)
    horizontal = normalize(axes[0])
    if np.dot(horizontal, planar[0]) < 0:
        horizontal = -horizontal
    vertical = normalize(np.cross(up, horizontal))
    radii = np.percentile(np.abs(planar @ np.stack([horizontal, vertical]).T), 90, axis=0)
    radii = np.maximum(radii, np.max(radii) * 0.25)
    if np.min(radii) <= 0:
        raise ValueError("A fly-around requires spatially separated cameras")
    width = int(np.median([record["width"] for record in records]))
    height = int(np.median([record["height"] for record in records]))
    scale = min(1.0, options["max_width"] / width, options["max_height"] / height)
    width, height = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    center = focus + up * np.median((positions - focus) @ up)
    frames = []
    for angle in np.linspace(0, 2 * np.pi, options["frames"], endpoint=False):
        position = center + horizontal * radii[0] * np.cos(angle) + vertical * radii[1] * np.sin(angle)
        frames.append({"position": position.tolist(), "forward": normalize(focus - position).tolist(),
                       "up": up.tolist(), "width": width, "height": height,
                       "fov_y": float(np.median([record["fov_y"] for record in records]))})
    return {"version": 1, "source": "training_camera_geometry_only", "fps": options["fps"],
            "codec": options["codec"], "crf": options["crf"], "frames": frames,
            "limitation": "360-degree extrapolation can expose uncaptured regions, especially on DTU"}


def make_camera(frame: dict, device: object) -> SimpleNamespace:
    import torch

    position = np.asarray(frame["position"])
    forward = normalize(np.asarray(frame["forward"]))
    right = normalize(np.cross(forward, frame["up"]))
    down = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack([right, down, forward])
    view[:3, 3] = -view[:3, :3] @ position
    tan_y = math.tan(frame["fov_y"] / 2)
    tan_x = tan_y * frame["width"] / frame["height"]
    projection = np.zeros((4, 4), dtype=np.float32)
    near, far = 0.01, 100.0
    projection[0, 0], projection[1, 1] = 1 / tan_x, 1 / tan_y
    projection[2, 2], projection[2, 3], projection[3, 2] = far / (far - near), -far * near / (far - near), 1
    view_tensor = torch.tensor(view.T.copy(), device=device)
    projection_tensor = torch.tensor(projection.T.copy(), device=device)
    return SimpleNamespace(image_width=frame["width"], image_height=frame["height"],
        FoVx=2 * math.atan(tan_x), FoVy=frame["fov_y"], znear=near, zfar=far,
        world_view_transform=view_tensor, projection_matrix=projection_tensor,
        full_proj_transform=view_tensor @ projection_tensor,
        camera_center=torch.tensor(position, device=device, dtype=torch.float32), image_name="orbit")
