"""Object-centered preview paths for partial-dome camera captures such as DTU."""

from __future__ import annotations

import math

import numpy as np

from scripts.nvs_trajectory import normalize


def object_orbit(poses: list[dict], reference: dict, margin: float = 2.0) -> dict:
    """Keep observed camera standoff instead of interpreting capture spread as radius."""
    if len(poses) < 3 or not math.isfinite(margin) or margin < 1:
        raise ValueError("Object orbits need at least three cameras and a margin >= 1")
    positions = np.asarray([pose["position"] for pose in poses], dtype=np.float64)
    forwards = np.asarray([normalize(np.asarray(pose["forward"])) for pose in poses])
    camera_ups = np.asarray([normalize(np.asarray(pose["up"])) for pose in poses])
    projectors = np.eye(3) - forwards[:, :, None] * forwards[:, None, :]
    focus, _, rank, _ = np.linalg.lstsq(
        projectors.sum(0), np.einsum("nij,nj->i", projectors, positions), rcond=None)
    if rank != 3:
        raise ValueError("Camera rays do not constrain an object focus")
    rights = np.asarray([normalize(right) for right in np.cross(forwards, camera_ups)])
    _, singular_values, axes = np.linalg.svd(rights, full_matrices=False)
    if singular_values[1] < singular_values[0] * 1e-6:
        raise ValueError("Camera right axes do not constrain a shared upright direction")
    up = axes[-1]
    if up @ camera_ups.mean(0) < 0:
        up = -up
    offsets = positions - focus
    distances = np.linalg.norm(offsets, axis=1)
    elevations = np.arcsin(np.clip(offsets @ up / distances, -1, 1))
    elevation = float(np.median(elevations))
    start = offsets[np.argmin(np.abs(elevations - elevation))]
    horizontal = normalize(start - up * (start @ up))
    tangent = normalize(np.cross(up, horizontal))
    distance = float(np.quantile(distances, 0.9) * margin)
    template = reference["frames"][0]
    frames = []
    for angle in np.linspace(0, 2 * np.pi, len(reference["frames"]), endpoint=False):
        position = focus + distance * (up * math.sin(elevation) + math.cos(elevation) *
                                     (horizontal * math.cos(angle) + tangent * math.sin(angle)))
        frames.append({"position": position.tolist(), "forward": normalize(focus - position).tolist(),
                       "up": up.tolist(), "width": template["width"], "height": template["height"],
                       "fov_y": template["fov_y"]})
    return {"version": 2, "source": "training_camera_geometry_only_object_standoff",
            "fps": reference["fps"], "codec": reference["codec"], "crf": reference["crf"], "frames": frames,
            "framing": {"focus": focus.tolist(), "up": up.tolist(), "distance": distance, "margin": margin,
                        "reference_distance_quantile": 0.9, "elevation_degrees": math.degrees(elevation),
                        "training_distance_range": [float(distances.min()), float(distances.max())],
                        "upright_singular_values": singular_values.tolist()},
            "limitation": "Full 360-degree preview still exposes uncaptured surfaces; framing does not repair geometry"}
