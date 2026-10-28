"""Bounded training-only visibility/appearance initialization for Hexel seeds."""

from __future__ import annotations

import numpy as np

from edge_bundle import Frame, project, quaternion_matrix


def bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample RGB at calibrated zero-based centers, clamped to image edges."""
    height, width = image.shape[:2]
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1-dy) * ((1-dx)*image[y0, x0] + dx*image[y0, x1])
            + dy * ((1-dx)*image[y1, x0] + dx*image[y1, x1])) / 255


def visible_parameters(parameters: np.ndarray, normals: np.ndarray,
                       frames: list[Frame], images: list[np.ndarray],
                       alphas: list[np.ndarray], *, sharp: bool = False) -> np.ndarray:
    """Initialize from at most two front-facing, approximately visible views.

    Uses selected training observations only. A coarse depth buffer rejects
    obvious opposite surfaces; this is an initializer, not a dense MVS claim.
    """
    count = len(parameters)
    if not (len(frames) == len(images) == len(alphas)) or count < 4:
        raise ValueError("Need matching training observations and at least four seeds")
    result = parameters.copy()
    points = result[:, :3].astype(np.float64)
    weights = np.zeros((count, len(frames)))
    fallback = np.zeros_like(weights)
    colors = np.zeros((count, len(frames), 3))
    scales = np.exp(result[:, 3:5].astype(np.float64))
    tolerance = float(np.median(scales)) * 2.5
    for view, (frame, image, alpha) in enumerate(zip(frames, images, alphas, strict=True)):
        xy, depth = project(points, frame)
        height, width = image.shape[:2]
        uv = np.rint(xy).astype(int)
        valid = ((depth > .2) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < height))
        ids = np.flatnonzero(valid)
        valid[ids] &= alpha[uv[ids, 1], uv[ids, 0]] > .5
        direction = frame.camera_to_world[:3, 3] - points
        direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-12)
        facing = (normals * direction).sum(axis=1).clip(0, 1)
        fallback[:, view] = valid * (.01 + facing**4)
        cells_x = (width + 7) // 8
        cells = (np.clip(uv[:, 1], 0, height-1)//8) * cells_x + np.clip(uv[:, 0], 0, width-1)//8
        minimum = np.full(cells_x * ((height + 7)//8), np.inf)
        np.minimum.at(minimum, cells[valid], depth[valid])
        visible = valid & (depth <= minimum[cells] + tolerance) & (facing > .05)
        weights[:, view] = visible * facing**4
        colors[:, view] = bilinear(image, xy)
    unsupported = weights.sum(axis=1) == 0
    weights[unsupported] = fallback[unsupported]
    if np.any(weights.sum(axis=1) == 0):
        raise ValueError("A seed is not observed by any selected training image")
    # Keep the best two views, avoiding unobserved/backside color contamination.
    top = np.argsort(weights, axis=1)[:, -2:]
    selected = np.zeros_like(weights)
    np.put_along_axis(selected, top, np.take_along_axis(weights, top, axis=1), axis=1)
    selected /= selected.sum(axis=1, keepdims=True)
    result[:, 9:12] = (colors * selected[:, :, None]).sum(axis=1)
    result[:, 18] = np.log(.5/.5)
    if sharp:
        result[:, 3:5] += np.log(.7)
        matrices = np.stack([quaternion_matrix(q) for q in result[:, 5:9]])
        for axis, slot in ((0, 12), (1, 15)):
            offset = matrices[:, :, axis] * np.exp(result[:, 3+axis:4+axis]) * .5
            gradient = np.zeros((count, 3))
            for view, (frame, image) in enumerate(zip(frames, images, strict=True)):
                plus = bilinear(image, project(points + offset, frame)[0])
                minus = bilinear(image, project(points - offset, frame)[0])
                gradient += selected[:, view, None] * (plus - minus)
            result[:, slot:slot+3] = gradient.clip(-.5, .5)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite visibility initialization")
    return result
