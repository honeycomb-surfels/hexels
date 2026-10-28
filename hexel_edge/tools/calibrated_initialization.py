"""Training-only SIFT-track triangulation with supplied pinhole calibration.

This recovers sparse observed surfaces rather than inventing visual-hull depth.
It does not estimate camera poses and does not use held-out observations.
"""

from __future__ import annotations

import logging
import numpy as np

from edge_bundle import Frame, pack_parameters, point_geometry
from quality_initialization import bilinear


LOGGER = logging.getLogger(__name__)


def calibrated_parameters(frames: list[Frame], images: list[np.ndarray],
                          alphas: list[np.ndarray], limit: int) -> np.ndarray:
    """Triangulate bounded, three-or-more-view geometrically checked tracks."""
    import cv2
    from scipy.spatial import cKDTree

    if not 3 <= len(frames) <= 64 or len(images) != len(frames) or len(alphas) != len(frames):
        raise ValueError("Calibrated initialization needs 3..64 matching training views")
    cv2.setNumThreads(2)
    cv2.setRNGSeed(0)
    feature_budget = min(12000, max(5000, max(image.shape[0]*image.shape[1] for image in images)//64))
    detector = cv2.SIFT_create(nfeatures=feature_budget, contrastThreshold=.02)
    keypoints, descriptors, offsets = [], [], [0]
    for image, alpha in zip(images, alphas, strict=True):
        mask = np.asarray(alpha > .8, dtype=np.uint8) * 255
        points, descriptor = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), mask)
        if descriptor is None or len(points) < 4:
            raise ValueError("A training image lacks enough SIFT features")
        keypoints.append(np.asarray([p.pt for p in points], dtype=np.float64))
        descriptors.append(descriptor)
        offsets.append(offsets[-1] + len(points))
    parent = np.arange(offsets[-1])

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    centers = np.stack([f.camera_to_world[:3, 3] for f in frames])
    distances = np.linalg.norm(centers[:, None] - centers[None], axis=2)
    pairs = {tuple(sorted((i, int(j)))) for i in range(len(frames))
             for j in np.argsort(distances[i])[1:5]}
    rotations = [f.camera_to_world[:3, :3].T for f in frames]
    projections = [f.intrinsics @ np.linalg.inv(f.camera_to_world)[:3] for f in frames]
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    accepted = 0
    for i, j in sorted(pairs):
        matches = matcher.knnMatch(descriptors[i], descriptors[j], k=2)
        matches = [a for a, b in matches if a.distance < .75 * b.distance]
        if not matches:
            continue
        first = np.column_stack((keypoints[i][[m.queryIdx for m in matches]], np.ones(len(matches))))
        second = np.column_stack((keypoints[j][[m.trainIdx for m in matches]], np.ones(len(matches))))
        translation = rotations[j] @ (centers[i] - centers[j])
        tx, ty, tz = translation
        skew = np.array([[0., -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        fundamental = np.linalg.inv(frames[j].intrinsics).T @ skew @ rotations[j] @ rotations[i].T @ np.linalg.inv(frames[i].intrinsics)
        line_j, line_i = first @ fundamental.T, second @ fundamental
        denominator = np.sqrt((line_j[:, :2]**2).sum(axis=1) + (line_i[:, :2]**2).sum(axis=1)).clip(1e-12)
        errors = np.abs((second * line_j).sum(axis=1)) / denominator
        for match, error in zip(matches, errors, strict=True):
            if error > .75:
                continue
            a, b = find(offsets[i] + match.queryIdx), find(offsets[j] + match.trainIdx)
            if a != b:
                parent[b] = a
            accepted += 1
    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for view, points in enumerate(keypoints):
        for index, xy in enumerate(points):
            tracks.setdefault(find(offsets[view] + index), []).append((view, xy))
    positions, colors = [], []
    for track in tracks.values():
        views = [v for v, _ in track]
        if len(track) < 3 or len(set(views)) != len(track):
            continue
        rows = []
        for view, xy in track:
            projection = projections[view]
            rows.extend((xy[0] * projection[2] - projection[0], xy[1] * projection[2] - projection[1]))
        _, _, vh = np.linalg.svd(rows)
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-10:
            continue
        point = homogeneous[:3] / homogeneous[3]
        if not np.isfinite(point).all() or np.any(abs(point) > 1.1):
            continue
        errors, rays = [], []
        for view, xy in track:
            projected = projections[view] @ np.append(point, 1)
            if projected[2] <= .2:
                break
            errors.append(float(np.linalg.norm(projected[:2] / projected[2] - xy)))
            ray = point - centers[view]
            rays.append(ray / np.linalg.norm(ray))
        if len(errors) != len(track) or max(errors) > 1.5 or np.max(np.linalg.norm(np.cross(rays[0], rays), axis=1)) < .02:
            continue
        positions.append(point)
        colors.append(np.mean([bilinear(images[v], xy[None])[0] for v, xy in track], axis=0))
    if len(positions) < 32:
        raise ValueError("Too few three-view calibrated tracks survived geometric checks")
    points = np.asarray(positions)
    _, unique = np.unique(np.rint(points / .003).astype(int), axis=0, return_index=True)
    unique = unique[np.linspace(0, len(unique)-1, min(limit, len(unique))).round().astype(int)]
    points, rgb = points[unique], np.asarray(colors)[unique]
    normals, _ = point_geometry(points, centers)
    nearest = cKDTree(points).query(points, k=min(9, len(points)), workers=1)[0]
    scales = (nearest[:, -1] * .65).clip(.001, .06)
    parameters = pack_parameters(points, normals, scales, rgb)
    parameters[:, 18] = 0
    parameters[:, 19] = np.log(15)
    LOGGER.info("Calibrated seeds: %d views, %d pairs, %d accepted links, %d seeds", len(frames), len(pairs), accepted, len(points))
    return parameters
