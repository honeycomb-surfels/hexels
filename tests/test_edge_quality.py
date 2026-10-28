"""Training-only visibility initialization checks for the Apple quality recipe."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "hexel_edge/tools"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
import edge_bundle as edge
from quality_initialization import bilinear, visible_parameters


def test_bilinear_samples_preserve_pixel_centers_and_linear_fields() -> None:
    image = np.array([[[0, 0, 0], [200, 0, 0]], [[0, 100, 0], [200, 100, 0]]], dtype=np.uint8)
    values = bilinear(image, np.array([[0., 0.], [.5, .5], [1., 1.]]))
    np.testing.assert_allclose(values * 255, [[0, 0, 0], [100, 50, 0], [200, 100, 0]])


def test_opposite_surface_color_does_not_contaminate_facing_seed() -> None:
    points = np.array([[-.02, -.02, 2], [.02, -.02, 2], [-.02, .02, 2], [.02, .02, 2.]])
    normals = np.tile([0., 0., -1.], (4, 1))
    parameters = edge.pack_parameters(points, normals, .1, np.full((4, 3), .5))
    k = np.array([[30., 0, 15.5], [0, 30., 15.5], [0, 0, 1.]])
    front = np.eye(4)
    back = np.diag([-1., 1, -1, 1])
    back[2, 3] = 4
    frames = [edge.Frame("front", Path("unused"), 32, 32, k, front),
              edge.Frame("back", Path("unused"), 32, 32, k, back)]
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    blue = np.zeros_like(red)
    blue[:, :, 2] = 255
    alphas = [np.ones((32, 32)), np.ones((32, 32))]
    actual = visible_parameters(parameters, normals, frames, [red, blue], alphas, sharp=True)
    np.testing.assert_allclose(actual[:, 9:12], np.tile([1, 0, 0], (4, 1)), atol=1e-6)
    np.testing.assert_allclose(actual[:, 12:18], 0, atol=1e-6)
    np.testing.assert_array_equal(actual[:, :3], parameters[:, :3])


def test_smoothed_hull_normals_do_not_change_seed_positions() -> None:
    k = np.array([[10., 0, 15.5], [0, 10., 15.5], [0, 0, 1.]])
    pose = np.eye(4)
    pose[2, 3] = -3
    frame = edge.Frame("training", Path("unused"), 32, 32, k, pose)
    mask = np.ones((32, 32))
    original = edge.visual_hull([frame], [mask], 1000, 12)
    smoothed = edge.visual_hull([frame], [mask], 1000, 12, smooth_normals=True)
    np.testing.assert_array_equal(original[0], smoothed[0])
    np.testing.assert_allclose(np.linalg.norm(smoothed[1], axis=1), 1, atol=1e-12)
    assert original[2] == smoothed[2]


def test_calibrated_tracks_recover_known_surface_and_reject_bad_geometry(monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from calibrated_initialization import calibrated_parameters
    rng = np.random.default_rng(4)
    points = np.column_stack((rng.uniform(-.5, .5, (40, 2)), np.zeros(40)))
    k = np.array([[100., 0, 63.5], [0, 100., 63.5], [0, 0, 1.]])
    frames, features = [], []
    for view, x in enumerate((-.4, 0., .4)):
        pose = np.eye(4)
        pose[:3, 3] = [x, 0, -3]
        frame = edge.Frame(str(view), Path("unused"), 128, 128, k, pose)
        xy = edge.project(points, frame)[0]
        if view == 2:
            xy[0, 0] += 12  # Three-view reprojection inconsistency.
        features.append([cv2.KeyPoint(float(u), float(v), 2) for u, v in xy])
        frames.append(frame)
    class Detector:
        index = 0
        def detectAndCompute(self, image, mask):
            value = features[self.index]
            self.index += 1
            return value, np.eye(40, 128, dtype=np.float32)
    monkeypatch.setattr(cv2, "SIFT_create", lambda **kwargs: Detector())
    image = np.empty((128, 128, 3), dtype=np.uint8)
    image[:] = [64, 128, 192]
    actual = calibrated_parameters(frames, [image]*3, [np.ones((128, 128))]*3, 100)
    assert 32 <= len(actual) < 40
    assert np.max(abs(actual[:, 2])) < 2e-5
    assert np.min(np.linalg.norm(actual[:, None, :3]-points[None, 1:], axis=2), axis=1).max() < 2e-5
    np.testing.assert_allclose(actual[:, 9:12], np.tile(np.array([64,128,192])/255, (len(actual), 1)), atol=1e-6)
