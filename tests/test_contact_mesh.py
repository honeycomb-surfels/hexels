from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

from hexels.contact_mesh import (
    ContactOptions, find_contacts, nondegenerate_patches, primitive_polygons, rotation_matrices,
    triangulate_patches, weld_contacts,
)
from scripts.extract_contact_mesh import write_colored_mesh


def polygons(centers: list[list[float]], family: str = "hexels") -> tuple[np.ndarray, np.ndarray]:
    count = len(centers)
    return primitive_polygons(np.array(centers), np.ones((count, 2)),
                              np.tile([1.0, 0, 0, 0], (count, 1)), family=family)


def test_hexel_vertices_match_cuda_half_planes() -> None:
    corners, _ = polygons([[0, 0, 0]])
    horizontal, vertical = np.abs(corners[0, :, :2]).T
    sdf = np.maximum(vertical, 0.5 * (np.sqrt(3) * horizontal + vertical)) - 1
    np.testing.assert_allclose(sdf, 0, atol=1e-14)


def test_gaussian_polygon_matches_half_peak_contour() -> None:
    corners, _ = polygons([[0, 0, 0]], "2dgs")
    np.testing.assert_allclose(np.exp(-0.5 * np.sum(corners**2, axis=2)), 0.5)


def test_learned_subprecision_scales_are_explicitly_degenerate() -> None:
    centers = np.ones((2, 3))
    corners, _ = primitive_polygons(centers, np.array([[1.0, 1.0], [1e-24, 1e-25]]),
                                    np.tile([1.0, 0, 0, 0], (2, 1)), family="2dgs")
    np.testing.assert_array_equal(nondegenerate_patches(corners, centers), [True, False])


def test_wxyz_rotation_preserves_local_tangent_frame() -> None:
    rotation = rotation_matrices(np.array([[np.sqrt(0.5), 0, np.sqrt(0.5), 0]]))[0]
    np.testing.assert_allclose(rotation @ [0, 0, 1], [1, 0, 0], atol=1e-14)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)


@pytest.mark.parametrize("family", ["hexels", "2dgs"])
def test_adjacent_polygons_share_exactly_one_boundary(family: str) -> None:
    angle = np.pi / (6 if family == "hexels" else 12)
    radius = 2 / np.sqrt(3) if family == "hexels" else np.sqrt(2 * np.log(2))
    distance = 2 * radius * np.cos(angle)
    corners, normals = polygons([[0, 0, 0], [distance * np.cos(angle), distance * np.sin(angle), 0]], family)
    pairs, endpoints, stats = find_contacts(corners, normals, ContactOptions())
    assert len(pairs) == 1
    assert stats["candidate_isolated_primitive_fraction"] == 0
    colors = np.array([[1, 0, 0], [0, 0, 1]])
    vertices, vertex_colors, corner_ids, stats = weld_contacts(corners, colors, endpoints, max_shift_fraction=0.2)
    assert stats["welded_vertices_removed"] == 2
    assert len(set(corner_ids[0]) & set(corner_ids[1])) == 2
    vertices, vertex_colors, faces, edges = triangulate_patches(vertices, vertex_colors, corner_ids,
                                                              corners.mean(axis=1), colors)
    assert len(faces) == corners.shape[1] * 2
    assert len(edges) == corners.shape[1] * 2 - 1
    assert np.all(np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                         vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1) > 0)


@pytest.mark.parametrize("shift", [[0, 0, 0], [0, 0, 0.3], [5, 0, 0], [np.sqrt(3), 1, 0.5]])
def test_stacks_and_distant_surfaces_are_not_welded(shift: list[float]) -> None:
    corners, normals = polygons([[0, 0, 0], shift])
    pairs, _, _ = find_contacts(corners, normals, ContactOptions())
    assert len(pairs) == 0


def test_large_transitive_welds_are_rejected() -> None:
    corners, _ = polygons([[0, 0, 0], [4, 0, 0]])
    vertices, _, _, stats = weld_contacts(corners, np.ones((2, 3)), np.array([[0, 6]]), max_shift_fraction=0.2)
    assert stats["rejected_vertex_groups"] == 1
    assert stats["welded_vertices_removed"] == 0
    assert stats["max_vertex_displacement"] == 0
    assert len(vertices) == 12


def test_colored_mesh_and_wireframe_roundtrip(tmp_path: Path) -> None:
    corners, _ = polygons([[0, 0, 0]])
    colors = np.array([[0.2, 0.5, 1.0]])
    vertices, vertex_colors, faces, edges = triangulate_patches(
        corners.reshape(-1, 3), np.repeat(colors, 6, axis=0), np.arange(6)[None], np.zeros((1, 3)), colors)
    write_colored_mesh(tmp_path / "mesh.ply", vertices, vertex_colors, faces, edges)
    mesh = PlyData.read(tmp_path / "mesh.ply")
    wireframe = PlyData.read(tmp_path / "wireframe.ply")
    assert len(mesh["face"]) == 6
    assert len(wireframe["edge"]) == 6
    assert np.all(mesh["vertex"]["blue"] == 255)
    assert np.all(wireframe["edge"]["green"] == 128)


def test_export_keeps_readonly_source_and_reports_contact_quality(tmp_path: Path, monkeypatch) -> None:
    from scripts import extract_contact_mesh as exporter

    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"unchanged")
    data = {"centers": np.array([[0, 0, 0], [np.sqrt(3), 1, 0]]), "scales": np.ones((2, 2)),
            "rotations": np.tile([1, 0, 0, 0], (2, 1)), "colors": np.ones((2, 3)), "opacities": np.ones(2)}
    monkeypatch.setattr(exporter, "load_primitives", lambda *args: (data, checkpoint))
    args = argparse.Namespace(output_dir=tmp_path / "export", model_path=tmp_path, family="hexels",
                              iteration=30000, mode="contact", workers=1, opacity_threshold=0.1, gaussian_sides=12)
    result = exporter.extract(args)
    assert result["diagnostics"]["welded_contacts"] == 1
    assert result["diagnostics"]["isolated_primitive_fraction"] == 0
    assert result["total_seconds"] >= result["extraction_seconds"] > 0
    assert checkpoint.read_bytes() == b"unchanged"
    with pytest.raises(FileExistsError):
        exporter.extract(args)
