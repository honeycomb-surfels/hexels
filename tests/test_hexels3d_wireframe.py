from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData

from hexels.wireframe import HexelWireframe, extract_hexel_wireframe
from scripts.extract_hexels_wireframe import extract_checkpoint_wireframe, write_wireframe_ply


def _identity_rotations(count: int) -> torch.Tensor:
    rotations = torch.zeros((count, 4), dtype=torch.float64)
    rotations[:, 0] = 1.0
    return rotations


def test_wireframe_edges_follow_canonical_hexel_sides() -> None:
    wireframe = extract_hexel_wireframe(
        torch.zeros((1, 3), dtype=torch.float64),
        torch.zeros((1, 2), dtype=torch.float64),
        _identity_rotations(1),
    )

    assert torch.equal(
        wireframe.edges,
        torch.tensor([[0, 1], [2, 5], [0, 4], [2, 3], [3, 4], [1, 5]]),
    )
    assert torch.equal(wireframe.primitive_ids, torch.zeros(6, dtype=torch.long))
    assert torch.equal(wireframe.side_ids, torch.arange(6))
    assert torch.equal(wireframe.faces, torch.tensor([[0, 4, 3, 2, 5, 1]]))
    assert wireframe.vertex_colors is None
    assert torch.allclose(
        wireframe.vertices[0],
        torch.tensor([2.0 / math.sqrt(3.0), 0.0, 0.0], dtype=torch.float64),
    )


def test_wireframe_preserves_original_ids_after_active_filtering() -> None:
    means3d = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    wireframe = extract_hexel_wireframe(
        means3d,
        torch.zeros((3, 2), dtype=torch.float64),
        _identity_rotations(3),
        active_mask=torch.tensor([True, False, True]),
    )

    assert wireframe.vertices.shape == (12, 3)
    assert torch.equal(
        wireframe.primitive_ids,
        torch.tensor([0] * 6 + [2] * 6),
    )
    assert torch.equal(wireframe.edges[6], torch.tensor([6, 7]))


def test_wireframe_geometry_is_differentiable() -> None:
    means3d = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    log_scales = torch.tensor([[0.2, -0.1]], requires_grad=True)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)

    wireframe = extract_hexel_wireframe(means3d, log_scales, rotations)
    loss = wireframe.vertices[0, 0] + wireframe.vertices[0, 1]
    loss.backward()

    assert means3d.grad is not None
    assert log_scales.grad is not None
    assert rotations.grad is not None
    assert torch.count_nonzero(rotations.grad) > 0


def test_wireframe_spatial_color_is_exact_and_differentiable() -> None:
    colors = torch.tensor([[0.25, 0.5, 0.75]], requires_grad=True)
    color_grads = torch.tensor(
        [[[0.3, 0.0, -0.3], [0.0, 0.2, 0.0]]],
        requires_grad=True,
    )
    wireframe = extract_hexel_wireframe(
        torch.zeros((1, 3)),
        torch.zeros((1, 2)),
        _identity_rotations(1).float(),
        colors=colors,
        color_grads=color_grads,
    )

    assert wireframe.vertex_colors is not None
    assert torch.allclose(
        wireframe.vertex_colors[0],
        colors[0] + (2.0 / math.sqrt(3.0)) * color_grads[0, 0],
    )
    wireframe.vertex_colors[0].sum().backward()
    assert colors.grad is not None
    assert color_grads.grad is not None
    assert torch.count_nonzero(color_grads.grad) > 0


def test_checkpoint_export_filters_and_writes_indexed_ply(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "ckpt.pt"
    output_path = tmp_path / "wireframe.ply"
    torch.save(
        {
            "model": {
                "means3d": torch.zeros((3, 3)),
                "log_scales": torch.zeros((3, 2)),
                "rotation_quats": _identity_rotations(3).float(),
                "opacity_logits": torch.tensor([2.0, 2.0, -2.0]),
                "colors": torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                "color_grads": torch.zeros((3, 2, 3)),
                "active_mask": torch.tensor([True, False, True]),
                "child_parent_idx": torch.empty(0, dtype=torch.long),
            }
        },
        checkpoint_path,
    )

    wireframe = extract_checkpoint_wireframe(
        checkpoint_path,
        output_path,
        scale_modifier=1.0,
        min_opacity=0.5,
        include_inactive=False,
        device=torch.device("cpu"),
        color_mode="spatial",
    )
    ply = PlyData.read(output_path)

    assert len(wireframe.plane_normals) == 1
    assert len(ply["vertex"]) == 6
    assert len(ply["edge"]) == 6
    assert len(ply["face"]) == 1
    assert set(ply["edge"]["primitive_id"]) == {0}
    assert list(ply["edge"]["side_id"]) == list(range(6))
    assert set(ply["vertex"]["red"]) == {255}
    assert set(ply["vertex"]["green"]) == {0}
    assert set(ply["vertex"]["blue"]) == {0}
    assert set(ply["face"]["primitive_id"]) == {0}


def test_export_matches_independent_numpy_sdf_zero_geometry() -> None:
    from hexels.contact_mesh import primitive_polygons

    generator = torch.Generator().manual_seed(37)
    means = torch.randn((7, 3), generator=generator, dtype=torch.float64)
    logs = torch.randn((7, 2), generator=generator, dtype=torch.float64) * 0.3
    rotations = torch.randn((7, 4), generator=generator, dtype=torch.float64)
    wireframe = extract_hexel_wireframe(means, logs, rotations, scale_modifier=0.7)
    independent, normals = primitive_polygons(
        means.numpy(), 0.7 * logs.exp().numpy(), rotations.numpy(), family="hexels",
    )
    cycle = [0, 4, 3, 2, 5, 1]
    np.testing.assert_allclose(wireframe.vertices.reshape(-1, 6, 3).numpy()[:, cycle],
                               independent, atol=1e-14)
    np.testing.assert_allclose(wireframe.plane_normals.numpy(), normals, atol=1e-14)


def test_selected_geometry_and_spatial_color_pass_finite_differences() -> None:
    generator = torch.Generator().manual_seed(65)
    inputs = tuple(tensor.requires_grad_() for tensor in (
        torch.randn((2, 3), generator=generator, dtype=torch.float64),
        torch.randn((2, 2), generator=generator, dtype=torch.float64) * 0.2,
        torch.randn((2, 4), generator=generator, dtype=torch.float64),
        torch.randn((2, 3), generator=generator, dtype=torch.float64),
        torch.randn((2, 2, 3), generator=generator, dtype=torch.float64),
    ))

    def converted(means, scales, rotations, colors, gradients):
        output = extract_hexel_wireframe(means, scales, rotations,
                                        colors=colors, color_grads=gradients)
        return output.vertices, output.vertex_colors

    assert torch.autograd.gradcheck(converted, inputs, eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("sides", [6, 12])
def test_face_parent_ids_support_variable_polygon_sizes(tmp_path: Path, sides: int) -> None:
    faces = torch.arange(2 * sides).reshape(2, sides)
    edges = torch.stack([faces, faces.roll(-1, dims=1)], dim=-1).reshape(-1, 2)
    wireframe = HexelWireframe(
        vertices=torch.zeros((2 * sides, 3)), edges=edges, faces=faces,
        primitive_ids=torch.tensor([7, 91]).repeat_interleave(sides),
        side_ids=torch.arange(sides).repeat(2), plane_normals=torch.zeros((2, 3)),
        vertex_colors=torch.ones((2 * sides, 3)),
    )
    output = tmp_path / "cells.ply"
    write_wireframe_ply(output, wireframe)
    data = PlyData.read(output)
    np.testing.assert_array_equal(data["face"]["primitive_id"], [7, 91])
    assert all(len(face) == sides for face in data["face"]["vertex_indices"])
