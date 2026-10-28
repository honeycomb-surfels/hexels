"""Independent signed-scale and log-scale checks for the flat 2D control."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("gsplat_hexels")
from gsplat_hexels import rasterize_hexels_sum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("log_scale", [False, True])
def test_flat_hexel_scale_gradients_match_reference(sign: float, log_scale: bool) -> None:
    centers = torch.tensor([[13.2, 15.1], [17.3, 14.2], [15.4, 18.3]], device="cuda",
                           requires_grad=True)
    initial = torch.tensor([[7.0, 6.0], [6.0, 8.0], [8.0, 5.0]], device="cuda")
    raw = (initial.log() if log_scale else initial).requires_grad_()
    scales = (raw.exp() if log_scale else raw) * raw.new_tensor([sign, 1.0])
    angles = torch.tensor([[0.15], [-0.2], [0.3]], device="cuda", requires_grad=True)
    colors = torch.tensor([[0.1, 0.3, 0.9], [0.8, 0.2, 0.4], [0.3, 0.7, 0.2]],
                          device="cuda", requires_grad=True)
    image = rasterize_hexels_sum(centers, scales, angles, colors, 32, 32, softness=10.0)
    y, x = torch.meshgrid(torch.arange(32, device="cuda"), torch.arange(32, device="cuda"),
                          indexing="ij")
    delta = centers[:, None, None] - torch.stack((x, y), dim=-1)[None]
    angle = angles[:, 0, None, None]
    sx, sy = (scales + 1e-8).unbind(-1)
    u = (delta[..., 0] * angle.cos() + delta[..., 1] * angle.sin()) / sx[:, None, None]
    v = (-delta[..., 0] * angle.sin() + delta[..., 1] * angle.cos()) / sy[:, None, None]
    distance = torch.stack((v.abs(), (math.sqrt(3) / 2 * u + v / 2).abs(),
                            (math.sqrt(3) / 2 * u - v / 2).abs()), dim=-1).amax(-1) - 1
    alpha = torch.sigmoid(-10 * distance)
    weights = alpha / (alpha.sum(0) + 1e-4)
    reference = (weights[..., None] * colors[:, None, None]).sum(0)
    torch.testing.assert_close(image[8:24, 8:24], reference[8:24, 8:24], rtol=2e-5, atol=2e-6)
    parameters = (centers, raw, angles, colors)
    actual = torch.autograd.grad(image[8:24, 8:24].square().mean(), parameters, retain_graph=True)
    expected = torch.autograd.grad(reference[8:24, 8:24].square().mean(), parameters)
    for found, wanted in zip(actual, expected):
        torch.testing.assert_close(found, wanted, rtol=2e-4, atol=2e-5)
