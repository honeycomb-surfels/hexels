"""Independent 2D Hexel checks for AA, appearance and selected-hit gradients."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

pytest.importorskip("gsplat_hexels")
from gsplat_hexels import rasterize_hexels_affine_sum, rasterize_hexels_sum

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _parameters(affine: bool, anisotropic: bool) -> dict[str, Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(39)
    values = {
        "xys": torch.tensor(
            [[13.2, 15.1], [17.3, 14.2], [15.4, 18.3], [18.2, 17.4]], device="cuda"
        ),
        "colors": torch.rand((4, 3), generator=generator, device="cuda"),
        "color_grads": torch.rand((4, 2, 3), generator=generator, device="cuda") * 0.2
        - 0.1,
        "sharpness": torch.tensor([[0.1], [0.15], [0.05], [0.2]], device="cuda"),
        "emission": torch.rand((4, 3), generator=generator, device="cuda") * 0.2,
        "softness": torch.full((4, 3) if anisotropic else (4,), 8.0, device="cuda"),
    }
    if anisotropic:
        values["softness"][:, 1] = 12.0
        values["softness"][:, 2] = 5.0
    if affine:
        values["affines"] = torch.tensor(
            [[[0.14, 0.03], [-0.02, 0.17]]] * 4, device="cuda"
        )
    else:
        values["scales"] = torch.tensor(
            [[7.0, 6.0], [6.0, 8.0], [8.0, 5.0], [6.0, 7.0]], device="cuda"
        )
        values["rotations"] = torch.tensor(
            [[0.15], [-0.2], [0.3], [0.6]], device="cuda"
        )
    return {name: value.requires_grad_() for name, value in values.items()}


def _render(values: dict[str, Tensor], affine: bool, aa: float) -> tuple[Tensor, ...]:
    function = rasterize_hexels_affine_sum if affine else rasterize_hexels_sum
    return function(
        **values,
        img_height=32,
        img_width=32,
        aa_kernel_size=aa,
        return_aux=True,
        return_topk=True,
    )


def _reference(values: dict[str, Tensor], affine: bool, aa: float) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(32, device="cuda"), torch.arange(32, device="cuda"), indexing="ij"
    )
    pixels = torch.stack((x, y), dim=-1)
    delta = values["xys"][:, None, None, :] - pixels[None]
    if affine:
        matrix = values["affines"]
        coordinates = torch.einsum("nhwi,nij->nhwj", delta, matrix)
        aa_scale = torch.rsqrt(1 + aa * matrix.square().sum(dim=1))
    else:
        angle = values["rotations"][:, 0, None, None]
        sx, sy = (values["scales"] + 1e-8).unbind(-1)
        u = (delta[..., 0] * angle.cos() + delta[..., 1] * angle.sin()) / sx[
            :, None, None
        ]
        v = (-delta[..., 0] * angle.sin() + delta[..., 1] * angle.cos()) / sy[
            :, None, None
        ]
        coordinates = torch.stack((u, v), dim=-1)
        aa_scale = torch.rsqrt(1 + aa / (values["scales"] + 1e-8).square())
    coordinates = coordinates * aa_scale[:, None, None, :]
    u, v = coordinates.unbind(-1)
    distances = torch.stack(
        (
            v.abs(),
            (math.sqrt(3) / 2 * u + 0.5 * v).abs(),
            (math.sqrt(3) / 2 * u - 0.5 * v).abs(),
        ),
        dim=-1,
    )
    distance, edge = distances.max(dim=-1)
    softness = values["softness"]
    if softness.ndim == 1:
        softness = softness[:, None].expand(-1, 3)
    selected = (
        softness[:, None, None, :]
        .expand(-1, 32, 32, -1)
        .gather(-1, edge[..., None])[..., 0]
    )
    peak = torch.exp(-values["sharpness"][:, :, None] * coordinates.square().sum(-1))
    alpha = torch.sigmoid(-(distance - 1) * selected) * peak
    weights = alpha / (alpha.sum(dim=0) + 1e-4)
    colors = values["colors"][:, None, None, :] + 0.1 * torch.einsum(
        "nhwi,nic->nhwc", delta, values["color_grads"]
    )
    return (
        colors * weights[..., None]
        + values["emission"][:, None, None, :] * peak[..., None]
    ).sum(dim=0)


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("aa", [0.0, 0.3])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_forward_and_gradients_match_reference(
    affine: bool, aa: float, anisotropic: bool
) -> None:
    values = _parameters(affine, anisotropic)
    image, _, _, _ = _render(values, affine, aa)
    reference = _reference(values, affine, aa)
    # Interior pixels avoid the existing finite tile-support truncation.
    torch.testing.assert_close(
        image[8:24, 8:24], reference[8:24, 8:24], rtol=2e-5, atol=2e-6
    )
    weight = torch.linspace(-1.0, 1.0, 16 * 16 * 3, device="cuda").reshape(16, 16, 3)
    actual_gradients = torch.autograd.grad(
        (image[8:24, 8:24] * weight).sum(), tuple(values.values()), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        (reference[8:24, 8:24] * weight).sum(), tuple(values.values())
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("affine", [False, True])
def test_selected_geometry_finite_difference(affine: bool) -> None:
    values = _parameters(affine, True)
    image = _render(values, affine, 0.3)[0]
    loss = image[8:24, 8:24].square().mean()
    loss.backward()
    for name in ("xys", "affines" if affine else "scales", "softness", "sharpness"):
        parameter = values[name]
        direction = torch.zeros_like(parameter)
        direction.reshape(-1)[0] = 1.0
        analytical = float((parameter.grad * direction).sum())
        epsilon = 1e-5
        with torch.no_grad():
            reference_values = {
                key: value.detach().double() for key, value in values.items()
            }
            plus = dict(
                reference_values, **{name: reference_values[name] + epsilon * direction}
            )
            minus = dict(
                reference_values, **{name: reference_values[name] - epsilon * direction}
            )
            high = _reference(plus, affine, 0.3)[8:24, 8:24].square().mean()
            low = _reference(minus, affine, 0.3)[8:24, 8:24].square().mean()
        numerical = float((high - low) / (2 * epsilon))
        assert analytical == pytest.approx(numerical, rel=0.02, abs=2e-5)
