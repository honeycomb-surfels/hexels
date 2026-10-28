"""Independent ray/plane checks for support clipping and cached RGB gradients."""
from __future__ import annotations

import math

import pytest
import torch

from gsplat import rasterize_hexels3d

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


def single_plane(angle: float, opacity: float, softness: tuple[float, ...], z: float = 1.0) -> tuple[dict, dict]:
    """A plane tilted toward +Z, with a camera independent of the CUDA transforms."""
    values = {
        'means3d': torch.tensor([[0.03, -0.02, z]], device='cuda', requires_grad=True),
        'scales': torch.tensor([[1.4, 0.7]], device='cuda', requires_grad=True),
        'colors': torch.tensor([[0.8, 0.5, 0.2]], device='cuda', requires_grad=True),
        'opacities': torch.tensor([opacity], device='cuda', requires_grad=True),
        'softnesses': torch.tensor([softness] if len(softness) == 3 else softness, device='cuda', requires_grad=True),
    }
    projection = torch.zeros((4, 4), device='cuda')
    projection[0, 0] = projection[1, 1] = 1.0
    projection[3, 2] = 1.0
    projection[2, 2], projection[2, 3] = 100 / 99.8, -20 / 99.8
    arguments = dict(
        rotations=torch.tensor([[math.cos(angle / 2), 0., -math.sin(angle / 2), 0.]], device='cuda'),
        color_grads=torch.empty(0, device='cuda'), view_grads=torch.empty(0, device='cuda'),
        viewmatrix=torch.eye(4, device='cuda'), projmatrix=projection.T.contiguous(),
        campos=torch.zeros(3, device='cuda'), background=torch.tensor([0.1, 0.2, 0.3], device='cuda'),
        image_width=197, image_height=149, tan_fovx=1., tan_fovy=1., softness=10.,
        color_model='flat', view_model='none', use_view_grads=False,
    )
    return values, arguments


def plane_reference(values: dict, arguments: dict, angle: float) -> torch.Tensor:
    """Intersect camera rays with z=mz+tan(angle)*(x-mx), then apply the SDF."""
    width, height = arguments['image_width'], arguments['image_height']
    yy, xx = torch.meshgrid(torch.arange(height, device='cuda', dtype=torch.float64),
                            torch.arange(width, device='cuda', dtype=torch.float64), indexing='ij')
    rx = (xx - (width - 1) / 2) / (width / 2)
    ry = (yy - (height - 1) / 2) / (height / 2)
    mx, my, mz = values['means3d'][0]
    depth = (mz - math.tan(angle) * mx) / (1 - math.tan(angle) * rx)
    u = (rx * depth - mx) / (math.cos(angle) * values['scales'][0, 0])
    v = (ry * depth - my) / values['scales'][0, 1]
    d0, d1, d2 = v.abs(), (math.sqrt(3) / 2 * u + 0.5 * v).abs(), (math.sqrt(3) / 2 * u - 0.5 * v).abs()
    edge = torch.where(d0 > torch.maximum(d1, d2), 0, torch.where(d2 >= d1, 2, 1))
    distances = torch.stack((d0, d1, d2))
    distance = distances.gather(0, edge[None])[0] - 1
    softness = values['softnesses'].flatten()
    effective = softness[edge] if softness.numel() == 3 else softness[0]
    alpha = (values['opacities'][0] * torch.sigmoid(-effective * distance)).clamp_max(0.99)
    alpha = torch.where((depth >= 0.2) & (alpha >= 1 / 255), alpha, 0.)
    return alpha[None] * values['colors'][0, :, None, None] + (1 - alpha[None]) * arguments['background'][:, None, None]


@pytest.mark.parametrize('angle,z,opacity,softness', [
    (45., 1., .8, (10.,)), (75., .3, .8, (.4,)),
    (35., .7, .8, (.7, 3., 12.)), (65., 1., .006, (.7, 3., 20.)),
    (25., 3., .004, (20., 30., 40.)), (45., 1., .001, (10.,)),
])
def test_clipped_support_matches_independent_plane(angle: float, z: float, opacity: float, softness: tuple) -> None:
    angle = math.radians(angle)
    values, arguments = single_plane(angle, opacity, softness, z)
    image, _ = rasterize_hexels3d(**values, **arguments, conservative_bounds=True)
    reference_values = {name: value.detach().double().requires_grad_() for name, value in values.items()}
    expected = plane_reference(reference_values, arguments, angle)
    torch.testing.assert_close(image.double(), expected, rtol=1e-4, atol=1e-5)
    actual_grads = torch.autograd.grad(image.square().mean(), tuple(values.values()))
    expected_grads = torch.autograd.grad(expected.square().mean(), tuple(reference_values.values()))
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual.double(), reference, rtol=2e-4, atol=2e-5)


def test_crossing_plane_restores_missing_coverage() -> None:
    values, arguments = single_plane(math.pi / 4, .8, (10.,))
    legacy, _ = rasterize_hexels3d(**values, **arguments)
    corrected, _ = rasterize_hexels3d(**values, **arguments, conservative_bounds=True)
    assert torch.count_nonzero((legacy - corrected).abs() > 0.1) > 100


def test_invisible_opacity_preserves_public_radius() -> None:
    values, arguments = single_plane(0., .001, (10.,), 3.)
    image, radii = rasterize_hexels3d(**values, **arguments)
    assert radii[0] > 0
    torch.testing.assert_close(image, arguments['background'][:, None, None].expand_as(image), rtol=0, atol=0)
    image.sum().backward()
    for value in values.values():
        assert torch.count_nonzero(value.grad) == 0


@pytest.mark.parametrize('count', [1, 384])
@pytest.mark.parametrize('conservative', [False, True])
@pytest.mark.parametrize('prefilter', [False, True])
def test_cached_rgb_matches_replayed_contribution_path(conservative: bool, count: int, prefilter: bool) -> None:
    values, arguments = single_plane(math.pi / 4, .8, (1., 5., 12.))
    values = {name: value.detach().repeat(count, *([1] * (value.ndim - 1))) for name, value in values.items()}
    values['means3d'][:, 2] += torch.linspace(0., .5, count, device='cuda')
    values = {name: value.requires_grad_() for name, value in values.items()}
    arguments['rotations'] = arguments['rotations'].repeat(count, 1)
    arguments['prefilter_edge_on'] = prefilter
    means2d = torch.zeros((count, 4), device='cuda', requires_grad=True)
    parameters = (*values.values(), means2d)
    image, radii = rasterize_hexels3d(**values, **arguments, means2d=means2d, conservative_bounds=conservative)
    gradients = torch.autograd.grad(image.square().mean(), parameters)
    replay, replay_radii, _ = rasterize_hexels3d(**values, **arguments, means2d=means2d,
                                               conservative_bounds=conservative, return_contrib=True)
    replay_gradients = torch.autograd.grad(replay.square().mean(), parameters)
    torch.testing.assert_close(image, replay, rtol=0, atol=0)
    torch.testing.assert_close(radii, replay_radii, rtol=0, atol=0)
    for actual, expected in zip(gradients, replay_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
