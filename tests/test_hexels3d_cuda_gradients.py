from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]

from gsplat import rasterize_hexels3d  # noqa: E402

from hexels.model import Hexels3DGeometry, Hexels3DModel  # noqa: E402
from hexels.refinement import directional_edge_children  # noqa: E402
from hexels.renderer import Hexels3DRenderer, _eval_linear_view_residual  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _projection_matrix(fov: float) -> torch.Tensor:
    tan_half_fov = math.tan(fov / 2.0)
    znear = 0.01
    zfar = 100.0
    matrix = torch.zeros((4, 4), device="cuda")
    matrix[0, 0] = 1.0 / tan_half_fov
    matrix[1, 1] = 1.0 / tan_half_fov
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix.T.contiguous()


def _prefilter_parameters() -> tuple[dict, dict, float]:
    """A thin, tilted patch seen through a translated perspective camera."""
    angle = math.radians(80)
    fov = math.radians(60)
    camera = torch.tensor([0.2, -0.1, 0.4], device="cuda")
    view = torch.eye(4, device="cuda")
    view[3, :3] = -camera
    values = {
        "means3d": torch.tensor([[0.235, -0.118, 3.0]], device="cuda", requires_grad=True),
        "scales": torch.tensor([[0.004, 0.09]], device="cuda", requires_grad=True),
        "colors": torch.tensor([[0.75, 0.45, 0.25]], device="cuda", requires_grad=True),
        "opacities": torch.tensor([0.8], device="cuda", requires_grad=True),
        "softnesses": torch.tensor([12.0], device="cuda", requires_grad=True),
    }
    arguments = dict(
        rotations=torch.tensor([[math.cos(angle / 2), 0., -math.sin(angle / 2), 0.]], device="cuda"),
        color_grads=torch.empty(0, device="cuda"), view_grads=torch.empty(0, device="cuda"),
        viewmatrix=view, projmatrix=(view @ _projection_matrix(fov)).contiguous(),
        campos=camera, background=torch.tensor([0.05, 0.1, 0.15], device="cuda"),
        image_width=73, image_height=59, tan_fovx=math.tan(fov / 2),
        tan_fovy=math.tan(fov / 2), softness=12., color_model="flat",
        view_model="none", use_view_grads=False, prefilter_edge_on=True,
    )
    return values, arguments, angle


def _prefilter_reference(values: dict, arguments: dict, angle: float) -> torch.Tensor:
    """Independent float64 ray/plane SDF and correctly projected screen filter."""
    width, height = arguments["image_width"], arguments["image_height"]
    yy, xx = torch.meshgrid(torch.arange(height, device="cuda", dtype=torch.float64),
                            torch.arange(width, device="cuda", dtype=torch.float64), indexing="ij")
    rx = (xx - (width - 1) / 2) * (2 * arguments["tan_fovx"] / width)
    ry = (yy - (height - 1) / 2) * (2 * arguments["tan_fovy"] / height)
    mx, my, mz = values["means3d"][0] - arguments["campos"].double()
    depth = (mz - math.tan(angle) * mx) / (1 - math.tan(angle) * rx)
    u = (rx * depth - mx) / (math.cos(angle) * values["scales"][0, 0])
    v = (ry * depth - my) / values["scales"][0, 1]
    dist = torch.maximum(v.abs(), torch.maximum((math.sqrt(3) / 2 * u + .5 * v).abs(),
                                               (math.sqrt(3) / 2 * u - .5 * v).abs())) - 1
    sdf_alpha = (values["opacities"][0] * torch.sigmoid(-values["softnesses"][0] * dist)).clamp_max(.99)
    point = torch.cat((values["means3d"][0], values["means3d"].new_ones(1)))
    clip = point @ arguments["projmatrix"].double()
    center = clip[:2] / (clip[3] + 1e-8)
    center = center * center.new_tensor([width / 2, height / 2]) + center.new_tensor([(width - 1) / 2, (height - 1) / 2])
    filter_alpha = (values["opacities"][0] * torch.exp(-((xx - center[0]) ** 2 + (yy - center[1]) ** 2))).clamp_max(.99)
    alpha = torch.maximum(sdf_alpha, filter_alpha)
    alpha = torch.where((depth >= .2) & (alpha >= 1 / 255), alpha, 0.)
    return alpha[None] * values["colors"][0, :, None, None] + (1 - alpha[None]) * arguments["background"].double()[:, None, None]


def test_prefilter_center_uses_row_vector_projection() -> None:
    from gsplat.cuda_hexels3d import _C

    values, arguments, _ = _prefilter_parameters()
    empty = torch.empty(0, device="cuda")
    _, radii, saved = _C.rasterize_hexels3d_forward_cached(
        values["means3d"], values["scales"], arguments["rotations"], values["colors"],
        empty, empty, empty, empty, empty, values["opacities"], values["softnesses"],
        arguments["viewmatrix"], arguments["projmatrix"], arguments["campos"],
        arguments["background"], arguments["image_width"], arguments["image_height"],
        arguments["tan_fovx"], arguments["tan_fovy"], arguments["softness"],
        1., 0, 0, False, True, False, False,
    )
    homogeneous = torch.cat((values["means3d"].double(), torch.ones((1, 1), device="cuda", dtype=torch.float64)), dim=1)
    clip = homogeneous @ arguments["projmatrix"].double()
    center = clip[:, :2] / (clip[:, 3:] + 1e-8)
    size = center.new_tensor([arguments["image_width"], arguments["image_height"]])
    center = center * (size / 2) + (size - 1) / 2
    assert radii[0] > 0
    torch.testing.assert_close(saved[6].double(), center, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("surface", [False, True])
def test_prefilter_image_and_gradients_match_independent_reference(surface: bool) -> None:
    values, arguments, angle = _prefilter_parameters()
    image = rasterize_hexels3d(**values, **arguments, return_surface=surface)[0]
    unfiltered = rasterize_hexels3d(**values, **{**arguments, "prefilter_edge_on": False})[0]
    assert torch.count_nonzero((image - unfiltered).abs() > 1e-3) > 3
    reference_values = {name: value.detach().double().requires_grad_() for name, value in values.items()}
    reference = _prefilter_reference(reference_values, arguments, angle)
    torch.testing.assert_close(image.double(), reference, rtol=1e-4, atol=1e-5)
    weights = torch.linspace(.2, .9, image.numel(), device="cuda").reshape_as(image)
    actual = torch.autograd.grad((image * weights).sum(), tuple(values.values()))
    expected = torch.autograd.grad((reference * weights.double()).sum(), tuple(reference_values.values()))
    for gradient, independent in zip(actual, expected):
        torch.testing.assert_close(gradient.double(), independent, rtol=2e-4, atol=2e-5)
    for axis in range(3):
        epsilon = 1e-4
        plus = {name: value.detach().clone() for name, value in values.items()}
        minus = {name: value.detach().clone() for name, value in values.items()}
        plus["means3d"][0, axis] += epsilon
        minus["means3d"][0, axis] -= epsilon
        positive = rasterize_hexels3d(**plus, **arguments)[0]
        negative = rasterize_hexels3d(**minus, **arguments)[0]
        finite = ((positive - negative) * weights).sum() / (2 * epsilon)
        torch.testing.assert_close(actual[0][0, axis], finite, rtol=3e-3, atol=2e-3)


def _parameters() -> dict[str, torch.Tensor]:
    angle = math.radians(75.0)
    values = {
        "means3d": torch.tensor([[0.08, 0.04, 3.0]], device="cuda"),
        "scales": torch.tensor([[0.65, 0.55]], device="cuda"),
        "rotation_quats": torch.tensor([[math.cos(angle), 0.0, math.sin(angle), 0.0]], device="cuda"),
        "colors": torch.tensor([[0.25, 0.35, 0.45]], device="cuda"),
        "color_grads": torch.tensor([[[0.07, -0.03, 0.04], [0.02, 0.05, -0.06]]], device="cuda"),
        "color_quads": torch.tensor(
            [[[0.01, 0.02, -0.01], [-0.02, 0.01, 0.03], [0.015, -0.01, 0.02]]],
            device="cuda",
        ),
        "view_grads": torch.tensor(
            [[[0.03, -0.02, 0.01], [0.01, 0.04, -0.03], [-0.02, 0.01, 0.05]]],
            device="cuda",
        ),
        "specular_ks": torch.tensor([[0.2, 0.3, 0.4]], device="cuda"),
        "specular_log_s": torch.tensor([1.0], device="cuda"),
        "opacities": torch.tensor([0.8], device="cuda"),
        "softnesses": torch.tensor([10.0], device="cuda"),
    }
    for value in values.values():
        value.requires_grad_(True)
    return values


def _render(
    values: dict[str, torch.Tensor],
    view_model: str,
    *,
    color_model: str = "quadratic",
    normalize_rotation: bool = True,
    linear_view_fast_path: bool = False,
) -> torch.Tensor:
    fov = math.radians(60.0)
    rotations = values["rotation_quats"]
    if normalize_rotation:
        rotations = torch.nn.functional.normalize(rotations, dim=-1)
    colors = values["colors"]
    native_view_model = view_model
    use_view_grads = view_model == "linear"
    if linear_view_fast_path:
        view_delta = values["means3d"]
        view_dirs = view_delta * torch.rsqrt(torch.sum(view_delta.square(), dim=-1, keepdim=True) + 1e-8)
        colors = colors + _eval_linear_view_residual(values["view_grads"], view_dirs)
        native_view_model = "none"
        use_view_grads = False
    image, _ = rasterize_hexels3d(
        means3d=values["means3d"],
        scales=values["scales"],
        rotations=rotations,
        colors=colors,
        color_grads=values["color_grads"],
        color_quads=values["color_quads"],
        view_grads=values["view_grads"],
        specular_ks=values["specular_ks"],
        specular_log_s=values["specular_log_s"],
        opacities=values["opacities"],
        softnesses=values["softnesses"],
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=_projection_matrix(fov),
        campos=torch.zeros(3, device="cuda"),
        background=torch.zeros(3, device="cuda"),
        image_width=64,
        image_height=64,
        tan_fovx=math.tan(fov / 2.0),
        tan_fovy=math.tan(fov / 2.0),
        softness=10.0,
        color_model=color_model,
        view_model=native_view_model,
        use_view_grads=use_view_grads,
    )
    return image


def _interior_weight() -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.zeros((3, 64, 64), device="cuda")
    weight[:, 24:40, 24:40] = torch.randn((3, 16, 16), generator=generator, device="cuda")
    return weight


def test_cuda_reports_sole_and_overdraw_hits() -> None:
    fov = math.radians(60.0)
    tan_half_fov = math.tan(fov / 2.0)
    means3d = torch.tensor([[-0.45, 0.0, 3.0], [0.45, 0.0, 3.0]], device="cuda")
    colors = torch.full((2, 3), 0.5, device="cuda")
    outputs = rasterize_hexels3d(
        means3d=means3d,
        scales=torch.full((2, 2), 0.55, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda"),
        colors=colors,
        color_grads=torch.zeros((2, 2, 3), device="cuda"),
        color_quads=torch.zeros((2, 3, 3), device="cuda"),
        view_grads=torch.zeros((2, 3, 3), device="cuda"),
        specular_ks=torch.zeros((2, 3), device="cuda"),
        specular_log_s=torch.zeros(2, device="cuda"),
        opacities=torch.full((2,), 0.8, device="cuda"),
        softnesses=torch.full((2,), 30.0, device="cuda"),
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=_projection_matrix(fov),
        campos=torch.zeros(3, device="cuda"),
        background=torch.zeros(3, device="cuda"),
        image_width=64,
        image_height=64,
        tan_fovx=tan_half_fov,
        tan_fovy=tan_half_fov,
        softness=30.0,
        color_model="flat",
        view_model="none",
        use_view_grads=False,
        return_contrib=True,
        return_overdraw=True,
        return_overdraw_hits=True,
        return_surface=True,
        overdraw_base_count=2,
        prefilter_edge_on=False,
    )
    _, _, _, _, hits, _, _, _ = outputs

    assert int((hits == 1.0).sum().item()) > 0
    assert int((hits > 1.0).sum().item()) > 0


@pytest.mark.parametrize("back_facing", [False, True])
def test_surface_normals_face_camera_without_changing_rgb(back_facing: bool) -> None:
    fov = math.radians(60)
    rotation = [0.0, 1.0, 0.0, 0.0] if back_facing else [1.0, 0.0, 0.0, 0.0]
    arguments = dict(
        means3d=torch.tensor([[0.0, 0.0, 3.0]], device="cuda"),
        scales=torch.full((1, 2), 0.7, device="cuda"),
        rotations=torch.tensor([rotation], device="cuda"),
        colors=torch.full((1, 3), 0.5, device="cuda"),
        opacities=torch.tensor([0.7], device="cuda", requires_grad=True),
        color_grads=torch.empty(0, device="cuda"), view_grads=torch.empty(0, device="cuda"),
        viewmatrix=torch.eye(4, device="cuda"), projmatrix=_projection_matrix(fov),
        campos=torch.zeros(3, device="cuda"), background=torch.zeros(3, device="cuda"),
        image_width=48, image_height=48, tan_fovx=math.tan(fov / 2), tan_fovy=math.tan(fov / 2),
        softness=10.0, color_model="flat", view_model="none", use_view_grads=False,
    )
    plain = rasterize_hexels3d(**arguments)
    surface = rasterize_hexels3d(**arguments, return_surface=True)
    assert torch.equal(plain[0], surface[0])
    assert torch.equal(plain[1], surface[1])
    alpha, normals = surface[3], surface[4]
    torch.testing.assert_close(normals[2], -alpha)
    torch.testing.assert_close(normals[:2], torch.zeros_like(normals[:2]))
    normal_gradient = torch.autograd.grad(normals[2].sum(), arguments["opacities"], retain_graph=True)[0]
    alpha_gradient = torch.autograd.grad(alpha.sum(), arguments["opacities"])[0]
    torch.testing.assert_close(normal_gradient, -alpha_gradient)


def _distortion(means3d: torch.Tensor) -> torch.Tensor:
    fov = math.radians(60.0)
    outputs = rasterize_hexels3d(
        means3d=means3d,
        scales=torch.full((2, 2), 0.8, device="cuda"),
        rotations=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 2,
            device="cuda",
        ),
        colors=torch.full((2, 3), 0.5, device="cuda"),
        color_grads=torch.empty(0, device="cuda"),
        view_grads=torch.empty(0, device="cuda"),
        opacities=torch.full((2,), 0.6, device="cuda"),
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=_projection_matrix(fov),
        campos=torch.zeros(3, device="cuda"),
        background=torch.zeros(3, device="cuda"),
        image_width=48,
        image_height=48,
        tan_fovx=math.tan(fov / 2.0),
        tan_fovy=math.tan(fov / 2.0),
        softness=20.0,
        color_model="flat",
        view_model="none",
        use_view_grads=False,
        return_surface=True,
        return_distortion=True,
        overdraw_base_count=2,
    )
    return outputs[-1]


def test_ray_distortion_forward_and_depth_gradient() -> None:
    means = torch.tensor(
        [[0.0, 0.0, 2.5], [0.0, 0.0, 3.5]],
        device="cuda",
        requires_grad=True,
    )
    loss = _distortion(means).mean()
    assert float(loss.item()) > 0.0
    loss.backward()
    analytical = float(means.grad[0, 2].item())

    epsilon = 1e-3
    with torch.no_grad():
        means[0, 2].add_(epsilon)
    plus = float(_distortion(means).mean().item())
    with torch.no_grad():
        means[0, 2].add_(-2.0 * epsilon)
    minus = float(_distortion(means).mean().item())
    with torch.no_grad():
        means[0, 2].add_(epsilon)
    numerical = (plus - minus) / (2.0 * epsilon)

    assert abs(analytical) > 1e-7
    assert analytical == pytest.approx(numerical, rel=0.03, abs=1e-5)


@pytest.mark.parametrize("parameter,index", [("means3d", (0, 2)), ("scales", (0, 0)),
                                           ("rotations", (0, 2)), ("opacities", (0,))])
@pytest.mark.parametrize("output", ["distortion", "surface", "median"])
def test_surface_geometry_vjps(parameter: str, index: tuple[int, ...], output: str) -> None:
    values = {
        "means3d": torch.tensor([[0.07, -0.09, 2.5], [-0.04, 0.06, 3.5]], device="cuda"),
        "scales": torch.tensor([[0.8, 0.7], [0.9, 0.75]], device="cuda"),
        "rotations": torch.tensor([[0.98, 0.1, 0.15, 0.03], [0.97, -0.12, 0.08, 0.15]], device="cuda"),
        "opacities": torch.tensor([0.6, 0.7], device="cuda"),
    }
    for value in values.values():
        value.requires_grad_(True)

    def objective(reference: bool = False) -> torch.Tensor:
        fov = math.radians(60)
        arguments = dict(
            color_grads=torch.empty(0, device="cuda"), view_grads=torch.empty(0, device="cuda"),
            viewmatrix=torch.eye(4, device="cuda"), projmatrix=_projection_matrix(fov),
            campos=torch.zeros(3, device="cuda"), background=torch.zeros(3, device="cuda"),
            image_width=48, image_height=48, tan_fovx=math.tan(fov / 2), tan_fovy=math.tan(fov / 2),
            softness=6.0, color_model="flat", view_model="none", use_view_grads=False,
            return_surface=True, return_distortion=True,
            return_median_depth=output == "median",
        )
        if reference:
            alpha_values = []
            depths = []
            for primitive in range(2):
                single = rasterize_hexels3d(
                    **{name: value[primitive:primitive + 1] for name, value in values.items()},
                    **arguments, colors=torch.full((1, 3), 0.5, device="cuda"),
                )
                alpha = single[3][24:30, 24:30].double()
                depth = single[2][24:30, 24:30].double() / alpha
                alpha_values.append(alpha)
                depths.append(100.0 / 99.8 * (1.0 - 0.2 / depth))
            return 1e5 * (alpha_values[0] * alpha_values[1] * (1 - alpha_values[0])
                          * (depths[0] - depths[1]).square()).mean()
        outputs = rasterize_hexels3d(**values, **arguments, colors=torch.full((2, 3), 0.5, device="cuda"))
        if output == "distortion":
            return 1e5 * outputs[-1][24:30, 24:30].mean()
        if output == "median":
            return outputs[-1][24:30, 24:30].mean()
        return (outputs[2][24:30, 24:30].mean() + 0.3 * outputs[3][24:30, 24:30].mean()
                + 0.7 * outputs[4][0, 24:30, 24:30].mean())

    objective().backward()
    if output == "distortion":
        assert objective().item() == pytest.approx(objective(reference=True).item(), rel=1e-3)
    analytical = values[parameter].grad[index].item()
    epsilon = 1e-3
    with torch.no_grad():
        values[parameter][index] += epsilon
        plus = objective(reference=output == "distortion").item()
        values[parameter][index] -= 2 * epsilon
        minus = objective(reference=output == "distortion").item()
        values[parameter][index] += epsilon
    numerical = (plus - minus) / (2 * epsilon)
    if output != "median" or parameter in {"means3d", "rotations"}:
        assert abs(analytical) > 1e-7
    assert analytical == pytest.approx(numerical, rel=0.03, abs=3e-4)


@pytest.mark.parametrize("opacity", [0.0, 0.2, 0.4, 0.8, 0.999])
@pytest.mark.parametrize("prefilter", [False, True])
def test_median_depth_matches_reference_and_preserves_outputs(opacity: float, prefilter: bool) -> None:
    fov = math.radians(60)
    depth_values = [2.5, 3.5, 4.5]
    arguments = dict(
        means3d=torch.tensor([[0.0, 0.0, depth] for depth in depth_values], device="cuda", requires_grad=True),
        scales=torch.full((3, 2), 0.8, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, device="cuda"),
        colors=torch.full((3, 3), 0.5, device="cuda"),
        color_grads=torch.empty(0, device="cuda"), view_grads=torch.empty(0, device="cuda"),
        opacities=torch.full((3,), opacity, device="cuda", requires_grad=True),
        viewmatrix=torch.eye(4, device="cuda"), projmatrix=_projection_matrix(fov),
        campos=torch.zeros(3, device="cuda"), background=torch.zeros(3, device="cuda"),
        image_width=49, image_height=49, tan_fovx=math.tan(fov / 2), tan_fovy=math.tan(fov / 2),
        softness=20.0, color_model="flat", view_model="none", use_view_grads=False,
        return_surface=True, return_distortion=True,
        prefilter_edge_on=prefilter,
    )
    expected_outputs = rasterize_hexels3d(**arguments)
    median_outputs = rasterize_hexels3d(**arguments, return_median_depth=True)
    for before, after in zip(expected_outputs, median_outputs[:-1], strict=True):
        assert torch.equal(before, after)
    transmittance = 1.0
    median = 0.0
    selected = -1
    alpha = min(opacity / (1.0 + math.exp(-20.0)), 0.99)
    for index, depth in enumerate(depth_values):
        if alpha < 1 / 255:
            continue
        remaining = transmittance * (1 - alpha)
        if remaining < 0.0001:
            break
        if transmittance > 0.5:
            median = depth
            selected = index
        transmittance = remaining
    assert median_outputs[-1][24, 24].item() == pytest.approx(median, abs=1e-5)
    median_outputs[-1][24, 24].backward()
    expected_gradient = torch.zeros_like(arguments["means3d"])
    if selected >= 0:
        expected_gradient[selected, 2] = 1.0
    torch.testing.assert_close(arguments["means3d"].grad, expected_gradient)
    torch.testing.assert_close(arguments["opacities"].grad, torch.zeros_like(arguments["opacities"]))


def test_median_depth_requires_surface() -> None:
    means = torch.zeros((2, 3), device="cuda", requires_grad=True)
    with pytest.raises(ValueError, match="requires surface"):
        rasterize_hexels3d(
            means, torch.ones((2, 2), device="cuda"), torch.ones((2, 4), device="cuda"),
            torch.ones((2, 3), device="cuda"), torch.empty(0, device="cuda"),
            torch.empty(0, device="cuda"), torch.ones(2, device="cuda"),
            torch.eye(4, device="cuda"), torch.eye(4, device="cuda"),
            torch.zeros(3, device="cuda"), torch.zeros(3, device="cuda"),
            4, 4, 1.0, 1.0, 10.0, color_model="flat", view_model="none",
            return_median_depth=True,
        )


def test_inactive_capacity_supports_moment_vjp_then_loss_backward() -> None:
    geometry = Hexels3DGeometry(
        means3d=torch.tensor([[0.0, 0.0, 3.0], [2.0, 0.0, 3.0]], device="cuda"),
        scales=torch.full((2, 2), 0.6, device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda"),
    )
    model = Hexels3DModel(
        geometry=geometry,
        colors_init=torch.full((2, 3), 0.4, device="cuda"),
        init_opacity=0.8,
        init_softness=10.0,
        view_basis_dim=3,
    )
    model.active_mask[1] = False
    fov = math.radians(60.0)
    camera = SimpleNamespace(
        image_width=32,
        image_height=32,
        FoVx=fov,
        FoVy=fov,
        world_view_transform=torch.eye(4, device="cuda"),
        full_proj_transform=_projection_matrix(fov),
        camera_center=torch.zeros(3, device="cuda"),
    )
    options = SimpleNamespace(
        softness=10.0,
        scale_modifier=1.0,
        color_model="linear",
        view_model="linear",
        use_view_grads=True,
        prefilter_edge_on=False,
        debug=False,
        enable_mcmc=False,
        repulsion_screen_weight=0.0,
        child_offset_frac=0.0,
        child_scale_ratio=1.0,
        child_rotation_delta=False,
        children_detach_geometry=True,
    )
    result = Hexels3DRenderer(options).render(
        camera,
        model,
        torch.zeros(3, device="cuda"),
        return_contrib=True,
        return_viewspace_points=True,
    )
    prediction = result["image"]
    residual = (prediction.detach() - 1.0).square() / 3.0
    color_gradient, gradient_gradient = torch.autograd.grad(
        torch.sum(prediction * residual),
        (model.colors, model.color_grads),
        retain_graph=True,
    )
    prediction.mean().backward()

    viewspace_gradient = result["viewspace_points"].grad
    assert viewspace_gradient is not None
    assert viewspace_gradient.shape == (2, 3)
    assert torch.count_nonzero(viewspace_gradient[0]) > 0
    assert torch.count_nonzero(viewspace_gradient[1]) == 0
    assert torch.isfinite(color_gradient).all()
    assert torch.isfinite(gradient_gradient).all()


def test_bounded_edge_child_materializes_through_active_mask() -> None:
    geometry = Hexels3DGeometry(
        means3d=torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]], device="cuda"),
        scales=torch.tensor([[0.8, 0.6], [0.8, 0.6]], device="cuda"),
        rotations=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device="cuda"),
    )
    model = Hexels3DModel(
        geometry=geometry,
        colors_init=torch.tensor([[0.25, 0.35, 0.45], [0.0, 0.0, 0.0]], device="cuda"),
        init_opacity=0.8,
        init_softness=10.0,
        view_basis_dim=3,
    )
    with torch.no_grad():
        model.color_grads[0] = torch.tensor(
            [[0.1, -0.05, 0.02], [0.03, 0.04, -0.02]],
            device="cuda",
        )
        model.active_mask[1] = False

    child = directional_edge_children(
        model.state_dict(),
        torch.tensor([0], device="cuda", dtype=torch.long),
        local_offsets=torch.tensor([[1.0, 0.0]], device="cuda"),
        scale_modifier=1.0,
        scale_ratio=0.5,
        init_opacity=0.5,
        extent_cap_fraction=0.2,
        parent_opacity_fraction=0.25,
        scene_extent=1.0,
    )
    assert all(value.is_cuda and torch.isfinite(value).all() for value in child.values())
    with torch.no_grad():
        for name, value in child.items():
            getattr(model, name)[1].copy_(value[0])
        model.active_mask[1] = True

    assert torch.allclose(
        torch.linalg.vector_norm(model.means3d[1] - model.means3d[0]),
        torch.tensor(0.2, device="cuda"),
    )
    assert torch.allclose(model.scales[1].max(), torch.tensor(0.2, device="cuda"))
    assert torch.allclose(model.opacities[1], torch.tensor(0.2, device="cuda"))

    fov = math.radians(60.0)
    camera = SimpleNamespace(
        image_width=64,
        image_height=64,
        FoVx=fov,
        FoVy=fov,
        world_view_transform=torch.eye(4, device="cuda"),
        full_proj_transform=_projection_matrix(fov),
        camera_center=torch.zeros(3, device="cuda"),
    )
    options = SimpleNamespace(
        softness=10.0,
        scale_modifier=1.0,
        color_model="linear",
        view_model="linear",
        use_view_grads=True,
        prefilter_edge_on=False,
        debug=False,
        enable_mcmc=False,
        repulsion_screen_weight=0.0,
        child_offset_frac=0.0,
        child_scale_ratio=1.0,
        child_rotation_delta=False,
        children_detach_geometry=True,
    )
    renderer = Hexels3DRenderer(options)
    background = torch.zeros(3, device="cuda")
    model.active_mask[1] = False
    base = renderer.render(camera, model, background)["image"]
    model.active_mask[1] = True
    proposal = renderer.render(camera, model, background)["image"]

    assert torch.isfinite(proposal).all()
    assert torch.count_nonzero(proposal - base) > 0
    proposal.mean().backward()
    assert model.colors.grad is not None
    assert model.opacity_logits.grad is not None
    assert torch.isfinite(model.colors.grad[1]).all()
    assert torch.isfinite(model.opacity_logits.grad[1])
    assert torch.count_nonzero(model.colors.grad[1]) > 0
    assert model.opacity_logits.grad[1].item() != 0.0


def _directional_derivatives(
    values: dict[str, torch.Tensor],
    name: str,
    weight: torch.Tensor,
    *,
    view_model: str,
    seed: int,
    color_model: str = "quadratic",
    epsilon: float = 1e-3,
    normalize_rotation: bool = True,
) -> tuple[float, float]:
    for value in values.values():
        value.grad = None
    loss = (
        _render(
            values,
            view_model,
            color_model=color_model,
            normalize_rotation=normalize_rotation,
        )
        * weight
    ).sum()
    loss.backward()
    parameter = values[name]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    direction = torch.randn(parameter.shape, generator=generator, device="cuda")
    direction /= direction.norm()
    analytical = float((parameter.grad * direction).sum().item())

    with torch.no_grad():
        parameter.add_(epsilon * direction)
    plus = float(
        (
            _render(
                values,
                view_model,
                color_model=color_model,
                normalize_rotation=normalize_rotation,
            )
            * weight
        ).sum().item()
    )
    with torch.no_grad():
        parameter.add_(-2.0 * epsilon * direction)
    minus = float(
        (
            _render(
                values,
                view_model,
                color_model=color_model,
                normalize_rotation=normalize_rotation,
            )
            * weight
        ).sum().item()
    )
    with torch.no_grad():
        parameter.add_(epsilon * direction)
    numerical = (plus - minus) / (2.0 * epsilon)
    return analytical, numerical


@pytest.mark.parametrize(
    ("name", "seed"),
    [
        ("means3d", 1),
        ("scales", 2),
        ("rotation_quats", 3),
        ("colors", 4),
        ("color_grads", 5),
        ("color_quads", 6),
        ("view_grads", 7),
        ("opacities", 8),
        ("softnesses", 9),
    ],
)
def test_interior_directional_gradients(name: str, seed: int) -> None:
    analytical, numerical = _directional_derivatives(
        _parameters(),
        name,
        _interior_weight(),
        view_model="linear",
        seed=seed,
    )
    assert analytical == pytest.approx(numerical, rel=0.02, abs=0.01)


@pytest.mark.parametrize(("name", "seed"), [("means3d", 20), ("color_quads", 21)])
def test_bounded_quadratic_halo_directional_gradients(name: str, seed: int) -> None:
    values = _parameters()
    values["color_quads"].data.mul_(5.0)
    with torch.no_grad():
        raw = _render(values, "none", color_model="quadratic")
        bounded = _render(values, "none", color_model="quadratic_bounded")
        halo = torch.amax(torch.abs(raw - bounded), dim=0) > 1e-6
    assert torch.count_nonzero(halo) > 0
    generator = torch.Generator(device="cuda").manual_seed(19)
    weight = torch.randn((3, 64, 64), generator=generator, device="cuda")
    weight *= halo.unsqueeze(0)

    analytical, numerical = _directional_derivatives(
        values,
        name,
        weight,
        view_model="none",
        color_model="quadratic_bounded",
        seed=seed,
        epsilon=3e-4 if name == "means3d" else 1e-3,
    )
    assert analytical == pytest.approx(numerical, rel=0.03, abs=0.01)


def test_bounded_quadratic_leaves_base_and_linear_paths_unchanged() -> None:
    raw_values = _parameters()
    raw_values["color_quads"].data.zero_()
    bounded_values = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in raw_values.items()
    }
    raw = _render(raw_values, "none", color_model="quadratic")
    bounded = _render(bounded_values, "none", color_model="quadratic_bounded")
    assert torch.equal(raw, bounded)

    weight = _interior_weight()
    (raw * weight).sum().backward()
    (bounded * weight).sum().backward()
    for name in ("means3d", "colors", "color_grads"):
        torch.testing.assert_close(
            raw_values[name].grad,
            bounded_values[name].grad,
            rtol=3e-6,
            atol=1e-5,
        )


@pytest.mark.parametrize(("name", "seed"), [("specular_ks", 10), ("specular_log_s", 11)])
def test_lobe_directional_gradients(name: str, seed: int) -> None:
    analytical, numerical = _directional_derivatives(
        _parameters(),
        name,
        _interior_weight(),
        view_model="lobe",
        seed=seed,
    )
    assert analytical == pytest.approx(numerical, rel=0.02, abs=0.01)


def test_raw_nonunit_quaternion_directional_gradient() -> None:
    values = _parameters()
    values["rotation_quats"].data.mul_(1.7)
    analytical, numerical = _directional_derivatives(
        values,
        "rotation_quats",
        _interior_weight(),
        view_model="linear",
        seed=12,
        normalize_rotation=False,
    )
    assert analytical == pytest.approx(numerical, rel=0.02, abs=0.01)
    quaternion = values["rotation_quats"]
    gradient = quaternion.grad
    assert gradient is not None
    radial_cosine = (gradient * quaternion).sum().abs() / (gradient.norm() * quaternion.norm())
    assert radial_cosine.item() < 1e-5


def test_linear_view_fast_path_matches_native_forward_and_gradients() -> None:
    native_values = _parameters()
    fast_values = {
        name: value.detach().clone().requires_grad_(True) for name, value in native_values.items()
    }
    native_image = _render(native_values, "linear")
    fast_image = _render(fast_values, "linear", linear_view_fast_path=True)
    torch.testing.assert_close(fast_image, native_image, rtol=1e-5, atol=1e-6)

    weight = _interior_weight()
    (native_image * weight).sum().backward()
    (fast_image * weight).sum().backward()
    for name in ("means3d", "colors", "view_grads"):
        torch.testing.assert_close(
            fast_values[name].grad,
            native_values[name].grad,
            rtol=2e-4,
            atol=2e-5,
        )


def test_lobe_sharpness_gradient_at_lower_clamp_boundary() -> None:
    values = _parameters()
    values["specular_log_s"].data.zero_()
    weight = _interior_weight()
    loss = (_render(values, "lobe") * weight).sum()
    loss.backward()
    analytical = float(values["specular_log_s"].grad.item())
    epsilon = 1e-3
    with torch.no_grad():
        values["specular_log_s"].add_(epsilon)
    numerical = (float((_render(values, "lobe") * weight).sum().item()) - float(loss.item())) / epsilon
    assert abs(analytical) > 1e-5
    assert analytical == pytest.approx(numerical, rel=0.02, abs=0.01)


@pytest.mark.parametrize("angle", [0.0, 0.7, 1.5707963267948966])
@pytest.mark.parametrize("shape", [(97, 65), (128, 128)])
@pytest.mark.parametrize("conservative_bounds", [False, True])
def test_anisotropic_plane_support_and_gradients(
    angle: float, shape: tuple[int, int], conservative_bounds: bool
) -> None:
    width, height = shape
    scales = torch.tensor([[0.8, 0.045]], device="cuda", requires_grad=True)
    opacity = torch.tensor([0.8], device="cuda", requires_grad=True)
    color = torch.tensor([[0.3, 0.6, 0.8]], device="cuda", requires_grad=True)
    softness = torch.tensor([12.0], device="cuda", requires_grad=True)
    background = torch.tensor([0.1, 0.2, 0.3], device="cuda")
    projection = torch.zeros((4, 4), device="cuda")
    projection[0, 0] = projection[1, 1] = math.sqrt(3)
    projection[3, 2] = 1
    projection[2, 2], projection[2, 3] = 100 / 99.99, -1 / 99.99
    image, _ = rasterize_hexels3d(
        means3d=torch.tensor([[0.0, 0.0, 3.0]], device="cuda"),
        scales=scales,
        rotations=torch.tensor(
            [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]], device="cuda"
        ),
        colors=color,
        color_grads=torch.empty(0, device="cuda"),
        view_grads=torch.empty(0, device="cuda"),
        opacities=opacity,
        softnesses=softness,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=projection.T.contiguous(),
        campos=torch.zeros(3, device="cuda"),
        background=background,
        image_width=width,
        image_height=height,
        tan_fovx=1 / math.sqrt(3),
        tan_fovy=1 / math.sqrt(3),
        softness=12.0,
        color_model="flat",
        view_model="none",
        use_view_grads=False,
        conservative_bounds=conservative_bounds,
    )
    y, x = torch.meshgrid(
        torch.arange(height, device="cuda"),
        torch.arange(width, device="cuda"),
        indexing="ij",
    )
    local_x = (x - (width - 1) / 2) * 3 / (width / 2 * math.sqrt(3))
    local_y = (y - (height - 1) / 2) * 3 / (height / 2 * math.sqrt(3))
    u = (local_x * math.cos(angle) + local_y * math.sin(angle)) / scales[0, 0]
    v = (-local_x * math.sin(angle) + local_y * math.cos(angle)) / scales[0, 1]
    distance = torch.maximum(v.abs(), 0.5 * (math.sqrt(3) * u.abs() + v.abs())) - 1
    alpha = (opacity * torch.sigmoid(-softness * distance)).clamp_max(0.99)
    alpha = torch.where(alpha >= 1 / 255, alpha, 0.0)
    reference = (
        alpha[None] * color[0, :, None, None]
        + (1 - alpha[None]) * background[:, None, None]
    )
    torch.testing.assert_close(image, reference, rtol=1e-4, atol=1e-5)
    parameters = (scales, opacity, color, softness)
    actual = torch.autograd.grad(image.square().mean(), parameters, retain_graph=True)
    expected = torch.autograd.grad(reference.square().mean(), parameters)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("shape", [(0,), (0, 3)])
def test_empty_softness_gradient_shape(shape: tuple[int, ...]) -> None:
    softness = torch.empty(shape, device="cuda", requires_grad=True)
    image, _ = rasterize_hexels3d(
        means3d=torch.empty((0, 3), device="cuda"),
        scales=torch.empty((0, 2), device="cuda"),
        rotations=torch.empty((0, 4), device="cuda"),
        colors=torch.empty((0, 3), device="cuda"),
        color_grads=torch.empty((0, 2, 3), device="cuda"),
        view_grads=torch.empty((0, 3, 3), device="cuda"),
        opacities=torch.empty((0,), device="cuda"),
        softnesses=softness,
        viewmatrix=torch.eye(4, device="cuda"),
        projmatrix=torch.eye(4, device="cuda"),
        campos=torch.zeros(3, device="cuda"),
        background=torch.tensor([0.1, 0.2, 0.3], device="cuda"),
        image_width=17,
        image_height=19,
        tan_fovx=1.0,
        tan_fovy=1.0,
        softness=10.0,
        color_model="flat",
        view_model="none",
        use_view_grads=False,
    )
    image.sum().backward()
    assert softness.grad.shape == softness.shape
    assert softness.grad.numel() == 0
