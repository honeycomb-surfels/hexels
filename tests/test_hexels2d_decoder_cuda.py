"""Optional decoder work must preserve renders and the training gradients."""

import pytest
import torch

pytest.importorskip("gsplat_hexels")
from gsplat_hexels import rasterize_hexels_sum

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("softness_mode", ["scalar", "isotropic", "edges"])
def test_optional_outputs(rich: bool, softness_mode: str) -> None:
    generator = torch.Generator(device="cuda").manual_seed(481)
    values = {
        "xys": torch.rand(23, 2, device="cuda", generator=generator) * 32,
        "scales": torch.rand(23, 2, device="cuda", generator=generator) * 5 + 3,
        "rotations": torch.rand(23, 1, device="cuda", generator=generator),
        "colors": torch.rand(23, 3, device="cuda", generator=generator),
    }
    values["scales"][::2, 0] *= -1
    if rich:
        values["color_grads"] = torch.randn(23, 2, 3, device="cuda", generator=generator) * 0.1
    softness = 10.0
    if softness_mode != "scalar":
        shape = (23, 3) if softness_mode == "edges" else (23,)
        softness = torch.rand(shape, device="cuda", generator=generator) * 4 + 6
        values["softness"] = softness
    for value in values.values():
        value.requires_grad_()
    options = {} if "softness" in values else {"softness": softness}
    baseline, weights, entropy, ids = rasterize_hexels_sum(
        **values, **options, img_height=31, img_width=35, return_aux=True, return_topk=True)
    optimized = rasterize_hexels_sum(**values, **options, img_height=31, img_width=35)
    torch.testing.assert_close(optimized, baseline, rtol=0, atol=0)
    before = torch.autograd.grad(baseline.square().mean(), tuple(values.values()), retain_graph=True)
    after = torch.autograd.grad(optimized.square().mean(), tuple(values.values()))
    for found, wanted in zip(after, before):
        torch.testing.assert_close(found, wanted, rtol=2e-5, atol=2e-6)
    assert ids.shape == (31, 35, 10)
    assert torch.isfinite(weights).all() and torch.isfinite(entropy).all()
    with torch.no_grad():
        decoded = rasterize_hexels_sum(**values, **options, img_height=31, img_width=35)
    torch.testing.assert_close(decoded, baseline, rtol=0, atol=0)
