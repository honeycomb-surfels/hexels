"""Check removal estimates against independent full CUDA rerenders."""

from __future__ import annotations

import pytest
import torch

from scripts.image_compression_allocation import load_prune_scores, removal_deltas, selected_colors_weights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("stress", [False, True])
def test_single_removal_matches_rerender(rich: bool, stress: bool) -> None:
    from gsplat_hexels import rasterize_hexels_sum

    torch.manual_seed(31)
    device = "cuda"
    values = {"centers": torch.tensor([[14.2, 15.3], [18.3, 11.7], [9.5, 19.1]], device=device),
              "scales": torch.tensor([[18.0, 16.0], [19.0, 17.0], [17.0, 20.0]], device=device),
              "rotations": torch.tensor([[0.2], [-0.3], [0.1]], device=device),
              "features": torch.rand(3, 3, device=device),
              "color_gradients": torch.randn(3, 2, 3, device=device) * 0.05,
              "softness": torch.tensor([7.0, 9.0, 11.0], device=device)}
    if stress:
        values["centers"] = torch.tensor([[1540.13, 1540.03], [1520.11, 1520.0], [1510.14, 1510.01]], device=device)
        values["scales"] = torch.tensor([[0.12, 2100.0], [0.11, 2200.0], [0.15, 2000.0]], device=device)
        values["rotations"].fill_(-torch.pi / 4)

    def render(keep: torch.Tensor, aux: bool = False):
        return rasterize_hexels_sum(values["centers"][keep], values["scales"][keep],
                                    values["rotations"][keep], values["features"][keep], 32, 32,
                                    color_grads=values["color_gradients"][keep] if rich else None,
                                    softness=values["softness"][keep] if rich else 10.0,
                                    return_aux=aux, return_topk=aux)

    all_ids = torch.arange(3, device=device)
    with torch.no_grad():
        image, denominator, _, ids = render(all_ids, True)
        y, x = torch.meshgrid(torch.arange(32, device=device), torch.arange(32, device=device), indexing="ij")
        pixels = torch.stack((x, y), dim=-1).reshape(-1, 2)
        colors, weights = selected_colors_weights(values, ids.reshape(-1, 10), pixels, rich)
        if not stress:
            torch.testing.assert_close(weights.sum(-1) + 1e-4, denominator.reshape(-1), rtol=1e-5, atol=1e-6)
        reference = (colors * weights[..., None]).sum(1) / (weights.sum(-1, keepdim=True) + 1e-4)
        if not stress:
            torch.testing.assert_close(reference, image.reshape(-1, 3), rtol=1e-5, atol=1e-6)
        target = torch.rand_like(image)
        deltas = removal_deltas(image.reshape(-1, 3), target.reshape(-1, 3), colors, weights)
        before = (image - target).square().mean(-1).reshape(-1)
        scores, error = load_prune_scores().removal_scores_cuda(
            values["centers"], values["scales"], values["rotations"], values["features"],
            values["color_gradients"], values["softness"],
            torch.cat((image, denominator[..., None]), dim=-1).contiguous(), target, ids, rich)
        torch.testing.assert_close(error, error.new_zeros(()), rtol=0, atol=2e-6)
        for primitive in range(3):
            removed = render(all_ids[all_ids != primitive])
            expected = (removed - target).square().mean(-1).reshape(-1) - before
            mask = ids.reshape(-1, 10) == primitive
            actual = (deltas * mask).sum(-1)
            if not stress:
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
            torch.testing.assert_close(scores[primitive], expected.sum(), rtol=2e-4, atol=2e-4)
