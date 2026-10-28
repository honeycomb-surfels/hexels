"""Independent removal-score, retained-optimizer-state and allocation-recipe checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.fit_image import recipe
from scripts.image_compression_allocation import removal_deltas, retain_rows


def test_removal_scores_match_individual_rerenders() -> None:
    torch.manual_seed(7)
    colors = torch.rand(4, 3, 3, dtype=torch.float64)
    weights = torch.rand(4, 3, dtype=torch.float64)
    weights[0] = torch.tensor([1.0, 0.0, 0.0])
    target = torch.rand(4, 3, dtype=torch.float64)
    image = (colors * weights[..., None]).sum(1) / (weights.sum(1, keepdim=True) + 1e-4)
    actual = removal_deltas(image, target, colors, weights)
    expected = []
    for index in range(3):
        changed = weights.clone()
        changed[:, index] = 0
        removed = (colors * changed[..., None]).sum(1) / (changed.sum(1, keepdim=True) + 1e-4)
        expected.append((removed - target).square().mean(-1) - (image - target).square().mean(-1))
    torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("step", [0, 3])
def test_pruning_preserves_adam_state_and_next_update(step: int) -> None:
    model = torch.nn.Module()
    model.xy = torch.nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(6, 2) / 12)
    model.vis_feat = torch.nn.Parameter(torch.ones(6, 3), requires_grad=False)
    model.num_gaussians = 6
    model.optimizer = torch.optim.Adam([{"params": model.xy, "lr": 0.007}], amsgrad=True)
    for _ in range(step):
        model.optimizer.zero_grad()
        model.xy.square().sum().backward()
        model.optimizer.step()
    original = model.xy
    state = model.optimizer.state[original]
    keep = torch.tensor([0, 2, 5])
    control = torch.nn.Parameter(original.detach()[keep].clone())
    optimizer = torch.optim.Adam([control], lr=0.007, amsgrad=True)
    optimizer.state[control] = {key: value[keep].clone() if value.ndim else value.clone()
                                for key, value in state.items()}
    retain_rows(model, keep)
    assert original not in model.optimizer.state
    assert model.optimizer.param_groups[0]["params"][0] is model.xy
    assert model.num_gaussians == 3 and model.vis_feat.shape == (3, 3)
    for parameter, opt in ((model.xy, model.optimizer), (control, optimizer)):
        opt.zero_grad()
        parameter.square().sum().backward()
        opt.step()
    torch.testing.assert_close(model.xy, control, rtol=0, atol=0)


def test_fit_image_rejects_gaussian_pruning() -> None:
    image = Path("anime-1_2k.png")
    with pytest.raises(ValueError, match="no Image-GS pruning arm"):
        recipe(image, "imagegs", "prune", 10000, 20000, 10000, "l1_ssim", 123)
    for variant, allocation in (("imagegs", "grow"), ("imagegs", "fixed"), ("flat", "prune"), ("rich", "prune")):
        job, _ = recipe(image, variant, allocation, 10000, 20000, 5100, "l1_ssim", 123)
        assert bool(job["prune_schedule"]) == (allocation == "prune")
