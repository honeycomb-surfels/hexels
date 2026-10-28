"""Error-aware population reduction for the optional 2D fitting pilot."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_prune_scores():
    """Build an isolated scorer from the renderer's own floating-point SDF helper."""
    import gsplat
    from torch.utils.cpp_extension import load_inline

    directory = Path(gsplat.__file__).parent / "cuda/csrc"
    helper = (directory / "hexels_forward.cu").read_text().split("namespace {", 1)[1].split("} // namespace", 1)[0]
    source = "#include <cuda_runtime.h>\n" + (directory / "config.h").read_text()
    source += "\nnamespace {" + helper + "}\n"
    source += Path(__file__).with_name("image_compression_prune_scores.cu").read_text()
    declaration = "std::vector<torch::Tensor> removal_scores_cuda(" + ", ".join(["torch::Tensor"] * 9) + ", bool);"
    return load_inline(name="hexel_prune_scores", cpp_sources=declaration, cuda_sources=source,
                       functions=["removal_scores_cuda"],
                       extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


def removal_deltas(image: torch.Tensor, target: torch.Tensor,
                   colors: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """MSE change after removing one selected weight, retaining the 1e-4 denominator.

    This holds the other selected IDs fixed; it does not insert the next tile
    candidate or estimate the joint effect of deleting several primitives.
    """
    denominator = weights.sum(-1, keepdim=True) + 1e-4
    removed = (image[:, None] * denominator[..., None] - weights[..., None] * colors)
    removed = removed / (denominator - weights).clamp_min(1e-4)[..., None]
    return ((removed - target[:, None]).square().mean(-1)
            - (image - target).square().mean(-1, keepdim=True))


def selected_colors_weights(values: dict[str, torch.Tensor], ids: torch.Tensor,
                            pixels: torch.Tensor, rich: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the existing Hexel SDF and spatial RGB for selected pixel IDs."""
    safe_ids = ids.clamp_min(0).long()
    delta = values["centers"][safe_ids] - pixels[:, None]
    angle = values["rotations"][safe_ids, 0]
    cosine, sine = angle.cos(), angle.sin()
    scales = values["scales"][safe_ids] + 1e-8
    u = (delta[..., 0] * cosine + delta[..., 1] * sine) / scales[..., 0]
    v = (-delta[..., 0] * sine + delta[..., 1] * cosine) / scales[..., 1]
    distance = torch.stack((v.abs(), (math.sqrt(3) / 2 * u + v / 2).abs(),
                            (math.sqrt(3) / 2 * u - v / 2).abs()), dim=-1).amax(-1) - 1
    softness = values["softness"][safe_ids] if rich else 10.0
    weights = torch.sigmoid(-distance * softness) * (ids >= 0)
    colors = values["features"][safe_ids]
    if rich:
        colors = colors + 0.1 * (delta[..., None] * values["color_gradients"][safe_ids]).sum(-2)
    return colors, weights


@torch.no_grad()
def retain_rows(model: torch.nn.Module, keep: torch.Tensor) -> None:
    """Slice parameters and Adam moments together, keeping the optimizer step and LR."""
    count = model.num_gaussians
    for name, parameter in list(model.named_parameters()):
        replacement = torch.nn.Parameter(parameter[keep].clone(), requires_grad=parameter.requires_grad)
        setattr(model, name, replacement)
        for group in model.optimizer.param_groups:
            if len(group["params"]) == 1 and group["params"][0] is parameter:
                group["params"][0] = replacement
                state = model.optimizer.state.pop(parameter, {})
                model.optimizer.state[replacement] = {
                    key: value[keep].clone() if isinstance(value, torch.Tensor)
                    and value.ndim and value.shape[0] == count else value
                    for key, value in state.items()
                }
    model.num_gaussians = len(keep)


@torch.no_grad()
def prune_model(model: torch.nn.Module, count: int) -> dict:
    """Rank individual removal errors using full-resolution renderer-selected IDs."""
    from scripts import image_compression_stable as stable

    values = stable.quantize_parameters(stable.storage_parameters(model))
    height, width = model.img_h, model.img_w
    values["centers"] = values["positions"] * values["positions"].new_tensor([width, height])
    values["scales"] = values["log_scales"].exp()
    image, denominator, _, ids = stable.legacy.rasterize_hexels_sum(
        values["centers"], values["scales"], values["rotations"], values["features"], height, width,
        color_grads=values["color_gradients"] if model.rich else None,
        softness=values["softness"] if model.rich else 10.0, return_aux=True, return_topk=True)
    packed = torch.cat((image, denominator[..., None]), dim=-1).contiguous()
    scores, max_weight_error = load_prune_scores().removal_scores_cuda(
        values["centers"], values["scales"], values["rotations"], values["features"],
        values["color_gradients"] if model.rich else image.new_zeros((model.num_gaussians, 2, 3)),
        values["softness"] if model.rich else image.new_full((model.num_gaussians,), 10.0),
        packed, model.gt_images.permute(1, 2, 0).contiguous(), ids, model.rich)
    torch.testing.assert_close(max_weight_error, image.new_zeros(()), rtol=0, atol=2e-4)
    old_count = model.num_gaussians
    # Scores estimate how much error each individual deletion would add.
    # Keep the most costly-to-remove rows; deleting a group is an approximation.
    keep = scores.argsort(descending=True, stable=True)[:count].sort().values
    retain_rows(model, keep)
    return {"step": model.step, "before": old_count, "after": count,
            "score_min": float(scores.min()), "score_max": float(scores.max()),
            "selected_weight_max_abs_error": float(max_weight_error)}
