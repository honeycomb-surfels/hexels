"""Surface-normal objectives, retaining the historical loss for ablations."""

import torch
from torch import Tensor
from torch.nn.functional import normalize


def surface_normal_loss(
    depth_normals: Tensor,
    rendered_normals: Tensor,
    alpha: Tensor,
    *,
    objective: str,
    alpha_threshold: float = 0.01,
) -> Tensor:
    """Compare world-space HWC depth normals to composited CHW primitive normals."""
    alpha_detached = alpha.detach()
    rendered = rendered_normals.permute(1, 2, 0)
    if objective == "2dgs":
        return (1.0 - (rendered * depth_normals).sum(dim=-1) * alpha_detached).mean()
    if objective != "legacy":
        raise ValueError(f"Unknown surface normal objective: {objective}")
    dot = (depth_normals * normalize(rendered, dim=-1)).sum(dim=-1).clamp(-1.0, 1.0)
    valid = (alpha_detached > alpha_threshold) & (torch.linalg.norm(depth_normals, dim=-1) > 0.0)
    weights = (alpha_detached * valid.to(dtype=alpha.dtype)).detach()
    return ((1.0 - dot) * weights).sum() / (weights.sum() + 1e-6)
