from __future__ import annotations

import math

import torch
from torch import Tensor

_N_MAX = 51


def _make_binoms(device: torch.device, dtype: torch.dtype) -> Tensor:
    binoms = torch.zeros((_N_MAX, _N_MAX), device=device, dtype=dtype)
    for n in range(_N_MAX):
        for k in range(n + 1):
            binoms[n, k] = float(math.comb(n, k))
    return binoms


def _logit(p: Tensor) -> Tensor:
    p = p.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(p / (1.0 - p))


def compute_relocation(
    opacity_old: Tensor,
    scales_old: Tensor,
    N: Tensor,
    binoms: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Adapted from 3dgs-mcmc `compute_relocation` (Eq. 9) for 2D (u,v) scales.

    Args:
        opacity_old: (K,) in (0,1)
        scales_old: (K,2) positive
        N: (K,) number of copies (>=1)
        binoms: (_N_MAX,_N_MAX) binomial coefficients (float)
    """
    device = opacity_old.device
    dtype = opacity_old.dtype

    N = N.to(torch.long).clamp(min=1, max=_N_MAX - 1)
    opacity_new = 1.0 - torch.pow(1.0 - opacity_old, 1.0 / N.to(dtype))

    p = torch.arange(1, _N_MAX, device=device, dtype=torch.long)
    p_f = p.to(dtype)
    sign = torch.where((p % 2) == 1, torch.ones_like(p_f), -torch.ones_like(p_f))
    coeff_base = sign / torch.sqrt(p_f)

    binom = binoms[N][:, p].to(dtype)
    denom = (binom * coeff_base * (opacity_new.unsqueeze(1) ** p_f)).sum(dim=1)
    coeff = opacity_old / denom.clamp_min(torch.finfo(dtype).eps)
    scales_new = scales_old * coeff.unsqueeze(1)
    return opacity_new, scales_new


def init_relocation_buffers(device: torch.device, dtype: torch.dtype) -> Tensor:
    return _make_binoms(device=device, dtype=dtype)


def opacity_to_logits(opacity: Tensor) -> Tensor:
    return _logit(opacity)
