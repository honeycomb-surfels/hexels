"""
Tile-based Hexel rasterization using Image-GS `gsplat` infrastructure.

This module adds a gsplat-style top-K normalized renderer for 2D affine hexagons
(rotation + anisotropic scale). It reuses gsplat's bin/sort tile infrastructure
and swaps the per-primitive evaluation from a Gaussian EWA to the Hexel SDF.

Important:
  - Requires the built hexel-kernel package (import name `gsplat`), whose
    extension includes the hexel kernels under `gsplat/cuda/csrc/`.
  - Supports flat colors, optional per-primitive linear color gradients
    (rotation+scale mode), and a full 2×2 affine mode (flat colors or gradients).
  - `topk` is compile-time in gsplat (TOP_K=10); use `topk=10`.
"""

from __future__ import annotations

import importlib
import math
from types import ModuleType
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function


def _load_gsplat_api() -> ModuleType:
    """Load the installed hexel-kernel `gsplat` API and reject namespace shadowing."""
    try:
        import gsplat as gsplat_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `gsplat` from the hexel-kernel package. From the hexels checkout, run "
            "`python -m pip install --no-build-isolation submodules/hexel-kernel` on a CUDA machine."
        ) from exc

    # A `gsplat` directory without `__init__.py` on sys.path, such as the
    # upstream `image-gs/gsplat` checkout, imports as an empty namespace package.
    if getattr(gsplat_module, "__file__", None) is None:
        raise ImportError(
            f"`gsplat` resolved to the namespace directories {list(getattr(gsplat_module, '__path__', []))} "
            "instead of the hexel-kernel package. Import gsplat before adding such a directory to sys.path."
        )

    return gsplat_module


_gsplat_api = _load_gsplat_api()
_gsplat_cuda = importlib.import_module("gsplat.cuda")

TOP_K = 10  # compile-time constant in gsplat CUDA kernels


def _compute_hexel_radii(scales: Tensor, softness: float | Tensor) -> Tensor:
    """
    Compute conservative per-hexel pixel radii for gsplat tile binning.

    We mirror gsplat's 3-sigma cutoff for Gaussians by choosing a distance where
    `sigmoid(-dist * softness)` drops to ~exp(-4.5) (same as 3-sigma Gaussian),
    i.e. `dist * softness ~= 4.5`.

    Args:
        scales: (N, 2) hexel scales in pixels.
        softness: scalar softness (>0) or per-primitive tensor (N,) / (N,3).

    Returns:
        radii: (N,) int32 radii in pixels.
    """
    if scales.ndim != 2 or scales.shape[1] != 2:
        raise ValueError(f"Expected scales with shape (N,2), got {tuple(scales.shape)}")
    if isinstance(softness, torch.Tensor):
        if softness.numel() == 0:
            raise ValueError("softness tensor must be non-empty")
        if softness.ndim == 0 or softness.numel() == 1:
            softness_eff = torch.full(
                (scales.shape[0],), float(softness.detach().item()), device=scales.device, dtype=scales.dtype
            )
        elif softness.ndim == 1:
            if softness.shape[0] != scales.shape[0]:
                raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            softness_eff = softness.to(device=scales.device, dtype=scales.dtype)
        elif softness.ndim == 2 and softness.shape[1] == 3:
            if softness.shape[0] != scales.shape[0]:
                raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            softness_eff = softness.to(device=scales.device, dtype=scales.dtype).amin(dim=1)
        else:
            raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
        softness_eff = softness_eff.clamp_min(1e-6)
    else:
        s = float(softness)
        if s <= 0:
            raise ValueError(f"softness must be > 0, got {softness!r}")
        softness_eff = torch.full((scales.shape[0],), s, device=scales.device, dtype=scales.dtype)

    # Base circumscribed radius for pointy-top hexagon in pixel space.
    # The maximal extent along the u-axis is (2/sqrt(3))*scale_x; along v-axis is scale_y.
    base = (2.0 / math.sqrt(3.0)) * torch.amax(scales.abs(), dim=-1)
    margin = 4.5 / softness_eff
    radii = torch.ceil(base * (1.0 + margin)).to(torch.int32)
    return torch.clamp(radii, min=1)


def _compute_num_tiles_hit(
    xys: Tensor,
    radii: Tensor,
    img_height: int,
    img_width: int,
    block_h: int,
    block_w: int,
) -> Tensor:
    """Compute per-primitive tile hit counts (non-differentiable)."""
    if xys.ndim != 2 or xys.shape[1] != 2:
        raise ValueError(f"Expected xys with shape (N,2), got {tuple(xys.shape)}")
    if radii.ndim != 1:
        raise ValueError(f"Expected radii with shape (N,), got {tuple(radii.shape)}")

    tile_bounds_x = (img_width + block_w - 1) // block_w
    tile_bounds_y = (img_height + block_h - 1) // block_h

    x = xys[:, 0]
    y = xys[:, 1]
    r = radii.to(dtype=xys.dtype)

    tile_center_x = x / float(block_w)
    tile_center_y = y / float(block_h)
    tile_radius_x = r / float(block_w)
    tile_radius_y = r / float(block_h)

    tile_min_x = torch.clamp((tile_center_x - tile_radius_x).to(torch.int32), 0, tile_bounds_x)
    tile_max_x = torch.clamp((tile_center_x + tile_radius_x + 1.0).to(torch.int32), 0, tile_bounds_x)
    tile_min_y = torch.clamp((tile_center_y - tile_radius_y).to(torch.int32), 0, tile_bounds_y)
    tile_max_y = torch.clamp((tile_center_y + tile_radius_y + 1.0).to(torch.int32), 0, tile_bounds_y)

    num_tiles_hit = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y)
    num_tiles_hit = torch.where(radii > 0, num_tiles_hit, torch.zeros_like(num_tiles_hit))
    return num_tiles_hit.to(torch.int32)


def _compute_hexel_radii_affine(affines: Tensor, softness: float | Tensor) -> Tensor:
    """
    Compute conservative per-hexel pixel radii for full 2x2 affine hexels.

    We bound the maximum pixel displacement by using the Frobenius norm of
    the inverse affine matrix (||A^{-1}||_2 <= ||A^{-1}||_F).

    Args:
        affines: (N, 2, 2) affine matrices A used in `delta @ A`.
        softness: scalar softness (>0) or per-primitive tensor (N,) / (N,3).

    Returns:
        radii: (N,) int32 radii in pixels.
    """
    if affines.ndim != 3 or affines.shape[1:] != (2, 2):
        raise ValueError(f"Expected affines with shape (N,2,2), got {tuple(affines.shape)}")
    if isinstance(softness, torch.Tensor):
        if softness.numel() == 0:
            raise ValueError("softness tensor must be non-empty")
        if softness.ndim == 0 or softness.numel() == 1:
            softness_eff = torch.full(
                (affines.shape[0],), float(softness.detach().item()), device=affines.device, dtype=affines.dtype
            )
        elif softness.ndim == 1:
            if softness.shape[0] != affines.shape[0]:
                raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            softness_eff = softness.to(device=affines.device, dtype=affines.dtype)
        elif softness.ndim == 2 and softness.shape[1] == 3:
            if softness.shape[0] != affines.shape[0]:
                raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            softness_eff = softness.to(device=affines.device, dtype=affines.dtype).amin(dim=1)
        else:
            raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
        softness_eff = softness_eff.clamp_min(1e-6)
    else:
        s = float(softness)
        if s <= 0:
            raise ValueError(f"softness must be > 0, got {softness!r}")
        softness_eff = torch.full((affines.shape[0],), s, device=affines.device, dtype=affines.dtype)

    a00 = affines[:, 0, 0]
    a01 = affines[:, 0, 1]
    a10 = affines[:, 1, 0]
    a11 = affines[:, 1, 1]

    det = a00 * a11 - a01 * a10
    det_abs = det.abs().clamp(min=1e-8)

    # Inverse(A) = (1/det) * [[a11, -a01], [-a10, a00]]
    b00 = a11 / det_abs
    b01 = (-a01) / det_abs
    b10 = (-a10) / det_abs
    b11 = a00 / det_abs

    # Frobenius norm upper-bounds operator norm.
    inv_frob = torch.sqrt(b00 * b00 + b01 * b01 + b10 * b10 + b11 * b11)

    base = (2.0 / math.sqrt(3.0)) * inv_frob
    margin = 4.5 / softness_eff
    radii = torch.ceil(base * (1.0 + margin)).to(torch.int32)
    return torch.clamp(radii, min=1)


def rasterize_hexels_sum(
    xys: Tensor,
    scales: Tensor,
    rotations: Tensor,
    colors: Tensor,
    img_height: int,
    img_width: int,
    *,
    softness: float | Tensor,
    aa_kernel_size: float = 0.0,
    sharpness: Optional[Tensor] = None,
    emission: Optional[Tensor] = None,
    color_grads: Optional[Tensor] = None,
    color_grad_scale: float = 0.1,
    block_h: int = 16,
    block_w: int = 16,
    topk_norm: bool = True,
    return_aux: bool = False,
    return_topk: bool = False,
) -> Tensor | Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Rasterize 2D hexels with gsplat-style tile binning and top-K normalization.

    Args:
        xys: (N, 2) hexel centers in pixel coordinates.
        scales: (N, 2) hexel scales in pixels.
        rotations: (N, 1) rotation angles in radians.
        colors: (N, C) per-hexel features/colors (unconstrained).
        img_height/img_width: output image size.
        softness: Either a scalar softness (float), or a per-primitive tensor:
            - (N,) isotropic softness per primitive
            - (N,3) anisotropic softness per primitive (3 edge orientations)
        aa_kernel_size: Anti-aliasing prefilter strength in pixel^2 (0 disables).
        sharpness: Optional (N, 1) tensor controlling Gaussian peak at center per primitive.
            None means flat hexels, positive values give peaked hexels for specular highlights.
        emission: Optional (N, C) additive emission features/colors (non-negative recommended).
        block_h/block_w: gsplat tile size (must match compiled constants; default 16).
        topk_norm: enable per-pixel top-K normalized blending (TOP_K=10).

    Returns:
        If return_aux is False:
            out_img: (H, W, C) float32 image.
        If return_aux is True:
            (out_img, weights_sum, entropy) where:
              - weights_sum: (H, W) float32 denom used for top-K normalization.
              - entropy: (H, W) float32 entropy of normalized top-K weights.
        If return_topk is True (requires return_aux=True):
            (out_img, weights_sum, entropy, pixel_topk) where:
              - pixel_topk: (H, W, TOP_K) int32 indices used for the top-K set.
    """
    if return_topk and (not return_aux):
        raise ValueError("return_topk=True requires return_aux=True")
    if not topk_norm:
        raise ValueError("gsplat_hexels currently supports topk_norm=True only")
    if block_h != 16 or block_w != 16:
        raise ValueError("gsplat_hexels currently requires block_h=block_w=16 (gsplat compile-time constants)")
    if xys.ndim != 2 or xys.shape[1] != 2:
        raise ValueError(f"Expected xys with shape (N,2), got {tuple(xys.shape)}")
    if scales.ndim != 2 or scales.shape[1] != 2:
        raise ValueError(f"Expected scales with shape (N,2), got {tuple(scales.shape)}")
    if rotations.ndim != 2 or rotations.shape[1] != 1:
        raise ValueError(f"Expected rotations with shape (N,1), got {tuple(rotations.shape)}")
    if colors.ndim != 2:
        raise ValueError(f"Expected colors with shape (N,C), got {tuple(colors.shape)}")
    if emission is not None:
        if emission.ndim != 2 or emission.shape != colors.shape:
            raise ValueError(f"Expected emission with shape {tuple(colors.shape)}, got {tuple(emission.shape)}")
        if not emission.is_cuda:
            raise ValueError("emission tensor must be CUDA when used with gsplat_hexels")
        if emission.dtype != torch.float32:
            raise ValueError(f"emission tensor must be float32 for gsplat_hexels, got {emission.dtype}")
    if color_grads is not None:
        if color_grads.ndim != 3 or color_grads.shape[1] != 2:
            raise ValueError(f"Expected color_grads with shape (N,2,C), got {tuple(color_grads.shape)}")
        if color_grads.shape[0] != xys.shape[0]:
            raise ValueError("color_grads must have the same N as xys")
        if color_grads.shape[2] != colors.shape[1]:
            raise ValueError("color_grads must have the same channel count as colors")
    if not xys.is_cuda:
        raise ValueError("gsplat_hexels requires CUDA tensors")

    aa_kernel_size_f = float(aa_kernel_size)
    if aa_kernel_size_f < 0.0:
        raise ValueError(f"aa_kernel_size must be >= 0, got {aa_kernel_size!r}")

    if isinstance(softness, torch.Tensor):
        if softness.numel() == 1:
            softness_arg: float | Tensor = float(softness.detach().item())
        else:
            if softness.ndim == 1:
                if softness.shape[0] != xys.shape[0]:
                    raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            elif softness.ndim == 2 and softness.shape[1] == 3:
                if softness.shape[0] != xys.shape[0]:
                    raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            else:
                raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            if not softness.is_cuda:
                raise ValueError("softness tensor must be CUDA when used with gsplat_hexels")
            if softness.dtype != torch.float32:
                raise ValueError(f"softness tensor must be float32 for gsplat_hexels, got {softness.dtype}")
            softness_arg = softness
    else:
        softness_arg = float(softness)

    with torch.no_grad():
        scales_eff = torch.sqrt(scales * scales + aa_kernel_size_f) if aa_kernel_size_f > 0.0 else scales
        radii = _compute_hexel_radii(scales_eff, softness=softness_arg)
        num_tiles_hit = _compute_num_tiles_hit(
            xys=xys,
            radii=radii,
            img_height=int(img_height),
            img_width=int(img_width),
            block_h=int(block_h),
            block_w=int(block_w),
        )

    if color_grads is None:
        out_img, weights_sum, entropy, pixel_topk = _RasterizeHexelsSum.apply(
            xys.contiguous(),
            radii.contiguous(),
            scales.contiguous(),
            rotations.contiguous(),
            num_tiles_hit.contiguous(),
            colors.contiguous(),
            int(img_height),
            int(img_width),
            int(block_h),
            int(block_w),
            bool(topk_norm),
            float(aa_kernel_size_f),
            softness_arg,
            sharpness.contiguous() if sharpness is not None else None,
            emission.contiguous() if emission is not None else None,
        )
        if not return_aux:
            return out_img
        if return_topk:
            return out_img, weights_sum, entropy, pixel_topk
        return out_img, weights_sum, entropy

    out_img, weights_sum, entropy, pixel_topk = _RasterizeHexelsSumColorGrad.apply(
        xys.contiguous(),
        radii.contiguous(),
        scales.contiguous(),
        rotations.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        color_grads.contiguous(),
        float(color_grad_scale),
        int(img_height),
        int(img_width),
        int(block_h),
        int(block_w),
        bool(topk_norm),
        float(aa_kernel_size_f),
        softness_arg,
        sharpness.contiguous() if sharpness is not None else None,
        emission.contiguous() if emission is not None else None,
    )
    if not return_aux:
        return out_img
    if return_topk:
        return out_img, weights_sum, entropy, pixel_topk
    return out_img, weights_sum, entropy


def rasterize_hexels_affine_sum(
    xys: Tensor,
    affines: Tensor,
    colors: Tensor,
    img_height: int,
    img_width: int,
    *,
    softness: float | Tensor,
    aa_kernel_size: float = 0.0,
    sharpness: Optional[Tensor] = None,
    emission: Optional[Tensor] = None,
    color_grads: Optional[Tensor] = None,
    color_grad_scale: float = 0.1,
    block_h: int = 16,
    block_w: int = 16,
    topk_norm: bool = True,
    return_aux: bool = False,
    return_topk: bool = False,
) -> Tensor | Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Rasterize 2D hexels parameterized by full 2x2 affine matrices.
    """
    if return_topk and (not return_aux):
        raise ValueError("return_topk=True requires return_aux=True")
    if block_h != 16 or block_w != 16:
        raise ValueError("gsplat_hexels currently requires block_h=block_w=16 (gsplat compile-time constants)")
    if xys.ndim != 2 or xys.shape[1] != 2:
        raise ValueError(f"Expected xys with shape (N,2), got {tuple(xys.shape)}")
    if affines.ndim != 3 or affines.shape[1:] != (2, 2):
        raise ValueError(f"Expected affines with shape (N,2,2), got {tuple(affines.shape)}")
    if colors.ndim != 2:
        raise ValueError(f"Expected colors with shape (N,C), got {tuple(colors.shape)}")
    if emission is not None:
        if emission.ndim != 2 or emission.shape != colors.shape:
            raise ValueError(f"Expected emission with shape {tuple(colors.shape)}, got {tuple(emission.shape)}")
        if not emission.is_cuda:
            raise ValueError("emission tensor must be CUDA when used with gsplat_hexels")
        if emission.dtype != torch.float32:
            raise ValueError(f"emission tensor must be float32 for gsplat_hexels, got {emission.dtype}")
    if color_grads is not None:
        if color_grads.ndim != 3 or color_grads.shape[1] != 2:
            raise ValueError(f"Expected color_grads with shape (N,2,C), got {tuple(color_grads.shape)}")
        if color_grads.shape[0] != xys.shape[0]:
            raise ValueError("color_grads must have the same N as xys")
        if color_grads.shape[2] != colors.shape[1]:
            raise ValueError("color_grads must have the same channel count as colors")
    if not xys.is_cuda:
        raise ValueError("gsplat_hexels requires CUDA tensors")

    aa_kernel_size_f = float(aa_kernel_size)
    if aa_kernel_size_f < 0.0:
        raise ValueError(f"aa_kernel_size must be >= 0, got {aa_kernel_size!r}")

    if isinstance(softness, torch.Tensor):
        if softness.numel() == 1:
            softness_arg: float | Tensor = float(softness.detach().item())
        else:
            if softness.ndim == 1:
                if softness.shape[0] != xys.shape[0]:
                    raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            elif softness.ndim == 2 and softness.shape[1] == 3:
                if softness.shape[0] != xys.shape[0]:
                    raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            else:
                raise ValueError("softness must have shape (N,), (N,3) or be a scalar")
            if not softness.is_cuda:
                raise ValueError("softness tensor must be CUDA when used with gsplat_hexels")
            if softness.dtype != torch.float32:
                raise ValueError(f"softness tensor must be float32 for gsplat_hexels, got {softness.dtype}")
            softness_arg = softness
    else:
        softness_arg = float(softness)

    with torch.no_grad():
        if aa_kernel_size_f > 0.0:
            col0_sq = affines[:, 0, 0] ** 2 + affines[:, 1, 0] ** 2
            col1_sq = affines[:, 0, 1] ** 2 + affines[:, 1, 1] ** 2
            s0 = torch.rsqrt(1.0 + aa_kernel_size_f * col0_sq)
            s1 = torch.rsqrt(1.0 + aa_kernel_size_f * col1_sq)
            aff_eff = affines.clone()
            aff_eff[:, 0, 0] *= s0
            aff_eff[:, 1, 0] *= s0
            aff_eff[:, 0, 1] *= s1
            aff_eff[:, 1, 1] *= s1
        else:
            aff_eff = affines
        radii = _compute_hexel_radii_affine(aff_eff, softness=softness_arg)
        num_tiles_hit = _compute_num_tiles_hit(
            xys=xys,
            radii=radii,
            img_height=int(img_height),
            img_width=int(img_width),
            block_h=int(block_h),
            block_w=int(block_w),
        )

    if color_grads is None:
        out_img, weights_sum, entropy, pixel_topk = _RasterizeHexelsSumAffine.apply(
            xys.contiguous(),
            radii.contiguous(),
            affines.contiguous(),
            num_tiles_hit.contiguous(),
            colors.contiguous(),
            int(img_height),
            int(img_width),
            int(block_h),
            int(block_w),
            bool(topk_norm),
            float(aa_kernel_size_f),
            softness_arg,
            sharpness.contiguous() if sharpness is not None else None,
            emission.contiguous() if emission is not None else None,
        )
        if not return_aux:
            return out_img
        if return_topk:
            return out_img, weights_sum, entropy, pixel_topk
        return out_img, weights_sum, entropy

    out_img, weights_sum, entropy, pixel_topk = _RasterizeHexelsSumAffineColorGrad.apply(
        xys.contiguous(),
        radii.contiguous(),
        affines.contiguous(),
        num_tiles_hit.contiguous(),
        colors.contiguous(),
        color_grads.contiguous(),
        float(color_grad_scale),
        int(img_height),
        int(img_width),
        int(block_h),
        int(block_w),
        bool(topk_norm),
        float(aa_kernel_size_f),
        softness_arg,
        sharpness.contiguous() if sharpness is not None else None,
        emission.contiguous() if emission is not None else None,
    )
    if not return_aux:
        return out_img
    if return_topk:
        return out_img, weights_sum, entropy, pixel_topk
    return out_img, weights_sum, entropy


class _RasterizeHexelsSum(Function):
    @staticmethod
    def forward(
        ctx,
        xys: Tensor,
        radii: Tensor,
        scales: Tensor,
        rotations: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        img_height: int,
        img_width: int,
        block_h: int,
        block_w: int,
        topk_norm: bool,
        aa_kernel_size: float,
        softness: float | Tensor,
        sharpness: Optional[Tensor],
        emission: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_points = xys.size(0)
        tile_bounds: Tuple[int, int, int] = (
            (img_width + block_w - 1) // block_w,
            (img_height + block_h - 1) // block_h,
            1,
        )
        block = (block_w, block_h, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = _gsplat_api.compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device, dtype=torch.int32)
            tile_bins = torch.zeros(0, 2, device=xys.device, dtype=torch.int32)
        else:
            (_, _, _, gaussian_ids_sorted, tile_bins) = _gsplat_api.bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.block_h = block_h
        ctx.block_w = block_w
        ctx.num_intersects = num_intersects
        ctx.topk_norm = bool(topk_norm)
        ctx.aa_kernel_size = float(aa_kernel_size)
        ctx.has_sharpness = sharpness is not None
        ctx.has_emission = emission is not None
        ctx.has_softness_tensor = isinstance(softness, torch.Tensor) and softness.numel() > 1
        if ctx.has_softness_tensor:
            ctx.softness_default = 0.0
        else:
            ctx.softness = float(softness)

        if not topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if num_intersects < 1:
            out_img = torch.zeros(
                img_height, img_width, colors.shape[-1], device=xys.device, dtype=torch.float32
            )
            weights_sum = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            entropy = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            pixel_topk = torch.full(
                (img_height, img_width, TOP_K), -1, device=xys.device, dtype=torch.int32
            )
            return out_img, weights_sum, entropy, pixel_topk

        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        if ctx.has_softness_tensor:
            out_img, pixel_topk, weights_sum, entropy = _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_softness(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                scales,
                rotations,
                colors,
                float(ctx.aa_kernel_size),
                softness,
                float(ctx.softness_default),
                sharpness_tensor,
                emission_tensor,
            )
        else:
            out_img, pixel_topk, weights_sum, entropy = _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                scales,
                rotations,
                colors,
                float(ctx.aa_kernel_size),
                float(ctx.softness),
                sharpness_tensor,
                emission_tensor,
            )
        saved_tensors = [
            gaussian_ids_sorted,
            tile_bins,
            xys,
            scales,
            rotations,
            colors,
            pixel_topk.contiguous(),
        ]
        if sharpness is not None:
            saved_tensors.append(sharpness)
        if emission is not None:
            saved_tensors.append(emission)
        if ctx.has_softness_tensor:
            saved_tensors.append(softness)
        ctx.save_for_backward(*saved_tensors)

        return out_img, weights_sum, entropy, pixel_topk

    @staticmethod
    def backward(
        ctx,
        v_out_img: Tensor,
        v_weights_sum: Optional[Tensor],
        v_entropy: Optional[Tensor],
        v_pixel_topk: Optional[Tensor],
    ):
        if ctx.num_intersects < 1:
            # No primitives hit any tiles => zero gradients.
            return (None,) * 15

        if not ctx.topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if v_weights_sum is None:
            v_weights_sum = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)
        if v_entropy is None:
            v_entropy = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)

        saved = ctx.saved_tensors
        idx = 0
        gaussian_ids_sorted = saved[idx]
        idx += 1
        tile_bins = saved[idx]
        idx += 1
        xys = saved[idx]
        idx += 1
        scales = saved[idx]
        idx += 1
        rotations = saved[idx]
        idx += 1
        colors = saved[idx]
        idx += 1
        pixel_topk = saved[idx]
        idx += 1
        sharpness = None
        if ctx.has_sharpness:
            sharpness = saved[idx]
            idx += 1
        emission = None
        if ctx.has_emission:
            emission = saved[idx]
            idx += 1
        softness_tensor: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            softness_tensor = saved[idx]
            idx += 1
        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )

        v_softness: Optional[Tensor] = None
        v_emission: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            assert softness_tensor is not None
            v_xy, v_scale, v_rot, v_colors, v_emission, v_sharpness, v_softness = (
                _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_softness(
                    ctx.img_height,
                    ctx.img_width,
                    ctx.block_h,
                    ctx.block_w,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    scales,
                    rotations,
                    colors,
                    softness_tensor,
                    v_out_img.contiguous(),
                    v_weights_sum.contiguous(),
                    v_entropy.contiguous(),
                    pixel_topk,
                    float(ctx.aa_kernel_size),
                    float(ctx.softness_default),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        else:
            v_xy, v_scale, v_rot, v_colors, v_emission, v_sharpness = _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels(
                ctx.img_height,
                ctx.img_width,
                ctx.block_h,
                ctx.block_w,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                scales,
                rotations,
                colors,
                v_out_img.contiguous(),
                v_weights_sum.contiguous(),
                v_entropy.contiguous(),
                pixel_topk,
                float(ctx.aa_kernel_size),
                float(ctx.softness),
                sharpness_tensor,
                emission_tensor,
            )

        return (
            v_xy,  # xys
            None,  # radii
            v_scale,  # scales
            v_rot,  # rotations
            None,  # num_tiles_hit
            v_colors,  # colors
            None,  # img_height
            None,  # img_width
            None,  # block_h
            None,  # block_w
            None,  # topk_norm
            None,  # aa_kernel_size
            v_softness if ctx.has_softness_tensor else None,  # softness
            v_sharpness if ctx.has_sharpness else None,  # sharpness
            v_emission if ctx.has_emission else None,  # emission
        )


class _RasterizeHexelsSumColorGrad(Function):
    @staticmethod
    def forward(
        ctx,
        xys: Tensor,
        radii: Tensor,
        scales: Tensor,
        rotations: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_grad_scale: float,
        img_height: int,
        img_width: int,
        block_h: int,
        block_w: int,
        topk_norm: bool,
        aa_kernel_size: float,
        softness: float | Tensor,
        sharpness: Optional[Tensor],
        emission: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_points = xys.size(0)
        tile_bounds: Tuple[int, int, int] = (
            (img_width + block_w - 1) // block_w,
            (img_height + block_h - 1) // block_h,
            1,
        )
        block = (block_w, block_h, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = _gsplat_api.compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device, dtype=torch.int32)
            tile_bins = torch.zeros(0, 2, device=xys.device, dtype=torch.int32)
        else:
            (_, _, _, gaussian_ids_sorted, tile_bins) = _gsplat_api.bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.block_h = block_h
        ctx.block_w = block_w
        ctx.num_intersects = num_intersects
        ctx.topk_norm = bool(topk_norm)
        ctx.aa_kernel_size = float(aa_kernel_size)
        ctx.has_sharpness = sharpness is not None
        ctx.has_emission = emission is not None
        ctx.color_grad_scale = float(color_grad_scale)
        ctx.has_softness_tensor = isinstance(softness, torch.Tensor) and softness.numel() > 1
        if ctx.has_softness_tensor:
            ctx.softness_default = 0.0
        else:
            ctx.softness = float(softness)

        if not topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if num_intersects < 1:
            out_img = torch.zeros(
                img_height, img_width, colors.shape[-1], device=xys.device, dtype=torch.float32
            )
            weights_sum = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            entropy = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            pixel_topk = torch.full(
                (img_height, img_width, TOP_K), -1, device=xys.device, dtype=torch.int32
            )
            return out_img, weights_sum, entropy, pixel_topk

        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        if ctx.has_softness_tensor:
            out_img, pixel_topk, weights_sum, entropy = (
                _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_color_grad_softness(
                    tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    scales,
                    rotations,
                    colors,
                    color_grads,
                    float(color_grad_scale),
                    float(ctx.aa_kernel_size),
                    softness,
                    float(ctx.softness_default),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        else:
            out_img, pixel_topk, weights_sum, entropy = _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_color_grad(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                scales,
                rotations,
                colors,
                color_grads,
                float(color_grad_scale),
                float(ctx.aa_kernel_size),
                float(ctx.softness),
                sharpness_tensor,
                emission_tensor,
            )
        saved_tensors = [
            gaussian_ids_sorted,
            tile_bins,
            xys,
            scales,
            rotations,
            colors,
            color_grads,
            pixel_topk.contiguous(),
        ]
        if sharpness is not None:
            saved_tensors.append(sharpness)
        if emission is not None:
            saved_tensors.append(emission)
        if ctx.has_softness_tensor:
            saved_tensors.append(softness)
        ctx.save_for_backward(*saved_tensors)

        return out_img, weights_sum, entropy, pixel_topk

    @staticmethod
    def backward(
        ctx,
        v_out_img: Tensor,
        v_weights_sum: Optional[Tensor],
        v_entropy: Optional[Tensor],
        v_pixel_topk: Optional[Tensor],
    ):
        if ctx.num_intersects < 1:
            return (None,) * 17

        if not ctx.topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if v_weights_sum is None:
            v_weights_sum = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)
        if v_entropy is None:
            v_entropy = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)

        saved = ctx.saved_tensors
        idx = 0
        gaussian_ids_sorted = saved[idx]
        idx += 1
        tile_bins = saved[idx]
        idx += 1
        xys = saved[idx]
        idx += 1
        scales = saved[idx]
        idx += 1
        rotations = saved[idx]
        idx += 1
        colors = saved[idx]
        idx += 1
        color_grads = saved[idx]
        idx += 1
        pixel_topk = saved[idx]
        idx += 1
        sharpness = None
        if ctx.has_sharpness:
            sharpness = saved[idx]
            idx += 1
        emission = None
        if ctx.has_emission:
            emission = saved[idx]
            idx += 1
        softness_tensor: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            softness_tensor = saved[idx]
            idx += 1
        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )

        v_softness: Optional[Tensor] = None
        v_emission: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            assert softness_tensor is not None
            v_xy, v_scale, v_rot, v_colors, v_color_grads, v_emission, v_sharpness, v_softness = (
                _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_color_grad_softness(
                    ctx.img_height,
                    ctx.img_width,
                    ctx.block_h,
                    ctx.block_w,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    scales,
                    rotations,
                    colors,
                    color_grads,
                    softness_tensor,
                    float(ctx.color_grad_scale),
                    v_out_img.contiguous(),
                    v_weights_sum.contiguous(),
                    v_entropy.contiguous(),
                    pixel_topk,
                    float(ctx.aa_kernel_size),
                    float(ctx.softness_default),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        else:
            v_xy, v_scale, v_rot, v_colors, v_color_grads, v_emission, v_sharpness = (
                _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_color_grad(
                    ctx.img_height,
                    ctx.img_width,
                    ctx.block_h,
                    ctx.block_w,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    scales,
                    rotations,
                    colors,
                    color_grads,
                    float(ctx.color_grad_scale),
                    v_out_img.contiguous(),
                    v_weights_sum.contiguous(),
                    v_entropy.contiguous(),
                    pixel_topk,
                    float(ctx.aa_kernel_size),
                    float(ctx.softness),
                    sharpness_tensor,
                    emission_tensor,
                )
            )

        return (
            v_xy,  # xys
            None,  # radii
            v_scale,  # scales
            v_rot,  # rotations
            None,  # num_tiles_hit
            v_colors,  # colors
            v_color_grads,  # color_grads
            None,  # color_grad_scale
            None,  # img_height
            None,  # img_width
            None,  # block_h
            None,  # block_w
            None,  # topk_norm
            None,  # aa_kernel_size
            v_softness if ctx.has_softness_tensor else None,  # softness
            v_sharpness if ctx.has_sharpness else None,  # sharpness
            v_emission if ctx.has_emission else None,  # emission
        )


class _RasterizeHexelsSumAffine(Function):
    @staticmethod
    def forward(
        ctx,
        xys: Tensor,
        radii: Tensor,
        affines: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        img_height: int,
        img_width: int,
        block_h: int,
        block_w: int,
        topk_norm: bool,
        aa_kernel_size: float,
        softness: float | Tensor,
        sharpness: Optional[Tensor],
        emission: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_points = xys.size(0)
        tile_bounds: Tuple[int, int, int] = (
            (img_width + block_w - 1) // block_w,
            (img_height + block_h - 1) // block_h,
            1,
        )
        block = (block_w, block_h, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = _gsplat_api.compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = torch.zeros(img_height, img_width, colors.shape[-1], device=xys.device)
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device, dtype=torch.int32)
            tile_bins = torch.zeros(0, 2, device=xys.device, dtype=torch.int32)
        else:
            (_, _, _, gaussian_ids_sorted, tile_bins) = _gsplat_api.bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                radii,
                cum_tiles_hit,
                tile_bounds,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.block_h = block_h
        ctx.block_w = block_w
        ctx.num_intersects = num_intersects
        ctx.topk_norm = bool(topk_norm)
        ctx.aa_kernel_size = float(aa_kernel_size)
        ctx.has_sharpness = sharpness is not None
        ctx.has_emission = emission is not None
        ctx.has_softness_tensor = isinstance(softness, torch.Tensor) and softness.numel() > 1
        if ctx.has_softness_tensor:
            ctx.softness_default = 0.0
        else:
            ctx.softness = float(softness)

        if not topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if num_intersects < 1:
            out_img = torch.zeros(
                img_height, img_width, colors.shape[-1], device=xys.device, dtype=torch.float32
            )
            weights_sum = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            entropy = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            pixel_topk = torch.full(
                (img_height, img_width, TOP_K), -1, device=xys.device, dtype=torch.int32
            )
            return out_img, weights_sum, entropy, pixel_topk

        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        if ctx.has_softness_tensor:
            out_img, pixel_topk, weights_sum, entropy = _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_affine_softness(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                affines,
                colors,
                float(ctx.aa_kernel_size),
                softness,
                float(ctx.softness_default),
                sharpness_tensor,
                emission_tensor,
            )
        else:
            out_img, pixel_topk, weights_sum, entropy = _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_affine(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                affines,
                colors,
                float(ctx.aa_kernel_size),
                float(ctx.softness),
                sharpness_tensor,
                emission_tensor,
            )
        saved_tensors = [
            gaussian_ids_sorted,
            tile_bins,
            xys,
            affines,
            colors,
            pixel_topk.contiguous(),
        ]
        if sharpness is not None:
            saved_tensors.append(sharpness)
        if emission is not None:
            saved_tensors.append(emission)
        if ctx.has_softness_tensor:
            saved_tensors.append(softness)
        ctx.save_for_backward(*saved_tensors)

        return out_img, weights_sum, entropy, pixel_topk

    @staticmethod
    def backward(
        ctx,
        v_out_img: Tensor,
        v_weights_sum: Optional[Tensor],
        v_entropy: Optional[Tensor],
        v_pixel_topk: Optional[Tensor],
    ):
        if ctx.num_intersects < 1:
            return (None,) * 14

        if not ctx.topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if v_weights_sum is None:
            v_weights_sum = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)
        if v_entropy is None:
            v_entropy = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)

        saved = ctx.saved_tensors
        idx = 0
        gaussian_ids_sorted = saved[idx]
        idx += 1
        tile_bins = saved[idx]
        idx += 1
        xys = saved[idx]
        idx += 1
        affines = saved[idx]
        idx += 1
        colors = saved[idx]
        idx += 1
        pixel_topk = saved[idx]
        idx += 1
        sharpness = None
        if ctx.has_sharpness:
            sharpness = saved[idx]
            idx += 1
        emission = None
        if ctx.has_emission:
            emission = saved[idx]
            idx += 1
        softness_tensor: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            softness_tensor = saved[idx]
            idx += 1
        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )

        v_softness: Optional[Tensor] = None
        v_emission: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            assert softness_tensor is not None
            v_xy, v_affines, v_colors, v_emission, v_sharpness, v_softness = (
                _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_affine_softness(
                    ctx.img_height,
                    ctx.img_width,
                    ctx.block_h,
                    ctx.block_w,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    affines,
                    colors,
                    softness_tensor,
                    v_out_img.contiguous(),
                    v_weights_sum.contiguous(),
                    v_entropy.contiguous(),
                    pixel_topk,
                    float(ctx.aa_kernel_size),
                    float(ctx.softness_default),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        else:
            v_xy, v_affines, v_colors, v_emission, v_sharpness = _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_affine(
                ctx.img_height,
                ctx.img_width,
                ctx.block_h,
                ctx.block_w,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                affines,
                colors,
                v_out_img.contiguous(),
                v_weights_sum.contiguous(),
                v_entropy.contiguous(),
                pixel_topk,
                float(ctx.aa_kernel_size),
                float(ctx.softness),
                sharpness_tensor,
                emission_tensor,
            )

        return (
            v_xy,  # xys
            None,  # radii
            v_affines,  # affines
            None,  # num_tiles_hit
            v_colors,  # colors
            None,  # img_height
            None,  # img_width
            None,  # block_h
            None,  # block_w
            None,  # topk_norm
            None,  # aa_kernel_size
            v_softness if ctx.has_softness_tensor else None,  # softness
            v_sharpness if ctx.has_sharpness else None,  # sharpness
            v_emission if ctx.has_emission else None,  # emission
        )


class _RasterizeHexelsSumAffineColorGrad(Function):
    @staticmethod
    def forward(
        ctx,
        xys: Tensor,
        radii: Tensor,
        affines: Tensor,
        num_tiles_hit: Tensor,
        colors: Tensor,
        color_grads: Tensor,
        color_grad_scale: float,
        img_height: int,
        img_width: int,
        block_h: int,
        block_w: int,
        topk_norm: bool,
        aa_kernel_size: float,
        softness: float | Tensor,
        sharpness: Optional[Tensor],
        emission: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        num_points = xys.size(0)
        tile_bounds: Tuple[int, int, int] = (
            (img_width + block_w - 1) // block_w,
            (img_height + block_h - 1) // block_h,
            1,
        )
        block = (block_w, block_h, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = _gsplat_api.compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = torch.zeros(
                img_height, img_width, colors.shape[-1], device=xys.device, dtype=torch.float32
            )
            weights_sum = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            entropy = torch.zeros(img_height, img_width, device=xys.device, dtype=torch.float32)
            ctx.img_width = img_width
            ctx.img_height = img_height
            ctx.block_h = block_h
            ctx.block_w = block_w
            ctx.num_intersects = num_intersects
            ctx.topk_norm = bool(topk_norm)
            ctx.aa_kernel_size = float(aa_kernel_size)
            ctx.has_sharpness = sharpness is not None
            ctx.color_grad_scale = float(color_grad_scale)
            ctx.has_softness_tensor = isinstance(softness, torch.Tensor) and softness.numel() > 1
            if ctx.has_softness_tensor:
                ctx.softness_default = 0.0
            else:
                ctx.softness = float(softness)
            pixel_topk = torch.full(
                (img_height, img_width, TOP_K), -1, device=xys.device, dtype=torch.int32
            )
            return out_img, weights_sum, entropy, pixel_topk

        (_, _, _, gaussian_ids_sorted, tile_bins) = _gsplat_api.bin_and_sort_gaussians(
            num_points,
            num_intersects,
            xys,
            radii,
            cum_tiles_hit,
            tile_bounds,
        )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.block_h = block_h
        ctx.block_w = block_w
        ctx.num_intersects = num_intersects
        ctx.topk_norm = bool(topk_norm)
        ctx.aa_kernel_size = float(aa_kernel_size)
        ctx.has_sharpness = sharpness is not None
        ctx.has_emission = emission is not None
        ctx.color_grad_scale = float(color_grad_scale)
        ctx.has_softness_tensor = isinstance(softness, torch.Tensor) and softness.numel() > 1
        if ctx.has_softness_tensor:
            ctx.softness_default = 0.0
        else:
            ctx.softness = float(softness)

        if not topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        if ctx.has_softness_tensor:
            out_img, pixel_topk, weights_sum, entropy = (
                _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_affine_color_grad_softness(
                    tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    affines,
                    colors,
                    color_grads,
                    float(color_grad_scale),
                    float(ctx.aa_kernel_size),
                    softness,
                    float(ctx.softness_default),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        else:
            out_img, pixel_topk, weights_sum, entropy = (
                _gsplat_cuda.nd_rasterize_forward_topk_norm_hexels_affine_color_grad(
                    tile_bounds,
                    block,
                    img_size,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    affines,
                    colors,
                    color_grads,
                    float(color_grad_scale),
                    float(ctx.aa_kernel_size),
                    float(ctx.softness),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        saved_tensors = [
            gaussian_ids_sorted,
            tile_bins,
            xys,
            affines,
            colors,
            color_grads,
            pixel_topk.contiguous(),
        ]
        if sharpness is not None:
            saved_tensors.append(sharpness)
        if emission is not None:
            saved_tensors.append(emission)
        if ctx.has_softness_tensor:
            saved_tensors.append(softness)
        ctx.save_for_backward(*saved_tensors)

        return out_img, weights_sum, entropy, pixel_topk

    @staticmethod
    def backward(
        ctx,
        v_out_img: Tensor,
        v_weights_sum: Optional[Tensor],
        v_entropy: Optional[Tensor],
        v_pixel_topk: Optional[Tensor],
    ):
        if ctx.num_intersects < 1:
            return (None,) * 16

        if not ctx.topk_norm:
            raise ValueError("gsplat_hexels currently supports topk_norm=True only")

        if v_weights_sum is None:
            v_weights_sum = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)
        if v_entropy is None:
            v_entropy = torch.zeros((ctx.img_height, ctx.img_width), device=v_out_img.device, dtype=torch.float32)

        saved = ctx.saved_tensors
        idx = 0
        gaussian_ids_sorted = saved[idx]
        idx += 1
        tile_bins = saved[idx]
        idx += 1
        xys = saved[idx]
        idx += 1
        affines = saved[idx]
        idx += 1
        colors = saved[idx]
        idx += 1
        color_grads = saved[idx]
        idx += 1
        pixel_topk = saved[idx]
        idx += 1
        sharpness = None
        if ctx.has_sharpness:
            sharpness = saved[idx]
            idx += 1
        emission = None
        if ctx.has_emission:
            emission = saved[idx]
            idx += 1
        softness_tensor: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            softness_tensor = saved[idx]
            idx += 1
        sharpness_tensor = (
            sharpness if sharpness is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )
        emission_tensor = (
            emission if emission is not None else torch.empty(0, device=xys.device, dtype=torch.float32)
        )

        v_softness: Optional[Tensor] = None
        v_emission: Optional[Tensor] = None
        if ctx.has_softness_tensor:
            assert softness_tensor is not None
            v_xy, v_affines, v_colors, v_color_grads, v_emission, v_sharpness, v_softness = (
                _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_affine_color_grad_softness(
                    ctx.img_height,
                    ctx.img_width,
                    ctx.block_h,
                    ctx.block_w,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    affines,
                    colors,
                    color_grads,
                    softness_tensor,
                    float(ctx.color_grad_scale),
                    v_out_img.contiguous(),
                    v_weights_sum.contiguous(),
                    v_entropy.contiguous(),
                    pixel_topk,
                    float(ctx.aa_kernel_size),
                    float(ctx.softness_default),
                    sharpness_tensor,
                    emission_tensor,
                )
            )
        else:
            v_xy, v_affines, v_colors, v_color_grads, v_emission, v_sharpness = (
                _gsplat_cuda.nd_rasterize_backward_topk_norm_hexels_affine_color_grad(
                    ctx.img_height,
                    ctx.img_width,
                    ctx.block_h,
                    ctx.block_w,
                    gaussian_ids_sorted,
                    tile_bins,
                    xys,
                    affines,
                    colors,
                    color_grads,
                    float(ctx.color_grad_scale),
                    v_out_img.contiguous(),
                    v_weights_sum.contiguous(),
                    v_entropy.contiguous(),
                    pixel_topk,
                    float(ctx.aa_kernel_size),
                    float(ctx.softness),
                    sharpness_tensor,
                    emission_tensor,
                )
            )

        return (
            v_xy,  # xys
            None,  # radii
            v_affines,  # affines
            None,  # num_tiles_hit
            v_colors,  # colors
            v_color_grads,  # color_grads
            None,  # color_grad_scale
            None,  # img_height
            None,  # img_width
            None,  # block_h
            None,  # block_w
            None,  # topk_norm
            None,  # aa_kernel_size
            v_softness if ctx.has_softness_tensor else None,  # softness
            v_sharpness if ctx.has_sharpness else None,  # sharpness
            v_emission if ctx.has_emission else None,  # emission
        )
