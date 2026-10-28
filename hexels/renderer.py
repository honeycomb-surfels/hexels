from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .options import Hexels3DOptions


_SH_C1 = 0.4886025119029199
_SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435)
_SH_COEFFS_BY_DEGREE = (0, 3, 8, 15)


def _eval_linear_view_residual(view_grads: Tensor, dirs: Tensor) -> Tensor:
    return torch.bmm(view_grads, dirs.unsqueeze(-1)).squeeze(-1)


def _uses_linear_view_fast_path(opt: Any) -> bool:
    # Native edge-prefilter fallback omits view-direction gradients.
    return (
        str(getattr(opt, "view_model", "linear")) == "linear"
        and bool(getattr(opt, "use_view_grads", True))
        and not bool(getattr(opt, "prefilter_edge_on", False))
    )


def _scatter_primitive_values(values: Tensor, primitive_ids: Tensor, capacity: int) -> Tensor:
    if values.shape[0] != primitive_ids.numel():
        raise ValueError(
            f"Rasterizer returned {values.shape[0]} primitive values for {primitive_ids.numel()} packed primitives"
        )
    if primitive_ids.numel() == capacity:
        return values
    scattered = values.new_zeros((capacity, *values.shape[1:]))
    return scattered.index_copy(0, primitive_ids, values)


def _map_packed_owner_ids(packed_owner_ids: Tensor, primitive_ids: Tensor) -> Tensor:
    if packed_owner_ids.dtype != torch.long:
        raise ValueError("packed_owner_ids must be int64")
    if primitive_ids.ndim != 1 or primitive_ids.dtype != torch.long:
        raise ValueError("primitive_ids must be a 1D int64 tensor")
    valid = packed_owner_ids >= 0
    valid_ids = packed_owner_ids[valid]
    if valid_ids.numel() > 0 and int(valid_ids.max().item()) >= primitive_ids.numel():
        raise ValueError("packed_owner_ids contains an out-of-range primitive index")
    owner_ids = torch.full_like(packed_owner_ids, -1)
    owner_ids[valid] = primitive_ids.index_select(0, valid_ids)
    return owner_ids


def _eval_sh3_residual(coeffs: Tensor, dirs: Tensor, degree: int = 3) -> Tensor:
    if degree < 0 or degree > 3:
        raise ValueError(f"SH degree must be in [0,3], got {degree}")
    x, y, z = dirs.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    basis = torch.stack(
        [
            -_SH_C1 * y,
            _SH_C1 * z,
            -_SH_C1 * x,
            _SH_C2[0] * xy,
            _SH_C2[1] * yz,
            _SH_C2[2] * (2.0 * zz - xx - yy),
            _SH_C2[3] * xz,
            _SH_C2[4] * (xx - yy),
            _SH_C3[0] * y * (3.0 * xx - yy),
            _SH_C3[1] * xy * z,
            _SH_C3[2] * y * (4.0 * zz - xx - yy),
            _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            _SH_C3[4] * x * (4.0 * zz - xx - yy),
            _SH_C3[5] * z * (xx - yy),
            _SH_C3[6] * x * (xx - 3.0 * yy),
        ],
        dim=-1,
    )
    num_coeffs = _SH_COEFFS_BY_DEGREE[degree]
    return torch.sum(basis[:, :num_coeffs].unsqueeze(-1) * coeffs[:, :num_coeffs], dim=1)


class Hexels3DRenderer:
    def __init__(self, opt: Hexels3DOptions) -> None:
        from gsplat import Hexels3DRasterizer

        self.opt = opt
        self._rasterizer_cls = Hexels3DRasterizer
        self._cache: dict[tuple[int, int, float, float], Any] = {}

    def _get_rasterizer(self, camera: Any) -> Any:
        tan_fovx = math.tan(float(camera.FoVx) * 0.5)
        tan_fovy = math.tan(float(camera.FoVy) * 0.5)
        key = (int(camera.image_width), int(camera.image_height), float(tan_fovx), float(tan_fovy))
        rast = self._cache.get(key)
        if rast is not None:
            return rast

        view_model = str(getattr(self.opt, "view_model", "linear"))
        linear_view_fast_path = _uses_linear_view_fast_path(self.opt)
        native_view_model = "none" if view_model == "sh3" or linear_view_fast_path else view_model
        rast = self._rasterizer_cls(
            image_width=key[0],
            image_height=key[1],
            tan_fovx=key[2],
            tan_fovy=key[3],
            softness=float(self.opt.softness),
            scale_modifier=float(self.opt.scale_modifier),
            color_model=str(getattr(self.opt, "color_model", "linear")),
            view_model=native_view_model,
            use_view_grads=bool(getattr(self.opt, "use_view_grads", True))
            and native_view_model == "linear",
            debug=bool(self.opt.debug),
        )
        self._cache[key] = rast
        return rast

    def render(
        self,
        camera: Any,
        model: Any,
        background: Tensor,
        return_contrib: bool | None = None,
        return_overdraw: bool | None = None,
        return_overdraw_hits: bool | None = None,
        return_surface: bool | None = None,
        return_distortion: bool | None = None,
        return_median_depth: bool = False,
        return_viewspace_points: bool | None = None,
        sh_degree: int = 3,
    ) -> dict[str, Tensor]:
        rasterizer = self._get_rasterizer(camera)
        enable_mcmc = bool(getattr(self.opt, "enable_mcmc", False))
        if return_contrib is None:
            return_contrib = enable_mcmc
        if return_overdraw is None:
            return_overdraw = float(getattr(self.opt, "repulsion_screen_weight", 0.0)) > 0.0
        if return_overdraw_hits is None:
            return_overdraw_hits = False
        if return_surface is None:
            return_surface = False
        if return_distortion is None:
            return_distortion = False
        if return_viewspace_points is None:
            return_viewspace_points = False
        pack = model.pack_for_rasterizer(
            child_offset_frac=float(getattr(self.opt, "child_offset_frac", 0.0)),
            child_scale_ratio=float(getattr(self.opt, "child_scale_ratio", 1.0)),
            child_rotation_delta=bool(getattr(self.opt, "child_rotation_delta", False)),
            detach_child_geometry=bool(getattr(self.opt, "children_detach_geometry", True)),
        )
        packed_count = int(pack["means3d"].shape[0])
        num_base = int(getattr(model, "num_base", packed_count))
        num_children = int(getattr(model, "num_children", max(0, packed_count - num_base)))
        primitive_capacity = max(packed_count, num_base + num_children)
        primitive_ids = pack.get("primitive_ids")
        if primitive_ids is None:
            primitive_ids = torch.arange(packed_count, device=pack["means3d"].device, dtype=torch.long)
        base_ids = pack.get("base_ids")
        active_base_count = num_base if base_ids is None else int(base_ids.numel())

        colors = pack["colors"]
        view_model = str(getattr(self.opt, "view_model", "linear"))
        use_view_grads = bool(getattr(self.opt, "use_view_grads", True))
        linear_view_fast_path = _uses_linear_view_fast_path(self.opt)
        if use_view_grads and (linear_view_fast_path or view_model == "sh3"):
            # V2's linear view term and V3's SH term are evaluated here so
            # autograd also carries their direction dependence to the centers.
            view_anchor3d = pack.get("view_anchor3d", pack["means3d"])
            if view_anchor3d.shape != pack["means3d"].shape:
                raise ValueError("view_anchor3d must match packed means3d")
            view_delta = view_anchor3d - camera.camera_center.unsqueeze(0)
            view_dirs = view_delta * torch.rsqrt(
                torch.sum(view_delta * view_delta, dim=-1, keepdim=True) + 1e-8
            )
            if linear_view_fast_path:
                colors = colors + _eval_linear_view_residual(pack["view_grads"], view_dirs)
            else:
                colors = colors + _eval_sh3_residual(pack["view_grads"], view_dirs, degree=sh_degree)
        viewspace_points = None
        packed_viewspace_points = None
        if return_viewspace_points:
            # These dummy coordinates collect screen-space gradients for growth.
            # Packed IDs map them back to the persistent rows, including masked rows.
            viewspace_points = torch.zeros(
                (primitive_capacity, 4 if bool(getattr(self.opt, "densify_absgrad_rank", False)) else 3),
                device=pack["means3d"].device,
                dtype=pack["means3d"].dtype,
                requires_grad=True,
            )
            packed_viewspace_points = viewspace_points.index_select(0, primitive_ids)
        out = rasterizer(
            means3d=pack["means3d"],
            means2d=packed_viewspace_points,
            scales=pack["scales"],
            rotations=pack["rotations"],
            colors=colors,
            color_grads=pack["color_grads"],
            color_quads=pack["color_quads"],
            view_grads=pack["view_grads"],
            specular_ks=pack["specular_ks"],
            specular_log_s=pack["specular_log_s"],
            opacities=pack["opacities"],
            softnesses=pack["softnesses"],
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            campos=camera.camera_center,
            background=background,
            return_contrib=return_contrib,
            return_overdraw=return_overdraw,
            return_overdraw_hits=return_overdraw_hits,
            return_surface=return_surface,
            return_distortion=return_distortion,
            return_median_depth=return_median_depth,
            overdraw_base_count=active_base_count,
            prefilter_edge_on=bool(getattr(self.opt, "prefilter_edge_on", False)),
            conservative_bounds=bool(getattr(self.opt, "conservative_bounds", False)),
        )
        idx = 0
        out_color = out[idx]
        radii = _scatter_primitive_values(out[idx + 1], primitive_ids, primitive_capacity)
        idx += 2
        result: dict[str, Tensor] = {"image": out_color, "radii": radii}
        if return_contrib:
            result["contrib"] = _scatter_primitive_values(out[idx], primitive_ids, primitive_capacity)
            idx += 1
        if return_overdraw:
            result["overdraw"] = out[idx]
            idx += 1
        if return_overdraw_hits:
            result["overdraw_hits"] = out[idx]
            idx += 1
        if return_surface:
            result["surf_depth"] = out[idx]
            result["surf_alpha"] = out[idx + 1]
            result["surf_normals"] = out[idx + 2]
            idx += 3
            if return_distortion:
                result["surf_distortion"] = out[idx]
                idx += 1
            if return_median_depth:
                result["surf_median_depth"] = out[idx]
                idx += 1
        if viewspace_points is not None:
            result["viewspace_points"] = viewspace_points
        return result

    @torch.no_grad()
    def render_owner(
        self,
        camera: Any,
        model: Any,
        background: Tensor,
        sh_degree: int = 3,
    ) -> dict[str, Tensor]:
        """Render the maximum-compositing-weight owner for every pixel."""
        rasterizer = self._get_rasterizer(camera)
        pack = model.pack_for_rasterizer(
            child_offset_frac=float(getattr(self.opt, "child_offset_frac", 0.0)),
            child_scale_ratio=float(getattr(self.opt, "child_scale_ratio", 1.0)),
            child_rotation_delta=bool(getattr(self.opt, "child_rotation_delta", False)),
            detach_child_geometry=bool(getattr(self.opt, "children_detach_geometry", True)),
        )
        packed_count = int(pack["means3d"].shape[0])
        num_base = int(getattr(model, "num_base", packed_count))
        num_children = int(getattr(model, "num_children", max(0, packed_count - num_base)))
        primitive_capacity = max(packed_count, num_base + num_children)
        primitive_ids = pack.get("primitive_ids")
        if primitive_ids is None:
            primitive_ids = torch.arange(packed_count, device=pack["means3d"].device, dtype=torch.long)

        colors = pack["colors"]
        view_model = str(getattr(self.opt, "view_model", "linear"))
        use_view_grads = bool(getattr(self.opt, "use_view_grads", True))
        linear_view_fast_path = _uses_linear_view_fast_path(self.opt)
        if use_view_grads and (linear_view_fast_path or view_model == "sh3"):
            view_anchor3d = pack.get("view_anchor3d", pack["means3d"])
            if view_anchor3d.shape != pack["means3d"].shape:
                raise ValueError("view_anchor3d must match packed means3d")
            view_delta = view_anchor3d - camera.camera_center.unsqueeze(0)
            view_dirs = view_delta * torch.rsqrt(
                torch.sum(view_delta * view_delta, dim=-1, keepdim=True) + 1e-8
            )
            if linear_view_fast_path:
                colors = colors + _eval_linear_view_residual(pack["view_grads"], view_dirs)
            else:
                colors = colors + _eval_sh3_residual(pack["view_grads"], view_dirs, degree=sh_degree)

        (
            image,
            packed_radii,
            packed_owner_ids,
            owner_weight,
            owner_uv,
            owner_sdf,
            owner_depth,
            alpha,
        ) = rasterizer.render_owner(
            means3d=pack["means3d"],
            scales=pack["scales"],
            rotations=pack["rotations"],
            colors=colors,
            color_grads=pack["color_grads"],
            color_quads=pack["color_quads"],
            view_grads=pack["view_grads"],
            specular_ks=pack["specular_ks"],
            specular_log_s=pack["specular_log_s"],
            opacities=pack["opacities"],
            softnesses=pack["softnesses"],
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            campos=camera.camera_center,
            background=background,
            prefilter_edge_on=bool(getattr(self.opt, "prefilter_edge_on", False)),
            conservative_bounds=bool(getattr(self.opt, "conservative_bounds", False)),
        )
        return {
            "image": image,
            "radii": _scatter_primitive_values(packed_radii, primitive_ids, primitive_capacity),
            "packed_owner_id": packed_owner_ids,
            "owner_id": _map_packed_owner_ids(packed_owner_ids, primitive_ids),
            "owner_weight": owner_weight,
            "owner_uv": owner_uv,
            "owner_sdf": owner_sdf,
            "owner_depth": owner_depth,
            "alpha": alpha,
        }
