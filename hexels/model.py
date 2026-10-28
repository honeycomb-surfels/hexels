from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _logit(p: float) -> float:
    p = float(p)
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class Hexels3DGeometry:
    means3d: Tensor
    scales: Tensor
    rotations: Tensor


def validate_active_mask_state(
    state: Mapping[str, object],
    num_base: int,
) -> None:
    active_mask = state.get("active_mask")
    if active_mask is None:
        return
    if (
        not isinstance(active_mask, Tensor)
        or active_mask.dtype != torch.bool
        or active_mask.shape != (num_base,)
    ):
        raise ValueError(f"active_mask must be bool with shape ({num_base},)")
    if not bool(active_mask.any().item()):
        raise ValueError("active_mask cannot deactivate every primitive")


def _quat_mul_wxyz(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def _rotvec_to_quat_wxyz(rotvec: Tensor) -> Tensor:
    if rotvec.shape[-1] != 3:
        raise ValueError(f"rotvec must have shape (...,3), got {tuple(rotvec.shape)}")
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * theta
    eps = 1e-8
    scale = torch.where(theta > eps, torch.sin(half) / theta, 0.5 - (theta * theta) / 48.0)
    xyz = rotvec * scale
    w = torch.cos(half)
    q = torch.cat([w, xyz], dim=-1)
    return F.normalize(q, dim=-1)


def _quat_to_normal_z(quat_wxyz: Tensor) -> Tensor:
    w, x, y, z = quat_wxyz.unbind(dim=-1)
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    n = torch.stack([nx, ny, nz], dim=-1)
    return F.normalize(n, dim=-1)


class Hexels3DModel(nn.Module):
    def __init__(
        self,
        geometry: Hexels3DGeometry,
        colors_init: Tensor,
        init_opacity: float,
        init_softness: float,
        softness_mode: Literal["iso", "aniso"] = "iso",
        view_basis_dim: int = 3,
    ) -> None:
        super().__init__()
        means3d, scales, rotations = geometry.means3d, geometry.scales, geometry.rotations
        if means3d.ndim != 2 or means3d.shape[1] != 3:
            raise ValueError("means3d must have shape (P,3)")
        if scales.ndim != 2 or scales.shape[1] != 2:
            raise ValueError("scales must have shape (P,2)")
        if rotations.ndim != 2 or rotations.shape[1] != 4:
            raise ValueError("rotations must have shape (P,4)")

        P = means3d.shape[0]
        if colors_init.shape != (P, 3):
            raise ValueError("colors_init must have shape (P,3)")
        if view_basis_dim < 1:
            raise ValueError("view_basis_dim must be positive")

        self.means3d = nn.Parameter(means3d)
        self.log_scales = nn.Parameter(torch.log(scales))
        self.rotation_quats = nn.Parameter(rotations)

        self.colors = nn.Parameter(colors_init)
        self.color_grads = nn.Parameter(torch.zeros((P, 2, 3), device=means3d.device, dtype=means3d.dtype))
        self.color_quads = nn.Parameter(torch.zeros((P, 3, 3), device=means3d.device, dtype=means3d.dtype))
        self.view_grads = nn.Parameter(
            torch.zeros((P, view_basis_dim, 3), device=means3d.device, dtype=means3d.dtype)
        )
        self.specular_ks = nn.Parameter(torch.zeros((P, 3), device=means3d.device, dtype=means3d.dtype))
        self.specular_log_s = nn.Parameter(torch.zeros((P,), device=means3d.device, dtype=means3d.dtype))

        opacity_logit = _logit(init_opacity)
        self.opacity_logits = nn.Parameter(torch.full((P,), opacity_logit, device=means3d.device, dtype=means3d.dtype))

        log_softness = math.log(float(init_softness))
        if softness_mode not in {"iso", "aniso"}:
            raise ValueError("softness_mode must be 'iso' or 'aniso'")
        self.softness_mode = softness_mode
        if softness_mode == "aniso":
            self.log_softness = nn.Parameter(
                torch.full((P, 3), log_softness, device=means3d.device, dtype=means3d.dtype)
            )
        else:
            self.log_softness = nn.Parameter(
                torch.full((P,), log_softness, device=means3d.device, dtype=means3d.dtype)
            )

        self.register_buffer("active_mask", torch.ones((P,), device=means3d.device, dtype=torch.bool))
        self._cached_active_mask_source: Tensor | None = None
        self._cached_active_mask_version = -1
        self._cached_active_base_ids: Tensor | None = None
        self.register_buffer("child_parent_idx", torch.empty((0,), device=means3d.device, dtype=torch.long))
        self.register_buffer("child_sign", torch.empty((0,), device=means3d.device, dtype=means3d.dtype))
        self.register_buffer("child_layer", torch.empty((0,), device=means3d.device, dtype=torch.long))
        self.child_rotvec = nn.Parameter(torch.empty((0, 3), device=means3d.device, dtype=means3d.dtype))
        self.child_colors = nn.Parameter(torch.empty((0, 3), device=means3d.device, dtype=means3d.dtype))
        self.child_color_grads = nn.Parameter(torch.empty((0, 2, 3), device=means3d.device, dtype=means3d.dtype))
        self.child_color_quads = nn.Parameter(torch.empty((0, 3, 3), device=means3d.device, dtype=means3d.dtype))
        self.child_view_grads = nn.Parameter(
            torch.empty((0, view_basis_dim, 3), device=means3d.device, dtype=means3d.dtype)
        )
        self.child_specular_ks = nn.Parameter(torch.empty((0, 3), device=means3d.device, dtype=means3d.dtype))
        self.child_specular_log_s = nn.Parameter(torch.empty((0,), device=means3d.device, dtype=means3d.dtype))
        self.child_opacity_logits = nn.Parameter(torch.empty((0,), device=means3d.device, dtype=means3d.dtype))
        if softness_mode == "aniso":
            self.child_log_softness = nn.Parameter(torch.empty((0, 3), device=means3d.device, dtype=means3d.dtype))
        else:
            self.child_log_softness = nn.Parameter(torch.empty((0,), device=means3d.device, dtype=means3d.dtype))

    @property
    def num_base(self) -> int:
        return int(self.opacity_logits.shape[0])

    def _active_base_ids(self) -> Tensor:
        mask = self.active_mask
        mask_version = int(mask._version)
        base_ids = self._cached_active_base_ids
        if (
            self._cached_active_mask_source is not mask
            or self._cached_active_mask_version != mask_version
            or base_ids is None
        ):
            base_ids = torch.nonzero(mask, as_tuple=False).flatten()
            self._cached_active_mask_source = mask
            self._cached_active_mask_version = mask_version
            self._cached_active_base_ids = base_ids
        return base_ids

    @property
    def num_active_base(self) -> int:
        return int(self._active_base_ids().numel())

    @property
    def num_children(self) -> int:
        return int(self.child_parent_idx.numel())

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self.opacity_logits)

    @property
    def scales(self) -> Tensor:
        return torch.exp(self.log_scales)

    @property
    def rotations(self) -> Tensor:
        return F.normalize(self.rotation_quats, dim=1)

    @property
    def softnesses(self) -> Tensor:
        # Softness is the SDF boundary slope, not an extra geometric thickness.
        return torch.exp(self.log_softness)

    def child_opacities(self) -> Tensor:
        return torch.sigmoid(self.child_opacity_logits)

    def child_softnesses(self) -> Tensor:
        return torch.exp(self.child_log_softness)

    def allocate_children(self, num_children: int) -> None:
        if num_children < 0:
            raise ValueError("num_children must be non-negative")
        device = self.means3d.device
        dtype = self.means3d.dtype
        requires_grad = {
            name: bool(getattr(self, name).requires_grad)
            for name in (
                "child_rotvec",
                "child_colors",
                "child_color_grads",
                "child_color_quads",
                "child_view_grads",
                "child_specular_ks",
                "child_specular_log_s",
                "child_opacity_logits",
                "child_log_softness",
            )
        }
        self.child_parent_idx = torch.zeros((num_children,), device=device, dtype=torch.long)
        self.child_sign = torch.zeros((num_children,), device=device, dtype=dtype)
        self.child_layer = torch.ones((num_children,), device=device, dtype=torch.long)
        parameter_shapes = {
            "child_rotvec": (num_children, 3),
            "child_colors": (num_children, 3),
            "child_color_grads": (num_children, 2, 3),
            "child_color_quads": (num_children, 3, 3),
            "child_view_grads": (num_children, *self.view_grads.shape[1:]),
            "child_specular_ks": (num_children, 3),
            "child_specular_log_s": (num_children,),
            "child_opacity_logits": (num_children,),
            "child_log_softness": (num_children, 3) if self.softness_mode == "aniso" else (num_children,),
        }
        for name, shape in parameter_shapes.items():
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.zeros(shape, device=device, dtype=dtype),
                    requires_grad=requires_grad[name],
                ),
            )

    def append_children(
        self,
        parent_idx: Tensor,
        *,
        sides: Literal["both", "plus", "minus"] = "both",
        layer: Tensor | int | None = None,
    ) -> None:
        if parent_idx.ndim != 1:
            raise ValueError("parent_idx must be a 1D tensor of base indices")
        if parent_idx.numel() == 0:
            return
        if parent_idx.dtype != torch.long:
            parent_idx = parent_idx.to(dtype=torch.long)

        if sides not in {"both", "plus", "minus"}:
            raise ValueError("sides must be 'both', 'plus' or 'minus'")

        parent_idx = parent_idx.to(device=self.means3d.device)
        parent_idx = parent_idx.clamp(min=0, max=self.num_base - 1)
        if layer is None:
            layer_t = torch.ones_like(parent_idx, dtype=torch.long)
        elif isinstance(layer, int):
            layer_t = torch.full_like(parent_idx, int(layer), dtype=torch.long)
        else:
            layer_t = layer.to(device=self.means3d.device, dtype=torch.long)
            if layer_t.ndim != 1 or layer_t.shape[0] != parent_idx.shape[0]:
                raise ValueError("layer must have shape (num_parents,)")
            layer_t = layer_t.clamp_min(1)
        if sides == "both":
            parent_idx2 = parent_idx.repeat_interleave(2)
            layer2 = layer_t.repeat_interleave(2)
            sign = torch.ones((parent_idx2.shape[0],), device=self.means3d.device, dtype=self.means3d.dtype)
            sign[1::2] = -1.0
        else:
            parent_idx2 = parent_idx
            layer2 = layer_t
            sign_val = 1.0 if sides == "plus" else -1.0
            sign = torch.full((parent_idx2.shape[0],), sign_val, device=self.means3d.device, dtype=self.means3d.dtype)

        with torch.no_grad():
            self.child_parent_idx = torch.cat([self.child_parent_idx, parent_idx2], dim=0)
            self.child_sign = torch.cat([self.child_sign, sign], dim=0)
            self.child_layer = torch.cat([self.child_layer, layer2], dim=0)
            self.child_rotvec = nn.Parameter(
                torch.cat(
                    [
                        self.child_rotvec.data,
                        torch.zeros((parent_idx2.shape[0], 3), device=self.means3d.device, dtype=self.means3d.dtype),
                    ],
                    dim=0,
                ),
                requires_grad=bool(self.child_rotvec.requires_grad),
            )

            self.child_colors = nn.Parameter(
                torch.cat([self.child_colors.data, self.colors.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_color_grads = nn.Parameter(
                torch.cat([self.child_color_grads.data, self.color_grads.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_color_quads = nn.Parameter(
                torch.cat([self.child_color_quads.data, self.color_quads.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_view_grads = nn.Parameter(
                torch.cat([self.child_view_grads.data, self.view_grads.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_specular_ks = nn.Parameter(
                torch.cat([self.child_specular_ks.data, self.specular_ks.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_specular_log_s = nn.Parameter(
                torch.cat([self.child_specular_log_s.data, self.specular_log_s.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_opacity_logits = nn.Parameter(
                torch.cat([self.child_opacity_logits.data, self.opacity_logits.data[parent_idx2].clone()], dim=0),
                requires_grad=True,
            )
            self.child_log_softness = nn.Parameter(
                torch.cat([self.child_log_softness.data, self.log_softness.data[parent_idx2].clone()], dim=0),
                requires_grad=bool(self.log_softness.requires_grad),
            )

    def pack_for_rasterizer(
        self,
        *,
        child_offset_frac: float,
        child_scale_ratio: float,
        child_rotation_delta: bool = False,
        detach_child_geometry: bool = True,
    ) -> dict[str, Tensor]:
        # Keep persistent optimizer rows while rendering only active rows.
        # The returned IDs let contribution and growth statistics scatter back.
        base_ids = self._active_base_ids()
        if base_ids.numel() == 0:
            raise ValueError("At least one base primitive must remain active")
        if self.num_children > 0 and base_ids.numel() != self.num_base:
            raise ValueError("Inactive base primitives are not supported while children are present")

        if base_ids.numel() == self.num_base:
            means3d = self.means3d
            scales = self.scales
            rotations = self.rotations
            colors = self.colors
            color_grads = self.color_grads
            color_quads = self.color_quads
            view_grads = self.view_grads
            specular_ks = self.specular_ks
            specular_log_s = self.specular_log_s
            opacities = self.opacities
            softnesses = self.softnesses
        else:
            means3d = self.means3d.index_select(0, base_ids)
            scales = torch.exp(self.log_scales.index_select(0, base_ids))
            rotations = F.normalize(self.rotation_quats.index_select(0, base_ids), dim=1)
            colors = self.colors.index_select(0, base_ids)
            color_grads = self.color_grads.index_select(0, base_ids)
            color_quads = self.color_quads.index_select(0, base_ids)
            view_grads = self.view_grads.index_select(0, base_ids)
            specular_ks = self.specular_ks.index_select(0, base_ids)
            specular_log_s = self.specular_log_s.index_select(0, base_ids)
            opacities = torch.sigmoid(self.opacity_logits.index_select(0, base_ids))
            softnesses = self.softnesses.index_select(0, base_ids)

        if self.num_children < 1:
            return {
                "means3d": means3d,
                "scales": scales,
                "rotations": rotations,
                "colors": colors,
                "color_grads": color_grads,
                "color_quads": color_quads,
                "view_grads": view_grads,
                "specular_ks": specular_ks,
                "specular_log_s": specular_log_s,
                "opacities": opacities,
                "softnesses": softnesses,
                "base_ids": base_ids,
                "primitive_ids": base_ids,
            }

        parent_idx = self.child_parent_idx
        sign = self.child_sign
        layer = self.child_layer.to(dtype=self.means3d.dtype)
        if detach_child_geometry:
            parent_means = means3d.detach()[parent_idx]
            parent_scales = scales.detach()[parent_idx]
            parent_rot = rotations.detach()[parent_idx]
        else:
            parent_means = means3d[parent_idx]
            parent_scales = scales[parent_idx]
            parent_rot = rotations[parent_idx]
        normal = _quat_to_normal_z(parent_rot)
        offset = float(child_offset_frac) * 0.5 * (parent_scales[:, 0] + parent_scales[:, 1]) * layer
        child_means = parent_means + (sign[:, None] * offset[:, None] * normal)
        child_scales = parent_scales * float(child_scale_ratio)
        if child_rotation_delta:
            q_delta = _rotvec_to_quat_wxyz(self.child_rotvec)
            child_rots = _quat_mul_wxyz(parent_rot, q_delta)
        else:
            child_rots = parent_rot

        primitive_ids = torch.arange(
            self.num_base + self.num_children,
            device=self.means3d.device,
            dtype=torch.long,
        )
        return {
            "means3d": torch.cat([means3d, child_means], dim=0),
            "scales": torch.cat([scales, child_scales], dim=0),
            "rotations": torch.cat([rotations, child_rots], dim=0),
            "colors": torch.cat([colors, self.child_colors], dim=0),
            "color_grads": torch.cat([color_grads, self.child_color_grads], dim=0),
            "color_quads": torch.cat([color_quads, self.child_color_quads], dim=0),
            "view_grads": torch.cat([view_grads, self.child_view_grads], dim=0),
            "specular_ks": torch.cat([specular_ks, self.child_specular_ks], dim=0),
            "specular_log_s": torch.cat([specular_log_s, self.child_specular_log_s], dim=0),
            "opacities": torch.cat([opacities, self.child_opacities()], dim=0),
            "softnesses": torch.cat([softnesses, self.child_softnesses()], dim=0),
            "base_ids": base_ids,
            "primitive_ids": primitive_ids,
        }
