from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

from .renderer import _eval_linear_view_residual, _eval_sh3_residual


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def teacher_dc_rgb(features_dc: torch.Tensor) -> torch.Tensor:
    if features_dc.ndim != 3 or features_dc.shape[1:] != (1, 3):
        raise ValueError("features_dc must have shape (P,1,3)")
    return 0.5 + SH_C0 * features_dc[:, 0]


def teacher_sh1_to_linear(features_rest: torch.Tensor) -> torch.Tensor:
    if features_rest.ndim != 3 or features_rest.shape[1] < 3 or features_rest.shape[2] != 3:
        raise ValueError("features_rest must have shape (P,K,3) with K >= 3")
    coefficients = features_rest[:, :3]
    linear = coefficients.new_empty((coefficients.shape[0], 3, 3))
    linear[:, :, 0] = -SH_C1 * coefficients[:, 2]
    linear[:, :, 1] = -SH_C1 * coefficients[:, 0]
    linear[:, :, 2] = SH_C1 * coefficients[:, 1]
    return linear


def teacher_v3_colors(
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    if features_rest.ndim != 3 or features_rest.shape[1:] != (15, 3):
        raise ValueError("V3 features_rest must have shape (P,15,3)")
    if directions.shape != (features_dc.shape[0], 3):
        raise ValueError("directions must have shape (P,3)")
    colors = teacher_dc_rgb(features_dc)
    colors = colors + _eval_sh3_residual(features_rest, directions, degree=3)
    return torch.clamp_min(colors, 0.0)


def teacher_v2_colors(
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    if directions.shape != (features_dc.shape[0], 3):
        raise ValueError("directions must have shape (P,3)")
    colors = teacher_dc_rgb(features_dc)
    colors = colors + _eval_linear_view_residual(
        teacher_sh1_to_linear(features_rest),
        directions,
    )
    return torch.clamp_min(colors, 0.0)


class TeacherHexelModel:
    def __init__(
        self,
        *,
        means3d: torch.Tensor,
        log_scales: torch.Tensor,
        rotations: torch.Tensor,
        opacity_logits: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        variant: Literal["v2", "v3"],
        softness: float,
    ) -> None:
        population = int(means3d.shape[0])
        if (
            means3d.shape != (population, 3)
            or log_scales.ndim != 2
            or log_scales.shape[0] != population
            or log_scales.shape[1] < 2
            or rotations.shape != (population, 4)
            or opacity_logits.numel() != population
            or features_dc.shape != (population, 1, 3)
            or features_rest.shape != (population, 15, 3)
        ):
            raise ValueError("2DGS teacher tensors have incompatible shapes")
        values = (
            means3d,
            log_scales,
            rotations,
            opacity_logits,
            features_dc,
            features_rest,
        )
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError("2DGS teacher tensors must be finite")
        if variant not in {"v2", "v3"}:
            raise ValueError("variant must be v2 or v3")
        if softness <= 0.0:
            raise ValueError("softness must be positive")

        self.means3d = means3d.detach()
        self.scales = torch.exp(log_scales[:, :2]).detach()
        self.rotations = F.normalize(rotations.detach(), dim=1)
        self.opacities = torch.sigmoid(opacity_logits.reshape(population)).detach()
        self.features_dc = features_dc.detach()
        self.features_rest = features_rest.detach()
        self.variant = variant
        self.softnesses = means3d.new_full((population,), float(softness))
        self._camera_center: torch.Tensor | None = None
        self._base_ids = torch.arange(
            population,
            device=means3d.device,
            dtype=torch.long,
        )
        self._color_grads = means3d.new_zeros((population, 2, 3))
        self._color_quads = means3d.new_zeros((population, 3, 3))
        self._view_grads = means3d.new_zeros((population, 3, 3))
        self._specular_ks = means3d.new_zeros((population, 3))
        self._specular_log_s = means3d.new_zeros((population,))

    @property
    def num_base(self) -> int:
        return int(self.means3d.shape[0])

    @property
    def num_children(self) -> int:
        return 0

    def set_camera_center(self, camera_center: torch.Tensor) -> None:
        if camera_center.shape != (3,):
            raise ValueError("camera_center must have shape (3,)")
        self._camera_center = camera_center

    def set_softness(self, softness: float) -> None:
        if softness <= 0.0:
            raise ValueError("softness must be positive")
        self.softnesses.fill_(float(softness))

    def pack_for_rasterizer(self, **_: object) -> dict[str, torch.Tensor]:
        if self._camera_center is None:
            raise ValueError("Set the camera center before rendering")
        directions = F.normalize(
            self.means3d - self._camera_center.unsqueeze(0),
            dim=1,
        )
        colors = (
            teacher_v3_colors(
                self.features_dc,
                self.features_rest,
                directions,
            )
            if self.variant == "v3"
            else teacher_v2_colors(
                self.features_dc,
                self.features_rest,
                directions,
            )
        )
        return {
            "means3d": self.means3d,
            "scales": self.scales,
            "rotations": self.rotations,
            "colors": colors,
            "color_grads": self._color_grads,
            "color_quads": self._color_quads,
            "view_grads": self._view_grads,
            "specular_ks": self._specular_ks,
            "specular_log_s": self._specular_log_s,
            "opacities": self.opacities,
            "softnesses": self.softnesses,
            "base_ids": self._base_ids,
            "primitive_ids": self._base_ids,
        }
