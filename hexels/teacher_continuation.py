from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import Hexels3DGeometry, Hexels3DModel
from .renderer import _eval_sh3_residual
from .teacher_transcode import teacher_dc_rgb


class TrainableTeacherHexelModel(Hexels3DModel):
    """Trainable Hexel state initialized from a 2DGS SH3 teacher."""

    def __init__(
        self,
        geometry: Hexels3DGeometry,
        colors_init: Tensor,
        opacity_logits: Tensor,
        sh_coefficients: Tensor,
        softness: float,
    ) -> None:
        super().__init__(
            geometry=geometry,
            colors_init=colors_init,
            init_opacity=0.1,
            init_softness=softness,
            view_basis_dim=15,
        )
        if opacity_logits.shape != (self.num_base,):
            raise ValueError("opacity_logits must have shape (P,)")
        if sh_coefficients.shape != (self.num_base, 15, 3):
            raise ValueError("sh_coefficients must have shape (P,15,3)")
        with torch.no_grad():
            self.opacity_logits.copy_(opacity_logits)
            self.view_grads.copy_(sh_coefficients)
        self.register_buffer(
            "_reference_log_softness",
            self.log_softness.detach()[0].clone(),
        )
        self.register_buffer(
            "_reference_softness",
            self.log_softness.detach().new_tensor(float(softness)),
        )
        self._camera_center: Tensor | None = None

    @property
    def softnesses(self) -> Tensor:
        return self._reference_softness * torch.exp(
            self.log_softness - self._reference_log_softness
        )

    def set_camera_center(self, camera_center: Tensor) -> None:
        if camera_center.shape != (3,):
            raise ValueError("camera_center must have shape (3,)")
        self._camera_center = camera_center

    def pack_for_rasterizer(self, **kwargs: object) -> dict[str, Tensor]:
        if self._camera_center is None:
            raise ValueError("Set the camera center before rendering")
        pack = super().pack_for_rasterizer(**kwargs)
        anchors = pack.get("view_anchor3d", pack["means3d"])
        directions = F.normalize(
            anchors - self._camera_center.unsqueeze(0),
            dim=1,
        )
        pack["colors"] = torch.clamp_min(
            pack["colors"] + _eval_sh3_residual(pack["view_grads"], directions),
            0.0,
        )
        return pack


def trainable_hexels_from_2dgs(
    teacher: Any,
    *,
    softness: float,
) -> TrainableTeacherHexelModel:
    means3d = teacher._xyz.detach().clone()
    log_scales = teacher._scaling.detach()
    rotations = teacher._rotation.detach()
    opacity_logits = teacher._opacity.detach()
    features_dc = teacher._features_dc.detach()
    features_rest = teacher._features_rest.detach()
    population = int(means3d.shape[0])
    if (
        means3d.shape != (population, 3)
        or log_scales.shape != (population, 2)
        or rotations.shape != (population, 4)
        or opacity_logits.shape != (population, 1)
        or features_dc.shape != (population, 1, 3)
        or features_rest.shape != (population, 15, 3)
    ):
        raise ValueError("2DGS teacher tensors have incompatible shapes")
    tensors = (
        means3d,
        log_scales,
        rotations,
        opacity_logits,
        features_dc,
        features_rest,
    )
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("2DGS teacher tensors must be finite")
    geometry = Hexels3DGeometry(
        means3d=means3d,
        scales=torch.exp(log_scales).clone(),
        rotations=rotations.clone(),
    )
    model = TrainableTeacherHexelModel(
        geometry=geometry,
        colors_init=teacher_dc_rgb(features_dc).clone(),
        opacity_logits=opacity_logits[:, 0].clone(),
        sh_coefficients=features_rest.clone(),
        softness=softness,
    )
    with torch.no_grad():
        model.log_scales.copy_(log_scales)
        model.rotation_quats.copy_(rotations)
    return model
