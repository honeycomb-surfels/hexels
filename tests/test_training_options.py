"""Preserve opt-in training controls and canonical held-out metrics."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hexels.options import Hexels3DOptions
from hexels.trainer import Hexels3DTrainer

def test_fast_training_options_can_be_disabled(tmp_path: Path) -> None:
    options = Hexels3DOptions().from_dict({"source_path": str(tmp_path), "model_path": str(tmp_path / "model"),
        "use_fused_ssim": False, "resident_training_targets": False, "memory_poll_interval": 1})
    options.process_args()
    assert not options.use_fused_ssim and not options.resident_training_targets
    options.memory_poll_interval = 0
    with pytest.raises(ValueError, match="memory_poll_interval"):
        options.process_args()

def test_training_defaults_keep_unreleased_optimizations_off() -> None:
    assert Hexels3DOptions().use_fused_ssim is False
    assert Hexels3DOptions().resident_training_targets is False

def test_fast_training_cli_uses_explicit_boolean_values(tmp_path: Path) -> None:
    options = Hexels3DOptions().parse_args(["--source_path", str(tmp_path),
        "--model_path", str(tmp_path / "model"), "--use_fused_ssim", "True",
        "--resident_training_targets", "False"])
    assert options.use_fused_ssim is True
    assert options.resident_training_targets is False

def test_heldout_ssim_uses_canonical_metric_with_a_different_training_loss() -> None:
    trainer = Hexels3DTrainer.__new__(Hexels3DTrainer)
    trainer.device = torch.device("cpu")
    trainer.model = SimpleNamespace(eval=lambda: None)
    trainer.bg_color = torch.zeros(3)
    trainer.renderer = SimpleNamespace(render=lambda *args: {"image": torch.full((3, 7, 9), .4)})
    trainer.metric_ssim = lambda prediction, target: torch.tensor(.9)

    def training_ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Training SSIM must not define reported held-out scores")

    trainer.ssim = training_ssim
    psnr, ssim = trainer._eval_psnr_ssim([SimpleNamespace(original_image=torch.full((3, 7, 9), .5))])
    assert psnr == pytest.approx(20, abs=1e-5)
    assert ssim == pytest.approx(.9)
