import math

import pytest
import torch

from hexels.benchmark_metrics import _aggregate, float_metrics, parse_namespace_config


def test_float_metrics_clamp_without_integer_quantization() -> None:
    prediction = torch.tensor([-0.1, 0.1001, 1.2], dtype=torch.float32).reshape(3, 1, 1)
    target = torch.zeros_like(prediction)
    values = float_metrics(prediction, target,
        lambda prediction, target: torch.tensor(0.75),
        lambda prediction, target: torch.tensor(0.25))
    expected_sse = prediction.clamp(0, 1).double().square().sum().item()
    assert values["rgb_sse"] == expected_sse
    assert values["psnr_db"] == pytest.approx(-10 * math.log10(expected_sse / 3))
    assert values["clipped_element_count"] == 2
    assert values["raw_float_sha256"] != values["metric_float_sha256"]


def test_view_macro_psnr_is_distinct_from_pooled_psnr() -> None:
    rows = [{"elements": 100, "rgb_sse": sse, "psnr_db": psnr, "ssim": .8, "lpips_vgg": .2}
        for sse, psnr in ((1., 20.), (4., 14.))]
    values = _aggregate(rows)
    assert values["mean_psnr_db"] == 17
    assert values["pooled_psnr_db"] == pytest.approx(-10 * math.log10(5 / 200))


def test_lpips_receives_minus_one_to_one_once_while_ssim_receives_rgb() -> None:
    prediction = torch.tensor([-0.1, 0.1001, 1.2], dtype=torch.float32).reshape(3, 1, 1)
    target = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).reshape(3, 1, 1)

    def perceptual(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        assert torch.equal(actual, prediction.clamp(0, 1)[None] * 2 - 1)
        assert torch.equal(reference, target[None] * 2 - 1)
        return torch.tensor(0.25)

    def structural(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        assert torch.equal(actual, prediction.clamp(0, 1))
        assert torch.equal(reference, target)
        return torch.tensor(0.75)

    assert float_metrics(prediction, target, structural, perceptual)["lpips_vgg"] == 0.25


def test_checkpoint_namespace_parser_does_not_execute_code() -> None:
    assert parse_namespace_config("Namespace(eval=True, images=None)") == {"eval": True, "images": None}
    with pytest.raises(ValueError, match="not literal"):
        parse_namespace_config("Namespace(source_path=__import__('os').getcwd())")
