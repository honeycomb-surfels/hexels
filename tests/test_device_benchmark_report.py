"""Independent guards for the exported iPad quality and battery analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE = Path(__file__).resolve().parents[1] / "hexel_edge/tools/summarize_device_benchmark.py"
SPEC = importlib.util.spec_from_file_location("device_benchmark_report", MODULE)
assert SPEC and SPEC.loader
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def sample(seconds: float, battery: float = .8, state: str = "unplugged", thermal: str = "nominal") -> dict[str, str]:
    """One serialized sensor sample with independent known expectations."""
    return {"elapsed_seconds": str(seconds), "step": str(int(seconds)),
            "battery_fraction": str(battery), "battery_state": state,
            "thermal_state": thermal, "footprint_bytes": "104857600",
            "gpu_bytes": "10485760", "available_memory_bytes": "1048576000", "step_seconds": ".02"}


def test_discharge_and_sampled_thermal_dwell() -> None:
    rows = REPORT.validate_samples([sample(0), sample(300, .79, thermal="fair"), sample(600, .77)])
    result = REPORT.summarize_samples(rows, {"battery_measurement_valid": True})
    assert result["battery_measurement_valid"]
    assert result["battery_drop_percentage_points"] == pytest.approx(3)
    assert result["estimated_percentage_points_per_hour"] == pytest.approx(18)
    assert result["sampled_thermal_seconds"] == {"nominal": 300, "fair": 300, "serious": 0, "critical": 0}
    assert result["peak_sampled_process_footprint_bytes"] == 104857600


@pytest.mark.parametrize("state", ["charging", "full", "unknown"])
def test_charging_and_unknown_state_invalidate_battery(state: str) -> None:
    rows = REPORT.validate_samples([sample(0), sample(600, .77, state=state)])
    result = REPORT.summarize_samples(rows, {"battery_measurement_valid": True})
    assert not result["battery_measurement_valid"]
    assert result["estimated_percentage_points_per_hour"] is None


def test_app_invalid_interval_and_zero_duration_are_not_drain_estimates() -> None:
    rows = REPORT.validate_samples([sample(0), sample(0)])
    result = REPORT.summarize_samples(rows, {"battery_measurement_valid": False})
    assert not result["battery_measurement_valid"]
    assert result["battery_drop_percentage_points"] is None


def test_interior_missing_or_increasing_battery_invalidates_interval() -> None:
    for middle in ({**sample(300), "battery_fraction": ""}, sample(300, .81)):
        rows = REPORT.validate_samples([sample(0), middle, sample(600, .78)])
        result = REPORT.summarize_samples(rows, {"battery_measurement_valid": True})
        assert not result["battery_measurement_valid"]
        assert result["estimated_percentage_points_per_hour"] is None


def test_zero_observed_drain_and_short_runs_do_not_claim_hourly_energy() -> None:
    for end in (sample(1200), sample(20, .78)):
        rows = REPORT.validate_samples([sample(0), end])
        result = REPORT.summarize_samples(rows, {"battery_measurement_valid": True})
        assert result["battery_measurement_valid"]
        assert result["estimated_percentage_points_per_hour"] is None


def test_requested_budget_is_distinct_from_longer_stress_test() -> None:
    rows = REPORT.validate_samples([sample(0), sample(300, .79), sample(600, .77)])
    result = REPORT.summarize_samples(rows, {"battery_measurement_valid": True, "target_step": 400})
    budget = result["requested_budget_observation"]
    assert budget["elapsed_seconds_range"] == [300, 600]
    assert budget["observed_step"] == 600
    assert budget["battery_drop_percentage_points_at_observation"] == pytest.approx(3)
    unfinished = REPORT.summarize_samples(rows, {"battery_measurement_valid": True, "target_step": 1000})
    assert unfinished["requested_budget_observation"] is None


def test_invalid_sensor_order_and_values_are_rejected() -> None:
    invalid_time = {**sample(0), "elapsed_seconds": "nan"}
    fractional_step = {**sample(0), "step": "1.5"}
    for rows in ([sample(2), sample(1)], [sample(0, 1.2)], [invalid_time], [fractional_step]):
        with pytest.raises(ValueError):
            REPORT.validate_samples(rows)


def test_ssim_identical_and_known_black_white_error() -> None:
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    identical = REPORT.display_image_metrics(black, black)
    assert identical["display_ssim"] == pytest.approx(1)
    assert identical["display_rgb_mse"] == 0
    assert identical["display_psnr_db"] is None
    opposite = REPORT.display_image_metrics(black, np.full_like(black, 255))
    assert opposite["display_rgb_mse"] == 1
    assert opposite["display_psnr_db"] == 0
    assert opposite["display_ssim"] == pytest.approx(.0001/1.0001)
