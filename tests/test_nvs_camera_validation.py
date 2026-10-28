"""Camera contract checks that tolerate float32 roundoff from other hardware."""
from __future__ import annotations

import copy

import pytest

from scripts.nvs_camera_validation import geometry_digest, validate


def _record(name: str, offset: float = 0.0) -> dict:
    return {
        "name": name,
        "width": 4,
        "height": 3,
        "fov_x": 0.5,
        "fov_y": 0.4,
        "world_view_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0], [0.1, 0.2, 2.0 + offset, 1.0]],
        "full_proj_transform": [[2.0, 0.0, 0.0, 0.0], [0.0, 2.5, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 1.0], [0.2, 0.5, 1.99 + offset, 2.0]],
        "camera_center": [-0.1, -0.2, -2.0 - offset],
    }


def _contract(records: dict[str, list[dict]]) -> dict:
    return {split: {"count": len(rows), "ordered_names_sha256": "names",
                    "ordered_target_tensors_sha256": "targets",
                    "ordered_geometry_sha256": geometry_digest(rows)}
            for split, rows in records.items()}


def _reference() -> dict[str, list[dict]]:
    return {"training": [_record("a")], "test": [_record("b")]}


def test_identical_contracts_need_no_geometry_records() -> None:
    assert validate(_contract(_reference()), _contract(_reference()))["valid"]


def test_float32_roundoff_in_camera_matrices_is_accepted() -> None:
    native = {"training": [_record("a", 2e-7)], "test": [_record("b", -2e-7)]}
    canonical = {**_contract(_reference()), "geometry": _reference()}
    result = validate(canonical, _contract(native), native)
    assert result["valid"]
    assert 0.0 < result["maximum_matrix_difference"] < 1e-6


def test_pose_changes_beyond_roundoff_are_rejected() -> None:
    native = {"training": [_record("a")], "test": [_record("b", 1e-3)]}
    canonical = {**_contract(_reference()), "geometry": _reference()}
    with pytest.raises(ValueError, match="test world_view_transform exceeds"):
        validate(canonical, _contract(native), native)


def test_image_sizes_stay_exact() -> None:
    native = copy.deepcopy(_reference())
    native["test"][0]["width"] = 5
    canonical = {**_contract(_reference()), "geometry": _reference()}
    with pytest.raises(ValueError, match="names, image sizes or fields of view"):
        validate(canonical, _contract(native), native)


def test_missing_reference_records_explain_the_fix() -> None:
    native = {"training": [_record("a", 2e-7)], "test": [_record("b")]}
    with pytest.raises(ValueError, match="re-download cameras/nvs"):
        validate(_contract(_reference()), _contract(native), native)
