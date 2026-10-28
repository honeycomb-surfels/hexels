"""Check identical image/pose inputs with a bounded float32 roundoff allowance."""

from __future__ import annotations

import hashlib
import json

import numpy as np

# Camera matrices are float32 results of GPU/LAPACK arithmetic, so other hardware can round them differently.
FLOAT_MATRICES = ("world_view_transform", "full_proj_transform", "camera_center")


def geometry_digest(records: list[dict]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def validate(canonical: dict, actual: dict, geometry: dict | None = None) -> dict:
    """Names, image sizes, fields of view and target tensors are exact; float32 camera matrices allow roundoff."""
    maximum = 0.0
    for split in ("training", "test"):
        expected = canonical[split]
        observed = actual[split]
        if expected == observed:
            continue
        field = "ordered_geometry_sha256"
        if {key: value for key, value in expected.items() if key != field} != {
                key: value for key, value in observed.items() if key != field}:
            raise ValueError(f"Native {split} image names, resolution, or target tensors differ")
        if geometry is None or "geometry" not in canonical:
            raise ValueError(
                f"Native {split} camera geometry differs from the contract hash, and the reference records "
                "needed for a roundoff-tolerant comparison are missing; use a contract that includes "
                "'geometry' (re-download cameras/nvs/<dataset>/<scene>/contract.json from the model repository)")
        reference, native = canonical["geometry"][split], geometry[split]
        if geometry_digest(reference) != expected[field] or geometry_digest(native) != observed[field]:
            raise ValueError("Camera geometry records do not match their hashes")
        if len(reference) != len(native):
            raise ValueError("Camera geometry counts differ")
        for first, second in zip(reference, native):
            if {key: value for key, value in first.items() if key not in FLOAT_MATRICES} != {
                    key: value for key, value in second.items() if key not in FLOAT_MATRICES}:
                raise ValueError(f"Native {split} names, image sizes or fields of view differ")
            for key in FLOAT_MATRICES:
                expected_value = np.asarray(first[key])
                native_value = np.asarray(second[key])
                if expected_value.shape != native_value.shape or not np.allclose(
                        expected_value, native_value, atol=1e-6, rtol=1e-6):
                    raise ValueError(f"Native {split} {key} exceeds float32 roundoff tolerance")
                maximum = max(maximum, float(np.max(np.abs(expected_value - native_value))))
    return {"valid": True, "maximum_matrix_difference": maximum,
            "policy": "exact names, sizes, fields of view and target tensors; camera matrices atol=rtol=1e-6"}
