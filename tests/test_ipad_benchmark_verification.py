"""Verify experiment binding and preserve original device evidence bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "hexel_edge/tools"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
import edge_bundle as edge
from verify_ipad_benchmark import bundle_fingerprint, verify, DETAIL_RATES


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = tmp_path/"bundle"
    manifest = edge.synthetic(bundle, long_edge=16, steps=100, recipe_hint="detail")
    selected = [i for i, frame in enumerate(manifest["frames"]) if frame["split"] == "train"]
    dimensions = [[frame["width"], frame["height"]] for frame in manifest["frames"]]
    selection = json.dumps({"training": selected, "dimensions": dimensions}, sort_keys=True, separators=(",", ":")).encode()
    raw = bundle_fingerprint(bundle, manifest)
    context = "edge2:"+raw+":"+hashlib.sha256(selection).hexdigest()
    parameters = np.fromfile(bundle/"parameters.f32", dtype="<f4")
    parameters.reshape(-1,20)[:,9:12] += .001
    header = {"schemaVersion": 1, "step": 100, "parameterCount": len(parameters),
              "learningRates": DETAIL_RATES, "context": context}
    encoded = json.dumps(header).encode()
    checkpoint = tmp_path/"original.hexstate"
    checkpoint.write_bytes(struct.pack("<I", len(encoded))+encoded
        + parameters.tobytes()+np.full_like(parameters,1e-4).tobytes()+np.full_like(parameters,1e-6).tobytes())
    report = {"run_id": "private-run-identifier", "selected_training_frames": [manifest["frames"][i]["name"] for i in selected],
              "held_out_frames": [f["name"] for f in manifest["frames"] if f["split"] == "test"],
              "image_long_edge": 16, "bundle_fingerprint": raw, "checkpoint_context": context,
              "last_step": 100, "initial_step": 0, "status": "completed", "battery_measurement_valid": True}
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report))
    return bundle, checkpoint, path


def test_context_normalization_preserves_payload_and_original(tmp_path: Path) -> None:
    bundle, checkpoint, report = fixture(tmp_path)
    original = checkpoint.read_bytes()
    copy = tmp_path/"explicit_comparison_copy.hextrain"
    result = verify(bundle, checkpoint, report, None, normalized_copy=copy)
    assert result["cli_comparison_copy"]["tensor_payload_byte_identical"]
    assert checkpoint.read_bytes() == original
    assert "private-run-identifier" not in json.dumps(result)
    old_size = struct.unpack_from("<I",original)[0]
    rewritten = copy.read_bytes()
    new_size = struct.unpack_from("<I",rewritten)[0]
    assert original[4+old_size:] == rewritten[4+new_size:]
    assert json.loads(rewritten[4:4+new_size])["context"] == result["raw_bundle_fingerprint"]


def test_heldout_selection_and_forged_context_are_rejected(tmp_path: Path) -> None:
    bundle, checkpoint, path = fixture(tmp_path)
    report = json.loads(path.read_text())
    report["selected_training_frames"][0] = report["held_out_frames"][0]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="held-out"):
        verify(bundle,checkpoint,path,None)


def test_charging_interval_invalidates_battery_comparison(tmp_path: Path) -> None:
    bundle, checkpoint, report = fixture(tmp_path)
    telemetry = tmp_path/"telemetry.csv"
    telemetry.write_text("battery_state,battery_fraction,thermal_state,footprint_bytes,gpu_bytes\n"
                         "unplugged,0.8,nominal,1000,100\ncharging,0.8,fair,1100,120\n")
    result = verify(bundle,checkpoint,report,telemetry)
    assert result["battery_reported_valid"]
    assert not result["telemetry"]["battery_comparison_valid"]
