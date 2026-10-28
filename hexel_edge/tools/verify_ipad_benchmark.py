"""Independently check a retrieved device benchmark without publishing its IDs.

Optional CLI-compatible checkpoint copies change only the context header after
the original edge2 context is validated; tensor payloads remain byte-identical.
Original device checkpoint bytes are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


STANDARD_RATES = [.0001]*3 + [.001]*2 + [.0005]*4 + [.0025]*3 + [.001]*6 + [.01, .001]
DETAIL_RATES = [.000025]*3 + [.001]*2 + [.0001]*4 + [.01]*3 + [.002]*6 + [.02, .0005]


def digest_file(path: Path) -> str:
    """Hash an existing local evidence file without retaining its bytes."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def bundle_fingerprint(root: Path, manifest: dict) -> str:
    """Reproduce SceneBundle's ordered metadata/asset SHA256 exactly."""
    digest = hashlib.sha256((root / "manifest.json").read_bytes())
    files = [manifest["parameters_file"], manifest["view_order_file"]] + [frame["image"] for frame in manifest["frames"]]
    for name in files:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Bundle asset escapes its root")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
    return digest.hexdigest()


def verify(bundle: Path, checkpoint: Path, report_path: Path, telemetry: Path | None,
           *, checkpoint_kind: str = "final", normalized_copy: Path | None = None) -> dict:
    """Validate experiment context, arrays, steps, recipe, and telemetry scope."""
    manifest = json.loads((bundle / "manifest.json").read_text())
    report = json.loads(report_path.read_text())
    names = {frame["name"]: i for i, frame in enumerate(manifest["frames"])}
    selected = [names[name] for name in report["selected_training_frames"]]
    if not selected or len(set(selected)) != len(selected) or any(manifest["frames"][i]["split"] != "train" for i in selected):
        raise ValueError("Selected observations include absent/duplicate/held-out frames")
    pool = [i for i, frame in enumerate(manifest["frames"]) if frame["split"] == "train"]
    if len(selected) == len(pool):
        expected_selection = pool
    elif len(selected) == 1:
        expected_selection = [pool[len(pool)//2]]
    else:
        expected_selection = [pool[math.floor(i*(len(pool)-1)/(len(selected)-1)+.5)] for i in range(len(selected))]
    if selected != expected_selection:
        raise ValueError("Frame selection does not match the deterministic device policy")
    expected_test = [frame["name"] for frame in manifest["frames"] if frame["split"] == "test"]
    if report["held_out_frames"] != expected_test:
        raise ValueError("Reported held-out cameras differ from the input bundle")
    edge = int(report["image_long_edge"])
    if edge < 1:
        raise ValueError("Invalid effective image resolution")
    dimensions = []
    for frame in manifest["frames"]:
        scale = min(1, edge / max(frame["width"], frame["height"]))
        dimensions.append([max(1, math.floor(frame["width"]*scale+.5)), max(1, math.floor(frame["height"]*scale+.5))])
    selection = json.dumps({"training": selected, "dimensions": dimensions}, sort_keys=True, separators=(",", ":")).encode()
    raw_context = bundle_fingerprint(bundle, manifest)
    context = "edge2:" + raw_context + ":" + hashlib.sha256(selection).hexdigest()
    if report["bundle_fingerprint"] != raw_context or report["checkpoint_context"] != context:
        raise ValueError("Report experiment context does not match independently derived input selection")
    if checkpoint.stat().st_size > 512*1024*1024:
        raise ValueError("Checkpoint exceeds verification admission limit")
    original = checkpoint.read_bytes()
    if len(original) < 4:
        raise ValueError("Truncated checkpoint")
    length = struct.unpack_from("<I", original)[0]
    if not 0 < length <= 65536 or len(original) < 4+length:
        raise ValueError("Invalid checkpoint header size")
    header = json.loads(original[4:4+length])
    count = int(header["parameterCount"])
    if header["schemaVersion"] != 1 or count != manifest["primitive_count"]*20 or len(original) != 4+length+count*12:
        raise ValueError("Checkpoint schema/population/binary length differs")
    expected_step = report["best_step"] if checkpoint_kind == "best" else report["last_step"]
    if header["step"] != expected_step or header["context"] != context:
        raise ValueError("Checkpoint step/context differs from the selected report state")
    recipe = manifest.get("recipe_hint", "standard")
    if recipe not in {"standard", "detail"}:
        raise ValueError("Unknown input recipe")
    rates = DETAIL_RATES if recipe == "detail" else STANDARD_RATES
    if not np.array_equal(np.asarray(header["learningRates"], dtype=np.float32), np.asarray(rates, dtype=np.float32)):
        raise ValueError("Checkpoint does not contain the declared recipe rates")
    payload = original[4+length:]
    arrays = np.frombuffer(payload, dtype="<f4").reshape(3, count)
    rows = arrays[0].reshape(-1, 20)
    if not np.isfinite(arrays).all() or np.any(arrays[2] < 0):
        raise ValueError("Nonfinite state or negative Adam second moments")
    norms = np.linalg.norm(rows[:, 5:9], axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-8) or np.any(abs(rows[:, 3:5]) >= 40) or np.any(abs(rows[:, 19]) >= 40):
        raise ValueError("Invalid quaternion/log-parameter domain")
    result = {
        "schema_version": 1, "verification_status": "passed",
        "checkpoint_kind": checkpoint_kind, "checkpoint_step": header["step"],
        "primitive_count": len(rows), "recipe": recipe,
        "raw_bundle_fingerprint": raw_context, "verified_experiment_context": context,
        "checkpoint_sha256": hashlib.sha256(original).hexdigest(),
        "tensor_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "report_sha256": digest_file(report_path),
        "selected_training_frames": report["selected_training_frames"],
        "held_out_frames": expected_test, "effective_dimensions": dimensions,
        "all_parameters_and_moments_finite": True, "second_moments_nonnegative": True,
        "measured_initial_step": report["initial_step"], "measured_last_step": report["last_step"],
        "status": report["status"], "quality_evaluations": report.get("evaluations", []),
        "battery_reported_valid": report["battery_measurement_valid"],
        "limits": ["Battery readings are coarse OS telemetry, not energy/power; thermal state is categorical, not temperature.",
                   "Input calibration and seeds were prepared separately; this does not validate image-only on-device reconstruction."],
    }
    initial = np.fromfile(bundle/manifest["parameters_file"], dtype="<f4").reshape(-1, 20)
    changes = rows.astype(np.float64) - initial
    groups = (("center", 0, 3), ("log_scale", 3, 5), ("quaternion_wxyz", 5, 9), ("rgb", 9, 12),
              ("gradient_u", 12, 15), ("gradient_v", 15, 18), ("opacity_logit", 18, 19), ("log_softness", 19, 20))
    result["updates_from_prepared_initial_parameters"] = {
        name: {"changed_scalars": int(np.count_nonzero(changes[:, start:end])),
               "max_absolute_change": float(np.max(abs(changes[:, start:end])))}
        for name, start, end in groups}
    if telemetry is not None:
        with telemetry.open(newline="") as handle:
            samples = list(csv.DictReader(handle))
        states = sorted({sample["battery_state"] for sample in samples})
        complete = bool(samples) and all(sample["battery_fraction"] and 0 <= float(sample["battery_fraction"]) <= 1 for sample in samples)
        result["telemetry"] = {
            "sha256": digest_file(telemetry), "sample_count": len(samples),
            "battery_states_observed": states,
            "thermal_states_observed": sorted({sample["thermal_state"] for sample in samples}),
            "peak_sampled_footprint_bytes": max((int(sample["footprint_bytes"]) for sample in samples), default=0),
            "peak_sampled_gpu_bytes": max((int(sample["gpu_bytes"]) for sample in samples), default=0),
            "battery_comparison_valid": bool(report["battery_measurement_valid"] and complete and states == ["unplugged"]),
        }
    if normalized_copy is not None:
        if normalized_copy.exists() or normalized_copy.resolve() == checkpoint.resolve():
            raise FileExistsError("Use a new path for the explicitly normalized comparison copy")
        normalized_header = dict(header, context=raw_context)
        encoded = json.dumps(normalized_header, separators=(",", ":"), sort_keys=True).encode()
        normalized_copy.parent.mkdir(parents=True, exist_ok=True)
        normalized_copy.write_bytes(struct.pack("<I", len(encoded)) + encoded + payload)
        restored = normalized_copy.read_bytes()
        size = struct.unpack_from("<I", restored)[0]
        if restored[4+size:] != payload:
            raise ValueError("Normalized comparison copy changed tensor payload")
        result["cli_comparison_copy"] = {"header_context_normalized_only": True,
            "original_device_checkpoint_unmodified": digest_file(checkpoint) == result["checkpoint_sha256"],
            "tensor_payload_byte_identical": True, "sha256": hashlib.sha256(restored).hexdigest()}
    return result


def main() -> None:
    """Write a sanitized independent verification record for a retrieved run."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "checkpoint", "report", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--checkpoint-kind", choices=("final", "best"), default="final")
    parser.add_argument("--normalized-checkpoint", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle, args.checkpoint, args.report, args.telemetry,
                    checkpoint_kind=args.checkpoint_kind, normalized_copy=args.normalized_checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
