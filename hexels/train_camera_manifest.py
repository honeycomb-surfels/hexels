from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_train_cameras(
    cameras: list[object],
    manifest_path: str | Path,
) -> tuple[list[object], dict[str, object]]:
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise ValueError(f"Training camera manifest does not exist: {path}")
    manifest_bytes = path.read_bytes()
    payload = json.loads(manifest_bytes)
    registered = isinstance(payload, Mapping)
    if registered:
        expected_keys = {"schema_version", "camera_names", "camera_names_sha256"}
        if set(payload) != expected_keys:
            raise ValueError(
                "Registered training camera manifest must contain exactly: "
                + ", ".join(sorted(expected_keys))
            )
        if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
            raise ValueError("Training camera manifest schema_version must be 1")
        names = payload["camera_names"]
    else:
        names = payload
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError("Training camera manifest must contain a non-empty list of camera names")
    if len(set(names)) != len(names):
        raise ValueError("Training camera manifest camera names must be unique")

    names_sha256 = _json_sha256(names)
    if registered and payload["camera_names_sha256"] != names_sha256:
        raise ValueError("Training camera manifest camera_names_sha256 mismatch")

    named_cameras: dict[str, object] = {}
    for camera in cameras:
        name = getattr(camera, "image_name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Every training camera must have a non-empty image_name")
        if name in named_cameras:
            raise ValueError("Training camera image_name values must be unique")
        named_cameras[name] = camera
    unknown = [name for name in names if name not in named_cameras]
    if unknown:
        raise ValueError(
            "Training camera manifest contains unknown camera names: " + ", ".join(unknown)
        )

    metadata = {
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "camera_count": len(names),
        "camera_names": list(names),
        "camera_names_sha256": names_sha256,
    }
    return [named_cameras[name] for name in names], metadata
