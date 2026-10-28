"""Standalone 2D image files with log scales and optional shared appearance."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"HXIMG002"
HEADER_BYTES = 1024
BASE_FIELDS = {"positions": 2, "log_scales": 2, "rotations": 1, "features": 3}
LAYOUTS = {"flat8", "linear14", "dense15", "sparse", "codebook"}


def encode(
    path: Path, parameters: dict[str, np.ndarray], *, method: str = "hexel",
    width: int, height: int, layout: str = "dense15", default_softness: float = 10.0,
    gradient_mask: np.ndarray | None = None, softness_mask: np.ndarray | None = None,
    codebook: np.ndarray | None = None, indices: np.ndarray | None = None,
    quantized: dict[str, tuple[np.ndarray, float, float]] | None = None,
) -> dict:
    """Write all decoding data, including masks, indices and per-image tables.

    Codebook rows contain RGB followed by the six flattened color gradients.
    Supplied quantized codes preserve CUDA rounding; sparse codes contain only
    retained rows. Log scales remain in log coordinates after decoding.
    """
    if layout not in LAYOUTS or method not in {"imagegs", "hexel"}:
        raise ValueError("Unknown method or compact layout")
    if method == "imagegs" and layout not in {"flat8", "linear14"}:
        raise ValueError("Gaussian images require flat8 or linear14")
    count = len(parameters["positions"])
    if count < 1 or width < 1 or height < 1:
        raise ValueError("Expected positive population and image dimensions")
    if not np.isfinite(default_softness) or default_softness <= 0:
        raise ValueError("Expected positive finite default softness")
    fields, payload = [], []

    def add_quantized(name: str, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name}")
        if quantized is not None and name in quantized:
            codes, minimum, step = quantized[name]
            if codes.shape != value.shape or codes.dtype != np.uint16:
                raise ValueError(f"Invalid supplied codes for {name}")
        else:
            minimum = float(value.min()) if value.size else 0.0
            step = max((float(value.max()) - minimum) / 65535, 1e-8) if value.size else 1.0
            codes = np.clip(np.rint((value - np.float32(minimum)) / np.float32(step)),
                            0, 65535).astype(np.uint16)
        if not np.isfinite(minimum) or not np.isfinite(step) or step <= 0:
            raise ValueError(f"Invalid quantization range for {name}")
        fields.append({"name": name, "shape": list(value.shape), "kind": "q16",
                       "min": float(minimum), "step": float(step)})
        payload.append(codes.astype("<u2").tobytes())

    for name, channels in BASE_FIELDS.items():
        if layout == "codebook" and name == "features":
            continue
        value = np.asarray(parameters[name], dtype=np.float32)
        if value.shape != (count, channels):
            raise ValueError(f"Invalid {name} shape")
        add_quantized(name, value)
    if layout in {"linear14", "dense15"}:
        gradients = np.asarray(parameters["color_gradients"], dtype=np.float32)
        if gradients.shape != (count, 2, 3):
            raise ValueError("Invalid color gradient shape")
        add_quantized("color_gradients", gradients)
        if layout == "dense15":
            softness = np.asarray(parameters["softness"], dtype=np.float32)
            if softness.shape != (count,):
                raise ValueError("Invalid softness shape")
            add_quantized("softness", softness)
    elif layout == "sparse":
        for name, mask in (("color_gradients", gradient_mask), ("softness", softness_mask)):
            if mask is None:
                mask = np.zeros(count, dtype=bool)
            mask = np.asarray(mask)
            if mask.shape != (count,) or mask.dtype != np.bool_:
                raise ValueError(f"Invalid {name} mask")
            value = np.asarray(parameters[name], dtype=np.float32)
            shape = (count, 2, 3) if name == "color_gradients" else (count,)
            if value.shape != shape:
                raise ValueError(f"Invalid {name} shape")
            fields.append({"name": f"{name}_mask", "shape": [count], "kind": "mask"})
            payload.append(np.packbits(mask, bitorder="little").tobytes())
            add_quantized(name, value[mask])
    elif layout == "codebook":
        table = np.asarray(codebook, dtype=np.float32)
        assignments = np.asarray(indices)
        if table.ndim != 2 or table.shape[1] != 9 or not 1 <= len(table) <= 65536:
            raise ValueError("Expected a codebook with shape (K,9), 1 <= K <= 65536")
        if (assignments.shape != (count,) or assignments.dtype.kind not in "ui"
                or assignments.min() < 0 or assignments.max() >= len(table)):
            raise ValueError("Invalid codebook indices")
        softness = np.asarray(parameters["softness"], dtype=np.float32)
        if softness.shape != (count,):
            raise ValueError("Invalid softness shape")
        add_quantized("softness", softness)
        add_quantized("codebook", table)
        index_kind = "u8" if len(table) <= 256 else "u16"
        fields.append({"name": "indices", "shape": [count], "kind": index_kind})
        payload.append(assignments.astype("u1" if index_kind == "u8" else "<u2").tobytes())
    header = {"method": method, "layout": layout, "width": width, "height": height,
              "count": count, "default_softness": float(default_softness),
              "topk": 10, "color_grad_scale": 0.1, "aa_kernel_size": 0.0, "fields": fields}
    metadata = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    prefix = MAGIC + struct.pack("<I", len(metadata)) + metadata
    if len(prefix) > HEADER_BYTES:
        raise ValueError("Compact codec header exceeds reserved space")
    with path.open("xb") as stream:
        stream.write(prefix.ljust(HEADER_BYTES, b"\0"))
        for block in payload:
            stream.write(block)
    return header


def decode(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Expand compact fields to the existing dense renderer's parameter arrays."""
    data = path.read_bytes()
    if data[:8] != MAGIC or len(data) < HEADER_BYTES:
        raise ValueError("Invalid compact image header")
    length = struct.unpack("<I", data[8:12])[0]
    if length > HEADER_BYTES - 12:
        raise ValueError("Invalid compact header length")
    header = json.loads(data[12:12 + length])
    count, layout = header["count"], header["layout"]
    if layout not in LAYOUTS or count < 1:
        raise ValueError("Invalid compact layout or population")
    values, offset = {}, HEADER_BYTES
    for field in header["fields"]:
        name, shape, kind = field["name"], tuple(field["shape"]), field["kind"]
        elements = int(np.prod(shape))
        if any(dimension < 0 for dimension in shape) or name in values:
            raise ValueError("Invalid compact field shape or duplicate name")
        size = (elements + 7) // 8 if kind == "mask" else elements * (1 if kind == "u8" else 2)
        if offset + size > len(data):
            raise ValueError("Compact representation is truncated")
        if kind == "mask":
            packed = np.frombuffer(data, dtype="u1", count=size, offset=offset)
            value = np.unpackbits(packed, bitorder="little", count=elements).astype(bool)
        elif kind in {"q16", "u8", "u16"}:
            value = np.frombuffer(data, dtype="u1" if kind == "u8" else "<u2",
                                  count=elements, offset=offset)
            if kind == "q16":
                value = value.astype(np.float32) * np.float32(field["step"]) + np.float32(field["min"])
        else:
            raise ValueError(f"Unknown compact field encoding: {kind}")
        values[name] = value.reshape(shape)
        offset += size
    if offset != len(data):
        raise ValueError("Compact representation has trailing data")
    parameters = {name: values[name] for name in BASE_FIELDS if name != "features" or layout != "codebook"}
    gradients = np.zeros((count, 2, 3), dtype=np.float32)
    softness = np.full(count, header["default_softness"], dtype=np.float32)
    if layout in {"linear14", "dense15"}:
        gradients = values["color_gradients"]
        if layout == "dense15":
            softness = values["softness"]
    elif layout == "sparse":
        gradients[values["color_gradients_mask"]] = values["color_gradients"]
        softness[values["softness_mask"]] = values["softness"]
    elif layout == "codebook":
        appearance = values["codebook"][values["indices"]]
        parameters["features"] = appearance[:, :3].copy()
        gradients = appearance[:, 3:].reshape(count, 2, 3).copy()
        softness = values["softness"]
    parameters.update(color_gradients=gradients, softness=softness)
    return header, parameters
