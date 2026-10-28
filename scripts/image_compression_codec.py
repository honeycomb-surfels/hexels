"""Optimizer-free 2D image representations with explicit, measured storage."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"HXIMG001"
HEADER_BYTES = 1024
FIELDS = {
    "imagegs": {"positions": 2, "scales": 2, "rotations": 1, "features": 3},
    "hexel": {
        "positions": 2, "scales": 2, "rotations": 1, "features": 3,
        "color_gradients": 6, "softness": 1,
    },
}


def stored_bytes(method: str, count: int, bits: int = 16) -> int:
    """Include the complete fixed header and every learned render parameter."""
    if bits not in (16, 32) or count < 1:
        raise ValueError("Expected positive count and 16 or 32 bits")
    return HEADER_BYTES + count * sum(FIELDS[method].values()) * (bits // 8)


def equal_storage_count(reference_count: int) -> int:
    """Largest Hexel population fitting the 16-bit Image-GS file budget."""
    return reference_count * sum(FIELDS["imagegs"].values()) // sum(FIELDS["hexel"].values())


def encode(
    path: Path, parameters: dict[str, np.ndarray], *, method: str,
    width: int, height: int, bits: int,
    quantized: dict[str, tuple[np.ndarray, float, float]] | None = None,
) -> dict:
    """Store render-space parameters as float32 or uniform unsigned 16-bit codes.

    Uniform quantization matches Image-GS's per-tensor min/step rule. The
    min/step values and renderer configuration are included in the header.
    """
    if set(parameters) != set(FIELDS[method]):
        raise ValueError("Unexpected parameter fields")
    count = len(parameters["positions"])
    size = stored_bytes(method, count, bits)
    fields = []
    payload = []
    for name, channels in FIELDS[method].items():
        value = np.asarray(parameters[name], dtype=np.float32)
        if value.size != count * channels or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} shape or values")
        field = {"name": name, "shape": list(value.shape)}
        if bits == 16:
            if quantized is None:
                minimum, maximum = float(value.min()), float(value.max())
                scale = max((maximum - minimum) / 65535, 1e-8)
                codes = np.rint((value - np.float32(minimum)) / np.float32(scale))
            else:
                codes, minimum, scale = quantized[name]
                if codes.shape != value.shape or codes.dtype != np.uint16:
                    raise ValueError(f"Invalid quantized codes for {name}")
            payload.append(np.clip(codes, 0, 65535).astype("<u2").tobytes())
            field.update(minimum=minimum, scale=scale)
        else:
            payload.append(value.astype("<f4").tobytes())
        fields.append(field)
    header = {
        "method": method, "width": width, "height": height, "count": count,
        "bits": bits, "topk": 10, "color_grad_scale": 0.1,
        "aa_kernel_size": 0.0, "fields": fields,
    }
    data = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    prefix = MAGIC + struct.pack("<I", len(data)) + data
    if len(prefix) > HEADER_BYTES:
        raise ValueError("Codec header exceeds reserved space")
    with path.open("xb") as stream:
        stream.write(prefix.ljust(HEADER_BYTES, b"\0"))
        for block in payload:
            stream.write(block)
    assert path.stat().st_size == size
    return header


def decode(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read a complete standalone representation without a training checkpoint."""
    data = path.read_bytes()
    if data[:8] != MAGIC or len(data) < HEADER_BYTES:
        raise ValueError("Invalid image representation header")
    length = struct.unpack("<I", data[8:12])[0]
    if length > HEADER_BYTES - 12:
        raise ValueError("Invalid header length")
    header = json.loads(data[12:12 + length])
    method, count, bits = header["method"], header["count"], header["bits"]
    if len(data) != stored_bytes(method, count, bits):
        raise ValueError("Representation is truncated or has trailing data")
    if [field["name"] for field in header["fields"]] != list(FIELDS[method]):
        raise ValueError("Unexpected codec fields")
    offset = HEADER_BYTES
    parameters = {}
    for field in header["fields"]:
        name, shape = field["name"], field["shape"]
        elements = count * FIELDS[method][name]
        if shape[0] != count or int(np.prod(shape)) != elements:
            raise ValueError(f"Invalid shape for {name}")
        values = np.frombuffer(data, dtype="<u2" if bits == 16 else "<f4",
                               count=elements, offset=offset).astype(np.float32)
        if bits == 16:
            values = values * np.float32(field["scale"]) + np.float32(field["minimum"])
        parameters[name] = values.reshape(shape)
        offset += elements * (bits // 8)
    return header, parameters
