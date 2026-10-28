"""Optional per-field bit packing for log-scale 2D Hexel files."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"HXIMG004"
HEADER_BYTES = 1024
FIELD_SHAPES = {"positions": (2,), "log_scales": (2,), "rotations": (1,),
                "features": (3,), "color_gradients": (2, 3), "softness": ()}
PROFILES = {
    "q16": dict.fromkeys(FIELD_SHAPES, 16),
    "appearance14": {"positions": 16, "log_scales": 16, "rotations": 16,
                     "features": 14, "color_gradients": 14, "softness": 14},
    "mixed14": {"positions": 16, "log_scales": 14, "rotations": 12,
                "features": 12, "color_gradients": 14, "softness": 10},
    "appearance12": {"positions": 16, "log_scales": 16, "rotations": 16,
                     "features": 12, "color_gradients": 12, "softness": 12},
    "balanced": {"positions": 16, "log_scales": 14, "rotations": 12,
                 "features": 10, "color_gradients": 8, "softness": 10},
    "rich128": {"positions": 14, "log_scales": 12, "rotations": 10,
                "features": 8, "color_gradients": 6, "softness": 6},
}
# Paper profiles: balanced for flat RGB (flat8), mixed14 for spatial RGB (dense15).
PAPER_PROFILES = {"flat8": "balanced", "dense15": "mixed14"}


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    """Pack each unsigned scalar least-significant bit first, including subbytes."""
    if (not 1 <= bits <= 16 or codes.dtype.kind not in "ui"
            or np.any(codes < 0) or np.any(codes >= 2**bits)):
        raise ValueError("Invalid precision or unsigned quantization code")
    bit_values = (codes.astype(np.uint32).reshape(-1, 1) >> np.arange(bits, dtype=np.uint32)) & 1
    return np.packbits(bit_values.reshape(-1), bitorder="little").tobytes()


def unpack_codes(data: bytes, elements: int, bits: int) -> np.ndarray:
    """Recover unsigned scalar codes without interpreting padding as parameters."""
    if not 1 <= bits <= 16 or elements < 0 or len(data) != (elements * bits + 7) // 8:
        raise ValueError("Invalid packed field length or precision")
    bit_values = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little",
                              count=elements * bits).reshape(elements, bits)
    return (bit_values.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(1).astype(np.uint16)


def encode(path: Path, parameters: dict[str, np.ndarray], *, width: int, height: int,
           rich: bool, field_bits: dict[str, int], quantized: dict[str, tuple] | None = None) -> dict:
    """Write the per-image ranges and bit allocation together with the packed codes."""
    count = len(parameters["positions"])
    fields, payload = [], []
    names = list(FIELD_SHAPES) if rich else list(FIELD_SHAPES)[:4]
    if count < 1 or width < 1 or height < 1:
        raise ValueError("Expected positive population and image dimensions")
    for name in names:
        value = np.asarray(parameters[name], dtype=np.float32)
        bits = field_bits[name]
        if value.shape != (count, *FIELD_SHAPES[name]) or not np.isfinite(value).all():
            raise ValueError(f"Invalid stored field: {name}")
        if not 1 <= bits <= 16:
            raise ValueError(f"Invalid bit allocation: {name}")
        if quantized is None:
            minimum = float(value.min())
            step = max((float(value.max()) - minimum) / (2**bits - 1), 1e-8)
            codes = np.clip(np.rint((value - np.float32(minimum)) / np.float32(step)),
                            0, 2**bits - 1).astype(np.uint16)
        else:
            codes, minimum, step = quantized[name]
            if codes.shape != value.shape or not np.isfinite([minimum, step]).all() or step <= 0:
                raise ValueError(f"Invalid supplied quantization: {name}")
        fields.append({"name": name, "bits": bits, "min": minimum, "step": step})
        payload.append(pack_codes(codes, bits))
    header = {"width": width, "height": height, "count": count, "rich": rich,
              "topk": 10, "color_grad_scale": 0.1, "default_softness": 10.0, "fields": fields}
    metadata = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    prefix = MAGIC + struct.pack("<I", len(metadata)) + metadata
    if len(prefix) > HEADER_BYTES:
        raise ValueError("Mixed-precision header exceeds reserved space")
    with path.open("xb") as stream:
        stream.write(prefix.ljust(HEADER_BYTES, b"\0"))
        for block in payload:
            stream.write(block)
    return header


def decode(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Expand a mixed-precision file into the existing float32 renderer fields."""
    data = path.read_bytes()
    if len(data) < HEADER_BYTES or data[:8] != MAGIC:
        raise ValueError("Invalid mixed-precision file")
    length = struct.unpack("<I", data[8:12])[0]
    if length > HEADER_BYTES - 12:
        raise ValueError("Invalid mixed-precision header length")
    header = json.loads(data[12:12 + length])
    names = list(FIELD_SHAPES) if header["rich"] else list(FIELD_SHAPES)[:4]
    if [field["name"] for field in header["fields"]] != names or header["count"] < 1:
        raise ValueError("Invalid mixed-precision fields or population")
    parameters, offset = {}, HEADER_BYTES
    for field in header["fields"]:
        name, bits = field["name"], field["bits"]
        shape = (header["count"], *FIELD_SHAPES[name])
        elements = int(np.prod(shape))
        size = (elements * bits + 7) // 8
        codes = unpack_codes(data[offset:offset + size], elements, bits)
        parameters[name] = (codes.astype(np.float32) * np.float32(field["step"])
                            + np.float32(field["min"])).reshape(shape)
        offset += size
    if offset != len(data):
        raise ValueError("Mixed-precision representation has trailing data")
    return header, parameters


def cuda_codes(parameters: dict[str, np.ndarray], field_bits: dict[str, int]) -> dict[str, tuple]:
    """Quantize each field over its own range exactly as the paper's CUDA export did.

    CUDA divides by a host scalar as a multiply by its float32 reciprocal. True
    division rounds some codes differently, so the multiply is written out.
    """
    quantized = {}
    for name, value in parameters.items():
        value = np.asarray(value, dtype=np.float32)
        bits = field_bits[name]
        minimum, maximum = float(value.min()), float(value.max())
        step = max((maximum - minimum) / (2**bits - 1), 1e-8)
        scaled = (value - np.float32(minimum)) * (np.float32(1) / np.float32(step))
        quantized[name] = (np.clip(np.rint(scaled), 0, 2**bits - 1).astype(np.uint16), minimum, step)
    return quantized


def export_paper_profile(checkpoint: Path, directory: Path) -> dict:
    """Pack raw FP32 `render_parameters_*.pt` fields with the paper's profile.

    The file is named after the checkpoint step and profile, for example
    `step_005100_mixed14.hxm`, and must decode to exactly the quantized values.
    """
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (saved["method"] != "hexel" or saved["scale_mode"] != "log"
            or saved["layout"] not in PAPER_PROFILES):
        raise ValueError("Paper precision profiles cover log-scale flat8 and dense15 Hexel fields")
    profile = PAPER_PROFILES[saved["layout"]]
    arrays = {name: value.numpy() for name, value in saved["parameters"].items()}
    quantized = cuda_codes(arrays, PROFILES[profile])
    path = directory / f"step_{saved['step']:06d}_{profile}.hxm"
    encode(path, arrays, width=saved["width"], height=saved["height"],
           rich=saved["layout"] == "dense15", field_bits=PROFILES[profile], quantized=quantized)
    _, decoded = decode(path)
    for name, value in decoded.items():
        codes, minimum, step = quantized[name]
        if not np.array_equal(value, codes.astype(np.float32) * np.float32(step) + np.float32(minimum)):
            raise RuntimeError(f"Packed field does not decode to its quantized values: {name}")
    data = path.read_bytes()
    return {"file": path.name, "profile": profile, "bytes": len(data),
            "bpp": len(data) * 8 / (saved["width"] * saved["height"]),
            "sha256": hashlib.sha256(data).hexdigest()}
