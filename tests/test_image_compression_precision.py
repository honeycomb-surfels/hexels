"""Packed byte counts and independent mixed-precision codec checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.image_compression_precision import (FIELD_SHAPES, PROFILES, cuda_codes, decode, encode,
                                                 export_paper_profile, pack_codes, unpack_codes)


@pytest.mark.parametrize("bits", [1, 6, 8, 10, 12, 14, 16])
def test_bitpacking_matches_independent_integer_stream(bits: int) -> None:
    codes = np.array([0, 1, 2**bits - 1, 2**bits // 2, 1], dtype=np.uint16)
    stream = sum(int(value) << (bits * index) for index, value in enumerate(codes))
    expected = stream.to_bytes((codes.size * bits + 7) // 8, "little")
    assert pack_codes(codes, bits) == expected
    np.testing.assert_array_equal(unpack_codes(expected, len(codes), bits), codes)


@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("profile", list(PROFILES))
def test_file_size_and_decoded_precision(tmp_path: Path, rich: bool, profile: str) -> None:
    rng = np.random.default_rng(3)
    names = list(FIELD_SHAPES) if rich else list(FIELD_SHAPES)[:4]
    values = {name: rng.normal(size=(7, *FIELD_SHAPES[name])).astype(np.float32) for name in names}
    path = tmp_path / "image.hxm"
    header = encode(path, values, width=2048, height=2048, rich=rich, field_bits=PROFILES[profile])
    actual_header, actual = decode(path)
    assert actual_header == header
    assert path.stat().st_size == 1024 + sum((value.size * PROFILES[profile][name] + 7) // 8
                                            for name, value in values.items())
    for field in header["fields"]:
        error = np.abs(actual[field["name"]] - values[field["name"]])
        assert error.max() <= field["step"] / 2 + 5e-7
        assert actual[field["name"]].dtype == np.float32


def test_invalid_codes_and_truncated_fields() -> None:
    with pytest.raises(ValueError, match="unsigned"):
        pack_codes(np.array([256], dtype=np.uint16), 8)
    with pytest.raises(ValueError, match="unsigned"):
        pack_codes(np.array([-1], dtype=np.int16), 8)
    with pytest.raises(ValueError, match="length"):
        unpack_codes(b"", 1, 12)


@pytest.mark.parametrize("rich", [False, True])
def test_public_decoder_reads_paper_precision_profiles(tmp_path: Path, rich: bool) -> None:
    from scripts.decode_image import load_representation

    rng = np.random.default_rng(19)
    names = list(FIELD_SHAPES) if rich else list(FIELD_SHAPES)[:4]
    values = {name: rng.normal(size=(5, *FIELD_SHAPES[name])).astype(np.float32) for name in names}
    profile = PROFILES["mixed14" if rich else "balanced"]
    path = tmp_path / "paper_model.bin"
    encode(path, values, width=256, height=128, rich=rich, field_bits=profile)
    header, parameters, legacy = load_representation(path)
    assert header["method"] == "hexel"
    assert header["layout"] == ("dense15" if rich else "flat8")
    assert not legacy
    for field in header["fields"]:
        value = parameters[field["name"]].numpy()
        assert value.shape == values[field["name"]].shape
        assert value.dtype == np.float32
        np.testing.assert_allclose(value, values[field["name"]], atol=field["step"] / 2 + 5e-7, rtol=0)


# Field range and one value from the published anime-1_2k paper files whose
# CUDA-written code differs from true float32 division.
@pytest.mark.parametrize("name,bits,minimum,maximum,value,published", [
    ("positions", 16, -0.042383164167404175, 1.0380816459655762, 0.670993983745575, 43270),
    ("log_scales", 14, -0.5054231286048889, 5.019032955169678, 3.0131595134735107, 10434),
    ("softness", 10, 5.789786338806152, 12.949824333190918, 9.579776763916016, 542),
])
def test_codes_reproduce_published_cuda_rounding(name: str, bits: int, minimum: float, maximum: float,
                                                 value: float, published: int) -> None:
    field = np.array([minimum, maximum, value], dtype=np.float32)
    codes, actual_minimum, step = cuda_codes({name: field}, {name: bits})[name]
    assert codes.tolist() == [0, 2**bits - 1, published]
    assert (actual_minimum, step) == (minimum, (maximum - minimum) / (2**bits - 1))
    assert np.rint((field[2] - np.float32(minimum)) / np.float32(step)) != published


@pytest.mark.parametrize("rich", [False, True])
def test_paper_profile_export_from_raw_checkpoint(tmp_path: Path, rich: bool) -> None:
    import torch

    rng = np.random.default_rng(23)
    names = list(FIELD_SHAPES) if rich else list(FIELD_SHAPES)[:4]
    values = {name: rng.normal(size=(9, *FIELD_SHAPES[name])).astype(np.float32) for name in names}
    checkpoint = tmp_path / "render_parameters_005100.pt"
    saved = {"step": 5100, "method": "hexel", "layout": "dense15" if rich else "flat8", "scale_mode": "log",
             "height": 48, "width": 64, "parameters": {name: torch.from_numpy(value) for name, value in values.items()}}
    torch.save(saved, checkpoint)
    record = export_paper_profile(checkpoint, tmp_path)
    profile = "mixed14" if rich else "balanced"
    path = tmp_path / f"step_005100_{profile}.hxm"
    data = path.read_bytes()
    assert record == {"file": path.name, "profile": profile, "bytes": len(data), "bpp": len(data) * 8 / (48 * 64),
                      "sha256": hashlib.sha256(data).hexdigest()}
    header, decoded = decode(path)
    assert (header["width"], header["height"], header["rich"]) == (64, 48, rich)
    assert {field["name"]: field["bits"] for field in header["fields"]} == {name: PROFILES[profile][name] for name in names}
    for name, (codes, minimum, step) in cuda_codes(values, PROFILES[profile]).items():
        np.testing.assert_array_equal(decoded[name], codes.astype(np.float32) * np.float32(step) + np.float32(minimum))
    with pytest.raises(FileExistsError):
        export_paper_profile(checkpoint, tmp_path)
    torch.save(dict(saved, layout="linear14"), checkpoint)
    with pytest.raises(ValueError, match="flat8 and dense15"):
        export_paper_profile(checkpoint, tmp_path)
