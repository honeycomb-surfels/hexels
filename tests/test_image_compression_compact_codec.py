"""Actual byte accounting and reconstruction for isolated compact image files."""

import numpy as np
import pytest

from scripts.image_compression_compact_codec import HEADER_BYTES, decode, encode


def parameters(count: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    return {
        "positions": rng.random((count, 2), dtype=np.float32),
        "log_scales": rng.normal(size=(count, 2)).astype(np.float32),
        "rotations": rng.normal(size=(count, 1)).astype(np.float32),
        "features": rng.random((count, 3), dtype=np.float32),
        "color_gradients": rng.normal(size=(count, 2, 3)).astype(np.float32),
        "softness": rng.uniform(5, 15, count).astype(np.float32),
    }


@pytest.mark.parametrize("layout,scalars", [("flat8", 8), ("linear14", 14), ("dense15", 15)])
@pytest.mark.parametrize("method", ["hexel", "imagegs"])
def test_dense_layouts_and_shared_defaults(tmp_path, layout, scalars, method):
    values = parameters(17)
    path = tmp_path / "image.hxi"
    if method == "imagegs" and layout == "dense15":
        with pytest.raises(ValueError, match="Gaussian images"):
            encode(path, values, method=method, width=2048, height=2048, layout=layout)
        return
    encode(path, values, method=method, width=2048, height=2048, layout=layout, default_softness=9.5)
    header, actual = decode(path)
    assert path.stat().st_size == HEADER_BYTES + 17 * scalars * 2
    assert header["layout"] == layout
    for field in header["fields"]:
        name = field["name"]
        np.testing.assert_allclose(actual[name], values[name], rtol=0, atol=field["step"] / 2 + 1e-6)
    if layout == "flat8":
        np.testing.assert_array_equal(actual["color_gradients"], 0.0)
    if layout != "dense15":
        np.testing.assert_array_equal(actual["softness"], 9.5)


def test_sparse_masks_count_padding_and_omitted_values(tmp_path):
    values = parameters(17)
    gradient_mask = np.arange(17) % 3 == 0
    softness_mask = np.arange(17) % 5 == 0
    path = tmp_path / "sparse.hxi"
    encode(path, values, width=64, height=64, layout="sparse", default_softness=8.0,
           gradient_mask=gradient_mask, softness_mask=softness_mask)
    header, actual = decode(path)
    expected = HEADER_BYTES + 17 * 8 * 2 + 2 * 3 + 12 * gradient_mask.sum() + 2 * softness_mask.sum()
    assert path.stat().st_size == expected
    for name, mask in (("color_gradients", gradient_mask), ("softness", softness_mask)):
        field = next(field for field in header["fields"] if field["name"] == name)
        np.testing.assert_allclose(actual[name][mask], values[name][mask], rtol=0,
                                   atol=field["step"] / 2 + 1e-6)
    np.testing.assert_array_equal(actual["color_gradients"][~gradient_mask], 0.0)
    np.testing.assert_array_equal(actual["softness"][~softness_mask], 8.0)


def test_empty_sparse_appearance_is_decodable(tmp_path):
    path = tmp_path / "empty.hxi"
    encode(path, parameters(7), width=64, height=64, layout="sparse")
    _, actual = decode(path)
    assert path.stat().st_size == HEADER_BYTES + 7 * 16 + 2
    np.testing.assert_array_equal(actual["color_gradients"], 0.0)
    np.testing.assert_array_equal(actual["softness"], 10.0)


@pytest.mark.parametrize("size,file_bytes,bpp", [
    (256, 135632, 0.258697509765625),
    (1024, 159456, 0.30413818359375),
])
def test_codebook_rate_counts_table_indices_and_softness(tmp_path, size, file_bytes, bpp):
    values = parameters(10000)
    table = np.arange(size * 9, dtype=np.float32).reshape(size, 9) / 1000
    indices = np.arange(10000, dtype=np.uint16) % size
    path = tmp_path / "codebook.hxi"
    header = encode(path, values, width=2048, height=2048, layout="codebook",
                    codebook=table, indices=indices)
    _, actual = decode(path)
    assert path.stat().st_size == file_bytes
    assert path.stat().st_size * 8 / 2048**2 == bpp
    field = next(field for field in header["fields"] if field["name"] == "codebook")
    appearance = np.concatenate((actual["features"], actual["color_gradients"].reshape(10000, 6)), axis=1)
    np.testing.assert_allclose(appearance, table[indices], rtol=0, atol=field["step"] / 2 + 1e-6)


def test_supplied_cuda_codes_and_imagegs_identity(tmp_path):
    values = parameters(5)
    quantized = {name: (np.full_like(value, 13, dtype=np.uint16), -0.5, 0.25)
                 for name, value in values.items()}
    path = tmp_path / "imagegs.hxi"
    encode(path, values, method="imagegs", width=32, height=32, layout="flat8", quantized=quantized)
    header, actual = decode(path)
    assert header["method"] == "imagegs"
    for name in ("positions", "log_scales", "rotations", "features"):
        np.testing.assert_array_equal(actual[name], 2.75)


@pytest.mark.parametrize("change", ["truncated", "trailing", "old_magic"])
def test_invalid_file_rejected(tmp_path, change):
    path = tmp_path / "image.hxi"
    encode(path, parameters(3), width=32, height=32, layout="flat8")
    content = path.read_bytes()
    content = content[:-1] if change == "truncated" else content + b"x" if change == "trailing" else b"HXIMG001" + content[8:]
    path.write_bytes(content)
    with pytest.raises(ValueError):
        decode(path)
