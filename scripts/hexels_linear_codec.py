"""V1/V2 RGB-gradient codebooks; base RGB and all geometry stay bitwise lossless."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from scripts.hexels_sh_codec import read_array, write_array, write_member

FORMAT = "hexels.linear_appearance_codebook"


def fields(checkpoint: dict, config: dict) -> list[str]:
    """Only compact, active, linear V1/V2 populations have this codec contract."""
    if checkpoint.get("resume_compatible") is not False or "optimizer" in checkpoint:
        raise ValueError("Expected an inference-only compact checkpoint")
    if config["color_model"] != "linear" or config["view_model"] not in {"none", "linear"}:
        raise ValueError("Linear appearance codebooks support V1/V2, not V3 SH")
    state = checkpoint["model"]
    count = len(state["means3d"])
    if state["child_parent_idx"].numel() or not bool(state["active_mask"].all()):
        raise ValueError("Expected active flat primitives without children")
    names = ["color_grads"] + (["view_grads"] if config["use_view_grads"] else [])
    for name in names:
        tensor = state[name]
        width = 2 if name == "color_grads" else 3
        if tensor.shape != (count, width, 3) or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Invalid active linear appearance field: {name}")
    return names


def fit(checkpoint: dict, names: list[str], clusters: int = 1024, sample_size: int = 32768,
        iterations: int = 20, seed: int = 0) -> tuple[dict, dict]:
    """Fit one RGB-vector dictionary per tangent/directional coefficient."""
    from sklearn.cluster import MiniBatchKMeans

    count = len(checkpoint["model"]["means3d"])
    if not 2 <= clusters <= min(count, sample_size, 65536):
        raise ValueError("Insufficient samples for the requested dictionary size")
    selected = np.random.default_rng(seed).choice(count, min(count, sample_size), replace=False)
    books, codes = {}, {}
    block = 0
    for name in names:
        coefficients = checkpoint["model"][name].numpy()
        width = coefficients.shape[1]
        books[name] = np.empty((width, clusters, 3), dtype=np.float32)
        codes[name] = np.empty((count, width), dtype=np.uint8 if clusters <= 256 else np.uint16)
        for coefficient in range(width):
            values = np.ascontiguousarray(coefficients[:, coefficient])
            estimator = MiniBatchKMeans(n_clusters=clusters, init="random", n_init=1, random_state=seed + block,
                batch_size=2048, max_iter=iterations, reassignment_ratio=0.0)
            estimator.fit(values[selected])
            books[name][coefficient] = estimator.cluster_centers_
            codes[name][:, coefficient] = estimator.predict(values)
            block += 1
    return books, codes


def encode(path: Path, checkpoint: dict, config: bytes, books: dict, codes: dict) -> None:
    """Pack all state, with explicit names for the only approximated tensors."""
    names = fields(checkpoint, json.loads(config))
    if set(books) != set(codes) or set(books) != set(names):
        raise ValueError("Dictionary fields differ from the appearance layout")
    metadata = io.BytesIO()
    torch.save({key: value for key, value in checkpoint.items() if key != "model"}, metadata)
    manifest = {"format": FORMAT, "version": 1, "fields": list(checkpoint["model"]), "arrays": {}, "codebooks": {}}
    with zipfile.ZipFile(path, "x") as archive:
        write_member(archive, "metadata.pt", metadata.getvalue())
        write_member(archive, "config.json", config)
        for name, tensor in checkpoint["model"].items():
            if name in names:
                manifest["codebooks"][name] = {"books": write_array(archive, f"{name}/books", books[name]),
                    "codes": write_array(archive, f"{name}/codes", codes[name])}
            else:
                manifest["arrays"][name] = write_array(archive, f"state/{name}", tensor.numpy())
        write_member(archive, "manifest.json", json.dumps(manifest, sort_keys=True).encode())


def decode(path: Path) -> tuple[dict, bytes]:
    """Expand the codebooks once; the renderer still receives ordinary FP32 arrays."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format"] != FORMAT or manifest["version"] != 1:
            raise ValueError("Unsupported linear appearance archive")
        checkpoint = torch.load(io.BytesIO(archive.read("metadata.pt")), map_location="cpu", weights_only=True)
        arrays = {name: read_array(archive, descriptor) for name, descriptor in manifest["arrays"].items()}
        for name, descriptors in manifest["codebooks"].items():
            books = read_array(archive, descriptors["books"])
            codes = read_array(archive, descriptors["codes"])
            arrays[name] = books[np.arange(books.shape[0])[None, :], codes]
        checkpoint["model"] = {name: torch.from_numpy(arrays[name]) for name in manifest["fields"]}
        checkpoint["deployment"] = {**checkpoint["deployment"], "appearance_quantization": {
            "codec": FORMAT, "fields": list(manifest["codebooks"]), "rendered_quality_verified": False}}
        return checkpoint, archive.read("config.json")


def verify(original: dict, decoded: dict, names: list[str]) -> None:
    """Require all untouched tensors, including base RGB, to remain bitwise exact."""
    if original["model"].keys() != decoded["model"].keys():
        raise ValueError("Decoded model fields differ")
    for name, tensor in original["model"].items():
        restored = decoded["model"][name]
        if restored.dtype != tensor.dtype or restored.shape != tensor.shape:
            raise ValueError(f"Decoded tensor layout differs: {name}")
        if name not in names and restored.numpy().tobytes() != tensor.numpy().tobytes():
            raise ValueError(f"Lossless field changed: {name}")
