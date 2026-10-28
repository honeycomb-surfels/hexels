"""CPU-only, export-time V3 SH product quantization; production loaders stay unchanged."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import platform
import resource
import statistics
import time
import zipfile

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)
FORMAT = "hexels.sh_codebook_pilot"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    archive.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def write_array(archive: zipfile.ZipFile, name: str, array: np.ndarray) -> dict:
    """Group equal-significance bytes without changing any stored bit."""
    array = np.ascontiguousarray(array)
    payload = array.view(np.uint8).reshape(-1, array.dtype.itemsize).T.tobytes()
    write_member(archive, name, payload)
    return {"member": name, "shape": list(array.shape), "dtype": array.dtype.str}


def read_array(archive: zipfile.ZipFile, descriptor: dict) -> np.ndarray:
    dtype = np.dtype(descriptor["dtype"])
    payload = np.frombuffer(archive.read(descriptor["member"]), dtype=np.uint8)
    return payload.reshape(dtype.itemsize, -1).T.copy().view(dtype).reshape(descriptor["shape"])


def validate_source(checkpoint: dict, config: dict) -> None:
    """The pilot accepts compact, flat-population FP32 V3 models only."""
    if checkpoint.get("resume_compatible") is not False or "optimizer" in checkpoint:
        raise ValueError("Use an existing compact deployment checkpoint, not a training checkpoint")
    if config.get("view_model") != "sh3" or not config.get("use_view_grads", True):
        raise ValueError("The appearance codebook pilot requires V3 SH3")
    state = checkpoint["model"]
    coefficients = state["view_grads"]
    if coefficients.dtype != torch.float32 or coefficients.shape != (len(state["means3d"]), 15, 3):
        raise ValueError("Expected view_grads with FP32 shape (N, 15, 3)")
    if state["child_parent_idx"].numel() or not bool(state["active_mask"].all()):
        raise ValueError("Export active, flat rows first; hierarchical children are not supported")
    if not bool(torch.isfinite(coefficients).all()):
        raise ValueError("SH coefficients must be finite")


def fit_codebooks(coefficients: np.ndarray, clusters: int, sample_size: int,
                  iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """One RGB-vector dictionary per SH coefficient; assign every original row."""
    from sklearn.cluster import MiniBatchKMeans

    count = len(coefficients)
    if not 2 <= clusters <= min(count, sample_size, 65536) or iterations < 1:
        raise ValueError("Need 2..65536 clusters, sufficient samples, and positive iterations")
    rng = np.random.default_rng(seed)
    selected = rng.choice(count, size=min(count, sample_size), replace=False)
    codebooks = np.empty((15, clusters, 3), dtype=np.float32)
    indices = np.empty((count, 15), dtype=np.uint8 if clusters <= 256 else np.uint16)
    for coefficient in range(15):
        values = np.ascontiguousarray(coefficients[:, coefficient])
        estimator = MiniBatchKMeans(n_clusters=clusters, init="random", n_init=1,
                                   random_state=seed + coefficient, batch_size=2048,
                                   max_iter=iterations, reassignment_ratio=0.0)
        estimator.fit(values[selected])
        codebooks[coefficient] = estimator.cluster_centers_
        indices[:, coefficient] = estimator.predict(values)
        LOGGER.info("Fitted K=%d coefficient %d/15", clusters, coefficient + 1)
    return codebooks, indices


def encode(path: Path, checkpoint: dict, config: bytes,
           codebooks: np.ndarray | None = None, indices: np.ndarray | None = None) -> None:
    """Store all model state and config; only SH values may be approximated."""
    if (codebooks is None) != (indices is None):
        raise ValueError("Codebooks and indices must be supplied together")
    metadata = {key: value for key, value in checkpoint.items() if key != "model"}
    buffer = io.BytesIO()
    torch.save(metadata, buffer)
    manifest = {"format": FORMAT, "version": 1, "quantized": codebooks is not None,
                "model_fields": list(checkpoint["model"]), "arrays": {}}
    with zipfile.ZipFile(path, "x") as archive:
        write_member(archive, "metadata.pt", buffer.getvalue())
        write_member(archive, "config.json", config)
        for name, tensor in checkpoint["model"].items():
            if name == "view_grads" and codebooks is not None:
                continue
            manifest["arrays"][name] = write_array(archive, f"state/{name}.bin", tensor.numpy())
        if codebooks is not None:
            manifest["codebooks"] = write_array(archive, "sh/codebooks.bin", codebooks)
            manifest["indices"] = write_array(archive, "sh/indices.bin", indices)
        write_member(archive, "manifest.json", json.dumps(manifest, sort_keys=True).encode())


def decode(path: Path) -> tuple[dict, bytes]:
    """Expand once to the existing FP32 checkpoint layout, not during rendering."""
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["format"] != FORMAT or manifest["version"] != 1:
            raise ValueError("Unsupported appearance archive")
        checkpoint = torch.load(io.BytesIO(archive.read("metadata.pt")), map_location="cpu", weights_only=True)
        arrays = {name: read_array(archive, descriptor) for name, descriptor in manifest["arrays"].items()}
        if manifest["quantized"]:
            codebooks = read_array(archive, manifest["codebooks"])
            indices = read_array(archive, manifest["indices"])
            arrays["view_grads"] = codebooks[np.arange(15)[None, :], indices]
            checkpoint["deployment"] = {**checkpoint["deployment"], "appearance_quantization": {
                "codec": FORMAT, "clusters": codebooks.shape[1], "rendered_quality_verified": False}}
        checkpoint["model"] = {name: torch.from_numpy(arrays[name]) for name in manifest["model_fields"]}
        return checkpoint, archive.read("config.json")


def verify_unchanged(original: dict, restored: dict, quantized: bool) -> None:
    """Compare untouched tensors bitwise, including order, signed zero and buffers."""
    if original["model"].keys() != restored["model"].keys():
        raise ValueError("Decoded model fields changed")
    for name, tensor in original["model"].items():
        if quantized and name == "view_grads":
            continue
        decoded = restored["model"][name]
        if tensor.shape != decoded.shape or tensor.dtype != decoded.dtype or tensor.numpy().tobytes() != decoded.numpy().tobytes():
            raise ValueError(f"Decoded non-quantized tensor changed: {name}")


def errors(original: np.ndarray, restored: np.ndarray) -> dict:
    difference = restored.astype(np.float64) - original
    energy = float(np.square(original.astype(np.float64)).sum())
    squared = float(np.square(difference).sum())
    return {"coefficient_rmse": float(np.sqrt(squared / original.size)),
            "coefficient_relative_l2": float(np.sqrt(squared / energy)) if energy else None,
            "coefficient_max_abs_error": float(np.abs(difference).max()),
            "per_coefficient_rmse": np.sqrt(np.mean(np.square(difference), axis=(0, 2))).tolist()}


def pilot(source: Path, config_path: Path, output: Path, clusters: list[int],
          sample_size: int, iterations: int, seed: int) -> None:
    """Measure full deployment archives, not just compressed SH payloads."""
    import sklearn

    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(source)
    config = config_path.read_bytes()
    original = torch.load(source, map_location="cpu", weights_only=True)
    validate_source(original, json.loads(config))
    coefficients = original["model"]["view_grads"].numpy()
    baseline_bytes = source.stat().st_size + len(config)
    load_times = []
    for _ in range(3):
        started = time.perf_counter()
        baseline = torch.load(source, map_location="cpu", weights_only=True)
        config_path.read_bytes()
        load_times.append(time.perf_counter() - started)
        del baseline
    report = {"complete": False, "checkpoint": str(source), "checkpoint_sha256": source_hash,
              "codec_source_sha256": sha256(Path(__file__)),
              "config_sha256": hashlib.sha256(config).hexdigest(), "primitives": len(coefficients),
              "baseline_deployment_bytes": baseline_bytes,
              "baseline_cached_load_median_seconds": statistics.median(load_times),
              "settings": {"clusters": clusters, "sample_size": sample_size, "iterations": iterations, "seed": seed},
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "torch": torch.__version__, "sklearn": sklearn.__version__},
              "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "threads": torch.get_num_threads(), "variants": [],
              "scope": "CPU storage pilot, no rendering; coefficient error is not image quality. Cached load timing includes file read and decode; no CUDA initialization/upload."}
    for size in [0, *clusters]:
        name = "lossless" if size == 0 else f"sh_pq{size}"
        started = time.perf_counter()
        codebooks, indices = fit_codebooks(coefficients, size, sample_size, iterations, seed) if size else (None, None)
        fit_seconds = time.perf_counter() - started
        archive_path = output / f"{name}.hexcodec"
        started = time.perf_counter()
        encode(archive_path, original, config, codebooks, indices)
        encode_seconds = time.perf_counter() - started
        load_times = []
        for _ in range(3):
            started = time.perf_counter()
            restored, decoded_config = decode(archive_path)
            load_times.append(time.perf_counter() - started)
            verify_unchanged(original, restored, bool(size))
            if decoded_config != config:
                raise ValueError("Config did not round-trip exactly")
        row = {"name": name, "bytes": archive_path.stat().st_size, "sha256": sha256(archive_path),
               "compression_ratio": baseline_bytes / archive_path.stat().st_size,
               "fit_seconds": fit_seconds, "encode_seconds": encode_seconds,
               "cached_decode_median_seconds": statistics.median(load_times),
               "non_sh_bitwise_exact": True, "rendered_quality_verified": False,
               "uncompressed_codebook_bytes": 0 if size == 0 else codebooks.nbytes,
               "uncompressed_index_bytes": 0 if size == 0 else indices.nbytes,
               **errors(coefficients, restored["model"]["view_grads"].numpy())}
        report["variants"].append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        LOGGER.info("%s: %.6f MB, %.4fx, SH RMSE %.8f", name, row["bytes"] / 1e6,
                    row["compression_ratio"], row["coefficient_rmse"])
        del restored, codebooks, indices
    if sha256(source) != source_hash:
        raise ValueError("Original deployment file changed during the pilot")
    report["complete"] = True
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("pilot")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--clusters", type=int, nargs="+", default=[256, 1024])
    run.add_argument("--sample-size", type=int, default=32768)
    run.add_argument("--iterations", type=int, default=20)
    run.add_argument("--seed", type=int, default=0)
    restore = commands.add_parser("decode")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    if args.action == "pilot":
        pilot(args.checkpoint.resolve(), args.config.resolve(), args.output.resolve(),
              args.clusters, args.sample_size, args.iterations, args.seed)
    else:
        checkpoint, config = decode(args.archive)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_bytes(config)
        (args.output / "checkpoints").mkdir()
        torch.save(checkpoint, args.output / "checkpoints" / f"ckpt_{checkpoint['iteration']:06d}.pt")


if __name__ == "__main__":
    main()
