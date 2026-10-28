"""Measure process-tree GPU/RAM usage and wall time for one benchmark phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


def verify_provenance(path: Path) -> None:
    """Reject code or extension drift before a queued job touches its output."""
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads(path.read_text())
    for group in ("source_sha256", "extension_sha256"):
        for relative, expected in provenance[group].items():
            if group == "source_sha256" and "runtime_paths" in provenance and relative not in provenance["runtime_paths"]:
                continue
            source = root / relative
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen runtime changed: {relative}")
    for filename, expected in provenance.get("contract_sha256", {}).items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen experiment contract changed: {filename}")


def integrate_samples(samples: list[dict], end_seconds: float) -> dict:
    """Integrate piecewise-constant samples, including startup and shutdown time."""
    if not samples or end_seconds <= 0:
        raise ValueError("At least one sample and a positive duration are required")
    result = {"samples": len(samples), "wall_seconds": end_seconds}
    for field in ("gpu_bytes", "rss_bytes"):
        integral = 0.0
        for index, sample in enumerate(samples):
            stop = samples[index + 1]["seconds"] if index + 1 < len(samples) else end_seconds
            integral += sample[field] * max(0.0, stop - sample["seconds"])
        result[f"peak_{field}"] = max(sample[field] for sample in samples)
        result[f"average_{field}"] = integral / end_seconds
    return result


def process_sample(pid: int, handles: list, nvml: object, include_graphics: bool = False) -> dict:
    processes = []
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        pass
    pids = {process.pid for process in processes}
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    gpu = 0
    for handle in handles:
        records = list(nvml.nvmlDeviceGetComputeRunningProcesses(handle))
        if include_graphics:
            records.extend(nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
        device_usage = {}
        for process in records:
            if process.pid in pids:
                used = process.usedGpuMemory
                if used is None:
                    raise RuntimeError("NVML cannot report process memory on this device")
                device_usage[process.pid] = max(device_usage.get(process.pid, 0), used)
        gpu += sum(device_usage.values())
    return {"gpu_bytes": gpu, "rss_bytes": rss}


def measure(command: list[str], output: Path, interval: float, cwd: Path,
            gpu_limit_bytes: int | None = None) -> int:
    """Run a command, preserve its exit status and write telemetry even on failure."""
    import pynvml

    if output.exists():
        raise FileExistsError(output)
    if interval <= 0 or not command:
        raise ValueError("A command and positive sampling interval are required")
    if gpu_limit_bytes is not None and gpu_limit_bytes <= 0:
        raise ValueError("The process VRAM limit must be positive")
    pynvml.nvmlInit()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
    started = time.perf_counter()
    samples = [{"seconds": 0.0, "gpu_bytes": 0, "rss_bytes": 0}]
    environment = dict(os.environ, PYTHONNOUSERSITE="1")
    executable = Path(command[0])
    if executable.name.startswith("python") and executable.parent.name == "bin":
        prefix = executable.parent.parent
        environment.update(PATH=f"{prefix / 'bin'}:{environment['PATH']}", CONDA_PREFIX=str(prefix),
                           TORCH_EXTENSIONS_DIR=str(Path(__file__).resolve().parents[1] / "data/torch_extensions" / prefix.name),
                           MAX_JOBS=str(min(4, int(environment.get("OMP_NUM_THREADS", "4")))))
        if (prefix / "bin/nvcc").exists():
            environment["CUDA_HOME"] = str(prefix)
            environment["LD_LIBRARY_PATH"] = f"{prefix / 'lib'}:{prefix / 'targets/x86_64-linux/lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    process = subprocess.Popen(command, cwd=cwd, env=environment, start_new_session=True)
    original_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    limit_exceeded = False

    def forward(number: int, frame: object) -> None:
        if process.poll() is None:
            os.killpg(process.pid, number)

    for number in original_handlers:
        signal.signal(number, forward)
    try:
        while process.poll() is None:
            sample = process_sample(process.pid, handles, pynvml, gpu_limit_bytes is not None)
            samples.append({"seconds": time.perf_counter() - started, **sample})
            if gpu_limit_bytes is not None and sample["gpu_bytes"] > gpu_limit_bytes:
                limit_exceeded = True
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(interval)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        returncode = process.wait()
        ended = time.perf_counter() - started
        result = {
            "complete": returncode == 0 and not limit_exceeded, "returncode": returncode, "command": command,
            "cwd": str(cwd), "sample_interval_seconds": interval,
            "scope": "command process tree; sampled NVML compute memory; includes startup and shutdown",
            "integration": "left hold, time-weighted; sampled peak, not allocator high-water mark",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **integrate_samples(samples, ended), "series": samples,
        }
        if gpu_limit_bytes is not None:
            result.update(gpu_limit_bytes=gpu_limit_bytes, gpu_limit_exceeded=limit_exceeded,
                          scope="command process tree; sampled NVML compute and graphics memory; deduplicated per PID/device",
                          enforcement="sampled watchdog; observed excess kills the process group; not a hardware partition")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        for number, handler in original_handlers.items():
            signal.signal(number, handler)
        pynvml.nvmlShutdown()
    return 124 if limit_exceeded else returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--gpu-limit-bytes", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.provenance:
        verify_provenance(args.provenance)
    raise SystemExit(measure(command, args.output, args.interval, args.cwd, args.gpu_limit_bytes))


if __name__ == "__main__":
    main()
