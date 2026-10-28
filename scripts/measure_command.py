"""Measure process-tree GPU/RAM usage and wall time for one benchmark phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import statistics
import subprocess
import threading
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.training_memory_policy import TRAINING_MEMORY_POLICY, TRAINING_METHODS


DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.005
LOGGER = logging.getLogger(__name__)


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


def process_rss_bytes(pid: int) -> int:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return rss


def is_descendant(pid: int, root_pid: int) -> bool:
    try:
        if os.getpgid(pid) == root_pid:
            return True
        return any(parent.pid == root_pid for parent in psutil.Process(pid).parents())
    except (ProcessLookupError, psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def process_sample(pid: int, handles: list, nvml: object, include_graphics: bool = False,
                   rss_bytes: int | None = None, owners: dict | None = None) -> dict:
    rss = process_rss_bytes(pid) if rss_bytes is None else rss_bytes
    family = {pid: True}
    gpu = 0
    for handle in handles:
        records = list(nvml.nvmlDeviceGetComputeRunningProcesses(handle))
        if include_graphics:
            records.extend(nvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
        device_usage = {}
        for process in records:
            if process.pid not in family:
                if owners is None:
                    family[process.pid] = is_descendant(process.pid, pid)
                else:
                    try:
                        key = (process.pid, psutil.Process(process.pid).create_time())
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        family[process.pid] = False
                        continue
                    if key not in owners:
                        owners[key] = is_descendant(process.pid, pid)
                    family[process.pid] = owners[key]
            if family[process.pid]:
                used = process.usedGpuMemory
                if used is None or used in (-1, 2**64 - 1):
                    raise RuntimeError("NVML cannot report process memory on this device")
                device_usage[process.pid] = max(device_usage.get(process.pid, 0), used)
        gpu += sum(device_usage.values())
    return {"gpu_bytes": gpu, "rss_bytes": rss}


class RssSampler:
    """Keep slow RAM/process-tree queries off the GPU sampling loop."""

    def __init__(self, pid: int, started: float) -> None:
        self.pid = pid
        self.started = started
        self.rss_bytes = 0
        self.error: Exception | None = None
        self.series = [{"seconds": 0., "gpu_bytes": 0, "rss_bytes": 0}]
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        try:
            while not self.stopped.is_set():
                self.rss_bytes = process_rss_bytes(self.pid)
                self.series.append({"seconds": time.perf_counter() - self.started,
                                    "gpu_bytes": 0, "rss_bytes": self.rss_bytes})
                self.stopped.wait(0.2)
        except Exception as error:
            LOGGER.exception("Process RAM measurement failed")
            self.error = error

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stopped.set()
        self.thread.join()


def measure(command: list[str], output: Path, interval: float, cwd: Path,
            gpu_limit_bytes: int | None = None, training_method: str | None = None) -> int:
    """Run a command, preserve its exit status and write telemetry even on failure."""
    import pynvml

    if output.exists():
        raise FileExistsError(output)
    if interval <= 0 or not command:
        raise ValueError("A command and positive sampling interval are required")
    if gpu_limit_bytes is not None and gpu_limit_bytes <= 0:
        raise ValueError("The process VRAM limit must be positive")
    requested_command = list(command)
    if training_method is not None:
        command = [command[0], str(ROOT / "scripts/training_memory_policy.py"),
                   "--method", training_method, "--", *command[1:]]
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
    measurement_error = None
    owners = {}
    rss_sampler = RssSampler(process.pid, started)
    rss_sampler.start()

    def forward(number: int, frame: object) -> None:
        if process.poll() is None:
            os.killpg(process.pid, number)

    for number in original_handlers:
        signal.signal(number, forward)
    try:
        while process.poll() is None:
            tick = time.perf_counter()
            if rss_sampler.error is not None:
                raise RuntimeError("Process RAM measurement failed") from rss_sampler.error
            sample = process_sample(process.pid, handles, pynvml, gpu_limit_bytes is not None,
                                    rss_bytes=rss_sampler.rss_bytes, owners=owners)
            samples.append({"seconds": time.perf_counter() - started, **sample})
            if gpu_limit_bytes is not None and sample["gpu_bytes"] > gpu_limit_bytes:
                limit_exceeded = True
                os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(max(0., interval - (time.perf_counter() - tick)))
    except BaseException as error:
        measurement_error = f"{type(error).__name__}: {error}"
        LOGGER.exception("Process-memory measurement failed")
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        returncode = process.wait()
        rss_sampler.stop()
        if rss_sampler.error is not None and measurement_error is None:
            measurement_error = f"{type(rss_sampler.error).__name__}: {rss_sampler.error}"
        ended = time.perf_counter() - started
        gaps = [b["seconds"] - a["seconds"] for a, b in zip(samples, samples[1:])]
        rss_stats = integrate_samples(rss_sampler.series, ended)
        result = {
            "complete": returncode == 0 and not limit_exceeded and measurement_error is None,
            "returncode": returncode, "command": command, "requested_command": requested_command,
            "measurement_error": measurement_error, "training_method": training_method,
            "training_memory_policy": TRAINING_MEMORY_POLICY if training_method else None,
            "cwd": str(cwd), "sample_interval_seconds": interval,
            "scope": "command process tree; sampled NVML compute memory; includes startup and shutdown",
            "integration": "left hold, time-weighted; sampled peak, not allocator high-water mark",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            **integrate_samples(samples, ended), "series": samples,
            "peak_rss_bytes": rss_stats["peak_rss_bytes"],
            "average_rss_bytes": rss_stats["average_rss_bytes"],
            "rss_sample_interval_seconds": 0.2, "rss_series": rss_sampler.series,
            "actual_gpu_sample_interval_seconds": {
                "mean": statistics.mean(gaps), "median": statistics.median(gaps), "max": max(gaps)
            } if gaps else None,
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
    if rss_sampler.error is not None:
        raise RuntimeError("Process RAM measurement failed") from rss_sampler.error
    return 124 if limit_exceeded else returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--interval", type=float, default=DEFAULT_SAMPLE_INTERVAL_SECONDS,
                        help="Requested NVML polling interval in seconds (default: 5 ms)")
    parser.add_argument("--training-method", choices=sorted(TRAINING_METHODS),
                        help="Apply the CPU-RGB-only training policy; exclude Nexels validation-only VGG")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--gpu-limit-bytes", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.provenance:
        verify_provenance(args.provenance)
    raise SystemExit(measure(command, args.output, args.interval, args.cwd, args.gpu_limit_bytes, args.training_method))


if __name__ == "__main__":
    main()
