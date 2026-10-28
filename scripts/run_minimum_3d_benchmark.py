from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs/minimum_3d/manifests/jobs.json"
DEFAULT_STATE_ROOT = ROOT / "outputs/minimum_3d/runner"
LOGGER = logging.getLogger("minimum_3d_runner")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jobs(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    jobs = manifest["jobs"]
    if any(job["status"] != "ready" for job in jobs):
        raise ValueError("The benchmark manifest contains jobs that are not ready")
    return jobs


def job_name(job: dict[str, Any]) -> str:
    return f"{job['dataset']}__{job['scene']}__{job['method']}"


def shard_jobs(
    jobs: list[dict[str, Any]],
    worker_index: int,
    worker_count: int,
) -> list[dict[str, Any]]:
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    return jobs[worker_index::worker_count]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def run_job(
    job: dict[str, Any],
    *,
    gpu: str,
    state_root: Path,
    manifest_sha256: str,
) -> bool:
    name = job_name(job)
    status_path = state_root / "status" / f"{name}.json"
    log_path = state_root / "logs" / f"{name}.log"
    previous_status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    if previous_status and previous_status.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"Job {name} status belongs to a different manifest")
    if previous_status.get("status") == "completed":
        LOGGER.info("Skipping completed job %s", name)
        return True

    command_count = len(job["commands"])
    completed_commands = min(previous_status.get("completed_commands", 0), command_count)
    started_at = previous_status.get("started_at", _timestamp())
    resumed_at = _timestamp()
    status: dict[str, Any] = {
        "job": name,
        "gpu": gpu,
        "pid": os.getpid(),
        "status": "running",
        "manifest_sha256": manifest_sha256,
        "command_count": command_count,
        "completed_commands": completed_commands,
        "started_at": started_at,
        "updated_at": resumed_at,
        "log": str(log_path),
        "attempt": previous_status.get("attempt", 0) + 1,
        "command_timings": previous_status.get("command_timings", []),
        "command_attempts": previous_status.get("command_attempts", []),
        "hostname": socket.gethostname(),
        "slurm": {name: value for name, value in os.environ.items() if name.startswith("SLURM_")},
    }
    if completed_commands:
        status["resumed_at"] = resumed_at
    _write_json(status_path, status)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a") as log:
        action = "RESUME" if completed_commands else "START"
        log.write(
            f"\n[{resumed_at}] {action} {name} gpu={gpu} "
            f"command={completed_commands + 1}/{command_count}\n"
        )
        log.flush()
        for command_index, command in enumerate(
            job["commands"][completed_commands:],
            start=completed_commands,
        ):
            status["current_command"] = command_index
            status["updated_at"] = _timestamp()
            command_started_at = status["updated_at"]
            _write_json(status_path, status)
            log.write(
                f"[{status['updated_at']}] COMMAND "
                f"{command_index + 1}/{len(job['commands'])}\n"
            )
            log.write(" ".join(command) + "\n")
            log.flush()
            command_start = time.monotonic()
            result = subprocess.run(
                command,
                cwd=job["cwd"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            command_finished_at = _timestamp()
            command_seconds = time.monotonic() - command_start
            timing = {
                "command": command_index,
                "started_at": command_started_at,
                "finished_at": command_finished_at,
                "seconds": command_seconds,
                "returncode": result.returncode,
                "stage": job.get("command_stages", ["unspecified"] * command_count)[command_index],
            }
            status["command_attempts"].append(timing)
            status["command_timings"] = [
                item
                for item in status["command_timings"]
                if item["command"] != command_index
            ]
            status["command_timings"].append(timing)
            if result.returncode != 0:
                status.update(
                    {
                        "status": "failed",
                        "returncode": result.returncode,
                        "failed_command": command_index,
                        "updated_at": command_finished_at,
                    }
                )
                _write_json(status_path, status)
                log.write(f"[{status['updated_at']}] FAILED returncode={result.returncode}\n")
                return False
            status["completed_commands"] = command_index + 1
            status["updated_at"] = command_finished_at
            _write_json(status_path, status)

        finished_at = _timestamp()
        status.update({"status": "completed", "finished_at": finished_at})
        status["total_seconds"] = (
            datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
        ).total_seconds()
        status["successful_command_seconds"] = sum(item["seconds"] for item in status["command_timings"])
        status["attempted_command_seconds"] = sum(item["seconds"] for item in status["command_attempts"])
        status.pop("current_command", None)
        _write_json(status_path, status)
        log.write(f"[{status['finished_at']}] COMPLETED\n")
    return True


def run_worker(
    *,
    manifest_path: Path,
    state_root: Path,
    gpu: str,
    worker_index: int,
    worker_count: int,
) -> int:
    jobs = shard_jobs(load_jobs(manifest_path), worker_index, worker_count)
    manifest_sha256 = _sha256(manifest_path)
    LOGGER.info("Worker %d on GPU %s received %d jobs", worker_index, gpu, len(jobs))
    for job in jobs:
        if not run_job(
            job,
            gpu=gpu,
            state_root=state_root,
            manifest_sha256=manifest_sha256,
        ):
            LOGGER.error("Worker %d stopped after %s failed", worker_index, job_name(job))
            return 1
    return 0


def run_named_job(
    *,
    manifest_path: Path,
    state_root: Path,
    gpu: str,
    name: str,
) -> int:
    jobs = [job for job in load_jobs(manifest_path) if job_name(job) == name]
    if len(jobs) != 1:
        raise ValueError(f"Expected exactly one manifest job named {name!r}")
    succeeded = run_job(
        jobs[0],
        gpu=gpu,
        state_root=state_root,
        manifest_sha256=_sha256(manifest_path),
    )
    return 0 if succeeded else 1


def launch_workers(
    *,
    manifest_path: Path,
    state_root: Path,
    gpus: list[str],
) -> dict[str, Any]:
    jobs = load_jobs(manifest_path)
    if not gpus or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError("GPU IDs must be a non-empty list of distinct integers")

    launch_path = state_root / "launch.json"
    if launch_path.is_file():
        previous = json.loads(launch_path.read_text())
        live_pids = [
            worker["pid"]
            for worker in previous["workers"]
            if Path(f"/proc/{worker['pid']}").exists()
        ]
        if live_pids:
            raise RuntimeError(f"Benchmark workers are already running: {live_pids}")

    worker_dir = state_root / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    workers = []
    for worker_index, gpu in enumerate(gpus):
        log_path = worker_dir / f"gpu{gpu}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(manifest_path),
            "--state-root",
            str(state_root),
            "--gpus",
            ",".join(gpus),
            "--worker-index",
            str(worker_index),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        workers.append(
            {
                "worker_index": worker_index,
                "gpu": gpu,
                "pid": process.pid,
                "job_count": len(shard_jobs(jobs, worker_index, len(gpus))),
                "log": str(log_path),
            }
        )

    launch = {
        "status": "launched",
        "launched_at": _timestamp(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "job_count": len(jobs),
        "workers": workers,
    }
    _write_json(launch_path, launch)
    return launch


def resume_workers(
    *,
    manifest_path: Path,
    state_root: Path,
    gpus: list[str],
    worker_indexes: list[int],
) -> dict[str, Any]:
    launch_path = state_root / "launch.json"
    launch = json.loads(launch_path.read_text())
    jobs = load_jobs(manifest_path)
    worker_count = len(gpus)
    selected_workers = []
    for worker_index in worker_indexes:
        if worker_index < 0 or worker_index >= worker_count:
            raise ValueError("worker indexes must be in [0, worker_count)")
        previous = next(
            worker
            for worker in launch["workers"]
            if worker["worker_index"] == worker_index
        )
        if Path(f"/proc/{previous['pid']}").exists():
            raise RuntimeError(
                f"Worker {worker_index} is still running with PID {previous['pid']}"
            )
        gpu = gpus[worker_index]
        log_path = state_root / "workers" / f"gpu{gpu}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--manifest",
            str(manifest_path),
            "--state-root",
            str(state_root),
            "--gpus",
            ",".join(gpus),
            "--worker-index",
            str(worker_index),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        previous.update({"pid": process.pid, "resumed_at": _timestamp()})
        selected_workers.append(previous.copy())

    launch.update(
        {
            "status": "resumed",
            "resumed_at": _timestamp(),
            "manifest_sha256": _sha256(manifest_path),
            "job_count": len(jobs),
        }
    )
    _write_json(launch_path, launch)
    return {"status": "resumed", "workers": selected_workers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimum 3D benchmark across GPUs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume-worker-indexes")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--job-name")
    parser.add_argument("--gpu")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    state_root = args.state_root.resolve()
    gpus = args.gpus.split(",")
    if args.launch:
        launch = launch_workers(
            manifest_path=manifest_path,
            state_root=state_root,
            gpus=gpus,
        )
        for worker in launch["workers"]:
            LOGGER.info(
                "GPU %s worker PID %d: %d jobs",
                worker["gpu"],
                worker["pid"],
                worker["job_count"],
            )
        return
    if args.resume_worker_indexes is not None:
        worker_indexes = [
            int(value) for value in args.resume_worker_indexes.split(",")
        ]
        resumed = resume_workers(
            manifest_path=manifest_path,
            state_root=state_root,
            gpus=gpus,
            worker_indexes=worker_indexes,
        )
        for worker in resumed["workers"]:
            LOGGER.info(
                "Resumed GPU %s worker PID %d",
                worker["gpu"],
                worker["pid"],
            )
        return
    if args.job_name is not None:
        if args.gpu is None:
            raise ValueError("--job-name requires --gpu")
        raise SystemExit(
            run_named_job(
                manifest_path=manifest_path,
                state_root=state_root,
                gpu=args.gpu,
                name=args.job_name,
            )
        )
    if args.worker_index is None:
        raise ValueError(
            "Specify --launch, --resume-worker-indexes, --worker-index, or --job-name"
        )
    if args.worker_count is not None and args.gpu is None:
        raise ValueError("--worker-count requires --gpu (the scheduler-assigned device)")
    raise SystemExit(
        run_worker(
            manifest_path=manifest_path,
            state_root=state_root,
            gpu=args.gpu if args.gpu is not None else gpus[args.worker_index],
            worker_index=args.worker_index,
            worker_count=args.worker_count if args.worker_count is not None else len(gpus),
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
