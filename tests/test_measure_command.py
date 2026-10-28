from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from scripts import measure_command


def test_peak_and_time_weighted_average_are_distinct() -> None:
    samples = [
        {"seconds": 0., "gpu_bytes": 0, "rss_bytes": 0},
        {"seconds": 1., "gpu_bytes": 100, "rss_bytes": 20},
        {"seconds": 2., "gpu_bytes": 200, "rss_bytes": 40},
    ]
    result = measure_command.integrate_samples(samples, 4.)
    assert result["peak_gpu_bytes"] == 200
    assert result["average_gpu_bytes"] == 125
    assert result["peak_rss_bytes"] == 40
    assert result["average_rss_bytes"] == 25


def process_fixture(monkeypatch, used: int | None = 100):
    child = SimpleNamespace(pid=12, memory_info=lambda: SimpleNamespace(rss=20))
    parent = SimpleNamespace(pid=11, memory_info=lambda: SimpleNamespace(rss=30), children=lambda recursive: [child])
    monkeypatch.setattr(measure_command.psutil, "Process", lambda pid: parent)
    monkeypatch.setattr(measure_command, "is_descendant", lambda pid, root: pid in {11, 12})
    records = [SimpleNamespace(pid=11, usedGpuMemory=used),
               SimpleNamespace(pid=12, usedGpuMemory=40),
               SimpleNamespace(pid=99, usedGpuMemory=1000000)]
    return SimpleNamespace(nvmlDeviceGetComputeRunningProcesses=lambda handle: records,
                           nvmlDeviceGetGraphicsRunningProcesses=lambda handle: records)


def test_process_sample_excludes_other_jobs_and_deduplicates_graphics(monkeypatch) -> None:
    nvml = process_fixture(monkeypatch)
    result = measure_command.process_sample(11, [0], nvml, include_graphics=True)
    assert result == {"gpu_bytes": 140, "rss_bytes": 50}


@pytest.mark.parametrize("used", [None, -1, 2**64 - 1])
def test_unavailable_nvml_memory_is_rejected(monkeypatch, used: int | None) -> None:
    with pytest.raises(RuntimeError, match="cannot report process memory"):
        measure_command.process_sample(11, [0], process_fixture(monkeypatch, used))


def test_cli_requests_five_ms_and_keeps_evaluation_unmodified(tmp_path: Path, monkeypatch) -> None:
    seen = {}

    def measure(command, output, interval, cwd, gpu_limit, training_method):
        seen.update(command=command, interval=interval, training_method=training_method)
        return 0

    monkeypatch.setattr(measure_command, "measure", measure)
    monkeypatch.setattr(sys, "argv", ["measure_command", "--output", str(tmp_path / "raw.json"),
                                      "--", "python", "evaluate.py"])
    with pytest.raises(SystemExit) as stopped:
        measure_command.main()
    assert stopped.value.code == 0
    assert seen == {"command": ["python", "evaluate.py"], "interval": .005, "training_method": None}


def test_sampling_failure_cannot_publish_a_complete_measurement(tmp_path: Path, monkeypatch) -> None:
    nvml = SimpleNamespace(nvmlInit=lambda: None, nvmlShutdown=lambda: None,
                           nvmlDeviceGetCount=lambda: 0)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr(measure_command, "process_rss_bytes", lambda pid: 0)
    polls = iter((None, 0))
    process = SimpleNamespace(pid=123, poll=lambda: next(polls), wait=lambda: 0)
    monkeypatch.setattr(measure_command.subprocess, "Popen", lambda *args, **kwargs: process)

    def failed_sample(*args, **kwargs):
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr(measure_command, "process_sample", failed_sample)
    output = tmp_path / "raw.json"
    with pytest.raises(RuntimeError, match="NVML unavailable"):
        measure_command.measure(["python", "train.py"], output, .005, tmp_path)
    result = json.loads(output.read_text())
    assert result["returncode"] == 0
    assert not result["complete"]
    assert result["measurement_error"] == "RuntimeError: NVML unavailable"


def test_nvs_wrapper_passes_configured_interval_and_training_policy(tmp_path: Path) -> None:
    from scripts.prepare_nvs_smoke import wrapped

    native = ["python", "train.py", "--iterations", "30000", "--data_device", "cuda"]
    command = wrapped(native, tmp_path, tmp_path / "resources.json", training_method="2dgs", interval=.013)
    assert command[command.index("--interval") + 1] == "0.013"
    assert command[command.index("--training-method") + 1] == "2dgs"
    assert command[command.index("--") + 1:] == native


def test_smoke_manifest_changes_training_sampling_without_changing_evaluation(tmp_path: Path) -> None:
    from scripts.nvs_campaign import CONFIG
    from scripts.prepare_nvs_smoke import prepare

    config = json.loads(CONFIG.read_text())
    prepare(tmp_path / "smoke", config, ["scan65"], ["3dgs_native"])
    jobs = json.loads((tmp_path / "smoke/manifests/smoke.json").read_text())["jobs"]
    train, evaluate = jobs[0]["commands"]
    assert train[train.index("--interval") + 1] == "0.005"
    assert train[train.index("--training-method") + 1] == "3dgs"
    assert evaluate[evaluate.index("--interval") + 1] == "0.2"
    assert "--training-method" not in evaluate


@pytest.mark.parametrize("wrong_interval,wrong_policy", [(False, False), (True, False), (False, True)])
def test_campaign_audit_checks_memory_protocol(tmp_path: Path, wrong_interval: bool, wrong_policy: bool) -> None:
    from scripts.audit_nvs_campaign import audit
    from scripts.training_memory_policy import TRAINING_MEMORY_POLICY

    run = tmp_path / "run"
    run.mkdir()
    result = run / "metrics.json"
    job = {"id": "dtu/scan24/2dgs", "method": "2dgs", "dataset": "dtu", "smoke": True,
           "measurement": {"resource_sample_seconds": .005, "evaluation_resource_sample_seconds": .2,
                           "training_memory_policy": TRAINING_MEMORY_POLICY}}
    (run / "job.json").write_text(json.dumps(job))
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/jobs.json").write_text(json.dumps({"jobs": [{"result": str(result)}]}))
    (run / "resources_training.json").write_text(json.dumps({
        "complete": True, "peak_gpu_bytes": 123,
        "sample_interval_seconds": .2 if wrong_interval else .005,
        "training_memory_policy": "native" if wrong_policy else TRAINING_MEMORY_POLICY,
        "training_method": "2dgs"}))
    (run / "resources_evaluation.json").write_text(json.dumps({
        "complete": True, "peak_gpu_bytes": 456, "sample_interval_seconds": .2}))
    row, = audit(tmp_path)["rows"]
    assert ("resource_training_interval" in row["issues"]) is wrong_interval
    assert ("resource_training_policy" in row["issues"]) is wrong_policy


def test_gpu_poll_does_not_enumerate_the_ram_process_tree(monkeypatch) -> None:
    nvml = process_fixture(monkeypatch)

    def forbidden(pid):
        raise AssertionError("RAM queries must not delay GPU samples")

    monkeypatch.setattr(measure_command, "process_rss_bytes", forbidden)
    assert measure_command.process_sample(11, [0], nvml, rss_bytes=50) == {
        "gpu_bytes": 140, "rss_bytes": 50}


def test_descendant_detection_covers_new_process_groups(monkeypatch) -> None:
    monkeypatch.setattr(measure_command.os, "getpgid", lambda pid: 99)
    monkeypatch.setattr(measure_command.psutil, "Process", lambda pid: SimpleNamespace(
        parents=lambda: [SimpleNamespace(pid=11)] if pid == 12 else []))
    assert measure_command.is_descendant(12, 11)
    assert not measure_command.is_descendant(99, 11)


def test_gpu_ownership_cache_distinguishes_reused_pids(monkeypatch) -> None:
    nvml = process_fixture(monkeypatch)
    stamp = {"child": 1.0}
    belongs = {"child": True}
    monkeypatch.setattr(measure_command.psutil, "Process", lambda pid: SimpleNamespace(
        create_time=lambda: stamp["child"] if pid == 12 else 0.0))
    monkeypatch.setattr(measure_command, "is_descendant", lambda pid, root: pid == 12 and belongs["child"])
    owners = {}
    assert measure_command.process_sample(11, [0], nvml, rss_bytes=0, owners=owners)["gpu_bytes"] == 140
    belongs["child"] = False
    assert measure_command.process_sample(11, [0], nvml, rss_bytes=0, owners=owners)["gpu_bytes"] == 140
    stamp["child"] = 2.0
    assert measure_command.process_sample(11, [0], nvml, rss_bytes=0, owners=owners)["gpu_bytes"] == 100
