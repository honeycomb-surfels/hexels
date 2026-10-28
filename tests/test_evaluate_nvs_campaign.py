"""Evaluation of decoded deployment archives without CUDA rendering."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts import evaluate_nvs_campaign


def test_decoded_archive_is_recorded_as_its_own_deployment(tmp_path: Path) -> None:
    model = tmp_path / "decoded"
    (model / "checkpoints").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    torch.save({"iteration": 30000, "model": {}, "deployment": {"format": "hexels.compact_appearance"}},
               model / "checkpoints" / "ckpt_030000.pt")
    artifact: dict = {}
    job = {"method": "hexels", "model_path": str(model), "iteration": 30000,
           "result": str(tmp_path / "evaluation" / "metrics.json")}
    evaluate_nvs_campaign.write_deployment(job, artifact)
    assert [Path(record["path"]).name for record in artifact["deployment_files"]] == ["ckpt_030000.pt", "config.json"]
    assert artifact["serialized_model_bytes"] == sum(record["bytes"] for record in artifact["deployment_files"])
    assert not (tmp_path / "evaluation").exists()


def test_model_path_override_keeps_the_training_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job = {"method": "hexels", "model_path": "pretrained/model", "result": "outputs/model/metrics.json"}
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    seen: dict = {}
    monkeypatch.setattr(evaluate_nvs_campaign, "evaluate", seen.update)
    monkeypatch.setattr(sys, "argv", ["evaluate_nvs_campaign", "evaluate", "--job", str(job_path),
                                      "--model-path", "outputs/decoded", "--result", "outputs/decoded/metrics.json"])
    evaluate_nvs_campaign.main()
    assert seen == {**job, "model_path": "outputs/decoded", "result": "outputs/decoded/metrics.json",
                    "summary_path": str(Path("pretrained/model") / "summary.json")}


def test_model_path_override_prefers_its_own_training_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trained = tmp_path / "trained"
    trained.mkdir()
    (trained / "summary.json").write_text("{}")
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"method": "hexels", "model_path": "pretrained/model", "result": "metrics.json"}))
    seen: dict = {}
    monkeypatch.setattr(evaluate_nvs_campaign, "evaluate", seen.update)
    monkeypatch.setattr(sys, "argv", ["evaluate_nvs_campaign", "evaluate", "--job", str(job_path),
                                      "--model-path", str(trained), "--result", str(tmp_path / "eval" / "metrics.json")])
    evaluate_nvs_campaign.main()
    assert seen["summary_path"] == str(trained / "summary.json")


def test_model_path_override_needs_a_fresh_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"method": "hexels", "model_path": "pretrained/model", "result": "metrics.json"}))
    monkeypatch.setattr(sys, "argv", ["evaluate_nvs_campaign", "evaluate", "--job", str(job_path),
                                      "--model-path", "outputs/decoded"])
    with pytest.raises(ValueError, match="--model-path needs a fresh --result"):
        evaluate_nvs_campaign.main()
