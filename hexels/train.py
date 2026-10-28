from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch

from .options import Hexels3DOptions
from .trainer import Hexels3DTrainer


def _prepare_quality_retry(opt: Hexels3DOptions) -> bool:
    if not bool(getattr(opt, "quality_safe_retry", False)):
        return False
    model_path = Path(str(opt.model_path))
    if not model_path.exists():
        return False
    checkpoint = model_path / "checkpoints" / f"ckpt_{int(opt.iterations):06d}.pt"
    if (model_path / "summary.json").is_file() and checkpoint.is_file():
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise RuntimeError(
                f"Complete quality output is missing its config: {config_path}"
            )
        stored = json.loads(config_path.read_text())
        values = (
            opt.as_dict()
            if hasattr(opt, "as_dict") and getattr(opt, "_parsed", False)
            else vars(opt)
        )
        current = json.loads(json.dumps(values, default=str))
        if stored != current:
            raise RuntimeError(
                f"Complete quality output has a different config: {model_path}"
            )
        return True
    if not model_path.is_dir():
        raise ValueError(f"Quality retry output is not a directory: {model_path}")
    shutil.rmtree(model_path)
    return False


def main() -> None:
    opt = Hexels3DOptions().parse_args()
    if _prepare_quality_retry(opt):
        return
    trainer = Hexels3DTrainer(opt)
    train_summary = trainer.train()

    postprocess_t = time.time()
    metrics_summary = None
    # All-view training (no --eval) has no held-out test split to render or score.
    if not bool(getattr(opt, "skip_postprocess", False)) and bool(getattr(opt, "eval", False)):
        ckpt = (
            Path(opt.model_path)
            / "checkpoints"
            / f"ckpt_{int(opt.iterations):06d}.pt"
        )
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from .metrics import evaluate
        from .render import run_render

        run_render(
            checkpoint=str(ckpt),
            model_path=str(opt.model_path),
            split="test",
            skip_path=True,
        )
        metrics_summary = evaluate(
            Path(opt.model_path),
            iteration=int(opt.iterations),
            split="test",
            device=torch.device(opt.data_device),
        )

    summary_path = Path(opt.model_path) / "summary.json"
    summary = dict(train_summary)
    summary["postprocess_seconds"] = float(time.time() - postprocess_t)
    summary["total_seconds"] = float(summary["train_seconds"]) + float(
        summary["postprocess_seconds"]
    )
    if metrics_summary is not None:
        summary["metrics"] = metrics_summary
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
