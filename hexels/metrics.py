from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torchvision.transforms.functional as tvf
from tap import Tap


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_2dgs_to_path() -> None:
    root = _repo_root()
    sys.path.insert(0, str(root / "2d-gaussian-splatting"))


class MetricsOptions(Tap):
    model_path: str = ""
    iteration: int = -1
    split: str = "test"  # "train" | "test"
    device: str = "cuda"
    render_dir: Optional[str] = None  # Explicit ours_* directory; metrics are written inside it.

    def process_args(self) -> None:
        if not self.model_path:
            raise ValueError("--model_path is required")
        if self.split not in {"train", "test"}:
            raise ValueError("--split must be one of: train, test")


def _latest_iteration(model_path: Path, split: str) -> int:
    root = model_path / split
    if not root.exists():
        raise ValueError(f"Missing split directory: {root}")
    iters = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("ours_"):
            continue
        try:
            iters.append(int(p.name.split("_", 1)[1]))
        except ValueError:
            continue
    if not iters:
        raise ValueError(f"No render dirs found under {root} (expected {root}/ours_<iter>/...)")
    return max(iters)


def _read_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    from PIL import Image

    img = Image.open(path)
    t = tvf.to_tensor(img).unsqueeze(0)[:, :3].to(device)
    return t


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = ((prediction - target) ** 2).reshape(prediction.shape[0], -1).mean(dim=1, keepdim=True)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


def _paired_png_names(renders_dir: Path, gt_dir: Path) -> list[str]:
    render_names = {p.name for p in renders_dir.iterdir() if p.suffix.lower() == ".png"}
    gt_names = {p.name for p in gt_dir.iterdir() if p.suffix.lower() == ".png"}
    if render_names != gt_names:
        missing_gt = sorted(render_names - gt_names)
        missing_renders = sorted(gt_names - render_names)
        raise ValueError(
            "Render/GT filename mismatch: "
            f"missing_gt={missing_gt}, missing_renders={missing_renders}"
        )
    if not render_names:
        raise ValueError(f"No paired PNG images found under {renders_dir} and {gt_dir}")
    return sorted(render_names)


def _evaluation_paths(
    model_path: Path,
    split: str,
    iteration: int,
    render_dir: Path | None,
) -> tuple[Path, Path]:
    if render_dir is None:
        return model_path / split / f"ours_{iteration}", model_path
    return render_dir, render_dir


def _iteration_from_render_dir(render_dir: Path) -> int:
    fields = render_dir.name.split("_")
    if len(fields) < 2 or fields[0] != "ours" or not fields[1].isdigit():
        raise ValueError(f"Cannot infer iteration from render directory: {render_dir}")
    return int(fields[1])


@torch.no_grad()
def evaluate(
    model_path: Path,
    *,
    iteration: int,
    split: str,
    device: torch.device,
    render_dir: Path | None = None,
) -> dict[str, object]:
    _add_2dgs_to_path()
    from utils.loss_utils import ssim

    import lpips

    ours_dir, output_dir = _evaluation_paths(model_path, split, iteration, render_dir)
    renders_dir = ours_dir / "renders"
    gt_dir = ours_dir / "gt"
    if not renders_dir.exists():
        raise ValueError(f"Missing renders dir: {renders_dir}")
    if not gt_dir.exists():
        raise ValueError(f"Missing gt dir: {gt_dir}")

    fnames = _paired_png_names(renders_dir, gt_dir)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    psnrs: list[float] = []
    ssims: list[float] = []
    lpipss: list[float] = []
    per_view: dict[str, dict[str, float]] = {"PSNR": {}, "SSIM": {}, "LPIPS": {}}

    for fname in fnames:
        pred = _read_image_tensor(renders_dir / fname, device)
        gt = _read_image_tensor(gt_dir / fname, device)

        psnr_v = float(_psnr(pred, gt).item())
        ssim_v = float(ssim(pred, gt).item())
        lpips_v = float(lpips_fn(pred * 2.0 - 1.0, gt * 2.0 - 1.0).item())

        psnrs.append(psnr_v)
        ssims.append(ssim_v)
        lpipss.append(lpips_v)
        per_view["PSNR"][fname] = psnr_v
        per_view["SSIM"][fname] = ssim_v
        per_view["LPIPS"][fname] = lpips_v

    results = {
        "iteration": int(iteration),
        split: {
            "PSNR": float(sum(psnrs) / len(psnrs)),
            "SSIM": float(sum(ssims) / len(ssims)),
            "LPIPS": float(sum(lpipss) / len(lpipss)),
        },
    }
    per_view_out = {"iteration": int(iteration), split: per_view}
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    (output_dir / "per_view.json").write_text(json.dumps(per_view_out, indent=2))
    return results


def main() -> None:
    opt = MetricsOptions().parse_args()
    model_path = Path(opt.model_path)
    render_dir = Path(opt.render_dir) if opt.render_dir is not None else None
    it = int(opt.iteration)
    if it < 0:
        it = _iteration_from_render_dir(render_dir) if render_dir is not None else _latest_iteration(model_path, opt.split)
    evaluate(
        model_path,
        iteration=it,
        split=str(opt.split),
        device=torch.device(opt.device),
        render_dir=render_dir,
    )


if __name__ == "__main__":
    main()
