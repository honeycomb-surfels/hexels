"""Archive the actual release sources, installed binaries and environment."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from scripts.run_paper_3d_benchmark import ROOT


def capture(output: Path) -> None:
    """Freeze dirty local source without requiring the development checkout."""
    output.mkdir(parents=True, exist_ok=False)
    repositories = [ROOT, ROOT / "submodules/hexel-kernel", ROOT / "submodules/hexel-viewer",
                    ROOT / "image-gs", ROOT / "2d-gaussian-splatting",
                    ROOT / "2d-gaussian-splatting/submodules/diff-surfel-rasterization",
                    ROOT / "2d-gaussian-splatting/submodules/simple-knn"]
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".inl", ".sh",
                ".json", ".yaml", ".yml", ".toml", ".patch", ".md", ".mjs", ".css", ".html"}
    files, revisions = set(), {}
    for repository in repositories:
        relative = str(repository.relative_to(ROOT))
        revisions[relative] = {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], cwd=repository, text=True),
        }
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=repository).decode().split("\0")
        files.update(repository / name for name in names if name and (repository / name).is_file()
                     and (repository / name).suffix in suffixes)
    hashes = {}
    with tarfile.open(output / "source.tar.gz", "w:gz") as archive:
        for path in sorted(files):
            relative = str(path.relative_to(ROOT))
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative)
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "environment.pip.txt").write_text(frozen)
    extensions = {}
    for package in ("gsplat", "diff_surfel_rasterization", "simple_knn", "fused_ssim"):
        specification = find_spec(package)
        if specification is None:
            raise ModuleNotFoundError(f"Install the release dependency before capture: {package}")
        directory = Path(specification.origin).parent
        paths = list(directory.rglob("*.so"))
        if package == "fused_ssim":
            native = find_spec("fused_ssim_cuda")
            if native is not None:
                paths.append(Path(native.origin))
        for path in paths:
            key = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            extensions[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    weights = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (ROOT / "data/torch_cache/hub/checkpoints").glob("*.pth")}
    record = {"captured_at": datetime.now(UTC).isoformat(), "python": sys.executable,
              "repositories": revisions, "source_sha256": hashes,
              "metric_weights_sha256": weights, "extension_sha256": extensions}
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args().output.resolve())
