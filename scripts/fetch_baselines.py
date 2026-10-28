"""Fetch pinned upstream baselines and apply the distributed adaptation patches."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "baselines"


def run(command: list[str], directory: Path) -> None:
    """Run a source operation and leave its error visible."""
    logging.info("%s", " ".join(command))
    subprocess.run(command, cwd=directory, check=True)


def apply_patch(repository: Path, patch: Path, expected: str) -> None:
    """Accept an already applied patch, but reject unrelated checkout changes."""
    if hashlib.sha256(patch.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Patch hash mismatch: {patch}")
    if not patch.stat().st_size:
        return
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=repository,
                           capture_output=True, text=True)
    if check.returncode == 0:
        run(["git", "apply", str(patch)], repository)
        return
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                             cwd=repository, capture_output=True, text=True)
    if reverse.returncode:
        raise RuntimeError(f"Conflicting baseline changes at {repository}: {check.stderr}")


def fetch(name: str, local_source_root: Path | None = None) -> Path:
    """Restore a baseline without replacing an existing checkout or its results."""
    specification = json.loads((SOURCES / name / "source.json").read_text())
    destination = ROOT / name
    local = local_source_root / name if local_source_root else None
    if not destination.exists():
        source = str(local) if local else specification["url"]
        run(["git", "clone", "--no-checkout", source, str(destination)], ROOT)
        run(["git", "checkout", "--detach", specification["revision"]], destination)
        run(["git", "remote", "set-url", "origin", specification["url"]], destination)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=destination,
                                       text=True).strip()
    if revision != specification["revision"]:
        raise ValueError(f"Unexpected revision at {destination}: {revision}")
    modules = destination / ".gitmodules"
    if local and modules.exists():
        entries = subprocess.check_output(
            ["git", "config", "-f", str(modules), "--get-regexp", r"submodule\..*\.path"],
            cwd=destination, text=True)
        for entry in entries.splitlines():
            key, path = entry.split(maxsplit=1)
            if (local / path / ".git").exists():
                run(["git", "config", key.removesuffix(".path") + ".url", str(local / path)], destination)
    if modules.exists():
        run(["git", "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive"], destination)
    if specification["patch"] is not None:
        apply_patch(destination, SOURCES / name / specification["patch"], specification["patch_sha256"])
    for child in specification["submodule_patches"]:
        apply_patch(destination / child["path"], SOURCES / name / child["patch"], child["patch_sha256"])
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    names = json.loads((SOURCES / "repositories.json").read_text())
    parser.add_argument("--methods", nargs="+", choices=names,
                        default=["image-gs", "2d-gaussian-splatting"])
    parser.add_argument("--local-source-root", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    for name in args.methods:
        fetch(name, args.local_source_root)


if __name__ == "__main__":
    main()
