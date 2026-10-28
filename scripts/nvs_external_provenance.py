"""Archive external native sources, environments and installed renderer binaries."""

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path

from scripts.setup_nvs_baselines import METHODS, ROOT, environment_prefix


def capture(output: Path, methods: set[str], record: dict) -> set[str]:
    """Extend core provenance with the actual native checkouts and dependency code."""
    files, extensions = set(), set()
    suffixes = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp", ".sh", ".json", ".yml", ".yaml", ".glsl"}
    for method in sorted(methods - {"hexels", "2dgs"}):
        repository = ROOT / METHODS[method][0]
        roots = [repository, *[repository / "submodules" / name for name in METHODS[method][5]]]
        if method == "milo":
            roots += [repository / "submodules/tetra_triangulation", repository / "submodules/nvdiffrast"]
            files.add(repository / "milo/configs/fast")
        for root in roots:
            relative = str(root.relative_to(ROOT))
            record["repositories"][relative] = {
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)}
            names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root).decode().split("\0")
            files.update(root / name for name in names if name and (root / name).is_file()
                         and (Path(name).suffix in suffixes or Path(name).name == "CMakeLists.txt"))
            extensions.update(root.rglob("*.so"))
        prefix = environment_prefix(method)
        sites = list((prefix / "lib").glob("python*/site-packages"))
        for site in sites:
            for package in site.iterdir():
                if package.name.startswith(("diff_", "simple_knn", "fused_ssim", "tinycudann", "nvdiffrast", "pytorch3d")):
                    extensions.update(package.rglob("*.so"))
                    files.update(package.rglob("*.py"))
        cache = ROOT / "data/torch_extensions" / prefix.name
        extensions.update(cache.rglob("*.so"))
        frozen = subprocess.check_output([str(prefix / "bin/python"), "-m", "pip", "freeze"], text=True)
        (output / f"{method}.pip.txt").write_text(frozen)
        explicit = subprocess.check_output(["conda", "list", "-p", str(prefix), "--explicit"], text=True)
        (output / f"{method}.explicit.txt").write_text(explicit)
    with tarfile.open(output / "external_source.tar.gz", "w:gz") as archive:
        for path in sorted(files):
            relative = str(path.relative_to(ROOT))
            record["source_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative)
    with tarfile.open(output / "external_extensions.tar.gz", "w:gz") as archive:
        for path in sorted(extensions):
            relative = str(path.relative_to(ROOT))
            record["extension_sha256"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative)
    return {str(path.relative_to(ROOT)) for path in files}
