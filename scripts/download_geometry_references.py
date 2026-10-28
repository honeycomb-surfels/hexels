from __future__ import annotations

import argparse
import io
import logging
import re
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from scripts.prepare_minimum_3d_benchmark import DTU


LOGGER = logging.getLogger(__name__)
BASE_URL = "https://roboimagedata2.compute.dtu.dk/data/MVS"
TNT_REFERENCE_URL = (
    "https://huggingface.co/hanlin-chen/VCR-GauS/resolve/"
    "7ab7a8798d7dadc4278ad634350236ff8b18074e/tnt.zip?download=true"
)


class RemoteZipFile(io.RawIOBase):
    """Read a public ZIP via verified HTTP ranges instead of downloading unused scans."""

    def __init__(self, url: str) -> None:
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
        self.position = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative ZIP offset")
        return self.position

    def read(self, size: int = -1) -> bytes:
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.position}-{end - 1}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            expected_range = f"bytes {self.position}-{end - 1}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise ValueError(f"Server did not honor requested ZIP range: {self.url}")
            data = response.read()
        if len(data) != end - self.position:
            raise EOFError(f"Incomplete ZIP range: {self.url}")
        self.position = end
        return data


def reference_destination(name: str, scans: set[int]) -> Path | None:
    """Select only the official masks, planes and STL clouds used by 2DGS."""
    filename = Path(name).name
    for pattern, directory in (
        (r"ObsMask(\d+)_10\.mat", "ObsMask"),
        (r"Plane(\d+)\.mat", "ObsMask"),
        (r"stl(\d+)_total\.ply", "Points/stl"),
    ):
        match = re.fullmatch(pattern, filename)
        if match and int(match.group(1)) in scans:
            return Path(directory) / filename
    return None


def download_members(url: str, output: Path, select: Callable[[str], Path | None]) -> None:
    """Extract selected ZIP entries with CRC verification and atomic per-file completion."""
    with RemoteZipFile(url) as remote, zipfile.ZipFile(remote) as zipped:
        for member in zipped.infolist():
            relative = select(member.filename)
            if relative is None:
                continue
            target = output / relative
            if target.is_file() and target.stat().st_size == member.file_size:
                LOGGER.info("Already present: %s", target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            LOGGER.info("Extracting %s (%d bytes)", relative, member.file_size)
            with zipped.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
            temporary.replace(target)


def download_dtu_references(output: Path) -> None:
    """Download the masks, planes and STL clouds for the 15 original 2DGS scans."""
    scans = {int(scene.removeprefix("scan")) for scene in DTU}
    for archive in ("SampleSet.zip", "Points.zip"):
        download_members(f"{BASE_URL}/{archive}", output, lambda name: reference_destination(name, scans))
    for scan in sorted(scans):
        for relative in (
            f"ObsMask/ObsMask{scan}_10.mat", f"ObsMask/Plane{scan}.mat",
            f"Points/stl/stl{scan:03}_total.ply",
        ):
            if not (output / relative).is_file():
                raise FileNotFoundError(output / relative)


def tnt_reference_destination(name: str) -> Path | None:
    """Select GT assets, not predicted meshes or VCR-preprocessed camera models."""
    parts = Path(name).parts
    scenes = {"Barn", "Caterpillar", "Courthouse", "Ignatius", "Meetingroom", "Truck"}
    if len(parts) != 3 or parts[0] != "tnt" or parts[1] not in scenes:
        return None
    if parts[2] in {f"{parts[1]}{suffix}" for suffix in (".ply", ".json", "_trans.txt", "_COLMAP_SfM.log")}:
        return Path(parts[1]) / parts[2]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download geometry references for the original 2DGS scene suite.")
    parser.add_argument("dataset", choices=("dtu", "tanks_and_temples"))
    parser.add_argument("--output", type=Path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser.parse_args()
    if args.dataset == "dtu":
        download_dtu_references(args.output or Path("data/dtu_official"))
    else:
        download_members(TNT_REFERENCE_URL, args.output or Path("data/tnt_official"), tnt_reference_destination)
