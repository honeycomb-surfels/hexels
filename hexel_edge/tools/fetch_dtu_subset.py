"""Download only selected DTU scan24 RGB/masks plus cameras from a public mirror.

No pretrained normal maps, full benchmark archives, or credentials are needed.
The full remote camera-name list fixes the evaluation pool before subsetting.
Downloaded bytes are checked against the mirror's LFS SHA-256 identities.
Dataset terms remain DTU's; this script does not relicense the images.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
from pathlib import Path
import re
import urllib.request

from edge_bundle import safe_child, select_views

REPOSITORY = "turandai/gaussian-surfels-dtu"
DEFAULT_REVISION = "70ec15cd824556d17493e1c18e7e6c8b79f3ad94"
LOGGER = logging.getLogger(__name__)


def _json(url: str) -> dict | list:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fetch(output: Path, *, train_count: int = 24, test_count: int = 6,
          max_megabytes: int = 120, revision: str = DEFAULT_REVISION) -> dict:
    """Fetch a revision-pinned, size-bounded public subset with exact hashes."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("revision must be an explicit 40-character lowercase commit SHA")
    entries = _json(f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/scan24?recursive=true&limit=1000")
    images = sorted(Path(item["path"]).name for item in entries if item["path"].startswith("scan24/image/") and item["type"] == "file")
    train, test = select_views(images, train_count, test_count)
    masks = {int(Path(item["path"]).stem): item["path"] for item in entries
             if item["path"].startswith("scan24/mask/") and item["type"] == "file"}
    wanted = {"scan24/cameras.npz"}
    for name in train+test:
        wanted.add(f"scan24/image/{name}")
        if int(Path(name).stem) not in masks:
            raise ValueError(f"Public mirror lacks matching mask for {name}")
        wanted.add(masks[int(Path(name).stem)])
    files = [item for item in entries if item["path"] in wanted]
    if len(files) != len(wanted):
        raise ValueError("Public mirror lacks selected images or matching masks")
    total = sum(item["size"] for item in files)
    if total > max_megabytes*1_000_000:
        raise ValueError(f"Subset requires {total/1e6:.1f} MB, exceeds {max_megabytes} MB limit")
    output.mkdir(parents=True, exist_ok=True)

    def download(item: dict) -> dict:
        relative = item["path"].removeprefix("scan24/")
        target = safe_child(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("lfs", {}).get("oid")
        if expected is None:
            raise ValueError(f"Mirror does not expose SHA-256 for {relative}")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return {"path": relative, "sha256": expected, "bytes": item["size"]}
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{item['path']}"
        temporary = target.with_suffix(target.suffix+".partial")
        digest, count = hashlib.sha256(), 0
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                while chunk := response.read(256*1024):
                    count += len(chunk)
                    if count > item["size"]:
                        raise ValueError(f"Download exceeded declared size: {relative}")
                    handle.write(chunk)
                    digest.update(chunk)
            if count != item["size"] or digest.hexdigest() != expected:
                raise ValueError(f"Download identity mismatch: {relative}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": relative, "sha256": expected, "bytes": count}

    LOGGER.info("Checking %d files (download missing/changed), %.1f MB, revision %s", len(files), total/1e6, revision)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, files))
    index = {"dataset": "DTU scan24", "mirror": f"https://huggingface.co/datasets/{REPOSITORY}",
             "revision": revision, "image_names": images, "training_views": train,
             "heldout_views": test, "downloaded_files": records, "total_bytes": total,
             "split_rule": "every_eighth_sorted_full_source_before_subsampling",
             "original_dataset": "https://roboimagedata.compute.dtu.dk/?page_id=36"}
    (output / "source_index.json").write_text(json.dumps(index, indent=2)+"\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-views", type=int, default=24)
    parser.add_argument("--test-views", type=int, default=6)
    parser.add_argument("--max-megabytes", type=int, default=120)
    parser.add_argument("--revision", default=DEFAULT_REVISION,
                        help="Pinned dataset commit SHA; defaults to the verified scan24 revision")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    index = fetch(args.output, train_count=args.train_views, test_count=args.test_views,
                  max_megabytes=args.max_megabytes, revision=args.revision)
    LOGGER.info("Verified %.1f MB at %s", index["total_bytes"]/1e6, args.output)


if __name__ == "__main__":
    main()
