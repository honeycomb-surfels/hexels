# Local scene preparation and verification

These optional Mac tools prepare calibrated inputs for the native engine and
analyze exported reports. They use supplied poses; they do not implement
image-only registration or send user scenes to a service. The downloader fetches
only explicitly requested public DTU benchmark files.

Run from the `hexel_edge` directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements-dev.txt
.venv/bin/python tools/edge_bundle.py --synthetic --long-edge 32 \
  --output outputs/synthetic32
```

For an undistorted calibrated COLMAP scene with images and a text/binary
`PINHOLE` or `SIMPLE_PINHOLE` sparse model:

```bash
.venv/bin/python tools/edge_bundle.py --source PATH_TO_SCENE \
  --format colmap --images images --long-edge 640 --max-points 20000 \
  --output outputs/my_scene
```

Use `--colmap-model sparse/1` for an explicit disconnected component. Camera
records must match source images; distorted models need undistortion first.
Initialization re-triangulates selected training tracks rather than silently
using all-view point positions or colors. Supplied pose calibration can still
have all-view provenance.

Optional public DTU acquisition and preparation:

```bash
.venv/bin/python tools/fetch_dtu_subset.py \
  --revision 70ec15cd824556d17493e1c18e7e6c8b79f3ad94 --output data/scan24
.venv/bin/python tools/edge_bundle.py --source data/scan24 --format dtu \
  --long-edge 640 --max-points 20000 --hull-resolution 48 \
  --output outputs/scan24_640
```

The source revision is pinned and downloaded bytes are verified. Camera names
reserve the evaluation pool before training subsampling. A visual hull is a
coarse silhouette initialization, not MVS. Optional known-camera feature seeding
requires `requirements-quality.txt`; inspect `edge_bundle.py --help` for its
explicit settings. No dataset, trained model or private capture is distributed
with these tools. Keep generated inputs and outputs outside version control.

`verify_ipad_benchmark.py` checks exact saved context, rates, steps and tensor
state against an original bundle/report. Its optional normalized comparison
checkpoint is a separate header-only comparison copy, never the original device
artifact. `summarize_device_benchmark.py` computes raw/report and display-image
metrics and rejects invalid battery comparisons; install
`requirements-analysis.txt` for plots. Both scripts document arguments with
`--help`. Charge is coarse whole-device telemetry, not watts, and thermal state
is not Celsius. Scene content stays local unless the user explicitly exports it.

The preparation, quality and report tests live in the parent research repository.
From that repository's root:

```bash
hexel_edge/.venv/bin/python -m pytest -q \
  tests/test_edge_bundle.py tests/test_edge_quality.py \
  tests/test_device_benchmark_report.py tests/test_ipad_benchmark_verification.py
```

Install `hexel_edge/tools/requirements-quality.txt` and
`hexel_edge/tools/requirements-analysis.txt` into the same environment when running
all four files. These tests use synthetic/local fixtures and require no dataset
download or training run. See the [Apple engine guide](../apple/README.md) for the
native bundle contract and runtime checks.
