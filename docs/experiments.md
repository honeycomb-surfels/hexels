# Experiment conventions

The generic NVS campaign and the older staged geometry-first suite are separate protocols. Use the matching config and do not merge their scene subsets or held-out/all-view checkpoints.

## Current NVS campaign

Registered cameras are ordered lexically by image name. Every eighth camera, starting at index zero, is held out. Initialization uses the shared COLMAP SfM data. Metrics consume float RGB clamped once to [0,1], with no PNG round trip; LPIPS uses VGG with inputs mapped to [-1,1].

| Dataset | Scenes | Image convention |
|---|---|---|
| mipnerf360 | bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai | images_4 outdoors; images_2 indoors |
| dtu | scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122 | images at resolution factor 2 |
| tanks_and_temples | Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck, Train, Francis, Horse, Lighthouse | images_2 at resolution factor 1 |

The extra T&T NVS scenes require their corresponding registered COLMAP/image data under the configured NVS root. The initial six-scene geometry dataset guide describes a different cohort. Missing data blocks preparation/evaluation and is not removed from an average.

Fixed and mixed-growth V1/V2/V3 use the same 30K schedule and seed policy in `configs/nvs.json`. Count controls use a ceiling while retaining native adaptive updates. Learned-state controls count every required scene component, including textures, grids and topology where applicable. Model state, deployment files and training GPU memory are separate measurements.

## Training memory

The NVS and capacity runners sample NVML memory for the training process and its
descendants every 5 ms by default. They filter by PID across GPU devices, so other
jobs on the same GPU are excluded. `resource_sample_seconds` configures the
requested training interval. Evaluation retains its previous 200 ms monitoring
interval via `evaluation_resource_sample_seconds`, preserving the existing FPS
measurement path. Each interval is passed explicitly to its phase. Query overhead and
scheduling can lengthen the actual interval; reported peaks are observed samples,
not guaranteed instantaneous maxima.
Slow RAM/process-tree enumeration runs separately at its original 200 ms rate;
it does not block GPU polling. The JSON records actual GPU sampling intervals
and a separate `rss_series` for RAM peaks/averages.

Use `resources_training.json`'s `peak_gpu_bytes / 2**30` for peak training VRAM in
GiB. This includes context, allocator reservations, initialization, checkpointing
and native in-training reporting. Separate evaluation/video uses
`resources_evaluation.json`. `average_gpu_bytes` is a time-weighted average and
must not replace the peak. For staged training, take the maximum training peak
across stages per scene, then average per-scene peaks for a dataset table.
The trainer's legacy `peak_gpu_used_gb` samples whole-device usage and can include
other jobs; its PyTorch allocated/reserved peaks are separate diagnostics.
The older staged suite's `peak_train_used_or_reserved_gb` and
`peak_training_gpu_gb` columns inherit those legacy counters, not process VRAM.

Training commands use `--training-method` to keep only cached RGB images on CPU.
Masks, optional depth, camera transforms, learned appearance and optimizer state
stay on their native GPU devices. Hexels and 2DGS already cache RGB on CPU; their
camera constructors are audited without changing their initialization. Other
baseline constructors change only RGB placement. The helper checks image
residency and logs the cache/device audit. It preserves RGB values, training
losses, budgets, seeds and native topology updates.

For Nexels only, three validation VGG-LPIPS statements are excluded. Its learned
appearance, training loss, validation renders and L1/PSNR reporting remain native.
No estimated RGB-cache or VGG allocation is subtracted from a measured peak.
NVML errors make the measurement incomplete even when the child exits cleanly.

Use fresh output roots and capture new provenance after changing this policy.
Existing resource files retain the interval and policy under which they ran.
The campaign audit checks recorded intervals and the declared training policy;
measurements from a different policy cannot satisfy a new run's resource checks.
For a standalone training measurement from the native trainer's directory:

```bash
CUDA_VISIBLE_DEVICES=0 python /path/to/hexels_main/scripts/measure_command.py \
  --training-method nexels --output /fresh/resources_training.json \
  -- /path/to/native/environment/bin/python train.py <native arguments> --data_device cuda
```

Hexels commands must include `--skip_postprocess` to keep separate final
rendering/metrics outside training memory. The generic wrapper without
`--training-method` preserves the command's own cache and validation policy.

## All-view geometry

The paper's surface models use `scripts.train_scene --all-views --surface`, a single 30K run with median-depth normal consistency and ray distortion, and `--init-geometry` with the published MVS pack for the `-mvs_` models. Official DTU/T&T surface evaluation uses all-view checkpoints. Mesh extraction defaults to median depth; TSDF settings follow the dataset-specific conventions in `datasets_2dgs.md`.

`scripts.run_surface_ablation` builds loss and extraction ablations on the older staged protocol (SfM initialization by default, staged 15K-to-30K continuation, expected-depth fusion for its training arms), so it is a template for new ablations rather than a rerun of the paper's surface ablation. Its editable protocol's `baseline_root` must point to your own completed geometry run. The extraction benchmark likewise takes your own checkpoint root. No archived development output directory is required by the single-scene training/extraction interfaces.

## Image fitting

The paper image controls use native Image-GS initialization, flat/rich Hexel fields, uint16 straight-through quantization during fitting and explicit allocation policies. The default pruned Hexel recipe reaches 10K from 20K and ends at step5100. Native Image-GS ends at step10000. `fit_image` exposes fixed/growing/pruned populations, losses, seeds and budgets.

For evaluation, `decode_image` and `evaluate_image` accept stored `.hxi`/`.hxc`/`.hxm` fields or raw `render_parameters_*.pt` tensors. `fit_image` writes the paper's per-field precision profile (balanced for flat, mixed14 for rich) as `step_<step>_<profile>.hxm`. Raw FP32 quality and decoded stored-file quality are distinct conditions; post-fit per-field precision profiles live in `image_compression_precision.py`. Do not compare their values without declaring the condition.

## Slurm and local runs

GPU compilation, training, rendering and GPU tests need an NVIDIA GPU. The submitters in `scripts.run_reproduction`, `scripts.run_nvs_campaign`, `scripts.run_surface_ablation` and `scripts.run_extraction_benchmark` accept `--partition`, `--qos` and `--gres`. Partition and QoS are passed only when given, so the scheduler defaults apply otherwise; `--gres` defaults to `gpu:1`. The paper runs used `--partition main --gres gpu:a5000:1`. CPU, memory and time requests stay fixed per runner (by default 8 CPUs and 32000M per task; T&T geometry jobs request 32 CPUs and 128000M), so adapt them to your site if needed. Scheduler limits are site policy and are not part of the scientific protocol.

Without Slurm, run a prepared manifest with the same worker the batch script uses, from the repository root, for example `python -m scripts.run_minimum_3d_benchmark --manifest <root>/nvs/manifests/jobs.json --state-root <root>/nvs/runner --worker-index 0 --worker-count 1 --gpu 0`. Replace the last three options with `--launch --gpus 0,1` to start one background worker per GPU. For an NVS campaign root, complete `manifests/cameras.json` (state root `runner/cameras`) before `manifests/jobs.json` (state root `runner/jobs`). The staged suite can also run locally with `python -m scripts.run_paper_3d_benchmark run-all --gpus 0,1`.
