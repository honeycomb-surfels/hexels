# Public DTU scan24 experiment

These files describe one actual native M2 iPad run, measured on 2026-10-28. They contain public-benchmark aggregates and relative-time telemetry only. Personal signing information, device/account identifiers, workstation paths, activity timestamps, private scenes and original checkpoint/scene files are excluded.

## Conditions and result

The sixth-generation 12.9-inch iPad Pro ran the Release app without a debugger, foregrounded, offline and unplugged, with brightness approximately 50%. Mac prepared the supplied calibration and training-only feature seeds. Optimization started from those seeds, not an optimized Mac checkpoint. There were 42 training images, 6 reserved images, 6,598 fixed V1 Hexels and 640×480 optimizer/evaluation pixels. Initialization used 1280-pixel training feature extraction.

The model reached **25.20971 dB pooled raw float-RGB PSNR** and **0.798455 mean display-PNG SSIM**. All 32,920 updates were trained on iPad; the 20-minute measured interval covers 32,880 updates after 40 warmup/prelude updates. Mean update-loop time was 35.40 ms. The final checkpoint was also best. Original finite optimizer state and exact experiment context were verified; an independent Mac render exactly reproduced all six metrics and exported render/reference pixel arrays.

All 233 thermal samples were nominal, and all 233 power samples were unplugged. OS-reported charge changed 100%→95%; sampled process peak was 64.88 MiB. This establishes no reported elevated thermal pressure in this run. It does **not establish absence of thermal throttling**: clocks and Celsius were not measured. Battery readings are coarse and include screen/system activity; there is no idle baseline. Process/GPU memory overlaps on Apple unified memory and must not be added together.

This is one development scene. Reserved cameras were reused for recipe selection, and supplied calibration may use broader observations. The results are not a blind benchmark, controlled Mac-vs-iPad comparison, best-possible quality claim or extrapolation to iPhone.

The tester confirmed witnessing the network-disabled state throughout the 20-minute run. Network state is recorded in the experiment metadata; the telemetry CSV records the measured power/thermal/runtime samples.

## Files

- [dtu_scan24_ipad.json](dtu_scan24_ipad.json): exact aggregate measurements, conditions and checkpoint verification hash.
- [dtu_scan24_telemetry.csv](dtu_scan24_telemetry.csv): all 233 samples with relative time, power/thermal categories, footprints and step timing. Absolute timestamps/headroom were removed.
- [dtu_scan24_quality.csv](dtu_scan24_quality.csv): all 34 recorded quality evaluations with original relative time and metrics.
- [Public comparison](../assets/ipad-dtu-comparison.png), [resource traces](../assets/ipad-dtu-resources.png), and [actual app screenshot](../assets/ipad-dtu-app.png): public DTU figures/screenshot, with hashes in [asset provenance](../assets/provenance.json). The resource plot title was simplified to “Sampled memory”; all plotted data and pixels outside that title are unchanged. The other figures remain byte-identical to their original exports. The screenshot records the earlier build's “Hexels Edge” name; the public app is now “Hexel Edge”.

## Reproducing the workflow

Use the [local preparation tools](../tools/README.md) to acquire the pinned public DTU subset and prepare a calibrated bundle. Use the [native app guide](../apple/iOS/README.md) to build/sign locally, transfer the prepared folder, then run **Device test** unplugged after setting controlled brightness. Preserve the raw report and checkpoints locally. Exact replay requires matching the 42-view/6,598-cell detail initialization and experiment context; a generic visual hull or different point budget is a different recipe. The 6,598 cell count is an initialization outcome, not a general preset limit.

Use the report analysis tool with its documented CLI to examine quality, raw telemetry and battery validity. [Native verification](../apple/README.md) covers rendering/gradients/checkpoints; [capture fixtures](../apple/iOS/Tests/README.md) cover camera pixel/depth preparation using synthetic buffers. No dataset or trained scene is bundled here. Dataset rights and citations remain separate from the code license; see [THIRD_PARTY.md](../THIRD_PARTY.md).
