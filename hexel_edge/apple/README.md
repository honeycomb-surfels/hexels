# Hexel Edge Apple engine

A fixed-population V1 Hexels renderer and trainer in Swift/Metal. The package
provides ray/plane rendering, soft hexagonal support, spatial RGB, analytic
backpropagation for all 20 parameters, Adam, bounded tile lists and exact local
optimizer checkpoints. Binning and depth sorting run on CPU; rendering,
gradients and updates run on Metal. No CUDA or third-party Swift packages are
required.

The [native iOS app](iOS/README.md) adds guided ARKit capture, private local bundle
preparation, training, calibrated and interpolated views, and surface exports.
The package targets macOS 14+ and iOS/iPadOS 18+ with an Apple7-or-later GPU check.
The standalone app and engine are experimental; physical iPhone and non-LiDAR
capture performance require their own validation.

## Build and verify on Mac

From the `hexel_edge` directory with an Apple-silicon Mac and full Xcode selected:

```bash
swift build -c release --package-path apple --product hexels-bench
swift run -c release --package-path apple hexels-tests --require-metal
```

The verification executable uses throwing assertions, an independent
Double-precision renderer, finite differences, geometry fixtures and checkpoint
continuation checks. `--require-metal` fails when no usable GPU exists; it does
not turn missing hardware into a passing result. No input dataset is needed for
the native checks.

## Prepared scene format

A scene folder contains `manifest.json`, little-endian `parameters.f32` with 20
Float32 values per primitive, `view_order.u32` containing training-frame indices,
and calibrated PNG images. Schema1 manifests use variant `v1`, parameter layout
`hexels_v1_log_wxyz_20`, intrinsics `[fx,fy,cx,cy]`, and row-major camera-to-world
matrices with right/down/forward axes. Every image declares `train` or `test`.
Image dimensions and calibration must agree. The app can prepare this format
locally from its camera; arbitrary imported photographs without poses require a
separate registration workflow.

Inspect and train an existing local bundle:

```bash
swift run -c release --package-path apple hexels-bench inspect --bundle PATH_TO_SCENE
swift run -c release --package-path apple hexels-bench train \
  --bundle PATH_TO_SCENE --output outputs/example --steps 1000
```

`--steps` counts additional updates. Optional manifest `recipe_hint` is
`standard` or `detail`; absence preserves standard rates. Checkpoints retain
parameters, moments, rates, step and experiment context. Incompatible scene or
mode resumes are rejected. RGB MSE/PSNR use raw encoded-sRGB floats; PNG display
clips to [0,1].

## Resource and geometry limits

The default CLI ceiling is 512 MiB of modeled workspace and two million tile
intersections. Admission is separate from measured process footprint and is not
an OS memory entitlement. Exceeding a limit rejects the frame rather than
silently discarding contributions.

Population is fixed. Growth, SH appearance, quantized training, image-only SfM,
mesh fusion and watertight/textured surface reconstruction are not implemented.
Captured ARKit OBJ geometry is distinct from trained colored Hexel patches.
Patch PLY exports represent independent hard-support surfaces, not the complete
soft optical renderer.

See [iOS setup](iOS/README.md) and [native capture fixtures](iOS/Tests/README.md).
