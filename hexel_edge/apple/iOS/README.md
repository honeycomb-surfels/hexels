# Hexel Edge for iPad and iPhone

An experimental SwiftUI app for guided local scene capture and Hexels training.
One ARSession supplies synchronized RGB, intrinsics, tracked poses and optional
scene depth. Confident training-frame depth initializes bounded geometry; the
fallback uses sparse tracked features with camera-facing normal priors. Images
retain native sensor orientation and matching calibration. Capture does not
recover poses retroactively from arbitrary existing photos or video.

## Build

1. Use an Apple-silicon Mac with full Xcode and its iOS SDK selected.
2. Open `HexelsEdge.xcodeproj` and select the `HexelsEdge` scheme.
3. Copy `Config/LocalSigning.example.xcconfig` to
   `Config/LocalSigning.xcconfig`, then set your own Team ID. The local file is
   ignored; the shared project contains no personal team/account identifiers.
   Choose your own unique bundle identifier when required.
4. Select a physical iPad or iPhone, enable Developer Mode and run. Camera
   permission is requested when starting capture. Prepared-scene training does
   not require LiDAR. LiDAR improves initial geometry and can supply a separate
   captured surface; physical non-LiDAR behavior needs validation.

A local unsigned SDK build, from the `hexel_edge` directory:

```bash
xcodebuild -project apple/iOS/HexelsEdge.xcodeproj \
  -scheme HexelsEdge -configuration Release \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
```

An unsigned build verifies compilation, not installation or camera behavior.
Simulator layout checks do not establish physical phone performance. Internal
project/module identifiers retain the original `HexelsEdge`/`HexelsMetal` names;
the displayed app name is **Hexel Edge**.

## Capture and refine

Choose **Capture scene**, use a small static well-lit scene, and move slowly
sideways and around it with overlapping views. The app accepts frames only with
normal tracking and translated viewpoints. Choose 24/48/64 frames and 640/1280 pixel
long edge within device limits. Finish requires at least 12 accepted views, a
12 cm translated baseline and 64 valid seed cells. Unsupported color metadata,
resource pressure, interruptions and failed publication produce explicit errors.

Finishing pauses capture and privately validates a complete calibrated bundle
before opening it in the scene studio. Choose a bounded quality preset, inspect
reference/reconstruction images, train, pause/save, and test interpolated camera
views. Resolution or training-view changes begin a separate fit from prepared
seeds; update-budget changes preserve the current fit. Population remains fixed.

Under **Surfaces**, inspect/export the captured ARKit OBJ when available, or
prepare colored PLY patches from trained Hexels. Captured geometry can use the
entire AR tracking session. Reserved RGB/depth seed frames are excluded from
optimization, while pose/feature tracking uses all-session observations; these
are not blind held-out geometry estimates. Surface patches are independent and
may overlap or have gaps; they are not fused or watertight meshes.

## Privacy and lifecycle

Camera images, depth, scene state and captured-scene evaluation remain in private
app storage, excluded from backup. Processing has no networking dependency or
scene-upload implementation. Export is an explicit user action. Cloud-backed
source files and destinations chosen in the system share sheet remain the
user's choice; no blanket claim is made about every OS service.

Prepared folders can also be transferred over USB Finder File Sharing to the
app's Documents/Benchmarks folder. A ZIP can be extracted in Files before
choosing **Open scene**. Import creates a separate private training copy. Public
benchmark reports may be placed in Documents/Reports; captured-scene reports
remain private with that scene until explicitly shared.

Capture and training keep the screen awake, restore its previous idle setting,
and stop or pause on backgrounding, memory/thermal pressure or battery limits.
Atomic periodic checkpoints limit lost work; they do not guarantee zero loss
from an abrupt process kill. Multiple simultaneous app scenes are disabled.

Battery percentages are coarse whole-device readings; charging intervals
invalidate discharge comparisons. Thermal state is categorical, not Celsius.
Process footprint and Metal allocation overlap and must not be added together.
No universal runtime, battery-life or phone-quality promise follows from the
available prototype.

See [engine and numeric checks](../README.md) and [capture fixture commands](Tests/README.md).
