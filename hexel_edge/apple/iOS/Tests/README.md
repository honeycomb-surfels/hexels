# Native capture checks

`CapturePipelineChecks.swift` executes the app's actual pixel decoder and private
bundle publisher using synthetic CoreVideo buffers and the shared engine. It
uses no ARKit stubs, camera inputs, private reports or networking.

With Xcode 27.1 / Swift 6.4 on Mac, run from the `hexel_edge` directory:

```bash
swift build -c release --package-path apple \
  --scratch-path /tmp/hexel-edge-checks --product hexels-tests
CAPTURE_MAC_SDK=$(xcrun --sdk macosx --show-sdk-path)
xcrun swiftc -O -swift-version 6 -sdk "$CAPTURE_MAC_SDK" \
  -module-cache-path /tmp/hexel-edge-module-cache \
  -I /tmp/hexel-edge-checks/release \
  apple/iOS/HexelsEdge/DevicePolicy.swift \
  apple/iOS/HexelsEdge/CapturePixels.swift \
  apple/iOS/HexelsEdge/CaptureBundleRecorder.swift \
  apple/iOS/Tests/CapturePipelineChecks.swift \
  /tmp/hexel-edge-checks/release/HexelsMetal.o \
  -o /tmp/hexel-edge-capture-checks
/tmp/hexel-edge-capture-checks
```

Xcode's SwiftPM engine emits the combined `.o` and module under `release/`.
Older build layouts may instead use per-file objects and `Modules/`. Real
CoreVideo/IOSurface allocation needs normal local process access, which a tool
sandbox may deny. These fixtures require no GPU.

The checks cover raw asymmetric rows/padding, half-pixel resizing, 601/709 matrix
and full/video range conversion, 709 transfer to sRGB, chroma siting, rejected P3/2020
metadata, depth/confidence layout, PNG round-trip, training/validation selection,
depth and sparse seed initialization, atomic publication, world-space OBJ and
cancellation cleanup. Synthetic files are deleted after the run.

The camera requests SDR. Supported inputs are 8-bit NV12, 601/709 YCbCr matrices,
709/sRGB primaries, and sRGB/709 transfer. Missing attachments use the camera's 601,
sRGB and centered-chroma defaults. Center/Left/Top/TopLeft registration is handled;
unsupported gamut, HDR, matrix or chroma metadata is rejected.

Pure policy checks, also from `hexel_edge`:

```bash
xcrun swiftc -module-cache-path /tmp/hexel-edge-module-cache \
  apple/iOS/HexelsEdge/DevicePolicy.swift \
  apple/iOS/Tests/DevicePolicyChecks.swift -o /tmp/hexel-edge-policy-checks
/tmp/hexel-edge-policy-checks
```

These numerical fixtures do not establish physical permission, AR tracking,
depth alignment, scene coverage or reconstruction quality. Test those separately
on each target device.
