# Related work and implementation scope

Primary sources checked on 2026-10-28. These references distinguish mobile reconstruction pipelines, scene representations and camera-registration methods.

| Work | Relevant capability |
|---|---|
| [Scaniverse release notes](https://www.nianticspatial.com/capture/scaniverse-release-notes) | Non-LiDAR support was introduced in September 2022; local Gaussian splatting followed in March 2024. Mesh and splat modes are separate. |
| [Scaniverse camera-only announcement](https://medium.com/scaniverse/scaniverse-2-0-adds-support-for-non-lidar-devices-e5b22b6b3300) | Describes camera-derived depth with ManyDepth and reconstruction on A12/iPhone XS and later, without LiDAR. |
| [Scaniverse splatting announcement](https://medium.com/scaniverse/scaniverse-introduces-support-for-3d-gaussian-splatting-9f7f63f5469b) | Describes local capture/training and optional PLY export. Current services include separate local and cloud workflows; see [Niantic's mode descriptions](https://nianticspatial.com/pricing). |
| [Apple Object Capture for iOS](https://developer.apple.com/videos/play/wwdc2023/10191/) | Demonstrates guided image/depth capture and local USDZ reconstruction. Supported mobile hardware uses LiDAR. |
| [Apple Mac photogrammetry](https://developer.apple.com/documentation/realitykit/capturing-photographs-for-realitykit-object-capture?changes=latest_m_3&language=objc) | Accepts ordinary photographs without depth on supported Macs; LiDAR is not a universal requirement for the Mac path. |
| [2D Gaussian Splatting](https://arxiv.org/abs/2403.17888) | Uses oriented planar Gaussian disks to model surfaces intrinsically. Its abstract does not establish smartphone training. |
| [PocketGS](https://arxiv.org/abs/2601.17354) | Reports smartphone geometry construction, Gaussian initialization and 3DGS optimization. The October 20, 2026 revision describes surface-aligned anisotropic Gaussians. These are Gaussian primitives, distinct from Hexels. A public paper does not by itself establish an openly licensed code release; code availability was not established in this check. |
| [Honeycomb-Surfels](https://arxiv.org/abs/2610.37668) | Defines the Hexels representation used here. The original research implementation includes CUDA training; Hexel Edge implements a native Swift/Metal workflow. |

## Current Hexel Edge pipeline

Guided capture uses ARKit camera tracking, intrinsics and admitted image frames. LiDAR depth initializes geometry when available; the implemented fallback uses sparse tracked points. The M2 iPad capture workflow is verified; physical non-LiDAR evaluation remains open. Calibrated image-bundle import is available, while standalone image-only SfM and video-file registration remain development tasks.

The scene representation is fixed-population V1 Hexels with FP32 optimization. NVS renders the trained representation. Captured ARKit OBJ geometry and colored Hexel-patch PLY exports are separate outputs; patch export does not fuse or texture-bake the surface.

## Open-source scope

The native Hexel Edge implementation is MIT licensed. A worldwide-priority claim would need to distinguish complete mobile capture/training from desktop pipelines, planar surface primitives from volumetric Gaussians, and an OSI-open-source release from noncommercial source-available code. The brief source check did not establish an earlier app meeting the exact conjunction, and does not prove worldwide first priority.
