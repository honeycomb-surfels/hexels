# Hexel Edge dependency and asset notices

Checked 2026-10-28. The package's [MIT license](LICENSE) covers its original code. It does not replace the terms of third-party software, SDKs, datasets or other assets. The full research repository's separate [inventory](../THIRD_PARTY.md) remains authoritative for its CUDA engines, baseline checkouts and submodules; those components are not silently relicensed by this package.

## Native app and shared core

[Package.swift](apple/Package.swift) declares no external Swift package dependencies. Its executable/test dependencies are the local `HexelsMetal` target. The iOS app and shared code import Apple system frameworks: ARKit, AVFoundation, UIKit, SwiftUI, Combine, Foundation, Metal, CoreGraphics, CoreVideo, ImageIO, UniformTypeIdentifiers, SceneKit, CryptoKit and SIMD, plus system Darwin APIs. SDK/framework use remains subject to Apple's applicable platform/developer terms. Their implementation source or SDK binaries are not vendored here.

The source audit found no vendored third-party implementation trees or copied vendor copyright/license headers in the package's Swift, Metal or Python files. The Hexels representation is attributed to [Honeycomb-Surfels](https://arxiv.org/abs/2610.37668), by Prakhar Kaushik, Soumava Paul and Alan Yuille; method/paper attribution is separate from software copyright ownership. This inventory does not assert that an algorithm or file has no possible third-party provenance beyond the inspected source.

**COLMAP, colmap-metal, Brush, PyTorch, CUDA, Image-GS, gsplat and the research baseline renderers are not native app runtime dependencies.** The support tools parse supplied COLMAP calibration/track files; parsing that format does not run or bundle COLMAP. Current guided capture uses ARKit tracking estimates and depth or sparse points, not a batch COLMAP executable.

## Separately installed Python support tools

Requirements are version ranges, **not exact pins or a lockfile**. The versions below describe the reference tool environment inspected for this inventory. Support libraries are installed separately; their code/wheels are not included in the iOS app. Complete publisher-supplied license snapshots are retained in [notices](notices/README.md), with byte hashes and distribution versions in [the manifest](notices/dependency_licenses.json).

| Distribution / scope | Declared requirement | Reference version | Upstream license/source |
|---|---|---|---|
| NumPy / bundle preparation and analysis | `numpy>=1.24` | 2.5.4 | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Pillow / image I/O and figures | `Pillow>=10` | 13.0.0 | [MIT-CMU / historical PIL notice](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| SciPy / optional seed geometry and analysis | `scipy>=1.10`; analysis uses `>=1.13,<2` | 1.18.1 | [BSD-3-Clause and bundled-component notices](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| OpenCV headless / optional known-camera SIFT initialization | `opencv-python-headless>=4.8` | 5.0.0.93 | [Packaging MIT; OpenCV Apache-2.0; wheel components have additional notices](https://github.com/opencv/opencv-python#licensing) |
| Matplotlib / optional report plots | `matplotlib>=3.8,<4` | 3.11.2 | [Matplotlib license agreement and retained notices](https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) |
| pytest / development checks | `pytest>=8` | 9.2.0 | [MIT](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| CairoSVG / optional SVG-to-PNG asset generator | Pinned in `assets/requirements.txt` (`CairoSVG==2.9.1`) | 2.9.1 | [LGPL-3.0-or-later](https://github.com/Kozea/CairoSVG/blob/main/LICENSE) |

Python, package transitive dependencies, and native libraries included in downloaded wheels retain their own terms. For example, OpenCV's packaging documentation identifies FFmpeg notices in its wheels; a headless wheel is not simply a redistribution of only Apache-licensed Python files. The retained license text files are reference notices, not a complete binary bill of materials. If distributing an environment or bundled wheels, retain the exact distribution's full license directory and bundled-component notices rather than substituting this summary.

## Dataset-derived figures and original assets

The public DTU comparison/screenshot assets contain **DTU scan24 benchmark imagery and rendered reconstructions**, not newly owned source photographs. The [DTU dataset homepage](https://roboimagedata.compute.dtu.dk/) describes its data as freely available citeware and requests citation of the related work. Its [MVS dataset page](https://roboimagedata.compute.dtu.dk/?page_id=36) and [publication list](https://roboimagedata.compute.dtu.dk/?page_id=39) are the primary provenance sources. Cite:

- Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, Henrik Aanæs. *Large Scale Multi-view Stereopsis Evaluation*. CVPR, 2014.
- Henrik Aanæs, Rasmus Jensen, George Vogiatzis, Engin Tola, Anders Dahl. *Large-Scale Data for Multiple-View Stereopsis*. IJCV, 2016.

Acquisition used the [public mirror](https://huggingface.co/datasets/turandai/gaussian-surfels-dtu) at revision `70ec15cd824556d17493e1c18e7e6c8b79f3ad94`. The mirror is a distribution source, not authority to replace the original dataset terms. Raw photos, prepared bundles and trained checkpoints remain experiment artifacts outside the published source package. The package MIT license does not relicense the underlying DTU material.

Logo/bee/hero SVG artwork is original code-native artwork; [asset provenance](assets/provenance.json) records source/output hashes and separates it from copied public benchmark figures. No stock logo, downloaded font files or private camera captures are included. System fonts used when rendering a figure are not redistributed as font binaries. User-created scene inputs/exports remain the user's data; the software license does not transfer ownership of them.
