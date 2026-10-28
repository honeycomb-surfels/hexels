# Third-party attribution and license inventory

The root MIT license covers original Hexels additions. It does not replace the terms of the following upstream sources, their adapted files, or nested dependencies. Baseline patches remain subject to the licenses of the files they modify. Full notices are retained in each linked folder. No baseline source is presented as newly licensed MIT code.

| Source | Revision | Retained license notices |
|---|---|---|
| [image-gs](https://github.com/NYU-ICL/image-gs) | `03088368d426` | [LICENSE](baselines/image-gs/LICENSE) |
| [2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting) | `335ad612f2e7` | [LICENSE.md](baselines/2d-gaussian-splatting/LICENSE.md) |
| [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) | `54c035f7834b` | [LICENSE.md](baselines/gaussian-splatting/LICENSE.md) |
| [convex-splatting](https://github.com/convexsplatting/convex-splatting) | `27f83f8b6fbe` | [LICENSE.md](baselines/convex-splatting/LICENSE.md), [LICENSE_GS.md](baselines/convex-splatting/LICENSE_GS.md) |
| [triangle-splatting](https://github.com/trianglesplatting/triangle-splatting) | `685ff270a827` | [LICENSE.md](baselines/triangle-splatting/LICENSE.md), [LICENSE_GS.md](baselines/triangle-splatting/LICENSE_GS.md) |
| [BBSplat](https://github.com/david-svitov/BBSplat) | `cff8d6a7bce2` | [LICENSE.md](baselines/BBSplat/LICENSE.md), [LICENSE_BBSplat.txt](baselines/BBSplat/LICENSE_BBSplat.txt) |
| [nexels](https://github.com/victor-rong/nexels) | `e975afad5b30` | [LICENSE.md](baselines/nexels/LICENSE.md) |
| [mesh-splatting](https://github.com/meshsplatting/mesh-splatting) | `2a810a6c3532` | [LICENSE.md](baselines/mesh-splatting/LICENSE.md), [LICENSE_GS.md](baselines/mesh-splatting/LICENSE_GS.md) |
| [MILo](https://github.com/Anttwo/MILo) | `0c38aa65f8b7` | [LICENSE.md](baselines/MILo/LICENSE.md) |
| [SAD](https://github.com/LuckyIYI/SAD) | `d86c653abdf1` | [LICENSE](baselines/SAD/LICENSE) |
| [fused-ssim](https://github.com/rahul-goel/fused-ssim.git) | `1272e21a2823` | [LICENSE](baselines/fused-ssim/LICENSE) |

The Gaussian-Splatting license in the listed folders limits use to research/evaluation and includes redistribution and notice conditions. Some repositories also contain Apache2.0, BSD or MIT components; the presence of those licenses does not remove the terms on their Gaussian-Splatting-derived components. Read the complete notices for the files you use.

The kernel retains the [Image-GS MIT notice](submodules/hexel-kernel/licenses/Image-GS.txt), inherited [gsplat Apache-2.0 license](submodules/hexel-kernel/licenses/gsplat-Apache-2.0.txt), and [GLM notice](submodules/hexel-kernel/gsplat/cuda/csrc/third_party/glm/copying.txt). Image-GS was developed by NYU Immersive Computing Lab and builds on gsplat. The Hexel image adapter uses Image-GS initialization, model and image utilities through the separately fetched checkout.

The viewer is original implementation. References credited in its README informed interface choices; their source is not included.

Upstream datasets, pretrained perceptual weights, fonts and optional model assets are downloaded separately and retain their respective terms. This repository does not distribute the experimental datasets or claim a common license for them.

All stored notices are tied to the pinned revisions, not to a moving default branch. `baselines/repositories.json` records the pinned revisions and patch hashes.
