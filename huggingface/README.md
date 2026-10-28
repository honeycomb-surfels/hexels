---
license: other
license_name: hexels-checkpoint-dataset-terms
license_link: LICENSE
library_name: pytorch
language:
- en
tags:
- hexels
- novel-view-synthesis
- 3d-reconstruction
- surface-reconstruction
- image-representation
- cuda
---
<p align="center"><a href="https://honeycomb-surfels.github.io/"><img src="assets/hexels-logo.gif" width="120" alt="Honeycomb-Surfels animated bee and honeycomb logo"></a></p>

# Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction

<p align="center"><a href="https://toshi2k2.github.io/">Prakhar Kaushik*</a> · <a href="https://mvp18.github.io/">Soumava Paul*</a> · <a href="https://www.cs.jhu.edu/~ayuille1/">Alan Yuille</a><br>CCVL, Johns Hopkins University<br><sub>* Equal contribution</sub></p>

<p align="center"><a href="https://honeycomb-surfels.github.io/">Project page</a> · <a href="https://honeycomb-surfels.github.io/h1.pdf">Paper</a> · <a href="https://arxiv.org/abs/2610.37668">arXiv</a> · <a href="https://github.com/honeycomb-surfels/hexels">GitHub code</a> · <a href="https://huggingface.co/jhu-ccvl/Hexels">Paper checkpoints</a></p>

## Run the models

Install the [training/inference code and both submodules](https://github.com/honeycomb-surfels/hexels#install), then
obtain the corresponding [datasets](https://github.com/honeycomb-surfels/hexels#download-the-four-datasets).
Source-file hashes for this release are recorded in `code/`.
These are custom PyTorch/CUDA scene models, not Transformers or hosted-inference
pipeline models.

Download a single model rather than the entire collection:

```bash
python -m pip install huggingface_hub
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  --include "nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000/*" --exclude "*.hexcodec"
model_dir=pretrained/Hexels/nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000
python -m hexels.render --checkpoint "$model_dir/checkpoints/ckpt_030000.pt" \
  --model_path "$model_dir" --out_dir outputs/pretrained_render/scan24 --split test --skip_path
```

Run from the code repository root. Supplied dataset paths are relative to
`data/`; change local paths if needed while preserving camera frames and image
scales. Rendering, evaluation and TSDF extraction need an NVIDIA GPU. The tested
environment is Linux, Python 3.11, PyTorch 2.5.1/CUDA 12.1, toolkit 12.3 and an
RTX A5000. See the code README for installation and hardware details.

Use the uncompressed FP32 checkpoint for the 3D table results. The example
download excludes optional `.hexcodec` companions; those have separate metrics
and do not replace the table checkpoint.

For the original direct-float paper evaluation, download the matching camera
settings and run the supplied evaluation job:

```bash
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  cameras/nvs/dtu/scan24/settings.json cameras/nvs/dtu/scan24/contract.json
python -m scripts.evaluate_nvs_campaign evaluate --job "$model_dir/evaluation_job.json"
```

It writes to a fresh `outputs/pretrained_eval/` location. Image names, sizes and
pixels must match the camera contract exactly, while camera matrices are compared
with its reference records within float32 roundoff, so GPUs other than the
paper's RTX A5000 pass the check. [The checkpoint guide](https://github.com/honeycomb-surfels/hexels/blob/main/docs/checkpoints.md)
also covers 2D decoding, surface mesh extraction with DTU Chamfer and T&T F1
evaluation, decoding and evaluating compressed scene files, and viewer export.
Training code remains available for fresh fits.

## Models and paper organization

This repository contains Hexels only. Adapted baseline weights are excluded.

| Collection | Models | Paper use |
|---|---:|---|
| `nvs/` | 204 | V1/V2/V3, fixed/grown, across 34 scenes; native comparisons and model-memory/primitive-budget references |
| `surface/` | 231 | 21 primary fixed-V3 MVS surface models plus the appearance, population, initialization and loss ablations, with reused models stored once |
| `images_2d/` | 90 | Flat/spatial RGB models on the 45-image Image-GS benchmark, including paper precision profiles and raw fields |
| `mvs/` | 21 packs | MVS initialization packs for the `-mvs_` surface models (15 DTU, 6 T&T), matching the SHA256 recorded in each checkpoint; needed only to retrain them |

Browse [catalog.json](https://huggingface.co/jhu-ccvl/Hexels/blob/main/catalog.json) for every model's roles,
expected metrics, file paths and checksums. Files serving several paper panels
are stored once. Equal model memory means learned scene state, not training
VRAM; a primitive ceiling need not equal a baseline's final population.

Names identify the task, dataset, scene/image, appearance, fixed/grown population,
SfM/MVS initialization, surface loss setting, seed and final iteration. V1 uses
spatial RGB, V2 adds linear view dependence, and V3 adds SH3.

Each model directory contains its checkpoint or raw fields, portable
configuration and metadata. NVS models also include training summaries,
per-view evaluation receipts and an evaluation-job config. Shared camera
contracts live under `cameras/`. `SHA256SUMS`, `FILES.json` and `release_config.json`
describe the released files. Model and optimizer tensor values are unchanged;
local paths and tracking-account fields are removed from embedded metadata as
well as sidecar configs. The catalog records source and published checksums and
sizes separately for rewritten files. Training checkpoints and compressed
deployment representations have separate quality/storage records.

The paper's `.hxm` image profiles are balanced for flat RGB and mixed14 for
spatial RGB. Uniform uint16 `.hxc` and raw FP32 `.pt` fields are additional
conditions. Scene `.hexcodec` files use the selected lossless or appearance-PQ
representation; their decoded quality may differ from the original checkpoint.

## Datasets

The four dataset downloads, COLMAP layouts, NVS/all-view differences and exact
image conventions are documented in the [code README](https://github.com/honeycomb-surfels/hexels#download-the-four-datasets)
and [dataset guide](https://github.com/honeycomb-surfels/hexels/blob/main/docs/datasets_2dgs.md):

- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/): both author archives; nine scenes.
- [DTU](https://roboimagedata.compute.dtu.dk/?page_id=36): 15 paper scans with the corresponding 2DGS preprocessing and official references.
- [Tanks and Temples](https://www.tanksandtemples.org/): ten NVS scenes and six all-view surface scenes, with separate input layouts.
- [Image-GS](https://github.com/NYU-ICL/image-gs#setup): the 45 RGB 2048-square benchmark images.

No raw dataset images or reference geometry are distributed here.

## Evaluation and limitations

The retained model tensors are distributed without numerical conversion. Embedded
metadata are made portable, which changes affected file checksums and sizes.
The documented evaluation command reproduces all 98 archived 3D test-view float
hashes and per-view metrics exactly across DTU scan24, Mip-NeRF 360 room and T&T
Barn, including camera-contract validation. Four packed-image samples match
their recorded paper-profile PSNR/SSIM/LPIPS. Full per-model expected values are retained in the catalog;
this publication check is not a fresh GPU evaluation of every model.

Saved-checkpoint evaluation and fresh training are separate reproducibility
conditions. CUDA backward reductions already vary in the historical binary,
so the same training seed can produce a different optimization outcome. Use the
published checkpoint and matching data/protocol to evaluate a paper model.
The original toy/artwork shape/color/overdraw illustration checkpoints were not
located; no substitute fits were uploaded. Hardware-specific speed and memory
figures are not device-independent guarantees.

## Abstract

Novel view synthesis and 3D reconstruction have advanced rapidly with radiance fields and splatting, yet their memory and computational demands limit their use in resource-constrained settings. We introduce Hexels: novel planar hexagonal surface elements with a first-order spatial color field for efficient 3D scene and 2D image representation. Hexels exploit locally planar regions with smoothly varying appearance using fewer, larger primitives. Their hexagonal support provides an area- and boundary-efficient domain, while the local color field captures appearance variation within each primitive. Across 34 scenes, Hexels improve PSNR over memory-matched 2DGS by 4.53–7.38 dB across datasets while requiring lower training VRAM. We further ablate Hexels by showing that per primitive, Hexels outperform 2DGS by upto 4.74 dB while using lower training VRAM and having 2.31× faster FPS. Even in unconstrained settings, Hexels offers complementary quality-memory trade-offs. For efficient 3D surface reconstruction, Hexels reduce error by 28.9% compared with memory-matched 2DGS across 15 DTU scans. On 6 Tanks and Temples surface scenes, Hexels achieves the highest mean F1 among compared methods (34.38%, versus 31.72% for 2DGS). We further also show that Hexels, as a general primitive, improves 2D image representation over Image-GS. Hexels push the Pareto frontier on the quality-efficiency trade-off of surface-aligned primitives.

## License and attribution

See [DATASET_TERMS.md](https://huggingface.co/jhu-ccvl/Hexels/blob/main/DATASET_TERMS.md) and
[LICENSE](https://huggingface.co/jhu-ccvl/Hexels/blob/main/LICENSE). The MIT source-code license does not relicense
dataset or depicted content. T&T's conflicting terms and the separate 2D
image-rights uncertainty remain disclosed. The original code notices and
upstream credits are retained in the linked code repositories.

## Citation

If our code was useful for your research, please consider citing and dropping a star.

```bibtex
@misc{kaushik2026honeycombsurfelscompacthexagonalprimitives,
      title={Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction},
      author={Prakhar Kaushik and Soumava Paul and Alan Yuille},
      year={2026},
      eprint={2610.37668},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.37668},
}
```
