<p align="center"><a href="https://honeycomb-surfels.github.io/"><img src="docs/assets/hexels-logo.gif" width="120" alt="Honeycomb-Surfels animated bee and honeycomb logo"></a></p>

# Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction

<p align="center"><a href="https://toshi2k2.github.io/">Prakhar Kaushik*</a> · <a href="https://mvp18.github.io/">Soumava Paul*</a> · <a href="https://www.cs.jhu.edu/~ayuille1/">Alan Yuille</a><br>CCVL, Johns Hopkins University<br><sub>* Equal contribution</sub></p>

<p align="center"><a href="https://honeycomb-surfels.github.io/">Project page</a> · <a href="https://honeycomb-surfels.github.io/h1.pdf">Paper</a> · <a href="https://arxiv.org/abs/2610.37668">arXiv</a> · <a href="https://github.com/honeycomb-surfels/hexels">GitHub code</a> · <a href="https://huggingface.co/jhu-ccvl/Hexels">Paper checkpoints</a></p>

Run the trainer from its source checkout using the editable (`-e`) install.
The kernel can also be installed as an ordinary wheel; the viewer needs the
`hexels` package installed alongside it.

The [kernel](https://github.com/honeycomb-surfels/hexel-kernel) and
[viewer](https://github.com/honeycomb-surfels/hexel-viewer) are Git submodules
under `submodules/`. Dataset images,
checkpoints, environments and generated experiment outputs are not bundled in
the source repositories. Paper checkpoints are distributed separately on Hugging Face below.

## Requirements

The supported reproduction environment is Linux x86-64, Python 3.11,
PyTorch 2.5.1 with CUDA 12.1, and a locally installed CUDA 12.1 or later in the 12.x family **toolkit**
(including `nvcc`) with a compatible C++17 compiler and NVIDIA driver. Builds are
tested with toolkit **12.3** and an RTX A5000, compute capability **8.6**, with
24 GB of GPU memory. Use this combination for numerical comparisons.
A CUDA runtime installed with PyTorch alone cannot compile the extensions.
See [PyTorch's version-specific installation commands](https://pytorch.org/get-started/previous-versions/)
and [NVIDIA's toolkit and driver requirements](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html).

One NVIDIA GPU is required for training, CUDA rendering and TSDF extraction.
There is no measured universal minimum VRAM: image size, primitive count and
requested diagnostics change memory use. 24 GB is the tested GPU capacity, not
a claim that every scene needs it.
Other GPU architectures require setting `TORCH_CUDA_ARCH_LIST` and rebuilding;
Windows, older GPUs and smaller-memory devices have not been validated.

The exported browser viewer needs only a WebGL2-capable browser and a local or
static HTTP server. Exporting a viewer bundle from a checkpoint can run on CPU.
External baselines have separate PyTorch/CUDA requirements; follow their
[installation notes](baselines/README.md) rather than installing every method
into the Hexels environment.

## Install

```bash
git clone --recurse-submodules https://github.com/honeycomb-surfels/hexels.git
cd hexels
CONDA_CHANNEL_PRIORITY=flexible conda env create -p .envs/hexels311 -f environment.yml
conda activate ./.envs/hexels311
# For an existing non-recursive checkout:
git submodule update --init --recursive
python -m pip install -e .
python -m pip install -e submodules/hexel-viewer
python -m scripts.fetch_baselines --methods image-gs 2d-gaussian-splatting fused-ssim

# Run the following builds on a GPU machine, inside its GPU allocation if using Slurm.
export CUDA_HOME=/path/to/cuda-12.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
python -m pip install --no-build-isolation submodules/hexel-kernel
python -m pip install --no-build-isolation 2d-gaussian-splatting/submodules/diff-surfel-rasterization
python -m pip install --no-build-isolation 2d-gaussian-splatting/submodules/simple-knn
python -m pip install --no-build-isolation ./fused-ssim
```

Two separate environments are only needed for specific steps, and the benchmark
runners expect them at these paths. The pinned Tanks and Temples F1 evaluator needs
Open3D 0.10, and building MVS initialization packs needs GPU COLMAP:

```bash
conda env create -p .envs/tnt-eval -f environment-tnt-eval.yml
CONDA_OVERRIDE_CUDA=12.1 CONDA_CHANNEL_PRIORITY=flexible conda env create -p .envs/colmap37 -f environment-colmap.yml
```

The [pinned fused-SSIM dependency](baselines/fused-ssim/README.md) is needed
for image fitting and optional for the default 3D loss.
Do not install the unrelated PyPI `gsplat` package over `hexel-kernel`: the
release intentionally keeps the `gsplat` Python import name for compatibility.

[The dataset guide](docs/datasets_2dgs.md) describes COLMAP layouts, scene lists,
image scales and reference geometry. Image-GS inputs are RGB images; the paper
uses 2048 by 2048 pixels. Use a new output directory for every changed
configuration, source version or seed.

## Download the four datasets

Download the data separately from the checkpoints and preserve the published
COLMAP coordinate frames and image scales.

| Dataset | Download source | Expected location |
|---|---|---|
| Mip-NeRF 360 | [Official project](https://jonbarron.info/mipnerf360/), [part 1](https://storage.googleapis.com/gresearch/refraw360/360_v2.zip), [part 2](https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip) | `data/mipnerf360/<scene>/{sparse/0,images_2,images_4}`; both archives are needed for all nine scenes |
| DTU | [2DGS training preprocessing](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), [official evaluation data](https://roboimagedata.compute.dtu.dk/?page_id=36) | Map the archive's `DTU/scan*/` folders to `data/dtu/scan*/`; keep images, masks and `cameras.npz`. Reference files go under `data/dtu_official/`. |
| Tanks and Temples | [GOF COLMAP pack](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/blob/main/TNT_GOF.zip), [pinned half-resolution NVS packs](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/6a638951a007c8b47eb8a0301d48b04d18e955cf/tanksandtemples), [official benchmark](https://www.tanksandtemples.org/) | Six GOF `TrainingSet/<Scene>/` folders go under `data/tanks_and_temples/`; the ten-scene NVS cohort uses `data/tnt_nvs/`. |
| Image-GS image benchmark | [Author setup instructions](https://github.com/NYU-ICL/image-gs#setup), [image/texture download](https://1drv.ms/u/c/3a8968df8a027819/EeshjZJlMtdCmvvmESiN2pABM71EDaoLYmEwuOvecg0tAA?e=GybqBv) | Put the 45 RGB images from `images/` in `data/imagegs/images/`, retaining names such as `anime-1_2k.png`. Images are 2048 by 2048. Textures are not needed. |

After obtaining the GOF geometry pack, prepare the complete ten-scene T&T NVS
inputs and fetch the official surface-evaluation references with:

```bash
python -m scripts.prepare_nvs_data --geometry-root data/tanks_and_temples
python -m scripts.download_geometry_references dtu
python -m scripts.download_geometry_references tanks_and_temples
```

The NVS helper verifies pinned downloads and prepares the five additional
half-size scenes from the GOF sparse frames. Use Mip `images_4` outdoors and
`images_2` indoors, T&T NVS `images_2` without further downsampling, and DTU
resolution factor 2. Official surface evaluation uses separate all-view models.
See [the dataset guide](docs/datasets_2dgs.md) for scene lists, archive layouts,
reference provenance and dataset terms.

## Paper checkpoints

[Download the Hexel paper models on Hugging Face](https://huggingface.co/jhu-ccvl/Hexels) and consult its
[catalog](https://huggingface.co/jhu-ccvl/Hexels/blob/main/catalog.json) for per-model names, comparisons and checksums.
The collection contains 204 NVS models, 231 surface models and 90 image models,
including paper ablations. It contains Hexels only; baseline weights are excluded.
Shared memory/primitive comparison references are stored once and labeled in
the catalog. Original checkpoints and compressed deployment files are distinct.

After installing the code and corresponding dataset, download one model:

```bash
python -m pip install huggingface_hub
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  --include "nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000/*" --exclude "*.hexcodec"
model_dir=pretrained/Hexels/nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000
python -m hexels.render --checkpoint "$model_dir/checkpoints/ckpt_030000.pt" \
  --model_path "$model_dir" --out_dir outputs/pretrained_render/scan24 --split test --skip_path
```

The [checkpoint guide](docs/checkpoints.md) includes direct-float paper evaluation,
2D decoding, surface extraction and format details. `pretrained/` is ignored by
Git. Dataset images and reference geometry are downloaded separately. Matching
configs, expected metrics and SHA256 checksums accompany the models.
The commands above use the uncompressed FP32 checkpoint for the 3D table
results. Compressed files are optional companions with separate metric records.

Fresh training remains available below, but the same seed does not reproduce the
paper's 3D numbers exactly: the CUDA backward pass accumulates gradients with
atomic additions, so their summation order, and with it the optimization
trajectory, varies between runs. Evaluating the published checkpoints reproduces
the paper's metrics.

## Fit and decode a 2D image

```bash
python -m scripts.fit_image --image data/imagegs/images/anime-1_2k.png \
  --output outputs/anime_rich --variant rich
python -m scripts.decode_image outputs/anime_rich/step_005100_mixed14.hxm outputs/anime.png
```

The default is the paper's 5100-step, 20K-to-10K pruning recipe with L1 plus
SSIM and seed 123. `--variant flat` removes learned spatial color gradients
and uses fixed softness. To run native Image-GS, use `--variant imagegs
--allocation grow --steps 10000 --loss l1_ssim`.

Flat and rich fits write the paper's packed per-field-precision file,
`step_005100_mixed14.hxm` for `--variant rich` and `step_005100_balanced.hxm`
for `--variant flat`, next to the uniform uint16 `step_005100_16bit.hxc` and the
raw FP32 `render_parameters_005100.pt`. `metrics.json` records its bytes, bits
per pixel and SHA256 under `mixed_precision`. Pruning compiles a small CUDA
scoring extension before training, so `nvcc` must be available when fitting; its
build time is reported as `kernel_setup_seconds`, separately from training time.

Ablations use `--allocation grow|fixed|prune` (pruning is Hexel-only; `--variant
imagegs` accepts `grow` or `fixed`), `--initial-count`, `--count`,
`--steps`, `--loss l1|l1_ssim|mse` and `--seed`. The `scripts/image_compression_*`
modules contain codecs, per-field precision and shape/allocation controls.
Model files include their decoding parameters; file bytes and GPU memory are
reported separately. `scripts.evaluate_image --model ... --image ... --output ...`
evaluates stored `.hxm`/`.hxc` files or raw `render_parameters_*.pt` FP32 fields. The selected
input condition is recorded in its metric JSON.

## Train and render a 3D scene

```bash
python -m scripts.train_scene --source data/dtu/scan24 --dataset dtu \
  --variant v3 --output outputs/scan24_v3
python -m hexels.render --checkpoint outputs/scan24_v3/checkpoints/ckpt_030000.pt \
  --model_path outputs/scan24_v3 --split test --skip_path
python -m hexels.metrics --model_path outputs/scan24_v3 --iteration 30000 --split test
```

V1 learns spatial color, V2 adds linear view-dependent color, and V3 uses SH3
view dependence. Their active learned-state sizes are 80, 116 and 260 bytes per
primitive. Training checkpoints also contain optimizer and inactive fields.
`--growth` selects the paper's mixed growth and coordinated weak pruning;
without it, the population stays fixed. The shared 30K recipe decays learning
rates at 15001, 20001 and 25001. `--prepare` prints the command on CPU.
`python -m hexels.train` on its own uses generic defaults, not the paper recipe.

`hexels.metrics` scores the saved 8-bit PNG renders, which is convenient but is not
the paper protocol. To score a trained model with the paper's direct-float metrics
and camera contract, reuse the published evaluation job for the same scene and
variant ([checkpoint guide](docs/checkpoints.md)) with your own model directory:

```bash
python -m scripts.evaluate_nvs_campaign evaluate \
  --job pretrained/Hexels/nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000/evaluation_job.json \
  --model-path outputs/scan24_v3 --result outputs/scan24_v3_eval/metrics.json
```

Additional method flags can follow `--`, for example
`-- --save_every 5000`. Inspect `python -m hexels.train --help` for
initialization, appearance, loss, growth/pruning and continuation ablations.
Experimental flags outside the paper recipes are not additional validated
quality claims.

## Experiment suites and baselines

[Experiment conventions](docs/experiments.md) distinguish the NVS and all-view protocols.
The [release configs](configs/README.md) contain production settings and dataset paths.
`configs/nvs.json` describes the common held-out-view experiments.
`python -m scripts.prepare_nvs_smoke --help` and
`python -m scripts.run_nvs_campaign --help` prepare the runtime gates and
production cohorts. A campaign runs `run_nvs_campaign prepare`, then `capture`,
which records the source, environment and kernel provenance every job verifies,
then `submit`. Slurm resources are set with `--partition`, `--qos` and `--gres`;
the [experiment conventions](docs/experiments.md#slurm-and-local-runs) show how to
run prepared manifests without Slurm. `scripts.prepare_nvs_budget_sweep` prepares count and
learned-state controls from completed Hexel anchors. Native/adaptive baseline
updates remain enabled for the primary constrained comparisons.

The older fixed-budget, staged protocol is available through
`python -m scripts.run_reproduction --help` with
`configs/train.json`. `scripts.run_surface_ablation` exposes geometry-loss and extraction ablations;
set its protocol's `baseline_root` to your own completed geometry experiment.
The single-scene wrappers expose the same growth, appearance and loss flags for
new ablation grids.
Manifests freeze inputs and commands. Missing scenes block aggregation;
partial results are never silently averaged into a dataset result.

[Each baseline folder](baselines/README.md) links the original implementation,
retains its license, pins its revision, and explains our changes in plain
language. Equal model memory means learned scene parameters. It does not mean
equal optimizer memory, checkpoint bytes or training VRAM. A primitive ceiling
also does not force adaptive methods to end with exactly the same population.

## Surface and wireframe extraction

The paper's surface models train on every registered view with normal-consistency
and ray-distortion losses. Most start from an MVS initialization pack; the 21 packs
used in the paper are on the Hub under `mvs/<dataset>/<scene>/`:

```bash
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels --include "mvs/dtu/scan24/*"
python -m scripts.train_scene --source data/dtu/scan24 --dataset dtu --variant v3 \
  --all-views --surface --init-geometry pretrained/Hexels/mvs/dtu/scan24/initial_geometry.pt \
  --output outputs/scan24_v3_surface
python -m scripts.extract_hexels_dtu_mesh \
  --checkpoint outputs/scan24_v3_surface/checkpoints/ckpt_030000.pt
```

`--surface` adds the paper's 2DGS-style normal consistency on median depth (weight
0.05 from step 7000) and ray distortion (weight 1000 from step 3000 with a
1000-step ramp). Omit `--init-geometry` for SfM initialization. For Tanks and
Temples, `--all-views` reads the full-size GOF `images/` at resolution 2.
`scripts.prepare_mvs_geometry` rebuilds a pack from COLMAP dense stereo in the
`.envs/colmap37` environment on a GPU; the paper packs used `--all-views --recipe
equal_sparse --max-image-size 777`, and a rebuilt pack can differ from the
published one.

The extractor writes `train/ours_30000/fuse.ply` and the filtered `fuse_post.ply`
under the model directory, or under `--output_model_path`. It defaults to median
depth and the DTU TSDF settings (voxel 0.004, SDF truncation 0.016, depth
truncation 3.0); T&T Courthouse and Meetingroom use `--voxel_size 0.006 --sdf_trunc
0.024 --depth_trunc 4.5`. Expected depth (`--depth_mode expected`) is the older
staged protocol's choice. The [checkpoint guide](docs/checkpoints.md) gives the DTU
Chamfer and T&T F1 evaluation commands.

```bash
python -m scripts.extract_hexels_wireframe --help
python -m scripts.run_extraction_benchmark --help
```

Analytic six-edge Hexel exports and contact-patch extraction are also included.
Use all-view checkpoints for official surface evaluation; a held-out NVS
checkpoint is a different experiment. Analytic wireframes and contact patches are
not guaranteed watertight surfaces.

## Viewers

```bash
python -m hexel_viewer.viewer_model outputs/scan24_v3/checkpoints/ckpt_030000.pt \
  outputs/scan24_site --site
python -m http.server 8081 --bind 127.0.0.1 --directory outputs/scan24_site
# Open http://127.0.0.1:8081/?model=model.hexview

# GPU/server viewer; run it on an NVIDIA GPU machine and forward its loopback port over SSH.
python -m hexel_viewer.viewer --help
python -m hexel_viewer.live_viewer --help
```

The exported page includes the dataset's camera views when the checkpoint's
`source_path` resolves from the current directory (run from the repository root)
or when `--source-path data/<dataset>/<scene>` is given. The whole exported site
can be copied to another machine. Rendering then runs in that machine's browser. The CUDA viewer supports pretrained checkpoints;
live viewing is an opt-in trainer wrapper. Connected live viewing adds work
and should not be used for paper training-time measurements.

## Abstract

Novel view synthesis and 3D reconstruction have advanced rapidly with radiance fields and splatting, yet their memory and computational demands limit their use in resource-constrained settings. We introduce Hexels: novel planar hexagonal surface elements with a first-order spatial color field for efficient 3D scene and 2D image representation. Hexels exploit locally planar regions with smoothly varying appearance using fewer, larger primitives. Their hexagonal support provides an area- and boundary-efficient domain, while the local color field captures appearance variation within each primitive. Across 34 scenes, Hexels improve PSNR over memory-matched 2DGS by 4.53–7.38 dB across datasets while requiring lower training VRAM. We further ablate Hexels by showing that per primitive, Hexels outperform 2DGS by upto 4.74 dB while using lower training VRAM and having 2.31× faster FPS. Even in unconstrained settings, Hexels offers complementary quality-memory trade-offs. For efficient 3D surface reconstruction, Hexels reduce error by 28.9% compared with memory-matched 2DGS across 15 DTU scans. On 6 Tanks and Temples surface scenes, Hexels achieves the highest mean F1 among compared methods (34.38%, versus 31.72% for 2DGS). We further also show that Hexels, as a general primitive, improves 2D image representation over Image-GS. Hexels push the Pareto frontier on the quality-efficiency trade-off of surface-aligned primitives.

## Licenses

Original Hexels code uses [MIT](LICENSE). Third-party sources and adaptations
retain their own licenses, including research/noncommercial restrictions where
present. See [third-party attribution](THIRD_PARTY.md); the root MIT license
does not relicense those dependencies. Cite the corresponding upstream papers
when using their methods.

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
