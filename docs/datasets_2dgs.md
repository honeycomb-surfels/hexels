# Datasets and evaluation

Data and generated outputs are not included in the release.

## Scenes and local paths

| Dataset | Scenes | Training root | Evaluation reference root |
|---|---|---|---|
| DTU | scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122 | `data/dtu` | `data/dtu_official` |
| Tanks and Temples | Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck | `data/tanks_and_temples` | `data/tnt_official` |
| Mip-NeRF360 | bicycle, flowers, garden, stump, treehill, room, counter, kitchen, bonsai | `data/mipnerf360` | Held-out source images |

The scene selection follows the pinned upstream
`2d-gaussian-splatting/scripts/{dtu_eval,tnt_eval,m360_eval}.py` scripts.
Church and Deep Blending are not part of the paper's evaluation.

## Sources and acquisition

Use the author-provided preprocessing below and preserve the COLMAP sparse models, image scales and geometry evaluation inputs. The held-out NVS cohort is listed separately in [experiment conventions](experiments.md).

The release uses four datasets: Mip-NeRF 360, DTU, Tanks and Temples, and the
Image-GS 2D image benchmark. Their download sources and expected folders are
listed below. Data are downloaded separately from the Hexel checkpoints.

- **Mip-NeRF360:** the [official project downloads](https://jonbarron.info/mipnerf360/)
  are `https://storage.googleapis.com/gresearch/refraw360/360_v2.zip` and
  `https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip`.
  Both are needed (flowers and treehill are in the extra archive). Extract their
  scene directories into `data/mipnerf360`, retaining original downsampled images
  and `sparse/0`. Use `images_4` outdoors and `images_2` indoors.
- **DTU training:** use `dtu.tar.gz` from the
  [2DGS preprocessing folder](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9),
  file ID `1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ`. Map its `DTU/scan*/` directories
  into `data/dtu/scan*/`. Preserve RGBA images, masks and `cameras.npz`.
- **DTU reference geometry:** the [official DTU evaluation data](https://roboimagedata.compute.dtu.dk/?page_id=36)
  provides `SampleSet.zip` and `Points.zip` at
  `https://roboimagedata2.compute.dtu.dk/data/MVS/`.
  Only the selected scans' `ObsMask/ObsMaskN_10.mat`, `ObsMask/PlaneN.mat` and
  `Points/stl/stlNNN_total.ply` are needed (45 files).
- **T&T training:** [GOF's author-provided `TNT_GOF.zip`](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/blob/main/TNT_GOF.zip),
  linked by [2DGS](https://github.com/hbb1/2d-gaussian-splatting), contains the
  normalized training scenes. Map only the six `TNT_GOF/TrainingSet/<Scene>/`
  directories into `data/tanks_and_temples/<Scene>/`. Preserve each
  `<Scene>_COLMAP_SfM.log` beside its images and sparse model.
- **T&T ground truth:** the [official evaluation download](https://github.com/isl-org/TanksAndTemples/blob/master/python_toolbox/evaluation/README.md)
  and the alternative link on the official download page both returned Google
  Drive quota-error HTML on 2026-09-14. The acquisition script instead uses the
  [VCR-GauS author's public data mirror](https://github.com/HLinChen/VCR-GauS#tanks-and-temples-dataset),
  at HF revision `7ab7a8798d7dadc4278ad634350236ff8b18074e`. It extracts only
  `<Scene>.ply`, `<Scene>.json`, `<Scene>_trans.txt`, and
  `<Scene>_COLMAP_SfM.log` into `data/tnt_official/<Scene>/` (24 files).
  These are the mirror's GT assets, not its predicted meshes or processed sparse
  clouds. Their byte identity to the quota-blocked official archive cannot be
  independently established here. GOF training trajectories remain separate
  from the reference trajectories and are aligned by the original evaluator.

Retain dataset licenses and attribution and consult the
[T&T terms](https://www.tanksandtemples.org/license/).
The current page contains a CC BY 4.0 statement alongside restrictive
language. This guide does not grant additional rights to the data.

### Ten-scene T&T held-out NVS inputs

The NVS cohort uses Barn, Caterpillar, Courthouse, Ignatius, Meetingroom, Truck,
Train, Francis, Horse and Lighthouse under `data/tnt_nvs/<Scene>/`. These are
separate from the six all-view geometry inputs above.

After downloading the GOF geometry pack, run:

```bash
python -m scripts.prepare_nvs_data --geometry-root data/tanks_and_temples
```

This downloads the five published half-resolution Train/Truck/Francis/Horse/
Lighthouse packs from the [pinned NerfBaselines data revision](https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/tree/6a638951a007c8b47eb8a0301d48b04d18e955cf/tanksandtemples),
checks their sizes and SHA256 values, and prepares the other five scenes from
the GOF sparse frames using the paper's half-size PIL LANCZOS images. It keeps
the original sparse coordinate frame. Use `images_2` with `resolution=1` for
NVS; do not halve those images again. An independently rerun COLMAP model can
use a different coordinate frame from a published checkpoint.

The helper needs the fetched `2d-gaussian-splatting` checkout for its COLMAP
reader and downloads about 720 MB to `data/downloads/tnt_nvs`. To prepare only the
five GOF-derived scenes without downloading the archive packs, pass `--scenes`
with no names: `python -m scripts.prepare_nvs_data --scenes --geometry-root
data/tanks_and_temples`; `--scenes Train Truck` downloads a subset. Each scene is
built in `data/tnt_nvs/.<Scene>.partial` and renamed only after validation; if a
run is interrupted, delete the reported `.<Scene>.partial` directory and rerun.
GOF frames are halved with PIL LANCZOS and re-saved as JPEG at Pillow's default
quality, which reproduces the paper's `images_2` files with the pinned Pillow
12.2.0. Truck's NVS inputs use the NerfBaselines frame, not GOF's.

### Image-GS 2D benchmark

Follow the [Image-GS setup page](https://github.com/NYU-ICL/image-gs#setup) and
download its [image and texture archive](https://1drv.ms/u/c/3a8968df8a027819/EeshjZJlMtdCmvvmESiN2pABM71EDaoLYmEwuOvecg0tAA?e=GybqBv).
Only the `images/` portion is needed for the paper's image experiments. Place
the 45 RGB images under `data/imagegs/images/`, retaining names such as
`anime-1_2k.png` and `photo-9_2k.png`. There are nine images in each of anime,
art, painting, photo and vector; each is 2048 by 2048 pixels. The checkpoint
catalog records the source-image hashes. The software's MIT license does not
establish a separate license for the image content; retain the source attribution.

The reference downloader reads verified HTTP byte ranges and checks ZIP CRCs
while extracting, so it does not download unused DTU scans or the entire 19.9 GB
VCR archive. Reruns skip complete-size files and replace incomplete `.part` files:

```bash
python -m scripts.download_geometry_references dtu
python -m scripts.download_geometry_references tanks_and_temples
```

## Evaluation conventions

- **DTU:** all views, resolution factor 2; mesh extraction at voxel 0.004,
  SDF truncation 0.016, depth truncation 3.0, largest component. The pinned
  `evaluate_single_scene.py` culls against masks and evaluates the official
  observation masks, planes and STL clouds. Its mesh-culling stage needs CUDA.
- **T&T:** all views, resolution factor 2. Barn/Caterpillar/Ignatius/Truck use
  voxel/SDF/depth values 0.004/0.016/3.0; Meetingroom/Courthouse use
  0.006/0.024/4.5. These are the two original extraction groups, not independent
  per-scene training presets. Both representations feed `fuse_post.ply` to the
  pinned `scripts/eval_tnt/run.py` for trajectory alignment, ICP, cropping and
  thresholded precision/recall/F1. Per-scene thresholds remain in its `config.py`.
  The local evaluator reports F1=0 for nonempty surfaces with zero precision and
  recall. Empty cropped surfaces or invalid distances fail explicitly; a zero
  score does not establish successful alignment or reconstruction quality.
- **Mip-NeRF360:** every eighth sorted camera is held out for NVS evaluation.
  The runner's existing paired image/efficiency evaluation remains unchanged.

T&T evaluation runs in `.envs/tnt-eval` (Python 3.8, Open3D 0.10), specified by
`evaluation_python` in the dataset config. The original code uses
`o3d.registration` and `RANSACConvergenceCriteria.max_validation`; aliasing the
namespace in Open3D 0.18 is not sufficient. An isolated old environment avoids
changing registration convergence semantics. Create it with
`conda env create -p .envs/tnt-eval -f environment-tnt-eval.yml`; do not use this
unsupported Python for services. The main Python 3.11 environment still performs
training and TSDF fusion. Preparing any manifest that contains T&T geometry
evaluation fails immediately if this interpreter is missing; create it or point
`evaluation_python` at an existing interpreter. `scripts.run_reproduction prepare`
likewise requires GPU COLMAP for the Mip-NeRF 360 MVS tier (`.envs/colmap37`, from
`environment-colmap.yml`, or pass `--colmap`).

The paper's surface checkpoints use the all-view recipe in the main README
(`scripts.train_scene --all-views --surface`, MVS packs from the model repository)
and median-depth fusion, the default of `scripts.extract_hexels_dtu_mesh`.

### Older staged geometry suite

`scripts.run_reproduction` and `scripts.run_paper_3d_benchmark` run an older
staged protocol that reuses the staged Hexel optimizer and budget controls. It is
not a claim of bit-identical reproduction of published training results. Its
Hexel fusion uses expected surface depth (passed explicitly); 2DGS uses its
median-depth option.
The pinned T&T script prints distortion weights that it does not pass to its
actual training commands; those commands use the default `lambda_dist=0`.
The runner follows the executed commands, without inventing scene-specific
distortion weights or fixing unrelated upstream script issues.

Use a fresh output root, such as `benchmark_outputs/2dgs_suite_seed0`.
There are 240 training jobs and 72 separate Mip NVS evaluations. Geometry jobs
include mesh extraction and their dataset evaluator. Summaries produce
`summary_dtu.json` (surface distances), `summary_tnt.json` (precision, recall and
F1 percentages, macro-averaged across selected scenes), and `summary_nvs.json`.
Missing GT/trajectories block jobs; missing results block aggregation rather
than producing partial dataset averages. Do not reuse held-out T&T checkpoints
as all-view surface-reconstruction runs.
