# Download and evaluate paper checkpoints

Models are hosted at [https://huggingface.co/jhu-ccvl/Hexels](https://huggingface.co/jhu-ccvl/Hexels). The code is installed from the main
repository together with the kernel and viewer submodules. Run the commands
below from the code repository root after following its installation steps.

## Names and comparison roles

Paths identify the task, dataset, scene/image, appearance, fixed/grown population,
SfM/MVS initialization, loss setting, seed and final step. For example:

```text
nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000/checkpoints/ckpt_030000.pt
surface/dtu/scan24/hexels_v3-sh3_fixed-mvs_normal-and-distortion_seed0_step030000/checkpoints/ckpt_030000.pt
images_2d/anime-1_2k/hexels_spatial-rgb_pruned-20000-to-10000_seed123_step005100/hexels_spatial-rgb_mixed14.hxm
```

V1 is spatial RGB, V2 adds linear view dependence, and V3 adds SH3. The same Hexel
checkpoint can appear in native, model-memory and primitive-budget comparisons;
its catalog roles record that use without duplicating its weights. Baseline
checkpoints are not distributed here. The original toy/artwork illustration
states were not located, and no substitute fits were included.

`expected_metrics_artifact` identifies the exact file evaluated for each metric
record. It points to the uncompressed checkpoint for 3D models. Optional
compressed representations have their own expected values; 2D precision-profile
metrics identify the corresponding packed image file.

## NVS rendering and direct-float evaluation

```bash
python -m pip install huggingface_hub
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  --include "nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000/*" --exclude "*.hexcodec"
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  cameras/nvs/dtu/scan24/settings.json cameras/nvs/dtu/scan24/contract.json
model_dir=pretrained/Hexels/nvs/dtu/scan24/hexels_v3-sh3_fixed-sfm_seed0_step030000
python -m hexels.render --checkpoint "$model_dir/checkpoints/ckpt_030000.pt" \
  --model_path "$model_dir" --out_dir outputs/pretrained_render/scan24 --split test --skip_path
python -m scripts.evaluate_nvs_campaign evaluate --job "$model_dir/evaluation_job.json"
```

The download above selects the uncompressed FP32 checkpoint for the 3D table
results and excludes compressed companions. The last command uses the saved
camera contract and direct-float metric protocol
with VGG-LPIPS. It writes to a fresh `outputs/pretrained_eval/` subfolder and
refuses to overwrite an existing result. PNG-render metrics are a separate
measurement. Use an NVIDIA GPU for rendering/evaluation. Image names, sizes and
pixels must match the contract exactly; camera matrices are compared with the
contract's reference camera records within float32 roundoff, so GPUs other than
the paper's RTX A5000 pass the check. Dataset paths in
the supplied configs are relative to the code repository's `data/` directory;
edit the local paths if using another location while retaining the same images,
camera frame and split. The supplied evaluation job is for the original FP32
checkpoint; compressed representations have separate expected values.

## Image models

```bash
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  --include "images_2d/anime-1_2k/hexels_spatial-rgb_pruned-20000-to-10000_seed123_step005100/*"
image_dir=pretrained/Hexels/images_2d/anime-1_2k/hexels_spatial-rgb_pruned-20000-to-10000_seed123_step005100
mkdir -p outputs
python -m scripts.decode_image "$image_dir/hexels_spatial-rgb_mixed14.hxm" outputs/anime.png
python -m scripts.evaluate_image --model "$image_dir/hexels_spatial-rgb_mixed14.hxm" \
  --image data/imagegs/images/anime-1_2k.png --output outputs/anime_metrics.json
```

Create `outputs/` before writing an image. The `.hxm` file is the paper's packed
profile: balanced for flat RGB and mixed14 for spatial RGB. Uniform uint16 `.hxc`
and raw FP32 `.pt` fields are also provided and carry different quality/storage
conditions. The public decoder detects all three formats. Image evaluation uses
AlexNet-LPIPS, unlike the VGG-LPIPS used by the 3D NVS suite. A fresh
`scripts.fit_image` run writes the same profile as `step_005100_<profile>.hxm`, and
`scripts.image_compression_precision.export_paper_profile` regenerates each
published `.hxm` byte for byte from its `hexels_raw-fp32.pt`.

## Surface checkpoints

Use `surface/` all-view checkpoints for official mesh evaluation; NVS checkpoints
are held-out-view models and are not substitutes. Each `metadata.json` records the
expected Chamfer distance or F1 and its TSDF settings. Extract and score one DTU
model from the code repository root (the DTU mesh-culling step needs CUDA):

```bash
hf download jhu-ccvl/Hexels --local-dir pretrained/Hexels \
  --include "surface/dtu/scan24/hexels_v3-sh3_fixed-mvs_normal-and-distortion_seed0_step030000/*"
model_dir=pretrained/Hexels/surface/dtu/scan24/hexels_v3-sh3_fixed-mvs_normal-and-distortion_seed0_step030000
python -m scripts.extract_hexels_dtu_mesh --checkpoint "$model_dir/checkpoints/ckpt_030000.pt" \
  --output_model_path outputs/surface/scan24
python 2d-gaussian-splatting/scripts/eval_dtu/evaluate_single_scene.py \
  --input_mesh outputs/surface/scan24/train/ours_30000/fuse_post.ply --scan_id 24 \
  --output_dir outputs/surface/scan24/dtu_eval --mask_dir data/dtu --DTU data/dtu_official
```

The extractor defaults to the paper's median depth and the DTU TSDF settings. For
Tanks and Temples Courthouse and Meetingroom add `--voxel_size 0.006 --sdf_trunc
0.024 --depth_trunc 4.5`, and score every T&T mesh with the pinned evaluator in the
separate `.envs/tnt-eval` environment:

```bash
.envs/tnt-eval/bin/python 2d-gaussian-splatting/scripts/eval_tnt/run.py \
  --dataset-dir data/tnt_official/Barn --traj-path data/tanks_and_temples/Barn/Barn_COLMAP_SfM.log \
  --ply-path outputs/surface/Barn/train/ours_30000/fuse_post.ply --out-dir outputs/surface/Barn/tnt_eval
```

The MVS packs behind the `-mvs_` models are under `mvs/<dataset>/<scene>/initial_geometry.pt`
and match the SHA256 recorded in each checkpoint. They are needed only for
retraining (`scripts.train_scene --all-views --surface --init-geometry ...`), not
for extraction or evaluation. `python -m scripts.extract_hexels_wireframe --help`
covers analytic wireframes.

## Compressed scene files and viewers

The `.hexcodec` files contain the selected lossless or appearance-PQ deployment
representation, with expected metrics under `representations` in `metadata.json`.
Decode one and evaluate it with the model's own job, cameras and protocol:

```bash
python -m scripts.decode_hexels_deployment --archive "$model_dir/hexels_sh_pq1024.hexcodec" \
  --output outputs/decoded/scan24_v3_pq1024
python -m scripts.evaluate_nvs_campaign evaluate --job "$model_dir/evaluation_job.json" \
  --model-path outputs/decoded/scan24_v3_pq1024 --result outputs/pretrained_eval/scan24_v3_pq1024/metrics.json
```

Here `model_dir` is the NVS model directory from the first section, downloaded
without `--exclude "*.hexcodec"`. Archive configs preserve training options with
portable dataset/output paths and no tracking account. The viewer exporter
accepts the ordinary checkpoint path; pass `--source-path data/<dataset>/<scene>`
(or run from the code repository root) so the exported page includes the
dataset's camera views.

## Coverage and terms

The catalog includes 204 NVS, 231 surface and 90 image identities. All underlying
model and optimizer tensor values are unchanged. Sidecar and embedded metadata
use portable paths without private tracking accounts. Affected file checksums
and sizes change; the catalog retains source checksums and sizes separately.
[Model terms](https://huggingface.co/jhu-ccvl/Hexels/blob/main/DATASET_TERMS.md) retain dataset/content
qualifications. [THIRD_PARTY.md](../THIRD_PARTY.md) lists the code notices.
