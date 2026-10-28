# Release configurations

These are production templates. They contain dataset paths, method settings and reproducible experiment definitions, without development run IDs, cleanup choices or machine-specific launch records.

| File | Purpose |
|---|---|
| `train.json` | Shared input paths, dataset conventions and the fixed Hexel training defaults. These five `hexels` settings are read from here by every Hexel preparer (staged suite, NVS campaign, `scripts.train_scene`); a config passed with `--config` can override them |
| `growth.json` | The published mixed growth/pruning settings; fixed-population training leaves growth disabled |
| `nvs.json` | Fixed/adaptive V1/V2/V3 and native baselines on the 34-scene paper cohort |
| `capacity.json` | Primitive/model-state controls, including the separately labeled MeshSplatting initialization-memory condition |
| `surface.json` | All-view normal/distortion and expected/median-depth ablations using your own completed geometry root |

The NVS and capacity templates contain 476 and 850 endpoints respectively, totaling 1326. The counts are checked before preparing a complete cohort. The staged runners (`scripts.run_reproduction`, `scripts.run_paper_3d_benchmark`) accept `--scenes`. For the NVS campaign and the capacity sweep, copy the template and adjust the declared matrix and its expected count together; `prepare_nvs_budget_sweep --scenes` applies only with `--smoke`. These files never submit jobs when read.

Set `surface.json`'s `baseline_root` to your completed geometry benchmark. For shared MVS initialization, set `initial_geometry_root` to a directory containing `<scene>/initial_geometry.pt`, such as `pretrained/Hexels/mvs/dtu` after downloading the paper packs from the model repository; leave it null for SfM initialization. This template runs on the older staged protocol; the paper's surface checkpoints come from `scripts.train_scene --all-views --surface`. Surface evaluation requires separate all-view checkpoints and ground truth.

Copy a template for a new experiment and use a fresh output directory. Manifests inside an output directory remain immutable.
