# Baseline sources and adaptations

Baseline source stays in its original repository. Each folder here contains
its upstream link, exact revision, retained license notices and, where we changed
the source, our adaptation patch. You never apply a patch by hand: fetching a
method clones that revision, checks the patch's SHA256 and applies it. It does not
overwrite a different checkout or hide conflicting edits. BBSplat, MILo,
Image-GS and fused-SSIM are used unmodified.

```bash
python -m scripts.fetch_baselines --methods image-gs 2d-gaussian-splatting fused-ssim
# Example external method:
python -m scripts.fetch_baselines --methods BBSplat
python -m scripts.setup_nvs_baselines install --methods bbsplat --output outputs/bbsplat_install --conda conda
```

Set `CUDA_HOME` to the appropriate toolkit and run CUDA builds on a GPU machine.
`setup_nvs_baselines` creates a separate environment for each external method.
The upstream README remains the authority for additional system dependencies.

| Source folder | Installer method | Pinned environment used for the comparison |
|---|---|---|
| [Image-GS](image-gs/README.md) | Main environment | Python3.11 / PyTorch2.5.1 / CUDA12.1 |
| [2DGS](2d-gaussian-splatting/README.md) | Main environment | Python3.11 / PyTorch2.5.1 / CUDA12.1 |
| [3DGS](gaussian-splatting/README.md) | `3dgs` | Python3.11 / PyTorch2.4.0 / CUDA12.4 |
| [Convex Splatting](convex-splatting/README.md) | `3d_convex` | Python3.11 / PyTorch2.4.0 / CUDA12.4 |
| [Triangle Splatting](triangle-splatting/README.md) | `triangle` | Python3.11 / PyTorch2.4.0 / CUDA12.4 |
| [BBSplat](BBSplat/README.md) | `bbsplat` | Python3.8 / PyTorch2.0.0 / CUDA11.8 |
| [Nexels](nexels/README.md) | `nexels` | Python3.8 / PyTorch2.1.2 / CUDA11.8 |
| [MeshSplatting](mesh-splatting/README.md) | `mesh_splatting` | Python3.11 / PyTorch2.7.1 / CUDA12.6 |
| [MILo](MILo/README.md) | `milo` | Python3.9 / PyTorch2.3.1 / CUDA11.8 |
| [SAD](SAD/README.md) | Upstream installation | Use its pinned upstream instructions |

The external installer builds for `TORCH_CUDA_ARCH_LIST` when it is set and
otherwise for 8.6 (sm_86, the A5000 used in the experiments).
`TCNN_CUDA_ARCHITECTURES` and `CMAKE_CUDA_ARCHITECTURES` follow that list unless
set explicitly, and `<method>_build.json` records the values used. CUDA11.8
methods get their toolkit inside the isolated environment; CUDA12.x methods
(3dgs, 3d_convex, triangle, mesh_splatting) require `CUDA_HOME` to name a local
CUDA 12.x toolkit and stop before installing anything when it is unset.
PyTorch3D, nvdiffrast, tiny-cuda-nn,
effrdel and other method-specific dependencies keep their own upstream licenses.
Do not merge these differing environments into one environment.

## What the adaptations change

The common adapters fix camera ordering and held-out views and measure model
state consistently. The pinned external trainers seed their random generators
with 0 and expose no seed option, so the NVS preparers reject other seeds for
them; Hexels and 2DGS take `--seed`. Count controls limit the number of
primitives. Model-memory controls limit learned scene parameters; they do not
match peak training GPU memory. Adaptive baselines keep their growth and
pruning unless a condition is explicitly labeled fixed-population.

BBSplat textures and Nexels' appearance network count toward their model
budgets. MeshSplatting's initialization budget is separate from final model
bytes. See each folder for the method-specific details and source patch.
Official surface evaluation and alignment conventions are preserved.

## Image-fitting dependency

[Fused SSIM](fused-ssim/README.md) is pinned separately because the image-fitting
adapter imports it even when the default 3D loss uses the unfused implementation.
Fetch it, then run `python -m pip install --no-build-isolation ./fused-ssim`.

## Licenses

These folders do not relicense upstream implementations. Many methods inherit
the Gaussian-Splatting research/noncommercial terms, including when another
part of that repository uses MIT, BSD or Apache2.0. The original full license
texts are retained next to each patch. Fetching keeps upstream source headers
and submodule notices.
Cite the upstream paper for each method you use. See [the license inventory](../THIRD_PARTY.md).
