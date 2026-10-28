# BBSplat

Upstream: [BBSplat](https://github.com/david-svitov/BBSplat), pinned to `cff8d6a7bce27c56b6938482dbdc72b882cd53bb`.

Runs with shared cameras and capacity controls. Model state includes the billboard textures, so equal billboard counts do not imply equal memory.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md), [LICENSE_BBSplat.txt](LICENSE_BBSplat.txt).
The upstream checkout and its copyright notices are fetched intact. The upstream source is used unmodified, so there is no patch; the shared cameras, capacity limits and evaluation come from `scripts/nvs_external.py` and `scripts/nvs_budgets.py` in the main repository. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods BBSplat` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.
