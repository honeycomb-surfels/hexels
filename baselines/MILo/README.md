# MILo

Upstream: [MILo](https://github.com/Anttwo/MILo), pinned to `0c38aa65f8b768513964f13246ef1ae54f001c70`.

Uses the upstream method with shared camera/evaluation adapters. Its Gaussian rendering state and full surface state are reported separately.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md).
The upstream checkout and its copyright notices are fetched intact. The upstream source is used unmodified, so there is no patch; the shared cameras, capacity limits and evaluation come from `scripts/nvs_external.py` and `scripts/nvs_budgets.py` in the main repository. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods MILo` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.
