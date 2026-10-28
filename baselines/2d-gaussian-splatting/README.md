# 2d-gaussian-splatting

Upstream: [2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting), pinned to `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`.

Adds primitive limits and optional fixed-population controls. Memory controls turn a Hexel learned-state budget into a primitive ceiling. Native growth and pruning stay active in the adaptive comparisons. This matches model state, not training GPU memory.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md).
The upstream checkout and its copyright notices are fetched intact. `adaptations.patch` contains our modifications for the Hexels experiments; modified files remain subject to their original terms. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods 2d-gaussian-splatting` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.

Modified Python files carry an explicit Hexels modification notice, preserving upstream copyright headers.
