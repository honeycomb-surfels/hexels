# gaussian-splatting

Upstream: [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), pinned to `54c035f7834b564019656c3e3fcc3646292f727d`.

Adds camera-consistent training, primitive ceilings, and measured model-state budgets while retaining the native population updates.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md).
The upstream checkout and its copyright notices are fetched intact. `adaptations.patch` contains our modifications for the Hexels experiments; modified files remain subject to their original terms. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods gaussian-splatting` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.

Modified Python files carry an explicit Hexels modification notice, preserving upstream copyright headers.
