# triangle-splatting

Upstream: [triangle-splatting](https://github.com/trianglesplatting/triangle-splatting), pinned to `685ff270a8273afc3a2b094ae3148dd0b98e31e9`.

Adds shared-view training and primitive/model-state ceilings while retaining native triangle updates.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md), [LICENSE_GS.md](LICENSE_GS.md).
The upstream checkout and its copyright notices are fetched intact. `adaptations.patch` contains our modifications for the Hexels experiments; modified files remain subject to their original terms. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods triangle-splatting` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.

Modified Python files carry an explicit Hexels modification notice, preserving upstream copyright headers.
