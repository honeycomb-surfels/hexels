# mesh-splatting

Upstream: [mesh-splatting](https://github.com/meshsplatting/mesh-splatting), pinned to `2a810a6c353215685307da3d4cc6ebd73b1c387b`.

Adds count and learned-state controls. The initial triangulation budget is reported separately from final vertices, faces and model bytes; initialization matching does not promise equal final memory.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md), [LICENSE_GS.md](LICENSE_GS.md).
The upstream checkout and its copyright notices are fetched intact. `adaptations.patch` contains our modifications for the Hexels experiments; modified files remain subject to their original terms. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods mesh-splatting` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.

Modified Python files carry an explicit Hexels modification notice, preserving upstream copyright headers.
