# convex-splatting

Upstream: [convex-splatting](https://github.com/convexsplatting/convex-splatting), pinned to `27f83f8b6fbeecef98530e6b56dbd76b73760685`.

Adds shared camera conventions and capacity controls around native convex primitives.

## License and attribution

Includes the Gaussian-Splatting research/noncommercial license. This dependency is not covered by our MIT license.

Retained notices: [LICENSE.md](LICENSE.md), [LICENSE_GS.md](LICENSE_GS.md).
The upstream checkout and its copyright notices are fetched intact. `adaptations.patch` contains our modifications for the Hexels experiments; modified files remain subject to their original terms. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods convex-splatting` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.

Modified Python files carry an explicit Hexels modification notice, preserving upstream copyright headers.
