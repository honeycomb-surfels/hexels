# image-gs

Upstream: [image-gs](https://github.com/NYU-ICL/image-gs), pinned to `03088368d42684fb54225c981cfd94b58cc0393a`.

Uses the original image fitting model and initialization. The Hexel adapter replaces the footprint and optionally adds spatial color gradients and learned softness. Flat and rich models report their actual saved file sizes.

## License and attribution

See the retained upstream license files and the licenses in upstream submodules. Our root MIT license does not replace them.

Retained notices: [LICENSE](LICENSE).
The upstream checkout and its copyright notices are fetched intact. The upstream source is used unmodified, so there is no patch; the Hexel image adapter in `scripts/image_compression_*.py` imports its model, initialization and utilities. Nested CUDA packages retain their own licenses.

Fetch with `python -m scripts.fetch_baselines --methods image-gs` from the main repository. Follow the upstream README for method-specific installation, or use our isolated environment installer where provided.
