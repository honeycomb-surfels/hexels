# App icon source

The canonical icon is the original bee artwork in
[`assets/app-icon.svg`](../../../assets/app-icon.svg); the opaque 1024 px export is
[`assets/app-icon.png`](../../../assets/app-icon.png). The app catalog contains an
unchanged copy of that PNG. A separate legacy icon generator is intentionally
not included.

From the parent repository root, regenerate with the asset pipeline and copy the
canonical result into the app:

```bash
python3 -m pip install -r hexel_edge/assets/requirements.txt
python3 hexel_edge/assets/render_assets.py
cp hexel_edge/assets/app-icon.png \
  hexel_edge/apple/iOS/HexelsEdge/Assets.xcassets/AppIcon.appiconset/AppIcon.png
```

The rendering helper also requires native Cairo. See
[`assets/provenance.json`](../../../assets/provenance.json) for provenance and
[`assets/requirements.txt`](../../../assets/requirements.txt) for dependencies.
