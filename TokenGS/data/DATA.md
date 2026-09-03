# Data Preparation

Defaults in [`tokengs/data/registry.py`](../tokengs/data/registry.py) resolve dataset roots under the **repository root**:

| Path | Role |
|------|------|
| `data/dl3dv` | Training zips for `DL3DV10K` (e.g. DL3DV-ALL 960p undistorted). |
| `data/dl3dv_eval` | Eval set for `DL3DVEval` (e.g. DL3DV-10K-Benchmark). |
| `data/kubric` | Kubric multi-view 4D tar dump used by `finetune_dl3dv_kubric_*` presets. |

The DL3DV reader accepts both the canonical training zip tree and unpacked
scenes from the Hugging Face `DL3DV-10K-Sample`. For the sample, it detects
`<scene>/colmap/transforms.json` and selects the 960x540 `images_4` directory;
no repacking or copying is required.

Implementation details for readers and transforms live in [`tokengs/data/static/dl3dv.py`](../tokengs/data/static/dl3dv.py).

## Symlinks (recommended)

From the **repository root**:

```bash
mkdir -p data
ln -snf /absolute/path/to/DL3DV-ALL-960P-undistorted data/dl3dv
ln -snf /absolute/path/to/DL3DV-10K-Benchmark data/dl3dv_eval
ln -snf /absolute/path/to/objaverse_4d/kubric_mv data/kubric
```

For an unpacked sample, point the same training link at the sample root:

```bash
ln -snf /absolute/path/to/DL3DV-10K-Sample data/dl3dv
```

Supported DL3DV training layouts are:

```text
# Canonical training release
data/dl3dv/{1K,2K,...}/<scene>.zip
  <scene>/transforms.json
  <scene>/images/*.png

# Unpacked Hugging Face sample
data/dl3dv/<scene>/
  colmap/transforms.json
  colmap/images_4/*.png
```

`-snf` creates or replaces a symlink. Relative targets (e.g. `../datasets/dl3dv`) are fine if paths stay stable.

## Kubric Layout

The Kubric reader expects split directories and per-camera tar files:

```text
data/kubric/
  v0/
    <scene>/
      output_000.tar
      output_001.tar
      ...
  v1/
  v2/
```

Each `output_{view:03d}.tar` should contain `metadata.json`, `rgba_{frame:05d}.png`, and `depth_{frame:05d}.tiff`. The dynamic presets enable pointmap camera scaling, so the depth TIFFs are loaded for the input frames.

## Overrides without symlinks

Pass kwargs the dataset constructor accepts (for example `root_path`) via Tyro. See `dataset_kwargs` on [`Options`](../tokengs/options.py) and run:

```bash
python -m tokengs.train --help
python -m tokengs.evaluate --help
```
