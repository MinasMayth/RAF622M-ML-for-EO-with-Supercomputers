# Lab 4 — Preprocessing & Patch Extraction

**Notebook A:** [lab4_1_data_preprocessing.ipynb](../../../notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb)
**Notebook B:** [lab4_2_patch_extraction.ipynb](../../../notebooks/iceland-ml/lab4_2_patch_extraction.ipynb)

Generated from sources in [`notebooks_src/`](../../../notebooks_src/); edit the
`.py` files and run `python scripts/nbbuild.py lab4`.

## Scope
Turn downloaded Sentinel-2 scenes and CORINE land cover into patch datasets that
carry enough provenance for lab 5 to build a leakage-free split.

## What changed for 2027
Four defects in 2025/26 made every downstream number untrustworthy:

- **Reflectance scaling was never applied.** Sentinel-2 L2A digital numbers are
  integers with `QUANTIFICATION_VALUE = 10000`. The old pipeline fed raw DN
  (order 1000–4000) straight into normalization and models.
- **`reproject` was called without `src_nodata`/`dst_nodata`.** Missing pixels
  became real value-0 pixels — dark water and bare soil invented out of nothing.
- **`glob("*_data.npz")` matched both the per-scene files and
  `combined_training_data.npz`.** Every patch was loaded twice, so duplicates
  appeared in train *and* test. Archives are now named
  `patches_*_scene.npz`, which makes the mistake structurally impossible.
- **Archives stored only `patches` and `labels`.** Nothing recorded which scene
  or pixel a patch came from, so no downstream lab could detect leakage. This is
  the root cause of every split bug in labs 5–7.

## Learning outcomes
- Convert digital numbers to reflectance and assert the resulting range.
- Reproject imagery with explicit nodata handling, and labels with `nearest` —
  and explain why interpolating a categorical raster manufactures class codes
  (averaging 321 and 333 gives 327, a real but wrong class).
- Extract patches that carry `scene`, `row`, `col`.
- Explain what a 3×3 patch (~30 m) against a 100 m CORINE label means for
  effective sample size: one label pixel supplies the label to ~9 neighbouring
  patches, and those nine are not independent observations.
- Write artifacts that describe themselves, so a stale output cannot be silently
  reused.

## Important course settings
- **4 bands**: B02, B03, B04, B08 (Blue, Green, Red, narrow-NIR), all 10 m.
- Patch size 3×3, stride 3.
- **Do not fit normalization in this lab.** The 2025/26 version fitted percentile
  bounds over the whole raster before any split existed, so every test patch
  contributed to the transform applied to every training patch. Normalization is
  fitted on the training split only, in lab 5.
- Do not truncate the patch list in raster order — taking the first N patches
  row-major keeps the north-west strip and discards the rest.

## Suggested flow (2 h)
1. Notebook A: scale to reflectance, reproject CORINE with `nearest`, verify the
   label codes against the class table, write `bands.json`.
2. Notebook B: extract patches with provenance, save, reload by key.
3. Count how many patches share each label pixel. That number is the input to
   the lab 5 split decision.
4. Read the gate board (duplicates, provenance).

## Expected outputs
- Reflectance-scaled stacks and aligned CORINE rasters, with manifests.
- `patches_*_scene.npz` archives, one per scene, provenance verified on load.
- A label-sharing statistic and a class-distribution report.
- A `results.json` record for the extraction.
