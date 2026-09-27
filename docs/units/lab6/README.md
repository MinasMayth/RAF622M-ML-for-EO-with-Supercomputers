# Lab 6 — Foundation Model Fine-tuning (`TerraTorch`)

**Notebook:** [lab6_terratorch_finetuning.ipynb](../../../notebooks/iceland-ml/lab6_terratorch_finetuning.ipynb)
(source: [`notebooks_src/lab6_terratorch_finetuning.py`](../../../notebooks_src/lab6_terratorch_finetuning.py))

**Graded.** See [Grading & Submission Contract](../GRADING.md).

## Scope
Fine-tune Prithvi-EO-2.0 on the course dataset with TerraTorch, and establish —
by measurement, not by reputation — whether the pretrained weights helped.

## What changed for 2027
This lab produced the worst result in the 2025/26 series. The fine-tuned
foundation model reported `test/Accuracy = 0.10000000149011612`. That is exactly
1/10 for a 10-class problem: **chance**. The majority-class baseline was 0.538.
The notebook presented it as a fine-tuning result and moved on.

Three independent causes, all silent:

1. **No normalization anywhere.** TerraTorch does not normalize your data for
   you. Patches were stretched to [0, 1] and fed to a backbone whose pretraining
   means are near 1000 DN — a ~1000× scale error.
2. **Wrong band order.** The config declared
   `["BLUE", "RED", "GREEN", "NIR_NARROW"]`, but the data is B02, B03, B04, B08 =
   Blue, **Green**, **Red**. TerraTorch permutes the pretrained patch-embed
   weights *by band name*, so channels 1 and 2 were silently transposed.
3. **Model selection on a degenerate metric.** Macro-accuracy over classes with
   one test sample, and no `ckpt_path="best"` at test time.

The commented-out alternative in the old config, `["B02", "B03", "B04", "B08"]`,
is **worse**, not better: those are not valid `HLSBands` values, so every input
filter falls back to random init and the pretrained input layer is discarded with
no error.

## Learning outcomes
- Explain pre-training vs fine-tuning for a geospatial foundation model, and what
  a patch-embed weight permutation actually is.
- Normalize input to a pretrained backbone using the pretraining statistics, and
  verify tensor units (`gates.gate_input_units`).
- Get band names right, and prove from the loaded weights whether pretraining was
  used at all.
- Run the three ablations that matter: normalization on/off, correct/swapped/
  invalid band order, pretrained/random-init.
- Report a foundation model losing to a linear probe as a finding.

## Important course settings
- **4 bands**: B02, B03, B04, B08 → `["BLUE", "GREEN", "RED", "NIR_NARROW"]`,
  which is `radiometry.PRITHVI_V2_BANDS[:4]`, with means
  `[1087, 1342, 1433, 2734]` and stds `[2248, 2179, 2178, 1850]`.
  Prithvi-EO-2.0 is **6 optical bands, no DEM/slope/aspect**; the full order is
  Blue, Green, Red, narrow-NIR, SWIR-1, SWIR-2 — *not* Landsat B-number order.
  If you swap in B11 (SWIR-1) you must take means from slot 5, not slot 4.
- Use `radiometry.prithvi_norm(...)`; do not retype the constants.
- The committed `lab6_config.yaml` is a historical artifact with a bad split
  (`test_ratio: 0.05`, `epochs: 1`, `split_strategy: stratified`). Generate a
  corrected config into your own run directory and diff it against that file.
- Fine-tuning a ViT does not belong on a login node. Use
  `slurm/train_jureca.sbatch` on `dc-gpu` (A100-40GB). Jupyter-JSC caps GPU at 1.

## Suggested flow (2 h)
1. Baselines first — the lab 5 table still applies to a foundation model.
2. Run the broken configuration deliberately; record the number.
3. Fix normalization; run again.
4. Run the band-order arms; inspect the patch-embed weights.
5. Pretrained vs random-init. Read the gate board.

## Expected outputs
- All ablation arms in `results.json`, with the split hash.
- A statement of which arm you expected to win and whether it did.
- A fine-tuned checkpoint that loads with `weights_only=True`.
