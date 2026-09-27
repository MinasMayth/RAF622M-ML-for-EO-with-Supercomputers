# Lab 5 — Model Training (PyTorch & Lightning)

**Notebooks:**
- [lab5_1_cnn_training.ipynb](../../../notebooks/iceland-ml/lab5_1_cnn_training.ipynb) — raw PyTorch
- [lab5_2_pytorch_lightning.ipynb](../../../notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb) — Lightning

Both notebooks are **generated** from sources in `notebooks_src/`. Edit the
`.py` source and run `python scripts/nbbuild.py lab5`; do not edit the `.ipynb`.

**Graded.** See [Grading & Submission Contract](../GRADING.md).

## Scope
- Establish whether a trained model beats not training one, and make that judgement reproducible.
- Handle severe class imbalance as a measured decision rather than a default.
- Organize code using PyTorch Lightning `Trainer`, `LightningModule` and `LightningDataModule` classes.

## What changed for 2027
The 2025/26 version trained a model and reported a number. That number was wrong
in three independent ways, none of which anything in the notebook could detect:

- **No baselines.** With no majority-class or random reference, "0.69 accuracy"
  means nothing. Lab 6's fine-tuned foundation model scored 0.1000 — exactly
  chance — against a majority baseline of 0.538, and it was reported as a result.
- **Normalization fitted on the whole raster before the split existed.** Every
  test patch contributed to the transform applied to every training patch.
- **A random split over 3×3 patches sharing 100 m CORINE labels.** Neighbouring
  patches are not independent, so the reported accuracy was an interpolation score.

Lab 5.2 additionally reported **validation** metrics (0.9231 / 0.8338 / 0.8376)
as headline results on a split where 5 of 10 classes had support ≤ 9 and two had
n = 1. Lab 7 hard-coded those numbers as ground truth.

## Learning outcomes
- Build/train a CNN on patch data.
- Compare class-imbalance strategies:
  - class-weighted cross-entropy
  - weighted random sampler
- Evaluate with imbalance-aware metrics (balanced accuracy, macro-F1).
- Combine PyTorch Lightning trainer and datamodules and to separate model, data and training logic
- Use Callback for added functionalities

## Important course settings
- Current lab setup uses **4 spectral bands**.
- Keep only one imbalance strategy active per run.
- Lab 5.2 uses 224x224 patches instead of 3x3
- Lab 5.2 uses labels from majority land cover instead of central pixel

## Suggested flow (2h)
1. Load prepared data and inspect class distribution
2. Configure CNN + imbalance strategy
3. Train, validate, and run diagnostics

## Expected outputs
- Trained CNN checkpoint/logs
- Confusion matrix + per-class recall
- Comparison notes between the two imbalance strategies
- CORINE-custom dataset and data module
