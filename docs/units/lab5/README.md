# Lab 5 — CNN Training & Class Imbalance

**Notebook:** [lab5_1_transformer_training.ipynb](../../../notebooks/iceland-ml/lab5_1_transformer_training.ipynb)

## Scope
Train a baseline CNN for land-cover classification and handle severe class imbalance.

## Learning outcomes
- Build/train a CNN on patch data.
- Compare class-imbalance strategies:
  - class-weighted cross-entropy
  - weighted random sampler
- Evaluate with imbalance-aware metrics (balanced accuracy, macro-F1).

## Important course settings
- Current lab setup uses **4 spectral bands**.
- Keep only one imbalance strategy active per run.

## Suggested flow (2h)
1. Load prepared data and inspect class distribution
2. Configure CNN + imbalance strategy
3. Train, validate, and run diagnostics

## Expected outputs
- Trained CNN checkpoint/logs
- Confusion matrix + per-class recall
- Comparison notes between the two imbalance strategies
