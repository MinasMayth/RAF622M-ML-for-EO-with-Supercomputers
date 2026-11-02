# Lab 5 — CNN Training & Class Imbalance

**Notebooks:** 
- [lab5_1_transformer_training.ipynb](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab5_1_cnn_training.ipynb)
- [lab5_2_pytorch_lightning.ipynb](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb)

## Scope
- Train a baseline CNN for land-cover classification and handle severe class imbalance.
- Organize code using PyTorch Lightning `Trainer`, `LightningModule` and `LightningDataModule` classes.

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
