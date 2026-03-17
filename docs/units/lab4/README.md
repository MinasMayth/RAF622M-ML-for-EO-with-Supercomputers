# Lab 4 — Preprocessing & Patch Extraction

**Notebook A:** [lab4_1_data_preprocessing.ipynb](../../../notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb)  
**Notebook B:** [lab4_2_patch_extraction.ipynb](../../../notebooks/iceland-ml/lab4_2_patch_extraction.ipynb)

## Scope
Prepare EO inputs and labels, then generate model-ready patch datasets.

## Learning outcomes
- Apply preprocessing/normalization for Sentinel-2 inputs.
- Align labels and create train/val/test partitions.
- Extract fixed-size patches and persist arrays + metadata.

## Suggested flow (2h)
1. Run Notebook A (preprocessing + splits)
2. Run Notebook B (patch extraction + storage)
3. Validate patch counts/shapes and class distribution

## Expected outputs
- Preprocessed arrays and split definitions
- Patch datasets for train/val/test
- Metadata files ready for model training
