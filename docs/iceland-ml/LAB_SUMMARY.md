# Lab Quick Reference Guide

Notebooks are generated from sources in [`notebooks_src/`](../../notebooks_src/)
by `python scripts/nbbuild.py`. They ship with **no outputs** — run the cells.
Graded labs are 5.1, 5.2, 6 and 7; see
[Grading & Submission Contract](../units/GRADING.md).

## 🗓️ Lab Schedule

| Lab | Topic | Duration | Main Deliverable | Graded |
|-----|-------|----------|------------------|--------|
| Lab 1 | Judoor & HPC Access | 120 min | Working SSH/JURECA access | |
| Lab 2 | Jupyter-JSC & Git | 120 min | Jupyter session + repo clone + kernel | |
| Lab 3 | Sentinel-2 Acquisition (Copernicus) | 120 min | 4 scenes + request manifest | |
| Lab 4.1 | Data Preprocessing | 120 min | Reflectance stacks + aligned CORINE | |
| Lab 4.2 | Patch Extraction | 120 min | Patch archives with provenance | |
| Lab 5.1 | Baseline Training (PyTorch) | 120 min | Baselines + 3 seeds + gate board | ✅ |
| Lab 5.2 | PyTorch Lightning | 120 min | Test metrics + reloadable checkpoints | ✅ |
| Lab 6 | TerraTorch / Prithvi Fine-tuning | 120 min | Normalization/band/pretrained ablations | ✅ |
| Lab 7 | Model Evaluation | 120 min | Metrics recomputed from saved predictions | ✅ |

---

## 📚 Lab Details

### Lab 1 — Judoor & HPC Access
**Notebook:** [`notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb`](../../notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb)

**Core steps**
1. Create Judoor account and join `training2600`
2. Configure SSH keys
3. Connect to JURECA and check storage areas (`HOME`, `PROJECT`, `SCRATCH`)
4. Run basic Slurm commands (`squeue`, `sinfo`, `sbatch`)

**Output**
- Working cluster access and basic HPC workflow

---

### Lab 2 — Jupyter-JSC & Git
**Notebook:** [`notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb`](../../notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb)

**Core steps**
1. Launch a Jupyter-JSC session
2. Clone and sync the course repository
3. Create a Python virtual environment
4. Register and test a custom Jupyter kernel

**Output**
- Usable notebook environment on JSC with reproducible Git setup

---

### Lab 3 — Sentinel-2 Data Acquisition
**Notebook:** [`notebooks/iceland-ml/lab3_1_data_acquisition.ipynb`](../../notebooks/iceland-ml/lab3_1_data_acquisition.ipynb)

**Core steps**
1. Authenticate against the Copernicus Data Space Ecosystem (OAuth2, environment variables)
2. Define AOI and temporal constraints
3. Filter scenes (≤30% cloud, four months across 2018)
4. Download scenes and write a request manifest

**Output**
- Four Sentinel-2 L2A scenes staged for preprocessing

---

### Lab 4.1 — Data Preprocessing
**Notebook:** [`notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb`](../../notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb)

**Core steps**
1. Convert digital numbers to reflectance (`÷ QUANTIFICATION_VALUE`)
2. Reproject CORINE with explicit nodata and `resampling=nearest`
3. Verify label codes against the CORINE class table
4. Write self-describing outputs (`bands.json`, per-output manifest)

**Output**
- Reflectance stacks and aligned CORINE rasters, range-asserted

---

### Lab 4.2 — Patch Extraction
**Notebook:** [`notebooks/iceland-ml/lab4_2_patch_extraction.ipynb`](../../notebooks/iceland-ml/lab4_2_patch_extraction.ipynb)

**Core steps**
1. Extract 3×3 patches carrying `scene`/`row`/`col`
2. Save as `patches_*_scene.npz` and reload by key
3. Count how many patches share each 100 m label pixel
4. Run the duplicate and provenance gates

**Output**
- Patch archives with provenance, and the label-sharing statistic lab 5 needs

**Note:** normalization is deliberately *not* fitted here. Fitting it before a
split exists is the bug that made every 2025/26 number an interpolation score.

---

### Lab 5.1 — Baseline Training (raw PyTorch) — graded
**Notebook:** [`notebooks/iceland-ml/lab5_1_cnn_training.ipynb`](../../notebooks/iceland-ml/lab5_1_cnn_training.ipynb)

**Core steps**
1. Score trivial baselines before training anything
2. Build a group (scene + spatial block) split; record its hash
3. Fit normalization on the training split only
4. Train one imbalance arm across three seeds
5. Read the gate board and diagnose anything red

**Output**
- Baseline table, split hash, three seeds with spread, gate board

---

### Lab 5.2 — PyTorch Lightning — graded
**Notebook:** [`notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb`](../../notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb)

**Core steps**
1. Restructure the lab 5.1 experiment as `LightningModule` / `LightningDataModule`
2. Get the scheduler `interval` and `EarlyStopping` direction right
3. Select on validation loss, test with `ckpt_path="best"`
4. Report per-class support; separate val metrics from test metrics
5. Reload the checkpoint with `weights_only=True`

**Output**
- Test metrics on the held-out split, reloadable checkpoints, `results.json` records

---

### Lab 6 — TerraTorch / Prithvi Fine-tuning — graded
**Notebook:** [`notebooks/iceland-ml/lab6_terratorch_finetuning.ipynb`](../../notebooks/iceland-ml/lab6_terratorch_finetuning.ipynb)

**Core steps**
1. Run the 2025/26 configuration deliberately and record the number
2. Fix normalization; run again
3. Run the band-order arms (correct / swapped / invalid names) and inspect the patch-embed weights
4. Compare pretrained against random init

**Output**
- Three ablation arms in `results.json`, and a statement of which you expected to win

---

### Lab 7 — Model Evaluation — graded
**Notebook:** [`notebooks/iceland-ml/lab7_model_evaluation.ipynb`](../../notebooks/iceland-ml/lab7_model_evaluation.ipynb)

**Core steps**
1. Load real saved predictions or a real checkpoint — never synthesise them
2. Recompute baselines on the same test split with the same metric code
3. Use a fixed label set, and see what `np.unique` was hiding
4. Compare i.i.d. against scene-clustered bootstrap intervals
5. Paired-compare lab 5 against lab 6, and reconcile every lab's `results.json` record

**Output**
- Metrics recomputed from saved predictions, CIs that respect scene correlation, a cross-lab reconciliation table

---

## 🔧 Common Issues & Solutions

### Jupyter-JSC job pending
Use fewer resources or wait for queue availability. Jupyter-JSC caps a session at
one GPU and warns you; it is not a training environment.

### Copernicus token authentication fails
Check that `COPERNICUS_CLIENT_ID` starts with `sh-` and that the secret was not
truncated. Token requests are rate-limited (HTTP 429) — reuse a token rather than
requesting one per call.

### Slurm job exits immediately
Check account (`training2600`) and partition (`dc-cpu` / `dc-gpu`, lowercase) and
inspect the stderr log. `/p/project/training2600` does not exist; it is
`/p/project1/training2600`.

### Training out-of-memory
Reduce batch size and/or workers; verify GPU allocation.

### Model predicts one class
Check the two usual causes in order: an input-unit mismatch
(`gates.gate_input_units`), then a batch size so large you get a handful of steps
per epoch.

### `GateFailure: run_id already recorded`
`results.json` is append-only by design. Name the run `<arm>_<seed>` or
`<arm>_<seed>_v2`; do not delete the old record.

---

## ✅ Completion Checklist

- [ ] Lab 1: SSH into JURECA; workspace layout created
- [ ] Lab 2: Jupyter-JSC session; repo cloned; custom kernel selected
- [ ] Lab 3: Four L2A scenes staged; request manifest written
- [ ] Lab 4.1: Reflectance scaling asserted; CORINE aligned with `nearest`
- [ ] Lab 4.2: Patches saved with provenance; duplicate gate green or diagnosed
- [ ] Lab 5.1: Baselines recorded **before** training; split hash recorded; 3 seeds run
- [ ] Lab 5.2: Test metrics separate from val; checkpoint reloads with `weights_only=True`
- [ ] Lab 6: Normalization, band-order and pretrained-vs-random arms all run
- [ ] Lab 7: Metrics recomputed from saved predictions; paired delta tested against seed noise
- [ ] All graded labs: `results.json` record written, gate board green or diagnosed in writing

---

## 📖 Core Resources

- [JSC JURECA Docs](https://apps.fz-juelich.de/jsc/hps/jureca/)
- [Jupyter-JSC Guide](https://apps.fz-juelich.de/jsc/hps/jupyter/)
- [Copernicus Data Space](https://dataspace.copernicus.eu)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TerraTorch](https://terratorch.readthedocs.io)

---

**Grading:** see [Grading & Submission Contract](../units/GRADING.md). A red gate
board with a correct diagnosis scores full marks; a green board you cannot
explain scores nothing.

---

## 📖 Core Resources

- [JSC JURECA Docs](https://apps.fz-juelich.de/jsc/hps/jureca/)
- [Jupyter-JSC Guide](https://apps.fz-juelich.de/jsc/hps/jupyter/)
- [Google Earth Engine Docs](https://developers.google.com/earth-engine)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Note:** This summary is aligned with the current course state. Future units will be added once finalized.