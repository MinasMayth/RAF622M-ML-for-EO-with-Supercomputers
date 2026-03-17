# Lab Quick Reference Guide


## 🗓️ Current Lab Schedule

| Lab | Topic | Duration | Main Deliverable |
|-----|-------|----------|------------------|
| Lab 1 | Judoor & HPC Access | 120 min | Working SSH/JURECA access |
| Lab 2 | Jupyter-JSC & Git | 120 min | Jupyter session + repo clone + kernel |
| Lab 3 | Sentinel-2 Data Acquisition | 120 min | Downloaded scenes + metadata |
| Lab 4 | Preprocessing & Patch Extraction | 120 min | Train/val/test patch datasets |
| Lab 5 | CNN Training & Class Imbalance | 120 min | Trained CNN + diagnostics |

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
1. Authenticate and access the imagery source (GEE workflow)
2. Define AOI and temporal constraints
3. Filter scenes (e.g., cloud threshold)
4. Export/download scenes and preserve metadata

**Output**
- Sentinel-2 scenes staged for preprocessing

---

### Lab 4 — Preprocessing & Patch Extraction
**Notebook A:** [`notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb`](../../notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb)  
**Notebook B:** [`notebooks/iceland-ml/lab4_2_patch_extraction.ipynb`](../../notebooks/iceland-ml/lab4_2_patch_extraction.ipynb)

**Core steps**
1. Preprocess imagery and labels
2. Build train/val/test splits
3. Extract patch datasets
4. Save arrays and metadata for model training

**Output**
- ML-ready patch tensors/arrays with split metadata

---

### Lab 5 — CNN Training & Class Imbalance
**Notebook:** [`notebooks/iceland-ml/lab5_1_transformer_training.ipynb`](../../notebooks/iceland-ml/lab5_1_transformer_training.ipynb)

**Current lab focus**
- Intro-level **CNN-only** training workflow
- **4 spectral bands** setup
- Class imbalance handling with two alternatives:
	- class-weighted cross-entropy, or
	- weighted random sampler

**Core steps**
1. Inspect class distribution and imbalance ratio
2. Train CNN with one selected balancing strategy
3. Evaluate using balanced accuracy, macro-F1, and confusion matrix

**Output**
- Trained model/checkpoint + imbalance-aware diagnostics

---

## 🔧 Common Issues & Solutions

### Jupyter-JSC job pending
Use fewer resources or wait for queue availability.

### Token authentication fails
Re-authenticate and retry token flow (clear stale browser session if needed).

### Slurm job exits immediately
Check account/project/partition and inspect stderr log.

### Training out-of-memory
Reduce batch size and/or workers; verify GPU allocation.

---

## ✅ Updated Completion Checklist

- [ ] Lab 1: SSH into JURECA successfully
- [ ] Lab 1: Create workspace structure on project/scratch
- [ ] Lab 2: Start Jupyter-JSC session
- [ ] Lab 2: Clone/sync repository and select custom kernel
- [ ] Lab 3: Acquire and stage Sentinel-2 scenes
- [ ] Lab 4: Run preprocessing notebook and create splits
- [ ] Lab 4: Run patch extraction notebook and save datasets
- [ ] Lab 5: Train CNN with one imbalance strategy
- [ ] Lab 5: Report balanced accuracy + macro-F1 + confusion matrix

---

## 📖 Core Resources

- [JSC JURECA Docs](https://apps.fz-juelich.de/jsc/hps/jureca/)
- [Jupyter-JSC Guide](https://apps.fz-juelich.de/jsc/hps/jupyter/)
- [Google Earth Engine Docs](https://developers.google.com/earth-engine)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Note:** This summary is aligned with the current course state. Future units will be added once finalized.