# Lab Quick Reference Guide


## 🗓️ Current Lab Schedule

| Lab | Topic | Duration | Main Deliverable |
|-----|-------|----------|------------------|
| Lab 1 | Judoor & HPC Access | 120 min | Working SSH/JURECA access + one completed batch job |
| Lab 2 | Jupyter-JSC & Git | 120 min | Jupyter session + repo clone + kernel |
| Lab 3 | Sentinel-2 Data Acquisition | 120 min | Downloaded scenes + metadata |
| Lab 4 | Preprocessing & Patch Extraction | 120 min | Train/val/test patch datasets |
| Lab 5 | CNN Training & Class Imbalance | 120 min | Trained CNN + diagnostics |
| Lab 6 | TerraTorch Fine-tuning | 120 min | Fine-tuned Prithvi-EO-2.0 checkpoint |
| Lab 7 | Model Evaluation & Metrics | 120 min | Metric comparison + per-class recall |

Labs 6 and 7 are listed in the order of the notebook files. If you work through the notebooks in
numeric order, Lab 6 (TerraTorch) comes before Lab 7 (evaluation), but the evaluation functions from
Lab 7 are useful for judging the Lab 6 model.

---

## 📚 Lab Details

### Lab 1 — Judoor & HPC Access
**Notebook:** [`notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb)  
**Landing page:** [units/lab1](../units/lab1/README.md)

**Core steps**
1. Create a Judoor account, enable multi-factor authentication, and join `training2600`
2. Generate an SSH key pair with a passphrase and register the public half in Judoor
3. Connect to JURECA and confirm where you are (`whoami`, `hostname`, `pwd`)
4. Activate the course project with `jutil env activate -p training2600`, then inspect
   `HOME`, `PROJECT` and `SCRATCH`
5. Build the standard course workspace layout under `$PROJECT`
6. Submit a batch job on `dc-cpu-devel`, watch it, read its `.out` and `.err` logs, and check
   the charge with `sacct`

**Output**
- Working cluster access, a course workspace, and one completed batch job with its job ID and
  `sacct` record

**Watch out for**
- `$PROJECT` and `$SCRATCH` are empty until the project is activated, so paths built from them
  silently resolve to the wrong place
- Omitting `--partition` puts a job on `dc-gpu`, not a CPU partition
- Compute nodes have no internet, so `pip install` inside a job script fails
- JSC rejects SSH keys without a passphrase

---

### Lab 2 — Jupyter-JSC & Git
**Notebook:** [`notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb)
**Landing page:** [units/lab2](../units/lab2/README.md)

**Core steps**
1. Launch a Jupyter-JSC session
2. Clone and sync the course repository
3. Create a Python virtual environment
4. Register and test a custom Jupyter kernel

**Output**
- Usable notebook environment on JSC with reproducible Git setup

---

### Lab 3 — Sentinel-2 Data Acquisition
**Notebook:** [`notebooks/iceland-ml/lab3_1_data_acquisition.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab3_1_data_acquisition.ipynb)
**Landing page:** [units/lab3](../units/lab3/README.md)

**Core steps**
1. Authenticate and access the imagery source (GEE workflow)
2. Define AOI and temporal constraints
3. Filter scenes (e.g., cloud threshold)
4. Export/download scenes and preserve metadata

**Output**
- Sentinel-2 scenes staged for preprocessing

---

### Lab 4 — Preprocessing & Patch Extraction
**Notebook A:** [`notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb)  
**Notebook B:** [`notebooks/iceland-ml/lab4_2_patch_extraction.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab4_2_patch_extraction.ipynb)
**Landing page:** [units/lab4](../units/lab4/README.md)

**Core steps**
1. Preprocess imagery and labels
2. Build train/val/test splits
3. Extract patch datasets
4. Save arrays and metadata for model training

**Output**
- ML-ready patch tensors/arrays with split metadata

---

### Lab 5 — CNN Training & Class Imbalance
**Notebook A:** [`notebooks/iceland-ml/lab5_1_cnn_training.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab5_1_cnn_training.ipynb)  
**Notebook B:** [`notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb)  
**Landing page:** [units/lab5](../units/lab5/README.md)

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

### Lab 6 — TerraTorch Fine-tuning
**Notebook:** [`notebooks/iceland-ml/lab6_terratorch_finetuning.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab6_terratorch_finetuning.ipynb)  
**Config:** [`notebooks/iceland-ml/lab6_config.yaml`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab6_config.yaml)  
**Landing page:** [units/lab6](../units/lab6/README.md)

**Core steps**
1. Understand the difference between pre-training and fine-tuning a geospatial foundation model
2. Assemble a TerraTorch task config for your own patch dataset
3. Fine-tune Prithvi-EO-2.0 on multi-spectral data
4. Save the checkpoint and the config that produced it

**Output**
- End-to-end TerraTorch workflow and a fine-tuned Prithvi-EO-2.0 checkpoint

---

### Lab 7 — Model Evaluation & Metrics
**Notebook:** [`notebooks/iceland-ml/lab7_model_evaluation.ipynb`](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab7_model_evaluation.ipynb)  
**Landing page:** [units/lab7](../units/lab7/README.md)

**Core steps**
1. Build the core evaluation functions
2. Evaluate a synthetic imbalanced test set and compare accuracy with balanced accuracy
3. Extract per-class recall and read the confusion matrix
4. Explore how metrics respond to imbalance, model quality, and error pattern
5. Apply the same functions to your Lab 5 model

**Output**
- Metric comparison across scenarios, plus per-class recall for your own model

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
- [ ] Lab 6: Fine-tune Prithvi-EO-2.0 and save the config used
- [ ] Lab 7: Report accuracy, balanced accuracy and macro-F1 side by side
- [ ] Lab 7: Report per-class recall for your own model

---

## 📖 Core Resources

- [JSC JURECA Docs](https://apps.fz-juelich.de/jsc/hps/jureca/)
- [Jupyter-JSC Guide](https://apps.fz-juelich.de/jsc/hps/jupyter/)
- [Google Earth Engine Docs](https://developers.google.com/earth-engine)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

**Note:** This summary covers the lab units that exist in the repository today. New units are added
here as their notebooks are finalised.