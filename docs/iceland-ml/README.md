# Machine Learning for Earth Observation powered by Supercomputers

## Course: TÖV606M

Welcome to the course! This module guides you through building a complete ML pipeline for Earth Observation using Sentinel-2 imagery, land cover classification, and HPC infrastructure at Jülich Supercomputing Centre.

### Course Details
- **Credits:** 6 ECTS
- **Instructors:** 
  - Gabriele Cavallaro (gcavallaro@hi.is) - Course Lead
  - Rocco Sedona (r.sedona@fz-juelich.de) - Technical Lead
  - Samy Hashim (s.hashim@fz-juelich.de) - Lab Instructor
  - Ehsan Zandi (e.zandi@fz-juelich.de)
- **Semester:** Spring 2025-2026
- **Modality:** Mixed in-person/online (Iceland + Germany)
- **HPC Resources:** JURECA (JSC/Judoor account required)

### Learning Outcomes
By completing this course, you will be able to:
- Access and manage HPC resources (Judoor, JURECA, SLURM)
- Acquire and preprocess satellite imagery (Sentinel-2 via Google Earth Engine)
- Build and train deep learning models for land cover classification
- Evaluate model performance using industry-standard metrics
- Deploy ML workflows on supercomputer infrastructure
- Fine-tune geospatial foundation models (TerraTorch, Prithvi)
- Apply models to generate classification maps and visualizations

---

## Lab Structure (9 sessions × 120 minutes)

Each lab builds progressively toward a complete ML pipeline. Labs are numbered
to match the notebooks and the course wiki: 1, 2, 3, 4.1, 4.2, 5.1, 5.2, 6, 7.

**Labs 5.1, 5.2, 6 and 7 are graded.** Read
[Grading & Submission Contract](../units/GRADING.md) before you start any of
them — submission is a machine-checked `results.json` record plus raw
predictions, not a description of what you did.

Every notebook is generated from a source file in `notebooks_src/` and ships
**with no outputs**. That is deliberate: a stored output is a number you did not
produce. Run the cells.

### Lab 1: Judoor Account and Access to HPC
**Week 1 | Duration: 120 min | Mode: Online**

**Topics:**
- Introduction to Jülich Supercomputing Centre (JSC)
- Creating and activating Judoor accounts
- SSH key setup and authentication
- First login to JURECA supercomputer
- Understanding HPC filesystem (home, project, scratch)
- Basic SLURM commands

**Deliverables:**
- ✅ Active Judoor account
- ✅ Membership in `training2653` project
- ✅ Successful SSH connection to JURECA
- ✅ Personal workspace directory structure

📓 **Notebook:** [`lab1_judoor_hpc_access.ipynb`](../../notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb)

---

### Lab 2: Jupyter-JSC and Git Basics
**Week 3 | Duration: 120 min | Mode: Online**

**Topics:**
- Launching Jupyter-JSC (web-based JupyterLab on HPC)
- Git fundamentals for version control
- Cloning course repository
- Creating Python virtual environments
- Registering custom Jupyter kernels
- First notebook execution on HPC

**Deliverables:**
- ✅ Running Jupyter-JSC session
- ✅ Cloned course repository
- ✅ Custom Python kernel (ML-EO Course)
- ✅ Git identity configuration

📓 **Notebook:** [`lab2_jupyter_jsc_git.ipynb`](../../notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb)

---

### Lab 3: Copernicus Data Space — Sentinel-2 Acquisition
**Week 6 | Duration: 120 min | Mode: Online**

**Topics:**
- OAuth2 authentication against the Copernicus Data Space Ecosystem
- Defining an Area of Interest and a temporal window
- Querying and filtering Sentinel-2 L2A (cloud cover, sensing date)
- Reading credentials from the environment (never from the notebook)
- Downloading and verifying imagery, preserving metadata

**Deliverables:**
- ✅ Credentials configured as environment variables, nothing printed
- ✅ 4 Sentinel-2 L2A scenes over one European tile, March–October 2018, ≤30% cloud
- ✅ A request manifest recording what you asked for and what you got
- ✅ Downloaded imagery with sidecar metadata

📓 **Notebook:** [`lab3_1_data_acquisition.ipynb`](../../notebooks/iceland-ml/lab3_1_data_acquisition.ipynb)

---

### Lab 4.1: Data Preprocessing
**Week 7 | Duration: 120 min | Mode: Online**

**Topics:**
- Digital numbers vs reflectance: the `QUANTIFICATION_VALUE = 10000` division
- Reprojection with explicit `src_nodata` / `dst_nodata`
- Why labels must be resampled with `nearest` and never with interpolation
- Aligning CORINE land cover with Sentinel-2 geometry
- Writing artifacts that describe themselves (band list, CRS, shape)

**Deliverables:**
- ✅ Reflectance-scaled imagery, range-asserted
- ✅ CORINE aligned to the image grid, label codes verified against the class table
- ✅ `bands.json` and a per-output manifest
- ✅ A preprocessing report that fails loudly on a stale output

📓 **Notebook:** [`lab4_1_data_preprocessing.ipynb`](../../notebooks/iceland-ml/lab4_1_data_preprocessing.ipynb)

---

### Lab 4.2: Patch Extraction
**Week 8 | Duration: 120 min | Mode: Online**

**Topics:**
- Extracting fixed-size patches with full provenance (scene, row, col)
- Why a 3×3 patch against a 100 m CORINE label is not an independent sample
- The cost of truncating a patch list in raster order
- Persisting patches so they cannot be double-loaded

**Deliverables:**
- ✅ Patch archives named `patches_*_scene.npz`, with provenance
- ✅ A count of how many patches share each label pixel
- ✅ Gate board for duplicates and provenance, green or diagnosed
- ✅ A `results.json` record for the extraction

📓 **Notebook:** [`lab4_2_patch_extraction.ipynb`](../../notebooks/iceland-ml/lab4_2_patch_extraction.ipynb)

---

### Lab 5.1: Baseline Training (raw PyTorch) — **graded**
**Week 10 | Duration: 120 min | Mode: Online**

**Topics:**
- Trivial baselines before any model: majority, prior-matched random, per-scene majority, NDVI rule, linear probe
- Group (scene + spatial block) splits, and a random split as a negative control
- Fitting normalization on the training split only
- Class imbalance as a decision with five comparable arms
- Three seeds, and whether your improvement survives seed noise

**Deliverables:**
- ✅ Baseline table recorded before training
- ✅ Split manifest hash for the split you actually used
- ✅ ≥3 seeds of the chosen arm, mean ± spread
- ✅ Gate board, green or diagnosed in writing

📓 **Notebook:** [`lab5_1_cnn_training.ipynb`](../../notebooks/iceland-ml/lab5_1_cnn_training.ipynb)

---

### Lab 5.2: PyTorch Lightning — **graded**
**Week 11 | Duration: 120 min | Mode: Online**

**Topics:**
- `LightningModule` / `LightningDataModule` structure, and why the split manifest is passed in
- Model selection: monitor `val/loss`, test with `ckpt_path="best"`
- Scheduler `interval="epoch"`, and what happens when you forget it
- Checkpoint round-trips: a result you cannot reload is not a result
- Moving the real run onto SLURM

**Deliverables:**
- ✅ Test metrics on the held-out split, separate from validation metrics
- ✅ Per-class support reported, thin classes merged or flagged
- ✅ Reloadable checkpoints (`weights_only=True`)
- ✅ `results.json` record with the split hash

📓 **Notebook:** [`lab5_2_pytorch_lightning.ipynb`](../../notebooks/iceland-ml/lab5_2_pytorch_lightning.ipynb)

---

### Lab 6: Foundation Model Fine-tuning (TerraTorch / Prithvi) — **graded**
**Week 12 | Duration: 120 min | Mode: Online**

**Topics:**
- Pre-training vs fine-tuning; what a geospatial foundation model gives you
- Band names and order: TerraTorch permutes pretrained weights *by name*
- Normalization: TerraTorch does not do it for you
- Verifying that pretrained weights were actually loaded
- Pretrained vs random-init, and normalization on/off as measured ablations

**Deliverables:**
- ✅ Three arms: correct bands, swapped bands, invalid band names — all three numbers
- ✅ Normalization on/off comparison
- ✅ Pretrained vs random-init comparison
- ✅ Gate board including `gate_input_units`

📓 **Notebook:** [`lab6_terratorch_finetuning.ipynb`](../../notebooks/iceland-ml/lab6_terratorch_finetuning.ipynb)

---

### Lab 7: Model Evaluation — **graded**
**Week 13 | Duration: 120 min | Mode: Online**

**Topics:**
- Evaluating a real checkpoint on the held-out test split
- Fixed label sets, and the class your model never predicts
- Scene-clustered bootstrap confidence intervals
- Paired comparison of two models on the same samples
- Error analysis: top confusions, and what they mean for CORINE

**Deliverables:**
- ✅ Test metrics recomputed from saved raw predictions
- ✅ Confidence intervals that respect scene correlation
- ✅ A paired delta against your lab 5 baseline, with seed spread
- ✅ A short written report whose numbers are all traceable to `results.json`

📓 **Notebook:** [`lab7_model_evaluation.ipynb`](../../notebooks/iceland-ml/lab7_model_evaluation.ipynb)

---

## Getting Started

### Required Knowledge
- Python programming (intermediate level)
- Basic machine learning concepts
- Familiarity with NumPy, Matplotlib
- Linux command line basics

### Required Accounts
- **Judoor Account:** https://judoor.fz-juelich.de/register
- **Google Earth Engine:** https://earthengine.google.com/signup
- **GitHub Account:** For cloning course repository

### Software Requirements
- SSH client (Terminal on Linux/Mac, PuTTY on Windows)
- Modern web browser (for Jupyter-JSC)
- Git (for version control)

---

## Resources

### Notebooks
All lab notebooks are available in [`notebooks/iceland-ml/`](../../notebooks/iceland-ml/):
- `lab1_judoor_hpc_access.ipynb`
- `lab2_jupyter_jsc_git.ipynb`
- `lab3_1_data_acquisition.ipynb`
- `lab4_1_data_preprocessing.ipynb`
- `lab4_2_patch_extraction.ipynb`
- `lab5_1_cnn_training.ipynb`
- `lab5_2_pytorch_lightning.ipynb`
- `lab6_terratorch_finetuning.ipynb`
- `lab7_model_evaluation.ipynb`

They are **generated** from the sources in [`notebooks_src/`](../../notebooks_src/)
by `python scripts/nbbuild.py`. Do not edit the `.ipynb` files directly, and do
not commit them with outputs you intend someone else to trust — run the cells.

### The `eo_course` package
Everything the labs share — paths, radiometry, labels, patch extraction, splits,
metrics, baselines, results, and the grading gates — lives in
[`eo_course/`](../../eo_course/). Install it with `pip install -e .` and read
the module docstrings; they explain *why* each piece exists, usually at the
expense of a specific 2025/26 bug. The test suite is `pytest`.

### Documentation
- **JSC Documentation:** https://apps.fz-juelich.de/jsc/hps/jureca/
- **Judoor Portal:** https://judoor.fz-juelich.de
- **Copernicus Data Space:** https://dataspace.copernicus.eu
- **PyTorch Tutorials:** https://pytorch.org/tutorials/
- **TerraTorch:** https://terratorch.readthedocs.io

### Communication
- **Slack Channel:** [Invite link provided by instructors]
- **Email Support:** s.hashim@fz-juelich.de
- **Office Hours:** By appointment (3 days advance notice)

---

## Assessment

### Lab Participation
- Complete all 8 lab exercises
- Submit working code and results
- Document preprocessing and training choices

### Final Project
- Apply learned techniques to custom AOI
- Train and evaluate classification model
- Present results (Week 14)

---

## Tips for Success

1. **Start Early:** HPC account setup takes time
2. **Test Incrementally:** Run code step-by-step, don't wait until deadline
3. **Ask Questions:** Use Slack channel for quick help
4. **Save Often:** Use checkpoints, Git commits, and backups
5. **Monitor Resources:** Check SLURM job status, GPU utilization
6. **Document Work:** Keep notes on experiments and results

---

## Next Steps

1. **Before Lab 1:** Create Judoor account (can take 1-2 days for approval)
2. **Before Lab 3:** Sign up for Google Earth Engine
3. **During Labs:** Follow notebooks sequentially, complete exercises
4. **After Labs:** Experiment with different datasets, architectures, hyperparameters

---

**Questions?** Contact Samy Hashim at s.hashim@fz-juelich.de

**Good luck, and enjoy your journey into ML for Earth Observation!** 🚀🛰️🌍