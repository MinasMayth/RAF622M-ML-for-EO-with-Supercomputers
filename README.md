# TÖV606M - Machine Learning for Earth Observation powered by Supercomputers

Welcome to the course repository! This repository contains all lab notebooks and materials for building a complete ML pipeline for Earth Observation using Sentinel-2 satellite imagery and HPC infrastructure.

## 📚 Course Overview

- **Credits:** 6 ECTS
- **Semester:** Spring 2025-2026 (January - April)
- **Modality:** Mixed in-person/online (Iceland + Germany)
- **HPC Resources:** JURECA at Jülich Supercomputing Centre

## 🎯 Learning Outcomes

By completing this course, you will learn to:
- Access and manage HPC resources (Judoor, JURECA, SLURM)
- Acquire and preprocess satellite imagery (Sentinel-2 via Google Earth Engine)
- Build and train deep learning models for land cover classification
- Evaluate model performance using industry-standard metrics
- Fine-tune geospatial foundation models (TerraTorch, Prithvi)

## 🗂️ Repository Structure

```
├── notebooks/iceland-ml/     # Lab notebooks (work here!)
│   ├── lab1_judoor_hpc_access.ipynb
│   ├── lab2_jupyter_jsc_git.ipynb
│   ├── lab3_1_data_acquisition.ipynb
│   ├── lab4_1_data_preprocessing.ipynb
│   ├── lab4_2_patch_extraction.ipynb
│   ├── lab5_1_cnn_training.ipynb
│   ├── lab5_2_pytorch_lightning.ipynb
│   ├── lab6_terratorch_finetuning.ipynb
│   ├── lab6_config.yaml
│   └── lab7_model_evaluation.ipynb
├── docs/                     # Documentation (published with MkDocs Material)
│   ├── iceland-ml/           # Course overview and lab summary
│   └── units/                # One landing page per lab
├── mkdocs.yml                # Site configuration
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project configuration
```

## 🚀 Getting Started

### Step 1: Create Required Accounts
1. **Judoor Account** (do this first - takes 1-2 days): https://judoor.fz-juelich.de/register
2. **Google Earth Engine** (needed for Lab 3): https://earthengine.google.com/signup

### Step 2: Join the Training Project
After your Judoor account is approved, join the `training2600` project.

### Step 3: Follow the Labs
Work through the notebooks sequentially starting with Lab 1.

## 📓 Lab Schedule

Each lab has a landing page under `docs/units/` and one or more notebooks under
`notebooks/iceland-ml/`. The table lists the notebook filenames as they exist in the repository.

| Lab | Topic | Notebook(s) | Landing page |
|-----|-------|-------------|--------------|
| 1 | Judoor & HPC Access | `lab1_judoor_hpc_access.ipynb` | [docs/units/lab1](docs/units/lab1/README.md) |
| 2 | Jupyter-JSC & Git | `lab2_jupyter_jsc_git.ipynb` | [docs/units/lab2](docs/units/lab2/README.md) |
| 3 | Sentinel-2 Acquisition | `lab3_1_data_acquisition.ipynb` | [docs/units/lab3](docs/units/lab3/README.md) |
| 4 | Preprocessing & Patch Extraction | `lab4_1_data_preprocessing.ipynb`, `lab4_2_patch_extraction.ipynb` | [docs/units/lab4](docs/units/lab4/README.md) |
| 5 | CNN Training & Class Imbalance | `lab5_1_cnn_training.ipynb`, `lab5_2_pytorch_lightning.ipynb` | [docs/units/lab5](docs/units/lab5/README.md) |
| 6 | TerraTorch Fine-tuning | `lab6_terratorch_finetuning.ipynb` | [docs/units/lab6](docs/units/lab6/README.md) |
| 7 | Model Evaluation & Metrics | `lab7_model_evaluation.ipynb` | [docs/units/lab7](docs/units/lab7/README.md) |

Lab 1 is the only one you can start before the course begins, and it is the one to finish first:
every later lab assumes the access and workspace it sets up.

## 📖 Documentation

- [Course Overview](docs/iceland-ml/README.md) - Detailed course information
- [Lab Summary](docs/iceland-ml/LAB_SUMMARY.md) - Quick reference for all labs

## 🔧 Prerequisites

- Python programming (intermediate level)
- Basic machine learning concepts
- Linux command line basics
- SSH client installed on your machine

## 💬 Getting Help

- **Slack Channel:** Check with instructors for invite link
- **Email:** s.hashim@fz-juelich.de
- **Office Hours:** By appointment (3 days advance notice)

## 📧 Instructors

- **Gabriele Cavallaro** (gcavallaro@hi.is) - Course Lead
- **Rocco Sedona** (r.sedona@fz-juelich.de) - Technical Lead  
- **Samy Hashim** (s.hashim@fz-juelich.de) - Lab Instructor
- **Ehsan Zandi** (e.zandi@fz-juelich.de)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Good luck with your learning journey!** 🛰️🌍
