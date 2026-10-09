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
│   ├── lab4_1_preprocessing.ipynb
│   ├── lab4_2_preprocessing_patches.ipynb
│   ├── lab5_1_baseline_training.ipynb
│   ├── lab5_2_model_evaluation.ipynb
│   └── lab6_finetune.ipynb
├── docs/                     # Documentation
│   ├── iceland-ml/           # Course guides
│   └── units/                # Lab-specific docs
├── requirements.txt          # Python dependencies
└── pyproject.toml           # Project configuration
```

## 🚀 Getting Started

### Step 1: Create Required Accounts
1. **Judoor Account** (do this first - takes 1-2 days): https://judoor.fz-juelich.de/register
2. **Google Earth Engine** (needed for Lab 3): https://earthengine.google.com/signup

### Step 2: Join the Training Project
After your Judoor account is approved, join the `training2653` project.

### Step 3: Follow the Labs
Work through the notebooks sequentially starting with Lab 1.

## 📓 Lab Schedule

| Lab | Topic | Notebook |
|-----|-------|----------|
| 1 | Judoor & HPC Access | `lab1_judoor_hpc_access.ipynb` |
| 2 | Jupyter-JSC & Git | `lab2_jupyter_jsc_git.ipynb` |
| 3 | Sentinel-2 Acquisition | `lab3_1_data_acquisition.ipynb` |
| 4 | Data Preprocessing | `lab4_1_preprocessing.ipynb` |
| 5 | Patch Extraction | `lab4_2_preprocessing_patches.ipynb` |
| 6 | Model Training | `lab5_1_baseline_training.ipynb` |
| 7 | Model Evaluation | `lab5_2_model_evaluation.ipynb` |
| 8 | TerraTorch Fine-tuning | `lab6_finetune.ipynb` |

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
