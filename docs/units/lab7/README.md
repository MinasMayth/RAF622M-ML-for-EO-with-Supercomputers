# Lab 7: Model Evaluation and Metrics

**Status:** Required  
**Duration:** 2 hours  
**Where you work:** JURECA, through Jupyter-JSC  
**Notebook:** [lab7_model_evaluation.ipynb](https://github.com/MinasMayth/RAF622M-ML-for-EO-with-Supercomputers/blob/main/notebooks/iceland-ml/lab7_model_evaluation.ipynb)

## What this lab is for

A model that is 94% accurate on your land-cover classes can be a model that never predicts anything
except the majority class. Lab 5 trained a classifier under severe class imbalance; Lab 7 gives you
the tools to tell whether it actually learned anything.

The lab works with synthetic predictions first, so you can see exactly how a metric behaves when you
change the data, before you point the same functions at your own Lab 5 model.

By the end you will be able to:

1. Explain why plain accuracy is misleading on imbalanced multiclass data.
2. Compute and interpret balanced accuracy and macro-averaged F1.
3. Read a confusion matrix and extract per-class recall for the classes you care about.
4. Choose a metric that matches the decision you are actually making.
5. Evaluate your own model's predictions with the same set of functions.

## Before you start

You need the trained model and test predictions from Lab 5 to complete the final part. Parts 1 to 9
run on synthetic data and need nothing from earlier labs beyond a working Python environment.

| What | Needed for |
| --- | --- |
| Lab 5 checkpoint and test predictions | Part 10, Scenario E, where you evaluate your own model |
| Working Jupyter-JSC kernel from Lab 2 | Everything |
| `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` | Everything |
| `torch`, `pytorch_lightning` | Only the cells that reload a trained model |

## How the session is structured

| Part | What happens | Rough time |
| --- | --- | --- |
| 1 to 2 | Setup, and the core evaluation functions you will reuse | 20 min |
| 3 to 6 | Synthetic imbalanced test set, overall metrics, per-class metrics | 35 min |
| 8 to 9 | Class distribution analysis and summary | 15 min |
| 10 | Metric behaviour across scenarios A to D, then your own data in Scenario E | 45 min |

Note that the notebook has no Part 7; the numbering in the source skips it. The sections run in
order regardless.

## What to look at carefully

!!! warning "Accuracy and balanced accuracy diverging is the whole point"
    In the synthetic example, accuracy is about 0.92 while balanced accuracy is far lower. If you
    report only the first number, you are reporting how well your model guesses the majority class.
    Watch the gap between the two as Scenario A increases the imbalance.

!!! warning "Macro-F1 and micro-F1 answer different questions"
    Macro averages each class equally, so a rare class you care about has the same weight as the
    dominant one. Micro pools all predictions together, which on imbalanced data behaves much like
    accuracy. Pick macro when rare classes matter, and say which you used.

!!! tip "Per-class recall is the number to put in a report"
    For land cover, "how often did we correctly map wetland" is a more useful statement than a single
    aggregate score. Part 6 produces it, and Scenario C shows how two models with identical accuracy
    can have very different per-class recall.

## What you hand in

- The overall metrics table for the synthetic test set, with accuracy, balanced accuracy and
  macro-F1 side by side.
- The per-class recall table, and one sentence on which classes the model handles worst.
- The confusion matrix figure.
- For Scenario E: the same three outputs computed on your Lab 5 model, plus a short note comparing
  it against the synthetic "good" model.

## Getting help

- If a metric looks wrong, check the class ordering in your ground-truth and prediction arrays
  first. A mismatch there produces numbers that look plausible and are meaningless.
- `sklearn.metrics.classification_report` prints most of what the notebook computes by hand. Use it
  as a cross-check, not as a replacement: the point of Parts 2 and 6 is that you can see how the
  numbers are formed.

## Next

Lab 6 fine-tunes Prithvi-EO-2.0 with TerraTorch. The evaluation functions you wrote here apply to
that model unchanged, so keep them in a module you can import rather than leaving them in one
notebook.
