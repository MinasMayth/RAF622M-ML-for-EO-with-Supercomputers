# Lab 7 — Model Evaluation

**Notebook:** [lab7_model_evaluation.ipynb](../../../notebooks/iceland-ml/lab7_model_evaluation.ipynb)
(source: [`notebooks_src/lab7_model_evaluation.py`](../../../notebooks_src/lab7_model_evaluation.py))

**Graded.** See [Grading & Submission Contract](../GRADING.md).

## Scope
Check whether everything before it was true. Lab 7 consumes the checkpoints and
saved predictions from labs 5 and 6, evaluates them on the held-out test split,
and reconciles every claim the earlier labs recorded.

## Why this notebook was rewritten from scratch
The 2025/26 version had **zero executed cells** and *synthesised* its own
predictions, described in the notebook as "matching Lab 5 performance". That is a
self-confirming loop: the lab could not find a problem because it was testing
whether its own assumption equalled itself. (It did not even reproduce the number
it claimed to match — the fabricated data scored 0.8590 balanced accuracy against
the 0.8338 it was meant to confirm.)

It also hard-coded lab 5.2's `0.9231 / 0.8338 / 0.8376` as ground truth. Those
were **validation** metrics on 455 samples where five of ten classes had support
≤ 9 and two had n = 1.

Scoring the old notebook's own fabricated predictions with the course metric code
(their cell 8, verbatim: `seed=42`, `n_background=800`, 20 per minority class)
gives `acc 0.9286 / balanced 0.8596 / macro-F1 0.7850`. That matches **none** of
the three numbers it was built to confirm — accuracy is close, balanced accuracy
is 0.026 high, macro-F1 is 0.053 low. The fabrication was circular *and*
internally inconsistent.

This notebook refuses to synthesise a stand-in. If the inputs from labs 5 and 6
are missing, it raises an error naming the lab that should have produced them.

## Learning outcomes
- Evaluate a real checkpoint on the held-out test split, and recompute every
  reported metric from saved raw predictions.
- Use a **fixed label set**, and quantify what deriving labels from
  `np.unique(y_pred)` was hiding — a collapsed model can score well that way.
- Explain why `balanced_acc − majority_accuracy` is meaningless: the first is the
  mean of per-class recalls, the second is the accuracy of a constant predictor.
  Only `balanced_acc − 1/K`, or a paired delta against a real baseline, is
  interpretable.
- Compare an i.i.d. bootstrap against a **scene-clustered** one and explain which
  interval is wider, and why patches within a scene are not independent.
- Compare two models on the *same* samples with `paired_bootstrap_delta`, and let
  `gates.gate_claim_supported` decide whether the difference survives seed noise.
- Read every lab's record out of `results.json` and flag any lab that reported
  validation metrics where test metrics were expected.

## Important course settings
- Metrics come from `eo_course.metrics.evaluate(..., labels=<fixed set>,
  groups=<scene ids>)`. Never from `np.unique`.
- Baselines are recomputed on the same test split with the same metric code, so
  every model number has a reference.
- Rare-class merging is decided on **train** support. Choosing merges by looking
  at test counts is test-set selection.
- Numbers are read from `results.json` via `results.runs_for_lab(...)`. Nothing
  from an earlier lab is ever retyped.

## Suggested flow (2 h)
1. Load the split of record and verify its hash against your lab 5 record.
2. Load real predictions (or a real checkpoint). Score the baselines.
3. Run the `np.unique` negative control and the meaningless-subtraction exercise.
4. Compare i.i.d. against scene-clustered confidence intervals.
5. Paired-compare lab 5 against lab 6; reconcile all labs; read the gate board.

## Expected outputs
- Test metrics recomputed from saved predictions, with scene-clustered CIs.
- A paired delta against the lab 5 baseline, judged against seed spread.
- A cross-lab reconciliation table built from `results.json`.
- A written report in which every number is traceable to a record.
- `results.record_run(lab="lab7", ...)`, raw predictions saved via
  `gates.gate_predictions_saved`, and `scripts/summarize_results.py --check`
  passing.
