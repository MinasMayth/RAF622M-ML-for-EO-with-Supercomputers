# Grading & Submission Contract

This page is the whole grading policy for the lab series. It exists because in
2025/26 the labs could be completed by running every cell in order, and they
were. Nothing below asks you to describe what you learned. It asks you to
produce artifacts that fail loudly when the work was not done.

## What you submit

Per graded lab (5.1, 5.2, 6, 7), one commit containing:

| Artifact | Where | Checked by |
|---|---|---|
| Run records | `results.json` (via `eo_course.results.record_run`) | `gates.gate_results_file` |
| Split provenance | `split_manifest_hash` inside each record | `gates.gate_split_hash_matches` |
| Raw predictions | `.npz` written by `gates.gate_predictions_saved` | grader recomputes every metric from these |
| Checkpoints | `paths.artifacts_dir()`, named `<run_id>.pt` | loaded with `weights_only=True` |
| Notebook, executed | `notebooks/iceland-ml/*.ipynb` | run cells must be present |

`results.json` is **append-only**. Reusing a `run_id` raises, so name runs
`<arm>_<seed>`. This is deliberate: a result you can silently overwrite is not
a result.

## The gate board

Every graded lab ends with `gates.print_gate_board(gates.run_all_gates(...))`.
Gates raise `GateFailure` (which extends `AssertionError`). A red board is not a
penalty — a red board that you have diagnosed in writing scores full marks; a
green board you cannot explain scores nothing.

| Gate | Threshold | Why this number |
|---|---|---|
| `gate_beats_baselines` | macro-F1 ≥ best baseline **+0.10** | The 2025/26 lab 6 reported 0.1000 — exactly 1/10, chance — against a majority baseline of 0.538. Nothing in the notebook noticed. |
| `gate_above_chance` | balanced-acc ≥ 1/K **+0.15** | Overall accuracy is uninformative under the class imbalance in this dataset. |
| `gate_class_support` | every scored class has **≥ 25** test samples | Lab 5.2's headline numbers were computed on a val split where 5 of 10 classes had support ≤ 9 and two had n = 1. |
| `gate_not_collapsed` | no class receives > **85%** of predictions | Catches the batch-256-on-578-samples collapse (3 steps/epoch). |
| `gate_split_is_grouped` | whole scenes held out, no scene straddles | Random splits over 3×3 patches sharing a 100 m CORINE label are interpolation, not prediction. |
| `gate_no_duplicate_patches` | no patch appears twice | `glob("*_data.npz")` matched per-scene files *and* the combined file; every patch was loaded twice. |
| `gate_input_units` | input in the units the model expects | TerraTorch does not normalize for you. |
| `gate_no_test_set_selection` | test split used exactly once | Lab 5.2 reported validation metrics as headline results; lab 7 hard-coded them as ground truth. |
| `gate_seeds_and_variance` | **≥ 3** seeds | A single-seed number is a sample, not a result. |
| `gate_claim_supported` | improvement > **2×** seed spread | Stops "0.71 vs 0.69, therefore better". |

Thresholds are constants at the top of `eo_course/gates.py`. Changing one is
allowed; hiding the change is not. If you alter a threshold, say so in the
`notes` field of the record and in your write-up.

## `results.json` record (schema v2)

```
run_id, lab, seed, n_seeds, split_manifest_hash, test_used_for_tuning,
config (incl. n_params), test{...}, val{...}, baselines{...}, git_sha, notes
```

`test{...}` must contain `n, overall_acc, balanced_acc, macro_f1, weighted_f1,
micro_f1, kappa, precision[], recall[], f1[], support[], predicted[],
confusion[]` — all produced by `eo_course.metrics.evaluate()` on the held-out
split, with a **fixed label set** (a class your model never predicts must still
appear, or it silently vanishes from the macro-average).

Print the exact contract with `eo_course.results.describe_contract()`.

## Decisions that must be recorded, not asserted

Each graded lab asks you to make choices. They are only gradeable if they land
in `config` or `notes`:

- which scenes you held out, and why those;
- which imbalance arm you ran, **and the arms you did not run**;
- which rare classes you merged (`metrics.merge_rare_classes`) and how the
  macro-average moved as a result;
- your predicted metric before running, and the gap you observed;
- the negative control (leaky split / unnormalized input / wrong band order),
  its number, and one sentence on why it is untrustworthy.

## What is *not* graded

- **Beating the baselines.** A well-diagnosed failure to beat `linear_probe` is
  a complete result. A fabricated success is not.
- **Number of runs.** Three seeds with variance reported beats thirty runs with
  one.
- **Accuracy.** We grade the measurement, not the magnitude.

## Late submissions

A submission whose gate board is red but whose diagnosis is correct will be
scored above a green submission with no diagnosis. If you are out of time,
submit the red board and say what you think is wrong.

## Academic integrity

Copying another student's notebook is visible: `split_manifest_hash` is derived
from your own scene choice and seed, and `git_sha` is recorded per run. Two
submissions with identical hashes will be treated as one submission by two
people. Working in a pair is fine for the project (max two members, per the
course syllabus); say so in `notes`.
