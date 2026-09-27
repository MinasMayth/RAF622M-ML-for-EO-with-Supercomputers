# Notebook source authoring spec — RAF622M / TÖV606M

Read this fully before writing any notebook source. It is the contract that keeps
nine notebooks written by different hands into one coherent course.

## 1. File format

Notebooks are **generated**. Write a `.py` file in `notebooks_src/` using VS Code /
jupytext cell markers, then run `python scripts/nbbuild.py`. Never hand-edit
`.ipynb` JSON.

```python
# %% [markdown]
# # Lab 5 — Training a baseline
#
# Markdown here. Every line must start with `# `. A blank markdown line is `#`.

# %%
import numpy as np        # code cell: ordinary Python, no comment prefix

# %% [markdown]
# ## Exercise 1 — predict before you run
```

Rules:
- Markdown lines are `# `-prefixed. A blank line inside markdown is a bare `#`.
- Code cells are plain Python and must be syntactically valid on their own.
- Cell order in the file is cell order in the notebook.
- Generated notebooks have **no outputs**. That is deliberate: a student must run
  them. Never write "expected output: 0.83" for a number that depends on data the
  student does not yet have.

`scripts/nbbuild.py` lints every build and **fails the build** on any of:
another user's username in a path, `/p/project/training2600` (it is `project1`),
`JURECA-DC_CPU`/`_GPU` (real names are `dc-cpu`/`dc-gpu`), `YOUR_ORG`,
`iceland-ml-course.git`, printed credential fragments, `CUDA_LAUNCH_BLOCKING`,
`multi_class=`, `rstrip('.SAFE')`, `path/to/model.pt`.
It warns on `np.random.rand*`, `np.load(...).values()`, `!source`, `!cd`,
`T_max=self.max_epochs`, and `trainer.test(model, datamodule=...)` with no
`ckpt_path`. Do not add code that trips a warning unless the cell's whole point
is to show the bug.

`nbbuild.py` lints; it does not parse. Run the compiler too:

```
python scripts/nbbuild.py lab5 && python scripts/check_notebooks.py
```

`check_notebooks.py` compiles every code cell of every generated notebook
(IPython magics are commented out first, so `%matplotlib inline` is fine). A
cell with a stray bracket builds happily and only fails in front of a student.

## 2. Use `eo_course`, do not re-implement

The package is the audited reference implementation. Labs must **call** it, not copy
it. Import at the top of the notebook:

```python
from eo_course import paths, radiometry, labels as lab_mod, patches as pat
from eo_course import splits, metrics, baselines as bl, results, gates
```

Available API (the module docstrings explain *why* each thing exists and are the
best source of prose for your markdown cells):

| module | use for |
|---|---|
| `paths` | every filesystem path: `paths.describe()`, `paths.training_data_dir()`, `paths.results_dir()`, `paths.matplotlib_cache_dir()`, `paths.require_existing()` |
| `radiometry` | `dn_to_reflectance`, `Norm` (fit on train only), `prithvi_norm`, `assert_reflectance_range`, `QUANTIFICATION_VALUE` |
| `labels` | `CORINE_CLASSES`, `CORINE_NODATA`, `valid_class_codes`, `LabelMap`, `class_counts`, `imbalance_ratio`, `level1_of` |
| `patches` | `extract_patches`, `save_patches`, `load_all_scenes`, `as_chw`, `SCENE_GLOB` |
| `splits` | `group_block_split`, `stratified_random_split` (negative control), `SplitManifest`, `block_ids`, `per_split_class_counts` |
| `metrics` | `evaluate` → `Metrics`, `bootstrap_ci`, `paired_bootstrap_delta`, `merge_rare_classes`, `top_confusions` |
| `baselines` | `run_all`, `majority_class`, `uniform_random`, `prior_matched_random`, `per_scene_majority`, `ndvi_rule`, `linear_probe` |
| `results` | `record_run`, `summary_table`, `write_csv`, `describe_contract` |
| `gates` | `gate_*` functions, `run_all_gates`, `print_gate_board`, threshold constants |
| `training` | `CorineModule`, `CorineDataModule`, `class_weights`, `FocalLoss`, `SmallCNN` |

## 3. Pedagogical structure — mandatory

The 2025/26 failure mode was: the notebook hands over a complete solution, the
student runs every cell, submits, passes, and never learns whether their model is
any good. Every notebook must contain, in this order:

1. **Header markdown** — what this lab produces, what it assumes from the previous
   lab, and an explicit "you must run this yourself" statement.
2. **Setup cell** — `paths.describe()`, MPLCONFIGDIR before matplotlib, seed.
3. **Concept markdown** — explain the *mechanism*, not the conclusion. The old
   notebooks asserted "ML models work better with normalized inputs" and moved on.
   Say what breaks, and by how much.
4. **Guided implementation** — where the toolkit does the work, call it. Where a
   student must think, leave a `# TODO(you):` with a *falsifiable* success
   condition, not a vague instruction.
5. **At least 2 predict-then-run exercises.** Format, every time:
   - state the setup,
   - ask for a **number or an ordering** written down *before* running,
   - the cell prints the actual,
   - the student records `predicted vs actual` and explains a disagreement.
   Grade the prediction, not the output. A cell that prints "Key insight: X" teaches
   nothing — code asserting the conclusion is the exact anti-pattern.
6. **A deliberate-bug or negative-control exercise** where possible: run the leaky
   split, watch accuracy rise, write one sentence on why the number is untrustworthy.
7. **Gate board** — call the relevant `gates.*` and `print_gate_board`. Gates raise;
   that is the point.
8. **Deliverable cell** — write to `results.json` via `results.record_run(...)` and
   print `results.summary_table(...)`.
9. **Submission checklist** — concrete artifacts, not self-attested "☐ I understand".

## 4. Voice

- Second person plural ("you", "we"). Present tense.
- Every warning names a **number** and a **consequence**. "Normalization must be
  fitted on train only" is weak. "Lab 4.2 fitted percentile bounds over the whole
  raster, so every test patch contributed to the transform applied to every training
  patch; the reported 0.6992 accuracy is an interpolation score" is teaching.
- Refer to real 2025/26 defects in the past tense as *course history*, never as
  "you might". Students respect specifics.
- No emoji in code cells. Sparse in markdown, only for status markers.
- Never say "simply", "obviously", or "trivially".

## 5. Hard technical rules

- **Never** print any part of a secret, even truncated. Print `'set'`/`'MISSING'`.
- Read credentials with `os.environ["NAME"]` so a missing one raises; do not use a
  silent default that lets a notebook run against the wrong account.
- `MPLCONFIGDIR` must be set **before** `import matplotlib`, via
  `paths.matplotlib_cache_dir()`.
- One `np.random.default_rng(seed)` per notebook; pass it down. No `np.random.*`.
- Every figure: `plt.savefig(...)` **and** `plt.show()`.
- No `!` line that depends on state from a previous `!` line. Use one
  `!bash -c "..."` per logical sequence, or `subprocess.run`.
- No hard-coded absolute paths. Everything comes from `eo_course.paths`.
- Long-running work: name the `slurm/` script that should run it and say what is
  wrong with doing it in the notebook (login-node I/O, GPU caps).
- If a cell needs data from a previous lab, use `paths.require_existing(...)` so the
  error names the lab that should have produced it.
- Keep code cells under ~40 lines. Split rather than scroll.

## 6. What "challenging" means here

Not more code to type. Decisions with measurable consequences the student must
justify:

- choose the held-out scenes and defend the choice;
- predict a metric, then explain the gap;
- pick an imbalance strategy *and* report the arms you did not pick;
- decide which rare classes to merge, and state how it changed the macro-average;
- report a result that failed a gate and diagnose it.

Every such decision must land in `results.json` (via `config` or `notes`) so it is
gradeable rather than asserted.
