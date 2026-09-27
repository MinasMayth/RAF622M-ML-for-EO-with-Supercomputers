# %% [markdown]
# # Lab 7 — Model evaluation: checking whether everything before it was true
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# This lab produces four things: a **baseline table scored on the same held-out test split
# as your model**, a set of **metrics with scene-clustered confidence intervals**, a
# **paired comparison between two real runs** that either survives seed noise or does not,
# and a **`results.json` record for the evaluation itself** that the grader's summariser can
# read. A gate board decides which claims you are allowed to write down.
#
# It assumes Labs 5.1, 5.2 and 6 left real artifacts on disk: raw test predictions written
# by `gates.gate_predictions_saved`, checkpoints under `paths.run_dir(run_id)/ckpt/`, and a
# `results.json` record carrying a `split_manifest_hash`. It produces the verdict the course
# report is built on.
#
# **You must run every cell yourself.** Generated notebooks ship with no outputs, on
# purpose. The 2025/26 version of this notebook had **zero executed cells and zero stored
# outputs**, and every number in its markdown was asserted rather than produced.

# %% [markdown]
# ## Why this notebook was rewritten from scratch
#
# The 2025/26 Lab 7 did not evaluate a model. Cell 8 *generated synthetic predictions* with
# "94% accuracy on background, 83% on minorities (**matching Lab 5 performance**)", and cells
# 9 and 17 then quoted Lab 5's numbers as ground truth. There was no model, no data, and no
# path from the notebook to any measurement. It could only ever confirm the number it started
# from — a self-confirming loop with no sensor, no checkpoint, and no possibility of finding
# a problem.
#
# Worse, the numbers it confirmed were themselves validation numbers. Lab 5.2's headline
# `Overall accuracy 0.9231 / Balanced accuracy 0.8338 / Macro-F1 0.8376` came from **455
# validation samples** on which five of ten classes had support ≤ 9 and two had support 1.
# Two misclassifications move that headline by roughly ±0.05. Those three numbers were then
# hard-typed into Lab 7 as "Lab 5 achieved these results on CORINE data" and used as the bar
# a submission had to clear. Beating them was free.
#
# > This notebook reads those numbers out of `results.json` instead of retyping them, and
# > compares **test against test**. If your record disagrees with a table on screen, the
# > record wins and the prose here is wrong.

# %% [markdown]
# ## The 2025/26 defects this notebook closes
#
# Each row names the old defect, what it cost, and the Part that fixes it.
#
# | old defect | consequence | fixed in |
# |---|---|---|
# | synthetic predictions "matching Lab 5 performance" | the lab could not find a problem; it tested whether its own assumption equalled itself | Part 3 |
# | zero executed cells, zero outputs, every number asserted | nobody knew, for a year, that the notebook measured nothing | Part 3, Part 12 |
# | Lab 5.2's 0.9231 / 0.8338 / 0.8376 hard-coded as ground truth | a 455-sample validation triple with two n=1 classes became the course's published result | Part 1, Part 9 |
# | `balanced_acc - majority_baseline` printed as "Model improvement" | a balanced accuracy minus a majority *accuracy*: two scales, meaningless difference, read as a marginal win | Part 6 |
# | `labels = np.unique(np.concatenate([y_true, y_pred]))` | a class the model never predicts silently left the macro-average, so a collapsed model could score well | Part 5 |
# | no baseline ever scored | Lab 6 shipped test accuracy exactly 1/10 — chance — against a majority baseline of 0.538, and the unit sheet was satisfied | Part 4 |
# | no uncertainty on any number | per-class recall printed to three decimals from n = 1–23 samples | Part 7 |
# | the only route to real evaluation was inside a `print("""…""")` | the documented deliverable "Comprehensive evaluation report" was unreachable | Part 3 |
# | four "scenarios" whose conclusion the code printed for you (`Key insight: …`) | two of the four printed conclusions were contradicted by their own tables | Part 6, Part 7 |
# | nothing written to disk, nothing asserted, no pass condition | a student could run all 32 cells and submit without evaluating anything | Part 12 |
# | "Class 4" meant a different CORINE class in each notebook | tables from two labs were not comparable | Part 3 |

# %% [markdown]
# ## Part 0 — Setup
#
# Three things happen here and the order matters.
#
# 1. `MPLCONFIGDIR` is set **before** matplotlib is imported. The 2025/26 notebooks set it
#    after the import, which does nothing, and shipped `Matplotlib created a temporary cache
#    directory …` in committed output.
# 2. **One** `np.random.default_rng(seed)` for the whole notebook, created here and passed
#    down. Every bootstrap and every random baseline receives `seed=SEED` explicitly.
# 3. Every path comes from `eo_course.paths`. No username appears anywhere in this file.

# %%
from eo_course import paths

print(paths.describe())

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = paths.matplotlib_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402

from eo_course import baselines as bl  # noqa: E402
from eo_course import gates, labels as lab_mod, metrics  # noqa: E402
from eo_course import patches as pat  # noqa: E402
from eo_course import paths, radiometry, results, splits  # noqa: E402

SEED = 0
rng = np.random.default_rng(SEED)
LAB = "lab7"
SOURCE_LABS = ("lab5.1", "lab5.2", "lab5", "lab6")
RARE_OTHER = 9001          # not a CORINE code, so it can never collide with one
print(f"  one rng, seed {SEED}; every bootstrap and random baseline takes seed=SEED")
print(f"  this lab writes records under lab='{LAB}'; it reads {SOURCE_LABS}")

# %% [markdown]
# ### Where the heavy work actually belongs
#
# Nothing in this notebook trains a model. It reads what Labs 5 and 6 produced, so the
# graded training runs belong in jobs, not here:
#
# ```bash
# sbatch slurm/train_jureca.sbatch          # one arm, one seed
# sbatch slurm/submit_sweep.sbatch          # the arm x seed grid
# ```
#
# What is wrong with training in a notebook on JURECA: heavy I/O is charged to a shared login
# node and will get you disconnected, there is no GPU there at all, and `module` is a shell
# function, which is why the 2025/26 `!source ...` lines failed under `/bin/sh`. Jupyter-JSC
# caps you at **one** GPU even when it allocated four. JURECA-DC partitions are `dc-cpu` and
# `dc-gpu`, lower case with hyphens; confirm live names with `sinfo`.

# %%
print(f"  on JURECA     : {paths.on_jureca()}")
print(f"  login node    : {paths.on_login_node()}")

import torch  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  torch {torch.__version__}  device for checkpoint inference: {DEVICE}")

def find_repo_root(start=None) -> Path:
    """Walk up from the notebook's cwd to the course checkout."""
    d = Path(start or Path().cwd()).resolve()
    for cand in (d, *d.parents):
        if (cand / "scripts" / "summarize_results.py").is_file():
            return cand
    raise FileNotFoundError(
        "scripts/summarize_results.py was not found at or above the current working "
        "directory. Start this notebook from the course repository root; the grader runs "
        "that script from there and your deliverable is checked by it.")


REPO = find_repo_root()
print(f"  repo root: {REPO}")
print("  Part 12 runs scripts/summarize_results.py --check from here and shows its exit code")

# %% [markdown]
# ## Part 1 — What the previous labs actually left you
#
# This is the step the 2025/26 notebook skipped, and skipping it is *why* it had to invent
# predictions: it never asked whether a checkpoint existed, so it never found out that none
# did.
#
# `results.json` is append-only and is the only artifact a grader can read without opening
# your kernel. `results.summary_table` exists so you print it rather than retype it — the
# 2025/26 Lab 7 retyped three numbers, and one of them was a validation score.

# %%
print(results.describe_contract())
ALL_RUNS = results.load_results()["runs"]
print(f"\n  ledger: {results.default_results_path()}")
print(f"  {len(ALL_RUNS)} record(s) total")
for _lab in SOURCE_LABS:
    print(f"    {_lab:<8} {len(results.runs_for_lab(_lab))} record(s)")

if not ALL_RUNS:
    raise RuntimeError(
        f"results.json is empty at {results.default_results_path()}. Lab 7 has nothing to "
        "evaluate. Run Lab 5.1, or scripts/train_cnn.py, which writes the same records. This "
        "notebook will not synthesise a stand-in: that is the defect it exists to remove.")

# %%
def predictions_path_for(rec):
    """Where a record's raw test predictions live, or None if they were never saved.

    Labs 5.1, 5.2 and scripts/train_cnn.py write `test_predictions.npz` into the run
    directory; Lab 6 stores the path it used in `extra["predictions"]`. Metrics are derived;
    predictions are the evidence, so a record without them cannot be audited.
    """
    p = (rec.get("extra") or {}).get("predictions")
    if isinstance(p, str) and Path(p).is_file():
        return Path(p)
    q = paths.run_dir(rec["run_id"]) / "test_predictions.npz"
    return q if q.is_file() else None


def checkpoint_path_for(rec):
    """The best checkpoint a run left behind, or None."""
    ck = sorted((paths.run_dir(rec["run_id"]) / "ckpt").glob("*.ckpt"))
    return ck[0] if ck else None


print(f"  {'run_id':<26} {'lab':<8} {'seed':>4} {'split':>8} {'preds':>6} {'ckpt':>5}")
for r in ALL_RUNS:
    print(f"  {r['run_id']:<26} {str(r.get('lab')):<8} {str(r.get('seed')):>4} "
          f"{str(r.get('split_manifest_hash'))[:8]:>8} "
          f"{str(predictions_path_for(r) is not None).lower():>6} "
          f"{str(checkpoint_path_for(r) is not None).lower():>5}")

# %% [markdown]
# Two columns above are load-bearing. A record with `preds=false` cannot be re-scored from
# evidence — you would have to trust the numbers already in it, which is exactly what Lab 7
# did in 2025/26. A record with `ckpt=false` cannot be re-derived at all.

# %% [markdown]
# ## Part 2 — The split of record
#
# Every number here is meaningless unless "the test split" names one specific thing. The
# manifest hash is what makes that checkable: it hashes the three index sets, so two runs
# share a test set if and only if their hashes match.
#
# The split of record is **derived from evidence, not chosen**. The course default is to hold
# out the alphabetically-last scenes so the whole class shares one test set and no team can
# select on it. Choosing your test scenes *after* seeing which gives the better number is
# test-set selection by another route, and Part 11 shows what a convenient split is worth.
#
# One property of the split matters for Part 7: `metrics.evaluate` bootstraps by scene, and a
# single-scene test set makes every scene-clustered interval degenerate — lower bound equal
# to upper bound, width exactly zero, an interval reporting a certainty it has not earned.

# %%
PATCH_DIR = paths.require_existing(
    paths.training_data_dir(),
    "Lab 4.2 patch archive directory (paths.training_data_dir())",
)
ARCHIVES = sorted(PATCH_DIR.glob(pat.SCENE_GLOB))
if not ARCHIVES:
    raise FileNotFoundError(
        f"no {pat.SCENE_GLOB} archives in {PATCH_DIR}. Lab 4.2 writes one archive per scene "
        "plus a sidecar carrying scene/row/col; without those no honest split exists.")
PS = pat.load_all_scenes(PATCH_DIR)
print(f"  {len(ARCHIVES)} archives, {len(PS)} patches, bands {list(PS.bands)}")
print(f"  scenes: {sorted(set(map(str, PS.scene)))}")
gates.gate_no_duplicate_patches(PS.patches, groups=PS.scene, seed=SEED)
print("  gate_no_duplicate_patches passed")

lo, hi = float(PS.patches.min()), float(PS.patches.max())
print(f"  stored range: [{lo:.4f}, {hi:.4f}]")
if hi <= 1.0 + 1e-3:
    UNITS, X_REFL = "reflectance", PS.patches
elif hi > 100.0:
    UNITS = "dn"
    X_REFL = radiometry.dn_to_reflectance(PS.patches)
    print(f"  -> raw DN converted once; X_REFL now spans "
          f"[{float(X_REFL.min()):.4f}, {float(X_REFL.max()):.4f}]")
else:
    raise ValueError(
        f"range [{lo:.4f}, {hi:.4f}] is neither reflectance nor DN — the archive was "
        "hand-stretched. Lab 5.1's bug was a second division on already-normalised data: the "
        "network saw values of order 1e-5, BatchNorm absorbed it, and nothing failed.")
radiometry.assert_reflectance_range(X_REFL[:4096], "X_REFL")
print(f"  units: {UNITS}; assert_reflectance_range passed")

# %% [markdown]
# ### Exercise 1 — predict the reconciliation before you run it
#
# > **P1.** Look at the run table in Part 1 and, before running the next three cells, write
# > down:
# >
# > 1. How many of your records you expect to share the split hash you are about to rebuild.
# > 2. Whether Lab 6's record (if you have one) shares it.
# > 3. Whether the three seeds of one Lab 5 arm share it.
# >
# > Question 3 is the trap. If a lab built its split with `test_scenes=None` and passed the
# > run seed into `group_block_split`, then each seed got a **different test set**, and the
# > "seed spread" it reports is partly split spread. Record predicted vs actual and explain
# > any disagreement by naming the code path that caused it.

# %%
P1 = {
    # TODO(you): fill in BEFORE running the next three cells.
    "records_sharing_split_of_record": None,
    "lab6_shares_split": None,        # True / False / "no lab6 record"
    "lab5_seeds_share_split": None,   # True / False
    "why": "TODO(you): name the argument that decides the answer to question 3.",
}

# %%
HASH_COUNTS: dict[str, int] = {}
for r in ALL_RUNS:
    h = str(r.get("split_manifest_hash"))
    HASH_COUNTS[h] = HASH_COUNTS.get(h, 0) + 1
TOP_HASH = max(HASH_COUNTS, key=lambda k: HASH_COUNTS[k])
print("  split hashes present in the ledger:")
for h, n in sorted(HASH_COUNTS.items(), key=lambda kv: -kv[1]):
    print(f"    {h:<18} {n} record(s)")
print(f"  most common: {TOP_HASH} — the split of record is the one your results were made on")

MF_FILE = paths.splits_dir() / f"split_{TOP_HASH}.json"
if MF_FILE.is_file():
    recorded = splits.SplitManifest.read(MF_FILE)
    TEST_SCENES = tuple(recorded.extra.get("test_scenes", ()))
    SPLIT_SEED = int(recorded.seed)
    print(f"  manifest file {MF_FILE.name}: test_scenes={list(TEST_SCENES)} seed={SPLIT_SEED}")
    print("  read from disk, not typed: the split your records were produced on is a fact,")
    print("  not a choice you get to revisit after seeing a number")
else:
    TEST_SCENES = tuple(sorted(set(map(str, PS.scene)))[-2:])
    SPLIT_SEED = 0
    print(f"  no manifest file for {TOP_HASH} under {paths.splits_dir()};")
    print(f"  falling back to the course default test scenes {list(TEST_SCENES)}. Re-run the")
    print("  producing lab so the manifest is on disk, or the hash in your record is a claim")
    print("  with nothing behind it.")

manifest = splits.group_block_split(PS.scene, PS.row, PS.col, block=10, val_ratio=0.15,
                                    test_scenes=list(TEST_SCENES), seed=SPLIT_SEED)
if MF_FILE.is_file():
    recorded.verify_against(manifest)
    print(f"  rebuilt manifest matches the recorded hash: {manifest.manifest_hash}")

# %%
paths.ensure(paths.splits_dir())
MANIFEST_PATH = manifest.write(paths.splits_dir() / f"split_{manifest.manifest_hash}.json")
gates.gate_split_is_grouped(manifest)
gates.gate_split_disjoint(manifest)
sm = manifest.summary()
train_idx, val_idx, test_idx = manifest.train_idx, manifest.val_idx, manifest.test_idx
print(f"  split of record {manifest.manifest_hash}  method={manifest.method}")
print(f"  n train/val/test {sm['n_train']} / {sm['n_val']} / {sm['n_test']} of {sm['n_total']}")
print(f"  test scenes {sm['test_scenes']} ({sm['n_scenes_test']} scene(s))   "
      f"val scenes {sm['val_scenes']}")
if sm["n_scenes_test"] < 2:
    print("  WARNING: one held-out scene makes the Part 7 scene-clustered bootstrap")
    print("  degenerate. Acquire scenes; do not switch to an i.i.d. bootstrap to make it move.")

print(f"  {'run_id':<26} {'lab':<8} {'recorded hash':>14} {'match':>6}  reason if not")
COMPARABLE = []
for r in ALL_RUNS:
    got = r.get("split_manifest_hash")
    ok = got == manifest.manifest_hash
    reason = "" if ok else (
        "recorded hash is a different split: different held-out scenes, or the split seed "
        "varied with the run seed" if got else "record carries no split hash at all")
    print(f"  {r['run_id']:<26} {str(r.get('lab')):<8} {str(got)[:14]:>14} "
          f"{('yes' if ok else 'NO'):>6}  {reason}")
    if ok:
        COMPARABLE.append(r)
print(f"\n  {len(COMPARABLE)} of {len(ALL_RUNS)} records were made on the split of record")
print(f"  P1 answers were: {P1}")

# %% [markdown]
# ### Why a seed-dependent split is a defect and not a detail
#
# A "three-seed" result measures run-to-run noise only if the three runs saw the same data.
# If the split moved, the spread you report as seed noise contains a change of test set, and
# every claim calibrated against it is calibrated against the wrong quantity. The cell below
# builds the split three times with `test_scenes=None` and different seeds, which is what the
# 2025/26 notebooks did by default.

# %%
alt = [splits.group_block_split(PS.scene, PS.row, PS.col, block=10, val_ratio=0.15,
                                test_scenes=None, seed=s) for s in (0, 1, 2)]
print(f"  {'seed':>4} {'manifest hash':>16} {'test index hash':>16}  test scenes")
for s, m in zip((0, 1, 2), alt):
    print(f"  {s:>4} {m.manifest_hash:>16} {m.index_hash('test_idx'):>16}  "
          f"{m.summary()['test_scenes']}")
print(f"\n  identical test index sets across seeds: "
      f"{len({m.index_hash('test_idx') for m in alt}) == 1}")
print("  If that is False, a seed sweep under that recipe is also a test-set sweep. The fix")
print("  is the argument you passed above: pin test_scenes, and the seed cannot move them.")

# %% [markdown]
# ## Part 3 — Real predictions, or an error that names the lab
#
# This is the cell the 2025/26 notebook did not have. It loads the raw `y_true` / `y_pred`
# arrays the previous labs saved and, if they are not there, raises an error naming the lab
# that should have produced them. It never, under any condition, generates a stand-in.
#
# Why this matters more than it looks: the old synthetic generator was parameterised *by the
# number it was supposed to verify*. Porting its cell 8 verbatim (`seed=42`, `n_background=800`,
# 20 samples per minority class) and scoring it with this course's metric code gives accuracy
# 0.9286 / balanced accuracy 0.8596 / macro-F1 0.7850 — which reproduces **none** of the
# 0.9231 / 0.8338 / 0.8376 it claimed to match. A fabricated input cannot even agree with the
# numbers it was fitted to, and a reader who ran this notebook would have had no way to notice.

# %%
def load_predictions(rec):
    """Load a run's saved test predictions, by key, or raise naming the lab."""
    p = predictions_path_for(rec)
    if p is None:
        raise paths.CoursePathError(
            f"no raw test predictions for run {rec['run_id']!r}. Labs 5.1, 5.2 and 6 write "
            f"them with gates.gate_predictions_saved into {paths.run_dir(rec['run_id'])}. "
            "Re-run the lab that produced this record and confirm it printed 'recorded'. This "
            "notebook will not synthesise a replacement — the 2025/26 version did, and that "
            "is the defect this rewrite exists to remove.")
    with np.load(paths.require_existing(p, f"test predictions of {rec['run_id']}")) as d:
        gates.gate_npz_loaded_by_key(d, required=("y_true", "y_pred"))
        return d["y_true"].copy(), d["y_pred"].copy()


def label_map_of(rec):
    """The class set a record was scored on, read from the record itself.

    Saved predictions are indices. Index 4 was CORINE 12 in Lab 5.1 and CORINE 20 in Lab
    5.2, and "Class 4" in the old Lab 7 — which is why three notebooks printed per-class
    tables that could not be compared. Carry the codes, never the indices.
    """
    codes = (rec.get("config") or {}).get("codes")
    if codes:
        return lab_mod.LabelMap(tuple(int(c) for c in codes))
    return None

# %%
TRAIN_CODES = sorted({int(c) for c in PS.labels[train_idx]})
# The fixed label set is the union of the train class set and every class set a recorded
# run was scored on, so a saved prediction array can always be decoded into it. Classes
# that never appear in train stay in the set with zero training support — deleting them
# would shrink the problem to fit the result, which is the defect Part 5 is about.
ALL_CODES = set(TRAIN_CODES)
for _r in ALL_RUNS:
    ALL_CODES |= {int(c) for c in ((_r.get("config") or {}).get("codes") or [])}
LM = lab_mod.LabelMap(tuple(sorted(ALL_CODES)))
LABELS = np.arange(LM.n_classes)
NAMES = list(LM.names)
if RARE_OTHER in ALL_CODES:
    NAMES[LM.index_of(RARE_OTHER)] = "rare-other (merged upstream)"
print(f"  fixed label set for every metric call in this lab: {LABELS.tolist()}")
print(f"  K = {LM.n_classes}   codes {list(LM.codes)}")

train_sup = lab_mod.class_counts(PS.labels[train_idx], LM.codes)
test_sup = lab_mod.class_counts(PS.labels[test_idx], LM.codes)
DROP_CODES = [int(c) for c, n in zip(LM.codes, train_sup) if n == 0]
print(f"  {'code':>5} {'class':<32} {'L1':<17} {'train':>6} {'test':>5}  note")
for i, c in enumerate(LM.codes):
    nm = (NAMES[i][:32] if c == RARE_OTHER else lab_mod.class_name(c)[:32])
    l1 = "merged bucket" if c == RARE_OTHER else lab_mod.level1_of(int(c))[:17]
    note = "NO TRAIN SUPPORT: unlearnable" if train_sup[i] == 0 else ""
    print(f"  {c:>5} {nm:<32} {l1:<17} {train_sup[i]:>6} {test_sup[i]:>5}  {note}")
print(f"  codes with zero training support: {DROP_CODES}")
print("  No loss weighting can predict a label the model never saw. They stay in the fixed")
print("  label set — deleting them would shrink the problem to fit the result — and Part 10")
print("  merges them on TRAIN support and reports what that did to the macro-average.")

EVAL_TEST_IDX = test_idx[np.isin(PS.labels[test_idx], TRAIN_CODES)]
print(f"  test patches scored: {EVAL_TEST_IDX.size} of {test_idx.size}")
if EVAL_TEST_IDX.size != test_idx.size:
    print(f"  EXCLUDED {test_idx.size - EVAL_TEST_IDX.size} test patches whose CORINE code "
          "never appears in train.")
    print("  This is a split-design failure, not an evaluation choice: LabelMap.encode with")
    print("  strict=True refuses such codes rather than dropping them quietly. Fix it")
    print("  upstream by acquiring or rebalancing scenes, not by shrinking the problem until")
    print("  the metric looks reasonable.")

# %%
PRED = {}
for r in COMPARABLE:
    try:
        PRED[r["run_id"]] = (load_predictions(r), label_map_of(r) or LM)
    except paths.CoursePathError as exc:
        print(f"  SKIP {r['run_id']}: {exc}")
print(f"\n  {len(PRED)} run(s) re-scorable from raw evidence on the split of record")
if not PRED:
    raise paths.CoursePathError(
        "no record on the split of record has saved predictions. Re-run Lab 5 with "
        f"test_scenes={list(TEST_SCENES)}, or rebuild the split of record from the manifest "
        "the runs actually used. Do not proceed with invented arrays.")

# %% [markdown]
# ### Which predictions belong to this split?
#
# Restrict to runs whose saved targets, decoded to CORINE codes, equal the split of record's
# test codes in order. This is the guard that stops you scoring a run against the wrong
# labels — a mistake that prints a perfectly plausible number and no warning, and which the
# 2025/26 notebooks could not have detected because they never had labels to check against.

# %%
EXPECTED_CODES = PS.labels[EVAL_TEST_IDX]
EXPECTED_LABELS = LM.encode(EXPECTED_CODES)
USABLE: dict[str, dict] = {}
for rid, ((yt, yp), lm_r) in PRED.items():
    codes_t = lm_r.decode(yt)
    n = min(codes_t.size, EXPECTED_CODES.size)
    if codes_t.size == EXPECTED_CODES.size and np.array_equal(codes_t, EXPECTED_CODES):
        codes_p = lm_r.decode(yp)
        USABLE[rid] = {"y": EXPECTED_LABELS, "p": LM.encode(codes_p), "lm": lm_r,
                       "saved": yp, "pred_codes": codes_p}
    else:
        print(f"  EXCLUDED {rid}: saved targets are not the split of record's test codes "
              f"({codes_t.size} vs {EXPECTED_CODES.size} samples, "
              f"{int((codes_t[:n] != EXPECTED_CODES[:n]).sum())} mismatched)")
print(f"\n  {len(USABLE)} run(s) usable for every comparison below")
print("  every y / p pair below lives in ONE index space, LM's, rebuilt through CORINE codes.")
print("  Index 4 was CORINE 12 in Lab 5.1 and CORINE 20 in Lab 5.2; codes do not drift.")
RUN_IDS = sorted(USABLE)
print(f"  {RUN_IDS}")

# %% [markdown]
# ### The round-trip: predictions must be reproducible from the checkpoint
#
# Saved predictions are evidence only if the checkpoint on disk regenerates them. If they
# disagree, one of the two is stale — usually the npz, written before a retrain overwrote the
# checkpoint. This check is what turns "I have a file" into "I have evidence".

# %%
NORM_DICT = next((r["config"]["normalisation"] for r in COMPARABLE
                  if (r.get("config") or {}).get("normalisation")), None)
if NORM_DICT:
    norm = radiometry.Norm.from_dict(NORM_DICT)
    print(f"  normalisation restored from a record: {norm}")
    print("  restored, not refitted: the transform the model saw is part of its definition")
else:
    norm = radiometry.Norm.fit(PS.patches[train_idx], mode="minmax", channel_axis=-1)
    print(f"  no normalisation in any record; refitting on train only: {norm}")

# A Norm is fitted in a specific unit system, and apply() does not know which. Feeding it
# an array in different units is Lab 5.1's bug wearing a different hat: it divided an
# already-normalised array by 10 000, the network saw values of order 1e-5, BatchNorm
# absorbed it, and nothing failed for a whole academic year.
X_MODEL = norm.apply(PS.patches)
gates.gate_input_units(X_MODEL, "recorded Norm applied to the stored patches")
print(f"  X_MODEL {X_MODEL.shape} range [{X_MODEL.min():.4f}, {X_MODEL.max():.4f}]")
print(f"  applied to PS.patches, the array the record's Norm was fitted on ({UNITS} stored)")

if UNITS == "dn":
    try:
        gates.gate_input_units(norm.apply(X_REFL), "recorded Norm applied to reflectance")
        print("  UNREACHABLE: a double division passed the units gate.")
    except gates.GateFailure as exc:
        print("  NEGATIVE CONTROL — the same Norm, applied to an array already in reflectance:")
        print(f"    {str(exc).splitlines()[0][:120]}")
    print("  That is the 2025/26 defect reproduced in one line, and the gate is what makes it")
    print("  loud instead of silent. Restore a transform onto the units it was fitted on.")

x_all = np.ascontiguousarray(np.transpose(X_MODEL, (0, 3, 1, 2)))
X_STORED_LAYOUT = np.ascontiguousarray(X_MODEL)
print(f"  x_all (N,C,H,W) {x_all.shape} — the layout the bands actually mean")
print(f"  X_STORED_LAYOUT (N,H,W,C) {X_STORED_LAYOUT.shape} — the layout on disk")
print("  A checkpoint records a channel COUNT, not a channel AXIS. If the code that trained")
print("  it guessed the axis, the weights are defined on the wrong view of the array, and")
print("  only the count is left to tell you. infer() below reads the count and says which")
print("  layout it had to feed to match it.")

# %%
from eo_course.training import CorineModule  # noqa: E402


@torch.no_grad()
def infer(ckpt, idx, batch=512):
    """Run one checkpoint over the samples at ``idx``. No dataloader, no shuffle.

    Architecture comes from the checkpoint's own hyperparameters. The channel axis is
    chosen to match the channel COUNT it recorded, because that is the only part of the
    input contract a checkpoint actually stores.
    """
    module = CorineModule.load_from_checkpoint(str(ckpt), train_counts=None)
    want = int(module.hparams["in_channels"])
    layout = "NCHW" if x_all.shape[1] == want else (
        "NHWC" if X_STORED_LAYOUT.shape[1] == want else None)
    if layout is None:
        raise ValueError(
            f"checkpoint {Path(ckpt).name} expects {want} channels; the data offers "
            f"{x_all.shape[1]} (NCHW) or {X_STORED_LAYOUT.shape[1]} (NHWC). Refusing to "
            "guess: a wrong channel axis trains and scores, and means nothing.")
    arr = x_all if layout == "NCHW" else X_STORED_LAYOUT
    module.to(DEVICE).eval()
    out = []
    for s in range(0, len(idx), batch):
        xb = torch.from_numpy(arr[idx[s:s + batch]])
        out.append(module(xb.to(DEVICE)).argmax(1).cpu().numpy())
    del module
    return np.concatenate(out).astype(np.int64), layout


print("  infer() defined. It returns the layout it used, so a mismatch is visible in the")
print("  output rather than absorbed by BatchNorm, which is exactly how the 2025/26 run")
print("  hid a unit error for a year.")

# %%
print(f"  {'run_id':<26} {'saved == regenerated':>19} {'n':>6}")
ROUNDTRIP = {}
LAYOUTS = {}
for rid in RUN_IDS:
    rec = next(r for r in COMPARABLE if r["run_id"] == rid)
    lm_r = USABLE[rid]["lm"]
    ck = checkpoint_path_for(rec)
    if ck is None:
        print(f"  {rid:<26} {'no checkpoint':>19} {USABLE[rid]['y'].size:>6}")
        continue
    if int((rec.get("config") or {}).get("n_classes", lm_r.n_classes)) != lm_r.n_classes:
        print(f"  {rid:<26} {'class-set mismatch':>19} {'-':>6}")
        continue
    # Compare in the checkpoint's OWN index space: both sides are argmax indices of the
    # same architecture. Translating through codes first would hide an off-by-one head.
    regen, layout = infer(ck, EVAL_TEST_IDX)
    LAYOUTS[rid] = layout
    saved = USABLE[rid]["saved"]
    agree = bool(np.array_equal(regen, saved))
    ROUNDTRIP[rid] = agree
    print(f"  {rid:<26} {str(agree):>19} {saved.size:>6}  fed as {layout}")
    if not agree:
        print(f"    -> {int((regen != saved).sum())} of {saved.size} predictions differ. "
              "The npz and the checkpoint are not the same run.")

print(f"  round-trip verified for {sum(ROUNDTRIP.values())} of {len(ROUNDTRIP)} checkable runs")
print("  An agreeing round-trip means a grader can rebuild every number in your report from")
print("  two files. That is the whole difference between a result and an assertion. A")
print("  disagreeing one means you must work out which artifact is stale before you write")
print("  anything down — and 'the nicer-looking one' is not a criterion.")

# %%
_bad = {r: l for r, l in LAYOUTS.items() if l != "NCHW"}
print(f"  bands in the archive : {list(PS.bands)}  ({len(PS.bands)} channels)")
print(f"  layouts the checkpoints required: {sorted(set(LAYOUTS.values())) or '(none)'}")
if _bad:
    print(f"  MISMATCH: {len(_bad)} checkpoint(s) were trained on the (N,H,W,C) view of "
          f"(N,H,W,{len(PS.bands)}) data, i.e. {X_STORED_LAYOUT.shape[1]} 'channels' of which")
    print(f"  the first {len(PS.bands)} are image ROWS. The first conv filter's weights are")
    print("  indexed against the wrong axis, so the model is a learned function of a fixed")
    print("  slice of the patch rather than of its spectra. It still trains, and BatchNorm")
    print("  keeps it numerically healthy, which is why nothing failed.")
    print("  This is a defect in the producing code, not in your evaluation: report it, name")
    print("  the runs affected, and do not present their scores as band-informed.")
    print("  TODO(you): name the runs and say what you would need to retrain to fix it.")
else:
    print("  every checkpoint consumed the (N,C,H,W) view, so its filters are indexed")
    print("  against bands. State this in your report — it is a property of the artifact you")
    print("  are grading, and nothing in a results record asserts it.")

# %% [markdown]
# ## Part 4 — Baselines, on the same split, with the same metric code
#
# The 2025/26 labs printed a majority-class *count*, called it a baseline, and never scored it
# with the same code as the model. Lab 6 then shipped a fine-tuned Prithvi with test accuracy
# `0.10000000149011612` — exactly 1/10, chance for ten classes — while the majority-class
# baseline on that same 455-sample test set was 245/455 = **0.538**. The model was worse than
# predicting one label and nothing in the notebook could catch it.
#
# `baselines.run_all` scores six trivial predictors through `eo_course.metrics.evaluate`, on
# the same test split with the same fixed label set as your model. That sameness is the entire
# point: a baseline scored by different code is not a baseline.
#
# | baseline | what it is | what it exposes |
# |---|---|---|
# | `majority` | always the most frequent training class | the accuracy floor of a constant predictor |
# | `uniform_random` | uniform guess | the balanced-accuracy floor 1/K |
# | `prior_random` | samples from the training class prior | what memorising the marginal buys |
# | `per_scene_majority` | most frequent class *within each scene* | how much a per-site constant gets you |
# | `ndvi` | two NDVI thresholds mapped onto the class set | a free, interpretable physics rule |
# | `linear_probe` | multinomial logistic regression on flattened patches | how much of the task is linear |

# %% [markdown]
# ### Exercise 2 — predict the floor
#
# > **P2.** Before running the baseline table, write down:
# >
# > 1. The **balanced accuracy** of `uniform_random`, to two decimals. It is a function of K
# >    alone.
# > 2. The **overall accuracy** of `majority`, to two decimals. It is the majority class's
# >    share of the *test* split, not of the training split.
# > 3. Which of the six will have the **highest macro-F1**.
# >
# > Question 3 decides your target, because `gate_beats_baselines` compares you against the
# > *winner*, not the easiest one. On small-patch data the winner is frequently not the one
# > people expect.

# %%
P2 = {
    # TODO(you): fill in BEFORE running the next cell.
    "uniform_random_bal_acc": None,
    "majority_overall_acc": None,
    "best_macro_f1_baseline": None,   # one of the six names
    "why": "TODO(you): why that one, at this patch size with these bands?",
}

# %%
BASELINES = bl.run_all(
    x_train=x_all[train_idx], y_train_codes=PS.labels[train_idx],
    x_test=x_all[EVAL_TEST_IDX], y_test_codes=PS.labels[EVAL_TEST_IDX],
    labels=LABELS, codes=LM.codes,
    train_scene_ids=PS.scene[train_idx], test_scene_ids=PS.scene[EVAL_TEST_IDX],
    seed=SEED)
BEST_BASE = max(BASELINES, key=lambda k: BASELINES[k]["macro_f1"])
print(f"  {'baseline':<20} {'acc':>8} {'bal_acc':>8} {'macro_F1':>9} {'kappa':>8}")
for name in sorted(BASELINES, key=lambda k: -BASELINES[k]["macro_f1"]):
    d = BASELINES[name]
    print(f"  {name:<20} {d['overall_acc']:>8.4f} {d['balanced_acc']:>8.4f} "
          f"{d['macro_f1']:>9.4f} {d['kappa']:>8.4f}")
print(f"\n  CHANCE FLOOR       : balanced_acc = 1/K = {1.0 / LM.n_classes:.4f}")
print(f"  STRONGEST BASELINE : {BEST_BASE}  macro-F1 "
      f"{BASELINES[BEST_BASE]['macro_f1']:.4f}  balanced_acc "
      f"{BASELINES[BEST_BASE]['balanced_acc']:.4f}")
print(f"  YOUR TARGETS       : macro-F1 > "
      f"{BASELINES[BEST_BASE]['macro_f1'] + gates.MIN_MACRO_F1_OVER_BASELINE:.4f}   "
      f"balanced_acc > "
      f"{BASELINES[BEST_BASE]['balanced_acc'] + gates.MIN_BALANCED_ACC_OVER_CHANCE:.4f}")
print(f"  P2 answers were    : {P2}")

fig, ax = plt.subplots(figsize=(8, 3.6))
_names = sorted(BASELINES, key=lambda k: BASELINES[k]["macro_f1"])
ax.barh(_names, [BASELINES[n]["macro_f1"] for n in _names],
        color=["#c0392b" if n == BEST_BASE else "#7f8c8d" for n in _names], label="macro-F1")
ax.barh(_names, [BASELINES[n]["balanced_acc"] for n in _names],
        color="#2980b9", alpha=0.55, label="balanced accuracy")
ax.axvline(1.0 / LM.n_classes, ls="--", c="k", lw=1,
           label=f"1/K = {1.0 / LM.n_classes:.3f}")
ax.set_xlabel("score on the split of record")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.set_title("the bar every model number in this lab is measured against")
fig.tight_layout()
FIG_BASE = paths.ensure(paths.results_dir())[0] / "lab7_baselines.png"
fig.savefig(FIG_BASE, dpi=110)
plt.show()
print(f"  saved {FIG_BASE}")

# %% [markdown]
# ## Part 5 — The fixed label set, and what `np.unique` actually cost you
#
# The 2025/26 metric code began with
#
# ```python
# labels = np.unique(np.concatenate([y_true, y_pred]))
# ```
#
# and then averaged the recalls that came back. The class set — the denominator of every
# macro-average, the size of the confusion matrix, and the 1/K that "above chance" is measured
# against — was chosen by the model's own predictions rather than by the problem.
#
# It is worth being precise about the damage, because the usual summary ("it inflates the
# score") is not always true, and a defect description that is wrong gets dismissed. What is
# always true is that **the metric became a function of the predictions as well as of the
# model**, so:
#
# * a class absent from the evaluation slice silently leaves the report — the confusion matrix
#   shrinks, the per-class table loses a row, and nobody sees a row go missing;
# * the chance floor `1/K` moves with it, so every "clears chance by X" claim is measured
#   against a bar the model set;
# * Cohen's κ changes, because κ depends on the label set;
# * and the score itself can move in **either** direction depending on whether the missing
#   class shows up in the predictions, which is the worst property a metric can have.
#
# `metrics.evaluate` takes `labels` as a required argument and **raises** if a prediction or a
# target falls outside it. The set here is `np.arange(K)` fixed in Part 3, so it cannot move
# between calls. Where a class genuinely has no test support, `evaluate` still keeps it in the
# matrix, prints `NO SUPPORT` in the table, and lets `gate_class_support` fail on it — the
# difference is that the omission is declared and gated rather than invisible.

# %%
def old_lab7_metrics(y_true, y_pred):
    """The 2025/26 metric code, transcribed faithfully from lab7 cells 5-6.

    labels = np.unique(concat(y_true, y_pred)); balanced_acc = recall.mean() and
    macro_f1 = f1.mean() over whatever rows that produced. The denominator is therefore
    chosen by the model's own predictions, not by the problem.
    """
    lab = np.unique(np.concatenate([y_true, y_pred]))
    cm = metrics.confusion(y_true, y_pred, lab)
    sup = cm.sum(axis=1).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    recall = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros_like(tp),
                   where=(prec + recall) > 0)
    return {"K": int(lab.size), "labels": lab.tolist(), "balanced_acc": float(recall.mean()),
            "macro_f1": float(f1.mean()), "floor": 1.0 / lab.size, "cm_shape": cm.shape}

# %% [markdown]
# ### Exercise 3 — predict what the derived label set loses
#
# > **P3.** Score your test predictions on a slice where one class is genuinely absent — from
# > the targets *and* the predictions — twice: once with the 2025/26 derived label set, once
# > with the fixed set. Then answer, before running:
# >
# > 1. How many classes does the derived set lose on that slice?
# > 2. Which class is it, by CORINE code?
# > 3. For each of three predictors — your model, a constant predictor, and one that sprays
# >    the absent class everywhere — is the derived balanced accuracy higher, lower or equal
# >    to the fixed one? Give three answers, not one.
# >
# > Question 3 is the one that separates reading this part from doing it. Most people give the
# > same answer three times, on the theory that dropping a hard class must flatter the model.
# > If the three come out differently, the metric's bias is being set by the predictions rather
# > than by the model, and no score computed with it is comparable to any other.

# %%
P3 = {
    # TODO(you): fill in BEFORE running the next cell.
    "classes_lost": None,
    "which_code": None,
    "your_model": None,               # "higher" / "lower" / "equal"
    "collapsed": None,
    "sprays_absent_class": None,
}

# %%
K = LM.n_classes
_y0, _p0 = USABLE[RUN_IDS[0]]["y"], USABLE[RUN_IDS[0]]["p"]
# Build a slice in which one class is genuinely absent — from the targets AND the
# predictions — which is what a held-out scene containing none of that class does. On the
# full test slice every class appears somewhere, derived and fixed coincide, and the bug is
# invisible. That is the case that hid it for a year.
_present = sorted({int(c) for c in _y0})
_absent = min(_present, key=lambda c: int((_y0 == c).sum()))
_slice = np.flatnonzero((_y0 != _absent) & (_p0 != _absent))
_y1, _p1 = _y0[_slice], _p0[_slice]
print(f"  full slice   : n={_y0.size}, classes present {len(_present)}")
print(f"  reduced slice: n={_y1.size}; class index {_absent} (code {int(LM.codes[_absent])}, "
      f"{int((_y0 == _absent).sum())} samples) absent from targets and predictions alike")

old = old_lab7_metrics(_y1, _p1)
new = metrics.evaluate(_y1, _p1, LABELS, codes=LM.codes, names=NAMES, n_boot=0)
lost = sorted(set(int(i) for i in LABELS) - set(old["labels"]))
print(f"\n  fixed label set   : K={K}  codes {list(LM.codes)}")
print(f"  derived label set : K={old['K']}  indices {old['labels']}")
print(f"  silently lost     : class indices {lost} -> codes "
      f"{[int(LM.codes[i]) for i in lost]}")
print(f"  confusion matrix  : derived {old['cm_shape']}  vs  fixed {new.confusion.shape}")
print(f"\n  {'metric':<16} {'derived (2025/26)':>18} {'fixed (this lab)':>17} {'delta':>9}")
print(f"  {'balanced_acc':<16} {old['balanced_acc']:>18.4f} {new.balanced_acc:>17.4f} "
      f"{old['balanced_acc'] - new.balanced_acc:>+9.4f}")
print(f"  {'macro-F1':<16} {old['macro_f1']:>18.4f} {new.macro_f1:>17.4f} "
      f"{old['macro_f1'] - new.macro_f1:>+9.4f}")
print(f"  {'chance floor 1/K':<16} {old['floor']:>18.4f} {new.floor_balanced_acc():>17.4f} "
      f"{old['floor'] - new.floor_balanced_acc():>+9.4f}")
_kd = metrics.evaluate(_y1, _p1, np.array(old["labels"]), n_boot=0).kappa
print(f"  {'kappa':<16} {_kd:>18.4f} {new.kappa:>17.4f} {_kd - new.kappa:>+9.4f}")
print(f"\n  P3 answers were: {P3}")

# %%
# Three regimes, because the direction of the error is the finding.
_absent_idx = _absent
_major = int(np.bincount(_y1).argmax())
cases = {
    "your model": _p1,
    "collapsed (one constant class)": np.full(_y1.size, _major, dtype=np.int64),
    "sprays the absent class everywhere": np.where(
        rng.random(_y1.size) < 0.5, _absent_idx, _p1),
}
print(f"  {'case':<36} {'K_d':>4} {'bal_d':>7} {'bal_f':>7} {'d':>8} "
      f"{'mf1_d':>7} {'mf1_f':>7} {'floor_d':>8} {'floor_f':>8}")
for nm, pred in cases.items():
    o = old_lab7_metrics(_y1, pred)
    n = metrics.evaluate(_y1, np.asarray(pred, dtype=np.int64), LABELS, n_boot=0)
    print(f"  {nm:<36} {o['K']:>4} {o['balanced_acc']:>7.4f} {n.balanced_acc:>7.4f} "
          f"{o['balanced_acc'] - n.balanced_acc:>+8.4f} {o['macro_f1']:>7.4f} "
          f"{n.macro_f1:>7.4f} {o['floor']:>8.4f} {n.floor_balanced_acc():>8.4f}")
print(f"\n  K_fixed = {K}; the derived K changed with the predictions, case by case.")
print("  Three things to read off that table.")
print("  1. The derived K moves with the predictions, so the chance floor 1/K moves with it.")
print("     'Clears chance by X' is then measured against a bar the model set for itself.")
print("  2. The score deltas go BOTH ways — zero for your model, negative when the absent")
print("     class is sprayed in. A metric whose bias direction is chosen by its own input")
print("     cannot be compared across models, across runs, or across labs.")
print("  3. The collapsed row is the subtle one. On a slice missing one class a constant")
print("     predictor scores 1/(K-1), and the derived floor is also 1/(K-1), so the old code")
print("     reports 'exactly chance' about a model that predicts one class. The fixed floor")
print("     is 1/K, so this lab reports 'above chance'. Both readings are wrong about what")
print("     the model actually is, which is the real lesson of this Part.")

# %%
# What actually catches a collapsed predictor is not a better label set. It is a gate that
# looks at the prediction histogram, which no score on that table can see.
_nc = metrics.evaluate(_y1, np.full(_y1.size, _major, dtype=np.int64), LABELS,
                       codes=LM.codes, names=NAMES, n_boot=0)
print("  GATES ON THE COLLAPSED PREDICTOR — the check the 2025/26 notebook never ran")
for nm, fn in (("above_chance", gates.gate_above_chance),
               ("not_collapsed", gates.gate_not_collapsed),
               ("class_support", gates.gate_class_support)):
    try:
        fn(_nc)
        print(f"    [PASS] {nm}: accepted a constant predictor")
    except gates.GateFailure as exc:
        print(f"    [FAIL] {nm}: {str(exc).splitlines()[0][:105]}")
print("  A constant predictor survives every label-set argument and fails a prediction-")
print("  histogram check. That is why this lab runs both: evaluate() fixes the denominator,")
print("  the gates ask what the model actually did. Lab 7 in 2025/26 had neither.")

try:
    metrics.evaluate(_y0, np.full(_y0.size, K - 1, dtype=np.int64), LABELS[: K - 1], n_boot=0)
    print("  UNREACHABLE: evaluate() accepted a prediction outside the label set it was given.")
    print("  If you ever see this line, the guard is broken and every label-set claim in this")
    print("  course is void.")
except ValueError as exc:
    print("  evaluate() refuses a widened problem instead of silently growing the label set:")
    print(f"    {str(exc).splitlines()[0][:110]}")
print("  That is the structural fix. The old code could not refuse, because it had no fixed")
print("  set to refuse against — it asked the predictions what the classes were.")

# %% [markdown]
# ## Part 6 — The subtraction that means nothing
#
# The 2025/26 Lab 7 cell was:
#
# ```python
# majority_baseline = cnts.max() / len(y_true)
# print(f"Model improvement:  {balanced_acc - majority_baseline:.4f}")
# ```
#
# On its own synthetic data that printed **+0.0427**, which reads as a narrow but real win. It
# is not a narrow win, it is a category error. `balanced_acc` is the unweighted mean of
# per-class recalls. `cnts.max() / len(y_true)` is the *overall accuracy* of a constant
# predictor. They live on different scales and their difference has no interpretation. The
# honest comparisons are accuracy-to-accuracy and balanced-accuracy-to-1/K, and there are
# exactly three of them.

# %% [markdown]
# ### Exercise 4 — predict the two floors
#
# > **P4.** For your own test split, before running:
# >
# > 1. `Metrics.majority_rate()` — the overall accuracy of always predicting the most common
# >    test class.
# > 2. `Metrics.floor_balanced_acc()` — the balanced accuracy of that same constant predictor.
# > 3. Your model's `balanced_acc - majority_rate()`. Is that number interpretable? Answer yes
# >    or no and give the reason, not a feeling.
# >
# > Question 3 is graded on the reason. "It is my improvement" is not a reason.

# %%
P4 = {
    # TODO(you): fill in BEFORE running the next cell.
    "majority_rate": None,
    "floor_balanced_acc": None,
    "subtraction_interpretable": None,   # "yes" / "no"
    "reason": "TODO(you): say what each of the two quantities actually measures.",
}

# %%
HEADLINE_RUN = RUN_IDS[0]      # TODO(you): pick deliberately and say why in your notes
yt, yp = USABLE[HEADLINE_RUN]["y"], USABLE[HEADLINE_RUN]["p"]
LM_RUN = USABLE[HEADLINE_RUN]["lm"]
M = metrics.evaluate(yt, yp, LABELS, codes=LM.codes, names=NAMES,
                     groups=PS.scene[EVAL_TEST_IDX], n_boot=400, seed=SEED)
print(M.table(min_support=gates.MIN_CLASS_SUPPORT))
print(f"\n  evaluated run: {HEADLINE_RUN}")

maj_rate = M.majority_rate()
floor = M.floor_balanced_acc()
print("  TWO FLOORS, SIDE BY SIDE")
print(f"    majority_rate()      = {maj_rate:.4f}   <- floor of OVERALL ACCURACY")
print(f"    floor_balanced_acc() = {floor:.4f}   <- floor of BALANCED ACCURACY = 1/K")
print(f"\n  THE THREE LEGITIMATE COMPARISONS")
print(f"    acc vs majority rate      : {M.overall_acc:.4f} vs {maj_rate:.4f}   "
      f"delta {M.overall_acc - maj_rate:+.4f}")
print(f"    balanced_acc vs 1/K       : {M.balanced_acc:.4f} vs {floor:.4f}   "
      f"delta {M.balanced_acc - floor:+.4f}")
print(f"    macro-F1 vs best baseline : {M.macro_f1:.4f} vs "
      f"{BASELINES[BEST_BASE]['macro_f1']:.4f}   "
      f"delta {M.macro_f1 - BASELINES[BEST_BASE]['macro_f1']:+.4f}")
print(f"\n  THE 2025/26 LINE, REPRODUCED: balanced_acc - majority_rate() = "
      f"{M.balanced_acc - maj_rate:+.4f}")
print("  That last number is what the old notebook printed as 'Model improvement'. It mixes a")
print("  mean of recalls with a share of samples. It is not small or large; it is undefined.")
print(f"  P4 answer was: {P4}")

# %% [markdown]
# Note which of the three comparisons is doing something the other two are not. Only the
# third subtracts two numbers produced by the *same* metric on the *same* samples, which is
# what makes it interpretable — and it is why Part 8 computes it with a paired bootstrap
# rather than by subtracting two independently reported scores.
#
# One more divergence worth naming, because the 2025/26 lab never said it: **balanced accuracy
# and macro-F1 disagree on purpose.** Balanced accuracy is a mean of recalls and ignores
# precision, so a model that sprays predictions onto rare classes is not punished by it.
# Macro-F1 includes precision and is. When the two separate, the model is over-predicting rare
# classes. Compare the `pred` column with the `sup` column in the table above and name the
# classes being over-predicted.

# %% [markdown]
# ## Part 7 — How wide is that interval, really?
#
# Lab 5.2's headline `Balanced accuracy 0.8338` came from 455 validation samples where five of
# ten classes had support ≤ 9 and two had support 1. A recall estimated from n = 1 is literally
# 0 or 1, printed to three decimals as though it were a measurement. Nothing in that notebook
# carried an interval, so nothing could be seen to be noise.
#
# `metrics.evaluate` bootstraps by default, in two modes, and the difference between them is
# the most instructive number in this lab:
#
# * **i.i.d. (`groups=None`)** resamples individual patches. It assumes patch *i* tells you
#   nothing about patch *j*.
# * **scene-clustered (`groups=<scene ids>`)** resamples whole scenes. Patches inside one scene
#   share illumination, atmosphere, sensor date, terrain, and often the same 100 m CORINE
#   polygon. They are not independent observations.
#
# Five hundred patches from two scenes are not 500 independent observations. They are closer
# to two observations that happen to be 500 patches long.

# %% [markdown]
# ### Exercise 5 — predict the interval widths
#
# > **P5.** Score the *same* predictions twice: once with `groups=None`, once with
# > `groups=<scene ids>`. Before running, write down:
# >
# > 1. Which one has the **wider** 95% interval for balanced accuracy.
# > 2. The **ratio** of the two widths, to within a factor of two.
# > 3. How many resampling clusters the clustered version actually has on your split.
# >
# > Question 3 decides whether question 1 means anything. With one held-out scene the
# > clustered bootstrap has one cluster, every resample is the same sample, and the interval
# > has width exactly zero — an interval reporting a certainty no dataset supports.

# %%
P5 = {
    # TODO(you): fill in BEFORE running the next two cells.
    "wider": None,                 # "iid" or "clustered"
    "width_ratio": None,           # wider / narrower
    "n_clusters": None,
}

# %%
M_iid = metrics.evaluate(yt, yp, LABELS, codes=LM.codes, names=NAMES,
                         groups=None, n_boot=400, seed=SEED)
M_clu = metrics.evaluate(yt, yp, LABELS, codes=LM.codes, names=NAMES,
                         groups=PS.scene[EVAL_TEST_IDX], n_boot=400, seed=SEED)
w_iid = M_iid.ci["balanced_acc"][1] - M_iid.ci["balanced_acc"][0]
w_clu = M_clu.ci["balanced_acc"][1] - M_clu.ci["balanced_acc"][0]
print(f"  {'bootstrap':<22} {'95% CI balanced_acc':>26} {'width':>8} {'clusters':>9}")
print(f"  {'i.i.d. over patches':<22} "
      f"{str(np.round(M_iid.ci['balanced_acc'], 4).tolist()):>26} {w_iid:>8.4f} "
      f"{M_iid.ci['n_clusters']:>9}")
print(f"  {'clustered by scene':<22} "
      f"{str(np.round(M_clu.ci['balanced_acc'], 4).tolist()):>26} {w_clu:>8.4f} "
      f"{M_clu.ci['n_clusters']:>9}")
_ratio = ("undefined: the narrower width is exactly 0" if min(w_iid, w_clu) <= 0
          else f"{max(w_iid, w_clu) / min(w_iid, w_clu):.2f}x")
print(f"\n  wider: {'clustered' if w_clu > w_iid else 'iid'}   ratio: {_ratio}   "
      f"(clusters: i.i.d. {M_iid.ci['n_clusters']}, clustered {M_clu.ci['n_clusters']})")
print(f"  i.i.d. note recorded in the metrics object: {M_iid.ci['note']}")
print(f"  P5 answers were: {P5}")
if M_clu.ci["n_clusters"] < 2:
    print("\n  DEGENERATE — read this before the ratio above.")
    print(f"  {M_clu.ci['n_clusters']} cluster means every bootstrap resample is the same")
    print("  sample, so the clustered interval has width exactly 0.0000. It is not evidence")
    print("  of certainty; it is the absence of a measurement, and the width ratio against")
    print("  it is meaningless arithmetic. The fix is more held-out scenes, never a switch")
    print("  back to the i.i.d. bootstrap, whose interval is optimistic but at least varies.")

# %%
fig, ax = plt.subplots(figsize=(7.4, 3.2))
for i, (nm, mm, w) in enumerate((("i.i.d. patches", M_iid, w_iid),
                                 ("clustered by scene", M_clu, w_clu))):
    lo_, hi_ = mm.ci["balanced_acc"]
    ax.plot([lo_, hi_], [i, i], lw=6, alpha=0.75,
            color="#c0392b" if nm.startswith("clustered") else "#2980b9")
    ax.plot(mm.balanced_acc, i, "k|", ms=18)
    ax.text(hi_, i + 0.2, f"width {w:.4f}", fontsize=8)
ax.set_yticks([0, 1])
ax.set_yticklabels(["i.i.d. patches", "clustered\nby scene"], fontsize=8)
ax.axvline(1.0 / K, ls="--", c="k", lw=1, label=f"1/K = {1.0 / K:.3f}")
ax.set_xlabel("balanced accuracy, 95% bootstrap interval")
ax.set_ylim(-0.5, 1.6)
ax.legend(fontsize=8)
ax.set_title("same predictions, two bootstrap assumptions, two different claims")
fig.tight_layout()
FIG_CI = paths.ensure(paths.results_dir())[0] / "lab7_bootstrap_intervals.png"
fig.savefig(FIG_CI, dpi=110)
plt.show()
print(f"  saved {FIG_CI}")

# %% [markdown]
# Write one sentence in your report naming both widths and the number of clusters. If the
# clustered interval is wider, say what that means for every claim in your report that used a
# patch-level interval. If it is narrower or zero, say how many clusters you actually have —
# because at one cluster you have not measured uncertainty, you have assumed it away.

# %% [markdown]
# ## Part 8 — Paired comparison: is one model actually better than another?
#
# Subtracting two reported scores is not a comparison. Two runs on one split differ by more
# than intuition suggests, and unpaired intervals overlap for a reason.
# `metrics.paired_bootstrap_delta` resamples the *same* indices into both models, so the
# per-cluster difference is what gets resampled and the correlation between the two models is
# respected rather than destroyed.
#
# The claim this lab exists to test is the one the course has always asserted and never
# measured: that the Lab 6 foundation model beats the Lab 5 CNN. Testing it needs three
# things, and Lab 7 in 2025/26 had none of them — the same test split, the same metric code,
# and a delta whose interval excludes zero.

# %%
def arm_of(rec):
    return (rec.get("config") or {}).get("arm", rec["run_id"])


by_lab: dict[str, list] = {}
for rid in RUN_IDS:
    rec = next(r for r in COMPARABLE if r["run_id"] == rid)
    by_lab.setdefault(str(rec.get("lab")), []).append(rid)
print("  comparable runs by lab:")
for k, v in sorted(by_lab.items()):
    print(f"    {k:<8} {v}")

# %% [markdown]
# ### Exercise 6 — predict the delta
#
# > **P6.** Choose two runs on the split of record: the Lab 5 CNN as model A and the Lab 6
# > foundation model as model B. If Lab 6 is not on this split, take the two Lab 5 arms that
# > are, and state in your report that the cross-lab claim is **not testable** on your data —
# > that is a finding, not a failure. Before running:
# >
# > 1. The **sign** of `macro_F1(B) - macro_F1(A)`.
# > 2. Its **magnitude**, to two decimals.
# > 3. Whether the 95% interval of the paired delta excludes zero.
# >
# > Question 3 is the only one that licenses the word "improvement". A delta inside its own
# > interval is a coin flip you happened to report in one direction.

# %%
P6 = {
    # TODO(you): fill in BEFORE running the next two cells.
    "sign": None,                 # "+" or "-"
    "magnitude": None,
    "interval_excludes_zero": None,
}

# %%
if by_lab.get("lab6") and by_lab.get("lab5.1"):
    RUN_A, RUN_B = by_lab["lab5.1"][0], by_lab["lab6"][0]
    PAIR_KIND = "cross-lab: Lab 5 CNN vs Lab 6 foundation model"
elif len(RUN_IDS) >= 2:
    RUN_A, RUN_B = RUN_IDS[0], RUN_IDS[-1]
    PAIR_KIND = "within-lab fallback: two runs on one split"
    print("  NOTE: no Lab 6 record on the split of record, so the cross-lab claim cannot be")
    print("  tested here. Lab 6 holds out a different pair of scenes. That mismatch is a real")
    print("  course defect and belongs in your report, not under a rug.")
else:
    raise RuntimeError(
        f"need two comparable runs to pair; have {RUN_IDS}. Re-run Lab 5 with a second arm")
ya, pa = USABLE[RUN_A]["y"], USABLE[RUN_A]["p"]
yb, pb = USABLE[RUN_B]["y"], USABLE[RUN_B]["p"]
assert np.array_equal(ya, yb), "the two runs are not on the same samples; do not pair them"
print(f"  pairing ({PAIR_KIND})")
print(f"    A = {RUN_A}")
print(f"    B = {RUN_B}")

DELTA = metrics.paired_bootstrap_delta(ya, pa, pb, LABELS,
                                       groups=PS.scene[EVAL_TEST_IDX], n_boot=2000, seed=SEED)
print(f"  paired delta (B - A), {DELTA['n_boot']} resamples of "
      f"{len(np.unique(PS.scene[EVAL_TEST_IDX]))} scene clusters")
print(f"    balanced_acc : mean {DELTA['mean_delta_balanced_acc']:+.4f}   "
      f"95% CI {np.round(DELTA['delta_balanced_acc'], 4).tolist()}   "
      f"excludes 0: {DELTA['significant_balanced_acc']}")
print(f"    macro-F1     : mean {DELTA['mean_delta_macro_f1']:+.4f}   "
      f"95% CI {np.round(DELTA['delta_macro_f1'], 4).tolist()}   "
      f"excludes 0: {DELTA['significant_macro_f1']}")
print(f"  P6 answers were: {P6}")
print("  An interval straddling zero is not 'probably an improvement'. It is an experiment")
print("  that has not resolved the question, and reporting it as a win is the 2025/26 error.")
if len(np.unique(PS.scene[EVAL_TEST_IDX])) < 2:
    print("\n  DEGENERATE, same cause as Part 7: one scene cluster, so every paired resample")
    print("  is the original sample and both interval bounds equal the point estimate. The")
    print("  interval is not tight, it is absent. Report the point delta and say plainly that")
    print("  its uncertainty is unmeasurable on this split — do not read a zero-width interval")
    print("  as a significant one.")

# %% [markdown]
# ### The second bar: seed noise
#
# A paired bootstrap says two prediction arrays differ on these samples. It does not say the
# difference would survive retraining. `gates.gate_claim_supported` requires at least
# `MIN_SEEDS = 3` runs carrying the metric and demands the delta exceed
# `MIN_SEED_MULTIPLIER = 2.0` times their pooled standard deviation. Below that, the only
# honest sentence is "no detectable effect at n = 3".

# %%
RECORD_OF = {r["run_id"]: r for r in ALL_RUNS}
ARM_A = arm_of(RECORD_OF[RUN_A])
ARM_RUNS = [RECORD_OF[r] for r in RUN_IDS if arm_of(RECORD_OF[r]) == ARM_A]
print(f"  runs carrying arm '{ARM_A}' on the split of record: "
      f"{[r['run_id'] for r in ARM_RUNS]}")
try:
    var = gates.gate_seeds_and_variance(ARM_RUNS, metric="macro_f1")
    print(f"    mean {var['mean']:.4f}  std {var['std']:.4f}  n {var['n']}  "
          f"claim threshold {var['threshold']:.4f}")
    if var["std"] == 0.0:
        print("    std is exactly 0.0000. Runs that agree to the last digit are almost never")
        print("    independent runs — check that the seed reached the sampler and the")
        print("    dataloader. Zero spread is a bug report, not precision.")
except gates.GateFailure as exc:
    var = None
    print(f"  gate_seeds_and_variance raises:\n    {exc}")

# %%
SEED_BAR_OK = False
if var is not None:
    try:
        gates.gate_claim_supported(DELTA["mean_delta_macro_f1"], ARM_RUNS, metric="macro_f1",
                                   label=f"{RUN_B} vs {RUN_A}")
        SEED_BAR_OK = True
        print(f"  gate_claim_supported: PASS on its own test — delta "
              f"{DELTA['mean_delta_macro_f1']:+.4f} exceeds "
              f"{gates.MIN_SEED_MULTIPLIER} x std = {var['threshold']:.4f}")
    except gates.GateFailure as exc:
        print(f"  gate_claim_supported raises:\n    {exc}")

# gate_claim_supported checks the delta against SEED spread only. It says nothing about the
# paired interval, and with a zero seed spread its threshold is zero, so it passes on any
# nonzero delta. Clearing it is necessary and nowhere near sufficient.
CLAIM_OK = bool(DELTA["significant_macro_f1"]) and SEED_BAR_OK
print(f"\n  bar 1, paired interval excludes 0 : {DELTA['significant_macro_f1']}")
print(f"  bar 2, delta > {gates.MIN_SEED_MULTIPLIER} x seed std          : {SEED_BAR_OK}")
print(f"  therefore the word 'improvement' is licensed: {CLAIM_OK}")
print("  Note what just nearly happened: bar 2 passed while bar 1 failed, because three")
print("  identical runs have zero spread and 2 x 0 = 0. If you read gate_claim_supported as")
print("  the end of the argument you would have written a conclusion the interval forbids.")
print("  Lab 7 in 2025/26 had neither bar, which is how a 122 M-parameter model on 8,278")
print("  patches became evidence of something.")

# %% [markdown]
# ## Part 9 — Cross-lab reconciliation
#
# The course's published result was three numbers typed into a markdown cell. This part prints
# them from the ledger instead, and then checks whether each record is admissible evidence for
# the sentence you would attach to it.
#
# The checks are mechanical, and each one has fired somewhere in the course's history:
#
# * **no `test` block** — Lab 5.1 in 2025/26 had no test result at all; `trainer.test()` was
#   unexecuted and every "result" was a validation number.
# * **`test.n` ≠ this split's `n_test`** — the record's "test" is a different slice.
# * **`test.n == val.n`** — one slice scored twice and renamed.
# * **no per-class `support`** — a per-class table without support is ungradeable, because
#   nothing tells the reader which recalls are noise.

# %%
print(results.summary_table())

print(f"  split of record n_test (scored) = {EVAL_TEST_IDX.size}\n")
print(f"  {'run_id':<26} {'lab':<8} {'test.n':>7} {'val.n':>7} {'sup?':>5}  flags")
ADMISSIBLE = []
for r in ALL_RUNS:
    t, v = r.get("test") or {}, r.get("val") or {}
    flags = []
    if not t:
        flags.append("NO test block: nothing was measured on test")
    else:
        if t.get("n") != int(EVAL_TEST_IDX.size):
            flags.append(f"test.n={t.get('n')} != {int(EVAL_TEST_IDX.size)} on this split")
        if not t.get("support"):
            flags.append("no per-class support recorded")
    if t and v and t.get("n") == v.get("n"):
        flags.append("test.n == val.n: one slice scored twice")
    if r.get("test_used_for_tuning"):
        flags.append("declares test_used_for_tuning=True")
    print(f"  {r['run_id']:<26} {str(r.get('lab')):<8} {str(t.get('n')):>7} "
          f"{str(v.get('n')):>7} {str(bool(t.get('support'))).lower():>5}  "
          f"{'; '.join(flags) or 'clean'}")
    if not flags:
        ADMISSIBLE.append(r["run_id"])
print(f"\n  {len(ADMISSIBLE)} of {len(ALL_RUNS)} records are admissible evidence as they stand")

# %% [markdown]
# ### The failure this replaces
#
# In 2025/26 the reconciliation cell was a markdown table with three numbers typed into it,
# labelled "Lab 5 achieved these results on CORINE data". Nothing connected those literals to
# any file. If Lab 5 had been rerun and improved, Lab 7 would still have printed the old
# numbers and still called them the bar. The cell below is the same table rebuilt from the
# ledger, with the split each number came from printed beside it.

# %%
print(f"  {'run_id':<26} {'lab':<8} {'acc':>8} {'bal_acc':>8} {'macro_F1':>9}  split")
for r in ALL_RUNS:
    t = r.get("test") or {}
    if not t:
        continue
    print(f"  {r['run_id']:<26} {str(r.get('lab')):<8} {t.get('overall_acc', 0):>8.4f} "
          f"{t.get('balanced_acc', 0):>8.4f} {t.get('macro_f1', 0):>9.4f}  "
          f"{str(r.get('split_manifest_hash'))[:8]}")
print(f"\n  this lab's own evaluation: {HEADLINE_RUN}  acc {M.overall_acc:.4f}  "
      f"bal_acc {M.balanced_acc:.4f}  macro-F1 {M.macro_f1:.4f}")
print(f"  best baseline {BEST_BASE}: macro-F1 {BASELINES[BEST_BASE]['macro_f1']:.4f}")
print("  Every figure above was read from results.json in this kernel. None was typed. If you")
print("  quote one in your report, quote the row, not your memory of it.")

# %% [markdown]
# ## Part 10 — Error analysis: which class, and confused with what
#
# A macro-average tells you how badly you did on average. It does not tell you what to fix.
# Per-class recall plus the confusion matrix does, and CORINE supplies the vocabulary to
# interpret it: `labels.level1_of` maps a level-3 code to its level-1 group, so "code 23 is
# called code 26" becomes "forest is called shrub", which is a claim about spectra you can act
# on.
#
# Two confusions are structurally different and you should say which you are looking at:
#
# * **spectral** — level-1 groups genuinely hard to separate at 3 × 3 or 5 × 5 pixels with four
#   bands. More context or more bands helps.
# * **geometric** — a boundary or narrow class whose 100 m CORINE polygon does not match the
#   10 m pixel grid. No model fixes that; it is label error, and reporting it as model error
#   sends someone chasing a bug that does not exist.

# %%
print(M.table(min_support=gates.MIN_CLASS_SUPPORT))
print(f"\n  worst class by F1: {M.worst_class()}")
print(f"  largest share of predictions in one class: "
      f"{M.predicted.max() / max(M.predicted.sum(), 1):.1%}")
print(f"  over-prediction check (pred/sup, worst three): "
      f"{sorted((M.predicted[i] / max(M.support[i], 1) for i in range(K)), reverse=True)[:3]}")

NAME_TO_IDX = {n: i for i, n in enumerate(NAMES)}
print("  Top confusions — 'true class is called predicted class', with counts:")
print(f"  {'true':<30} {'predicted':<30} {'n':>5}  level-1 true -> predicted")
for true_c, pred_c, n in metrics.top_confusions(M, k=8):
    i, j = NAME_TO_IDX.get(true_c), NAME_TO_IDX.get(pred_c)
    if i is None or j is None or int(LM.codes[i]) == RARE_OTHER or int(LM.codes[j]) == RARE_OTHER:
        l1 = "(merged bucket: not a CORINE code)"
    else:
        l1 = (f"{lab_mod.level1_of(int(LM.codes[i]))[:13]} -> "
              f"{lab_mod.level1_of(int(LM.codes[j]))[:13]}")
    print(f"  {true_c:<30} {pred_c:<30} {n:>5}  {l1}")
print("\n  TODO(you): take the single largest off-diagonal cell and write one sentence naming")
print("  it spectral or geometric, with the evidence you used to decide.")

# %%
_cm = M.confusion.astype(np.float64)
_row = _cm / np.maximum(_cm.sum(axis=1, keepdims=True), 1)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
for ax, mat, ttl in ((axes[0], _cm, "counts"), (axes[1], _row, "row-normalised = recall")):
    im = ax.imshow(mat, cmap="magma")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(NAMES, rotation=90, fontsize=6)
    ax.set_yticklabels(NAMES, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("rows true, columns predicted; row-normalised shows recall, "
             "column-normalised shows precision", fontsize=9)
fig.tight_layout()
FIG_CM = paths.ensure(paths.results_dir())[0] / "lab7_confusion.png"
fig.savefig(FIG_CM, dpi=110)
plt.show()
print(f"  saved {FIG_CM}")

# %% [markdown]
# ### Merging rare classes is a train decision
#
# `gates.MIN_CLASS_SUPPORT = 25` exists because recall on n = 25 has a standard error near 0.10
# and recall on n = 1 is not an estimate at all. Lab 5.2 in 2025/26 reported per-class metrics
# to three decimals on supports of `9, 1, 245, 99, 6, 1, 5, 52, 30, 7`.
#
# The merge decision must be made on **train** support. Deciding it on test support is choosing
# which classes to be scored on after seeing the model's answers, and it lets a model dodge
# its hardest classes. Lab 5.2's class-drop threshold `< 10` was computed over train+val+test,
# which silently changed the denominator of every reported metric — and removed Peat bogs and
# Water courses, two of the most relevant classes in Iceland.
#
# Merging is not free. It changes the macro-average, usually upward, because a class the model
# gets entirely wrong stops being a row of zeros. Report the change.

# %%
_, _, MAP = metrics.merge_rare_classes(PS.labels[train_idx], LM.codes,
                                       gates.MIN_CLASS_SUPPORT)
MERGED = [int(c) for c in MAP["merged"]]


def to_merged(codes_arr):
    """Apply the TRAIN-decided merge to an array of CORINE codes."""
    out = np.asarray(codes_arr).copy()
    for _c in MERGED:
        out[out == _c] = RARE_OTHER
    return out


print(f"  merge decided on TRAIN support < {gates.MIN_CLASS_SUPPORT}")
print(f"    merged codes : {MERGED}")
print(f"    names        : {[lab_mod.class_name(c) for c in MERGED]}")
print(f"    merge_rare_classes proposed rare-other code {MAP['rare_other_code']}")
if MAP["rare_other_code"] is not None and int(MAP["rare_other_code"]) in lab_mod.CORINE_CLASSES:
    print(f"  WARNING: max(codes)+1 = {MAP['rare_other_code']} IS a real CORINE code "
          f"({lab_mod.class_name(int(MAP['rare_other_code']))}). This lab uses {RARE_OTHER} "
          "instead, so the bucket cannot be mistaken for a land-cover class.")

# %%
if MERGED:
    codes_m = tuple(c for c in LM.codes if int(c) not in MERGED and int(c) != RARE_OTHER)
    codes_m = codes_m + (RARE_OTHER,)
    LM_M = lab_mod.LabelMap(codes_m)
    NAMES_M = list(LM_M.names)
    NAMES_M[LM_M.index_of(RARE_OTHER)] = f"rare-other (merged {MERGED})"
    M_MERGED = metrics.evaluate(LM_M.encode(to_merged(EXPECTED_CODES)),
                                LM_M.encode(to_merged(LM.decode(yp))),
                                np.arange(LM_M.n_classes), codes=LM_M.codes, names=NAMES_M,
                                groups=PS.scene[EVAL_TEST_IDX], n_boot=400, seed=SEED)
    print(f"  {'':<20} {'K':>3} {'bal_acc':>9} {'macro_F1':>9}")
    print(f"  {'unmerged':<20} {K:>3} {M.balanced_acc:>9.4f} {M.macro_f1:>9.4f}")
    print(f"  {'merged':<20} {LM_M.n_classes:>3} {M_MERGED.balanced_acc:>9.4f} "
          f"{M_MERGED.macro_f1:>9.4f}")
    print(f"  effect of merging: macro-F1 {M_MERGED.macro_f1 - M.macro_f1:+.4f}, "
          f"balanced_acc {M_MERGED.balanced_acc - M.balanced_acc:+.4f}")
    print("  Report both rows. A merged macro-average quoted without the unmerged row beside")
    print("  it is a number chosen to look better, not a number measured.")
else:
    M_MERGED = M
    print("  nothing fell below the threshold on train support; no merge applied")

# %% [markdown]
# ## Part 11 — Negative control: the split that makes the number go up
#
# Every lab in this course ends with a control designed to fail. Here it is: the 2025/26
# splitting rule. `splits.stratified_random_split` splits per class at random, so edge-adjacent
# patches — which share pixels, illumination and the same 100 m CORINE polygon — land on both
# sides. It is not merely a worse split, it is inadmissible, and `gate_split_is_grouped` rejects
# it by method name before any metric is computed.
#
# No retraining is needed to see the effect. Take the checkpoint you already have, which was
# trained on the honest train split, and score it on a "test" slice that overlaps that training
# data. The number rises. That is the whole 2025/26 result, in one cell.

# %%
# The 2025/26 notebooks deleted rare classes in order to make a random split constructible.
# Try it on the real class set first and let the split say what it refuses to do.
try:
    LEAKY = splits.stratified_random_split(PS.labels, train_ratio=0.7, val_ratio=0.15,
                                           seed=SEED, min_per_class=2)
    LEAKY_LABELS = PS.labels
    print("  the random split was constructible on the full class set")
except splits.SplitLeakError as exc:
    print("  stratified_random_split REFUSES the real class set:")
    print(f"    {exc}")
    print("  That refusal is the 2025/26 story in one line: to run a per-class random split on")
    print("  imbalanced EO data you must first delete the classes that make it imbalanced, and")
    print("  the reported metric is then computed on the survivors. Retrying on the merged")
    print("  class set, which is what 'drop anything under 25' actually means.")
    LEAKY_LABELS = to_merged(PS.labels) if MERGED else PS.labels
    LEAKY = splits.stratified_random_split(LEAKY_LABELS, train_ratio=0.7, val_ratio=0.15,
                                           seed=SEED, min_per_class=2)
print(f"\n  leaky manifest {LEAKY.manifest_hash}  method={LEAKY.method}")
print(f"  index overlap: {[k for k, v in LEAKY.overlaps().items() if v] or 'none'}")
print("  Indices are disjoint, so a disjointness check passes. That is the check the 2025/26")
print("  notebook ran, and it is the wrong check.")

LEAKY_TEST = LEAKY.test_idx
n_leaked = int(np.isin(LEAKY_TEST, train_idx).sum())
n_adjacent = int(np.isin(LEAKY_TEST, val_idx).sum())
print(f"  leaky 'test' slice: {LEAKY_TEST.size} patches")
print(f"  of which {n_leaked} ({n_leaked / max(LEAKY_TEST.size, 1):.1%}) were in the honest")
print(f"  TRAIN split, and {n_adjacent} ({n_adjacent / max(LEAKY_TEST.size, 1):.1%}) in VAL.")
print("  That percentage is the size of the lie. A per-class random split draws its 'test'")
print("  from every patch in the archive, so the model is scored on the data it fitted.")
print("  Even the part it did not fit is drawn from scenes it fitted, so the patches beside")
print("  every test patch were in training.")

try:
    gates.gate_split_is_grouped(LEAKY)
    gates.report("gate_rejects_leaky_split", False, "the gate accepted a stratified split")
except gates.GateFailure as exc:
    gates.report("gate_rejects_leaky_split", True, str(exc).splitlines()[0][:120])

# %%
_ck = checkpoint_path_for(RECORD_OF[HEADLINE_RUN])
if _ck is None:
    print("  no checkpoint for the headline run; the negative control needs one. Re-run Lab 5.")
    HONEST = LEAKED = None
else:
    def score(idx, pred_idx, lm_r):
        """Score predictions (in lm_r's index space) against the truth at idx.

        Both sides go through CORINE codes into LM's index space, and any sample whose
        true or predicted code is outside LM is dropped and counted rather than encoded
        to a wrong class. evaluate() would raise on it, which is the right behaviour but
        the wrong place to discover it.
        """
        tc, pc = PS.labels[idx], lm_r.decode(pred_idx)
        keep = np.isin(tc, LM.codes) & np.isin(pc, LM.codes)
        m = metrics.evaluate(LM.encode(tc[keep]), LM.encode(pc[keep]), LABELS,
                             codes=LM.codes, names=NAMES, n_boot=0)
        return m, int((~keep).sum())

    HONEST, drop_h = score(EVAL_TEST_IDX, infer(_ck, EVAL_TEST_IDX)[0], LM_RUN)
    LEAKED, drop_l = score(LEAKY_TEST, infer(_ck, LEAKY_TEST)[0], LM_RUN)
    print(f"  samples dropped for codes outside the fixed label set: "
          f"honest {drop_h}, leaky {drop_l}")
    print(f"  {'slice':<26} {'acc':>8} {'bal_acc':>8} {'macro_F1':>9} {'n':>6}")
    print(f"  {'honest held-out scenes':<26} {HONEST.overall_acc:>8.4f} "
          f"{HONEST.balanced_acc:>8.4f} {HONEST.macro_f1:>9.4f} {HONEST.n:>6}")
    print(f"  {'stratified_random':<26} {LEAKED.overall_acc:>8.4f} "
          f"{LEAKED.balanced_acc:>8.4f} {LEAKED.macro_f1:>9.4f} {LEAKED.n:>6}")
    print(f"  the leaky slice buys {LEAKED.macro_f1 - HONEST.macro_f1:+.4f} macro-F1 for free")

print("\n  TODO(you): one sentence, with both macro-F1 numbers and the overlap percentage in")
print("  it, on why the higher number is worth less. It must name what is on both sides of")
print("  the leaky split. A sentence you could have written without running the cell has not")
print("  been earned.")

# %% [markdown]
# ## Part 12 — Gate board and deliverable
#
# Gates raise. That is the point. In 2025/26 a student could finish Lab 7, see four pretty
# tables, and submit without ever touching a trained model, because nothing in the notebook
# asserted anything and nothing wrote a byte to disk. Each line on the board is a claim you are
# making in your report.
#
# `results.record_run` writes first, because `run_all_gates` reads the record to check that the
# split hash you reported is the split in use and that you did not select on test. `results.json`
# is append-only and refuses a duplicate `run_id`; that refusal is correct, and if it fires,
# change the id rather than deleting the earlier attempt — the earlier attempt is evidence you
# did the work.

# %%
LINES = []
try:
    prior = gates.gate_results_file(lab="lab5.1")
    LINES.append(f"[PASS] results_file: {len(prior)} upstream lab5.1 record(s) well-formed")
except gates.GateFailure as exc:
    LINES.append(f"[FAIL] results_file: {exc}")

try:
    gates.gate_split_hash_matches(RECORD_OF[HEADLINE_RUN], manifest)
    LINES.append(f"[PASS] split_hash: {HEADLINE_RUN} reports {manifest.manifest_hash}")
except gates.GateFailure as exc:
    LINES.append(f"[FAIL] split_hash: {exc}")

try:
    _p = gates.gate_predictions_saved(
        yt, yp, paths.run_dir(f"lab7_eval_{manifest.manifest_hash}") / "test_predictions.npz",
        manifest.manifest_hash)
    PRED_PATH = _p
    LINES.append(f"[PASS] predictions_saved: {_p.name}, every metric re-derivable from it")
except (gates.GateFailure, OSError) as exc:
    PRED_PATH = None
    LINES.append(f"[FAIL] predictions_saved: {exc}")

LINES += gates.run_all_gates(M, BASELINES, manifest, RECORD_OF[HEADLINE_RUN])
LINES.append(f"[{'PASS' if CLAIM_OK else 'FAIL'}] claim_supported: paired delta "
             f"{DELTA['mean_delta_macro_f1']:+.4f}; bar 1 interval excludes 0: "
             f"{DELTA['significant_macro_f1']}; bar 2 seed spread: {SEED_BAR_OK}. Both are "
             "required, and a zero seed spread makes bar 2 vacuous.")
LINES.append(f"[{'PASS' if M_clu.ci['n_clusters'] >= 2 else 'FAIL'}] bootstrap_clusters: "
             f"{M_clu.ci['n_clusters']} scene cluster(s) resampled; below 2 the interval is "
             "meaningless")
LINES.append(f"[{'PASS' if all(ROUNDTRIP.values()) and ROUNDTRIP else 'FAIL'}] "
             f"roundtrip: {sum(ROUNDTRIP.values())}/{len(ROUNDTRIP)} runs reproduce their "
             "saved predictions from the checkpoint on disk")

gates.print_gate_board(LINES)

# %% [markdown]
# ### Reading a failed board
#
# A `FAIL` here is not a formatting problem. It means a conclusion you would have written down
# is not supported, and the message names the number and the threshold.
#
# * **`class_support`** — a scored class has fewer than 25 test samples. Merge it (Part 10) and
#   report the merged class. Do not average over what is left and call it macro.
# * **`beats_baselines`** — the central gate. Before concluding the model is bad, check the two
#   usual causes in order: an input-unit mismatch (`gate_input_units`), then a collapsed head —
#   look at the `pred` column; if one class absorbs nearly everything, the model learned the
#   prior and nothing else.
# * **`above_chance`** — balanced accuracy at or below 1/K. Overall accuracy can look far higher
#   and mean exactly the same thing.
# * **`bootstrap_clusters`** — you cannot measure uncertainty from one cluster. This is a data
#   acquisition problem, not a statistics problem.
# * **`roundtrip`** — the npz and the checkpoint disagree, so one is stale. Until you know
#   which, neither is evidence.
#
# Reporting a failed gate with a correct diagnosis is a passing submission. Reporting a green
# board obtained by quietly changing the split is not.
#
# ### The deliverable
#
# Every decision lands in `results.json` so it is gradeable rather than asserted: the split of
# record and how it was chosen, the fixed label set, the merge decision, the baseline table, the
# paired delta, the bootstrap mode, and the honest `test_used_for_tuning: false`.
#
# This record is for the **evaluation**, not a training arm. `summarize_results.py --check`
# groups records by `config["arm"]` and asks for `MIN_SEEDS` runs per arm, so a single
# evaluation record will not satisfy that gate. That is the summariser telling you something
# true: an evaluation inherits its seed spread from the runs it evaluated, and Part 8 is where
# that spread lives.

# %%
RUN_ID = f"lab7_eval_{manifest.manifest_hash}"
CONFIG = {
    "arm": "evaluation",
    "evaluated_run": HEADLINE_RUN,
    "pair": {"A": RUN_A, "B": RUN_B, "kind": PAIR_KIND},
    "split_of_record": {"hash": manifest.manifest_hash, "method": manifest.method,
                        "test_scenes": sm["test_scenes"], "chosen_by": "ledger majority"},
    "n_classes": LM.n_classes, "codes": list(LM.codes), "class_names": NAMES,
    "zero_train_support_codes": DROP_CODES,
    "merge_min_support": gates.MIN_CLASS_SUPPORT, "merged_codes": MERGED,
    "rare_other_code": RARE_OTHER,
    "bootstrap": {"mode": "scene-clustered", "n_clusters": int(M_clu.ci["n_clusters"]),
                  "iid_width": float(w_iid), "clustered_width": float(w_clu)},
    "baselines_scored": sorted(BASELINES), "best_baseline": BEST_BASE,
    "n_scored_test": int(EVAL_TEST_IDX.size), "n_test_total": int(test_idx.size),
    "n_params": int((RECORD_OF[HEADLINE_RUN].get("config") or {}).get("n_params") or 0),
    "evaluated_model_input_layout": LAYOUTS.get(HEADLINE_RUN),
    "bands": list(PS.bands), "patch_size": PS.patch_size, "block": sm["block"],
    "normalisation": norm.to_dict(), "norm_fitted_on": "train only",
    "input_units": UNITS, "seed": SEED, "torch": str(torch.__version__),
}

# %%
try:
    REC = results.record_run(
        RUN_ID, lab=LAB, config=CONFIG, split_manifest_hash=manifest.manifest_hash,
        seed=SEED, test_metrics=M.to_dict(), baselines=BASELINES,
        test_used_for_tuning=False, n_seeds=len(ARM_RUNS),
        notes=(f"evaluated {HEADLINE_RUN} on the split of record chosen from the ledger, not "
               f"by the author. Best baseline {BEST_BASE} macro-F1 "
               f"{BASELINES[BEST_BASE]['macro_f1']:.4f}; model macro-F1 {M.macro_f1:.4f}. "
               f"Paired delta {RUN_B} minus {RUN_A} on macro-F1 "
               f"{DELTA['mean_delta_macro_f1']:+.4f} "
               f"CI {np.round(DELTA['delta_macro_f1'], 4).tolist()}, claim supported: "
               f"{CLAIM_OK}. Scene-clustered CI width {w_clu:.4f} vs i.i.d. {w_iid:.4f} over "
               f"{M_clu.ci['n_clusters']} clusters. Merged codes {MERGED} on train support "
               f"below {gates.MIN_CLASS_SUPPORT}; macro-F1 effect "
               f"{M_MERGED.macro_f1 - M.macro_f1:+.4f}. "
               "TODO(you): append your predicted-vs-actual explanations for P1-P6 here."),
        extra={"paired_delta": DELTA, "iid_ci": M_iid.ci, "clustered_ci": M_clu.ci,
               "merged_metrics": M_MERGED.to_dict(), "gate_board": LINES,
               "predictions": str(PRED_PATH) if PRED_PATH else None,
               "roundtrip": ROUNDTRIP,
               "leaky_split": {"hash": LEAKY.manifest_hash,
                               "n_test": int(LEAKY_TEST.size),
                               "n_also_in_train": n_leaked,
                               "macro_f1": None if LEAKED is None else LEAKED.macro_f1},
               "predictions_exercises": {"P1": P1, "P2": P2, "P3": P3, "P4": P4,
                                         "P5": P5, "P6": P6}})
    print(f"  recorded run_id={REC['run_id']}")
except results.ResultsError as exc:
    print(f"  not recorded: {exc}")
    print("  results.json is append-only. Change the run_id rather than overwriting the")
    print("  earlier attempt; the earlier attempt is part of your record of work.")

print(results.summary_table(LAB))
CSV_PATH = results.write_csv(LAB)
print(f"\n  csv for team comparison: {CSV_PATH}")

# %%
_proc = subprocess.run([sys.executable, str(REPO / "scripts" / "summarize_results.py"),
                        "--lab", LAB, "--check"], cwd=str(REPO), capture_output=True,
                       text=True)
print(_proc.stdout[-2600:])
print(f"  summarize_results.py --check exit code: {_proc.returncode}")
if _proc.returncode != 0:
    print("  Read every FAIL line above and diagnose it; do not explain the board away. Two")
    print("  are structural and two are findings:")
    print("   * seeds[arm] — the summariser wants MIN_SEEDS runs per arm and an evaluation")
    print("     record has one. Its seed spread is Part 8's, not its own. That one is the")
    print("     summariser's grouping rule, and you should say so rather than pad the ledger.")
    print("   * beats_baselines — if this is red, the model you just evaluated does not beat")
    print("     a logistic regression or an NDVI threshold on this split. That is a result.")
    print("     Report it, name the baseline that won, and diagnose it in the order the gate")
    print("     message gives: input units first, then a collapsed head, then the model.")
    print("  TODO(you): write the diagnosis for every FAIL here before you submit.")

# %% [markdown]
# ## Submission checklist
#
# Everything below is a file a grader can open. Self-attestation is not an artifact.
#
# * `results/results.json` — one `lab7` record with `split_manifest_hash`, `seed`, `config`
#   (evaluated run, the pair compared, how the split was chosen, the fixed label set and its
#   zero-support codes, the merge decision, the bootstrap mode and cluster count, the input
#   units), `test` metrics from `metrics.evaluate`, the full `baselines` block, and
#   `test_used_for_tuning: false`.
# * `results/results.csv` — from `results.write_csv("lab7")`.
# * `results/lab7_baselines.png`, `lab7_bootstrap_intervals.png`, `lab7_confusion.png`.
# * `results/runs/lab7_eval_<hash>/test_predictions.npz` plus the `.split` sidecar.
# * `data/splits/split_<hash>.json` — the manifest whose hash your record carries.
#
# In the write-up:
#
# 1. Predicted vs actual for **P1–P6**, every disagreement explained by a mechanism. A
#    prediction that matched still needs the mechanism stated.
# 2. The baseline table, with the chance floor and the strongest baseline named **before** any
#    model number appears.
# 3. Both bootstrap widths and the cluster count, with a sentence on what the clustered width
#    does to your claims.
# 4. The paired delta, its interval, the seed threshold, and whether the claim survived. If it
#    did not, write "no detectable effect at n = <seeds>" and mean it.
# 5. The cross-lab reconciliation table, with every flagged record diagnosed — in particular any
#    lab reporting validation metrics where test metrics were expected.
# 6. The merge decision on **train** support, and the macro-average effect of it, both rows.
# 7. The negative-control sentence, with both macro-F1 numbers and the overlap percentage.
# 8. The gate board with every `FAIL` diagnosed.
# 9. One claim you cannot make from this evaluation, and why.
