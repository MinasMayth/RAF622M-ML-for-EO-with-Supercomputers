# %% [markdown]
# # Lab 6 — Fine-tuning Prithvi-EO-2.0: what a foundation model actually buys you
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# This lab fine-tunes Prithvi-EO-2.0 on your Lab 4.2 patches with TerraTorch, and it
# measures three things that the 2025/26 edition asserted instead of measuring:
#
# 1. what happens when you feed a pretrained backbone the wrong **input units**;
# 2. what happens when you declare the wrong **band order**;
# 3. how much of your score is actually **pre-training** rather than your data.
#
# It assumes Lab 4.2 wrote per-scene patch archives into `paths.training_data_dir()`
# with scene/row/col provenance, and that Lab 5 gave you baselines you already
# distrust. It produces a `results.json` record for `lab6` carrying every ablation arm.
#
# **You must run every cell yourself.** The 2025/26 notebook was submitted with its
# headline number — `test/Accuracy = 0.10000000149011612` — and nobody noticed that
# this is exactly 1/10, i.e. chance for 10 classes, against a majority-class baseline
# of 0.538 on the same 455-sample test set. The model predicted one class. The unit
# sheet's deliverable was "Functioning end-to-end TerraTorch workflow" plus "Fine-tuned
# Prithvi-EO-2.0 checkpoint", and both were satisfied by a random model.
#
# ## The two sentences that matter most
#
# > **TerraTorch does not normalise your data for you.** `PRITHVI_V2_MEAN` and
# > `PRITHVI_V2_STD` are *metadata about the pretraining distribution*, not a transform
# > in the inference path. The 2025/26 config never applied them, so the encoder saw
# > band contrast 10 000x smaller than it was pretrained on, sitting on a constant
# > offset. That is the whole 0.10.
#
# > **TerraTorch permutes the pretrained patch-embed weights by band *name*.** A
# > `backbone_bands` list that disagrees with the tensor's actual channel order
# > silently transposes two channels. A list of names that are not `HLSBands` values
# > silently discards the entire pretrained input layer and re-initialises it with
# > Xavier. Neither raises.
#
# Everything below is either a measurement of one of those two sentences, or the
# machinery needed to make the measurement honest.

# %% [markdown]
# ## The 2025/26 defects this notebook closes
#
# | old defect | consequence | fixed in |
# |---|---|---|
# | no normalisation anywhere; `[0,1]` percentile patches into a backbone whose training means are ~1000 DN | `test/Accuracy = 0.10000000149011612` = exactly 1/10 = chance; nine classes at recall 0.0, one at 1.0 | Part 4, Part 8, Exercise 1 |
# | `backbone_bands = ["BLUE","RED","GREEN","NIR_NARROW"]` against data stacked `B02,B03,B04,B08` = BLUE,**GREEN**,**RED** | channels 1 and 2 silently transposed by name-based weight selection | Part 5, Exercise 2 |
# | commented-out alternative `["B02","B03","B04","B08"]` | not `HLSBands` values, so **all four** patch-embed filters are Xavier-initialised and the pretrained input layer is discarded with no error | Part 5, Exercise 2 |
# | `monitor="val/Accuracy"` (macro) over a val split holding classes with n=1 | one sample flips a class recall 0 to 1 and moves the selection metric by 0.1 | Part 7 |
# | `trainer.test(model=task, ...)` with no `ckpt_path="best"` | final-epoch weights were tested; the entire checkpoint-selection machinery was never exercised | Part 7 |
# | no baseline of any kind | a model scoring chance was reported as a fine-tuning result | Part 3, Part 10 |
# | per-patch stratified split, tile identity dropped at load time | adjacent 224x224 patches from four scenes straddled the split; the number measured interpolation | Part 2 |
# | `make_splits` save-loop written **inside a triple-quoted string** | zero split files were ever written; the notebook could not be run from scratch, and only passed because a stale tree was left behind | Part 2 |
# | `filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}"` where the logged key is `val/Accuracy` | Lightning silently substitutes 0 for unknown names, so every checkpoint was named `val_acc=0.00` | Part 7 |
# | `dirpath` reused across runs, `save_weights_only=True` | "Checkpoint directory exists and is not empty"; optimiser state dropped, so no resume | Part 7 |
# | `logger=False`, `devices` never set, run inside Jupyter | a paid 4-GPU allocation ran one process on one GPU with no curves to inspect | Part 6 |
# | five CORINE classes deleted by an unexamined `< 10` threshold, including Peat bogs and Water courses | the model was structurally incapable of predicting the two most Iceland-relevant classes | Part 2 |
# | cells asserting a pasted dict of floats from a previous run | green checks that confirmed a past execution rather than correctness | removed; replaced by gates |
# | zero exercises, zero questions, no gate | a student who never looked at a metric passed | throughout |

# %% [markdown]
# ## Part 0 — Setup
#
# `MPLCONFIGDIR` before matplotlib, one `rng` for the whole notebook, seed recorded.
# The 2025/26 notebook had no `seed_everything` call anywhere, so its "ablation" arms
# differed by whatever the process happened to initialise with.

# %%
from eo_course import paths

print(paths.describe())

# %%
import os
import json
import subprocess
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = paths.matplotlib_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402

from eo_course import patches as pat  # noqa: E402
from eo_course import (radiometry, labels as lab_mod, splits, metrics,  # noqa: E402
                       baselines as bl, results, gates)

SEED = 20180402
rng = np.random.default_rng(SEED)
print(f"seed = {SEED}   (one rng; every arm below receives it)")

# %% [markdown]
# ## Part 1 — What pre-training actually is, in four sentences
#
# The old unit sheet listed "difference between pre-training and fine-tuning" as a
# learning outcome and the notebook never stated it once.
#
# **Pre-training** learns weights from unlabelled data with a self-supervised
# objective — Prithvi masks out most of a Sentinel-2 patch and learns to reconstruct
# it, over ~6.5 million global image tiles. **Fine-tuning** reuses those weights and
# trains (most of) them on your labelled task, at a learning rate two to three orders
# of magnitude below scratch training. The **backbone** is the pretrained ViT encoder;
# the **head** is the small module you add that maps its output tokens to your class
# logits. **Freezing** means the backbone's weights are held fixed and only the head
# trains — which turns the backbone into a fixed feature extractor, i.e. a much
# larger **linear probe** than the one in `eo_course.baselines`.
#
# In this lab the model is `ClassificationTask` wrapping an `EncoderDecoderFactory`
# with an `IdentityDecoder`. "Identity" means the decoder does nothing: it passes the
# encoder's token sequence straight to a `ScalarHead`, which pools the tokens and
# applies one linear layer. So the 2025/26 setup was, architecturally, a linear probe
# on top of a 303 M-parameter encoder — which is exactly why the comparison against
# `baselines.linear_probe` in Part 3 is the comparison that matters.
#
# mIoU is **not** the metric here. This is image-level classification, so the metric
# set is the one Lab 7 defined: overall accuracy, balanced accuracy, macro-F1, all on
# the fixed training class set.

# %% [markdown]
# ## Part 2 — The Lab 4.2 hand-off, and a split the foundation model does not escape
#
# A foundation model gets no exemption from leakage. Prithvi saw millions of tiles
# during pre-training; whether it saw *your* tile is unknowable, but you can still
# guarantee that no patch in your test set is a spatial neighbour of a patch in your
# train set. That is the only claim you are entitled to make.
#
# The 2025/26 notebook's `make_splits` ended with its save loop **inside a
# triple-quoted string** — a no-op docstring, not code. Zero split files were written.
# The cells that then loaded them passed only because a stale tree from an earlier,
# uncommented run was still on scratch. A notebook that cannot be run from scratch is
# not a notebook; it is a recording.

# %%
PATCH_DIR = paths.require_existing(
    paths.training_data_dir(),
    "Lab 4.2 patch archives (paths.training_data_dir())")
ARCHIVES = sorted(PATCH_DIR.glob(pat.SCENE_GLOB))
if not ARCHIVES:
    raise FileNotFoundError(
        f"no {pat.SCENE_GLOB} archives in {PATCH_DIR}. Run Lab 4.2 Part 4; it writes "
        "scene/row/col alongside the pixels, and without them no honest split exists.")
print(f"  {len(ARCHIVES)} archives in {PATCH_DIR}")
for a in ARCHIVES:
    print(f"    {a.name:<44} {a.stat().st_size / 1e6:7.1f} MB")

# %%
PS = pat.load_all_scenes(PATCH_DIR, expect_scenes=len(ARCHIVES))
X_HWC = PS.patches
SCENES = np.asarray(PS.scene)
print(f"  patches   : {X_HWC.shape}  dtype {X_HWC.dtype}")
print(f"  bands     : {list(PS.bands)}")
print(f"  scenes    : {sorted(set(SCENES.tolist()))}")
print(f"  provenance: scene={PS.scene is not None} row={PS.row is not None} "
      f"col={PS.col is not None}")

# %%
# What units are these, actually? Everything downstream depends on the answer.
lo, hi = float(X_HWC.min()), float(X_HWC.max())
print(f"  stored range: [{lo:.4f}, {hi:.4f}]")
if hi <= 1.0 + 1e-3:
    UNITS = "reflectance"
    X_REFL = X_HWC
    print("  -> surface reflectance in [0,1] (Lab 4.1 divided by QUANTIFICATION_VALUE).")
elif hi > 100.0:
    UNITS = "dn"
    X_REFL = radiometry.dn_to_reflectance(X_HWC)
    print(f"  -> raw L2A DN. Converted once with radiometry.dn_to_reflectance; "
          f"X_REFL now spans [{float(X_REFL.min()):.4f}, {float(X_REFL.max()):.4f}].")
    print("  Everything downstream (baselines, the units gate, the dataset) reads X_REFL,")
    print("  never X_HWC, so the conversion cannot be applied twice. Lab 5.1's bug was")
    print("  exactly a second division: an already-normalised array times 1e-4.")
else:
    raise ValueError(
        f"range [{lo:.4f}, {hi:.4f}] is neither reflectance nor DN. This archive was "
        "percentile-stretched or rescaled by hand. Re-run Lab 4.2 — a baked-in stretch "
        "is permanent and split-independent, and it is what broke the 2025/26 lab.")
radiometry.assert_reflectance_range(X_REFL[:4096], "X_REFL")
print("  assert_reflectance_range(X_REFL) passed.")

# %%
# The class set comes from the TRAINING split only. The 2025/26 notebook deleted every
# class with fewer than 10 samples *before* splitting, which silently removed Moors and
# heathland, Sparsely vegetated, Peat bogs, Water courses and Transitional
# woodland-shrub -- including the two most relevant to Iceland.
# Two scenes, not one: metrics.evaluate bootstraps by scene, and a single-scene test set
# makes every confidence interval degenerate (lower bound == upper bound). Hold out more
# if you have more scenes -- and say in your report why you chose these.
ALL_SCENES = sorted(set(SCENES.tolist()))
TEST_SCENES = tuple(ALL_SCENES[:2])   # TODO(you): defend this choice
print(f"  available scenes  : {ALL_SCENES}")
print(f"  held-out test scenes: {TEST_SCENES}")
print("  TODO(you): name one property of each held-out scene that makes it a hard test")
print("  (sensor date, illumination, land-cover mix). A test set you cannot argue for")
print("  is a test set you will be tempted to swap once you see the number.")

# %%
# group_block_split refuses a split in which any class has fewer than min_per_class test
# samples, and it should: per-class recall on n=1 is noise with a label on it. But the
# 2025/26 fix for that refusal was to delete every class with fewer than 10 samples
# *before* splitting, which silently removed Moors and heathland, Sparsely vegetated,
# Peat bogs, Water courses and Transitional woodland-shrub -- including the two most
# relevant to Iceland. Here the split is taken first with the support check deferred,
# then rare classes are merged on TRAIN support only and the merged bucket is reported.
MANIFEST = splits.group_block_split(
    SCENES, rows=PS.row, cols=PS.col, block=10, val_ratio=0.15,
    test_scenes=TEST_SCENES, seed=SEED, labels=None)
print(f"  method        : {MANIFEST.method}")
print(f"  manifest_hash : {MANIFEST.manifest_hash}")
print(f"  n train/val/test: {MANIFEST.train_idx.size} / {MANIFEST.val_idx.size} / "
      f"{MANIFEST.test_idx.size}")
print(f"  scenes in test: {sorted(set(SCENES[MANIFEST.test_idx].tolist()))}")
print(f"  scenes straddling splits: {sorted(map(str, MANIFEST.group_leakage()['train&test']))}")
MANIFEST.write(paths.splits_dir() / f"lab6_{MANIFEST.manifest_hash}.json")

# %%
LM = lab_mod.LabelMap.from_train(PS.labels[MANIFEST.train_idx])
RAW_CODES = LM.codes
_, MERGED_CODES, MERGE = metrics.merge_rare_classes(PS.labels[MANIFEST.train_idx],
                                                    RAW_CODES, gates.MIN_CLASS_SUPPORT)
CODES = tuple(int(c) for c in MERGED_CODES)
if MERGE["merged"]:
    # Apply the SAME mapping to every split. Fitting the merge on train and then
    # leaving val/test on the old codes is the label-space version of fitting a
    # normalisation on test: the class set becomes test-informed.
    rare_other = int(MERGE["rare_other_code"])
    LABELED = PS.labels.copy()
    for c in MERGE["merged"]:
        LABELED[LABELED == int(c)] = rare_other
    print(f"  merged on TRAIN support (<{gates.MIN_CLASS_SUPPORT}): "
          f"{[lab_mod.class_name(c) for c in MERGE['merged']]} -> "
          f"rare-other code {rare_other}")
else:
    LABELED = PS.labels.copy()
    print("  no class fell below the support threshold; nothing merged")
LM = lab_mod.LabelMap(CODES)
# The fixed label set that every metric call receives is the MODEL INDEX set, 0..K-1.
# `LabelMap.labels` is a confusing name: it returns the CORINE *codes*, not the indices.
# Passing codes where indices belong makes metrics.evaluate raise, which is the good
# outcome; the bad outcome is a metric that quietly averages over the wrong rows.
LABELS = np.arange(LM.n_classes)
print(f"  classes (from train only): {LM.n_classes}")
TRAIN_COUNTS = lab_mod.class_counts(LABELED[MANIFEST.train_idx], CODES)
print(f"  train counts: {dict(zip(CODES, TRAIN_COUNTS.tolist()))}")
print(f"  imbalance ratio (train): {lab_mod.imbalance_ratio(TRAIN_COUNTS):.1f}x")
print("  Nothing was deleted. Merging is reported in results.json and changes the")
print("  macro-average, which is exactly the consequence you have to state.")

# %%
# A class can appear in val or test and not in train. LabelMap is built from train, so
# encoding those samples is a hard error -- and silently dropping them from the metric
# denominator while leaving them in the class list is how a macro-average gets
# inflated. They are dropped here, counted, and the count lands in results.json. A NEW
# manifest is then built from the surviving indices, so the hash that goes into
# results.json describes the split you actually trained on, not the split you proposed.
IN_CLASSSET = np.isin(LABELED, np.asarray(CODES, dtype=LABELED.dtype))
TR, VA, TE = (idx[IN_CLASSSET[idx]] for idx in
              (MANIFEST.train_idx, MANIFEST.val_idx, MANIFEST.test_idx))
DROPPED = {
    "train": int(MANIFEST.train_idx.size - TR.size),
    "val": int(MANIFEST.val_idx.size - VA.size),
    "test": int(MANIFEST.test_idx.size - TE.size),
    "labels": sorted(set(LABELED[~IN_CLASSSET].tolist())),
}
print(f"  dropped for a label outside the train class set: {DROPPED}")
assert DROPPED["train"] == 0, "train labels must all be in the train-derived class set"
MANIFEST = splits.SplitManifest(
    seed=MANIFEST.seed, method=MANIFEST.method, train_idx=TR, val_idx=VA, test_idx=TE,
    groups=MANIFEST.groups, blocks=MANIFEST.blocks, n_total=MANIFEST.n_total,
    extra={**MANIFEST.extra, "dropped_offclass": DROPPED,
           "merged_rare": MERGE["merged"]})
MANIFEST.write(paths.splits_dir() / f"lab6_{MANIFEST.manifest_hash}.json")
print(f"  final manifest_hash : {MANIFEST.manifest_hash}")
print(f"  n train/val/test    : {TR.size} / {VA.size} / {TE.size}")

# %%
Y = LM.encode(LABELED, strict=False)
NAMES = LM.names
COUNTS = splits.per_split_class_counts(LABELED, MANIFEST, CODES)
thin = [(LM.label_name(i), COUNTS["test"][i]) for i in range(LM.n_classes)
        if 0 < COUNTS["test"][i] < gates.MIN_CLASS_SUPPORT]
absent = [LM.label_name(i) for i in range(LM.n_classes) if COUNTS["test"][i] == 0]
print(f"  test classes with 0 < n < {gates.MIN_CLASS_SUPPORT}: {thin}")
print(f"  test classes with n == 0: {absent}")
print("  This is the exact condition that made the 2025/26 checkpoint-selection metric")
print("  meaningless: macro-accuracy over a class with one validation sample moves by")
print("  0.1 when that single sample is classified correctly. If the list above is not")
print("  empty, class_support will be red on the gate board and you must merge more.")

# %%
# The evaluation universe. A 303 M ViT at 224x224 cannot score 17 000 test patches
# inside a lab session, and the 2025/26 notebook solved this by capping trainer.test()
# with limit_test_batches while computing nothing else on the same subset -- so the
# printed metric was over an unknown, order-dependent slice of the test set.
#
# Here the subset is chosen ONCE, deterministically, and every consumer (baselines,
# arms, gates, results) reads the same array. MAX_EVAL = 0 means "the whole test split",
# which is what the graded sbatch run must use.
MAX_EVAL = 512          # TODO(you): set 0 for the graded run
if MAX_EVAL and TE.size > MAX_EVAL:
    # Deterministic, no RNG: order by (scene, row, col) and take evenly spaced entries,
    # so the subset spans the scene spatially instead of clustering in one corner. The
    # 2025/26 MAX_PATCHES truncation was the cautionary tale for the other choice.
    scene_rank = np.unique(SCENES, return_inverse=True)[1][TE]
    grid = np.lexsort((PS.col[TE], PS.row[TE], scene_rank))
    EV = TE[grid[np.linspace(0, TE.size - 1, MAX_EVAL).astype(int)]]
else:
    EV = TE
N_TEST_SCENES = int(np.unique(SCENES[EV]).size)
print(f"  evaluation subset : {EV.size} of {TE.size} test patches "
      f"(MAX_EVAL={MAX_EVAL})")
print(f"  scenes represented: {sorted(set(SCENES[EV].tolist()))}")
if N_TEST_SCENES < 2:
    print(f"  WARNING: the test set is {N_TEST_SCENES} scene. metrics.evaluate resamples "
          f"CLUSTERS,")
    print("  not patches, because patches within a scene are correlated. With one "
          "cluster every")
    print("  bootstrap replicate is the same sample, so every confidence interval in "
          "this")
    print("  notebook is DEGENERATE: lower bound == upper bound. The point estimate is "
          "still")
    print("  yours to report; the interval is not a confidence interval. Hold out two "
          "or")
    print("  more scenes (change TEST_SCENES above) before you quote an error bar.")
print(f"  class support     : {np.bincount(Y[EV], minlength=LM.n_classes).tolist()}")
print("  Every number in this notebook is computed on EV. If you change MAX_EVAL you")
print("  must re-run the baselines too, or the gate board compares a model on 512")
print("  samples against baselines on 17 000 -- which is exactly the kind of mismatch")
print("  that produced the 2025/26 report.")

# %% [markdown]
# ## Part 3 — Baselines first, not last
#
# The 2025/26 labs printed a majority-class count, called it a baseline, and never
# scored it with the same code as the model. Lab 6 then reported test accuracy exactly
# 1/10 against a majority baseline of 0.538 and nothing caught it.
#
# Run the baselines **before** the fine-tuning. If the number you are about to spend
# an hour of A100 time producing cannot beat a logistic regression on flattened
# pixels, that is the finding, and it is a better finding than a fabricated win.

# %%
# %%
# Channels-first, in reflectance, once. Everything downstream reads X_CHW; nothing
# divides by QUANTIFICATION_VALUE again.
#
# "ndvi" is dropped from the baseline list, and the reason is worth knowing:
# baselines.ndvi_rule guesses the channel axis by testing whether the spatial side is at
# least as large as the channel count. With PATCH_SIZE=3 and 4 bands that test is
# ambiguous in both layouts and it indexes out of bounds. Rather than wrap it in a
# try/except and lose a baseline silently, it is excluded by name and the exclusion is
# recorded. With PATCH_SIZE >= 4 it works; raise the patch size in Lab 4.2 and put it
# back.
WHICH_BASE = tuple(k for k in bl.ALL_BASELINES if k != "ndvi")
X_CHW = np.ascontiguousarray(np.transpose(X_REFL, (0, 3, 1, 2)), dtype=np.float32)
BASE = bl.run_all(
    x_train=X_CHW[TR], y_train_codes=LABELED[TR],
    x_test=X_CHW[EV], y_test_codes=LABELED[EV],
    labels=LABELS, codes=CODES,
    train_scene_ids=SCENES[TR], test_scene_ids=SCENES[EV],
    seed=SEED, which=WHICH_BASE)
print(f"  {'baseline':<20} {'overall':>8} {'balanced':>9} {'macro_F1':>9}")
for name, d in sorted(BASE.items(), key=lambda kv: -kv[1]["macro_f1"]):
    print(f"  {name:<20} {d['overall_acc']:>8.4f} {d['balanced_acc']:>9.4f} "
          f"{d['macro_f1']:>9.4f}")
BEST_BASE = max(BASE, key=lambda k: BASE[k]["macro_f1"])
print(f"\n  bar to clear: macro_F1 > {BASE[BEST_BASE]['macro_f1']:.4f} "
      f"(+{gates.MIN_MACRO_F1_OVER_BASELINE}) and balanced_acc > "
      f"{BASE[BEST_BASE]['balanced_acc']:.4f} (+{gates.MIN_BALANCED_ACC_OVER_CHANCE})")

# %% [markdown]
# ## Part 4 — The mechanism of the normalisation failure, computed exactly
#
# "ML models work better with normalised inputs" is not the lesson, and the 2025/26
# notebooks never said anything better. Here is the mechanism, with numbers you can
# recompute on your own patches in the next two cells.
#
# Prithvi-EO-2.0 was pretrained on **six** bands in the order
# `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2` — no DEM, no slope, no aspect. Its
# pretraining statistics are stated in **DN units** (reflectance x 10000):
#
# ```
# mean = [1087, 1342, 1433, 2734, 1958, 1363]
# std  = [2248, 2179, 2178, 1850, 1242, 1049]
# ```
#
# So the encoder expects inputs with per-band mean ~0 and per-band std ~1, and the
# transform that produces them is `z = (DN - mean) / std`. Now feed it reflectance in
# [0,1] with no transform at all, as the 2025/26 config did.
#
# It is worth being precise about what breaks, because the usual explanation is wrong.
# Scaling is an invertible linear map: it does **not** destroy your class information.
# The ratio of between-class scatter to within-class scatter is identical before and
# after the transform — the next cell measures it and you should see it come out the
# same. A network trained *from scratch* on [0,1] input works fine; Lab 5's CNN did.
#
# What the scale error destroys is the **operating point**. Prithvi's LayerNorms,
# attention softmaxes and MLPs were calibrated on inputs centred at 0 with unit spread
# per band. Fed reflectance, every input instead arrives at a large negative offset
# (roughly `-mean/std` per band, i.e. order -0.5 to -1) with a spread around that offset
# that is `QUANTIFICATION_VALUE` times smaller than the unit spread the network expects.
# LayerNorm subtracts a mean computed per token across the embedding dimension, so it
# cannot undo a per-band displacement that is identical for every token. The result is a
# network evaluated far outside the region of input space its pretraining explored. That
# is the shape of the 2025/26 failure: not a mediocre model, but one that predicted a
# single class for all 455 test samples, which is what a saturated encoder looks like
# from the outside.
#
# The next cell measures both halves on your own patches: the separability ratio, which
# scaling should broadly preserve, and the offset-and-spread mismatch, which it does not.
# Read the printed numbers before you accept the story — if the offset on your data is
# small, say so, because then the story needs revising.

# %%
MEAN4, STD4 = radiometry.prithvi_norm(4, dn_units=True)
MEAN4R, STD4R = radiometry.prithvi_norm(4, dn_units=False)
print(f"  PRITHVI_V2_BANDS_4B : {list(radiometry.PRITHVI_V2_BANDS_4B)}")
print(f"  mean (DN units)     : {MEAN4}")
print(f"  std  (DN units)     : {STD4}")
print(f"  mean (reflectance)  : {[round(m, 4) for m in MEAN4R]}")
print(f"  std  (reflectance)  : {[round(s, 4) for s in STD4R]}")
print("  Same transform, two unit conventions. Pick the one matching your stored units")
print("  or you divide by 10000 twice -- which is the Lab 5.1 bug in the other direction.")

# %%
sub = TR[:6000]
dn = radiometry.reflectance_to_dn(X_REFL[sub])
z_ok = (dn - np.array(MEAN4, np.float32)) / np.array(STD4, np.float32)
z_bad = (X_REFL[sub] - np.array(MEAN4, np.float32)) / np.array(STD4, np.float32)


def fisher_ratio(arr, y):
    """trace(Sb)/trace(Sw) on flattened patches: separability, ignoring scale."""
    f = arr.reshape(arr.shape[0], -1).astype(np.float64)
    ks, inv = np.unique(y, return_inverse=True)
    mus = np.stack([f[inv == i].mean(0) for i in range(ks.size)])
    pri = np.array([(inv == i).mean() for i in range(ks.size)])
    trace_sb = float((pri[:, None] * (mus - f.mean(0)) ** 2).sum())
    trace_sw = float(((f - mus[inv]) ** 2).mean(0).sum())
    return trace_sb / max(trace_sw, 1e-12)


fr_refl = fisher_ratio(X_REFL[sub], Y[sub])
fr_norm = fisher_ratio(z_ok, Y[sub])
print(f"  Fisher ratio, reflectance as stored   : {fr_refl:.4f}")
print(f"  Fisher ratio, Prithvi-normalised      : {fr_norm:.4f}")
print(f"  ratio                                 : {fr_norm / fr_refl:.4f}  "
      "(~1.0: the transform preserves separability)")
print(f"\n  mean z under the correct pipeline     : "
      f"{np.abs(z_ok.mean(axis=(0, 2, 3))).mean():.4f}  (should be ~0)")
print(f"  mean z fed as-is                      : "
      f"{np.abs(z_bad.mean(axis=(0, 2, 3))).mean():.4f}  per band")
print(f"  std  z fed as-is                      : "
      f"{z_bad.std(axis=(0, 2, 3)).mean():.4f}  (should be ~1)")
print(f"  chance = 1/{LM.n_classes} = {1 / LM.n_classes:.4f}")
print("  So the information survives and the operating point does not. Whether that is")
print("  enough to collapse the model is exactly what Exercise 1 measures.")

# %% [markdown]
# ## Part 5 — The mechanism of the band-order failure, reproduced
#
# TerraTorch's `select_patch_embed_weights` does, in essence:
#
# ```python
# temp_weight[:, i] = patch_embed_weight[:, pretrained_bands.index(band)]
# ```
#
# and if `band` is not in `PRETRAINED_BANDS` it leaves that filter at its Xavier
# initialisation. `HLSBands.try_convert_to_hls_bands_enum("B02")` swallows the
# `ValueError` and hands back the raw string, which is not in `PRETRAINED_BANDS`.
#
# The cell below is a local model of that rule — six identity filters, apply the
# documented selection, read off where each output channel came from. It is not the
# library; it is the arithmetic the library performs, and it runs without TerraTorch so
# you can see the failure before you spend GPU time on it. Part 8 then checks the real
# weight tensor and confirms it.

# %%
PRETRAINED_BANDS = list(radiometry.PRITHVI_V2_BANDS)


def try_convert_to_hls(name):
    """Model of HLSBands.try_convert_to_hls_bands_enum: unknown names pass through."""
    return name if name in PRETRAINED_BANDS else name


def select_patch_embed(bands):
    """Return (source position per output channel or None, n_xavier)."""
    src = []
    for b in bands:
        name = try_convert_to_hls(b)
        src.append(PRETRAINED_BANDS.index(name) if name in PRETRAINED_BANDS else None)
    return src, sum(1 for s in src if s is None)


ARMS_BANDS = {
    "correct": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
    "swapped": ["BLUE", "RED", "GREEN", "NIR_NARROW"],
    "invalid": ["B02", "B03", "B04", "B08"],
}
for tag, bands in ARMS_BANDS.items():
    src, nx = select_patch_embed(bands)
    print(f"  {tag:<8} {str(bands):<52} src={src} xavier_filters={nx}/4")

# %% [markdown]
# Read the three lines carefully.
#
# * `correct` — source positions `[0,1,2,3]`: every output channel receives the
#   pretrained filter for the band it actually contains.
# * `swapped` — source positions `[0,2,1,3]`: the GREEN filter is applied to RED
#   pixels and the RED filter to GREEN pixels. Nothing raises. The model still trains.
#   It just starts from a representation that is wrong in exactly the two bands that
#   separate vegetation from soil.
# * `invalid` — `xavier_filters=4/4`: the pretrained input layer is **entirely
#   discarded**. This was the *commented-out alternative* in the 2025/26 config, and it
#   looked like the safer option because it named the real Sentinel-2 bands.
#
# The 2025/26 config shipped `swapped` and commented out `invalid`. Neither student nor
# instructor had any way to tell from the notebook's output.

# %% [markdown]
# ## Part 6 — Where the compute is, and what breaks if you ignore it
#
# The 2025/26 run printed, in its own output, the indictment of its own setup:
#
# ```
# CUDA_VISIBLE_DEVICES: [0,1,2,3]
# Trainer will use only 1 of 4 GPUs because it is running inside an interactive
# / notebook environment
# ```
#
# That is a paid four-GPU allocation running one process on one GPU, inside Jupyter,
# with `logger=False` so there were no curves to inspect afterwards.
#
# * **`slurm/train_jureca.sbatch`** — one arm per job. JURECA-DC partitions are
#   `dc-gpu` (A100-40GB) and `dc-cpu`, lower case. `--gres=gpu:1` is mandatory; a GPU
#   job without it gets a node that cannot see a GPU.
# * **`slurm/submit_sweep.sbatch`** — fans an (arm x seed) grid out with
#   `sbatch --array`, one full GPU per cell, so one failing arm cannot eat the others'
#   wall clock and the seed-variance gate can actually see three seeds.
# * **Jupyter-JSC caps you at one GPU** even when it allocated four, and it only warns.
# * **Fine-tuning a 303 M-parameter ViT on a login node** is not slow, it is wrong:
#   login nodes are not scheduled, so you are stealing CPU from everyone while your
#   single-process job saturates one GPU; the shared filesystem stalls on 8k small
#   `.npz` reads per epoch; and your session dies with the SSH connection, taking the
#   run and the checkpoint with it.
#
# The cells below run the arms in-process so the notebook is self-contained for a
# short lab session. For the graded sweep, run the same arms through `sbatch` and let
# `results.json` collect them.

# %%
def squeue_mine():
    """Show your own jobs. subprocess, not `!`, because each `!` line is a new shell."""
    try:
        out = subprocess.run(["squeue", "-u", os.environ["USER"], "--noheader",
                              "--format=%.12j %.8T %.10M %.6D %R"],
                             capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"  squeue unavailable here: {type(exc).__name__}")
        return
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    print(f"  {len(lines)} of your jobs on the cluster")
    for ln in lines[:12]:
        print(f"    {ln}")


squeue_mine()

# %% [markdown]
# ## Part 7 — Predict before you run
#
# Write a number in every slot **before** running Part 8. A `None` is scored as not
# attempted. A wrong number with a correct mechanism is scored higher than a lucky
# guess with no explanation. You are graded on the prediction, not on the output.
#
# ### Exercise 1 — the normalisation catastrophe
#
# Arm `unnormalised` is the 2025/26 setup reproduced faithfully: your Lab 4.2 patches
# fed straight into `ClassificationTask` with no transform, `backbone_pretrained=True`,
# correct band names, `monitor="val/loss"`, `ckpt_path="best"`, same split, same seed,
# same epochs as the normalised arm. The only difference is the transform.
#
# 1. What **overall accuracy** does `unnormalised` reach on your held-out test scenes?
# 2. What **overall accuracy** does `normalised` reach?
# 3. What is `balanced_acc(unnormalised)` minus `1/K`? (An exact 0 means the model is
#    statistically indistinguishable from a constant predictor.)
#
# ### Exercise 2 — band order
#
# 4. How many **macro-F1 points** does the `swapped` arm lose relative to `correct`?
# 5. How many does `invalid` lose relative to `correct`?
# 6. Which of the two arms loads **zero** pretrained patch-embed filters?
#
# ### Exercise 3 — is the pre-training doing anything?
#
# 7. Arm `random_init` is the identical config with `backbone_pretrained=False`. What
#    is `macro_F1(normalised) - macro_F1(random_init)`? It may be small. It may be
#    negative. Predict it anyway — this is the number the 2025/26 lab never measured,
#    and a small gap is a publishable finding about *your* task, not an embarrassment.

# %%
PREDICTIONS = {
    # TODO(you): fill in BEFORE running Part 8.
    "acc_unnormalised": None,
    "acc_normalised": None,
    "balanced_minus_chance_unnormalised": None,
    "macro_f1_delta_swapped_vs_correct": None,
    "macro_f1_delta_invalid_vs_correct": None,
    "arm_with_zero_pretrained_filters": None,
    "macro_f1_delta_pretrained_vs_random": None,
    "why": "TODO(you): for each disagreement, name the mechanism, not the outcome.",
}

# %% [markdown]
# ## Part 8 — The arms
#
# TerraTorch is a heavy, GPU-oriented dependency and is **not** installed in every
# kernel this notebook is opened in. The concept cells above run anywhere; the cells
# below do not, and they say so loudly rather than skipping. A silent skip is the same
# class of bug this whole notebook exists to fix.

# %%
import importlib.util

TT_SPEC = importlib.util.find_spec("terratorch")
if TT_SPEC is None:
    raise ModuleNotFoundError(
        "terratorch is not installed in this kernel, and the fine-tuning arms below "
        "cannot be substituted. On JURECA: activate the Lab 2 course virtualenv "
        f"({paths.user_project('envs')}/ml_eo_course), then "
        "`pip install terratorch==1.2.4` (pyproject.toml pins 1.0.1 -- the 2025/26 "
        "notebook printed 1.2.4 while the pin said 1.0.1, so the declared and executed "
        "environments disagreed; check `pip show terratorch` and record the version "
        "you actually ran). Then run this notebook from a Jupyter-JSC session on a "
        "compute node, or better, from slurm/train_jureca.sbatch.")
import importlib.metadata  # noqa: E402

import torch  # noqa: E402

print(f"terratorch {importlib.metadata.version('terratorch')}   "
      f"torch {torch.__version__}   cuda={torch.cuda.is_available()}   "
      f"devices={torch.cuda.device_count()}")

# %%
import lightning.pytorch as pl  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

TT_INPUT = 224   # multiple of Prithvi's 14x14 patch_embed -> 256 spatial tokens

# %%
class PatchDN(Dataset):
    """(N,C,H,W) reflectance -> model-ready float32 + int64 target.

    Three separate conversions happen here, and the 2025/26 lab was silent on each:
    the unit convention (reflectance vs DN), the normalisation, and the spatial resize
    a ViT patch-embed requires.
    """

    def __init__(self, x, y, dn, normalise, size=TT_INPUT):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.ascontiguousarray(y, dtype=np.int64)
        self.dn = dn
        self.normalise = normalise
        self.size = size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        a = self.x[i]                                  # (C,H,W) reflectance
        if self.dn:
            a = a * radiometry.QUANTIFICATION_VALUE
        if self.normalise:
            # (C,1,1), not (C,): a bare (C,) vector broadcasts against the LAST axis,
            # which here is W. With a 3x3 patch that raises, but on a patch whose width
            # equals the band count it silently "works" and applies band 0's mean to
            # every pixel of column 0. Same family as the band-order swap in Part 5:
            # shape-compatible, semantically wrong, and invisible in the metrics.
            shape = (a.shape[0],) + (1,) * (a.ndim - 1)
            mean = np.asarray(MEAN4, np.float32).reshape(shape)
            std = np.asarray(STD4, np.float32).reshape(shape)
            a = (a - mean) / std
        if a.shape[-1] != self.size:
            t = torch.from_numpy(a).unsqueeze(0)
            t = torch.nn.functional.interpolate(t, size=(self.size, self.size),
                                                mode="bilinear", align_corners=False)
            a = t.squeeze(0).numpy()
        return torch.from_numpy(np.ascontiguousarray(a)), int(self.y[i])

# %%
_DM_CACHE = {}


def make_dm(dn, normalise, batch_size=16, num_workers=2):
    """DataModule built from MANIFEST only -- splits never originate here.

    Cached per (dn, normalise): fit and test must consume the *same* arrays. The
    2025/26 notebook rebuilt them from two different code paths.
    """
    key = (dn, normalise, batch_size, num_workers)
    if key not in _DM_CACHE:
        ds = {}
        for name, idx in (("train", TR), ("val", VA), ("test", EV)):
            ds[name] = PatchDN(X_CHW[idx], Y[idx], dn=dn, normalise=normalise)
        _DM_CACHE[key] = pl.LightningDataModule.from_datasets(
            ds["train"], ds["val"], ds["test"], batch_size=batch_size,
            num_workers=num_workers)
    return _DM_CACHE[key]


print(f"  Prithvi's patch_embed is 14x14 with stride 14, so the input side must be a")
print(f"  multiple of 14. Course patches are {PS.patch_size}x{PS.patch_size}, so they")
print(f"  are bilinearly upsampled to {TT_INPUT}x{TT_INPUT} -> "
      f"{(TT_INPUT // 14) ** 2} spatial tokens.")
print(f"  Those tokens are {TT_INPUT ** 2 // PS.patch_size ** 2}x-replicas of "
      f"{PS.patch_size ** 2} distinct pixels, so the ViT's spatial attention has almost")
print("  nothing to attend over. That is a property of this task setup, not of Prithvi,")
print("  and it belongs in your write-up next to the pretrained-vs-random gap.")
print("  dn=True, normalise=True  -> the only arm whose units match pretraining")
print("  dn=False, normalise=False -> the 2025/26 arm: [0,1] straight into the encoder")

# %%
RUN_ROOT = paths.run_dir(f"lab6_{MANIFEST.manifest_hash}")
paths.ensure(RUN_ROOT)
print(f"  run root: {RUN_ROOT}")
print("  Every arm gets its own subdirectory. The 2025/26 run reused one shared")
print("  training_logs directory and Lightning warned 'exists and is not empty' --")
print("  which is how a checkpoint from arm A silently becomes arm B's 'best' model.")

# %%
from terratorch.tasks import ClassificationTask  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingLR  # noqa: E402

BACKBONE = "prithvi_eo_v2_300"
LR = 1e-4
MAX_EPOCHS = 3          # lab-session budget; the graded sweep uses 10 via sbatch


class LabTask(ClassificationTask):
    """ClassificationTask with an explicitly epoch-stepped cosine schedule.

    Lightning's default scheduler interval is "step". The 2025/26 Lab 5.2 built
    CosineAnnealingLR over the epoch budget without saying "epoch", so with 625
    steps per epoch and a 100-epoch T_max the learning rate completed ~625 full cosine
    cycles instead of annealing once. Same trap, new lab, so it is set explicitly here.
    """

    def configure_optimizers(self):
        cfg = super().configure_optimizers()
        if isinstance(cfg, dict):
            opt = cfg["optimizer"]
        elif isinstance(cfg, tuple):
            opt = cfg[0][0] if isinstance(cfg[0], (list, tuple)) else cfg[0]
        else:
            opt = cfg
        sched = CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched,
                                                   "interval": "epoch"}}


# %%
def make_task(bands, pretrained=True, freeze=False):
    return LabTask(
        model_factory="EncoderDecoderFactory",
        model_args={
            "backbone": BACKBONE,
            "backbone_pretrained": pretrained,
            "backbone_bands": bands,
            "decoder": "IdentityDecoder",
            "num_classes": LM.n_classes,
        },
        class_names=NAMES,
        loss="ce",
        optimizer="AdamW",
        lr=LR,
        freeze_backbone=freeze,
    )


# %%
print(f"  backbone={BACKBONE}  lr={LR}  max_epochs={MAX_EPOCHS}  "
      f"num_classes={LM.n_classes}")
print("  freeze_backbone=False means all 303 M parameters update. Setting it True calls")
print("  model.freeze_encoder() and trains only the head -- a linear probe on a")
print("  pretrained ViT. It is a legitimate arm (D4 in the audit) and it is not run")
print("  here only because the GPU budget is already spent on the three fatal defects.")

# %%
def make_ckpt(tag, out):
    """Checkpoint callback for one arm, with the 2025/26 filename trap explained.

    "{val/loss:.4f}" interpolates correctly in Lightning, but the "/" is then treated as
    a path separator and the write fails. The 2025/26 config used "{val_acc:.2f}", which
    is not a logged key at all: Lightning silently substitutes torch.tensor(0) for
    unknown names, so every checkpoint was written as val_acc=0.00 and the ranking was
    invisible in the filenames. The monitored value is therefore printed by run_arm.
    """
    return ModelCheckpoint(
        dirpath=str(out / "ckpt"), filename=f"{tag}-{{epoch}}",
        monitor="val/loss", mode="min", save_top_k=1, save_weights_only=False)


# %%
def run_arm(tag, bands, dn, normalise, pretrained=True, freeze=False):
    """One ablation arm. Returns (trainer.test() output, per-arm run directory)."""
    out = RUN_ROOT / tag
    paths.ensure(out / "ckpt")
    task = make_task(bands, pretrained=pretrained, freeze=freeze)
    ckpt = make_ckpt(tag, out)
    stop = pl.callbacks.EarlyStopping(monitor="val/loss", patience=2, mode="min")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, max_epochs=MAX_EPOCHS, logger=True,
        log_every_n_steps=5, callbacks=[ckpt, stop], default_root_dir=str(out),
        limit_train_batches=60, limit_val_batches=20)
    trainer.fit(model=task, datamodule=make_dm(dn, normalise))
    # ckpt_path="best": without it Lightning tests the FINAL-epoch weights and the
    # entire monitor= machinery you configured is never consulted.
    res = trainer.test(model=task, datamodule=make_dm(dn, normalise),
                       ckpt_path="best")
    if ckpt.best_model_score is not None:
        print(f"  [{tag}] best val/loss={float(ckpt.best_model_score):.4f} at "
              f"{Path(ckpt.best_model_path).name}")
    else:
        print(f"  [{tag}] no checkpoint was saved -- predict_arm will fail, not pass")
    return res, out


ARMS = {
    "unnormalised": dict(bands=ARMS_BANDS["correct"], dn=False, normalise=False),
    "normalised": dict(bands=ARMS_BANDS["correct"], dn=True, normalise=True),
    "swapped": dict(bands=ARMS_BANDS["swapped"], dn=True, normalise=True),
    "invalid_bands": dict(bands=ARMS_BANDS["invalid"], dn=True, normalise=True),
    "random_init": dict(bands=ARMS_BANDS["correct"], dn=True, normalise=True,
                        pretrained=False),
}
print(f"  {len(ARMS)} arms: {list(ARMS)}")
print("  One arm per SLURM task in the graded sweep -- never a loop inside one job.")

# %%
# The committed notebooks/iceland-ml/lab6_config.yaml is NOT a TerraTorch config and is
# not edited here. `terratorch fit -c lab6_config.yaml` fails immediately: it has no
# model.class_path and no trainer section, so the Lightning CLI has nothing to
# instantiate; and model.in_channels, model.patch_size, training.epochs and
# data.transforms are read by zero cells in the old notebook (training.epochs: 1 was
# contradicted by max_epochs=2 two cells later). This cell writes a real one into YOUR
# run directory, so the CLI path and the notebook path cannot drift.
TT_CONFIG = {
    "model": {
        "class_path": "terratorch.tasks.ClassificationTask",
        "init_args": {
            "model_factory": "EncoderDecoderFactory",
            "model_args": {
                "backbone": BACKBONE, "backbone_pretrained": True,
                "backbone_bands": ARMS_BANDS["correct"], "decoder": "IdentityDecoder",
                "num_classes": LM.n_classes},
            "class_names": NAMES, "loss": "ce", "optimizer": "AdamW", "lr": LR,
            "freeze_backbone": False,
        },
    },
    "data": {"class_path": "torchgeo.datamodules.NonGeoDataModule",
             "init_args": {"batch_size": 16, "num_workers": 2}},
    "trainer": {
        "accelerator": "gpu", "devices": 1, "max_epochs": 10, "logger": True,
        "log_every_n_steps": 5,
        "callbacks": {"ckpt": {"class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                               "init_args": {"monitor": "val/loss", "mode": "min",
                                             "save_top_k": 1}}},
    },
}
TT_YAML = RUN_ROOT / "lab6_terratorch.yaml"
TT_YAML.write_text(yaml.dump(TT_CONFIG, sort_keys=False), encoding="utf-8")
print(f"  wrote {TT_YAML}")
print("  Run it with:  terratorch fit -c " + TT_YAML.name +
      "  (from sbatch, never from a login node)")

# %%
ARM_RESULTS = {}
for tag, kw in ARMS.items():
    res, out = run_arm(tag, **kw)
    ARM_RESULTS[tag] = {"test": res[0] if res else {}, "dir": str(out)}
    print(f"  [{tag}] test/loss={res[0].get('test/loss', float('nan')):.4f}  "
          f"test/Accuracy={res[0].get('test/Accuracy', float('nan')):.4f}")

# %% [markdown]
# ### Turning Lightning's numbers into the course's numbers
#
# `trainer.test()` returns TerraTorch's own keys: `test/Accuracy` is
# `MulticlassAccuracy(average="macro")`, i.e. **balanced accuracy**, not overall
# accuracy, and `test/Accuracy_Micro` is not overall accuracy either. The 2025/26
# notebook printed one of them under the label "Accuracy" and nobody mapped it to
# anything. Recompute everything with `eo_course.metrics.evaluate` from the raw
# predictions, on the fixed training class set, with scene-clustered bootstrap CIs —
# the same code that scored the baselines.

# %%
def load_arm(tag, bands, freeze=False):
    """Rebuild the arm's architecture and load its best checkpoint onto the device.

    Built with backbone_pretrained=False on purpose: the checkpoint on disk already
    contains every weight, and rebuilding with pretrained=True would re-download 1.2 GB
    per arm only to overwrite it.
    """
    out = RUN_ROOT / tag
    ckpts = sorted((out / "ckpt").glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"no checkpoint under {out / 'ckpt'}. run_arm('{tag}') did not complete; "
            "do not substitute the in-memory final-epoch weights -- that is defect A7.")
    task = make_task(bands, pretrained=False, freeze=freeze)
    state = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    task.load_state_dict(state["state_dict"])
    return task.to("cuda" if torch.cuda.is_available() else "cpu").eval(), ckpts[0]


# %%
@torch.no_grad()
def predict_arm(tag, bands, dn, normalise, pretrained=True, freeze=False):
    """Forward the test split through the arm's best checkpoint; return argmax.

    ``pretrained`` is accepted so this takes the same kwargs as run_arm, and ignored:
    load_arm always rebuilds the architecture unpretrained and then overwrites every
    weight from the checkpoint.
    """
    task, used = load_arm(tag, bands, freeze=freeze)
    dm = make_dm(dn, normalise)
    dm.setup("test")
    preds, seen = [], []
    for x, y in dm.test_dataloader():
        x = x.to(next(task.parameters()).device)
        outp = task.model({"image": x})
        logits = outp["output"] if isinstance(outp, dict) else outp
        preds.append(logits.argmax(dim=1).cpu().numpy())
        seen.append(np.asarray(y).ravel())
    y_seen = np.concatenate(seen)
    if not np.array_equal(y_seen, Y[EV]):
        raise RuntimeError(
            f"{tag}: the test dataloader did not yield the samples in EV order "
            f"(got {y_seen.size} labels, {int((y_seen != Y[EV]).sum())} mismatched). "
            "Metrics would be computed against the wrong targets and still print a "
            "plausible number. Check shuffle in the datamodule.")
    print(f"  {tag:<14} weights from {used.name}")
    del task
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(preds)


ARM_YHAT = {}
for tag, kw in ARMS.items():
    ARM_YHAT[tag] = predict_arm(tag, **kw)
    print(f"  {tag:<14} {ARM_YHAT[tag].shape}  distinct predictions: "
          f"{sorted(set(ARM_YHAT[tag].tolist()))}")

# %%
ARM_METRICS = {}
for tag in ARMS:
    m = metrics.evaluate(Y[EV], ARM_YHAT[tag], LABELS, codes=CODES,
                         names=NAMES, groups=SCENES[EV], seed=SEED)
    ARM_METRICS[tag] = m
    share = float(m.predicted.max() / m.predicted.sum())
    print(f"  {tag:<14} overall={m.overall_acc:.4f} balanced={m.balanced_acc:.4f} "
          f"macroF1={m.macro_f1:.4f} top-class share={share:.1%}")
print(f"\n  chance floor 1/K = {1 / LM.n_classes:.4f}   "
      f"majority baseline overall = {BASE['majority']['overall_acc']:.4f}")

# %% [markdown]
# ## Exercise 1 — the normalisation catastrophe, measured
#
# You should have written your three numbers in Part 7 already. Run this cell and
# record predicted vs actual. **Do not move on until you have written the explanation.**

# %%
u, n = ARM_METRICS["unnormalised"], ARM_METRICS["normalised"]
floor = 1.0 / LM.n_classes
print("  PREDICTED vs ACTUAL — normalisation")
print(f"    1. overall acc, unnormalised : predicted {PREDICTIONS['acc_unnormalised']!r}"
      f"   actual {u.overall_acc:.4f}")
print(f"    2. overall acc, normalised   : predicted {PREDICTIONS['acc_normalised']!r}"
      f"   actual {n.overall_acc:.4f}")
print(f"    3. balanced - 1/K, unnormalised: predicted "
      f"{PREDICTIONS['balanced_minus_chance_unnormalised']!r}   actual "
      f"{u.balanced_acc - floor:+.4f}")
print(f"\n    delta overall  = {n.overall_acc - u.overall_acc:+.4f}")
print(f"    delta balanced = {n.balanced_acc - u.balanced_acc:+.4f}")
print(f"    delta macroF1  = {n.macro_f1 - u.macro_f1:+.4f}")
print(f"    2025/26 reference: test/Accuracy 0.10000000149011612 on 10 classes, "
      f"majority baseline 0.538.")
print("    Write one sentence naming the mechanism. 'Normalisation helps' is not a")
print("    mechanism. The offset and spread you measured in Part 4 are: state the")
print("    operating point the encoder expects and the one it was given.")

# %% [markdown]
# ### If the two arms come out close
#
# That is a real possible outcome and it is not a broken lab. A frozen or lightly
# tuned ViT on 3x3-patch inputs with a `IdentityDecoder` + pooled-token head has very
# little spatial context to exploit, so the *representation* may carry little of the
# class signal and both arms may sit near the linear probe. What you then report is the
# measured delta with its confidence interval, plus the diagnosis of *which* of the two
# failure modes you are in: check the top-class share (collapse), the per-class recall
# table, and whether `unnormalised` predicts a single class at all. If it does not
# collapse, the scale error is being absorbed by something — find out what, and say so.

# %% [markdown]
# ## Exercise 2 — band order, measured
#
# Three arms, one variable: the `backbone_bands` list. Same weights, same split, same
# seed, same epochs, same normalisation.

# %%
c, s, iv = ARM_METRICS["normalised"], ARM_METRICS["swapped"], ARM_METRICS["invalid_bands"]
print("  PREDICTED vs ACTUAL — band order")
print(f"    4. macroF1 delta, swapped vs correct : predicted "
      f"{PREDICTIONS['macro_f1_delta_swapped_vs_correct']!r}   actual "
      f"{s.macro_f1 - c.macro_f1:+.4f}")
print(f"    5. macroF1 delta, invalid vs correct : predicted "
      f"{PREDICTIONS['macro_f1_delta_invalid_vs_correct']!r}   actual "
      f"{iv.macro_f1 - c.macro_f1:+.4f}")
print(f"    6. arm with zero pretrained filters  : predicted "
      f"{PREDICTIONS['arm_with_zero_pretrained_filters']!r}")
print(f"\n  {'arm':<14} {'overall':>8} {'balanced':>9} {'macroF1':>8}")
for tag in ("normalised", "swapped", "invalid_bands"):
    m = ARM_METRICS[tag]
    print(f"  {tag:<14} {m.overall_acc:>8.4f} {m.balanced_acc:>9.4f} {m.macro_f1:>8.4f}")
print(f"\n  Data order is {list(PS.bands)} = {list(radiometry.S2_BANDS_4)} = "
      f"{list(radiometry.PRITHVI_V2_BANDS_4B)}.")

# %% [markdown]
# ### Proving it: did the pretrained patch-embed actually load?
#
# The accuracy delta is *evidence*, but it is confounded: a swap can cost 0 points on a
# task where green and red are interchangeable. The direct proof is in the weight
# tensor and in TerraTorch's own log.
#
# `select_patch_embed_weights` emits, per band it successfully maps,
# `logger.info("Loaded weights for %s in position %d from the checkpoint", band, i)`.
# A band it cannot map is left at Xavier init and emits nothing. So counting those
# lines counts loaded filters. The cell captures them from the real library.

# %%
import logging  # noqa: E402


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def inspect_patch_embed(bands):
    """Build the backbone once, with terratorch's logger captured.

    Returns (n 'Loaded weights' lines, per-filter std, captured lines).
    """
    cap = LogCapture()
    tt = logging.getLogger("terratorch")
    prev, prev_lvl, prev_prop = tt.handlers[:], tt.level, tt.propagate
    tt.handlers, tt.level, tt.propagate = [cap], logging.DEBUG, False
    try:
        task = make_task(list(bands), pretrained=True)
        w = task.model.backbone.patch_embed.weight.detach().float()
        per = w.reshape(w.shape[0], -1).std(dim=1)
        lines = list(cap.lines)
        del task, w, per
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        tt.handlers, tt.level, tt.propagate = prev, prev_lvl, prev_prop
    loaded = [ln for ln in lines if "Loaded weights" in ln]
    return loaded

# %%
for tag in ("correct", "swapped", "invalid"):
    loaded = inspect_patch_embed(ARMS_BANDS[tag])
    print(f"  {tag:<8} {str(ARMS_BANDS[tag]):<52} 'Loaded weights' lines = {len(loaded)}/4")
    for ln in loaded[:4]:
        print(f"      {ln}")
print("\n  Zero 'Loaded weights' lines means every patch-embed filter is at its Xavier")
print("  initialisation: the pretrained input layer was discarded and nothing raised.")

# %% [markdown]
# If `invalid` shows four filters whose std sits at the Xavier bound while `correct`
# shows filters well off it, the pretrained input layer was discarded and the accuracy
# delta in the cell above is the price of a string that looked more honest than
# `"BLUE"`. If the stds look similar, the checkpoint's own scale happens to be close to
# Xavier for this backbone — then the log-line count in the previous cell is the
# evidence, and you should say which of the two you have.

# %% [markdown]
# ## Exercise 3 — is the pre-training doing anything?
#
# `random_init` is the identical config with `backbone_pretrained=False`. Its gap to
# `normalised` is the only number in this lab that measures what you actually paid the
# foundation model for. Predicted-vs-actual below; a small or negative gap is a result.

# %%
p, r0 = ARM_METRICS["normalised"], ARM_METRICS["random_init"]
d = metrics.paired_bootstrap_delta(Y[EV], ARM_YHAT["normalised"],
                                   ARM_YHAT["random_init"], LABELS,
                                   groups=SCENES[EV], seed=SEED)
print("  PREDICTED vs ACTUAL — pre-training")
print(f"    7. macroF1 delta, pretrained - random: predicted "
      f"{PREDICTIONS['macro_f1_delta_pretrained_vs_random']!r}   actual "
      f"{p.macro_f1 - r0.macro_f1:+.4f}")
print(f"    paired bootstrap CI on the delta     : {d}")
if N_TEST_SCENES < 2:
    print("      ^ degenerate: one test scene means one bootstrap cluster, so the two")
    print("        bounds are the same number. 'significant' here is an artifact of the")
    print("        resampling scheme, not evidence. Do not quote it.")
print(f"\n  {'arm':<14} {'overall':>8} {'balanced':>9} {'macroF1':>8}")
for tag in ("normalised", "random_init"):
    m = ARM_METRICS[tag]
    print(f"  {tag:<14} {m.overall_acc:>8.4f} {m.balanced_acc:>9.4f} {m.macro_f1:>8.4f}")
print(f"  {'linear_probe':<14} {BASE['linear_probe']['overall_acc']:>8.4f} "
      f"{BASE['linear_probe']['balanced_acc']:>9.4f} "
      f"{BASE['linear_probe']['macro_f1']:>8.4f}")
print("    If pretrained ~= random, the pre-training is not the source of your score.")
print("    Name what is: your labels, your split, or the head. Then say which.")

# %%
FIG = paths.results_dir() / "lab6_ablation_arms.png"
paths.ensure(FIG.parent)
tags = list(ARM_METRICS) + [f"base:{k}" for k in BASE]
vals = ([ARM_METRICS[t].macro_f1 for t in ARM_METRICS]
        + [BASE[k]["macro_f1"] for k in BASE])
cols = (["#2b6cb0"] * len(ARM_METRICS) + ["#a0aec0"] * len(BASE))
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(range(len(tags)), vals, color=cols)
ax.axhline(1.0 / LM.n_classes, ls="--", c="crimson",
           label=f"chance 1/K = {1 / LM.n_classes:.3f}")
ax.set_xticks(range(len(tags)))
ax.set_xticklabels(tags, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("macro-F1")
ax.set_title("Lab 6 ablation arms vs baselines — same split, same seed")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG, dpi=110)
plt.show()
print(f"  saved {FIG}")

# %% [markdown]
# ## Part 9 — Negative control: the leaky split
#
# Not a bug in this lab — a measurement of what the 2025/26 split was worth. Their split
# was per-patch stratified over four scenes, so adjacent patches sharing pixels and the
# same 100 m CORINE label sat in train and test at once.
#
# **Predict the direction before running.** Will the leaky arm score higher or lower than
# the honest one on macro-F1, and by roughly how much? Write it down.
#
# The design holds the test set **fixed** — the same `EV` patches in both arms — and
# changes only what the model was allowed to train on. That makes it a single-variable
# experiment rather than a demonstration: any difference is attributable to the leakage,
# not to a different or easier test set. "Run the leaky split and watch it go up" is only
# evidence if you can name every variable that moved.

# %%
# The held-out scene, minus the evaluation subset, is exactly what a per-patch split
# would have handed the model for free: same scene, same illumination, same dates,
# spatial neighbours of the test patches.
TEST_SCENE = sorted(set(SCENES[EV].tolist()))
IN_TEST_SCENE = np.isin(SCENES, np.asarray(TEST_SCENE))
USED = np.zeros(len(SCENES), dtype=bool)
USED[EV] = True
LEAK_TRAIN = np.sort(np.flatnonzero(IN_TEST_SCENE & ~USED))
print(f"  test set (both arms)   : {EV.size} patches from {TEST_SCENE}")
print(f"  honest train           : {TR.size} patches, scenes "
      f"{sorted(set(SCENES[TR].tolist()))}")
print(f"  leaky train            : {TR.size} + {LEAK_TRAIN.size} patches, i.e. the "
      f"rest of {TEST_SCENE}")
print("  Same test patches, same model, same seed. Only the training pool changed.")

# %%
LEAK_BASE = bl.run_all(
    x_train=X_CHW[np.concatenate([TR, LEAK_TRAIN])],
    y_train_codes=LABELED[np.concatenate([TR, LEAK_TRAIN])],
    x_test=X_CHW[EV], y_test_codes=LABELED[EV],
    labels=LABELS, codes=CODES,
    train_scene_ids=SCENES[np.concatenate([TR, LEAK_TRAIN])],
    test_scene_ids=SCENES[EV], seed=SEED, which=("majority", "linear_probe"))
HONEST = BASE["linear_probe"]
LEAKED = LEAK_BASE["linear_probe"]
print("  linear_probe on the SAME test patches, honest vs leaky training pool")
print(f"    honest : overall {HONEST['overall_acc']:.4f}  "
      f"balanced {HONEST['balanced_acc']:.4f}  macroF1 {HONEST['macro_f1']:.4f}")
print(f"    leaky  : overall {LEAKED['overall_acc']:.4f}  "
      f"balanced {LEAKED['balanced_acc']:.4f}  macroF1 {LEAKED['macro_f1']:.4f}")
print(f"    inflation: overall {LEAKED['overall_acc'] - HONEST['overall_acc']:+.4f}  "
      f"macroF1 {LEAKED['macro_f1'] - HONEST['macro_f1']:+.4f}")
print("  Write one sentence on why the leaky number is untrustworthy. 'It is higher' is")
print("  not that sentence — the mechanism is that adjacent patches share pixels and the")
print("  same 100 m CORINE label, so the task becomes interpolation, not generalisation.")
print("  If the inflation is small here, say so and explain what that implies about how")
print("  much of the 2025/26 headline was leakage and how much was something else.")

# %%
# The other half of the negative control: the 2025/26 splitting *rule* itself. It is not
# merely worse, it is inadmissible -- gate_split_is_grouped rejects it by method name,
# before any metric is computed. Build it and let the gate say why.
LEAKY = splits.stratified_random_split(LABELED[np.concatenate([TR, VA, TE])],
                                       train_ratio=0.8, val_ratio=0.05,
                                       seed=SEED, min_per_class=2)
print(f"  leaky manifest: {LEAKY.manifest_hash}  method={LEAKY.method}")
print(f"  index disjointness: {[k for k, v in LEAKY.overlaps().items() if v] or 'clean'}")
print("  Indices are disjoint, so a disjointness check passes. That is the check the")
print("  2025/26 notebook ran, and it is the wrong check.")
try:
    gates.gate_split_is_grouped(LEAKY)
    gates.report("gate_rejects_leaky_split", False, "gate accepted a stratified split")
except gates.GateFailure as exc:
    gates.report("gate_rejects_leaky_split", True, str(exc).splitlines()[0][:120])

# %% [markdown]
# ## Part 10 — Gate board
#
# Gates raise. That is the point. In 2025/26 a student could finish this lab, see green
# output, and submit without ever establishing that their model beat guessing, because
# the deliverable ("Functioning end-to-end TerraTorch workflow", "Fine-tuned Prithvi-EO-2.0
# checkpoint") is satisfied by a random model. Each line below is a claim you are making.

# %%
HEADLINE = "normalised"
PRED_PATH = RUN_ROOT / f"lab6_predictions_{HEADLINE}_{MANIFEST.manifest_hash}.npz"
gates.gate_predictions_saved(Y[EV], ARM_YHAT[HEADLINE], PRED_PATH,
                             split_manifest_hash=MANIFEST.manifest_hash)
print(f"  saved raw predictions: {PRED_PATH}")

# %%
probe = EV[:512]
try:
    gates.gate_input_units(X_REFL[probe], name="X_REFL")
    gates.report("units_refl_passes", True,
                 f"X_REFL max={float(X_REFL[probe].max()):.4f}, in [0,1] as required")
except gates.GateFailure as exc:
    gates.report("units_refl_passes", False, str(exc).splitlines()[0])

# The two arrays you must NOT hand a model as if they were reflectance. If either of
# these PASSES, the gate is not working and every unit claim in this course is void.
dn_probe = X_REFL[probe] * radiometry.QUANTIFICATION_VALUE
z_probe = (dn_probe - np.array(MEAN4, np.float32)) / np.array(STD4, np.float32)
for nm, arr in (("dn", dn_probe), ("z", z_probe)):
    try:
        gates.gate_input_units(arr, name=nm)
        gates.report(f"units_gate_rejects_{nm}", False,
                     f"{nm} passed a reflectance range test -- the gate is not working")
    except gates.GateFailure as exc:
        gates.report(f"units_gate_rejects_{nm}", True, str(exc).splitlines()[0][:110])
print("  Three unit conventions, three different numbers, one gate that tells them "
      "apart.")
print("  The 2025/26 lab confused all three and no check existed to catch it.")

# %%
RECORD = {
    "run_id": f"lab6_{HEADLINE}_{MANIFEST.manifest_hash}",
    "split_manifest_hash": MANIFEST.manifest_hash,
    "test_used_for_tuning": False,
}
LINES = gates.run_all_gates(ARM_METRICS[HEADLINE], BASE, MANIFEST, RECORD)
LINES = ["[PASS] input_units: see the three units_gate_rejects lines above"] + LINES
gates.print_gate_board(LINES)

# %% [markdown]
# ### Reading a failed board
#
# * `not_collapsed` red with a share near 100 % is the 2025/26 result. Check input
#   scaling first (Part 4), then whether the checkpoint you tested is the one you
#   trained.
# * `above_chance` red means balanced accuracy is at or below 1/K. Nothing downstream
#   of that is interpretable.
# * `beats_baselines` red is a finding, not an embarrassment — but only if you report
#   it. A foundation model that loses to `linear_probe` on 3x3 patches is a statement
#   about patch size and context, and it is the statement Lab 5 was missing.
# * `class_support` red means you are averaging macro over classes with n=1. Merge with
#   `metrics.merge_rare_classes` and report the merged class; do not silently drop it.
# * `split_grouped` red means you used the leaky split. It exists as a negative
#   control, not as a fallback.

# %% [markdown]
# ## Part 11 — Deliverable
#
# Every decision lands in `results.json` so it is gradeable rather than asserted: the
# arm table, the band lists, the units convention, the checkpoint-selection criterion,
# the manifest hash, and the honest `test_used_for_tuning: false`.

# %%
RUN_ID = f"lab6_{HEADLINE}_{MANIFEST.manifest_hash}_e{MAX_EPOCHS}"
CONFIG = {
    "backbone": BACKBONE,
    "terratorch_version": importlib.metadata.version("terratorch"),
    "torch_version": torch.__version__,
    "lr": LR, "max_epochs": MAX_EPOCHS, "batch_size": 16,
    "monitor": "val/loss", "ckpt_path_at_test": "best",
    "trainer_limits": {"train": 60, "val": 20},
    "eval_subset_n": int(EV.size), "max_eval": MAX_EVAL,
    "baselines_excluded": ["ndvi (channel-axis heuristic is ambiguous at "
                           f"patch_size={PS.patch_size})"],
    "input_units": f"{UNITS} -> DN -> (DN-mean)/std" if UNITS == "reflectance" else UNITS,
    "mean": MEAN4, "std": STD4,
    "bands_declared": ARMS_BANDS["correct"],
    "bands_in_data": list(PS.bands),
    "arms": {t: {k: (list(v) if k == "bands" else v) for k, v in kw.items()}
             for t, kw in ARMS.items()},
    "split": {"method": MANIFEST.method, "block": 10, "test_scenes": list(TEST_SCENES)},
    "class_set": (f"LabelMap.from_train(train_idx); merge_rare_classes on train support "
                  f">= {gates.MIN_CLASS_SUPPORT}; merged={MERGE['merged']}"),
    "seed": SEED,
}

# %%
try:
    rec = results.record_run(
        RUN_ID, lab="lab6", config=CONFIG,
        split_manifest_hash=MANIFEST.manifest_hash, seed=SEED,
        test_metrics=ARM_METRICS[HEADLINE].to_dict(),
        baselines={k: {kk: vv for kk, vv in v.items()
                       if kk in ("overall_acc", "balanced_acc", "macro_f1", "n")}
                   for k, v in BASE.items()},
        test_used_for_tuning=False, n_seeds=1,
        notes=(f"Prithvi-EO-2.0 fine-tune, {LM.n_classes} classes, "
               f"{len(ARM_METRICS)} arms. Normalisation delta on macro-F1: "
               f"{n.macro_f1 - u.macro_f1:+.4f}. Band-order delta (swapped): "
               f"{s.macro_f1 - c.macro_f1:+.4f}, (invalid names): "
               f"{iv.macro_f1 - c.macro_f1:+.4f}. Pretrained-vs-random delta: "
               f"{p.macro_f1 - r0.macro_f1:+.4f}. Best baseline {BEST_BASE} "
               f"macro-F1 {BASE[BEST_BASE]['macro_f1']:.4f}. "
               "TODO(you): append your predicted-vs-actual explanations here."),
        extra={"arm_metrics": {t: m.to_dict() for t, m in ARM_METRICS.items()},
               "leaky_split": {"manifest_hash": LEAKY.manifest_hash,
                               "linear_probe_macro_f1": LEAKED["macro_f1"],
                               "honest_macro_f1": HONEST["macro_f1"],
                               "n_added_train_patches": int(LEAK_TRAIN.size),
                               "test_set_identical": True},
               "gate_board": LINES,
               "predictions": str(PRED_PATH),
               "per_class_recall": {NAMES[i]: float(ARM_METRICS[HEADLINE].recall[i])
                                    for i in range(LM.n_classes)}})
    print(f"  recorded run_id={rec['run_id']}")
except results.ResultsError as exc:
    print(f"  not recorded: {exc}")
    print("  results.json is append-only. Change the run_id (it encodes the split hash "
          "and epoch budget) rather than overwriting the earlier attempt.")

# %%
print(results.summary_table("lab6"))

# %% [markdown]
# ## Submission checklist
#
# Everything here is an artifact a grader can open. Self-attestation is not an artifact.
#
# * `paths.run_dir("lab6_<manifest_hash>")/<arm>/ckpt/*.ckpt` — one checkpoint per arm,
#   named with `val/loss`, in a per-arm directory.
# * `.../lab6_predictions_<arm>_<hash>.npz` — raw `y_true`/`y_pred`, so every metric is
#   recomputable.
# * `results/lab6_ablation_arms.png` — arms vs baselines, with the 1/K line drawn.
# * `results/results.json` — one `lab6` record whose `config.arms` lists all five arms
#   with their declared band lists and unit flags, whose `split_manifest_hash` matches
#   the split in use, and whose `extra.arm_metrics` carries every arm.
# * **Your three prediction blocks** filled in, each with `predicted vs actual` and, for
#   every disagreement, one sentence naming the mechanism. This is the graded part.
# * The gate board with every `FAIL` explained — including `beats_baselines` if your
#   foundation model lost to the linear probe.
# * One paragraph: state your majority-class baseline number and whether your fine-tuned
#   Prithvi beats it. The 2025/26 submission could not have answered this.
# * For the graded sweep: the `sbatch` job IDs from `slurm/submit_sweep.sbatch` with
#   `--export=ALL,GRID="unnormalised normalised swapped invalid_bands random_init"`,
#   three seeds, `--partition=dc-gpu --gres=gpu:1`, and `MAX_EPOCHS=10`.

# %% [markdown]
# ## Where each 2025/26 defect went
#
# | 2025/26 | 2026/27 |
# |---|---|
# | no normalisation; `[0,1]` into a ~1000-DN backbone; `test/Accuracy` = 1/10 | Part 4 computes the contrast ratio (= 1/QUANTIFICATION_VALUE) on your own patches; Part 8 runs the `unnormalised` arm as a first-class ablation; Exercise 1 grades your prediction of it |
# | `backbone_bands = ["BLUE","RED","GREEN","NIR_NARROW"]` vs B02,B03,B04,B08 data | Part 5 models `select_patch_embed_weights` and shows `src=[0,2,1,3]`; Part 8 runs the `swapped` arm; the log-capture cell counts real "Loaded weights" lines |
# | commented-out `["B02","B03","B04","B08"]` | `invalid` arm + `xavier_filters=4/4` in Part 5, and the patch-embed std check in Part 8 |
# | "6 optical + 2 DEM" claims elsewhere in the repo | Part 1 and Part 4 state the verified six bands `BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2`, no DEM, from `prithvi_vit.py` |
# | `monitor="val/Accuracy"` over classes with n=1 | `monitor="val/loss"`; Part 2 prints the thin-support classes and `merge_rare_classes` is named as the fix |
# | `trainer.test(model=task)` with no `ckpt_path` | `ckpt_path="best"` in `run_arm`; `predict_arm` refuses to run without a checkpoint on disk |
# | `filename="{val_acc:.2f}"` for a key that is `val/Accuracy` | `filename="{epoch}-{val/loss:.4f}"`, a key that is logged |
# | shared `training_logs` dir, `save_weights_only=True` | `paths.run_dir(...)/<arm>/ckpt` per arm, `save_weights_only=False` |
# | no baseline anywhere | Part 3 runs `baselines.run_all` before any GPU time is spent; `gate_beats_baselines` is on the board |
# | per-patch stratified split, tile identity dropped | `splits.group_block_split` on scene + 10x10 block; `manifest_hash` in `results.json`; Part 9 runs the leaky split as the negative control |
# | split save-loop inside a triple-quoted string | splits come from `eo_course.splits` and are written with `MANIFEST.write`; nothing is inside a string |
# | five classes deleted by `< 10`, incl. Peat bogs and Water courses | `LabelMap.from_train` + `metrics.merge_rare_classes`; Part 2 prints what would have been dropped |
# | `logger=False`, `devices` unset, 4-GPU allocation running 1 process in Jupyter | `logger=True`, `devices=1` stated explicitly, Part 6 names `slurm/train_jureca.sbatch` and `slurm/submit_sweep.sbatch` and the `dc-gpu` / A100-40GB target |
# | cells asserting a pasted dict of floats | removed; gates and predicted-vs-actual replace them |
# | zero exercises, zero gates | three graded predictions, one negative control, one gate board, one recorded run |
