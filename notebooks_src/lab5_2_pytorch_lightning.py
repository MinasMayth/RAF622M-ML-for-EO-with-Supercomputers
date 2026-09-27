# %% [markdown]
# # Lab 5.2 — Lightning mechanics, and never reporting a validation number as a test number
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# Lab 5.1 made your model earn its number: baselines first, a hashed scene-held-out split,
# three seeds, a gate board. This lab keeps that discipline and adds the half of Lightning
# that decides whether the number you write down is the number you computed. It produces:
#
# * a **checkpoint you can reload** — named `<run_id>.pt`, in `paths.artifacts_dir()`, read
#   back with `weights_only=True` so a result becomes checkable rather than remembered;
# * a **selection rule** that monitors an imbalance-aware metric with adequate support, and
#   an early-stopping rule that is not pointed backwards;
# * **validation and test metrics recorded separately** in one `results.json` record, with
#   `ckpt_path="best"` doing the weight selection instead of your memory;
# * a **support column** you have to look at before you are allowed to average over it.
#
# It assumes Lab 4.2 wrote the patch archives and Lab 5.1 fixed the class set and the split
# convention. It re-derives both here so the notebook runs standalone, and it calls the same
# `eo_course` functions rather than re-implementing them.
#
# **You must run every cell yourself.** The 2025/26 version of this exact notebook was
# committed with `trainer.test()` printing two scalars, no per-class table anywhere on the
# test split, and two cells in an error state — and its summary cell reported
# `0.9231 accuracy / 0.8338 F1 / 0.8376 kappa`. Those were **validation** numbers on 455
# samples where 5 of 10 classes had support ≤ 9 and two classes had **n = 1**. Lab 7 then
# hard-coded all three as ground truth. The back half of the course was validating a number
# of which two misclassified samples moved it by ±0.05.

# %% [markdown]
# ## Why this notebook is separate from 5.1
#
# Lab 5.1 is about *what* a number means: chance floors, split leakage, seed spread. Lab 5.2
# is about the *machinery* that decides which weights produced it and which split scored
# them. In 2025/26 that machinery is where the fatal error lived, and it stayed invisible
# precisely because Lightning made it quiet:
#
# * `trainer.test(model, datamodule)` with no `ckpt_path` evaluates the **in-memory
#   end-of-training weights**. The `ModelCheckpoint` that supposedly selected the best epoch
#   — and in the old notebook was commented out anyway — never touched the evaluation.
# * `EarlyStopping(monitor="val_acc", mode="min")` is syntactically valid, prints nothing
#   alarming, and stops at the **worst** accuracy.
# * A cosine scheduler whose `T_max` is in epochs, returned without `interval="epoch"`, is
#   stepped per **batch**. At 625 steps/epoch against a `T_max` of 100 the learning rate
#   completed ~625 full cosine cycles instead of annealing once.
# * A numpy array anywhere in `hyper_parameters` makes `torch.load(weights_only=True)` —
#   the default since torch 2.6 — raise `UnpicklingError` at the moment you try to check
#   your own result.
#
# Each of those is a mechanism, not an opinion. You will run all four in this notebook and
# read the number each one produces.

# %% [markdown]
# ## The 2025/26 defects this notebook closes
#
# Cell numbers are the old 57-cell notebook's, so you can check them against the audit.
#
# | old defect | consequence | fixed in |
# |---|---|---|
# | cell 53 computed the headline on `val_dataloader()`; cell 51's `trainer.test()` result was discarded | the only per-class analysis in the course ran on the tuning set; 0.9231/0.8338/0.8376 were validation numbers | Part 8, Part 9 |
# | those numbers were on 455 samples with supports `9, 1, 245, 99, 6, 1, 5, 52, 30, 7` | five of ten recalls came from 1–9 samples; two were literally 0 or 1 | Part 3 |
# | no `ModelCheckpoint`; `trainer.test(model, datamodule)` | the last epoch's weights were evaluated while the text implied "best" | Part 7, Part 8 |
# | `EarlyStopping(monitor="val_acc", mode="min")` (cell 51) | stops training at the worst accuracy — in practice epoch 1 | Part 5 |
# | cosine scheduler returned with no interval (cells 40, 43) | 259 cosine cycles in a 20-epoch run | Part 4 |
# | `batch_size=256` against a few thousand patches | 3–23 steps/epoch; the head collapses onto the majority class | Part 6 |
# | `save_hyperparameters` absent from the VGG cell (cell 40) | `load_from_checkpoint` — what lab 7 told students to do — could not work | Part 2 |
# | a numpy array in hparams | `torch.load(weights_only=True)` raises `UnpicklingError`; the result cannot be re-opened | Part 2 |
# | a `*_data.npz` glob plus a positional npz read (cell 9) | every patch loaded twice; the positional read can swap patches and labels | Part 1 |
# | `get_split` allowed `num_train < 0`; `setup()` re-split on every call | the same sample landed in val **and** test | Part 1 |
# | class-drop threshold `< 10` computed over all data (cell 13) | the metric's denominator was test-informed | Part 1 |
# | label set derived from the evaluation slice | a class the model never predicts silently left the macro-average | Part 11 |
# | zero `seed_everything`, one split, two conflicting splits in one file | run-to-run noise exceeded every effect the lab compared | Part 9 |
# | zero `to_csv` / `json.dump` / `torch.save`; no `default_root_dir` | nothing persisted, so nothing was gradeable | Part 7, Part 12 |
# | ran on an interactive GPU node; no `.sbatch` existed in the repo | a paid 4-GPU allocation ran one process on one GPU, with no curves to inspect | Part 0 |

# %% [markdown]
# ## Part 0 — Setup
#
# The order is not cosmetic. `MPLCONFIGDIR` must exist **before** matplotlib is imported;
# the 2025/26 notebooks set it afterwards, which does nothing, and shipped
# `Matplotlib created a temporary cache directory` in committed output. One
# `np.random.default_rng(seed)` is created here and passed down; there is no legacy global
# RNG call anywhere in this file, because that is what made the old notebook's `seed=42`
# recover nothing.

# %%
from eo_course import paths

print(paths.describe())

# %%
import math
import os

import numpy as np

os.environ["MPLCONFIGDIR"] = paths.matplotlib_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402  (must follow MPLCONFIGDIR)

from eo_course import baselines as bl  # noqa: E402
from eo_course import gates, labels as lab_mod, metrics  # noqa: E402
from eo_course import patches as pat  # noqa: E402
from eo_course import paths, radiometry, results, splits  # noqa: E402

SEED = 0
rng = np.random.default_rng(SEED)
# Windows starts dataloader workers by spawn; Linux (JURECA) forks. Zero workers costs
# nothing at this patch size, so the notebook stays portable across both.
NUM_WORKERS = 0 if os.name == "nt" else 2
LAB = "lab5.2"
print(f"  one rng seeded {SEED}; every stochastic helper below takes seed=SEED explicitly "
      f"(split, bootstrap, duplicate-sampler) — no legacy np.random.* call exists here")
print(f"  dataloader workers: {NUM_WORKERS}   lab tag: {LAB}")

# %% [markdown]
# ### Where the real runs actually go
#
# Nothing in this notebook is a graded training run. It exists so you can watch the
# mechanisms fire. The graded work is one job per (arm, seed):
#
# ```bash
# sbatch slurm/train_jureca.sbatch                              # one arm, one seed
# sbatch --export=ALL,ARM_TAG=weights_sqrt slurm/train_jureca.sbatch
# sbatch slurm/submit_sweep.sbatch                              # the whole arm x seed grid
# ```
#
# Both scripts call `scripts/train_cnn.py`, which is the same pipeline you are about to run
# cell by cell — one implementation, so the notebook cannot drift from the graded path.
# JURECA-DC partitions are `dc-cpu` and `dc-gpu`, **lower case with hyphens**; the 2025/26
# lab 1 taught upper-case underscored variants that are not partition names at all, and its
# example had no `--gres` line, so a student who copied it asked for a GPU and got a job
# that could not see one. Confirm live names with `sinfo`.
#
# What breaks on a login node: heavy I/O is charged to a shared machine and will get you
# disconnected; there is no GPU, so a "quick" 50-epoch run is hours of someone else's CPU;
# and `module` is a shell function, which is why the old notebooks' `!source ...` lines
# failed under `/bin/sh`. Jupyter-JSC is its own trap: it caps you at **one** GPU even when
# it allocated four, printing `Trainer will use only 1 of 4 GPUs because it is running
# inside an interactive / notebook environment`. That is exactly what the 2025/26 Lab 6
# output showed.

# %%
print(f"  on JURECA     : {paths.on_jureca()}")
print(f"  login node    : {paths.on_login_node()}")

import torch  # noqa: E402

ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
print(f"  accelerator for this notebook: {ACCELERATOR}")
if paths.on_login_node() and ACCELERATOR == "cpu":
    print(
        "  WARNING: login node, no GPU. Verify the pipeline here, then submit\n"
        "  slurm/train_jureca.sbatch. A long run on a login node is slow for you and\n"
        "  rude to everyone else."
    )
# %% [markdown]
# ## Part 1 — The hand-off: archives, split, class set, transform
#
# Lab 5.1 argued each of these at length. They are re-derived here in five cells because a
# notebook that trains must not inherit a split from a variable someone left in a kernel.
# Three rules, each of which was broken in 2025/26:
#
# 1. **Load by key, from the per-scene pattern only.** The 2025/26 glob matched the per-scene
#    files *and* the combined file, which was a permutation of the same patches, so every
#    sample entered the array twice and the split put duplicates on both sides. A positional
#    npz read then returns arrays in insertion order, so one extra metadata key silently
#    turns patches into labels.
# 2. **The split comes from a manifest, not a permutation.** Whole scenes held out, spatial
#    blocks inside the rest, and a hash so "the split I reported == the split I used" is
#    checkable. The old `get_split` allowed `num_train < 0`, and its assert
#    `num_train + num_val + num_test == len(idx)` still passed — because `-1 + 1 + 1 == 1` —
#    while two different slices both selected the same single index into val and test.
# 3. **The class set and the transform are decided on TRAIN only.** The old class-drop
#    threshold `< 10` was computed over train+val+test, which silently changed the
#    denominator of every reported metric.

# %%
TRAIN_DIR = paths.require_existing(
    paths.training_data_dir(),
    "Lab 4.2 patch archive directory (paths.training_data_dir())",
)
ARCHIVES = sorted(TRAIN_DIR.glob(pat.SCENE_GLOB))
if not ARCHIVES:
    raise FileNotFoundError(
        f"no {pat.SCENE_GLOB} archives in {TRAIN_DIR}. Run Lab 4.2 "
        "(extract_patches + save_patches); it writes one archive per scene plus a sidecar JSON."
    )
PS0 = pat.load_all_scenes(TRAIN_DIR)
print(f"  {len(ARCHIVES)} archives matched by {pat.SCENE_GLOB!r}; a combined file cannot match it")
print(f"  {len(PS0)} patches, {len(np.unique(PS0.scene))} scenes, bands {list(PS0.bands)}")
gates.gate_no_duplicate_patches(PS0.patches, groups=PS0.scene, seed=SEED)
print("  gate_no_duplicate_patches passed")

# %%
SCENES = sorted(map(str, np.unique(PS0.scene)))
TEST_SCENES = SCENES[-1:]
man_pre = splits.group_block_split(PS0.scene, PS0.row, PS0.col, block=10, val_ratio=0.15,
                                   test_scenes=TEST_SCENES, seed=SEED)
gates.gate_split_is_grouped(man_pre)
gates.gate_split_disjoint(man_pre)
print(f"  provisional split {man_pre.manifest_hash}: "
      f"{len(man_pre.train_idx)}/{len(man_pre.val_idx)}/{len(man_pre.test_idx)}")
print(f"  test scenes {TEST_SCENES}   val scenes {man_pre.summary()['val_scenes']}")

# %%
MERGE_MIN = gates.MIN_CLASS_SUPPORT
RARE_OTHER = 9001  # not a CORINE code, so it can never collide with one

CODES_ALL = tuple(int(c) for c in PS0.class_codes)
counts_train_pre = lab_mod.class_counts(PS0.labels[man_pre.train_idx], CODES_ALL)
DROP_CODES = [int(c) for c, n in zip(CODES_ALL, counts_train_pre) if n == 0]
MERGED_CODES = [int(c) for c, n in zip(CODES_ALL, counts_train_pre) if 0 < n < MERGE_MIN]
KEEP_CODES = [int(c) for c, n in zip(CODES_ALL, counts_train_pre) if n >= MERGE_MIN]
print(f"  {'code':>5} {'class':<32} {'train':>7}  decision")
for i, c in enumerate(CODES_ALL):
    dec = ("DROP (unlearnable)" if c in DROP_CODES else
           f"merge -> {RARE_OTHER}" if c in MERGED_CODES else "keep")
    print(f"  {c:>5} {lab_mod.class_name(c)[:32]:<32} {counts_train_pre[i]:>7}  {dec}")
if not KEEP_CODES:
    raise RuntimeError(
        f"no class reaches {MERGE_MIN} training samples. Your dataset is too small to "
        "supervise at all; go back to Lab 4.2 and extract more scenes rather than lowering "
        "the threshold until something passes."
    )

# %%
drop_mask = np.isin(PS0.labels, DROP_CODES)
keep_idx = np.flatnonzero(~drop_mask)
PS = PS0.subset(keep_idx)
LABELS_RAW = PS0.labels[keep_idx]              # unmerged: the negative control needs these
LABEL_CODES = LABELS_RAW.copy()
for c in MERGED_CODES:
    LABEL_CODES[LABELS_RAW == c] = RARE_OTHER

manifest = splits.group_block_split(PS.scene, PS.row, PS.col, block=10, val_ratio=0.15,
                                    test_scenes=TEST_SCENES, seed=SEED)
sm = manifest.summary()
train_idx, val_idx, test_idx = manifest.train_idx, manifest.val_idx, manifest.test_idx
paths.ensure(paths.splits_dir())
MANIFEST_PATH = manifest.write(paths.splits_dir() / f"split_{manifest.manifest_hash}.json")
gates.gate_split_is_grouped(manifest)
gates.gate_split_disjoint(manifest)
CODES_USED = KEEP_CODES + ([RARE_OTHER] if MERGED_CODES else [])
print(f"  manifest of record {manifest.manifest_hash}  ({MANIFEST_PATH.name})")
print(f"  n train/val/test {sm['n_train']} / {sm['n_val']} / {sm['n_test']} of {sm['n_total']}")
print(f"  class set ({len(CODES_USED)}): {CODES_USED}")

# %%
norm = radiometry.Norm.fit(PS.patches[train_idx], mode="minmax", channel_axis=-1)
x_all = np.ascontiguousarray(np.transpose(norm.apply(PS.patches), (0, 3, 1, 2)))
LM = lab_mod.LabelMap(tuple(CODES_USED))
y_all = LM.encode(LABEL_CODES)
LABELS = np.arange(LM.n_classes)
# The merged bucket is not a CORINE code, so labels.class_name would print it as
# "UNKNOWN CORINE 9001" on the gate board. Name it: a gate message a student cannot
# read is a gate message that gets ignored.
CLASS_NAMES = list(LM.names)
if MERGED_CODES:
    CLASS_NAMES[LM.index_of(RARE_OTHER)] = f"rare-other (merged {MERGED_CODES})"
IN_CHANNELS = x_all.shape[1]
gates.gate_input_units(x_all, "normalised patches")
print(f"  {norm}  fitted on {norm.n_samples_seen} of {len(PS)} patches (train only)")
print(f"  x_all {x_all.shape} range [{x_all.min():.4f}, {x_all.max():.4f}]")
print(f"  LabelMap from TRAIN: {LM}")
print("  gate_input_units passed with the default expect='reflectance'")

# %% [markdown]
# ### `gate_input_units` takes an `expect`, and you should name it
#
# The default is the strictest option because the course convention is reflectance in
# `[0, 1]`. But "is it in `[0, 1]`" is not the real question — the real question is *is this a
# scale a sensor produced, or an artifact of arithmetic*. Lab 5.1 divided an already-stretched
# array by 10 000, the network saw values of order 1e-5, BatchNorm absorbed it, and the bug
# survived a full academic year because nothing raised.
#
# | `expect` | what it accepts | who wants it |
# |---|---|---|
# | `"reflectance"` (default) | per-band values in `[0, 1]` with a real spread | `Norm(mode="minmax")`, the course default |
# | `"standardized"` | mean near 0, std near 1; ranges in `[-3, 5]` are fine | `Norm(mode="zscore")`, Prithvi |
# | `"dn"` | order 1e3–1e4 | raw L2A digital numbers |
# | `"auto"` | any sane scale; rejects only pathological ones | quick checks |
#
# Related trap, because it bites the same way: `Norm`'s `clip` default is **per mode**.
# `minmax` and `percentile` default to `(0, 1)`; `zscore` defaults to `(-3, 3)`. A
# mode-independent `(0, 1)` default would clip every negative z-score to 0.0 — on a symmetric
# distribution that is half your data, silently replaced by a constant.

# %%
x_z = radiometry.Norm.fit(PS.patches[train_idx], mode="zscore", channel_axis=-1).apply(PS.patches)
gates.gate_input_units(x_z, "z-scored patches", expect="standardized")
print(f"  z-scored array: mean {x_z.mean():.3f} std {x_z.std():.3f} "
      f"range [{x_z.min():.2f}, {x_z.max():.2f}]  -> expect='standardized' passes")
print(f"  Norm(mode='zscore').clip defaults to {radiometry.Norm(mode='zscore').clip}, not (0, 1)")
try:
    gates.gate_input_units(x_all, "reflectance patches", expect="dn")
    print("  UNREACHABLE: reflectance passed a DN check")
except gates.GateFailure as exc:
    print(f"  expect='dn' on reflectance raises:\n    {exc}")
# %% [markdown]
# ## Part 2 — Why Lightning splits the code the way it does
#
# `LightningDataModule` owns *which samples, in which order, with which transform*.
# `LightningModule` owns *the weights, the loss, the optimiser, and what gets logged*. The
# `Trainer` owns the loop. The payoff is not tidiness: it is that `ModelCheckpoint`,
# `EarlyStopping`, logging and `ckpt_path` all read from one place, so the thing that selected
# your weights and the thing that scored them cannot silently disagree — which is exactly what
# happened in 2025/26, where the callbacks were commented out and the evaluation used whatever
# was in memory.
#
# `training.CorineDataModule` takes a **`SplitManifest`, not an index list**. That is the
# design decision worth two minutes. With a bare `(train_idx, val_idx, test_idx)` tuple the
# module cannot tell you what produced them, whether they are disjoint, whether a scene
# straddles splits, or whether the split you trained on is the split you reported. The old
# DataModule recomputed its split on **every** `setup()` call and relied on a hard-coded
# `seed=42` default that its config never passed — so a student who changed the seed got a
# "test" subset drawn from a different permutation of the same pool: direct train/test
# overlap, printed as a generalisation score. Passing the manifest in means the hash travels
# with the data, into the hyperparameters, and into `results.json`.

# %%
from eo_course.training import CorineDataModule, CorineModule  # noqa: E402

dm = CorineDataModule(x_all, y_all, manifest, batch_size=32, imbalance="sqrt_inverse_freq",
                      num_workers=NUM_WORKERS, seed=SEED)
dm.setup("fit")
print(f"  datamodule hparams keys: {sorted(dm.hparams)}")
print(f"  split_manifest_hash in hparams: {dm.hparams['split_manifest_hash']}")
print(f"  matches the manifest in use: "
      f"{dm.hparams['split_manifest_hash'] == manifest.manifest_hash}")
print(f"  train_counts (TRAIN only): {dm.train_counts.tolist()}")
print(f"  test transform: {dm._datasets['test'].transform}   (must be None)")

# %%
print("  What the DataModule refuses to do, and which 2025/26 defect each refusal closes:")
print("    - invent its own split            -> setup() re-split on every call, seed=42 default")
print("    - augment val or test             -> test_dataset read the val transform key")
print("    - build sampler weights off-train -> 7,358 weights over an 8,278-sample dataset,")
print("                                        so 920 patches were never drawable")
print("    - evaluate at batch_size=1        -> 455 sequential launches, and per-batch acc")
print("                                        silently equals per-sample acc")
print("  save_hyperparameters(ignore=['x', 'y', 'manifest']) keeps the arrays and the")
print("  manifest out of the hparams dict; the hash is recorded separately, as a scalar.")

# %% [markdown]
# ### `save_hyperparameters` — the two traps that actually fire
#
# **Trap 1: there is no `hyperparameters=` kwarg.** `save_hyperparameters()` inspects the
# caller's signature. It accepts `ignore=[...]` and a couple of positional frame arguments.
# Passing a dict under the name of the resulting attribute — which is what the name tempts you
# to write — is a `TypeError` at construction time, in every cell that touches the class.
#
# **Trap 2: a numpy array in `hyper_parameters` makes your checkpoint unreadable.** Since
# torch 2.6 `torch.load` defaults to `weights_only=True`, which refuses any global not on its
# allow-list. `numpy._core.multiarray._reconstruct` is not on it. So a checkpoint whose
# `hyper_parameters` contains an array raises `UnpicklingError` — not when you save it, but
# weeks later when you or a grader try to open it. The fix is structural, not a flag: keep
# hparams scalar, and put arrays in `register_buffer`, which travels in `state_dict` and is
# restored by `load_from_checkpoint`.
#
# `CorineModule` does exactly that: `save_hyperparameters(ignore=["train_counts"])` plus
# `self.register_buffer("train_counts", ...)`.

# %%
probe = CorineModule(n_classes=LM.n_classes, in_channels=IN_CHANNELS,
                     imbalance="sqrt_inverse_freq", train_counts=dm.train_counts,
                     max_epochs=1)
print(f"  hparams: {dict(probe.hparams)}")
print(f"  hparam value types: {sorted({type(v).__name__ for v in dict(probe.hparams).values()})}")
print(f"  train_counts is a registered buffer: "
      f"{'train_counts' in dict(probe.named_buffers())}")
print(f"  buffer value: {probe.train_counts.tolist()}")

# %%
# Trap 2, reproduced. Same idea, one word changed: the array is NOT ignored.
import lightning as pl  # noqa: E402


class BadModule(pl.LightningModule):
    def __init__(self, n_classes=2, counts=None):
        super().__init__()
        self.save_hyperparameters()          # counts stays in hyper_parameters
        self.layer = torch.nn.Linear(2, n_classes)


_bad = BadModule(counts=dm.train_counts)
_art = paths.ensure(paths.artifacts_dir())[0]
p_bad = _art / "_probe_numpy_hparam.pt"
p_ok = _art / "_probe_clean_hparam.pt"
torch.save({"state_dict": _bad.state_dict(), "hyper_parameters": dict(_bad.hparams)}, p_bad)
torch.save({"state_dict": probe.state_dict(), "hyper_parameters": dict(probe.hparams)}, p_ok)
for label, path in (("numpy array in hparams", p_bad), ("scalars only", p_ok)):
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        print(f"  weights_only=True, {label}: loads")
    except Exception as exc:
        line = [ln.strip() for ln in str(exc).splitlines() if "Unsupported global" in ln]
        print(f"  weights_only=True, {label}: {type(exc).__name__} {line[:1]}")

# %%
try:
    class KwargModule(pl.LightningModule):
        def __init__(self, lr=1e-3):
            super().__init__()
            self.save_hyperparameters(hyperparameters={"lr": lr})

    KwargModule()
    print("  UNREACHABLE: a hyperparameters= kwarg was accepted")
except TypeError as exc:
    print(f"  save_hyperparameters(hyperparameters={{...}}) -> TypeError: {exc}")
print("  The valid kwarg is ignore=[...]. Everything else is a frame argument.")

# %% [markdown]
# ## Part 3 — Support is a number you have to look at
#
# Here is the arithmetic behind this lab's headline defect. The 2025/26 headline was
# `Balanced accuracy 0.8338`, the unweighted mean of ten per-class recalls whose supports were
# `9, 1, 245, 99, 6, 1, 5, 52, 30, 7`. Five of those ten recalls came from 1–9 samples. Class 1
# had **n = 1** and scored recall 1.000; class 5 had **n = 1** and scored 0.000. Two
# misclassified samples moved the headline by about ±0.05 — larger than every difference the
# lab asked students to compare, and larger than the gap Lab 7 later analysed at length.
#
# `gates.MIN_CLASS_SUPPORT = 25` exists because a recall on n = 25 has a standard error near
# 0.10, and on n = 1 it is literally 0 or 1 printed to three decimals as if it were a
# measurement.

# %% [markdown]
# ### Exercise 1 — predict the support before you print it
#
# Score the class set **as Lab 4.2 delivered it** — before Part 1's merge — on the
# **validation** split. This is the exact set of rows the 2025/26 headline averaged over.
#
# > **P1.** Write down, before running:
# >
# > 1. The **minimum** per-class support on the val split, as an integer.
# > 2. How many classes fall below `gates.MIN_CLASS_SUPPORT`.
# > 3. Whether `gates.gate_class_support` passes or raises on that table.
# >
# > Use the per-split counts Part 1 already computed and the split ratios; do not guess blind.
# > Question 3 is the one that matters: a gate that raises is a claim you are not allowed to
# > make.

# %%
P1 = {
    # TODO(you): fill in BEFORE running the next two cells. Leaving these None is scored as
    # not attempted.
    "min_val_support": None,
    "n_classes_below_min": None,
    "gate_class_support": None,  # "pass" or "raises"
    "why": "TODO(you): in one sentence, why is a macro-average over these rows not a "
           "measurement of the thing the headline claimed to measure?",
}

# %%
val_counts = lab_mod.class_counts(LABELS_RAW[val_idx], CODES_ALL)
per_split = splits.per_split_class_counts(LABELS_RAW, manifest, CODES_ALL)
print(f"  {'code':>5} {'class':<32} {'train':>7} {'val':>6} {'test':>6}")
for i, c in enumerate(CODES_ALL):
    print(f"  {c:>5} {lab_mod.class_name(c)[:32]:<32} "
          f"{per_split['train'][i]:>7} {val_counts[i]:>6} {per_split['test'][i]:>6}")
print(f"  labels.class_counts on the VAL split, ordered by {CODES_ALL}")
print(f"  val n = {int(val_counts.sum())}   imbalance on val = "
      f"{lab_mod.imbalance_ratio(val_counts):.1f}x")
print(f"\n  gates.MIN_CLASS_SUPPORT = {gates.MIN_CLASS_SUPPORT}")
print(f"  classes with 0 < val support < {gates.MIN_CLASS_SUPPORT}: "
      f"{int(((val_counts > 0) & (val_counts < gates.MIN_CLASS_SUPPORT)).sum())}")
print(f"  P1 answers were         : {P1}")

# %%
# The 2025/26 headline procedure, on the val split, with no training required: a constant
# predictor scored over the unmerged class set. The number is not the point; the support
# column is.
LM_ALL = lab_mod.LabelMap(tuple(CODES_ALL))
y_val_all = LM_ALL.encode(LABELS_RAW[val_idx])
pred_const = bl.majority_class(LABELS_RAW[train_idx], CODES_ALL, len(val_idx))
val_unmerged = metrics.evaluate(y_val_all, pred_const, np.arange(LM_ALL.n_classes),
                                codes=LM_ALL.codes, names=LM_ALL.names,
                                groups=PS.scene[val_idx], n_boot=200, seed=SEED)
print(val_unmerged.table(min_support=gates.MIN_CLASS_SUPPORT))
try:
    gates.gate_class_support(val_unmerged, min_support=gates.MIN_CLASS_SUPPORT)
    print("  gate_class_support: PASS")
except gates.GateFailure as exc:
    print(f"  gate_class_support raises:\n    {exc}")

# %% [markdown]
# > Record predicted vs actual for P1 and explain any disagreement. Then answer in one
# > sentence: the gate message lists the offending supports. Why is *merging* those classes the
# > honest response rather than deleting the rows and averaging over what is left — and what
# > does the merged number no longer mean?
# %% [markdown]
# ## Part 4 — The scheduler: `T_max` is in epochs, the default interval is per batch
#
# This defect ran for a year in three separate model classes and it is entirely silent.
# Lightning's scheduler config defaults to `interval="step"`. So the old code:
#
# ```python
# # lab5_1 cell 11, lab5_2 cells 40 and 43, verbatim
# return {"optimizer": opt, "lr_scheduler": CosineAnnealingLR(opt, T_max=self.max_epochs)}
# ```
#
# `T_max` is the half-period **in whatever unit the scheduler is stepped in**. Stepped per
# batch, with 625 steps/epoch and `T_max=100`, the LR ran 62 500 steps against a 100-step
# half-period: hundreds of full cosine cycles, the learning rate sawtoothing between 3e-4 and
# 0 dozens of times per epoch. Nothing warns. The loss curve looks plausible.
#
# `CorineModule.configure_optimizers` returns `interval="epoch", frequency=1` explicitly and
# says so in a comment.

# %%
STEPS_PER_EPOCH = math.ceil(len(train_idx) / 32)
print(f"  n_train = {len(train_idx)}, batch 32 -> {STEPS_PER_EPOCH} steps/epoch")
print(f"  {'EPOCHS':>7} {'total steps':>12} {'T_max':>6} {'cycles if stepped per batch':>28}")
for ep in (5, 10, 15, 20):
    print(f"  {ep:>7} {ep * STEPS_PER_EPOCH:>12} {ep:>6} "
          f"{ep * STEPS_PER_EPOCH / ep:>28.1f}")
print("  The old notebook's own numbers were 159,996 train patches at batch 256 = 625")
print("  steps/epoch, T_max=100. Same formula, 625 cycles.")

# %% [markdown]
# ### Exercise 2 — predict the number of cycles
#
# You will now run the buggy configuration on **your** split: `EPOCHS = 15`, `T_max = EPOCHS`,
# stepped once per training step, exactly as Lightning's default interval does it.
#
# > **P2.** Write down, before running:
# >
# > 1. How many complete cosine cycles the LR trace contains.
# > 2. The learning rate at the very end of training, as a multiple of the initial LR (`0`,
# >    `~1×`, or something in between).
# > 3. Then: if you fix only the interval and leave `T_max` in epochs, what is the final LR?

# %%
P2 = {
    # TODO(you): fill in BEFORE running the next two cells.
    "n_cycles": None,
    "final_lr_over_initial": None,   # 0, 1, or a number in between
    "final_lr_after_fix": None,
    "why": "TODO(you): which of the two schedules actually anneals? Name the config key.",
}

# %%
EPOCHS = 15
LR0 = 3e-4
opt_b = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR0)
# The old bug, reproduced: T_max in epochs, stepped per batch (Lightning's default interval).
sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=EPOCHS)
lrs = []
for _ in range(EPOCHS * STEPS_PER_EPOCH):
    lrs.append(opt_b.param_groups[0]["lr"])
    opt_b.step()
    sched_b.step()
lrs = np.asarray(lrs)
restarts = int(((lrs[1:] > 0.99 * LR0) & (lrs[:-1] <= 0.99 * LR0)).sum())
print(f"  per-batch interval: {len(lrs)} steps, T_max={EPOCHS}")
print(f"    cycles observed   : {restarts + 1}")
print(f"    min LR            : {lrs.min():.3e}   final LR: {lrs[-1]:.3e}")
print(f"    P2 answers were   : {P2}")

# %%
opt_e = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=LR0)
sched_e = torch.optim.lr_scheduler.CosineAnnealingLR(opt_e, T_max=EPOCHS)
lr_ep = []
for _ in range(EPOCHS):
    for _ in range(STEPS_PER_EPOCH):
        opt_e.step()
    lr_ep.append(opt_e.param_groups[0]["lr"])
    sched_e.step()  # once per epoch: interval="epoch", frequency=1
print(f"  per-epoch interval: {EPOCHS} steps, T_max={EPOCHS}")
print(f"    final LR          : {opt_e.param_groups[0]['lr']:.3e}  (annealed once, to ~0)")

# %%
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(lrs, lw=0.8, label="interval='step' (the 2025/26 bug)")
ax.plot(np.linspace(0, len(lrs) - 1, EPOCHS), lr_ep, "o-", label="interval='epoch' (the fix)")
ax.set_xlabel("training step"); ax.set_ylabel("learning rate")
ax.set_title(f"same T_max={EPOCHS}, two intervals")
ax.legend(fontsize=8)
fig.tight_layout()
FIG_SCHED = paths.ensure(paths.results_dir())[0] / "lab5_2_scheduler_interval.png"
fig.savefig(FIG_SCHED, dpi=110)
plt.show()
print(f"  saved {FIG_SCHED}")

# %% [markdown]
# ## Part 5 — `EarlyStopping` direction, and why the two callbacks watch different things
#
# `mode="min"` means "lower is better". On an accuracy that is a backwards monitor: the
# callback waits for improvement in the *wrong* direction, sees one at epoch 1, and stops. The
# 2025/26 notebook shipped `EarlyStopping(monitor="val_acc", mode="min")` commented out; a
# student who uncommented it got a one-epoch model and a green notebook.
#
# There is a second, quieter failure: monitoring a metric that was never logged.
# `CorineModule` logs `val_loss`, `val_balanced_acc` and `val_macro_f1`. It does **not** log
# `val_acc`, because per-batch accuracy averaged over batches is not per-sample accuracy — a
# 124-sample final batch counted the same as a 256-sample one. Monitoring `val_acc` raises
# `RuntimeError: Early stopping conditioned on metric ... which is not available`, which is the
# *good* outcome. The old notebooks logged `val_acc` and used it, so nothing raised and the
# number was simply wrong.

# %%
class Tiny(pl.LightningModule):
    """Toy module that logs a constant 'accuracy', to watch a callback fire."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, b, i):
        loss = torch.nn.functional.cross_entropy(self(b[0]), b[1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, b, i):
        self.log("val_balanced_acc", 0.9, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())


# %%
class TwoFeatureDS(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        return torch.randn(2), torch.tensor(i % 2)


from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402  (loggers, not callbacks)

_loader = torch.utils.data.DataLoader(TwoFeatureDS(), batch_size=8)
_es_backwards = EarlyStopping(monitor="val_balanced_acc", mode="min", patience=1)
_tr_bwd = pl.Trainer(max_epochs=12, accelerator="cpu", devices=1, logger=False,
                     enable_progress_bar=False, limit_train_batches=2, limit_val_batches=1,
                     num_sanity_val_steps=0, callbacks=[_es_backwards])
_tr_bwd.fit(Tiny(), train_dataloaders=_loader, val_dataloaders=_loader)
print(f"  EarlyStopping(mode='min') on an accuracy: stopped after epoch "
      f"{_tr_bwd.current_epoch + 1} of 12")

# %%
_es_missing = EarlyStopping(monitor="val_acc", mode="max", patience=2)
_tr_miss = pl.Trainer(max_epochs=3, accelerator="cpu", devices=1, logger=False,
                      enable_progress_bar=False, limit_train_batches=2, limit_val_batches=1,
                      num_sanity_val_steps=0, callbacks=[_es_missing])
try:
    _tr_miss.fit(Tiny(), train_dataloaders=_loader, val_dataloaders=_loader)
    print("  UNREACHABLE: monitored a metric that was never logged")
except RuntimeError as exc:
    print(f"  monitor='val_acc' on CorineModule's logging:\n    {type(exc).__name__}: "
          f"{str(exc)[:170]}")

# %% [markdown]
# ### Why `scripts/train_cnn.py` decouples the two callbacks
#
# The graded path does this, and the choice is deliberate:
#
# ```python
# best = ModelCheckpoint(monitor="val_macro_f1", mode="max", save_top_k=1, save_last=True)
# stop = EarlyStopping(monitor="val_loss", mode="min", patience=8)
# ```
#
# **Selection** and **termination** answer different questions and want different noise
# profiles.
#
# * `ModelCheckpoint` picks one epoch out of many. It wants the metric the claim is about —
#   macro-F1, imbalance-aware — and it is robust to noise because it takes an argmax and keeps
#   the winner.
# * `EarlyStopping` decides "training is over". A macro-F1 computed over a validation split
#   with thin per-class support is a noisy quantity; halting on it stops on a fluctuation.
#   Validation loss is smoother, is not the reported metric, and therefore does not become a
#   selection criterion by being the stopping criterion.
#
# Lab 5.1 used `EarlyStopping(monitor="val_macro_f1", mode="max")`, which is defensible and is
# not what the graded script does. Here you use the graded configuration, and you should be
# able to say which one you used and why. What is *not* defensible is monitoring accuracy on
# classes with n = 1, where accuracy is a coin flip with a label on it.

# %% [markdown]
# ## Part 6 — Batch size is steps-per-epoch, and steps-per-epoch is everything
#
# Batch 256 is a reasonable default at 160 000 patches. Against a training pool of a few
# hundred it is **three gradient steps per epoch**: the optimiser visits the data three times
# before the epoch ends, the BatchNorm running statistics barely move, and the head collapses
# onto the majority class. Lab 5.1 in 2025/26 used 256 and reported a model that predicted one
# class.
# %%
print(f"  n_train = {len(train_idx)}")
print(f"  {'batch':>6} {'steps/epoch':>12} {'steps in 15 epochs':>19}")
for bs in (256, 128, 64, 32, 16):
    print(f"  {bs:>6} {math.ceil(len(train_idx) / bs):>12} "
          f"{math.ceil(len(train_idx) / bs) * EPOCHS:>19}")
BATCH = 32  # TODO(you): justify against the row above, not against habit
STEPS_PER_EPOCH = math.ceil(len(train_idx) / BATCH)
print(f"\n  using batch {BATCH}: {STEPS_PER_EPOCH} steps/epoch, "
      f"{STEPS_PER_EPOCH * EPOCHS} total steps over {EPOCHS} epochs")
print("  This also fixes what T_max means: with interval='epoch', T_max=EPOCHS is the")
print("  half-period in the unit the scheduler is actually stepped in.")

# %% [markdown]
# ## Part 7 — Train one arm, and write a checkpoint you can open
#
# The evaluation contract, stated before the code:
#
# * `pl.seed_everything(seed, workers=True)` before construction; `deterministic="warn"`.
# * `ModelCheckpoint(monitor="val_macro_f1", mode="max", save_top_k=1, save_last=True)` —
#   imbalance-aware, and `save_last` so best-versus-last is comparable rather than assumed.
# * `EarlyStopping(monitor="val_loss", mode="min")`, exactly as the graded script does it.
# * The cosine schedule stepped per **epoch**.
# * A `CSVLogger` in the run directory, so curves exist. Lab 6 in 2025/26 ran a paid 4-GPU
#   allocation with `logger=False` and there was nothing to inspect afterwards.
# * Metrics computed **once** from the full arrays, over a **fixed** label set, on **test**,
#   from the **best** checkpoint. All four of those words were violated in 2025/26.

# %%
ARMS = {
    "none": {"imbalance": None, "note": "plain cross-entropy, no rebalancing"},
    "weights_sqrt": {"imbalance": "sqrt_inverse_freq", "note": "class weights ~ 1/sqrt(n)"},
    "weights_inv": {"imbalance": "inverse_freq", "note": "class weights ~ 1/n"},
    "sampler": {"imbalance": "weighted_sampler", "note": "WeightedRandomSampler, ~1/n"},
    "focal": {"imbalance": "focal", "note": "focal loss, gamma=2"},
}
ARM = "weights_sqrt"  # TODO(you): none / weights_sqrt / weights_inv / sampler / focal
# results.json is append-only and shared with Lab 5.1, whose records are also named
# <arm>_<seed>. record_run refuses a duplicate id, and that refusal is correct. If it fires
# here, set RUN_SUFFIX = "_52" and re-run; do not delete the earlier records.
RUN_SUFFIX = ""  # TODO(you): "" normally, "_52" if record_run refuses the id
assert ARM in ARMS, f"ARM must be one of {sorted(ARMS)}"
REJECTED = {a: ("chosen" if a == ARM else "TODO(you): why not") for a in ARMS}
print(f"  chosen arm: {ARM} ({ARMS[ARM]['note']})")
print("  The full arm x seed grid is slurm/submit_sweep.sbatch's job, not this kernel's.")


def collect_predictions(ckpt_path, datamodule, split="test"):
    """Evaluate a named checkpoint, not whatever happens to be in memory."""
    module = CorineModule.load_from_checkpoint(str(ckpt_path),
                                               train_counts=datamodule.train_counts)
    module.eval()
    loader = (datamodule.test_dataloader() if split == "test"
              else datamodule.val_dataloader())
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(module(xb).argmax(1).cpu())
            targets.append(yb.cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), module


# %%
def run_arm(arm, seed, man, epochs=EPOCHS, batch=BATCH, tag=""):
    """One arm, one seed. Mirrors scripts/train_cnn.py's callbacks exactly."""
    pl.seed_everything(seed, workers=True)
    imb = ARMS[arm]["imbalance"]
    run_id = f"{arm}_{seed}{tag}"
    dm_run = CorineDataModule(x_all, y_all, man, batch_size=batch, imbalance=imb,
                              num_workers=NUM_WORKERS, seed=seed)
    dm_run.setup("fit")
    module = CorineModule(n_classes=LM.n_classes, in_channels=IN_CHANNELS, imbalance=imb,
                          train_counts=dm_run.train_counts, max_epochs=epochs)
    run_root = paths.ensure(paths.run_dir(run_id))[0]
    best = ModelCheckpoint(dirpath=str(run_root / "ckpt"), monitor="val_macro_f1", mode="max",
                           save_top_k=1, save_last=True,
                           filename="best-{epoch}-{val_macro_f1:.4f}")
    stop = EarlyStopping(monitor="val_loss", mode="min", patience=8)
    trainer = pl.Trainer(max_epochs=epochs, accelerator=ACCELERATOR, devices=1,
                         deterministic="warn", default_root_dir=str(run_root),
                         enable_progress_bar=False,
                         logger=CSVLogger(str(run_root), name="logs"), callbacks=[best, stop])
    trainer.fit(module, datamodule=dm_run)
    dm_run.setup("test")
    n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"run_id": run_id, "arm": arm, "seed": seed, "dm": dm_run, "module": module,
            "trainer": trainer, "best": best, "n_params": n_params}


print("  run_arm defined: it seeds, takes its split from the manifest, monitors val_macro_f1,")
print("  stops on val_loss, logs to CSV, and returns the checkpoint paths.")

# %%
res = run_arm(ARM, SEED, manifest, tag=RUN_SUFFIX)
best, dm_run = res["best"], res["dm"]
print(f"  run {res['run_id']}: {res['n_params']:,} trainable parameters")
print(f"  checkpoint monitor: {best.monitor} mode={best.mode}")
print(f"  best  : {os.path.basename(best.best_model_path)}  "
      f"score {float(best.best_model_score):.4f}")
print(f"  last  : {os.path.basename(best.last_model_path)}")
print("  Lab 5.2 in 2025/26 trained a 122 M-parameter VGG-16 on 8,278 patches and never")
print("  discussed the ratio. Parameter count belongs in the config for that reason.")

# %% [markdown]
# ### The checkpoint round-trip
#
# A result becomes checkable when someone else can open the file that produced it. Save to
# `paths.artifacts_dir()` — project space, which survives the 90-day scratch sweep — name the
# file `<run_id>.pt`, and **read it back with `weights_only=True`** in the same notebook. If
# the reload works here it works for the grader. If it does not, you find out now rather than
# in the week you hand in.

# %%
CKPT_PT = paths.ensure(paths.artifacts_dir())[0] / f"{res['run_id']}.pt"
torch.save({
    "state_dict": res["module"].state_dict(),
    "hyper_parameters": dict(res["module"].hparams),
    "run_id": res["run_id"],
    "split_manifest_hash": manifest.manifest_hash,
    # str(), not torch.__version__ itself: that is a TorchVersion object, and an object in
    # the payload is exactly the class of thing weights_only=True refuses. Same lesson as
    # the numpy-array-in-hparams trap two Parts back, caught here by loading it back.
    "torch": str(torch.__version__),
}, CKPT_PT)
reloaded = torch.load(CKPT_PT, map_location="cpu", weights_only=True)
fresh = CorineModule(**reloaded["hyper_parameters"])
fresh.load_state_dict(reloaded["state_dict"])
max_diff = max((a - b).abs().max().item()
               for a, b in zip(fresh.state_dict().values(), res["module"].state_dict().values()))
print(f"  wrote {CKPT_PT.name} ({CKPT_PT.stat().st_size / 1e6:.2f} MB)")
print(f"  reloaded with weights_only=True; run_id={reloaded['run_id']}")
print(f"  max |state_dict difference| after the round-trip: {max_diff:.3e}")
print(f"  hparams travelled with it: {reloaded['hyper_parameters']}")

# %% [markdown]
# ## Part 8 — `ckpt_path="best"`, and what you get if you omit it
#
# `trainer.test(model, datamodule)` with a model supplied and `ckpt_path=None` uses **the
# weights currently in memory** — the end of the last epoch that ran. Lightning does not warn.
# `trainer.test(model, datamodule, ckpt_path="best")` restores the checkpoint
# `ModelCheckpoint` selected. In 2025/26 the callbacks were commented out *and* the test call
# passed the in-memory module, so the entire checkpoint-selection machinery was never
# exercised, and the notebook's promise of "model checkpoints for evaluation" was false.
#
# The uncomfortable part is that both calls print the same-shaped table. Below you run both,
# and then compare the in-memory weights against the selected checkpoint tensor by tensor,
# because the printed output alone will not tell you which weights produced it.

# %%
out_default = res["trainer"].test(res["module"], datamodule=dm_run)
out_best = res["trainer"].test(res["module"], datamodule=dm_run, ckpt_path="best")
print(f"  test_loss, ckpt_path omitted : {out_default[0]['test_loss']:.6f}")
print(f"  test_loss, ckpt_path='best'  : {out_best[0]['test_loss']:.6f}")

_, _, from_best = collect_predictions(best.best_model_path, dm_run, "test")
in_mem = {k: v.detach().cpu() for k, v in res["module"].state_dict().items()}
ckpt_sd = from_best.state_dict()
n_diff = sum(1 for k in in_mem if not torch.equal(in_mem[k], ckpt_sd[k].cpu()))
print(f"  tensors where in-memory != best checkpoint: {n_diff} of {len(in_mem)}")
print("  If that count is non-zero, the two tables above are two different models' answers and")
print("  nothing in the printed output says which is which. If it is zero, this run's best")
print("  epoch happened to be its last one — which is luck, not a reason to omit ckpt_path.")
# %% [markdown]
# ## Part 9 — Validation and test, recorded separately
#
# The contract for the rest of the course: **validation** is what you tune on — the split
# `ModelCheckpoint` and `EarlyStopping` are allowed to see. **Test** is scored once, after every
# decision is frozen, and never informs one. `results.record_run` takes `test_metrics` and
# `val_metrics` as separate arguments precisely so the two cannot be confused in the artifact,
# and `gates.gate_no_test_set_selection` reads the record and fails it closed if you declare
# that you tuned on test.
#
# The 2025/26 notebook had one diagnostics cell, and it evaluated the validation loader. There
# was no code path in the whole file that produced a per-class test number.

# %% [markdown]
# ### Exercise 3 — predict the val-versus-test gap
#
# Your validation split is a scene that contributed nothing to training, and your test split is
# a *different* held-out scene. Both are honest. They are not the same terrain, the same date,
# or the same illumination.
#
# > **P3.** Before running, write down:
# >
# > 1. The **sign** of `test_balanced_acc − val_balanced_acc` for your best checkpoint (`+` or
# >    `−`), and how confident you are.
# > 2. The **magnitude**, to two decimals.
# > 3. Whether the **minimum per-class test support** clears `gates.MIN_CLASS_SUPPORT`.
# >
# > Most people predict a drop, because that is what the phrase "generalisation gap" implies.
# > Record what you actually predicted against what actually came out, and explain the
# > disagreement in terms of the two scenes rather than in terms of hope.

# %%
P3 = {
    # TODO(you): fill in BEFORE running the next two cells.
    "gap_sign": None,           # "+" or "-"
    "gap_magnitude": None,      # e.g. 0.05
    "min_test_support_clears_min": None,  # True / False
    "why": "TODO(you): name one property of the test scene that could move the gap either way.",
}

# %%
def score_run(r, split):
    idx = test_idx if split == "test" else val_idx
    preds, targets, _ = collect_predictions(r["best"].best_model_path, r["dm"], split)
    return metrics.evaluate(targets, preds, LABELS, codes=LM.codes, names=CLASS_NAMES,
                            groups=PS.scene[idx], n_boot=1000, seed=r["seed"])


val_m = score_run(res, "val")
test_m = score_run(res, "test")
print("=== VALIDATION (the tuning set; never a headline) ===")
print(val_m.table(min_support=gates.MIN_CLASS_SUPPORT))
print("\n=== TEST (held-out scene, scored once, from the best checkpoint) ===")
print(test_m.table(min_support=gates.MIN_CLASS_SUPPORT))

# %%
gap_bal = test_m.balanced_acc - val_m.balanced_acc
gap_f1 = test_m.macro_f1 - val_m.macro_f1
print(f"  val  acc {val_m.overall_acc:.4f}  bal_acc {val_m.balanced_acc:.4f}  "
      f"macro-F1 {val_m.macro_f1:.4f}")
print(f"  test acc {test_m.overall_acc:.4f}  bal_acc {test_m.balanced_acc:.4f}  "
      f"macro-F1 {test_m.macro_f1:.4f}")
print(f"  gap    acc {test_m.overall_acc - val_m.overall_acc:+.4f}  "
      f"bal_acc {gap_bal:+.4f}  macro-F1 {gap_f1:+.4f}")
print(f"  min per-class support: val "
      f"{int(val_m.support[val_m.support > 0].min())}   "
      f"test {int(test_m.support[test_m.support > 0].min())}")
if test_m.ci:
    print(f"  scene-clustered 95% CI on test: bal_acc "
          f"{np.round(test_m.ci['balanced_acc'], 4).tolist()}  macro-F1 "
          f"{np.round(test_m.ci['macro_f1'], 4).tolist()} over "
          f"{test_m.ci['n_clusters']} clusters")
print(f"  P3 answers were: {P3}")
print("  Write predicted vs actual above. A gap smaller than the CI is not a gap.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
idx_k = np.arange(LM.n_classes)
w = 0.38
axes[0].bar(idx_k - w / 2, val_m.support, w, label="val")
axes[0].bar(idx_k + w / 2, test_m.support, w, label="test")
axes[0].axhline(gates.MIN_CLASS_SUPPORT, c="#c0392b", ls="--", lw=1,
                label=f"MIN_CLASS_SUPPORT={gates.MIN_CLASS_SUPPORT}")
axes[0].set_ylabel("support"); axes[0].set_yscale("log")
axes[0].set_title("per-class support: the denominator of every recall")
axes[0].legend(fontsize=7)
axes[1].bar(idx_k - w / 2, val_m.recall, w, label="val recall")
axes[1].bar(idx_k + w / 2, test_m.recall, w, label="test recall")
axes[1].set_xticks(idx_k)
axes[1].set_xticklabels([n[:14] for n in CLASS_NAMES], rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("recall"); axes[1].set_ylim(0, 1.05)
axes[1].set_title("per-class recall, same checkpoint")
axes[1].legend(fontsize=7)
fig.tight_layout()
FIG_SUP = paths.ensure(paths.results_dir())[0] / "lab5_2_val_vs_test_support.png"
fig.savefig(FIG_SUP, dpi=110)
plt.show()
print(f"  saved {FIG_SUP}")

# %% [markdown]
# ## Part 10 — Three seeds, or it is not a result
#
# `gates.MIN_SEEDS = 3` and `gates.MIN_SEED_MULTIPLIER = 2.0`: an improvement must exceed
# **twice the seed spread** to be written down as an improvement. Below that, the honest
# sentence is "no detectable effect at n = 3", and `gate_claim_supported` raises until you
# write that sentence instead.
#
# This is not pedantry aimed at your model. It is aimed at the 2025/26 headline, whose own
# sampling noise was about ±0.05 on a single split with n = 1 classes — larger than the entire
# effect the lab asked students to explain.

# %%
SEEDS = [0, 1, 2]
SEED_RUNS = [{"run_id": res["run_id"], "seed": SEED, "arm": ARM, "res": res,
              "test": test_m.to_dict(), "val": val_m.to_dict()}]
for s in SEEDS[1:]:
    r = run_arm(ARM, s, manifest, tag=RUN_SUFFIX)
    SEED_RUNS.append({"run_id": r["run_id"], "seed": s, "arm": ARM, "res": r,
                      "test": score_run(r, "test").to_dict(),
                      "val": score_run(r, "val").to_dict()})
    print(f"  seed {s}: test acc {SEED_RUNS[-1]['test']['overall_acc']:.4f}  "
          f"bal_acc {SEED_RUNS[-1]['test']['balanced_acc']:.4f}  "
          f"macro-F1 {SEED_RUNS[-1]['test']['macro_f1']:.4f}")
print(f"  {len(SEED_RUNS)} seeds; run_ids are <arm>_<seed>: "
      f"{[r['run_id'] for r in SEED_RUNS]}")

# %%
var = gates.gate_seeds_and_variance(SEED_RUNS, metric="balanced_acc")
var_f1 = gates.gate_seeds_and_variance(SEED_RUNS, metric="macro_f1")
bal = np.array([r["test"]["balanced_acc"] for r in SEED_RUNS])
f1 = np.array([r["test"]["macro_f1"] for r in SEED_RUNS])
val_bal = np.array([r["val"]["balanced_acc"] for r in SEED_RUNS])
print(f"  {ARM} over {var['n']} seeds")
print(f"    test balanced_acc  mean {var['mean']:.4f}  std {var['std']:.4f}  "
      f"range [{bal.min():.4f}, {bal.max():.4f}]")
print(f"    test macro-F1      mean {var_f1['mean']:.4f}  std {var_f1['std']:.4f}  "
      f"range [{f1.min():.4f}, {f1.max():.4f}]")
print(f"    val   balanced_acc mean {val_bal.mean():.4f}  "
      f"std {val_bal.std(ddof=1):.4f}")
print(f"    claim threshold: an improvement must exceed {var['threshold']:.4f} "
      f"({gates.MIN_SEED_MULTIPLIER} x std)")
# %%
BASELINES = bl.run_all(
    x_train=x_all[train_idx], y_train_codes=LABEL_CODES[train_idx],
    x_test=x_all[test_idx], y_test_codes=LABEL_CODES[test_idx],
    labels=LABELS, codes=LM.codes,
    train_scene_ids=PS.scene[train_idx], test_scene_ids=PS.scene[test_idx], seed=SEED,
)
BEST_BASE = max(BASELINES, key=lambda k: BASELINES[k]["macro_f1"])
print(f"  {'baseline':<20} {'acc':>8} {'bal_acc':>8} {'macro_F1':>9}")
for name in sorted(BASELINES, key=lambda k: -BASELINES[k]["macro_f1"]):
    d = BASELINES[name]
    print(f"  {name:<20} {d['overall_acc']:>8.4f} {d['balanced_acc']:>8.4f} "
          f"{d['macro_f1']:>9.4f}")
print(f"  strongest baseline: {BEST_BASE}  macro-F1 "
      f"{BASELINES[BEST_BASE]['macro_f1']:.4f}")
print("  Scored on the same test split, with the same label set, through the same")
print("  metrics.evaluate call as your model. That sameness is the whole point.")

# %%
delta_f1 = float(f1.mean() - BASELINES[BEST_BASE]["macro_f1"])
print(f"  seed-mean macro-F1 {f1.mean():.4f} - best baseline "
      f"{BASELINES[BEST_BASE]['macro_f1']:.4f} = {delta_f1:+.4f}")
try:
    gates.gate_claim_supported(delta_f1, SEED_RUNS, metric="macro_f1",
                               label=f"{ARM} vs {BEST_BASE}")
    print("  gate_claim_supported: the margin exceeds 2 x seed spread")
except gates.GateFailure as exc:
    print(f"  gate_claim_supported raises:\n    {exc}")
    print("  That is a legitimate finding. Report it as 'no detectable effect at n=3',")
    print("  not as an improvement.")

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot([r["seed"] for r in SEED_RUNS], bal, "o-", label=f"{ARM} test balanced_acc")
ax.plot([r["seed"] for r in SEED_RUNS], val_bal, "s--", label="same runs, val balanced_acc")
ax.axhline(var["mean"], c="#2c3e50", lw=1, ls=":", label=f"test mean {var['mean']:.3f}")
ax.axhline(BASELINES[BEST_BASE]["balanced_acc"], c="#c0392b", lw=1.5,
           label=f"best baseline ({BEST_BASE}) {BASELINES[BEST_BASE]['balanced_acc']:.3f}")
ax.set_xlabel("seed"); ax.set_ylabel("balanced accuracy"); ax.legend(fontsize=8)
ax.set_title("seed spread versus the val-test gap: which one is bigger?")
fig.tight_layout()
FIG_SEED = paths.ensure(paths.results_dir())[0] / "lab5_2_seed_spread.png"
fig.savefig(FIG_SEED, dpi=110)
plt.show()
print(f"  saved {FIG_SEED}")
print("  If the val-test gap is smaller than the seed spread, do not narrate the gap.")

# %% [markdown]
# ## Part 11 — Deliberate bug: reproduce the 2025/26 headline procedure
#
# Same trained model, same split. Two changes, each of which the old notebook made: evaluate
# the **validation** split, and derive the label set from the evaluation slice with
# `np.unique(np.concatenate([y_true, y_pred]))`. Nothing here needs retraining — which is the
# point. The old numbers were not produced by a worse model. They were produced by a worse
# *procedure* applied to a perfectly ordinary model.

# %%
preds_v, targets_v, _ = collect_predictions(best.best_model_path, dm_run, "val")
old_labels = np.unique(np.concatenate([targets_v, preds_v]))
cm_old = metrics.confusion(targets_v, preds_v, old_labels)
tp = np.diag(cm_old).astype(float)
rec_old = tp / np.maximum(cm_old.sum(1), 1)
prec_old = tp / np.maximum(cm_old.sum(0), 1)
f1_old = 2 * prec_old * rec_old / np.maximum(prec_old + rec_old, 1e-12)
print("  THE 2025/26 PROCEDURE   (val split, label set derived from the slice)")
print(f"    accuracy {cm_old.trace() / cm_old.sum():.4f}   "
      f"balanced_acc {rec_old.mean():.4f}   macro-F1 {f1_old.mean():.4f}")
print(f"    label set derived from the slice: {len(old_labels)} classes {old_labels.tolist()}")
print(f"    fixed label set                 : {len(LABELS)} classes {LABELS.tolist()}")
print(f"    classes that vanished from the average: "
      f"{sorted(set(LABELS.tolist()) - set(old_labels.tolist()))}")
print(f"    per-class val supports behind those recalls: "
      f"{np.bincount(targets_v, minlength=LM.n_classes).tolist()}")

# %%
print("  THE HONEST PROCEDURE    (test split, fixed label set, best checkpoint)")
print(f"    accuracy {test_m.overall_acc:.4f}   balanced_acc {test_m.balanced_acc:.4f}   "
      f"macro-F1 {test_m.macro_f1:.4f}   kappa {test_m.kappa:.4f}")
print(f"    difference in balanced_acc: {rec_old.mean() - test_m.balanced_acc:+.4f}")
print("  Two numbers, one model. The gap is made of the split you scored on and the")
print("  denominator you averaged over — not of the weights.")

# %%
# And the gate that closes the door. This is what would have caught the 2025/26 headline.
bad_record = {"run_id": "old_procedure", "test_used_for_tuning": True}
try:
    gates.gate_no_test_set_selection(bad_record)
    print("  UNREACHABLE: a record declaring test_used_for_tuning=True passed")
except gates.GateFailure as exc:
    print(f"  gate_no_test_set_selection raises on a record that admits tuning on test:\n"
          f"    {exc}")
print("  It reads the record, not your confidence. Declare the truth and let the gate judge.")

# %% [markdown]
# > **Write this sentence in your report, in your own words, with the numbers above in it:**
# > one sentence explaining why the *validation* number is untrustworthy as a headline. It must
# > name both mechanisms — which split the tuning happened on, and which classes can leave the
# > denominator when the label set is derived from the slice — and not just say "leakage".
# >
# > Note what the sign of the difference does **not** tell you. If the old procedure happens to
# > come out lower on your data, it is no more trustworthy: it is still a number computed on the
# > split the checkpoint was selected on, averaged over a denominator chosen by the model's own
# > predictions. The direction is luck. The provenance is the defect. A sentence you could have
# > written without running the cell has not been earned.

# %% [markdown]
# ## Part 12 — Gate board and deliverable
#
# `results.record_run` writes first, because `run_all_gates` reads the record to check that the
# split hash you reported is the split in use and that you did not select on test. Then the
# board prints.
#
# `results.json` is **append-only** and `record_run` refuses to reuse a `run_id`. Name your runs
# `<arm>_<seed>` — which is what `run_arm` already returns — so a different configuration gets a
# different id and an earlier attempt stays in the record. Overwriting a previous run destroys
# the evidence that you did the work.

# %%
print(results.describe_contract())

# %%
_probe = CorineModule(n_classes=LM.n_classes, in_channels=IN_CHANNELS,
                      imbalance=ARMS[ARM]["imbalance"], train_counts=dm.train_counts,
                      max_epochs=EPOCHS)
N_PARAMS = int(sum(p.numel() for p in _probe.parameters() if p.requires_grad))
CONFIG = {
    "arm": ARM, "imbalance": ARMS[ARM]["imbalance"], "rejected_arms": REJECTED,
    "epochs": EPOCHS, "batch_size": BATCH, "steps_per_epoch": STEPS_PER_EPOCH,
    "n_train": int(len(train_idx)), "n_params": N_PARAMS, "lr": 3e-4,
    "n_classes": LM.n_classes, "in_channels": IN_CHANNELS, "codes": list(LM.codes),
    "class_names": CLASS_NAMES,
    "bands": list(PS.bands), "patch_size": PS.patch_size, "block": sm["block"],
    "test_scenes": sm["test_scenes"], "val_scenes": sm["val_scenes"],
    "normalisation": norm.to_dict(), "norm_fitted_on": "train only",
    "merge_min_support": MERGE_MIN, "merged_codes": MERGED_CODES,
    "dropped_zero_train_codes": DROP_CODES, "rare_other_code": RARE_OTHER,
    "imbalance_ratio": lab_mod.imbalance_ratio(dm.train_counts),
    "checkpoint_monitor": best.monitor, "checkpoint_mode": best.mode,
    "early_stopping_monitor": "val_loss", "early_stopping_mode": "min",
    "scheduler_interval": "epoch", "seed": SEED, "seeds": SEEDS,
    "accelerator": ACCELERATOR, "torch": torch.__version__, "lightning": pl.__version__,
}
print(f"  trainable parameters: {N_PARAMS:,}")
print(f"  checkpoint selected on {best.monitor} ({best.mode}); "
      f"training stopped on val_loss (min)")
# %%
RECORDED = []
for r in SEED_RUNS:
    preds_t, targets_t, _ = collect_predictions(r["res"]["best"].best_model_path,
                                                r["res"]["dm"], "test")
    try:
        rec = results.record_run(
            r["run_id"], lab=LAB, config={**CONFIG, "seed": r["seed"]},
            split_manifest_hash=manifest.manifest_hash, seed=r["seed"],
            test_metrics=r["test"], val_metrics=r["val"], baselines=BASELINES,
            test_used_for_tuning=False, n_seeds=len(SEED_RUNS),
            notes=(f"arm={ARM} ({ARMS[ARM]['note']}); rejected "
                   f"{[k for k, v in REJECTED.items() if v != 'chosen']}; best baseline "
                   f"{BEST_BASE} macro-F1 {BASELINES[BEST_BASE]['macro_f1']:.4f}; seed spread "
                   f"test bal_acc {var['std']:.4f}; val-test gap on bal_acc {gap_bal:+.4f}; "
                   f"merged rare codes {MERGED_CODES} into {RARE_OTHER} on train support < "
                   f"{MERGE_MIN}; dropped zero-train codes {DROP_CODES}; checkpoint "
                   f"{os.path.basename(best.best_model_path)}"),
            extra={"predictions": {"P1": P1, "P2": P2, "P3": P3},
                   "seed_mean_test_balanced_acc": var["mean"],
                   "seed_std_test_balanced_acc": var["std"],
                   "claim_threshold": var["threshold"],
                   "old_procedure_val_metrics": {
                       "balanced_acc": float(rec_old.mean()),
                       "macro_f1": float(f1_old.mean()),
                       "n_label_classes": int(len(old_labels))},
                   "checkpoint_pt": str(CKPT_PT),
                   "best_checkpoint": best.best_model_path,
                   "last_checkpoint": best.last_model_path},
        )
        RECORDED.append(rec)
        gates.gate_predictions_saved(
            targets_t, preds_t, paths.run_dir(r["run_id"]) / "test_predictions.npz",
            manifest.manifest_hash)
        print(f"  recorded {rec['run_id']}  (raw predictions saved so every metric is "
              f"re-derivable)")
    except results.ResultsError as exc:
        print(f"  {r['run_id']} not recorded: {exc}")
print(f"  {len(RECORDED)} of {len(SEED_RUNS)} runs written this session")

# %%
record = RECORDED[0] if RECORDED else results.get_run(SEED_RUNS[0]["run_id"])
lines = gates.run_all_gates(test_m, BASELINES, manifest, record)
lines.append(f"[{'PASS' if var['n'] >= gates.MIN_SEEDS else 'FAIL'}] seed_variance: "
             f"{var['n']} seeds (need {gates.MIN_SEEDS}), std {var['std']:.4f}, "
             f"claim threshold {var['threshold']:.4f}")
lines.append(f"[{'PASS' if CKPT_PT.exists() else 'FAIL'}] checkpoint_reload: "
             f"{CKPT_PT.name} opens with weights_only=True")
ok = gates.print_gate_board(lines)

# %%
print(results.summary_table(LAB))
csv_path = results.write_csv(LAB)
print(f"\n  csv for team comparison: {csv_path}")

# %% [markdown]
# ### Reading a failed board
#
# A `FAIL` here is not a formatting problem. It means a conclusion you would have written down
# is not supported, and the message names the number and the threshold.
#
# * **`class_support`** — a scored class has fewer than 25 test samples. This one can fail even
#   after Part 1's merge, because the merge is decided on **train** counts (deciding it on test
#   support would be test-informed problem definition) and a class can therefore still be thin
#   on test. The fix is more scenes, not a lower threshold. Report the failing supports; do not
#   average over what is left and call it macro.
# * **`beats_baselines`** — the central gate. Before concluding the model is bad, check the two
#   usual causes in order: (1) an input-unit mismatch (`gate_input_units`), and (2) a collapsed
#   head — look at the `pred` column of the metrics table; if one class absorbs nearly
#   everything, the model learned the prior and nothing else.
# * **`above_chance`** — balanced accuracy at or below 1/K. A constant predictor scores this,
#   and overall accuracy can look far higher while meaning exactly the same thing.
# * **`not_collapsed`** — more than 85 % of predictions in one class.
# * **`split_hash`** — the record's `split_manifest_hash` is not the current split. You are
#   reporting numbers from a split you no longer have. Retrain or re-report.
# * **`seed_variance`** — fewer than `MIN_SEEDS` runs carry the metric. Nothing about a single
#   run on this data is a result.
#
# A failing board that you diagnose and report is worth more than a passing board you do not
# understand.

# %% [markdown]
# ## Submission checklist
#
# Everything below is a file the grader can open. Self-attestation is not an artifact.
#
# * `results/results.json` — one `lab5.2` record **per seed**, each carrying **both** `test` and
#   `val` metric blocks, `split_manifest_hash`, `seed`, `n_seeds`, the full `baselines` block,
#   `test_used_for_tuning: false`, and a `config` naming the checkpoint monitor, the
#   early-stopping monitor, `scheduler_interval`, `steps_per_epoch`, `batch_size`, `n_params`,
#   the normalisation dict and the merge decision.
# * `results/results.csv` — from `results.write_csv("lab5.2")`.
# * `results/lab5_2_scheduler_interval.png`, `lab5_2_val_vs_test_support.png`,
#   `lab5_2_seed_spread.png`.
# * `artifacts/<arm>_<seed>.pt` — reloadable with `weights_only=True`.
# * `artifacts/runs/<arm>_<seed>/ckpt/best-*.ckpt`, `last.ckpt`, `test_predictions.npz`, and the
#   `logs/` directory the CSV logger wrote.
# * `data/splits/split_<hash>.json` — the manifest whose hash is in every record.
#
# In the write-up:
#
# 1. Predicted vs actual for **P1, P2 and P3**, with every disagreement explained by a
#    mechanism. A prediction that matched still needs the mechanism stated.
# 2. The val and test numbers **side by side**, labelled as such, plus the seed spread — and a
#    statement of whether the val-test gap is larger than the seed spread.
# 3. `mean ± spread` over three seeds, and "no detectable effect at n = 3" wherever the margin
#    is under `2 × std`.
# 4. The gate board with every `FAIL` diagnosed, including `class_support` if it fired.
# 5. The negative-control sentence from Part 11, with its numbers in it.
# 6. Which metric selected the checkpoint, which metric stopped training, and why they differ —
#    one paragraph, naming the noise argument.
# 7. The arms you did **not** pick and one reason each, as recorded in `config`.
# 8. One claim you cannot make from this experiment, and why.

# %% [markdown]
# ## Where each 2025/26 defect went
#
# | 2025/26 | 2026/27 |
# |---|---|
# | headline `0.9231 / 0.8338 / 0.8376` computed on the validation loader | val and test evaluated from the same checkpoint and recorded as separate `val` / `test` blocks |
# | supports `9, 1, 245, 99, 6, 1, 5, 52, 30, 7` never printed | `Metrics.table(min_support=25)` prints the support column; `gate_class_support` raises on it |
# | label set derived from `np.unique` of the evaluation slice | fixed `LABELS = np.arange(K)`; Part 11 shows the difference on your own predictions |
# | `trainer.test(model, datamodule)`, no checkpoint | `ckpt_path="best"`, plus an explicit in-memory-versus-checkpoint tensor comparison |
# | `ModelCheckpoint` commented out, no `default_root_dir` | `ModelCheckpoint(monitor="val_macro_f1", mode="max", save_last=True)` into `paths.run_dir(run_id)` |
# | `EarlyStopping(monitor="val_acc", mode="min")` | `EarlyStopping(monitor="val_loss", mode="min")`, and the backwards direction demonstrated on a toy module |
# | cosine scheduler returned with no interval | `interval="epoch"` in `CorineModule`; Part 4 counts the cycles you actually got |
# | `batch_size=256` → 3 steps/epoch → collapse | Part 6 prints `ceil(n_train/batch)` for five batch sizes; batch 32 used here |
# | no `save_hyperparameters` in the VGG cell | `CorineModule.save_hyperparameters(ignore=["train_counts"])`; Part 2 reproduces both failure modes |
# | a numpy array in hparams | `register_buffer`, and a `weights_only=True` round-trip in the same notebook |
# | a `*_data.npz` glob and a positional npz read | `patches.SCENE_GLOB` + `load_all_scenes`; `gate_no_duplicate_patches` on load |
# | `get_split` allowed `num_train < 0`; `setup()` re-split per call | `splits.group_block_split` plus a hashed `SplitManifest` passed into the DataModule |
# | class-drop threshold `< 10` computed over all data | merge/drop decided on **train** counts only, and the decision recorded in `config` |
# | zero `seed_everything`, one split, two conflicting splits | `pl.seed_everything(seed, workers=True)`, `deterministic="warn"`, three seeds, one manifest |
# | 122 M-parameter VGG on 8,278 patches, unremarked | `n_params` in `config`; `SmallCNN` is ~20k here |
# | zero `to_csv` / `json.dump` / `torch.save`, `logger=False` | `results.record_run` per seed, `write_csv`, `<run_id>.pt`, checkpoints, raw predictions, CSVLogger |
# | ran on an interactive GPU node, no `.sbatch` in the repo | `slurm/train_jureca.sbatch` and `slurm/submit_sweep.sbatch`, `dc-cpu`/`dc-gpu`, one arm per job |
# | summary cell described a model that was not the one evaluated | the summary is `results.summary_table("lab5.2")`, generated from the records |
