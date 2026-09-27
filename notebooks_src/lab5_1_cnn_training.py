# %% [markdown]
# # Lab 5.1 — Baselines, a defensible split, and a model that has to earn its number
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# This lab trains a small CNN on the CORINE patch archives Lab 4.2 wrote, and produces
# four things: a **baseline table scored before any neural network exists**, a **split
# with a hash** that makes "the split I reported is the split I used" checkable, a
# **three-seed result with a spread**, and a **`results.json` record** carrying every
# decision you made. The gate board at the end decides whether your claim survives.
#
# It assumes Lab 4.2 wrote `patches_<scene>_scene.npz` archives with `scene`, `row` and
# `col` into `paths.training_data_dir()`. It produces the arm comparison Lab 5.2
# continues and Lab 7 reports.
#
# **You must run every cell yourself.** Generated notebooks ship with no outputs, on
# purpose. The 2025/26 version of this notebook was committed with `trainer.test()`
# unexecuted — zero `test_acc`, zero `Testing` table anywhere in its stored output — and
# every "result" in its summary cell was a **validation** number. The notebook read as
# complete. A green-looking notebook is not evidence.

# %% [markdown]
# ## Why this notebook exists
#
# In 2025/26 the workflow was: open the notebook, run every cell, read the printed
# accuracy, write it in the report, pass. Nothing in that chain could distinguish a model
# that had learned land cover from a model that had learned the class prior. The numbers
# that came out of it — `Overall accuracy: 0.6992 / Balanced accuracy: 0.5196` here, and
# `0.9231 / 0.8338 / 0.8376` in Lab 5.2 — were then quoted forward into Lab 7 as the
# course's published result.
#
# This notebook closes that path. Before a single weight is initialised you will know
# what chance scores on your test split. After training, a gate board raises if your model
# does not clear it by a margin larger than your own seed noise.
#
# > A model that does not beat the table in Part 6 is an expensive random number
# > generator, and this notebook says so out loud.

# %% [markdown]
# ## The 2025/26 defects this notebook closes
#
# Each row names the old defect, what it cost, and the Part that fixes it.
#
# | old defect | consequence | fixed in |
# |---|---|---|
# | a classmate's username hard-coded as the data path | only one student in the class could run the notebook; three other spellings of the same path existed and two resolved to nothing | Part 1 |
# | `glob("*_data.npz")` matched the per-scene files **and** `combined_training_data.npz` | every patch loaded twice; the random split then put duplicates in train *and* test | Part 1 |
# | `rng.permutation` over edge-adjacent 3 × 3 patches | neighbours sharing a 100 m CORINE label landed on both sides of the split; 0.6992 was partly an interpolation score | Part 2 |
# | no group or block variable anywhere in the notebook | a leakage-free split was not constructible even for someone who wanted one | Part 2 |
# | `np.unique(y)` over the whole dataset set `num_classes` | the class set was test-informed | Part 3 |
# | per-class metrics reported from n = 1–23 samples | recall on n = 1 is literally 0 or 1, printed to three decimals as if it were a measurement | Part 3, Part 12 |
# | no split record at all | "the split I reported == the split I used" was unverifiable | Part 2 |
# | percentile bounds fitted over the whole raster, before any split existed | every test patch contributed to the transform applied to every training patch | Part 5 |
# | `X = X.astype(np.float32) * 0.0001` on data Lab 4.2 had already stretched to [0, 1] | the network saw values of order 1e-5; BatchNorm absorbed it, so nothing crashed and nobody noticed for a year | Part 5 |
# | no baseline was ever *scored* | Lab 6 shipped a fine-tuned Prithvi at test accuracy exactly 1/10 — chance — against a majority baseline of 0.538, and the unit sheet was satisfied | Part 6 |
# | `np.unique(np.concatenate([y_true, y_pred]))` as the metric label set | a class the model never predicts silently left the average, so balanced accuracy was computed over fewer classes than exist | Part 9 |
# | `batch_size = 256` against a few hundred training patches | 3 gradient steps per epoch; the head collapsed onto the majority class | Part 8 |
# | `EarlyStopping(monitor="val_acc", mode="min")` | stops training at the **worst** accuracy — in practice epoch 1 | Part 9 |
# | cosine scheduler returned without `interval="epoch"` | Lightning steps schedulers per batch by default, so the LR ran hundreds of full cosine cycles instead of annealing once | Part 9 |
# | zero `seed_everything` calls in the whole notebook | run-to-run noise exceeded every effect the lab asked students to compare | Part 0, Part 10 |
# | zero `to_csv` / `json.dump` / `torch.save` | nothing was persisted, so nothing was checkable and nothing was gradeable | Part 13 |
# | summary cell said "Batch Size 64" (code: 256) and "10x more frequent" (code printed 396.0x) | the report described a configuration the code did not run | Part 13 |

# %% [markdown]
# ## Part 0 — Setup
#
# Three things happen here and the order matters.
#
# 1. `MPLCONFIGDIR` is set **before** matplotlib is imported. The 2025/26 notebooks set it
#    after the import, which does nothing, and shipped
#    `Matplotlib created a temporary cache directory …` in committed output.
# 2. **One** `np.random.default_rng(seed)` for the whole notebook, created here and passed
#    down. The old notebook called `np.random.permutation` and `np.random.choice` with no
#    seed anywhere in it — `pl.seed_everything`, `torch.manual_seed` and `np.random.seed`
#    appear zero times — so its dataset could not be rebuilt and its `seed=42` recovered
#    nothing.
# 3. Every path comes from `eo_course.paths`. No username appears anywhere in this file.

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
# Windows starts dataloader workers by spawn and does not clean up persistent workers
# between repeated DataLoaders in one kernel; Linux (JURECA) forks and is fine. Zero
# workers costs nothing at this patch size, so the notebook stays portable.
NUM_WORKERS = 0 if os.name == "nt" else 2
print(f"  one rng, seed {SEED}; every subsample and bootstrap below receives it")
print(f"  dataloader workers: {NUM_WORKERS}")

# %% [markdown]
# ### Where the heavy training actually belongs
#
# This notebook trains small, short runs so you can see the machinery work. The graded
# run — full epochs, all five arms, three seeds — belongs in a job:
#
# ```bash
# sbatch slurm/train_jureca.sbatch                          # one arm, one seed
# sbatch --export=ALL,ARM_TAG=weights_sqrt slurm/train_jureca.sbatch
# sbatch slurm/submit_sweep.sbatch                          # the whole arm x seed grid
# ```
#
# Both call `scripts/train_cnn.py`, which is the same pipeline you are about to run
# cell-by-cell. Keeping one implementation is the point: a notebook that drifts from the
# graded script produces numbers nobody can reproduce.

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
        "  WARNING: login node, no GPU. A login node is shared, I/O-throttled and has no\n"
        "  GPU allocation, so a long run here is slow for you and rude to everyone else.\n"
        "  Use these cells to verify the pipeline, then submit slurm/train_jureca.sbatch."
    )

# %% [markdown]
# ## Part 1 — The Lab 4.2 hand-off
#
# `paths.require_existing` names the producing lab when an artifact is missing, so a
# skipped step points at the step instead of surfacing as a mystery `FileNotFoundError`.
#
# The 2025/26 notebook hard-coded a classmate's username as its data path. That is not a
# style problem: the notebook's provenance was someone else's directory, and the student
# running it was reading data they had not produced and could not regenerate. Everything
# here resolves from environment variables through `paths`.

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
print(f"  reading {len(ARCHIVES)} archives matched by patches.SCENE_GLOB = {pat.SCENE_GLOB!r}")
for a in ARCHIVES:
    print(f"    {a.name:<40} {a.stat().st_size / 1e6:6.2f} MB")
print("  The suffix is `_scene.npz`, not `_data.npz`, on purpose: a combined file cannot")
print("  collide with it, so the 2025/26 double-load is unrepresentable in this design.")

# %%
PS = pat.load_all_scenes(TRAIN_DIR)
print(f"  {len(PS)} patches, {len(np.unique(PS.scene))} scenes, bands {list(PS.bands)}")
print(PS.summary())
with np.load(ARCHIVES[0]) as _d:
    gates.gate_npz_loaded_by_key(_d, required=("patches", "labels", "scene", "row", "col"))
    print(f"  {ARCHIVES[0].name} keys: {sorted(_d.files)}")
print("  loaded by key, not by position; provenance (scene/row/col) present")

# %%
# Lab 5.2's `glob("*_data.npz")` matched per-scene files AND a combined permutation of
# them, so every patch entered the array twice and the random split put duplicates on both
# sides. This gate catches that class of bug however it arose.
gates.gate_no_duplicate_patches(PS.patches, groups=PS.scene, seed=SEED)
print("  gate_no_duplicate_patches passed: no patch appears twice")

# %%
print(f"  raw stored range: [{PS.patches.min():.4f}, {PS.patches.max():.4f}]")
print(f"  patch {PS.patches.shape[1]}x{PS.patches.shape[2]} x {PS.patches.shape[3]} bands, "
      f"stride {PS.stride}")
print("  Nothing is normalised here. Lab 4.2 deliberately stored raw values and left the")
print("  transform to this lab, because a transform fitted before a split exists cannot be")
print("  honest. Part 5 does the fitting.")

# %% [markdown]
# ## Part 2 — The split you choose and defend
#
# Lab 4.2 measured the fact that governs this whole Part. Quote it back:
#
# > A 3 × 3 patch spans 30 m. CORINE is **100 m**, so one label pixel is a 10 × 10 block of
# > image pixels and is the centre pixel of roughly `floor(10/3)² = 9` patches. Those ~9
# > patches are **not independent samples** — they share a label exactly, and at
# > stride = patch size they share image pixels at their edges.
#
# That gives two independent leaks, and a random permutation is exposed to both:
#
# 1. **Neighbour leakage.** Near-duplicate patches carrying the same label land in train
#    and test. The model is asked to interpolate, not to generalise.
# 2. **Scene leakage.** Each Sentinel-2 scene carries its own illumination, atmosphere,
#    sensor state and date. If every split receives patches from every scene,
#    "generalisation" reduces to "generalising within a scene you have already seen".
#
# `splits.group_block_split` removes both: whole scenes are held out, and within the
# remaining scenes the unit of assignment is a spatial **block** of patches, never a patch.

# %%
SCENES = sorted(map(str, np.unique(PS.scene)))
print(f"  {len(SCENES)} scenes available:")
print(f"  {'scene':<26} {'n':>6} {'classes':>8} {'mean NIR':>10}")
for s in SCENES:
    m = np.asarray(PS.scene, dtype=object) == s
    print(f"  {s:<26} {int(m.sum()):>6} {len(np.unique(PS.labels[m])):>8} "
          f"{float(PS.patches[m, ..., -1].mean()):>10.4f}")
print("  The last column is the NIR band mean. If it differs materially between scenes,")
print("  that difference is exactly what a scene-held-out split refuses to let you exploit.")

# %% [markdown]
# ### Exercise 1 — predict the split before you build it
#
# You are going to hold out whole scenes for test and carve validation out of one of the
# remaining scenes, with `block=10` and `val_ratio=0.15`.
#
# > **P1.** Write down, before running:
# >
# > 1. How many of your scenes will be in the **test** split?
# > 2. What fraction of all patches that puts in test (nearest 10 %).
# > 3. How many **distinct** scenes appear in the **train** split.
# > 4. If you rebuilt the split with the same `seed` and the same archives, would
# >    `manifest_hash` change?
# >
# > Question 4 is the one the whole Part exists to make checkable.

# %%
P1 = {
    # TODO(you): fill in BEFORE running the next two cells. Leaving these None is scored as
    # not attempted.
    "n_test_scenes": None,
    "test_fraction_pct": None,
    "n_train_scenes": None,
    "hash_changes_on_rebuild": None,  # True or False
    "why": "TODO(you): in one sentence, what does manifest_hash make verifiable that a "
           "printed split ratio does not?",
}

# %%
TEST_SCENES = SCENES[-1:]
manifest = splits.group_block_split(PS.scene, PS.row, PS.col, block=10, val_ratio=0.15,
                                    test_scenes=TEST_SCENES, seed=SEED)
sm = manifest.summary()
print(f"  method          : {sm['method']}   hash {manifest.manifest_hash}")
print(f"  n train/val/test: {sm['n_train']} / {sm['n_val']} / {sm['n_test']} of {sm['n_total']}")
print(f"  test fraction   : {100 * sm['n_test'] / sm['n_total']:.1f}%")
print(f"  scenes in train : {sm['n_scenes_train']}   val {sm['n_scenes_val']}   "
      f"test {sm['n_scenes_test']}")
print(f"  test scenes     : {sm['test_scenes']}   val scenes: {sm['val_scenes']}")
print(f"  P1 answers were : {P1}")

# %%
print("  This split is provisional. Part 3 filters the class set on TRAIN counts and then")
print("  rebuilds it, because the manifest of record must describe the data you actually")
print("  train on, not an earlier draft of it.")
print(f"  provisional manifest_hash = {manifest.manifest_hash}")

# %%
train_idx0, test_idx0 = manifest.train_idx, manifest.test_idx
gates.gate_split_is_grouped(manifest)
gates.gate_split_disjoint(manifest)
leaks = {k: v for k, v in manifest.group_leakage().items() if v}
print(f"  gate_split_is_grouped passed; scenes straddling splits: {leaks or 'none'}")
ov = {k: len(v) for k, v in manifest.overlaps().items() if v}
print(f"  gate_split_disjoint passed; index overlaps: {ov or 'none'}")

# %% [markdown]
# ### The choice you own
#
# `TEST_SCENES = SCENES[-1:]` above is the course default, not a recommendation you should
# stop questioning. Holding out a fixed scene means the whole class shares one test set, so
# teams are comparable and nobody can select on it — that is why it is the default. But it
# also means one scene's weather, date and sensor state define your generalisation score.
#
# `# TODO(you):` choose the scenes you will actually report, and state the criterion. A
# falsifiable one: "the held-out scene's NIR mean differs from the training scenes' by more
# than 0.02, so the split is not radiometrically convenient." "It gave better accuracy" is
# not a criterion — it is selection on the test set, and `gate_no_test_set_selection` fails
# it closed.

# %% [markdown]
# ## Part 3 — The class set is a decision, made on TRAIN counts only
#
# Lab 5.1 in 2025/26 reported `class 0 | support 23 | recall 0.565` to three decimals. A
# recall on n = 23 has a standard error near 0.10; on n = 1 it is literally 0 or 1.
# `gates.MIN_CLASS_SUPPORT = 25` exists because per-class metrics below that are noise with
# a label attached.
#
# There are three responses and only two of them are honest. **Deleting** a row and
# averaging over what remains is what the old code did, and it is the label-set bug from
# Part 9 wearing a different hat. **Merging** collapses under-supported codes into one
# `rare-other` bucket, which you then report. **Dropping** is correct only for a code with
# *zero* training samples: no weight, sampler or loss can predict a label the model never
# saw, and folding such a code into `rare-other` would create a bucket the model cannot
# lose and cannot win.
#
# Every one of these decisions is made from **training** counts only. Deciding the class
# set by looking at test support is test-informed problem definition — the same category of
# error as deriving `num_classes` from `np.unique(y)` over the whole dataset, which the old
# notebook did inside its split cell.

# %%
MERGE_MIN = gates.MIN_CLASS_SUPPORT
RARE_OTHER = 9001  # deliberately not a CORINE code, so it can never collide with one

# The candidate list is the taxonomy present in the archive. What is decided on TRAIN
# counts alone is which codes survive. Deciding the class set by looking at test support
# would be test-informed problem definition.
CODES_ALL = PS.class_codes
counts_all = lab_mod.class_counts(PS.labels, CODES_ALL)
counts_train_pre = lab_mod.class_counts(PS.labels[train_idx0], CODES_ALL)

DROP_CODES = [int(c) for c, n in zip(CODES_ALL, counts_train_pre) if n == 0]
MERGED_CODES = [int(c) for c, n in zip(CODES_ALL, counts_train_pre) if 0 < n < MERGE_MIN]
KEEP_CODES = [int(c) for c, n in zip(CODES_ALL, counts_train_pre) if n >= MERGE_MIN]

print(f"  {'code':>5} {'class':<32} {'all':>7} {'train':>7}  decision")
for i, c in enumerate(CODES_ALL):
    c = int(c)
    dec = ("DROP (unlearnable)" if c in DROP_CODES else
           f"merge -> {RARE_OTHER}" if c in MERGED_CODES else "keep")
    print(f"  {c:>5} {lab_mod.class_name(c)[:32]:<32} {counts_all[i]:>7} "
          f"{counts_train_pre[i]:>7}  {dec}")
print(f"\n  threshold: {MERGE_MIN} training samples")
print(f"  kept {len(KEEP_CODES)}: {KEEP_CODES}")
print(f"  merged {len(MERGED_CODES)}: {MERGED_CODES}")
print(f"  dropped {len(DROP_CODES)}: {DROP_CODES}")
if not KEEP_CODES:
    raise RuntimeError(
        f"no class reaches {MERGE_MIN} training samples. Your dataset is too small to "
        "supervise at all -- go back to Lab 4.2 and extract more scenes; do not lower the "
        "threshold until something passes."
    )

# %%
# A code with ZERO training samples is not a rare class, it is an unlearnable one: no
# weight, sampler or loss can predict a label the model never saw. Those patches are
# dropped, not merged -- folding them into rare-other would give that bucket a label the
# training set never contained, which is a class the model cannot lose and cannot win.
drop_mask = np.isin(PS.labels, DROP_CODES)
keep_idx = np.flatnonzero(~drop_mask)
LABEL_CODES = PS.labels.copy()
for c in MERGED_CODES:
    LABEL_CODES[PS.labels == c] = RARE_OTHER
N_BEFORE = len(PS)
PS = PS.subset(keep_idx)          # everything downstream uses the filtered set
LABEL_CODES = LABEL_CODES[keep_idx]
print(f"  dropped {int(drop_mask.sum())} patches whose label has no training support")
print(f"  relabelled {int(np.isin(PS.labels, MERGED_CODES).sum())} patches into "
      f"rare-other {RARE_OTHER}")
print(f"  working set now: {len(PS)} patches (was {N_BEFORE}), "
      f"{len(np.unique(PS.scene))} scenes, labels {sorted(set(int(c) for c in LABEL_CODES))}")
print(f"  note: {RARE_OTHER} is not a CORINE code, so tables will show it as")
print(f"  'UNKNOWN CORINE {RARE_OTHER}'. That is correct and worth saying in your report.")

# %%
# Rebuild the split on the filtered set, so the manifest of record describes the data you
# actually train on. Same seed, same held-out scenes, fewer patches.
manifest = splits.group_block_split(PS.scene, PS.row, PS.col, block=10,
                                    val_ratio=0.15, test_scenes=TEST_SCENES, seed=SEED)
sm = manifest.summary()
train_idx, val_idx, test_idx = manifest.train_idx, manifest.val_idx, manifest.test_idx
paths.ensure(paths.splits_dir())
MANIFEST_PATH = manifest.write(paths.splits_dir() / f"split_{manifest.manifest_hash}.json")
gates.gate_split_is_grouped(manifest)
gates.gate_split_disjoint(manifest)
print(f"  manifest of record: {manifest.manifest_hash}  ({MANIFEST_PATH.name})")
print(f"  n train/val/test  : {sm['n_train']} / {sm['n_val']} / {sm['n_test']} "
      f"of {sm['n_total']}  (was {N_BEFORE} patches before the filter)")
print(f"  test scenes       : {sm['test_scenes']}   val scenes: {sm['val_scenes']}")
print("  This hash goes into every results.json record. It is what makes 'the split I")
print("  reported == the split I used' checkable. 2025/26 had no split record at all.")

# %%
CODES_USED = KEEP_CODES + ([RARE_OTHER] if MERGED_CODES else [])
counts_train_pre = lab_mod.class_counts(LABEL_CODES[train_idx], CODES_USED)
print(f"  class set used from here on ({len(CODES_USED)}): {CODES_USED}")
print(f"  imbalance ratio over classes present in TRAIN = "
      f"{lab_mod.imbalance_ratio(counts_train_pre):.1f}x")
print("  Both decisions -- what was merged and what was dropped -- go into results.json.")
print("  Merging changes the macro-average, so Part 12 measures by how much, and the number")
print("  gets reported rather than absorbed.")

# %% [markdown]
# ## Part 4 — Negative control: the split that makes the number look better
#
# `splits.stratified_random_split` is in the package for one reason: so you can run it,
# watch the metric rise, and write down why you do not believe it. It is the exact split
# Labs 1–5 of the 2025/26 edition used and reported as a result.
#
# Stratification is not the fix. Equalising class frequencies across splits changes the
# *class prior*; it does nothing about *spatial adjacency*, which is where the leak lives.

# %%
manifest_leaky = splits.stratified_random_split(LABEL_CODES, train_ratio=0.7, val_ratio=0.15,
                                                seed=SEED, min_per_class=2)
sl = manifest_leaky.summary()
print(f"  method          : {sl['method']}   hash {manifest_leaky.manifest_hash}")
print(f"  n train/val/test: {sl['n_train']} / {sl['n_val']} / {sl['n_test']}")
print(f"  scenes in train : {sl['n_scenes_train']}  (None: no group variable was recorded)")
print(f"  extra           : {manifest_leaky.extra}")
print("  Now try the same call on the RAW labels, before Part 3 filtered anything:")
try:
    splits.stratified_random_split(PS.labels, train_ratio=0.7, val_ratio=0.15, seed=SEED,
                                   min_per_class=2)
    print("    it succeeded: every class in this archive has enough support to be split")
except splits.SplitLeakError as exc:
    print(f"    it raises:\n      {exc}")
print("  So the merge was not only a reporting decision: without it, this split could not")
print("  be constructed at all. The grouped split never had that problem, because it")
print("  assigns whole scenes and never asks a rare class to supply three subsets.")

# %%
try:
    gates.gate_split_is_grouped(manifest_leaky)
    print("  UNREACHABLE: the gate accepted a per-patch split")
except gates.GateFailure as exc:
    print(f"  gate_split_is_grouped raises on it:\n    {exc}")
print("  GateFailure extends AssertionError, so this is not a warning you can scroll past.")
print("  Part 11 runs the same model on this split and compares.")

# %% [markdown]
# ## Part 5 — Normalisation fitted on TRAIN ONLY
#
# In the 2025/26 edition, Lab 4.2 computed percentile bounds over the **whole raster**,
# which later became train + val + test. Every test patch therefore contributed to the
# transform applied to every training patch. The reported 0.6992 accuracy is partly an
# interpolation score, and no experiment in that notebook could have detected it, because
# the leak happened upstream of the split.
#
# Two further defects sat inside that same function: the bounds were **pooled across all
# four bands** (one shared `[low, high]` for Blue, Green, Red and NIR — NIR is far brighter
# than Blue over vegetation, so a pooled stretch squeezes Blue toward a constant and
# destroys the band that most separates urban fabric from vegetation), and each tile was
# normalised **independently** before concatenation, so four tiles meant four incompatible
# radiometric scales reported as one dataset.
#
# `radiometry.Norm` makes the fit/apply separation structural: `fit` takes only the train
# array, and `apply` refuses to run on statistics that were never fitted.

# %%
norm = radiometry.Norm.fit(PS.patches[train_idx], mode="percentile", channel_axis=-1,
                           low_pct=2.0, high_pct=98.0)
print(f"  {norm}")
print(f"  fitted on {norm.n_samples_seen} patches (train only; n_total = {len(PS)})")
print(f"  per-band low  {np.round(norm.low_, 4).tolist()}")
print(f"  per-band high {np.round(norm.high_, 4).tolist()}")

# %%
x_all = norm.apply(PS.patches)
# Archives store channels-last (rasterio order); PyTorch wants channels-first. Lab 5.2's
# transformer did `x.view(B, seq_len, C)` on a CHW tensor -- a view, not a permute -- which
# scrambled band and position semantics without ever raising. Transpose once, here.
x_all = np.ascontiguousarray(np.transpose(x_all, (0, 3, 1, 2)))
# Built from CODES_USED, which Part 3 derived from TRAIN counts only. LabelMap.from_train
# cannot be used here: it runs clean_labels(), which keeps only real CORINE codes, and
# rare-other is deliberately not one. The assertion below is what replaces from_train's
# guarantee -- every scored class must have training support.
LM = lab_mod.LabelMap(tuple(CODES_USED))
_no_support = [c for c in LM.codes
               if int((LABEL_CODES[train_idx] == c).sum()) == 0]
if _no_support:
    raise RuntimeError(
        f"class(es) {_no_support} are in the scored set but have zero training samples. "
        "Part 3 should have dropped them. Fix the filter; do not let the model be scored "
        "on a class it never saw."
    )
y_all = LM.encode(LABEL_CODES)
LABELS = np.arange(LM.n_classes)
IN_CHANNELS = x_all.shape[1]
print(f"  LabelMap from TRAIN: {LM}")
print(f"  x_all {x_all.shape} range [{x_all.min():.4f}, {x_all.max():.4f}] dtype {x_all.dtype}")
gates.gate_input_units(x_all, "normalised patches")
print("  gate_input_units passed: the array is in the units the model expects")

# %%
# The 2025/26 double-normalisation, in one line of history. Lab 4.2 had already stretched
# the raster to [0, 1] and clipped it; Lab 5.1 then multiplied by 1e-4 anyway, so the
# network saw values of order 1e-5. BatchNorm rescaled it and nothing crashed, which is
# exactly why the bug survived a full academic year.
print(f"  what the old notebook fed the network: max {float(x_all.max() * 1e-4):.2e}")
print("  BatchNorm would have absorbed that silently. gate_input_units does not.")

# %%
norm_leaky = radiometry.Norm.fit(PS.patches, mode="percentile", channel_axis=-1,
                                 low_pct=2.0, high_pct=98.0)
a = norm.apply(PS.patches[test_idx])
b = norm_leaky.apply(PS.patches[test_idx])
delta = np.abs(a - b)
print(f"  train-only fit  saw {norm.n_samples_seen} patches")
print(f"  whole-raster fit saw {norm_leaky.n_samples_seen} patches, including "
      f"{len(test_idx)} test patches")
print(f"  held-out pixels differ by up to {delta.max():.4f}, mean {delta.mean():.4f}")
print(f"  pixels moved by more than 0.05: {100 * (delta > 0.05).mean():.2f}%")
print("  Same held-out array both times. The only thing that changed is who was allowed to")
print("  vote on the transform.")

# %%
try:
    radiometry.Norm(mode="percentile").apply(PS.patches[:8])
    print("  UNREACHABLE: an unfitted Norm applied itself")
except RuntimeError as exc:
    print(f"  an unfitted Norm refuses:\n    {exc}")

# %% [markdown]
# ## Part 6 — Baselines, before any neural network exists
#
# The 2025/26 labs printed a majority-class *count*, called it a baseline, and never scored
# it with the same code as the model. Lab 6 then shipped a fine-tuned Prithvi with test
# accuracy `0.10000000149011612` — exactly 1/10, chance for 10 classes — while the
# majority-class baseline on that same 455-sample test set was 245/455 = **0.538**. The
# model was worse than predicting one label, and nothing in the notebook could have caught
# it.
#
# `baselines.run_all` scores six trivial predictors through `eo_course.metrics.evaluate`,
# on the **same test split** with the **same label set** as your model. That sameness is
# the whole point.
#
# | baseline | what it is | what it exposes |
# |---|---|---|
# | `majority` | always the most frequent training class | the accuracy floor of a constant predictor |
# | `uniform_random` | uniform guess | balanced-accuracy floor = 1/K |
# | `prior_random` | samples from the training class prior | what memorising the prior alone buys you |
# | `per_scene_majority` | most frequent class *within each scene* | how much a per-site constant gets you |
# | `ndvi` | two NDVI thresholds mapped onto the class set | a free, interpretable physics rule |
# | `linear_probe` | multinomial logistic regression on flattened patches | how much of the task is linear in reflectance |

# %% [markdown]
# ### Exercise 2 — predict the floor
#
# > **P2.** Before running the baseline table, write down:
# >
# > 1. The **balanced accuracy** of `uniform_random` on your split, to two decimals.
# >    (Hint: it is a function of K alone.)
# > 2. The **overall accuracy** of `majority`, to two decimals. (Hint: it is the majority
# >    class share of the *test* split.)
# > 3. Which of the six baselines will have the **highest macro-F1**.
# >
# > Then run the cell and record predicted vs actual. Question 3 is the one that matters:
# > your model must beat the *winner*, not the easiest one.

# %%
P2 = {
    # TODO(you): fill in BEFORE running the next cell.
    "uniform_random_bal_acc": None,
    "majority_overall_acc": None,
    "best_macro_f1_baseline": None,  # one of the six names
    "why": "TODO(you): why is that one likely to win on this data?",
}

# %%
BASELINES = bl.run_all(
    x_train=x_all[train_idx], y_train_codes=LABEL_CODES[train_idx],
    x_test=x_all[test_idx], y_test_codes=LABEL_CODES[test_idx],
    labels=LABELS, codes=LM.codes,
    train_scene_ids=PS.scene[train_idx], test_scene_ids=PS.scene[test_idx],
    seed=SEED,
)
BEST_BASE = max(BASELINES, key=lambda k: BASELINES[k]["macro_f1"])
print(f"  {'baseline':<20} {'acc':>8} {'bal_acc':>8} {'macro_F1':>9} {'kappa':>8}")
for name in sorted(BASELINES, key=lambda k: -BASELINES[k]["macro_f1"]):
    d = BASELINES[name]
    print(f"  {name:<20} {d['overall_acc']:>8.4f} {d['balanced_acc']:>8.4f} "
          f"{d['macro_f1']:>9.4f} {d['kappa']:>8.4f}")
print(f"\n  CHANCE FLOOR      : balanced_acc = 1/K = {1.0 / LM.n_classes:.4f}")
print(f"  STRONGEST BASELINE: {BEST_BASE}  macro-F1 "
      f"{BASELINES[BEST_BASE]['macro_f1']:.4f}, "
      f"balanced_acc {BASELINES[BEST_BASE]['balanced_acc']:.4f}")
print(f"  YOUR TARGETS      : macro-F1 > "
      f"{BASELINES[BEST_BASE]['macro_f1'] + gates.MIN_MACRO_F1_OVER_BASELINE:.4f}"
      f"   balanced_acc > "
      f"{BASELINES[BEST_BASE]['balanced_acc'] + gates.MIN_BALANCED_ACC_OVER_CHANCE:.4f}")
print(f"  P2 answers were   : {P2}")

# %%
maj = BASELINES["majority"]
print(f"  majority OVERALL accuracy   {maj['overall_acc']:.4f}  <- compare your accuracy to this")
print(f"  majority BALANCED accuracy  {maj['balanced_acc']:.4f}  = 1/K for a constant predictor")
print("  In 2025/26 the line `print(f'Model improvement: {balanced_acc - majority_baseline}')`")
print("  subtracted a majority-class accuracy from a balanced accuracy. Those are different")
print("  quantities and their difference is not an improvement.")

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
names = sorted(BASELINES, key=lambda k: BASELINES[k]["macro_f1"])
ax.barh(names, [BASELINES[n]["macro_f1"] for n in names],
        color=["#c0392b" if n == BEST_BASE else "#7f8c8d" for n in names], label="macro-F1")
ax.barh(names, [BASELINES[n]["balanced_acc"] for n in names],
        color="#2980b9", alpha=0.6, label="balanced accuracy")
ax.axvline(1.0 / LM.n_classes, ls="--", c="k", lw=1, label=f"1/K = {1.0 / LM.n_classes:.3f}")
ax.set_xlabel("score"); ax.legend(fontsize=8); ax.tick_params(labelsize=8)
ax.set_title("baselines on the held-out test split — the bar your model must clear")
fig.tight_layout()
FIG1 = paths.ensure(paths.results_dir())[0] / "lab5_1_baselines.png"
fig.savefig(FIG1, dpi=110)
plt.show()
print(f"  saved {FIG1}")

# %% [markdown]
# ## Part 7 — Imbalance as a decision, not a default
#
# `scripts/train_cnn.py` defines five arms in `ARMS`. Each differs from `none` in **exactly
# one thing**, which is what makes a measured difference attributable to that thing rather
# than to a coincidence of three changes at once.
#
# | arm | what changes | mechanism |
# |---|---|---|
# | `none` | — | plain cross-entropy |
# | `weights_sqrt` | loss weights ∝ 1/√n | gentle reweighting; keeps some majority gradient |
# | `weights_inv` | loss weights ∝ 1/n | aggressive; capped at `clamp=50` so the optimiser does not spend the epoch chasing single samples |
# | `sampler` | `WeightedRandomSampler` ∝ 1/n | changes *which* samples you see, not the loss; an epoch no longer covers the data once |
# | `focal` | focal loss, γ = 2 | down-weights *easy* examples; a different axis entirely |
#
# Weights are normalised to mean 1 in every arm. Without that, the weighted arms have a
# loss orders of magnitude different from `none`, and the training curves the lab asks you
# to compare are not comparable.

# %%
counts_train = lab_mod.class_counts(LABEL_CODES[train_idx], LM.codes)
from eo_course.training import class_weights  # noqa: E402

w_sqrt = class_weights(counts_train, "sqrt_inverse_freq").numpy()
w_inv = class_weights(counts_train, "inverse_freq").numpy()
print(f"  K = {LM.n_classes} classes in TRAIN (after the Part 3 merge)")
print(f"  {'code':>5} {'class':<34} {'n_train':>8} {'share':>7} {'w_sqrt':>8} {'w_inv':>8}")
for i, c in enumerate(LM.codes):
    nm = lab_mod.class_name(c)[:24] + (" [rare-other]" if c == RARE_OTHER else "")
    print(f"  {c:>5} {nm:<34} {counts_train[i]:>8} "
          f"{counts_train[i] / counts_train.sum():>6.1%} {w_sqrt[i]:>8.3f} {w_inv[i]:>8.3f}")
IMBALANCE = lab_mod.imbalance_ratio(counts_train)
print(f"  imbalance ratio (max/min over present classes) = {IMBALANCE:.1f}x")
print("  The 2025/26 summary cell asserted '10x more frequent than rare classes' while its")
print("  own cell 9 printed 396.0x. Quote the number you measured, not the one you expect.")

# %% [markdown]
# ### Exercise 3 — predict the ordering
#
# > **P3.** Before running any arm, write down an **ordering** of the five arms by test
# > macro-F1 on your split, best first. Then write the one sentence you would give a
# > reviewer for why your chosen arm beats `none` on *this* data — the sentence has to
# > mention the imbalance ratio you just printed, not just "imbalance hurts".
# >
# > You will not run all five here; the full grid is the `slurm/submit_sweep.sbatch`
# > assignment. You will run your chosen arm three times, and the gate board judges whether
# > the margin exceeds your seed spread.

# %%
ARM = "weights_sqrt"  # TODO(you): none / weights_sqrt / weights_inv / sampler / focal
ARMS = {
    "none": {"imbalance": None, "note": "plain cross-entropy, no rebalancing"},
    "weights_sqrt": {"imbalance": "sqrt_inverse_freq", "note": "class weights ~ 1/sqrt(n)"},
    "weights_inv": {"imbalance": "inverse_freq", "note": "class weights ~ 1/n"},
    "sampler": {"imbalance": "weighted_sampler", "note": "WeightedRandomSampler, ~1/n"},
    "focal": {"imbalance": "focal", "note": "focal loss, gamma=2"},
}
assert ARM in ARMS, f"ARM must be one of {sorted(ARMS)}"
REJECTED = {a: ("chosen" if a == ARM else "TODO(you): why not") for a in ARMS}
P3 = {"ordering": None, "why": "TODO(you): name the mechanism, citing the imbalance ratio."}
print(f"  chosen arm : {ARM}  ({ARMS[ARM]['note']})")
print(f"  imbalance  : {ARMS[ARM]['imbalance']!r}")
print("  REJECTED is written into results.json, which is what makes the decision gradeable")
print("  rather than asserted. Fill every TODO in it before you submit.")

# %% [markdown]
# ## Part 8 — Batch size is not a free parameter at this data volume
#
# Lab 5.1 in 2025/26 used `batch_size = 256`. Against a training pool of a few hundred
# patches that is **three gradient steps per epoch**, and the reported model predicted a
# single class. Batch size is not a style choice when the epoch is that short: the
# optimiser barely visits the data before the schedule has moved.
#
# Do the arithmetic on your own split before you pick.

# %%
n_train = len(train_idx)
print(f"  n_train = {n_train}")
print(f"  {'batch':>6} {'steps/epoch':>12} {'steps in 30 epochs':>19}")
for bs in (256, 128, 64, 32, 16):
    print(f"  {bs:>6} {math.ceil(n_train / bs):>12} {math.ceil(n_train / bs) * 30:>19}")
BATCH = 32  # TODO(you): justify against the row above, not against habit
EPOCHS = 30
STEPS = math.ceil(n_train / BATCH)
print(f"\n  using batch {BATCH}: {STEPS} steps/epoch, {STEPS * EPOCHS} total steps")
print("  256 is defensible at 160k patches and indefensible here.")

# %% [markdown]
# ## Part 9 — Train one arm, evaluate it once, on test
#
# The evaluation contract, stated before the code:
#
# * `pl.seed_everything(seed, workers=True)` before construction, `deterministic="warn"`.
# * `ModelCheckpoint(monitor="val_macro_f1", mode="max")` — an imbalance-aware metric, so
#   the checkpoint can select for rare-class performance. The old notebook logged only
#   `val_loss` and `val_acc`, so no checkpoint could ever have done this.
# * `EarlyStopping(monitor="val_macro_f1", mode="max")`. The 2025/26 callback was
#   `EarlyStopping(monitor="val_acc", mode="min")`, which stops at the **worst** accuracy.
# * The cosine schedule steps per **epoch**. The old code returned a cosine scheduler whose
#   `T_max` was in epochs with no interval, and Lightning's default is `"step"`: at 625
#   steps/epoch against `T_max=100` the LR completed ~625 full cosine cycles instead of
#   annealing once. `training.CorineModule.configure_optimizers` sets `interval="epoch"`
#   explicitly and says so in a comment.
# * Metrics computed **once** from the full arrays, over a **fixed** label set, on **test**,
#   from the **best** checkpoint. Every one of those four words was violated in 2025/26.

# %%
import lightning as pl  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from eo_course.training import CorineDataModule, CorineModule  # noqa: E402


def collect_predictions(ckpt_path, dm, split="test"):
    """Evaluate the BEST checkpoint, not the in-memory last-epoch weights."""
    module = CorineModule.load_from_checkpoint(str(ckpt_path), train_counts=dm.train_counts)
    module.eval()
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(module(xb).argmax(1).cpu())
            targets.append(yb.cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


print(f"  lightning {pl.__version__}; checkpoint monitor = val_macro_f1 (mode=max)")

# %%
def run_arm(arm, seed, man, x=None, epochs=EPOCHS, batch=BATCH, tag=""):
    """One arm, one seed, one held-out evaluation. Mirrors scripts/train_cnn.py."""
    pl.seed_everything(seed, workers=True)
    imb = ARMS[arm]["imbalance"]
    run_id = f"{arm}_{seed}{tag}"
    dm = CorineDataModule(x_all if x is None else x, y_all, man, batch_size=batch,
                          imbalance=imb, num_workers=NUM_WORKERS, seed=seed)
    dm.setup("fit")
    module = CorineModule(n_classes=LM.n_classes, in_channels=IN_CHANNELS, imbalance=imb,
                          train_counts=dm.train_counts, max_epochs=epochs)
    ckpt_dir = paths.ensure(paths.run_dir(run_id) / "ckpt")[0]
    best = ModelCheckpoint(dirpath=str(ckpt_dir), monitor="val_macro_f1", mode="max",
                           save_top_k=1, save_last=True, filename="best-{epoch}-{val_macro_f1:.4f}")
    stop = EarlyStopping(monitor="val_macro_f1", mode="max", patience=8)
    trainer = pl.Trainer(max_epochs=epochs, accelerator=ACCELERATOR, devices=1,
                         deterministic="warn", logger=False, enable_progress_bar=False,
                         default_root_dir=str(paths.run_dir(run_id)), callbacks=[best, stop])
    trainer.fit(module, datamodule=dm)
    dm.setup("test")
    preds, targets = collect_predictions(best.best_model_path, dm)
    return {"run_id": run_id, "arm": arm, "seed": seed, "preds": preds, "targets": targets,
            "ckpt": best.best_model_path, "best_val_macro_f1": best.best_model_score}


print("  run_arm defined. It seeds, takes its split from the manifest, monitors val_macro_f1")
print("  and evaluates the best checkpoint on test exactly once.")

# %%
res = run_arm(ARM, SEED, manifest)
test_m = metrics.evaluate(res["targets"], res["preds"], LABELS, codes=LM.codes, names=LM.names,
                          groups=PS.scene[test_idx], n_boot=1000, seed=SEED)
print(test_m.table(min_support=gates.MIN_CLASS_SUPPORT))
print(f"  best val macro-F1 was {float(res['best_val_macro_f1']):.4f}")
if test_m.ci:
    print(f"  scene-clustered 95% CI  balanced_acc "
          f"{np.round(test_m.ci['balanced_acc'], 4).tolist()}  "
          f"macro-F1 {np.round(test_m.ci['macro_f1'], 4).tolist()} "
          f"over {test_m.ci['n_clusters']} clusters")

# %%
print("  Top confusions — 'class X is being called Y', with counts:")
for true_c, pred_c, n in metrics.top_confusions(test_m, k=8):
    print(f"    {true_c:<38} -> {pred_c:<38} {n:>5}")
print("\n  Read this before you write anything. Which class absorbs your worst class, and is")
print("  that a spectral confusion (two classes that look alike) or a geometric one (a")
print("  boundary class whose 100 m label is a coin flip)?")

# %% [markdown]
# ### The label-set bug, demonstrated on your own predictions
#
# The old diagnostics cell did this:
#
# ```python
# labels = np.unique(np.concatenate([y_true, y_pred]))
# ...
# balanced_acc = recall.mean()
# ```
#
# A class that appears in **neither** `y_true` nor `y_pred` drops out of `labels`, so it
# never enters the confusion matrix and `recall.mean()` averages over **fewer classes than
# exist** — and comes out higher. The denominator shrinks and nobody prints the
# denominator. Note the precise trigger: a class the model never predicts but that *is*
# present in `y_true` survives this, because the union includes `y_true`. What kills a
# class is having no support in the slice being scored, which is the normal state of the
# rare classes in this dataset. `metrics.evaluate` takes the label set as an argument and
# refuses arrays containing classes outside it.

# %%
def derived_vs_fixed(y_true, y_pred, fixed_labels):
    """Score the same predictions three ways and return all three pairs of numbers.

    ``derived``  the 2025/26 code: labels = np.unique(concat(y_true, y_pred)) then
                 recall.mean(). K becomes whatever the slice happens to contain.
    ``all_K``    average over every class in the fixed set, absent ones counted as 0.
    ``present``  what eo_course.metrics.evaluate does: average over classes with
                 support > 0 -- but it prints the support per class, and
                 gate_class_support refuses the under-supported regime outright.
    """
    lab = np.unique(np.concatenate([y_true, y_pred]))
    cm = metrics.confusion(y_true, y_pred, lab)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    cm_k = metrics.confusion(y_true, y_pred, fixed_labels)
    tp_k = np.diag(cm_k).astype(float)
    rec_k = tp_k / np.maximum(cm_k.sum(1), 1)
    prec_k = tp_k / np.maximum(cm_k.sum(0), 1)
    f1_k = 2 * prec_k * rec_k / np.maximum(prec_k + rec_k, 1e-12)
    ev = metrics.evaluate(y_true, y_pred, fixed_labels, n_boot=0)
    return {"k_derived": int(lab.size), "k_fixed": int(fixed_labels.size),
            "derived": (float(rec.mean()), float(f1.mean())),
            "all_K": (float(rec_k.mean()), float(f1_k.mean())),
            "present": (ev.balanced_acc, ev.macro_f1),
            "zero_support": int((cm_k.sum(1) == 0).sum())}


# %%
def show(tag, d):
    print(f"  {tag}")
    print(f"    K derived {d['k_derived']}  vs  K fixed {d['k_fixed']}  "
          f"(classes with zero test support: {d['zero_support']})")
    print(f"    balanced_acc  derived {d['derived'][0]:.4f}   all-K {d['all_K'][0]:.4f}"
          f"   present-only {d['present'][0]:.4f}")
    print(f"    macro-F1      derived {d['derived'][1]:.4f}   all-K {d['all_K'][1]:.4f}"
          f"   present-only {d['present'][1]:.4f}")
    print(f"    derived minus all-K: bal {d['derived'][0] - d['all_K'][0]:+.4f}, "
          f"F1 {d['derived'][1] - d['all_K'][1]:+.4f}")


# A slice with one class missing entirely, which is the normal state of the rare classes
# in this dataset. A collapsed predictor is NOT the case where this bites: a constant
# model still leaves every class present in y_true, so the derived set is unchanged.
syn_true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
syn_pred = np.array([0, 0, 1, 1, 0, 2, 2, 2])
show(f"constructed slice, K = {len(LABELS)}, class {int(LABELS[-1])} absent:",
     derived_vs_fixed(syn_true, syn_pred, LABELS))

show("your model's actual test predictions:",
     derived_vs_fixed(res["targets"], res["preds"], LABELS))
print("  Read the left column against the middle one. The old code's number is not")
print("  'slightly off': it silently changes K, and the more classes are missing, the")
print("  more it flatters you. eo_course.metrics.evaluate averages over classes with")
print("  support > 0 too, but it prints the support per class and gate_class_support")
print("  refuses to let you be scored at all below 25. The fix is a visible denominator")
print("  plus a gate, not a different formula.")

# %% [markdown]
# > **P4.** Look at the `derived minus all-K` line for the constructed slice and answer in
# > two sentences. (a) The derived label set is *smaller* than the fixed one. Which
# > direction does that bias balanced accuracy, and why is the bias largest on exactly the
# > rare classes you care about? (b) On your own predictions the two columns may be close
# > or identical. What does that tell you about *this run*, and what does it tell you about
# > the bug? The second question is the one that separates reading from running.

# %%
P4 = {"direction": "TODO(you): sign of the derived-vs-all-K bias, and why",
      "own_run": "TODO(you): what your own gap does and does not tell you"}

# %% [markdown]
# ## Part 10 — Three seeds, or it is not a result
#
# `gates.MIN_SEEDS = 3`. Lab 5.2's headline `Balanced accuracy 0.8338` came from 455
# validation samples where five of ten classes had support ≤ 9 and two had support 1. Two
# misclassifications move that number by about 0.05 — larger than every difference the lab
# asked students to compare. A single run cannot resolve them.
#
# `gates.MIN_SEED_MULTIPLIER = 2.0`: an improvement must exceed **2 × the seed spread** to
# be written down as an improvement. Below that, the honest sentence is "no detectable
# effect at n = 3".

# %%
SEEDS = [0, 1, 2]
SEED_RUNS = [{"run_id": res["run_id"], "seed": SEED, "arm": ARM, "test": test_m.to_dict(),
              "preds": res["preds"], "targets": res["targets"], "ckpt": res["ckpt"]}]
for s in SEEDS[1:]:
    r = run_arm(ARM, s, manifest)
    m = metrics.evaluate(r["targets"], r["preds"], LABELS, codes=LM.codes, names=LM.names,
                         groups=PS.scene[test_idx], n_boot=1000, seed=s)
    SEED_RUNS.append({"run_id": r["run_id"], "seed": s, "arm": ARM, "test": m.to_dict(),
                      "preds": r["preds"], "targets": r["targets"], "ckpt": r["ckpt"]})
    print(f"  seed {s}: acc={m.overall_acc:.4f} bal_acc={m.balanced_acc:.4f} "
          f"macro_F1={m.macro_f1:.4f}")

# %%
var = gates.gate_seeds_and_variance(SEED_RUNS, metric="balanced_acc")
var_f1 = gates.gate_seeds_and_variance(SEED_RUNS, metric="macro_f1")
bal = np.array([r["test"]["balanced_acc"] for r in SEED_RUNS])
f1 = np.array([r["test"]["macro_f1"] for r in SEED_RUNS])
print(f"  {ARM} over {var['n']} seeds")
print(f"    balanced_acc  mean {var['mean']:.4f}  std {var['std']:.4f}  "
      f"range [{bal.min():.4f}, {bal.max():.4f}]")
print(f"    macro-F1      mean {var_f1['mean']:.4f}  std {var_f1['std']:.4f}  "
      f"range [{f1.min():.4f}, {f1.max():.4f}]")
print(f"    claim threshold: an improvement must exceed {var['threshold']:.4f} "
      f"({gates.MIN_SEED_MULTIPLIER} x std) to be written as an improvement")

# %%
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot([r["seed"] for r in SEED_RUNS], f1, "o-", label=f"{ARM} macro-F1 per seed")
ax.axhline(var_f1["mean"], c="#2c3e50", lw=1, ls=":", label=f"mean {var_f1['mean']:.3f}")
ax.axhline(BASELINES[BEST_BASE]["macro_f1"], c="#c0392b", lw=1.5,
           label=f"best baseline ({BEST_BASE}) {BASELINES[BEST_BASE]['macro_f1']:.3f}")
ax.axhline(BASELINES[BEST_BASE]["macro_f1"] + gates.MIN_MACRO_F1_OVER_BASELINE, c="#c0392b",
           lw=1, ls="--", label=f"+{gates.MIN_MACRO_F1_OVER_BASELINE} gate threshold")
ax.set_xlabel("seed"); ax.set_ylabel("test macro-F1"); ax.legend(fontsize=8)
ax.set_title("is the seed spread smaller than the margin over the baseline?")
fig.tight_layout()
FIG2 = paths.ensure(paths.results_dir())[0] / "lab5_1_seed_spread.png"
fig.savefig(FIG2, dpi=110)
plt.show()
print(f"  saved {FIG2}")

# %%
delta_f1 = test_m.macro_f1 - BASELINES[BEST_BASE]["macro_f1"]
print(f"  seed {SEED} macro-F1 {test_m.macro_f1:.4f} - best baseline "
      f"{BASELINES[BEST_BASE]['macro_f1']:.4f} = {delta_f1:+.4f}")
try:
    gates.gate_claim_supported(delta_f1, SEED_RUNS, metric="macro_f1",
                               label=f"{ARM} vs {BEST_BASE}")
    print("  gate_claim_supported: the margin exceeds 2 x seed spread")
except gates.GateFailure as exc:
    print(f"  gate_claim_supported raises:\n    {exc}")
    print("  That is a legitimate finding. Report it as 'no detectable effect at n=3', not")
    print("  as an improvement.")

# %% [markdown]
# ## Part 11 — Negative control: the same model on the leaky split
#
# Same arm, same seed, same epochs, same callbacks — and the same normalisation
# *procedure*: refitted on the leaky split's own training indices, so the split is the
# only variable. Fitting one norm on everything and the other on train would confound two
# leaks at once, and this notebook's whole theme is one-variable-at-a-time attribution.
#
# Do not predict the sign of the difference. Neighbour leakage usually inflates the leaky
# number, but stratification also equalises the class prior, and on a small test split
# those two effects can point in opposite directions. What is fixed, and what you are being
# asked to report, is that **the leaky number is not a generalisation estimate either
# way** — a higher one is inflated and a lower one is not evidence of anything about
# generalisation, because the two numbers are not measuring the same task.

# %%
norm_leaky_split = radiometry.Norm.fit(PS.patches[manifest_leaky.train_idx],
                                       mode="percentile", channel_axis=-1,
                                       low_pct=2.0, high_pct=98.0)
x_all_leaky = np.ascontiguousarray(
    np.transpose(norm_leaky_split.apply(PS.patches), (0, 3, 1, 2)))
res_leaky = run_arm(ARM, SEED, manifest_leaky, x=x_all_leaky, tag="_leakysplit")
leaky_m = metrics.evaluate(res_leaky["targets"], res_leaky["preds"], LABELS, codes=LM.codes,
                           names=LM.names, groups=PS.scene[manifest_leaky.test_idx],
                           n_boot=1000, seed=SEED)
print(f"  group_block split : acc {test_m.overall_acc:.4f}  bal_acc {test_m.balanced_acc:.4f}"
      f"  macro-F1 {test_m.macro_f1:.4f}   (test scenes {sm['test_scenes']}, "
      f"n={test_m.n}, {test_m.ci['n_clusters'] if test_m.ci else '?'} bootstrap clusters)")
print(f"  stratified_random : acc {leaky_m.overall_acc:.4f}  "
      f"bal_acc {leaky_m.balanced_acc:.4f}  macro-F1 {leaky_m.macro_f1:.4f}   "
      f"(no scene held out, n={leaky_m.n})")
print(f"  difference        : acc {leaky_m.overall_acc - test_m.overall_acc:+.4f}  "
      f"bal_acc {leaky_m.balanced_acc - test_m.balanced_acc:+.4f}  "
      f"macro-F1 {leaky_m.macro_f1 - test_m.macro_f1:+.4f}")
print(f"  n_train differs too: {sm['n_train']} grouped vs {sl['n_train']} random — a random "
      "split never has to give a whole scene away, which is part of why it looks cheap.")
try:
    gates.gate_split_is_grouped(manifest_leaky)
except gates.GateFailure:
    print("  The leaky row is not reportable: gate_split_is_grouped refuses it.")

# %% [markdown]
# > **Write this sentence in your report, in your own words, with the numbers above in
# > it:** one sentence explaining why the stratified-random number cannot be read as
# > generalisation, whichever way the difference came out. It must name the mechanism —
# > held-out patches that share image pixels, or sit inside the same 100 m CORINE label
# > pixel, as training patches — and not just say "leakage". A sentence that could have
# > been written without running the cell has not been earned.

# %% [markdown]
# ## Part 12 — What the merge bought, and what it cost
#
# Part 3 merged the under-supported classes on training counts. That decision moves the
# headline number, so the move has to be measured rather than asserted. The *same*
# predictions are scored twice: on the merged class set you trained against, and on the full
# class set where the rare codes are still separate rows that the model can never hit.

# %%
pred_codes = LM.decode(res["preds"])
FULL_CODES = sorted({int(c) for c in KEEP_CODES} | {int(c) for c in MERGED_CODES}
                    | ({RARE_OTHER} if MERGED_CODES else set()))
LM_FULL = lab_mod.LabelMap(tuple(FULL_CODES))
unmerged_m = metrics.evaluate(LM_FULL.encode(PS.labels[test_idx]), LM_FULL.encode(pred_codes),
                              np.arange(LM_FULL.n_classes), codes=LM_FULL.codes,
                              names=LM_FULL.names, groups=PS.scene[test_idx],
                              n_boot=1000, seed=SEED)
print(unmerged_m.table(min_support=gates.MIN_CLASS_SUPPORT))
print(f"  merged class set   : K={LM.n_classes}  macro-F1 {test_m.macro_f1:.4f}  "
      f"balanced_acc {test_m.balanced_acc:.4f}")
print(f"  unmerged class set : K={LM_FULL.n_classes}  macro-F1 {unmerged_m.macro_f1:.4f}  "
      f"balanced_acc {unmerged_m.balanced_acc:.4f}")
print(f"  the merge moved macro-F1 by {test_m.macro_f1 - unmerged_m.macro_f1:+.4f}")

# %% [markdown]
# > State that number in your report, with its sign. Merging usually raises the
# > macro-average because it removes rows the model had no realistic chance at this
# > support — that is a legitimate reason to merge, and it is also exactly why a merged
# > number must never be compared against someone else's unmerged one. If your merge moved
# > the number the other way, say so; it means the model was doing real work on the rare
# > classes and the merge threw that away.

# %% [markdown]
# ## Part 13 — Gate board and deliverable
#
# `results.record_run` writes first, because `run_all_gates` reads the record to check that
# the split hash you reported is the split in use and that you did not select on test. Then
# the board prints.
#
# `results.json` is **append-only** and `record_run` refuses to reuse a `run_id`. Name your
# runs `<arm>_<seed>` so a different configuration gets a different id and an earlier
# attempt stays in the record. Overwriting a previous run destroys the evidence that you did
# the work.

# %%
print(results.describe_contract())

# %%
_probe = CorineModule(n_classes=LM.n_classes, in_channels=IN_CHANNELS,
                      imbalance=ARMS[ARM]["imbalance"], train_counts=counts_train,
                      max_epochs=EPOCHS)
N_PARAMS = int(sum(p.numel() for p in _probe.parameters() if p.requires_grad))
print(f"  trainable parameters: {N_PARAMS:,}")
print("  Lab 5.2 in 2025/26 trained a 122 M-parameter VGG-16 on 8,278 patches and never")
print("  discussed it. Parameter count belongs in the config for exactly that reason.")

CONFIG = {
    "arm": ARM, "imbalance": ARMS[ARM]["imbalance"], "rejected_arms": REJECTED,
    "epochs": EPOCHS, "batch_size": BATCH, "steps_per_epoch": STEPS, "n_train": int(n_train),
    "n_params": N_PARAMS, "lr": 3e-4, "n_classes": LM.n_classes, "in_channels": IN_CHANNELS,
    "codes": list(LM.codes), "bands": list(PS.bands), "patch_size": PS.patch_size,
    "block": sm["block"], "test_scenes": sm["test_scenes"], "val_scenes": sm["val_scenes"],
    "normalisation": norm.to_dict(), "norm_fitted_on": "train only",
    "merge_min_support": MERGE_MIN, "merged_codes": MERGED_CODES,
    "dropped_zero_train_codes": DROP_CODES, "rare_other_code": RARE_OTHER,
    "imbalance_ratio": IMBALANCE, "seed": SEED, "seeds": SEEDS, "accelerator": ACCELERATOR,
    "torch": torch.__version__, "lightning": pl.__version__,
}

# %%
RECORDED = []
for r in SEED_RUNS:
    try:
        rec = results.record_run(
            r["run_id"], lab="lab5.1", config={**CONFIG, "seed": r["seed"]},
            split_manifest_hash=manifest.manifest_hash, seed=r["seed"],
            test_metrics=r["test"], baselines=BASELINES,
            test_used_for_tuning=False, n_seeds=len(SEED_RUNS),
            notes=(f"arm={ARM} ({ARMS[ARM]['note']}); rejected "
                   f"{[k for k, v in REJECTED.items() if v != 'chosen']}; best baseline "
                   f"{BEST_BASE} macro-F1 {BASELINES[BEST_BASE]['macro_f1']:.4f}; seed "
                   f"spread bal_acc {var['std']:.4f}; merged rare codes {MERGED_CODES} into "
                   f"{RARE_OTHER} on train support < {MERGE_MIN}; dropped zero-train "
                   f"codes {DROP_CODES}"),
            extra={"predictions": {"P1": P1, "P2": P2, "P3": P3, "P4": P4},
                   "seed_mean_balanced_acc": var["mean"],
                   "seed_std_balanced_acc": var["std"],
                   "claim_threshold": var["threshold"],
                   "leaky_split_metrics": leaky_m.to_dict(),
                   "leaky_split_hash": manifest_leaky.manifest_hash,
                   "unmerged_class_metrics": unmerged_m.to_dict(),
                   "checkpoint": r["ckpt"]},
        )
        RECORDED.append(rec)
        gates.gate_predictions_saved(
            r["targets"], r["preds"], paths.run_dir(r["run_id"]) / "test_predictions.npz",
            manifest.manifest_hash)
        print(f"  recorded {rec['run_id']}  (predictions saved so every metric is re-derivable)")
    except results.ResultsError as exc:
        print(f"  {r['run_id']} not recorded: {exc}")
print(f"  {len(RECORDED)} of {len(SEED_RUNS)} runs written this session")

# %%
record = RECORDED[0] if RECORDED else results.get_run(f"{ARM}_{SEED}")
lines = gates.run_all_gates(test_m, BASELINES, manifest, record)
lines.append(f"[{'PASS' if var['n'] >= gates.MIN_SEEDS else 'FAIL'}] seed_variance: "
             f"{var['n']} seeds (need {gates.MIN_SEEDS}), std {var['std']:.4f}, "
             f"claim threshold {var['threshold']:.4f}")
ok = gates.print_gate_board(lines)

# %%
print(results.summary_table("lab5.1"))
csv_path = results.write_csv("lab5.1")
print(f"\n  csv for team comparison: {csv_path}")

# %% [markdown]
# ### Reading a failed board
#
# A `FAIL` here is not a formatting problem. It means a conclusion you would have written
# down is not supported, and the message names the number and the threshold.
#
# * **`class_support`** — a scored class has fewer than 25 test samples. Go back to Part 3
#   and raise the merge threshold or add scenes. Do not average over what is left and call
#   it macro.
# * **`beats_baselines`** — the central gate. Before concluding the model is bad, check the
#   two usual causes in order: (1) an input-unit mismatch (`gate_input_units`), and (2) a
#   collapsed head — look at the `pred` column of the metrics table; if one class absorbs
#   nearly everything, the model learned the prior and nothing else.
# * **`above_chance`** — balanced accuracy at or below 1/K. A constant predictor scores this.
#   Overall accuracy can look far higher and mean exactly the same thing.
# * **`not_collapsed`** — more than 85 % of predictions in one class.
# * **`split_hash`** — the record's `split_manifest_hash` is not the current split. You are
#   reporting numbers from a split you no longer have. Retrain or re-report.
# * **`no_test_selection`** — you declared `test_used_for_tuning=True`. Any number chosen by
#   looking at test is a training number with a test label on it.
#
# Reporting a failed gate with a correct diagnosis is a passing submission. Reporting a
# green board obtained by quietly changing the split is not.

# %% [markdown]
# ## Submission checklist
#
# Everything below is a file the grader can open. Self-attestation is not an artifact.
#
# * `results/results.json` — one `lab5.1` record **per seed**, each with
#   `split_manifest_hash`, `seed`, `n_seeds`, `config` (arm, rejected arms, batch size,
#   steps/epoch, `n_params`, the normalisation dict, test scenes, the merge decision),
#   `test` metrics from `metrics.evaluate`, the full `baselines` block, and
#   `test_used_for_tuning: false`.
# * `results/results.csv` — from `results.write_csv("lab5.1")`.
# * `results/lab5_1_baselines.png`, `results/lab5_1_seed_spread.png`.
# * `artifacts/runs/<arm>_<seed>/ckpt/best-*.ckpt` and `test_predictions.npz`.
# * `data/splits/split_<hash>.json` — the manifest whose hash is in your records.
#
# In the write-up:
#
# 1. The baseline table, with the chance floor and the strongest baseline's macro-F1 stated
#    **before** the model's numbers appear.
# 2. Predicted vs actual for **P1, P2 and P3**, plus both answers to **P4**, with the
#    disagreement explained. A prediction that matched still needs the mechanism stated.
# 3. `mean ± spread` over at least three seeds, and "no detectable effect at n = 3" wherever
#    the margin is under `2 × std`.
# 4. The gate board with every `FAIL` diagnosed.
# 5. The negative-control sentence from Part 11, with its numbers in it.
# 6. The arms you did **not** pick and one reason each, as recorded in `config`.
# 7. How the rare-class merge moved the macro-average, and why that is not cheating.
# 8. One claim you cannot make from this experiment, and why.

# %% [markdown]
# ## Where each 2025/26 defect went
#
# | 2025/26 | 2026/27 |
# |---|---|
# | a classmate's username hard-coded as the data path | `paths.require_existing(paths.training_data_dir())`; no username appears anywhere |
# | `glob("*_data.npz")` matched per-scene **and** combined files | `patches.SCENE_GLOB` + `load_all_scenes`; `gate_no_duplicate_patches` on load |
# | `np.load(...).values()` positional read | keyed loading inside `load_scene`; `gate_npz_loaded_by_key` on the archive |
# | `rng.permutation` over edge-adjacent patches sharing a 100 m label | `splits.group_block_split(scene, row, col, block=10)`; `gate_split_is_grouped` |
# | no split record, so reported ≠ used was undetectable | `manifest.manifest_hash` written to disk and into every results record; `gate_split_hash_matches` |
# | `np.unique(y)` over the whole dataset set `num_classes` | the class set is fixed on TRAIN counts only, and an assertion refuses any scored class with zero training support |
# | per-class metrics from n = 1–23 samples | `gates.MIN_CLASS_SUPPORT = 25`; Part 3 merges or drops, Part 12 measures what merging cost |
# | percentile bounds fitted over the whole raster | `radiometry.Norm.fit(x[train_idx])`; Part 5 measures the difference on held-out pixels |
# | `* 0.0001` on already-normalised data | `gate_input_units` on the array actually fed to the model |
# | no baseline ever scored | `baselines.run_all` in Part 6, before training, through the same `metrics.evaluate` |
# | `np.unique(y_pred)`-derived label set | fixed `LABELS = np.arange(K)`; Part 9 demonstrates the difference on your own predictions |
# | metrics averaged over batches, rank-0 shards | `metrics.evaluate` once over the full arrays; `sync_dist=True` inside `CorineModule` |
# | validation numbers reported as results | test evaluated exactly once, from the best checkpoint, recorded under `test` |
# | `batch_size=256` → 3 steps/epoch → collapse | Part 8 prints `ceil(n_train/batch)` for five batch sizes; batch 32 used here |
# | `EarlyStopping(monitor="val_acc", mode="min")` | `EarlyStopping(monitor="val_macro_f1", mode="max")` |
# | cosine scheduler stepped per batch against `T_max` in epochs | `CorineModule.configure_optimizers` returns `interval="epoch"` |
# | zero `seed_everything` calls | `pl.seed_everything(seed, workers=True)` inside `run_arm`, `deterministic="warn"`, one `default_rng` per notebook |
# | single-seed claims | `gates.MIN_SEEDS = 3` via `gate_seeds_and_variance`; `gate_claim_supported` at 2 × std |
# | zero `to_csv` / `json.dump` / `torch.save` | `results.record_run` per seed, `write_csv`, checkpoints and raw predictions on disk |
# | summary cell contradicted the code | the summary is `results.summary_table("lab5.1")`, generated from the records |
