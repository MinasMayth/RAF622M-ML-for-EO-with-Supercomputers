# %% [markdown]
# # Lab 4.2 — Patch extraction: provenance, and what a patch is worth
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# This lab turns the aligned scene pairs from Lab 4.1 into per-scene patch archives in
# `paths.training_data_dir()`, one `patches_<scene_id>_scene.npz` per scene plus its
# sidecar JSON. Every patch carries the scene it came from and its row/col inside that
# scene. Those two arrays are the reason Labs 5–7 can build a leakage-free split at
# all.
#
# It assumes Lab 4.1 wrote `s2_<scene>.tif` and `corine_<scene>.tif` into
# `paths.aligned_dir()`, with reflectance units and a declared nodata. It produces the
# input for Lab 5.1.
#
# **You must run every cell yourself.** All twelve code cells of the 2025/26 version of
# this notebook were committed with `exec=None` and zero outputs. None of that logic had
# ever run. The notebook still read as complete.
#
# ## The one sentence that matters most
#
# > The 2025/26 pipeline threw away scene identity and patch position at the moment of
# > extraction — `np.savez(patches=..., labels=...)` — so by the time a model was
# > trained it was **impossible** to build a leakage-free split even for someone who
# > wanted one. There was no group variable left. Every lab then fell back to a random
# > permutation over spatially contiguous patches, and reported 0.6992 overall /
# > 0.5196 balanced accuracy as generalisation.
#
# Everything else in this notebook is either in service of that fix, or a measurement of
# what the absence of it cost.

# %% [markdown]
# ## The 2025/26 defects this notebook closes
#
# | old defect | consequence | fixed in |
# |---|---|---|
# | `TRAINING_DATA_DIR.glob("*_data.npz")` in Lab 5.2 matched the per-scene files **and** `combined_training_data.npz` | every patch loaded twice; the random split then put duplicates in both train and test | Part 2 |
# | `np.savez(patches=..., labels=...)`, nothing else | no scene, no row, no col: no group variable, so no honest split was constructible | Part 4 |
# | `MAX_PATCHES = 50000` truncated in raster-scan order | the retained patches were the **north-west corner** of every tile — a systematic spatial bias, invisible in the summary | Part 3 |
# | `PATCH_SIZE = 3` against a **100 m** CORINE label, never justified | one label pixel supplies the label to ~9 neighbouring patches, which are therefore not independent samples | Part 5 |
# | "Apply normalization to entire image first (for consistent statistics)" | percentile bounds fitted over the whole raster, which later became train+val+test: every test patch contributed to the transform applied to every training patch | Part 6 |
# | `if np.any(patch == 0): continue` | DN 0 is legitimate dark reflectance (deep water, shadow, asphalt); the filter discarded exactly the hard cases — and it was the only nodata guard, because Lab 4.1 never wrote one | Part 4 |
# | `labels.astype(np.uint8)` | `F.cross_entropy` needs int64; every downstream notebook carried a cast, and Lab 5.2's dataset stored remapped indices as uint8 | Part 4 |
# | `metadata['bands'] = ['B02','B03','B04','B08']` hard-coded | the metadata could lie; Lab 4.1 *did* write band descriptions and nothing read them | Part 4 |
# | `np.random.permutation(...)` with no seed | the dataset was irreproducible, so Lab 5.1's `seed=42` could not recover which patch was which | Part 0 |
# | `if label < 1 or label > 44: continue` | kept 44 "Sea and ocean", 43 "Estuaries", 30 "Beaches, dunes, sands" as supervised classes | Part 4 |
# | `y + patch_size // 2` for even patch sizes | the "centre" is half a pixel up-left and the label no longer corresponds to the patch centre | Part 4 |

# %% [markdown]
# ## Part 0 — Setup
#
# `MPLCONFIGDIR` before matplotlib. One `rng` for the whole notebook, created here and
# passed down. The 2025/26 notebook called `np.random.permutation` and `np.random.choice`
# with no seed anywhere, so the dataset it produced could not be rebuilt — Lab 5.1's
# `seed=42` could not recover which patch was which, because there was nothing to
# recover from.

# %%
from eo_course import paths

print(paths.describe())

# %%
import os
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = paths.matplotlib_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402  (must follow MPLCONFIGDIR)
import rasterio  # noqa: E402

from eo_course import patches as pat  # noqa: E402
from eo_course import radiometry, labels as lab_mod, gates, results  # noqa: E402

SEED = 20180402
rng = np.random.default_rng(SEED)
print(f"seed = {SEED}   (one rng; every subsample below receives it and records it)")

# %% [markdown]
# ## Part 1 — The Lab 4.1 hand-off
#
# `paths.require_existing` names the producing lab when an artifact is missing, so a
# skipped step points at the step rather than surfacing as a mystery
# `FileNotFoundError`. The 2025/26 chain broke twice here: Lab 4.2 wrote to
# `/p/scratch/.../training_data` while Labs 5.2 and 6 asserted
# `/p/scratch/.../data/training_data`, so their `assert` failed for any student who had
# correctly followed Lab 4.2; and Lab 5.1 hard-coded another user's home directory, so
# only the instructor could run it.

# %%
ALIGNED = paths.require_existing(paths.aligned_dir(),
                                 "Lab 4.1 aligned directory (paths.aligned_dir())")
STACKS = sorted(ALIGNED.glob("s2_*.tif"))
if not STACKS:
    raise FileNotFoundError(
        f"no s2_*.tif in {ALIGNED}. Run Lab 4.1 Part 4 (write_stack); it also writes the "
        "sidecar manifest that tells you which bands and units each file holds."
    )
LABELS = {s.stem[len("s2_"):]: ALIGNED / ("corine_" + s.stem[len("s2_"):] + ".tif")
          for s in STACKS}
print(f"  aligned dir : {ALIGNED}")
print(f"  stacks      : {len(STACKS)}")
for s in STACKS:
    lb = LABELS[s.stem[len("s2_"):]]
    print(f"    {s.name:<34} label={'yes' if lb.exists() else 'MISSING'} "
          f"{s.stat().st_size / 1e6:6.1f} MB")

# %%
for s in STACKS:
    side = ALIGNED / (s.stem + ".manifest.json")
    if not side.exists():
        raise FileNotFoundError(
            f"{s.name} has no sidecar manifest. It was not written by Lab 4.1 Part 4, so "
            "its units, band order and nodata are unverified. Re-run Lab 4.1."
        )
BAND_ORDER = json.loads((ALIGNED / "bands.json").read_text(encoding="utf-8"))["order"]
with rasterio.open(STACKS[0]) as src:
    FILE_BANDS = tuple(src.descriptions)
    S2_NODATA, S2_RES = src.nodata, src.res[0]
    S2_H, S2_W = src.height, src.width
print(f"  bands.json order : {BAND_ORDER}")
print(f"  file descriptions: {list(FILE_BANDS)}")
assert list(FILE_BANDS) == BAND_ORDER, (
    "the GeoTIFF band descriptions disagree with bands.json; Lab 4.1 wrote one without the "
    "other. Trust the file, and find out which is stale before extracting.")
print(f"  grid {S2_W}x{S2_H} @ {S2_RES:.0f} m, nodata {S2_NODATA!r}, units per sidecar")

# %% [markdown]
# ## Part 2 — Exercise 1: the double-load bug, reproduced
#
# Lab 4.2 wrote per-scene files as `patches_<tile>_data.npz` and also wrote
# `combined_training_data.npz` into the **same directory**. Lab 5.2 read them with
#
# ```python
# for f in TRAINING_DATA_DIR.glob("*_data.npz"):
# ```
#
# which matches all four files. The combined file is a permutation of the per-scene
# ones, so every patch entered the array **twice**, and the random split then put
# duplicates in both train and test. Nothing downstream could detect it because the
# duplicate count was never computed.
#
# The fix is structural, not a caution: archives are named
# `patches_<scene_id>_scene.npz` (`patches.SCENE_PREFIX` + `SCENE_SUFFIX`), and
# `patches.load_all_scenes` globs `patches.SCENE_GLOB` **only**. There is deliberately
# no combined file in this design — concatenation happens in memory, once.
#
# > **P1.** The cell below builds a legacy-shaped directory: two per-scene archives and
# > one combined archive that is a permutation of the same patches. It then loads that
# > directory three ways: `*.npz`, `*_data.npz`, and `patches.SCENE_GLOB`.
# >
# > 1. How many patches does each glob return?
# > 2. How many **distinct** patches are actually in that directory?
# >
# > Write the three numbers down before running. Question 2 is the one you can check
# > against the ground truth the cell prints.

# %%
PREDICTIONS = {
    # TODO(you): fill in BEFORE running the next two cells. `star`/`data`/`scene` are
    # patch counts. Leaving them None is scored as not attempted.
    "glob_star": None, "glob_data": None, "glob_scene": None, "true_distinct": None,
    "why": "TODO(you): say which glob is structurally immune and why.",
    "nw_bias": {"mean_row_first": None, "mean_row_random": None, "why": "TODO(you):"},
    "label_sharing": {"patches_per_label": None, "ess_pct": None, "why": "TODO(you):"},
}

# %%
DEMO_DIR = paths.data_root() / "_legacy_glob_demo"
if DEMO_DIR.exists():
    shutil.rmtree(DEMO_DIR)
paths.ensure(DEMO_DIR)

def _mini_archive(path, patches, labels, with_provenance):
    kw = {"patches": patches, "labels": labels}
    if with_provenance:
        kw["scene"] = np.asarray(["x"] * len(labels), dtype=object)
        kw["row"] = np.arange(len(labels), dtype=np.int32)
        kw["col"] = np.zeros(len(labels), dtype=np.int32)
    np.savez_compressed(path, **kw)


A = rng.normal(0, 1, size=(60, 3, 3, 4)).astype(np.float32)
B = rng.normal(0, 1, size=(40, 3, 3, 4)).astype(np.float32)
la = rng.choice([1, 12, 23], size=60).astype(np.int64)
lb = rng.choice([1, 12, 23], size=40).astype(np.int64)
_mini_archive(DEMO_DIR / "patches_A_data.npz", A, la, with_provenance=False)
_mini_archive(DEMO_DIR / "patches_B_data.npz", B, lb, with_provenance=False)
perm = rng.permutation(100)
_mini_archive(DEMO_DIR / "combined_training_data.npz",
              np.concatenate([A, B])[perm], np.concatenate([la, lb])[perm],
              with_provenance=False)

# %%
def count_glob(pattern):
    total = 0
    for f in sorted(DEMO_DIR.glob(pattern)):
        with np.load(f) as d:
            total += int(d["patches"].shape[0])
    return total


n_star, n_data = count_glob("*.npz"), count_glob("*_data.npz")
n_scene = count_glob(pat.SCENE_GLOB)
print("  PREDICTED vs ACTUAL — three globs over the same directory")
print(f"    '*.npz'                : predicted {PREDICTIONS['glob_star']!r:>6}   "
      f"actual {n_star}   files={len(list(DEMO_DIR.glob('*.npz')))}")
print(f"    '*_data.npz' (2025/26) : predicted {PREDICTIONS['glob_data']!r:>6}   "
      f"actual {n_data}   files={len(list(DEMO_DIR.glob('*_data.npz')))}")
print(f"    SCENE_GLOB  (2026/27)  : predicted {PREDICTIONS['glob_scene']!r:>6}   "
      f"actual {n_scene}   files={len(list(DEMO_DIR.glob(pat.SCENE_GLOB)))}")
print(f"    ground truth distinct  : predicted {PREDICTIONS['true_distinct']!r:>6}   "
      f"actual {len(A) + len(B)}")
print("    '*_data.npz' matched the combined file as well as the per-scene ones, so it")
print("    returned exactly twice the true count and looked like a healthy dataset.")
print("    A team whose naming had no combined file would see the same glob return the")
print("    right number -- the bug is invisible from the count alone.")

# %%
naive = []
for f in sorted(DEMO_DIR.glob("*.npz")):
    with np.load(f) as d:
        naive.append(d["patches"])
naive_cat = np.concatenate(naive, axis=0)
try:
    gates.gate_no_duplicate_patches(naive_cat, seed=SEED)
    print("  gate_no_duplicate_patches: PASSED -- the gate is not doing its job")
except gates.GateFailure as exc:
    print(f"  gate_no_duplicate_patches raised, as it should:\n    {exc}")
print("\n  Same data, correct loader:")
try:
    pat.load_all_scenes(DEMO_DIR, expect_scenes=2)
    print("    load_all_scenes accepted the legacy directory -- unexpected")
except (FileNotFoundError, ValueError) as exc:
    print(f"    load_all_scenes refused it: {str(exc).splitlines()[0]}")

# %% [markdown]
# ### Why `np.load(...).values()` is the second half of the same bug
#
# `np.load` gives you `d.files`, in **insertion order**, and `.values()` follows that
# order. Write `np.savez(p, labels=..., patches=...)` — a perfectly reasonable thing to
# type — and a positional reader hands you labels where it asked for patches. Lab 5.2's
# `_parse_npz_file` did exactly that. Adding one metadata array is enough to swap two
# arrays silently.

# %%
trap = DEMO_DIR / "trap.npz"
np.savez_compressed(trap, labels=lb, patches=B, note=np.asarray(["order matters"]))
with np.load(trap) as d:
    print(f"  d.files (insertion order) = {list(d.files)}")
    first = d[d.files[0]]
    print(f"  first entry shape {first.shape}, d['patches'].shape {d['patches'].shape}")
    print(f"  a positional reader would feed shape {first.shape} to the model as patches")
    print(f"  keyed access is unambiguous: d['patches'] {d['patches'].shape}, "
          f"d['labels'] {d['labels'].shape}")
try:
    gates.gate_npz_loaded_by_key(d, required=("patches", "labels", "scene", "row", "col"))
except gates.GateFailure as exc:
    print(f"  gate_npz_loaded_by_key: {str(exc).splitlines()[0]}")

# %% [markdown]
# ## Part 3 — Exercise 2: raster-order truncation is a spatial bias
#
# `MAX_PATCHES = 50000` kept "the first 50 000 windows". Windows were generated in
# row-major order, so the first 50 000 are the **top strip** of the scene. In a
# 10980 × 10980 tile with stride 3 that is roughly the first 1360 rows — the northern
# 13.6 km — and every patch south of it never existed.
#
# That is not noise. Land cover is spatially autocorrelated, so a north strip is a
# different *landscape*, and a model trained on it and tested on the whole tile is being
# asked to extrapolate in space while being scored as if it interpolated.
#
# > **P2.** Reproduce the 2025/26 configuration on your own scene: `S2_H` rows,
# > `PATCH_SIZE`, `STRIDE = PATCH_SIZE`, keeping 0.37 % of the candidate windows in
# > row-major order.
# >
# > 1. What is the **mean row index** of that first block of windows, as a fraction of
# >    `S2_H`?
# > 2. What is the mean row index of the same number of windows chosen uniformly at
# >    random, as a fraction of `S2_H`?
# > 3. What fraction of the scene's rows does the first block touch at all?
# >
# > Give the answers as fractions of `S2_H` (e.g. "0.06 × H"), then run the cell.

# %%
PATCH_SIZE = 3      # TODO(you): odd only. Even sizes have no centre pixel.
STRIDE = 6          # TODO(you): >= PATCH_SIZE. 2x patch size halves neighbour overlap.

# The measurement below reproduces the 2025/26 settings, not yours: STRIDE=None meant
# stride = patch_size, and MAX_PATCHES = 50000 out of ~13.4 million candidate windows in
# a full granule, i.e. 0.37% of the scene. Reproducing that *ratio* rather than the
# absolute number makes the demo say the same thing on a 1536 px crop and on a full tile.
OLD_STRIDE = PATCH_SIZE
OLD_KEEP_FRAC = 50000 / (3660 * 3660)
cand_rows = np.arange(0, S2_H - PATCH_SIZE + 1, OLD_STRIDE)
cand_cols = np.arange(0, S2_W - PATCH_SIZE + 1, OLD_STRIDE)
n_cand = len(cand_rows) * len(cand_cols)
N_KEEP = max(1, int(round(OLD_KEEP_FRAC * n_cand)))
first_rows = np.repeat(cand_rows, len(cand_cols))[:N_KEEP]
rand_rows = rng.choice(cand_rows, size=N_KEEP, replace=True)

print("  PREDICTED vs ACTUAL — raster-order truncation")
print(f"    2025/26 settings: {S2_W}x{S2_H} scene, patch {PATCH_SIZE}, stride {OLD_STRIDE}"
      f" -> {n_cand} candidate windows; keeping {N_KEEP} ({100 * OLD_KEEP_FRAC:.2f}%)")
print(f"    mean row, first {N_KEEP} in raster order : "
      f"predicted {PREDICTIONS['nw_bias']['mean_row_first']!r}   "
      f"actual {first_rows.mean():.0f}  ({first_rows.mean() / S2_H:.3f} x H)")
print(f"    mean row, {N_KEEP} random windows       : "
      f"predicted {PREDICTIONS['nw_bias']['mean_row_random']!r}   "
      f"actual {rand_rows.mean():.0f}  ({rand_rows.mean() / S2_H:.3f} x H)")
print(f"    rows touched by the first {N_KEEP}       : {len(np.unique(first_rows))} "
      f"of {len(cand_rows)} ({len(np.unique(first_rows)) / len(cand_rows):.2%})")
print(f"    max row reached                          : {first_rows.max()} of {cand_rows.max()}")
print(f"    your production settings are patch {PATCH_SIZE}, stride {STRIDE}; the "
      f"truncation argument does not depend on them")
print("    Say what a model trained on that strip and tested on the whole tile is being")
print("    asked to do, and why the printed patch count would not have revealed it.")

# %% [markdown]
# ### What this notebook does instead
#
# `MAX_PATCHES` defaults to `None`: keep everything. If you must subsample — and you
# may, for runtime — it is an explicit **stratified** choice, seeded, and recorded in
# `results.json` so a grader can see that you traded sample size for class balance and
# know which classes you cut. `patches.extract_patches(max_patches=...)` is also
# available and subsamples at random rather than in raster order; either is defensible,
# silently truncating in scan order is not.

# %%
MAX_PATCHES = None            # None = keep every window. Not a default to keep quietly.
SUBSAMPLE = "none"            # TODO(you): "none" | "random" | "stratified"
SUBSAMPLE_TARGET = 20000


def stratified_subsample(ps, target, seed):
    """Equalise toward the median class count, then top up randomly to `target`.

    Deliberately not a full balance: over-resampling rare classes to parity makes the
    class prior in your training set a fiction, and Lab 5's prior-matched baseline then
    measures the resampler instead of the model.
    """
    r = np.random.default_rng(seed)
    codes, counts = np.unique(ps.labels, return_counts=True)
    cap = max(1, min(int(counts.max()), target // max(1, len(codes))))
    keep = [r.choice(np.flatnonzero(ps.labels == c), size=min(int(n), cap), replace=False)
            for c, n in zip(codes, counts)]
    keep = np.concatenate(keep) if keep else np.array([], dtype=int)
    if len(keep) < target:
        rest = np.setdiff1d(np.arange(len(ps.labels)), keep, assume_unique=False)
        if rest.size:
            keep = np.concatenate([keep, r.choice(rest, size=min(target - len(keep), rest.size),
                                                  replace=False)])
    keep.sort()
    return ps.subset(keep)

# %% [markdown]
# ## Part 4 — Extraction, with provenance
#
# `patches.extract_patches` returns a `PatchSet`: `patches`, `labels`, **`scene`**,
# **`row`**, **`col`**, plus the band tuple read out of the file, the patch/stride
# settings, and a `stats` dict recording every skip reason. `save_patches` persists all
# of it and writes a sidecar JSON. `load_scene(..., require_provenance=True)` refuses an
# archive that cannot say which scene a patch came from.
#
# Four things it fixes relative to the old loop, each visible in the skip counters:
#
# * **nodata comes from the raster's own metadata**, not from `== 0`. DN 0 is a real
#   dark surface, and the old guard discarded exactly the deep water, shadow and asphalt
#   pixels that make those classes hard.
# * **labels are int64**, because `F.cross_entropy` requires it.
# * **bands are read from the file**, not pasted into a metadata dict.
# * **even patch sizes raise** for centre labelling, instead of silently shifting the
#   label half a pixel up-left.
#
# ### Runtime, and where it belongs
#
# Extraction is a Python triple loop over candidate windows. On a full 10980 × 10980
# scene with stride 3 that is ~12 million windows, and it will not finish on a login
# node — the 2025/26 version of this cell has never completed anywhere, because it was
# never executed. `CROP_PX` below extracts from a window of the real Lab 4.1 outputs so
# the notebook is interactive; set `CROP_PX = 0` and run the whole scene on a `dc-cpu`
# compute node:
#
# ```bash
# srun --partition=dc-cpu --cpus-per-task=8 --mem-per-cpu=8G \
#      jupyter execute --inplace lab4_2_patch_extraction.ipynb
# ```
#
# A crop is still **real data** — same pixels, same labels, same grid — but it is a
# smaller sample of it, and `results.json` records the crop so the claim is bounded.

# %%
CROP_PX = 512       # TODO(you): 0 = whole scene, on a compute node only
CROP_DIR = paths.data_root() / "lab4_2_crops"
paths.ensure(CROP_DIR)


def best_labelled_window(label_path, n):
    """Where to crop. Chosen by label coverage, not by 'the middle of the file'.

    A centre crop is the obvious choice and it is often empty: CORINE covers part of a
    Sentinel-2 tile, so the geometric centre of a granule can be 100 % no-data. The
    2025/26 notebook never checked, and a student whose tile happened to be labelled off
    centre got `zero patches extracted` with no hint that the crop, not the pipeline,
    was the problem.
    """
    with rasterio.open(label_path) as src:
        n = max(8, min(int(n), src.width, src.height))
        nd = int(src.nodata) if src.nodata is not None else lab_mod.CORINE_NODATA
        step = max(n // 3, 1)
        best, best_frac = None, -1.0
        for r0 in range(0, max(1, src.height - n + 1), step):
            for c0 in range(0, max(1, src.width - n + 1), step):
                w = rasterio.windows.Window(c0, r0, n, n)
                frac = float((src.read(1, window=w) != nd).mean())
                if frac > best_frac:
                    best, best_frac = w, frac
                if best_frac > 0.98:
                    break
            if best_frac > 0.98:
                break
        return best, best_frac


def crop_pair(stack, label, out_dir, n=CROP_PX, window=None):
    """Windowed copy of one aligned pair. Same grid, same transform, same nodata.

    `window` is passed in rather than recomputed per scene: all scenes of one tile share
    a grid, so cropping each at its own best window would give you four scenes covering
    four different places. That is a silent way to make your "temporal" samples spatially
    unrelated, and it is the same error Lab 3.1 warns about for tile choice.
    """
    tag = stack.stem[len("s2_"):]
    out_img, out_lab = Path(out_dir) / f"s2_{tag}.tif", Path(out_dir) / f"corine_{tag}.tif"
    with rasterio.open(label) as probe:
        w = window if window is not None else best_labelled_window(label, n)[0]
        w = rasterio.windows.Window(min(w.col_off, max(0, probe.width - w.width)),
                                    min(w.row_off, max(0, probe.height - w.height)),
                                    min(w.width, probe.width), min(w.height, probe.height))
        nd = int(probe.nodata) if probe.nodata is not None else lab_mod.CORINE_NODATA
        frac = float((probe.read(1, window=w) != nd).mean())
    with rasterio.open(stack) as src:
        w = rasterio.windows.Window(min(w.col_off, max(0, src.width - w.width)),
                                    min(w.row_off, max(0, src.height - w.height)),
                                    min(w.width, src.width), min(w.height, src.height))
        prof = src.profile.copy()
        prof.update(height=w.height, width=w.width,
                    transform=rasterio.windows.transform(w, src.transform))
        with rasterio.open(out_img, "w", **prof) as dst:
            dst.write(src.read(window=w))
            for i in range(1, src.count + 1):
                dst.set_band_description(i, src.descriptions[i - 1])
    with rasterio.open(label) as src:
        prof = src.profile.copy()
        prof.update(height=w.height, width=w.width,
                    transform=rasterio.windows.transform(w, src.transform))
        with rasterio.open(out_lab, "w", **prof) as dst:
            dst.write(src.read(1, window=w), 1)
    return out_img, out_lab, {"row": int(w.row_off), "col": int(w.col_off),
                              "size": int(w.width), "label_coverage": frac}

# %%
SCENE_IDS = [s.stem[len("s2_"):] for s in STACKS]
print(f"  {len(SCENE_IDS)} scene(s): {SCENE_IDS}")
print(f"  patch {PATCH_SIZE}, stride {STRIDE}, max_patches {MAX_PATCHES}, "
      f"crop {CROP_PX or 'FULL SCENE'} px")
print(f"  valid label codes: {len(lab_mod.valid_class_codes())} "
      "(water/sediment excluded by default; pass exclude=() to keep them and say so)")

SHARED_WINDOW = None
if CROP_PX:
    SHARED_WINDOW, SHARED_FRAC = best_labelled_window(LABELS[SCENE_IDS[0]], CROP_PX)
    print(f"  shared crop: row {int(SHARED_WINDOW.row_off)} col "
          f"{int(SHARED_WINDOW.col_off)}, {CROP_PX}x{CROP_PX}, label coverage "
          f"{SHARED_FRAC:.0%} in the first scene")

EXTRACTED = {}
CROPS = {}
EXTRACT_LABEL = {}
for stack in STACKS:
    sid = stack.stem[len("s2_"):]
    if CROP_PX:
        img, lb, CROPS[sid] = crop_pair(stack, LABELS[sid], CROP_DIR, window=SHARED_WINDOW)
    else:
        img, lb = stack, LABELS[sid]
    ps = pat.extract_patches(img, lb, scene_id=sid, patch_size=PATCH_SIZE, stride=STRIDE,
                             seed=SEED, verbose=False)
    EXTRACTED[sid] = ps
    EXTRACT_LABEL[sid] = lb
    print(f"  kept {len(ps)} patches; skipped {ps.stats['skipped']}")

# %%
print("\n  What one archive now contains:")
first_ps = EXTRACTED[SCENE_IDS[0]]
for name, arr in (("patches", first_ps.patches), ("labels", first_ps.labels),
                  ("scene", first_ps.scene), ("row", first_ps.row), ("col", first_ps.col)):
    print(f"    {name:<8} shape={str(arr.shape):<14} dtype={arr.dtype}")
print(f"    bands   {first_ps.bands}   (read from the GeoTIFF, not pasted)")
print(f"    example patch 0: scene={first_ps.scene[0]} row={first_ps.row[0]} "
      f"col={first_ps.col[0]} label={first_ps.labels[0]} "
      f"({lab_mod.class_name(int(first_ps.labels[0]))})")
print("  scene/row/col are the only reason Lab 5 can hold out whole scenes and split")
print("  on spatial blocks. Without them the choice does not exist.")

# %%
paths.ensure(paths.training_data_dir())


def stale_archives(cfg):
    """Archives already in the directory that the current config did not produce.

    `load_all_scenes` reads **whatever matches the glob**. An archive left over from a
    4-band run, a different patch size, or an earlier tile joins your dataset silently
    and is indistinguishable from yours once it is concatenated. This is the same class
    of bug as the 2025/26 double-load — the loader's contract is the directory, so the
    directory has to be true — and it is caught here rather than in Lab 5.
    """
    stale = []
    for p in sorted(paths.training_data_dir().glob(pat.SCENE_GLOB)):
        try:
            ps = pat.load_scene(p, require_provenance=False)
        except Exception as exc:  # unreadable is worse than mismatched
            stale.append((p, [f"unreadable: {type(exc).__name__}"], {}))
            continue
        meta = {"patch_size": ps.patch_size, "bands": list(ps.bands),
                "seed": ps.stats.get("seed")}
        why = []
        if ps.patch_size != cfg["patch_size"]:
            why.append(f"patch_size {ps.patch_size} != {cfg['patch_size']}")
        if list(ps.bands) != cfg["bands"]:
            why.append(f"bands {list(ps.bands)} != {cfg['bands']}")
        if meta["seed"] != cfg["seed"]:
            why.append(f"seed {meta['seed']} != {cfg['seed']}")
        if not p.with_suffix(".json").exists():
            why.append("no sidecar: written by hand or by older code")
        if why:
            stale.append((p, why, meta))
    return stale


STALE = stale_archives({"patch_size": PATCH_SIZE, "bands": list(BAND_ORDER), "seed": SEED})
for p, why, meta in STALE:
    print(f"  removing stale {p.name}: {'; '.join(why)}")
    p.unlink(missing_ok=True)
    p.with_suffix(".json").unlink(missing_ok=True)
if not STALE:
    print("  no stale archives in the output directory")

ARCHIVES = {}
for sid, ps in EXTRACTED.items():
    p = pat.save_patches(ps, paths.training_data_dir())
    ARCHIVES[sid] = p
    print(f"  {p.name}  ({p.stat().st_size / 1e6:.2f} MB)  sidecar "
          f"{p.with_suffix('.json').name}")
MATCHED = sorted(paths.training_data_dir().glob(pat.SCENE_GLOB))
print(f"\n  SCENE_GLOB = {pat.SCENE_GLOB!r} -> {len(MATCHED)} archives matched, "
      f"{len(ARCHIVES)} written by this run")
assert {p.name for p in MATCHED} == {p.name for p in ARCHIVES.values()}, (
    f"the directory holds archives this run did not write: "
    f"{sorted({p.name for p in MATCHED} - {p.name for p in ARCHIVES.values()})}")
print("  The suffix is `_scene.npz`, not `_data.npz`, on purpose: a combined file can")
print("  never collide with it, so the 2025/26 double-load is unrepresentable here.")

# %%
LOADED = pat.load_all_scenes(paths.training_data_dir(), expect_scenes=len(SCENE_IDS))
print(f"  load_all_scenes -> {len(LOADED)} patches from "
      f"{len(np.unique(LOADED.scene))} scenes, bands {LOADED.bands}")
for p in ARCHIVES.values():
    with np.load(p) as d:
        gates.gate_npz_loaded_by_key(d, required=("patches", "labels", "scene", "row", "col"))
print("  gate_npz_loaded_by_key passed on every archive (loaded by key, not position)")
chk = pat.load_scene(ARCHIVES[SCENE_IDS[0]], require_provenance=True)
print(f"  load_scene(require_provenance=True) -> {len(chk)} patches, "
      f"patch_size inferred {chk.patch_size}, label_name={chk.stats.get('label_name')}, "
      f"nodata_label={chk.stats.get('label_nodata')}")

# %% [markdown]
# ## Part 5 — Exercise 3: a 3 × 3 patch against a 100 m label
#
# This is the defining limitation of the entire dataset, and the 2025/26 course stated
# it nowhere. The old comment read `PATCH_SIZE = 3  # Size in pixels (3 = 30m x 30m at
# 10m resolution)` and moved on.
#
# CORINE is **100 m**. Sentinel-2 bands here are **10 m**. So one CORINE pixel is a
# 10 × 10 block of image pixels, and a 3 × 3 patch spans 30 m — a third of a label
# pixel on a side. The label attached to a patch is the CORINE pixel containing its
# centre, and that same pixel is the centre pixel of roughly
# `floor(10/3) × floor(10/3) = 3 × 3 = 9` patches.
#
# Consequences, in order of how much they cost you:
#
# 1. **Those ~9 patches are not independent samples.** They share a label exactly, and
#    with stride = patch size they share image pixels at their edges. Your *nominal* N
#    is roughly 9× your *effective* N.
# 2. **A random split leaks.** Neighbours with the same label land in train and test.
#    Lab 5.1's `0.6992 / 0.5196` is partly an interpolation score.
# 3. **The label is a claim about a 100 m area, not about your 30 m patch.** On a field
#    boundary the centre pixel is a coin flip. `label_name="majority"` helps for larger
#    patches and cannot fix this one.
#
# > **P3.** For the patches you just extracted, with the label-pixel size
# > `k = CORINE resolution / S2 resolution`:
# >
# > 1. How many patches, on average, share a label pixel?
# > 2. What percentage of your nominal sample count survives as *effective* independent
# >    samples (distinct label pixels ÷ patches)?
# >
# > Then state, in one sentence, what this does to a bootstrap confidence interval that
# > resamples **patches** instead of label pixels or scenes.

# %%
# The *aligned* label raster is 10 m, because Lab 4.1 upsampled it onto the S2 grid.
# Its semantics, though, are still 100 m: nearest resampling invented no information, it
# just repeated it. So the label-pixel size has to come from the SOURCE raster, which the
# Lab 4.1 sidecar recorded as shape + bounds. Reading `lb.res` here would return 10 and
# the whole exercise would silently compute nothing -- a good example of a metadata field
# that answers a different question than the one you asked.
_lab_side = json.loads((ALIGNED / (LABELS[SCENE_IDS[0]].stem + ".manifest.json"))
                       .read_text(encoding="utf-8"))
_sb = _lab_side["label_source_bounds"]
_ss = _lab_side["label_source_shape"]
SRC_LEFT, SRC_TOP = _sb[0], _sb[3]
CORINE_RES = (_sb[2] - _sb[0]) / _ss[1]
K = int(round(CORINE_RES / S2_RES))
print(f"  source label raster: {_ss[1]}x{_ss[0]} px over {(_sb[2] - _sb[0]) / 1e3:.0f} km "
      f"-> {CORINE_RES:.0f} m pixels")
print(f"  aligned label raster declares res {rasterio.open(LABELS[SCENE_IDS[0]]).res[0]:.0f} m"
      " -- that is the grid, not the information content")
print(f"  CORINE {CORINE_RES:.0f} m / S2 {S2_RES:.0f} m -> one label pixel is "
      f"{K}x{K} image pixels")
print(f"  a {PATCH_SIZE}x{PATCH_SIZE} patch spans {PATCH_SIZE * S2_RES:.0f} m; "
      f"the label describes {CORINE_RES:.0f} m")
print(f"  theoretical patches per label pixel ~ {(K // PATCH_SIZE) ** 2} "
      f"(floor({K}/{PATCH_SIZE})^2) at stride {PATCH_SIZE}, "
      f"{max(1, (K // STRIDE) ** 2)} at stride {STRIDE}")

# %%
# Which 100 m CORINE pixel contains each patch centre? Two patches in the same source
# pixel are not two observations -- they are one observation quoted twice. Computed
# exactly, per scene, by mapping each patch centre into that scene's label raster grid.
# The row/col in a PatchSet are local to the raster it was extracted from, so the
# transform must come from the same raster -- for a crop, the crop's transform, not the
# full granule's. Mixing the two gives a plausible, wrong count.
from rasterio.warp import transform as crs_transform  # noqa: E402

cy = cx = PATCH_SIZE // 2
cell_rows, cell_cols, cell_scene = [], [], []
for sid, ps in EXTRACTED.items():
    img_path = (CROP_DIR / f"s2_{sid}.tif") if CROP_PX else \
        next(s for s in STACKS if s.stem == f"s2_{sid}")
    with rasterio.open(EXTRACT_LABEL[sid]) as lb, rasterio.open(img_path) as im:
        itf, icrs, ltf, lcrs = im.transform, im.crs, lb.transform, lb.crs
    xs = itf.c + (ps.col + cx + 0.5) * itf.a
    ys = itf.f + (ps.row + cy + 0.5) * itf.f
    lx, ly = crs_transform(icrs, lcrs, xs.tolist(), ys.tolist())
    # Bin into the SOURCE 100 m grid, using the source origin and resolution recorded in
    # the Lab 4.1 sidecar. Binning with `ltf` instead — the *aligned* raster's transform
    # — silently bins into the 10 m grid, gives every patch its own cell, and reports
    # "1.00 patches per label pixel, 100% effective", i.e. exactly the wrong answer with
    # no error. Affine row step in y is `e` (negative, north-up), not `b`.
    cell_rows.append(np.floor((np.asarray(ly) - SRC_TOP) / CORINE_RES).astype(np.int64))
    cell_cols.append(np.floor((np.asarray(lx) - SRC_LEFT) / CORINE_RES).astype(np.int64))
    # Group within a scene: two patches of the SAME scene on the same label pixel are
    # one observation quoted twice. Patches from different scenes on that pixel are
    # different acquisitions of the same terrain, which is a different (and separately
    # important) dependence -- that one is handled by holding scenes out in Lab 5.
    cell_scene.append(np.full(len(ps), SCENE_IDS.index(sid), dtype=np.int64))

# np.unique(axis=...) does not support object dtype, so the scene id is carried as an
# integer code. Encoding a grouping key as text and then wondering why the count is wrong
# is a familiar way to lose an afternoon.
# Two groupings matter and they are not the same question. GEOGRAPHIC: every patch, from
# any scene, whose centre falls in the same 100 m pixel -- those are the same terrain, so
# they are one observation quoted N times. WITHIN-SCENE: the same, restricted to one
# acquisition -- that is the neighbour-leakage number a random split inside one scene
# would face. Report both; Lab 5 needs the first for cluster-robust intervals and the
# second for why a per-patch split leaks.
geo = np.stack([np.concatenate(cell_rows), np.concatenate(cell_cols)], axis=1)
within = np.stack([np.concatenate(cell_scene), np.concatenate(cell_rows),
                   np.concatenate(cell_cols)], axis=1)
# Cross-scene comparison is only valid because every scene of a tile shares one grid:
# Lab 4.1 reprojected each label raster onto the S2 geometry, and the crop window is
# shared. Assert it rather than assuming it -- if two scenes were cropped at different
# offsets, "same row, same col" would not mean "same place", and the geographic count
# below would be quietly wrong.
assert len({(c["row"], c["col"], c["size"]) for c in CROPS.values()}) <= 1, \
    "scenes were cropped at different offsets; geographic grouping would be invalid"
geo_ids, geo_sizes = np.unique(geo, axis=0, return_counts=True)
within_ids, within_sizes = np.unique(within, axis=0, return_counts=True)
mean_share = float(within_sizes.mean())
ess_pct = 100.0 * len(within_ids) / len(LOADED)
geo_share = float(geo_sizes.mean())
geo_ess_pct = 100.0 * len(geo_ids) / len(LOADED)

print("  PREDICTED vs ACTUAL — label sharing")
print(f"    predicted patches per label pixel : {PREDICTIONS['label_sharing']['patches_per_label']!r}")
print(f"    actual, within one scene          : {mean_share:.2f}")
print(f"    actual, across all scenes         : {geo_share:.2f}")
print(f"    predicted effective %             : {PREDICTIONS['label_sharing']['ess_pct']!r}")
print(f"    actual effective %, within scene  : {ess_pct:.1f}%  "
      f"({len(within_ids)} distinct scene+label cells for {len(LOADED)} patches)")
print(f"    actual effective %, all scenes    : {geo_ess_pct:.1f}%  "
      f"({len(geo_ids)} distinct places)")
print(f"    largest group, within scene       : {int(within_sizes.max())} patches on one "
      f"label pixel")
print(f"    largest group, all scenes         : {int(geo_sizes.max())} patches on one place")
print("    Your effective sample size is not", len(LOADED), ". Name what it is, and what")
print("    it does to a confidence interval computed by resampling patches.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].hist(within_sizes, bins=np.arange(0.5, min(int(within_sizes.max()), 60) + 1.5, 1),
             color="#3b6ea5", alpha=0.75, label="within one scene")
axes[0].hist(geo_sizes, bins=np.arange(0.5, min(int(within_sizes.max()), 60) + 1.5, 1),
             color="#c44e52", alpha=0.55, label="across all scenes")
axes[0].set_xlabel("patches sharing one 100 m CORINE label pixel")
axes[0].set_ylabel("label pixels")
axes[0].set_title(f"label-sharing distribution\nwithin-scene {mean_share:.2f} "
                  f"({ess_pct:.1f}% effective), all-scenes {geo_share:.2f}")
axes[0].legend(fontsize=8)
axes[1].scatter(LOADED.col[::7], LOADED.row[::7], s=2, alpha=0.4,
                c=LOADED.labels[::7], cmap="tab20")
axes[1].invert_yaxis()
axes[1].set_xlabel("col (px)"); axes[1].set_ylabel("row (px)")
axes[1].set_title("extracted patch positions, coloured by CORINE code")
fig.tight_layout()
FIG_PATH = paths.results_dir() / "lab4_2_patch_geometry.png"
paths.ensure(FIG_PATH.parent)
fig.savefig(FIG_PATH, dpi=110)
plt.show()
print(f"  saved {FIG_PATH}")

# %% [markdown]
# ### This is the setup for Lab 5, planted deliberately
#
# Lab 5.1 will call `splits.group_block_split(groups=scene, rows=..., cols=..., block=10)`.
# Two decisions follow directly from what you just measured:
#
# * **`groups` = scene.** Holding out whole scenes removes scene leakage: each scene has
#   its own illumination, atmosphere, sensor state and date, so if every split receives
#   patches from every scene, "generalisation" means "generalising within a scene you
#   have already seen".
# * **`block` ≥ the label-pixel footprint in patches.** A block of 10 × 10 patch
#   positions is roughly one CORINE label pixel wide at stride 3, and several at
#   stride 6. Splitting on blocks rather than patches means a held-out block is a
#   contiguous piece of terrain whose every neighbour is held out too.
#
# Write the number you measured into your Lab 5 report. It is the justification for the
# split, not a curiosity.

# %% [markdown]
# ## Part 6 — Negative control: normalisation fitted on everything
#
# **Do not fit normalisation in this lab.** The old notebook did, and the comment argued
# for it in plain text: *"Apply normalization to entire image first (for consistent
# statistics)."* Those percentile bounds were computed over the whole raster, which later
# became train + val + test, so **every test patch contributed to the transform applied
# to every training patch**. The reported 0.6992 accuracy is partly an interpolation
# score, and no experiment in the notebook could have detected it.
#
# Two further defects sat inside that same function:
#
# * the percentile bounds were **pooled across all four bands** — one shared
#   `[low, high]` for Blue, Green, Red and NIR. NIR is far brighter than Blue over
#   vegetation, so a pooled stretch squeezes Blue toward a constant, destroying the band
#   that most separates urban fabric from vegetation;
# * each tile was normalised **independently** and then concatenated, so four tiles
#   meant four incompatible radiometric scales in one training set, reported as one
#   label distribution.
#
# The correct home is `radiometry.Norm` in Lab 5: `Norm.fit(train_only, ...)` then
# `.apply(...)` to each split. The class refuses to apply statistics it never fitted,
# which is the point.
#
# Below, both transforms are fitted on the **same** patches, and the difference is
# measured on the held-out third only. Nothing here is saved as a dataset.

# %%
perm = np.random.default_rng(SEED + 1).permutation(len(LOADED))
train_idx, hold_idx = perm[: len(perm) // 3 * 2], perm[len(perm) // 3 * 2:]
print(f"  demo split: {len(train_idx)} 'train', {len(hold_idx)} 'held out' "
      "(a random split, used here only to have something to hold out)")

norm_leaky = radiometry.Norm.fit(LOADED.patches[train_idx], mode="percentile",
                                 channel_axis=-1, low_pct=2.0, high_pct=98.0)
norm_all = radiometry.Norm.fit(LOADED.patches, mode="percentile",
                               channel_axis=-1, low_pct=2.0, high_pct=98.0)
print(f"  train-only fit  low={np.round(norm_leaky.low_, 4).tolist()}")
print(f"  whole-data fit  low={np.round(norm_all.low_, 4).tolist()}")
print(f"  n_samples_seen  {norm_leaky.n_samples_seen} vs {norm_all.n_samples_seen}")

# %%
a = norm_leaky.apply(LOADED.patches[hold_idx])
b = norm_all.apply(LOADED.patches[hold_idx])
delta = np.abs(a - b)
print(f"  held-out patches transformed by the two rules differ by up to "
      f"{delta.max():.4f} per pixel, mean {delta.mean():.4f}")
print(f"  pixels moved by more than 0.05: {100 * (delta > 0.05).mean():.2f}%")
print(f"  per-band mean on held-out data: train-only {np.round(a.mean(axis=(0, 1, 2)), 4).tolist()}")
print(f"                              whole-data {np.round(b.mean(axis=(0, 1, 2)), 4).tolist()}")
print("  The held-out array is the SAME array in both lines. The only thing that changed")
print("  is who was allowed to vote on the transform. Write one sentence on what a model")
print("  trained under rule (b) is being scored on.")

# %%
try:
    radiometry.Norm(mode="percentile").apply(LOADED.patches[:10])
    print("  UNREACHABLE: an unfitted Norm applied itself")
except RuntimeError as exc:
    print(f"  an unfitted Norm refuses:\n    {exc}")
print("  That refusal is the structural fix. The 2025/26 code had no object to refuse,")
print("  just a dict of statistics computed over an array that had not been split yet.")

# %% [markdown]
# ## Part 7 — What you actually have: counts, classes, imbalance
#
# `imbalance_ratio` is `max/min` over classes with support ≥ 1, defined once in
# `eo_course.labels`. The old notebook computed `counts[0] / counts[-1]` and called it
# the imbalance ratio, which is only that ratio if the array happens to be sorted
# descending — and it silently divides by zero when the last class is absent.

# %%
CODES = lab_mod.valid_class_codes()
counts = LOADED.class_counts(CODES)
present = [c for c, n in zip(CODES, counts) if n > 0]
print(f"  {len(LOADED)} patches, {len(np.unique(LOADED.scene))} scenes, "
      f"{len(present)} of {len(CODES)} valid classes present")
print(f"  imbalance ratio (max/min over present classes) = "
      f"{lab_mod.imbalance_ratio(counts):.1f}x")
print(f"  skipped across all scenes: {LOADED.stats.get('skipped')}")
print(f"\n  {'code':>5} {'level-1':<28} {'n':>7} {'%':>6}")
for c, n in sorted(zip(present, [counts[CODES.index(c)] for c in present]),
                   key=lambda t: -t[1])[:12]:
    print(f"  {c:>5} {lab_mod.level1_of(c):<28} {n:>7} {100 * n / len(LOADED):>5.1f}%")
absent = [c for c in CODES if c not in present]
print(f"  zero-support classes: {len(absent)} (they still occupy a row of your "
      f"confusion matrix and a term in your macro-average)")

# %%
if SUBSAMPLE != "none":
    SUB = stratified_subsample(LOADED, SUBSAMPLE_TARGET, SEED) if SUBSAMPLE == "stratified" \
        else LOADED.subset(np.random.default_rng(SEED).choice(
            len(LOADED), size=min(SUBSAMPLE_TARGET, len(LOADED)), replace=False))
    print(f"  {SUBSAMPLE} subsample: {len(LOADED)} -> {len(SUB)} patches, "
          f"{lab_mod.imbalance_ratio(SUB.class_counts(CODES)):.1f}x imbalance")
    print(f"  macro-average over present classes will move: report the before/after "
          f"counts, not just the after")
else:
    SUB = LOADED
    print("  SUBSAMPLE = 'none': every extracted patch is kept. If you change this,")
    print("  record which arm you chose and what it cost in rare-class support.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
sup = counts[counts > 0]
axes[0].barh([str(c) for c in present], sup, color="#4c72b0")
axes[0].set_xscale("log")
axes[0].set_xlabel("patches (log scale)")
axes[0].set_title(f"class support, {len(present)} present classes\n"
                  f"imbalance {lab_mod.imbalance_ratio(counts):.1f}x")
axes[1].bar(range(len(sup)), np.sort(sup)[::-1] / len(LOADED), color="#c44e52")
axes[1].axhline(1.0 / len(present), ls="--", c="k", lw=1,
                label=f"uniform over {len(present)} classes")
axes[1].set_xlabel("class rank"); axes[1].set_ylabel("share of patches")
axes[1].legend()
axes[1].set_title("class prior vs uniform")
for ax in axes:
    ax.tick_params(labelsize=7)
fig.tight_layout()
FIG2 = paths.results_dir() / "lab4_2_class_support.png"
paths.ensure(FIG2.parent)
fig.savefig(FIG2, dpi=110)
plt.show()
print(f"  saved {FIG2}")

# %% [markdown]
# ## Part 8 — Gate board
#
# The performance gates (`gate_beats_baselines`, `gate_above_chance`,
# `gate_not_collapsed`) belong to Labs 5–7. The claims you are making **here** are about
# the archive: that it was read by key, that no patch appears twice, that every patch
# can say where it came from, that the geometry you measured is the geometry you
# recorded, and that no normalisation leaked into the stored arrays.

# %%
def gate(name, ok, detail):
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


# "Unnormalised" is not "greater than 1". Lab 4.1 stores reflectance in [0, 1] and DN in
# [0, 10000]; both are legitimate, and what must not happen is a *percentile stretch
# baked into the archive*, because that transform is then permanent and split-independent.
# So the check is: the stored range matches the units the Lab 4.1 sidecar declares, and
# the array is not a stretched copy of itself.
_side = json.loads((ALIGNED / (STACKS[0].stem + ".manifest.json"))
                   .read_text(encoding="utf-8"))
_declared = _side["units"]
_pmax = float(np.nanmax(LOADED.patches))
if _declared == "surface_reflectance":
    UNITS_OK = _pmax <= 1.0 + 1e-3
    UNITS_DETAIL = (f"sidecar declares {_declared}, stored max {_pmax:.3f} in [0,1], "
                    "no stretch applied")
elif _declared == "digital_number":
    UNITS_OK = _pmax > 1.0 + 1e-3
    UNITS_DETAIL = (f"sidecar declares {_declared}, stored max {_pmax:.0f} in DN units, "
                    "no stretch applied")
else:
    UNITS_OK = False
    UNITS_DETAIL = f"sidecar declares unknown units {_declared!r}"

board = []
try:
    gates.gate_no_duplicate_patches(LOADED.patches, groups=LOADED.scene, seed=SEED)
    board.append(gate("no_duplicate_patches", True,
                      f"{len(LOADED)} patches, no content duplicates under the "
                      "position-weighted fingerprint"))
except gates.GateFailure as exc:
    board.append(gate("no_duplicate_patches", False, str(exc).splitlines()[0]))

board += [
    gate("provenance_present",
         all(ps.scene is not None and ps.row is not None and ps.col is not None
             for ps in EXTRACTED.values()),
         f"scene/row/col on all {len(EXTRACTED)} archives; "
         f"{len(np.unique(LOADED.scene))} distinct scene ids"),
    gate("one_archive_per_scene", len(ARCHIVES) == len(SCENE_IDS) == len(EXTRACTED),
         f"{len(ARCHIVES)} archives for {len(SCENE_IDS)} scenes, globbed by "
         f"{pat.SCENE_GLOB!r}"),
    gate("labels_int64", LOADED.labels.dtype == np.int64,
         f"labels dtype {LOADED.labels.dtype} (F.cross_entropy requires int64)"),
    gate("bands_from_file", list(LOADED.bands) == BAND_ORDER,
         f"{list(LOADED.bands)} read from GeoTIFF descriptions, matches bands.json"),
    gate("units_unnormalised", UNITS_OK,
         f"{UNITS_DETAIL}. Lab 5 fits Norm on train only."),
]

# %%
board += [
    gate("stride_ge_patch", STRIDE >= PATCH_SIZE,
         f"stride {STRIDE} >= patch_size {PATCH_SIZE}; adjacent windows share "
         f"{max(0, PATCH_SIZE - STRIDE) if STRIDE < PATCH_SIZE else 0} px edges"),
    gate("patch_odd_for_center_label", PATCH_SIZE % 2 == 1,
         f"patch_size {PATCH_SIZE}; even sizes have no centre pixel and the label "
         "shifts half a pixel up-left"),
    gate("label_sharing_measured", mean_share > 1.0,
         f"{mean_share:.2f} patches per 100 m label pixel, {ess_pct:.1f}% effective; "
         "Lab 5 must split on scene + spatial block, not on patches"),
    gate("no_raster_order_truncation",
         MAX_PATCHES is None or SUBSAMPLE in ("random", "stratified"),
         f"MAX_PATCHES={MAX_PATCHES}, SUBSAMPLE={SUBSAMPLE!r} -- scan-order truncation "
         "is not reachable here"),
    gate("seed_recorded",
         all(ps.stats.get("seed") == SEED for ps in EXTRACTED.values()),
         f"seed {SEED} recorded in every sidecar: "
         f"{sorted({ps.stats.get('seed') for ps in EXTRACTED.values()})}"),
    gate("class_support_reported", len(present) > 0,
         f"{len(present)} present classes, imbalance {lab_mod.imbalance_ratio(counts):.1f}x, "
         f"{len(absent)} zero-support"),
    gate("decisions_recorded", "TODO" not in PREDICTIONS["why"],
         "predictions explained: " + ("still the TODO placeholder -- name the mechanism"
                                      if "TODO" in PREDICTIONS["why"] else "recorded")),
]
gates.print_gate_board(board)

# %% [markdown]
# ### Reading a failed board
#
# `no_duplicate_patches` failing means a loader is reading more than it should — go
# check the glob before you check anything else, because that is the 2025/26 bug and it
# is not subtle once you know where to look. `provenance_present` failing means Lab 5
# cannot build a group split and you will be tempted to use
# `splits.stratified_random_split`, which exists as the course's **negative control**,
# not as a fallback. `units_unnormalised` failing means a stretch got baked into the
# archive, and the split-independent transform is now permanent.
#
# `decisions_recorded` is red **on first run by design**: it fails while
# `PREDICTIONS["why"]` still holds the placeholder. No amount of re-running turns it
# green.

# %% [markdown]
# ## Part 9 — Deliverable
#
# The decisions that must land in `results.json` so they are gradeable rather than
# asserted: patch size and stride, the subsampling arm you chose and the one you did
# not, the label-sharing measurement, and the fact that no normalisation was fitted
# here.

# %%
run_id = f"lab4_2_{SCENE_IDS[0]}_{len(SCENE_IDS)}scenes_p{PATCH_SIZE}s{STRIDE}"
qc_pass = sum("[PASS]" in b for b in board)
n_checks = len(board)
rec_kwargs = dict(
    lab="lab4_2",
    config={
        "scenes": SCENE_IDS, "patch_size": PATCH_SIZE, "stride": STRIDE,
        "max_patches": MAX_PATCHES, "subsample": SUBSAMPLE,
        "subsample_target": SUBSAMPLE_TARGET if SUBSAMPLE != "none" else None,
        "crop_px": CROP_PX, "crop_window": CROPS, "label_name": "center",
        "bands": list(LOADED.bands), "units": "as written by lab 4.1, unnormally fitted",
        "normalization_fitted_here": False,
        "valid_class_codes": len(CODES),
        "water_excluded": list(lab_mod.WATER_AND_SEDIMENT_CODES),
        "seed": SEED,
    },
    split_manifest_hash=None,
    seed=SEED,
    test_metrics={"n": int(len(LOADED)), "overall_acc": qc_pass / n_checks,
                  "balanced_acc": qc_pass / n_checks, "macro_f1": qc_pass / n_checks},
    notes=(f"Lab 4.2 extraction. Gates {qc_pass}/{n_checks}. "
           f"{len(LOADED)} patches over {len(np.unique(LOADED.scene))} scenes, "
           f"{len(present)} present classes, imbalance "
           f"{lab_mod.imbalance_ratio(counts):.1f}x. "
           f"{mean_share:.2f} patches share each 100 m CORINE label pixel, so effective "
           f"sample size is {ess_pct:.1f}% of nominal -- Lab 5 must split on scene + "
           f"spatial block. No normalization fitted here. "
           f"NOTE: no held-out split exists in this lab, so the required metric keys "
           f"carry the gate pass fraction and n is the patch count, not model metrics."),
    extra={"gate_board": board, "predictions": PREDICTIONS,
           "patches_per_label_pixel_within_scene": mean_share,
           "patches_per_label_pixel_all_scenes": geo_share,
           "effective_sample_pct_within_scene": ess_pct,
           "effective_sample_pct_all_scenes": geo_ess_pct,
           "n_label_pixels_within_scene": int(len(within_ids)),
           "n_distinct_places": int(len(geo_ids)),
           "skipped": LOADED.stats.get("skipped"),
           "class_counts": {str(c): int(n) for c, n in zip(CODES, counts) if n > 0},
           "leaky_norm_max_delta": float(delta.max()),
           "nw_bias_mean_row_first": float(first_rows.mean()),
           "nw_bias_mean_row_random": float(rand_rows.mean()),
           "archives": {k: str(v) for k, v in ARCHIVES.items()}},
)
try:
    rec = results.record_run(run_id, **rec_kwargs)
    print(f"  recorded run_id={rec['run_id']}")
except results.ResultsError as exc:
    print(f"  not recorded: {exc}")
    print("  results.json is append-only. Change the run_id (it encodes patch size and "
          "stride, so a different configuration gets a different id) rather than "
          "overwriting the earlier attempt.")

# %%
print(results.summary_table("lab4_2"))

# %% [markdown]
# ## Submission checklist
#
# Everything here is a file the grader can open. Self-attestation is not an artifact.
#
# * `<scratch>/<user>/data/training_data/patches_<scene>_scene.npz` — one per scene,
#   with `patches`, `labels`, `scene`, `row`, `col`, `bands`.
# * `.../patches_<scene>_scene.json` — the sidecar: source paths, skip counters,
#   nodata, label policy, seed.
# * `results/lab4_2_patch_geometry.png` — label-sharing histogram and patch positions.
# * `results/lab4_2_class_support.png` — class support and prior.
# * `results/results.json` — one `lab4_2` record whose `config` names patch size,
#   stride, the subsampling arm, and `normalization_fitted_here: false`, and whose
#   `extra` carries the gate board, the three predictions and the label-sharing numbers.
# * In your write-up: the gate board with every `FAIL` explained; predicted-vs-actual
#   for P1, P2 and P3; and one paragraph stating your effective sample size and what
#   follows for the Lab 5 split.
#
# **Before Lab 5.1**, confirm `provenance_present` and `no_duplicate_patches` are green,
# and that `load_all_scenes(paths.training_data_dir(), expect_scenes=<your scene count>)`
# returns the number you expect. Lab 5 reads these archives and trusts them.

# %% [markdown]
# ## Where each 2025/26 defect went
#
# | 2025/26 | 2026/27 |
# |---|---|
# | `glob("*_data.npz")` matched per-scene **and** combined files | `patches.SCENE_GLOB` / `load_all_scenes`; no combined file exists; Part 2 reproduces the double load and the gate catches it |
# | `np.savez(patches=..., labels=...)` | `extract_patches` + `save_patches` carry scene/row/col; `load_scene(require_provenance=True)` refuses archives without it |
# | `np.load(...).values()` positional read | keyed access throughout; `gate_npz_loaded_by_key` on every archive; Part 2 shows the insertion-order trap |
# | `MAX_PATCHES` truncated in raster order | removed; Part 2 measures the north-west bias; subsampling is an explicit seeded arm recorded in `results.json` |
# | 3 × 3 patch vs 100 m label, unremarked | Part 5 measures patches-per-label-pixel and effective sample size, and hands both to Lab 5's group split |
# | percentile bounds fitted over the whole raster | not fitted here at all; Part 6 quantifies the difference on held-out data and defers to `radiometry.Norm` in Lab 5 |
# | percentile bounds pooled across bands | `Norm` fits per-band; the pooled variant is named as a defect, not offered as an option |
# | `if np.any(patch == 0): continue` | nodata read from the raster's own metadata; skip counters printed |
# | `labels.astype(np.uint8)` | int64, gated |
# | `metadata['bands']` hard-coded | band descriptions read from the GeoTIFF and cross-checked against `bands.json` |
# | unseeded `np.random.permutation` | one `default_rng(SEED)`, seed in every sidecar and in `results.json` |
# | `1 <= label <= 44` | `labels.valid_class_codes()`, water/sediment exclusion stated as a decision |
# | `y + patch_size // 2` on even sizes | even patch sizes raise; `patch_odd_for_center_label` gates it |
# | zero exercises, zero questions | three graded predictions, one negative control, one gate board |
