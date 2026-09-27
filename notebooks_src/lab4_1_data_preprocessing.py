# %% [markdown]
# # Lab 4.1 — Preprocessing: units, grids, and what you invent when you are careless
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# This lab turns the four Level-2A granules you acquired in Lab 3.1 into the two
# artifacts Lab 4.2 consumes, per scene:
#
# * `s2_<scene_id>.tif` — the band stack, in **surface reflectance**, with a declared
#   `nodata`, per-band descriptions written into the file, and a sidecar
#   `manifest.json` recording where every band came from;
# * `corine_<scene_id>.tif` — CORINE 2018 reprojected onto the S2 grid with
#   **nearest** resampling and nodata handled on both ends.
#
# It assumes Lab 3.1 produced `<results>/manifest.json` and flat `*.SAFE` directories
# under `paths.data_root()`. It produces the input for Lab 4.2.
#
# **You must run every cell yourself.** These notebooks ship with no outputs on
# purpose. The 2025/26 copy of this notebook was committed with its main processing
# loop reporting `Successful: 0 / Skipped: 4` — every tile was skipped because outputs
# already existed on the instructor's scratch — while the cell that stacked the bands
# had never been executed at all. The notebook looked complete. Nothing in it had run.
#
# ## What this lab is actually about
#
# Not "run rasterio functions". Three failure modes, each of which silently corrupts
# every number reported in Labs 5–7:
#
# | failure | what it does to your result |
# |---|---|
# | units never converted from DN to reflectance | every reflectance threshold is wrong by 10⁴; a pretrained backbone sees a ~1000× scale error and collapses to one class |
# | reprojection without `src_nodata`/`dst_nodata` | missing data becomes **class 0 / DN 0**, i.e. invented as a dark surface, and then trained on |
# | interpolating a categorical label raster | codes that do not exist in the taxonomy appear, and are silently cast into codes that do |
#
# All three happened in 2025/26. All three are invisible in a printed summary.

# %% [markdown]
# ## The 2025/26 defects this notebook closes
#
# Kept as a table because you should be able to point at the cell that fixes each one.
#
# | old defect | consequence | fixed in |
# |---|---|---|
# | `reflectance = DN / 10000` appeared nowhere in Lab 4.1 or 4.2 | raw DN ~1000–4000 fed to models; Lab 6 reported test accuracy exactly 0.10000000149011612 = 1/10 = chance | Part 2 |
# | `reproject(...)` with no `src_nodata`/`dst_nodata` | pixels outside the CORINE extent filled with `0`, which is not the declared nodata (−128); "no data" became ambiguous between two values | Part 6 |
# | resampling choice never justified or tested | bilinear/cubic on a label raster invents class codes | Part 6, Exercise 3 |
# | `successful_results[2]` | `IndexError` for anyone with 1 or 2 processed tiles; and index 2 is a different scene depending on sort order | Part 1 |
# | `SKIP_EXISTING = True` with no cache key | stale outputs from an older, different request reused silently; this is what produced the "0 successful" commit | Part 4 |
# | band descriptions written but never read back; `src.read(3)` assumed to be Red | a reordered band list silently swaps R and B in every figure and every patch | Part 3 |
# | band pattern `**/R10m/*_B02_10m.jp2` only | pre-Processor-1.05 products have no `R10m/` directory; the old cell printed `⚠ B02 not found` and returned `None` | Part 3 |
# | stacked GeoTIFF written with `nodata=None` | Lab 4.2 had to guess that 0 meant no-data, and discarded real dark water and shadow | Part 4 |
# | `valid_mask = (data >= 1) & (data <= 44)` | class 44 "Sea and ocean", 43 "Estuaries", 30 "Beaches, dunes, sands" become supervised classes in a coastal tile | Part 5 |
# | `os.environ['MPLCONFIGDIR']` set *after* `import matplotlib` | the `/tmp/matplotlib-…` warning was in committed output for three notebooks | Part 0 |

# %% [markdown]
# ## Part 0 — Setup
#
# `MPLCONFIGDIR` is set **before** matplotlib is imported, and it points at scratch,
# not at `/tmp`. One `rng` for the whole notebook; anything that needs randomness
# receives it.

# %%
from eo_course import paths

print(paths.describe())

# %%
import os
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = paths.matplotlib_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402  (must follow MPLCONFIGDIR)
import rasterio  # noqa: E402
from rasterio.windows import Window, transform as window_transform  # noqa: E402
from rasterio.warp import reproject, transform  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from eo_course import radiometry, labels as lab_mod, gates, results  # noqa: E402

SEED = 20180401
rng = np.random.default_rng(SEED)
print(f"seed = {SEED}   (one rng for the whole notebook; pass it down)")
print(f"rasterio {rasterio.__version__}   numpy {np.__version__}")

# %% [markdown]
# ## Part 1 — The Lab 3.1 hand-off, read by key
#
# Lab 3.1 wrote `manifest.json` into `paths.results_dir()`. It records the tile, the
# four sensing dates, the product IDs, the band list, and whether the run was
# synthetic. Read it. Do not re-glob scratch and hope you find the same four granules
# you chose — that is how a team ends up preprocessing a scene nobody selected.
#
# `paths.require_existing` is used rather than a bare `Path(...)`, so a missing
# artifact names the lab that should have produced it instead of surfacing as a
# mystery `FileNotFoundError`.

# %%
MANIFEST_PATH = paths.require_existing(
    paths.results_dir() / "manifest.json",
    "Lab 3.1 manifest (Lab 3.1, Part 13 writes it to paths.results_dir())",
)
lab3 = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
print(f"  manifest      : {MANIFEST_PATH}")
print(f"  tile          : {lab3['tile']}")
print(f"  bands claimed : {lab3['bands']}")
print(f"  quantisation  : {lab3['quantification_value']}")
print(f"  mode          : {lab3['mode']}   synthetic={lab3['synthetic']}")
if lab3["synthetic"]:
    print("  WARNING: Lab 3.1 ran in demo mode. This preprocessing is a dry run on")
    print("           synthetic stubs and is not submittable course data.")

# %% [markdown]
# ### `successful_results[2]` — why a positional index is a bug, not a style choice
#
# The 2025/26 notebook picked the scene to inspect with `successful_results[2]`. Two
# separate problems:
#
# 1. **It assumes a length nobody controls.** A team with two processed tiles gets
#    `IndexError`, and the notebook dies at the *verification* cell — after an hour of
#    processing — which is the worst possible place to fail.
# 2. **It assumes an order nobody set.** The list came out of a `glob`, then a filter.
#    Sorting by date, by cloud cover, or not at all gives a different scene at index 2.
#    The number you report is then a function of Python's iteration order.
#
# The fix is not `successful_results[0]`. The fix is a dict keyed by something the
# student chose and can name. Below, the same four results are ordered two ways and
# index 2 is shown to be a different scene each time.

# %%
scenes = sorted(lab3["scenes"], key=lambda s: s["sensing_date"])
by_date = {s["sensing_date"]: s for s in scenes}          # keyed: stable, nameable
by_cloud = sorted(scenes, key=lambda s: s["reported_cloud_pct"])

print(f"  {len(scenes)} scene(s) in the manifest\n")
print(f"  {'index':<6} {'by date':<14} {'by reported cloud':<14}  same scene?")
for i in range(max(len(scenes), 1)):
    a = scenes[i]["sensing_date"] if i < len(scenes) else "IndexError"
    b = by_cloud[i]["sensing_date"] if i < len(by_cloud) else "IndexError"
    print(f"  [{i}]    {a:<14} {b:<14}  {a == b}")
print("\n  Index 2 is not 'the third scene'; it is 'whatever the sort happened to put third'.")
short = scenes[:2]
try:
    short[2]
except IndexError as exc:
    print(f"  and on a two-tile run, `successful_results[2]` -> IndexError: {exc}")

# %%
# TODO(you): choose the scene you will inspect by DATE, not by position.
# Falsifiable success condition: the cell below raises unless INSPECT_DATE is exactly
# one of the keys of `by_date`. Leaving the placeholder raises, so an unfilled TODO
# cannot pass. If you write an index instead of a date, the check still fails -- and
# it fails for a team whose scene list is shorter than yours, which is the 2025/26 bug.
INSPECT_DATE = "TODO(you): one of " + ", ".join(sorted(by_date))

if INSPECT_DATE not in by_date:
    raise KeyError(
        f"INSPECT_DATE={INSPECT_DATE!r} is not a sensing date in {MANIFEST_PATH.name}. "
        f"Pick one of {sorted(by_date)}."
    )
INSPECT = by_date[INSPECT_DATE]
INSPECT_SAFE = Path(INSPECT["safe_dir"])
print(f"  inspecting {INSPECT_DATE}: {INSPECT_SAFE.name}")
print(f"  reported scene-level cloud {INSPECT['reported_cloud_pct']:.1f}%, "
      f"AOI cloud {INSPECT['aoi_cloud_mask_pct']}")

# %% [markdown]
# ## Part 2 — Units: what a Level-2A band actually stores
#
# Lab 3.1 stated the definition. Here you *measure* it on your own pixels, because a
# definition you have never seen in your own data is a definition you will not apply.
#
# Sentinel-2 L2A stores an integer quantised reflectance:
#
# $$\text{DN} = \operatorname{round}\!\left(\frac{\rho}{\text{QUANTIFICATION\_VALUE}}\right),
# \qquad \text{QUANTIFICATION\_VALUE} = 10000$$
#
# DN 4000 is 40 % reflectance. The 2025/26 Lab 4.2 asserted the imagery was "12-bit,
# values 0–4095" — that is L1C's radiometric resolution, not L2A storage — and then
# applied a percentile stretch, so the ÷10000 step appeared nowhere in the course.
#
# ### Exercise 1 — predict before you run
#
# **Write your answers down before running the next two cells.**
#
# > **P1.** Over a 512 × 512 window of the scene you selected:
# >
# > 1. What is the **maximum DN** in the B08 (Narrow-NIR) band? Give an order of
# >    magnitude, not a guess to three decimals.
# > 2. After `radiometry.dn_to_reflectance`, what is the **maximum of the same
# >    pixels**?
# > 3. How many of your bands have a **DN maximum greater than 1.0**? (0, 1, some, or
# >    all of them.)
# >
# > Question 3 is the one that matters: `radiometry.assert_reflectance_range` fails on
# > any array whose max exceeds 1, so it is a test for "did someone divide by 10000".

# %%
PREDICTIONS = {
    # TODO(you): fill in BEFORE running the next two cells. Leaving a value as None is
    # scored as "not attempted"; a wrong value with a correct explanation is scored
    # higher than a lucky guess with no explanation.
    "dn_max_B08": None,          # a number, e.g. 500 or 5000
    "refl_max_B08": None,        # a number in [0, 1]
    "bands_with_dn_max_gt_1": None,   # an integer, or "all"
    "why": "TODO(you): name the mechanism behind each disagreement.",
    "nodata_invented": {"pct": None, "why": "TODO(you): finish the sentence."},
    "cubic_labels": {"n_codes_after": None, "impossible_codes": None,
                     "why": "TODO(you): name the mechanism, not the outcome."},
}

# %%
def band_path(safe_dir, band, resolution_m):
    """Locate one band raster, tolerating both SAFE layouts.

    Processor >= 1.05 puts bands in `IMG_DATA/R10m/`, `R20m/`, `R60m/`. Older
    products put them flat in `IMG_DATA/` with no resolution suffix. The 2025/26
    notebook only tried the first pattern, printed `⚠ B02 not found` and returned
    `None` -- Lab 3.1 handled both, Lab 4.1 did not, with no explanation.
    """
    pats = (
        f"**/R{resolution_m}m/*_{band}_{resolution_m}m.jp2",
        f"**/IMG_DATA/*_{band}.jp2",
        f"**/*_{band}_*.jp2",
    )
    for pat in pats:
        hits = sorted(Path(safe_dir).glob(pat))
        if hits:
            return hits[0], pat
    return None, None

# %%
# `rasterio.open` opens rasters, not product directories. The 2025/26 notebook wrapped
# every function in `try/except Exception: print(...); return None`, so an error of
# exactly this kind became a printed line plus a `None` that the next cell indexed
# into. Here the exception is shown once, deliberately, and then we do the right thing.
BAND_WINDOW = 512
try:
    rasterio.open(INSPECT_SAFE)
    print("  UNREACHABLE: rasterio opened a product directory as a raster")
except Exception as exc:
    print(f"  rasterio.open({INSPECT_SAFE.name}) -> {type(exc).__name__}")
    print("  expected: a .SAFE is a directory of per-band rasters, not one raster.")

first_band, _ = band_path(INSPECT_SAFE, "B02", 10)
if first_band is None:
    raise FileNotFoundError(
        f"no B02 raster under {INSPECT_SAFE}. That directory holds "
        f"{len(list(Path(INSPECT_SAFE).rglob('*.jp2')))} .jp2 files; re-run Lab 3.1 Part 10."
    )
print(f"  reference band raster: {first_band.name}")
with rasterio.open(first_band) as src:
    print(f"  grid    : {src.width} x {src.height} @ {src.res[0]:.0f} m")
    print(f"  crs     : {src.crs}   (read from the file, never derived from the MGRS letter)")
    print(f"  nodata  : {src.nodata!r}   dtype {src.dtypes[0]}")
    print(f"  bounds  : {src.bounds}")
    REF_H, REF_W, REF_CRS, REF_TF = src.height, src.width, src.crs, src.transform

# %%
win = Window(REF_W // 3, REF_H // 3, BAND_WINDOW, BAND_WINDOW)
DN_STATS = {}
for bname in lab3["bands"]:
    res = 10 if bname in ("B02", "B03", "B04", "B08") else 20
    p, _ = band_path(INSPECT_SAFE, bname, res)
    if p is None:
        DN_STATS[bname] = None
        continue
    with rasterio.open(p) as s:
        dn = s.read(1, window=win, out_shape=(1, BAND_WINDOW, BAND_WINDOW)).astype(np.float64)
    DN_STATS[bname] = {"dn_min": float(dn.min()), "dn_max": float(dn.max()),
                       "dn_mean": float(dn.mean())}

print(f"  window {win.width}x{win.height} at row {int(win.row_off)} col {int(win.col_off)}\n")
print(f"  {'band':<6} {'DN min':>9} {'DN max':>9} {'DN mean':>9}   reflectance min/max")
for b, st in DN_STATS.items():
    if st is None:
        print(f"  {b:<6}  MISSING from product")
        continue
    rmin = radiometry.dn_to_reflectance(st["dn_min"])
    rmax = radiometry.dn_to_reflectance(st["dn_max"])
    print(f"  {b:<6} {st['dn_min']:>9.0f} {st['dn_max']:>9.0f} {st['dn_mean']:>9.0f}   "
          f"{float(rmin):.4f} / {float(rmax):.4f}")

# %%
b08 = DN_STATS.get("B08")
n_gt1 = sum(1 for st in DN_STATS.values() if st and st["dn_max"] > 1.0)
print("  PREDICTED vs ACTUAL — units")
if b08:
    print(f"    P1.1 DN max B08         : predicted {PREDICTIONS['dn_max_B08']!r}   "
          f"actual {b08['dn_max']:.0f}")
    print(f"    P1.2 reflectance max B08: predicted {PREDICTIONS['refl_max_B08']!r}   "
          f"actual {float(radiometry.dn_to_reflectance(b08['dn_max'])):.4f}")
else:
    print("    P1.1/P1.2: B08 is absent from this product, so there is nothing to compare.")
print(f"    P1.3 bands with DN max > 1: predicted "
      f"{PREDICTIONS['bands_with_dn_max_gt_1']!r}   actual {n_gt1} of {len(DN_STATS)}")
print("    Record predicted-vs-actual and explain any disagreement in one sentence.")
print("    If you predicted a small number for P1.1, the mechanism to name is")
print("    QUANTIFICATION_VALUE. If you predicted 0 for P1.3, name which downstream")
print("    consumer of yours would then have divided twice.")

# %% [markdown]
# ### The assert that would have caught 2025/26
#
# `radiometry.assert_reflectance_range` fails loudly if an array claimed to be
# reflectance is not. It is not a style check. Lab 5.1 multiplied an already
# percentile-normalised array by `1e-4`, so the network saw values of order 1e-5;
# BatchNorm absorbed it and nothing failed, which is precisely why it survived a full
# academic year. Lab 6 fed a percentile stretch in [0, 1] against Prithvi means near
# 1000 and reported test accuracy of exactly 1/10.

# %%
demo_dn = np.array([[0, 120, 1500, 4000, 9000]], dtype=np.uint16)
rho = radiometry.dn_to_reflectance(demo_dn)
radiometry.assert_reflectance_range(rho, "rho (correctly divided)")
print(f"  DN          {demo_dn[0].tolist()}")
print(f"  reflectance {[round(float(v), 4) for v in rho[0]]}   assert passed")
try:
    radiometry.assert_reflectance_range(demo_dn, "raw DN mislabelled as reflectance")
    print("  UNREACHABLE: the assert did not fire")
except AssertionError as exc:
    print(f"  fired, as intended -> {str(exc).splitlines()[0]}")

# %% [markdown]
# ## Part 3 — `bands.json`: write it, then read it back
#
# The 2025/26 notebook wrote band descriptions into the GeoTIFF
# (`dst.set_band_description(i, band_name)`) and then, in the very next section, read
# `src.read(3)` with the comment `# B04 (Red)`. That is correct only because the band
# dict happened to be built in B02, B03, B04, B08 order. Nothing verified it. Add one
# band to the list and every figure and every patch silently swaps Red for Blue.
#
# So the band table is an **artifact** here, not a comment: `bands.json` is written
# with name, wavelength, native resolution and the role the band plays in this
# classification, and then read back and used to index the stack **by name**.
#
# The `role` field is not decoration. B11 and B12 (SWIR) are what separate CORINE
# arable land from forest and from bare soil; a 4-band configuration does not "lose a
# little detail", it removes the axis on which that separation lives.

# %%
BAND_CATALOG = {
    "B02": {"wavelength_nm": 490, "resolution_m": 10, "role": "Blue: aerosol/water scattering; separates urban fabric from vegetation"},
    "B03": {"wavelength_nm": 560, "resolution_m": 10, "role": "Green: peak vegetation reflectance; chlorophyll"},
    "B04": {"wavelength_nm": 665, "resolution_m": 10, "role": "Red: chlorophyll absorption; the NIR/Red pair is NDVI"},
    "B08": {"wavelength_nm": 842, "resolution_m": 10, "role": "Narrow-NIR: biomass and canopy structure"},
    "B11": {"wavelength_nm": 1610, "resolution_m": 20, "role": "SWIR-1: moisture, lignin; separates arable from forest"},
    "B12": {"wavelength_nm": 2190, "resolution_m": 20, "role": "SWIR-2: soil mineralogy and moisture; bare soil vs burnt"},
}

# %%
BANDS = list(lab3["bands"])
missing_from_catalog = [b for b in BANDS if b not in BAND_CATALOG]
if missing_from_catalog:
    raise KeyError(f"bands {missing_from_catalog} are not in BAND_CATALOG; add them with a "
                   "wavelength, native resolution and role rather than guessing an index")

bands_doc = {
    "schema": "lab4-bands/1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "scene_inspected": INSPECT_DATE,
    "units": "surface_reflectance",
    "quantization_value_applied": radiometry.QUANTIFICATION_VALUE,
    "stack_nodata": -1.0,
    "order": BANDS,
    "bands": [{"name": b, **BAND_CATALOG[b]} for b in BANDS],
}
BANDS_PATH = paths.aligned_dir() / "bands.json"
paths.ensure(BANDS_PATH.parent)
BANDS_PATH.write_text(json.dumps(bands_doc, indent=2), encoding="utf-8")
print(f"  wrote {BANDS_PATH} ({BANDS_PATH.stat().st_size} bytes)")

# %%
bands_back = json.loads(paths.require_existing(BANDS_PATH, "lab4 bands.json")
                        .read_text(encoding="utf-8"))
assert bands_back["order"] == BANDS, "bands.json order does not match the stack order"
BAND_INDEX = {b["name"]: i for i, b in enumerate(bands_back["bands"])}
print(f"  read back: order={bands_back['order']}  units={bands_back['units']}  "
      f"nodata={bands_back['stack_nodata']}")
print(f"  name -> stack band index: {BAND_INDEX}")
print("  Every later cell indexes by name through this dict, never by a literal integer.")

# %% [markdown]
# ## Part 4 — Stacking, with a cache key
#
# ### The staleness trap
#
# The old notebook had `SKIP_EXISTING = True` and nothing else. If the output file
# existed, the tile was skipped. There was no record of *what produced it*, so a
# student who changed the band list, swapped the CORINE vintage, or re-downloaded a
# different date silently trained on the previous request's pixels. That flag is
# removed.
#
# In its place: a **sidecar `manifest.json` per output**, recording source paths, band
# list, CRS, shape, units and nodata. Reuse is allowed only when every one of those
# fields agrees with what the current code would produce. Disagreement is printed
# field by field, and the output is rewritten.

# %%
SIDECAR_KEYS = ("sources", "bands", "crs", "shape", "units", "nodata", "grid",
                "resampling", "label_source_shape", "label_source_bounds")


def sidecar_of(out_path):
    return Path(out_path).with_suffix("").with_suffix(".manifest.json")


def write_sidecar(out_path, info):
    sidecar_of(out_path).write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")


def reuse_reasons(out_path, info):
    """Return the list of fields that disagree. Empty list means reuse is safe."""
    sc = sidecar_of(out_path)
    if not Path(out_path).exists():
        return ["output does not exist"]
    if not sc.exists():
        return ["no sidecar: written by older code, or by hand"]
    old = json.loads(sc.read_text(encoding="utf-8"))
    return [k for k in SIDECAR_KEYS if old.get(k) != info.get(k)]

# %%
NODATA = float(bands_back["stack_nodata"])   # read back from bands.json, not re-typed


def corine_extent(crs_src):
    """The S2 grid's four corners, reprojected into `crs_src`. Returns a bbox tuple."""
    xs, ys = transform(REF_CRS, crs_src,
                       [REF_TF.c, REF_TF.c + REF_W * REF_TF.a,
                        REF_TF.c, REF_TF.c + REF_W * REF_TF.a],
                       [REF_TF.f, REF_TF.f, REF_TF.f + REF_H * REF_TF.e,
                        REF_TF.f + REF_H * REF_TF.e])
    return min(xs), min(ys), max(xs), max(ys)


def corine_window(label_path, n=600):
    """A window of the label raster that actually overlaps the S2 tile *and* has data.

    Taking `Window(0, 0, n, n)` on the full European CORINE grid samples the north
    Atlantic, which is all nodata, and every demo below then prints zeros for the wrong
    reason. Candidate offsets are scanned and the one with the most valid pixels wins.
    """
    with rasterio.open(label_path) as src:
        n = max(8, min(int(n), src.width, src.height))   # small rasters: clamp, don't lie
        left, bottom, right, top = corine_extent(src.crs)
        c0 = int((left - src.bounds.left) / src.res[0])
        r0 = int((src.bounds.top - top) / src.res[1])
        c0 = max(0, min(c0, src.width - n))
        r0 = max(0, min(r0, src.height - n))
        best, best_valid = Window(c0, r0, n, n), -1
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                w = Window(min(max(c0 + dc * n, 0), src.width - n),
                           min(max(r0 + dr * n, 0), src.height - n), n, n)
                v = int((src.read(1, window=w) != CORINE_NODATA).sum())
                if v > best_valid:
                    best, best_valid = w, v
                if best_valid > 0.8 * n * n:
                    break
        data = src.read(1, window=best)
        return best, window_transform(best, src.transform), data


def stack_scene(safe_dir, scene_id, out_dir):
    """Write one scene's band stack in reflectance units, with nodata and descriptions.

    Every band is resampled onto the B02 grid, so a 6-band configuration mixing 10 m
    and 20 m rasters is legal; the 20 m values are duplicated by nearest, not
    interpolated, because interpolation would invent pixel values that were never
    measured.
    """
    out = Path(out_dir) / f"s2_{scene_id}.tif"
    sources = {}
    for b in BANDS:
        p, _ = band_path(safe_dir, b, BAND_CATALOG[b]["resolution_m"])
        if p is None:
            raise FileNotFoundError(f"{scene_id}: band {b} not found under {safe_dir}")
        sources[b] = str(p)
    info = {"sources": sources, "bands": BANDS, "crs": str(REF_CRS), "shape": [REF_H, REF_W],
            "units": "surface_reflectance", "nodata": NODATA,
            "grid": [float(v) for v in REF_TF], "scene_id": scene_id,
            "resampling": "nearest",
            "quantization_value": radiometry.QUANTIFICATION_VALUE}
    return out, info

# %%
def write_stack(safe_dir, scene_id, out_dir):
    out, info = stack_scene(safe_dir, scene_id, out_dir)
    reasons = reuse_reasons(out, info)
    if not reasons:
        print(f"  [{scene_id}] reuse {out.name} (sidecar agrees on all {len(SIDECAR_KEYS)} fields)")
        return out, info, "reused"

    print(f"  [{scene_id}] computing {out.name} -- "
          + ("; ".join(reasons[:3]) if reasons != ["output does not exist"] else "new output"))
    prof = {"driver": "GTiff", "height": REF_H, "width": REF_W, "count": len(BANDS),
            "dtype": "float32", "crs": REF_CRS, "transform": REF_TF, "nodata": NODATA,
            "compress": "lzw", "tiled": True, "blockxsize": 256, "blockysize": 256}
    # Band by band, into an open destination, instead of building a (C, H, W) array
    # first. Lab 4.2's old code did `src.read()` on a whole 10980x10980x4 scene and
    # then a float32 copy of it -- ~2.9 GB resident, on a login node.
    with rasterio.open(out, "w", **prof) as dst:
        for i, b in enumerate(BANDS, start=1):
            with rasterio.open(info["sources"][b]) as s:
                if s.crs != REF_CRS:
                    raise ValueError(f"{scene_id}/{b}: crs {s.crs} != reference {REF_CRS}")
                m = s.read(1, out_shape=(1, REF_H, REF_W), resampling=Resampling.nearest,
                           masked=True)
            rho_b = radiometry.dn_to_reflectance(m.filled(0))
            rho_b = np.where(np.ma.getmaskarray(m), NODATA, rho_b)   # nodata stays nodata
            dst.write(rho_b.astype(np.float32), i)
            dst.set_band_description(i, b)
            del m, rho_b
    write_sidecar(out, info)
    return out, info, "written"

# %% [markdown]
# ### Where the batch belongs
#
# A full granule is 10980 × 10980 per band. Reading six of them, converting to
# float32 and writing an LZW tile is roughly 2 GB of traffic per scene. On a JURECA
# **login node** that is the kind of I/O that gets a training project's session
# throttled, and the 2025/26 notebooks did all of it there. The `DEMO_SCENES` cap
# below keeps the notebook interactive; the real run belongs on a `dc-cpu` compute
# node, e.g.
#
# ```bash
# srun --partition=dc-cpu --cpus-per-task=8 --mem-per-cpu=4G \
#      jupyter execute --inplace lab4_1_data_preprocessing.ipynb
# ```
#
# There is no `slurm/lab4_preprocess.sbatch` in the repository yet — `slurm/`
# currently holds only the two training scripts. Writing that batch file is a
# legitimate Lab 4 extension and is worth marks if you submit it with the notebook.

# %%
DEMO_SCENES = len(scenes)   # TODO(you): lower this to 1 if you are on a login node
SCENES_TO_PROCESS = scenes[:DEMO_SCENES]
print(f"  processing {len(SCENES_TO_PROCESS)} of {len(scenes)} scene(s) -> {paths.aligned_dir()}")
STACKS = {}
for s in SCENES_TO_PROCESS:
    sid = f"{lab3['tile']}_{s['sensing_date'].replace('-', '')}"
    out, info, status = write_stack(Path(s["safe_dir"]), sid, paths.aligned_dir())
    STACKS[sid] = {"stack": out, "info": info, "status": status, "date": s["sensing_date"],
                   "safe": s["safe_dir"]}
print()
for sid, r in STACKS.items():
    print(f"  {sid:<20} {r['status']:<8} {r['stack'].name}")

# %% [markdown]
# ### Negative control — prove the cache key can fail
#
# A check you have never seen reject something is a check you have not verified. The
# cell below rewrites one sidecar so it claims a **different band list** than the file
# on disk actually has, then asks `reuse_reasons` whether reuse is safe. This is
# exactly the state a student creates by editing `BANDS` after a run.

# %%
victim = next(iter(STACKS.values()))
sc = sidecar_of(victim["stack"])
tampered = json.loads(sc.read_text(encoding="utf-8"))
tampered["bands"] = ["B02", "B03", "B04"]
tampered["units"] = "digital_number"
sc.write_text(json.dumps(tampered, indent=2, sort_keys=True), encoding="utf-8")

bad = reuse_reasons(victim["stack"], victim["info"])
print(f"  tampered sidecar for {victim['stack'].name}")
print(f"  reuse_reasons -> {bad}")
if not bad:
    raise AssertionError("the cache key accepted a sidecar that lies about bands and units")
write_sidecar(victim["stack"], victim["info"])
print(f"  restored; reuse_reasons now -> {reuse_reasons(victim['stack'], victim['info'])}")
print("  Write one sentence on what would have happened with the old SKIP_EXISTING flag.")

# %% [markdown]
# ## Part 5 — CORINE: what the raster actually is
#
# Three facts, none of which the 2025/26 notebook stated correctly:
#
# * The U2018 44-class raster stores no-data as **−128** in an int8 band. Not 0, not
#   48. The old notebook pasted a `CORINE_CLASSES` dict containing `48: "No data"` and
#   printed "Defined 45 CORINE land cover classes" while its own cell 11 output said
#   `NoData value: -128.0`. Three copies of that dict existed across the course and
#   all three carried the phantom class.
# * The taxonomy is `eo_course.labels.CORINE_CLASSES`. You do not paste it again.
# * `1 <= label <= 44` is not a validity test. It keeps 44 "Sea and ocean",
#   43 "Estuaries", 39 "Intertidal flats" and 30 "Beaches, dunes, sands" as supervised
#   classes. In a coastal tile those become real training samples; in an inland tile
#   they are zero-support classes that still occupy a confusion-matrix row and a term
#   in your macro-average.

# %%
CORINE_CANDIDATES = [
    paths.scratch_root() / "CORINE" / "u2018_clc2018_v2020_20u1_raster100m" / "DATA"
    / "U2018_CLC2018_V2020_20u1.tif",
    paths.data_root() / "CORINE" / "U2018_CLC2018_V2020_20u1.tif",
]
CORINE_PATH = next((p for p in CORINE_CANDIDATES if p.exists()), None)
print(f"  taxonomy in eo_course.labels: {len(lab_mod.CORINE_CLASSES)} classes, "
      f"no-data {lab_mod.CORINE_NODATA}")
print(f"  default supervised set excludes water/sediment "
      f"{list(lab_mod.WATER_AND_SEDIMENT_CODES)} -> {len(lab_mod.valid_class_codes())} codes")
print(f"  CORINE raster: {CORINE_PATH if CORINE_PATH else 'NOT FOUND -> synthetic fallback'}")

# %%
def build_synthetic_corine(out_path):
    """A 100 m EPSG:3035 label raster covering the S2 grid, for when CORINE is absent.

    STAMPED SYNTHETIC. It exists so the reprojection mechanics in Part 6 are
    demonstrable on a laptop or before the EEA download finishes. It is not course
    data and the gate board fails the run while it is in use.
    """
    left, bottom, right, top = corine_extent("EPSG:3035")
    pad = 20000.0   # 200 px at 100 m: the real EEA raster is 65000 x 46000, so a
                    # generous surround is realistic and leaves room for the warp demos
    left, right, bottom, top = left - pad, right + pad, bottom - pad, top + pad
    w = int(math.ceil((right - left) / 100.0))
    h = int(math.ceil((top - bottom) / 100.0))
    codes = [1, 12, 23, 26, 41]
    coarse = rng.integers(0, len(codes), size=(max(2, h // 9), max(2, w // 9)))
    fy = np.clip((np.arange(h) / h * coarse.shape[0]).astype(int), 0, coarse.shape[0] - 1)
    fx = np.clip((np.arange(w) / w * coarse.shape[1]).astype(int), 0, coarse.shape[1] - 1)
    arr = np.array(codes)[coarse[np.ix_(fy, fx)]].astype("int16")
    arr[: max(1, h // 5), :] = lab_mod.CORINE_NODATA      # a nodata strip, on purpose
    tf = rasterio.transform.from_origin(left, top, 100.0, 100.0)
    with rasterio.open(out_path, "w", driver="GTiff", height=h, width=w, count=1,
                       dtype="int16", crs="EPSG:3035", nodata=lab_mod.CORINE_NODATA,
                       transform=tf, compress="lzw", tiled=True) as dst:
        dst.write(arr, 1)
    return out_path


if CORINE_PATH is None:
    CORINE_PATH = build_synthetic_corine(paths.aligned_dir() / "CORINE_synthetic.tif")
    print(f"  built synthetic label raster {CORINE_PATH.name} "
          f"({CORINE_PATH.stat().st_size / 1e6:.2f} MB) -- NOT course data")

# %%
with rasterio.open(CORINE_PATH) as src:
    print(f"  file      : {src.name}")
    print(f"  size      : {src.width} x {src.height} @ {src.res[0]:.0f} m")
    print(f"  crs       : {src.crs}")
    print(f"  nodata    : {src.nodata!r}   dtype {src.dtypes[0]}")
    print(f"  bounds    : {src.bounds}")
    CORINE_NODATA = int(src.nodata) if src.nodata is not None else lab_mod.CORINE_NODATA
if CORINE_NODATA != lab_mod.CORINE_NODATA:
    print(f"  WARNING: this raster declares nodata {CORINE_NODATA}, but the course "
          f"constant is {lab_mod.CORINE_NODATA}. Read yours from the file; do not assume.")

# %% [markdown]
# ## Part 6 — Reprojection: the mechanism, then the damage
#
# CORINE is 100 m in EPSG:3035 (ETRS89-LAEA Europe, a Lambert azimuthal
# equal-area projection chosen so that *areas* are comparable across Europe).
# Sentinel-2 is 10 m in a UTM projection, where *angles and shapes* are preserved and
# areas are not. Your model needs one grid. The S2 grid is the only sensible choice
# because it is the grid the features live on.
#
# Two things can go wrong, and the 2025/26 notebook did both.
#
# ### Exercise 2 — how much data does a careless warp invent?
#
# The old call was:
#
# ```python
# reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
#           src_transform=..., src_crs=..., dst_transform=..., dst_crs=...,
#           resampling=Resampling.nearest)
# ```
#
# No `src_nodata`, no `dst_nodata`. GDAL therefore treats −128 as an ordinary class
# value and resamples it like one, and destination pixels that receive **no source
# data at all** keep whatever the newly created file was initialised to: `0`. Since
# CORINE has no class 0, every such pixel is invented information — and it is invented
# as *the same value a genuinely dark surface would have* if you later store
# reflectance.
#
# > **P2.** Warp a 300 × 300 CORINE window into a grid shifted 180 px down-right, so a
# > large part of the destination receives no source data. **What percentage of the
# > destination pixels does the naive warp turn into `0` that the correct warp marks
# > as nodata?** Give a percentage. Then finish the sentence: a model trained on those
# > pixels has been taught to predict …

# %%
def warp_demo(label_path, n=300, shift=180):
    """Naive vs nodata-aware warp of the same window into a shifted grid.

    The source is the window read into an array with the window's own transform, so no
    `src_window` bookkeeping can drift. The two warps differ in exactly one thing:
    whether nodata is declared on the source and on the destination.
    """
    _win, src_tf, data = corine_window(label_path, n=n)
    with rasterio.open(label_path) as src:
        # Translation in WORLD metres, composed on the LEFT. `src_tf.translation(dx, dy)`
        # would translate the *pixel* arguments first, i.e. by dx pixels, and silently
        # move the grid by dx * res metres. That is a real rasterio footgun, and it makes
        # a warp return all-nodata while looking like it ran.
        dst_tf = rasterio.Affine.translation(shift * src.res[0], -shift * src.res[1]) * src_tf
        prof = {"driver": "GTiff", "height": n, "width": n, "count": 1,
                "dtype": src.dtypes[0], "crs": src.crs, "transform": dst_tf}

        naive_path = paths.aligned_dir() / "_warp_naive.tif"
        with rasterio.open(naive_path, "w", **prof) as dst:      # no nodata declared
            reproject(source=data, destination=rasterio.band(dst, 1),
                      src_transform=src_tf, src_crs=src.crs,
                      dst_transform=dst_tf, dst_crs=src.crs,
                      resampling=Resampling.nearest)
        with rasterio.open(naive_path) as r:
            naive = r.read(1)

        good_path = paths.aligned_dir() / "_warp_nodata_aware.tif"
        with rasterio.open(good_path, "w", nodata=CORINE_NODATA, **prof) as dst:
            reproject(source=data, destination=rasterio.band(dst, 1),
                      src_transform=src_tf, src_crs=src.crs, src_nodata=CORINE_NODATA,
                      dst_transform=dst_tf, dst_crs=src.crs, dst_nodata=CORINE_NODATA,
                      resampling=Resampling.nearest)
        with rasterio.open(good_path) as r:
            good = r.read(1)
    return naive, good


NAIVE, GOOD = warp_demo(CORINE_PATH)
invented = int(((NAIVE == 0) & (GOOD == CORINE_NODATA)).sum())
n_px = NAIVE.size
pct_invented = 100.0 * invented / n_px
print(f"  destination {NAIVE.shape[0]}x{NAIVE.shape[1]} = {n_px} px")
print(f"  naive warp: {(NAIVE == 0).sum()} px are 0;   nodata-aware warp: "
      f"{(GOOD == CORINE_NODATA).sum()} px are {CORINE_NODATA}")
print(f"  pixels where the naive warp INVENTED a value the correct warp calls nodata: "
      f"{invented} ({pct_invented:.1f}%)")

# %%
print("  PREDICTED vs ACTUAL — invented nodata")
print(f"    predicted % invented : {PREDICTIONS['nodata_invented']['pct']!r}")
print(f"    actual % invented    : {pct_invented:.1f}%")
print(f"    naive unique codes   : {sorted(np.unique(NAIVE).tolist())}")
print(f"    correct unique codes : {sorted(np.unique(GOOD).tolist())}")
print("    The naive output contains a code that exists in neither raster. Name it.")
print("    Also count how many naive pixels still carry the raw", CORINE_NODATA,
      f"-> {(NAIVE == CORINE_NODATA).sum()}. Nearest propagated it as if it were a class.")

# %% [markdown]
# ### Exercise 3 — what interpolating a label raster does
#
# Nearest neighbour is not a "quality" choice for labels, it is the only choice that
# preserves the *type*. Bilinear and cubic compute weighted averages of neighbouring
# **code numbers**. Class codes are not numbers — they are identifiers that happen to
# be written as digits. Averaging 321 and 333 gives 327, which is a real CORINE code
# ("Transitional woodland-shrub" in the full CLC nomenclature) that appears nowhere in
# your window. The warp does not warn. It produces a class you will then report
# per-class recall for.
#
# > **P3.** Take a 60 × 60 CORINE window and warp it **10× upsampled** — the same
# > 100 m → 10 m geometry the production warp uses — with `Resampling.cubic`, then cast
# > the float result back to integer the way a `dtype='int16'` GeoTIFF would.
# >
# > 1. How many **distinct** codes are in the source window?
# > 2. How many distinct codes after the cubic warp and integer cast?
# > 3. How many of those codes were **absent from the source window**?
# > 4. How many are not in `labels.valid_class_codes()` at all?
# >
# > Question 3 is the silent one: an interpolated code can be a *legal* CORINE code that
# > simply is not present in your tile, so nothing downstream flags it — you just start
# > reporting per-class recall for a class that never existed in your data.

# %%
def cubic_demo(label_path, n=60, factor=10):
    """The production geometry — 100 m labels onto a 10 m grid — with the wrong kernel.

    Same CRS, same window, same nodata on both ends. The only variable is the
    resampling method. A 1:1 warp would show nothing, because interpolation only has
    anything to average where two codes meet, and at 100 m those pixels are a small
    minority. Upsampling 10x is exactly what aligning CORINE to Sentinel-2 does.
    """
    with rasterio.open(label_path) as src:
        _win, src_tf, data = corine_window(label_path, n=n)
        src_codes = np.unique(data)
        # Affine.scale multiplies a, b, d -- the pixel steps -- but ALSO c and f, which
        # are the origin coordinates. Composing it on the right therefore moves the
        # window to a point 1000x closer to the origin, i.e. into the Atlantic, and the
        # warp returns nothing but nodata while running without error. Scale the
        # coefficients explicitly and keep the origin.
        dst_tf = rasterio.Affine(src_tf.a / factor, src_tf.b, src_tf.c,
                                 src_tf.d, src_tf.e / factor, src_tf.f)
        m = n * factor
        out_c, out_n = (np.full((m, m), float(CORINE_NODATA), dtype="float32") for _ in range(2))
        for res, dest in ((Resampling.cubic, out_c), (Resampling.nearest, out_n)):
            reproject(source=data.astype("float32"), destination=dest,
                      src_transform=src_tf, src_crs=src.crs, src_nodata=CORINE_NODATA,
                      dst_transform=dst_tf, dst_crs=src.crs, dst_nodata=float(CORINE_NODATA),
                      resampling=res)
    return src_codes, out_c, out_n


SRC_CODES, CUBIC, NEAREST = cubic_demo(CORINE_PATH)
CUBIC_CAST = np.rint(CUBIC).astype("int16")
NEAR_CAST = np.rint(NEAREST).astype("int16")
valid = set(lab_mod.valid_class_codes())
src_set = {int(c) for c in SRC_CODES}
after = [int(c) for c in np.unique(CUBIC_CAST)]
after_near = [int(c) for c in np.unique(NEAR_CAST)]
invented_codes = [c for c in after if c != CORINE_NODATA and c not in src_set]
invalid_codes = [c for c in after if c != CORINE_NODATA and c not in valid]

print("  PREDICTED vs ACTUAL — cubic resampling of a label raster, 100 m -> 10 m")
print(f"    distinct codes in source  : {len(src_set)}  {sorted(src_set)}")
print(f"    predicted after           : {PREDICTIONS['cubic_labels']['n_codes_after']!r}")
print(f"    actual after (cubic)      : {len(after)}  {after[:14]}{' ...' if len(after) > 14 else ''}")
print(f"    actual after (nearest)    : {len(after_near)}  {after_near[:14]}"
      f"{' ...' if len(after_near) > 14 else ''}   <- same window, same grid, same nodata")
print(f"    predicted invented codes  : {PREDICTIONS['cubic_labels']['impossible_codes']!r}")
print(f"    invented by cubic (absent from the source window): {len(invented_codes)} -> "
      f"{invented_codes[:14]}{' ...' if len(invented_codes) > 14 else ''}")
print(f"    invented codes that are not valid CORINE at all  : {len(invalid_codes)} -> "
      f"{invalid_codes[:14]}{' ...' if len(invalid_codes) > 14 else ''}")
print(f"    cubic pixels that are neither a source code nor nodata: "
      f"{100 * float(np.isin(CUBIC_CAST, invented_codes).mean()):.2f}%")
print("    Two different kinds of damage are counted above. Name both, and say which one")
print("    a per-class recall table would silently report as if it were a real class.")

# %% [markdown]
# ### The aligned warp, done once, correctly
#
# `out_profile = src.profile.copy()` then update geometry is the standard rasterio
# idiom, and the classic bug inside it is forgetting to override `dtype` and
# `nodata`. Here both are explicit, the resampling is nearest and named in the
# sidecar, and the destination grid is taken from the S2 stack rather than computed.

# %%
def align_label(label_path, stack_path, out_dir, scene_id):
    out = Path(out_dir) / f"corine_{scene_id}.tif"
    with rasterio.open(stack_path) as s2, rasterio.open(label_path) as lsrc:
        # The source label raster's own shape and bounds go into the key. Without them,
        # replacing the CORINE file with a different one at the same path -- a rebuilt
        # fallback, a new vintage, a re-download -- leaves the sidecar agreeing and the
        # stale warp being reused. That is the SKIP_EXISTING trap again, one level up.
        info = {"sources": {"label": str(label_path), "reference": str(stack_path)},
                "bands": ["CORINE_L3"], "crs": str(s2.crs), "shape": [s2.height, s2.width],
                "units": "corine_level3_code", "nodata": float(CORINE_NODATA),
                "grid": [float(v) for v in s2.transform], "resampling": "nearest",
                "scene_id": scene_id,
                "label_source_shape": [lsrc.height, lsrc.width],
                "label_source_bounds": [float(v) for v in lsrc.bounds]}
    reasons = reuse_reasons(out, info)
    if not reasons:
        print(f"  [{scene_id}] reuse {out.name} (sidecar agrees)")
        return out, "reused"
    print(f"  [{scene_id}] computing {out.name} -- "
          + ("; ".join(reasons[:3]) if reasons != ["output does not exist"] else "new output"))
    with rasterio.open(label_path) as src:
        prof = src.profile.copy()
        prof.update(driver="GTiff", crs=s2.crs, transform=s2.transform,
                    width=s2.width, height=s2.height, count=1, dtype="int16",
                    nodata=CORINE_NODATA, compress="lzw", tiled=True)
        with rasterio.open(out, "w", **prof) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=CORINE_NODATA,
                      dst_transform=s2.transform, dst_crs=s2.crs,
                      dst_nodata=CORINE_NODATA,
                      resampling=Resampling.nearest)
    write_sidecar(out, info)
    return out, "written"

# %%
for sid, r in STACKS.items():
    lab_out, status = align_label(CORINE_PATH, r["stack"], paths.aligned_dir(), sid)
    r["label"] = lab_out
    r["label_status"] = status
print()
print(f"  aligned label rasters in {paths.aligned_dir()}: "
      f"{len(list(paths.aligned_dir().glob('corine_*.tif')))}")

# %% [markdown]
# ## Part 7 — Verification: does the label grid actually match?
#
# Lab 4.2 will read the image and the label raster as if they were the same grid.
# `patches.extract_patches` raises if the shapes or geotransforms differ, which is the
# right behaviour — reading two rasters at mismatched shapes silently pairs pixels
# with the wrong label. But raising after an hour of extraction is a bad experience,
# so the check happens here, on the artifacts, before they are consumed.

# %%
ALIGN_REPORT = []
for sid, r in STACKS.items():
    with rasterio.open(r["stack"]) as im, rasterio.open(r["label"]) as lb:
        same_shape = (im.height, im.width) == (lb.height, lb.width)
        same_crs = im.crs == lb.crs
        same_tf = np.allclose(np.asarray(im.transform), np.asarray(lb.transform), atol=1e-9)
        declared = im.nodata
        # Windowed, not `im.read(1)`: a full 10980x10980 band is ~480 MB of float32,
        # and the 2025/26 notebook did exactly that on a login node, twice per tile.
        probe = Window(max(0, im.width // 2 - 512), max(0, im.height // 2 - 512), 1024, 1024)
        frac_nodata = (float((im.read(1, window=probe) == declared).mean())
                       if declared is not None else None)
    ALIGN_REPORT.append({"scene_id": sid, "same_shape": bool(same_shape),
                         "same_crs": bool(same_crs), "same_transform": bool(same_tf),
                         "img_nodata": None if declared is None else float(declared),
                         "img_nodata_frac": frac_nodata})
    print(f"  {sid:<20} shape={same_shape} crs={same_crs} transform={same_tf} "
          f"nodata={declared!r} "
          f"({frac_nodata:.2%} of band 1)" if frac_nodata is not None else f"  {sid}")

# %%
COVERAGE = []
for sid, r in STACKS.items():
    with rasterio.open(r["label"]) as lb:
        # Decimated read: 1/4 linear resolution is 1/16 the pixels, and class
        # frequencies and coverage fractions are stable under it. A full read of a
        # 10980x10980 label band is ~240 MB per tile per call.
        arr = lb.read(1, out_shape=(1, lb.height // 4, lb.width // 4),
                      resampling=Resampling.nearest)
        nd = int(lb.nodata) if lb.nodata is not None else CORINE_NODATA
    mask = lab_mod.clean_labels(arr, nodata=nd)          # not (1 <= x <= 44)
    present = np.unique(arr[mask])
    counts = lab_mod.class_counts(arr[mask], present)
    order = np.argsort(-counts)
    COVERAGE.append({"scene_id": sid, "labelled_frac": float(mask.mean()),
                     "nodata_frac": float((arr == nd).mean()),
                     "n_codes": int(len(present)),
                     "top": [(int(present[i]), int(counts[i])) for i in order[:5]]})
    top = COVERAGE[-1]["top"]
    print(f"  {sid:<20} supervised {mask.mean():6.2%}  nodata {COVERAGE[-1]['nodata_frac']:6.2%}  "
          f"{len(present)} codes")
    print(f"    top 5: " + ", ".join(f"{c} {lab_mod.class_name(c)[:22]} ({n})" for c, n in top))

# %% [markdown]
# ### Read the numbers before you move on
#
# Three questions, and the answers are in the cell you just ran:
#
# 1. Your `labelled_frac` is the fraction of the tile that will produce a patch at
#    all. If it is low, most of the granule you downloaded is unusable — that is an
#    acquisition decision (Lab 3.1), not a preprocessing one.
# 2. Which class dominates, and does it match what you see in the RGB panel in
#    Part 8? A tile that is 70 % class 12 in January is arable land under a winter
#    crop; the same tile in July is a different surface with the same label.
# 3. Are any of `WATER_AND_SEDIMENT_CODES` present? They are excluded by
#    `valid_class_codes()` by default. Keeping them is legal — pass `exclude=()` — but
#    it is a decision you must record, because "Sea and ocean" as a supervised class
#    in a coastal tile changes your macro-average.

# %% [markdown]
# ## Part 8 — Visualisation, indexed by name
#
# Two defects in the 2025/26 stretch, both of which change what you can see:
#
# * **Pooled percentiles.** `np.percentile(rgb, (2, 98))` over a stacked `(H, W, 3)`
#   array gives Blue, Green and Red *one shared* pair of bounds. Narrow-NIR is far
#   brighter than Blue over vegetation, so a shared stretch drags the bounds up and
#   squeezes Blue toward a constant — destroying the band that most separates urban
#   fabric from vegetation.
# * **`rgb > 0` as the nodata test.** In L2A, DN 0 is a legitimate dark reflectance:
#   deep open water, cast shadow, asphalt. Discarding it brightens exactly the classes
#   you are trying to separate. The sentinel is the declared `nodata`, which is what
#   this notebook wrote into the file.
#
# The bands are selected through `BAND_INDEX`, built from `bands.json`, so reordering
# the band list changes the file and the figure together rather than desynchronising
# them.

# %%
VIS_SID = next(iter(STACKS))
VIS_STACK = STACKS[VIS_SID]["stack"]
VIS_LABEL = STACKS[VIS_SID]["label"]
VIS_WIN = Window(REF_W // 3, REF_H // 3, 512, 512)
RGB_NAMES = [n for n in ("B04", "B03", "B02") if n in BAND_INDEX]
if len(RGB_NAMES) != 3:
    raise KeyError(f"bands.json has no B02/B03/B04; cannot build a natural-colour panel "
                   f"from {list(BAND_INDEX)}")

with rasterio.open(VIS_STACK) as src:
    img = src.read(window=VIS_WIN, masked=True)   # nodata already masked by rasterio
rgb = np.stack([img[BAND_INDEX[n]].filled(np.nan) for n in RGB_NAMES], axis=-1)
with rasterio.open(VIS_LABEL) as lb:
    lab_vis = lb.read(1, window=VIS_WIN)
print(f"  panel {VIS_SID} rows {int(VIS_WIN.row_off)}+{VIS_WIN.height} "
      f"cols {int(VIS_WIN.col_off)}+{VIS_WIN.width}")
print(f"  RGB bands by name: {RGB_NAMES} -> stack indices {[BAND_INDEX[n] for n in RGB_NAMES]}")

# %%
# nanpercentile, not percentile: nodata must not enter the stretch. This is the same
# per-band operation as radiometry.per_band_percentiles, applied to a masked array.
with np.errstate(invalid="ignore"):
    lo = np.nanpercentile(rgb, 2, axis=(0, 1))
    hi = np.nanpercentile(rgb, 98, axis=(0, 1))
print(f"  per-band 2%: {np.round(lo, 4).tolist()}   98%: {np.round(hi, 4).tolist()}")
print("  If these bounds were pooled across bands, the widest band would set all three.")
stretch = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
stretch = np.where(np.isfinite(stretch), stretch, 0.0)

# %%
codes_here = [int(c) for c in np.unique(lab_vis) if c != CORINE_NODATA]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
axes[0].imshow(stretch)
axes[0].set_title(f"{VIS_SID} natural colour (per-band 2-98%)\n{RGB_NAMES}")
axes[1].imshow(np.ma.masked_equal(lab_vis, CORINE_NODATA), cmap="tab20", interpolation="nearest")
axes[1].set_title(f"aligned CORINE, {len(codes_here)} codes\nnearest + nodata on both ends")
axes[2].hist(stretch.reshape(-1, 3), bins=48, range=(0, 1), color=["r", "g", "b"], alpha=0.6)
axes[2].set_title("per-band histograms of the stretched panel")
axes[2].set_xlabel("stretched reflectance")
for a in axes[:2]:
    a.axis("off")
fig.tight_layout()
FIG_PATH = paths.results_dir() / "lab4_1_alignment.png"
paths.ensure(FIG_PATH.parent)
fig.savefig(FIG_PATH, dpi=110)
plt.show()
print(f"  saved {FIG_PATH}")

# %% [markdown]
# ## Part 9 — Gate board
#
# `eo_course.gates` raises rather than warns, because a gate is a claim. The
# performance gates (`gate_beats_baselines`, `gate_above_chance`) belong to Labs 5–7.
# The claims here are about the data: that the units are what you say they are, that
# nothing was invented at the warp, that the label grid matches the image grid, and
# that the artifacts Lab 4.2 needs actually exist.

# %%
def gate(name, ok, detail):
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def stack_units_ok(path, n=1024):
    """Range check on a window, not the whole stack: a full granule is ~1.4 GB."""
    with rasterio.open(path) as src:
        w = Window(max(0, src.width // 2 - n // 2), max(0, src.height // 2 - n // 2), n, n)
        arr = src.read(window=w, masked=True).filled(np.nan)
    try:
        radiometry.assert_reflectance_range(arr[np.isfinite(arr)], path.name)
        return True, "reflectance in [0,1] over a 1024x1024 window"
    except (AssertionError, ValueError) as exc:
        return False, str(exc).splitlines()[0]


board = []
ok_units, why_units = stack_units_ok(STACKS[VIS_SID]["stack"])
board += [
    gate("units_are_reflectance", ok_units, why_units),
    gate("nodata_declared",
         all(a["img_nodata"] is not None for a in ALIGN_REPORT),
         f"declared nodata = {[a['img_nodata'] for a in ALIGN_REPORT]} "
         "(None here is the 2025/26 defect that made Lab 4.2 guess 0)"),
    gate("grids_identical",
         all(a["same_shape"] and a["same_crs"] and a["same_transform"] for a in ALIGN_REPORT),
         f"{sum(a['same_shape'] and a['same_crs'] and a['same_transform'] for a in ALIGN_REPORT)}"
         f"/{len(ALIGN_REPORT)} scenes share shape+crs+transform with their label raster"),
    gate("labels_nearest_only",
         all(json.loads(sidecar_of(r["label"]).read_text(encoding="utf-8"))
             .get("resampling") == "nearest" for r in STACKS.values()),
         f"cubic demo put {len(invalid_codes)} code(s) outside valid_class_codes() and "
         f"{len(invented_codes)} interpolated ones; every production sidecar must record "
         "resampling=nearest"),
]

# %%
def label_codes_in(r, n=1024):
    """Distinct codes in a 1024x1024 window of one aligned label raster."""
    with rasterio.open(r["label"]) as lb:
        w = Window(max(0, lb.width // 2 - n // 2), max(0, lb.height // 2 - n // 2), n, n)
        return set(int(c) for c in np.unique(lb.read(1, window=w)))


LABEL_CODES = {sid: label_codes_in(r) for sid, r in STACKS.items()}
board += [
    gate("no_invented_nodata_in_production",
         all(codes <= (valid | {CORINE_NODATA}) for codes in LABEL_CODES.values()),
         f"codes outside valid_class_codes()+nodata: "
         f"{ {k: sorted(v - (valid | {CORINE_NODATA})) for k, v in LABEL_CODES.items()} }"),
    gate("supervised_coverage_measured",
         all(c["labelled_frac"] > 0 for c in COVERAGE),
         f"labelled fraction = {[round(100 * c['labelled_frac'], 1) for c in COVERAGE]}%"),
    gate("bands_json_roundtrip",
         json.loads(BANDS_PATH.read_text(encoding="utf-8"))["order"] == BANDS,
         f"{BANDS_PATH.name}: {len(BANDS)} bands with wavelength, resolution and role"),
    gate("sidecar_per_output",
         all(sidecar_of(r["stack"]).exists() and sidecar_of(r["label"]).exists()
             for r in STACKS.values()),
         f"{sum(sidecar_of(r['stack']).exists() for r in STACKS.values())}/"
         f"{len(STACKS)} stacks and "
         f"{sum(sidecar_of(r['label']).exists() for r in STACKS.values())}/"
         f"{len(STACKS)} labels have a manifest sidecar"),
    gate("keyed_lookup_used", INSPECT_DATE in by_date,
         f"scene selected by date {INSPECT_DATE}, not by list position"),
    gate("handoff_to_lab4_2",
         all(Path(r["stack"]).exists() and Path(r["label"]).exists() for r in STACKS.values()),
         f"{len(STACKS)} scene pairs in {paths.aligned_dir()}"),
    gate("not_synthetic_labels", CORINE_PATH.name != "CORINE_synthetic.tif",
         f"label source = {CORINE_PATH.name}"),
    gate("decisions_recorded", "TODO" not in PREDICTIONS["why"],
         "predictions explained: " + ("still the TODO placeholder -- name the mechanism"
                                      if "TODO" in PREDICTIONS["why"] else "recorded")),
]
gates.print_gate_board(board)

# %% [markdown]
# ### Reading a failed board
#
# A red line is not a formatting problem. `units_are_reflectance` failing means the
# stack still holds DN and every threshold in Labs 5–7 is off by 10⁴.
# `grids_identical` failing means Lab 4.2 will raise after extracting for an hour.
# `not_synthetic_labels` failing means you never obtained CORINE and your labels are
# noise. Report the failure and its cause; do not re-run until it is green by quietly
# loosening a threshold.
#
# `decisions_recorded` is red **on first run by design**: it fails while
# `PREDICTIONS["why"]` still holds the placeholder. It is not a code bug and no amount
# of re-running turns it green.

# %% [markdown]
# ## Part 10 — Deliverable
#
# `results.json` is what gets graded. Nothing in the 2025/26 labs was persisted, so
# there was no artifact to grade and no way to tell whether a cell had ever run.
#
# This lab has no held-out split, so the required metric keys carry the **gate pass
# fraction**, not model metrics, and the record says so in `notes`. The gradeable
# content is in `config` and `extra`.

# %%
run_id = f"lab4_1_{lab3['tile']}_{INSPECT_DATE.replace('-', '')}"
qc_pass = sum("[PASS]" in b for b in board)
n_checks = len(board)
rec_kwargs = dict(
    lab="lab4_1",
    config={
        "tile": lab3["tile"], "scenes": sorted(STACKS), "bands": BANDS,
        "units": "surface_reflectance",
        "quantization_value": radiometry.QUANTIFICATION_VALUE,
        "stack_nodata": NODATA, "label_nodata": CORINE_NODATA,
        "label_resampling": "nearest",
        "valid_class_codes": len(lab_mod.valid_class_codes()),
        "water_excluded": list(lab_mod.WATER_AND_SEDIMENT_CODES),
        "corine_source": CORINE_PATH.name,
        "cache": "sidecar manifest.json, no SKIP_EXISTING",
        "seed": SEED,
    },
    split_manifest_hash=None,
    seed=SEED,
    test_metrics={"n": int(sum(1 for _ in STACKS)), "overall_acc": qc_pass / n_checks,
                  "balanced_acc": qc_pass / n_checks, "macro_f1": qc_pass / n_checks},
    notes=(f"Lab 4.1 preprocessing. Gates {qc_pass}/{n_checks}. "
           f"Naive warp invented {pct_invented:.1f}% of a demo window as value 0; the "
           f"production warp does not. Cubic demo produced {len(invented_codes)} "
           f"interpolated codes and {len(invalid_codes)} outside valid_class_codes(). "
           f"NOTE: no held-out split exists in this lab, so the required metric keys "
           f"carry the gate pass fraction and n is the scene count, not model metrics. "
           f"Graded content is in config and extra."),
    extra={"gate_board": board, "predictions": PREDICTIONS, "coverage": COVERAGE,
           "alignment": ALIGN_REPORT, "dn_stats": DN_STATS,
           "naive_warp_invented_pct": pct_invented,
           "cubic_interpolated_codes": invented_codes,
           "cubic_invalid_codes": invalid_codes, "bands_json": str(BANDS_PATH)},
)
try:
    rec = results.record_run(run_id, **rec_kwargs)
    print(f"  recorded run_id={rec['run_id']}")
except results.ResultsError as exc:
    print(f"  not recorded: {exc}")
    print("  results.json is append-only. Re-run with a new run_id (it changes when you "
          "change your tile or inspected date) so the earlier attempt stays in the record.")

# %%
print(results.summary_table("lab4_1"))

# %% [markdown]
# ## Submission checklist
#
# Everything here is a file the grader can open. Self-attestation is not an artifact.
#
# * `results/bands.json` — every band with wavelength, native resolution and role, and
#   an `order` that matches the stack.
# * `<scratch>/<user>/data/aligned_data/s2_<scene>.tif` — one per scene, float32
#   reflectance, `nodata` declared, band descriptions set.
# * `.../aligned_data/s2_<scene>.manifest.json` and `corine_<scene>.manifest.json` —
#   sidecars with sources, bands, crs, shape, units, nodata, grid.
# * `.../aligned_data/corine_<scene>.tif` — one per scene, nearest, nodata on both ends.
# * `results/lab4_1_alignment.png` — per-band stretch, label panel, band histograms.
# * `results/results.json` — one `lab4_1` record with the gate board, your three
#   predictions and the coverage table inside `extra`.
# * In your write-up: the gate board with every `FAIL` explained; the
#   predicted-vs-actual lines for P1, P2 and P3; and one paragraph naming what the
#   naive warp teaches a model.
#
# **Before Lab 4.2**, confirm `units_are_reflectance` and `grids_identical` are green.
# Lab 4.2 reads these files and trusts them.

# %% [markdown]
# ## Where each 2025/26 defect went
#
# | 2025/26 | 2026/27 |
# |---|---|
# | `reflectance = DN/10000` never stated or applied | Part 2 measures it on your own window; stack is written in reflectance; `assert_reflectance_range` gates it |
# | `reproject` without `src_nodata`/`dst_nodata` | Part 6 quantifies the invented pixels, then warps with both |
# | resampling "explored" as an unguided suggestion | Exercise 3 forces the discovery with a number attached |
# | `successful_results[2]` | Part 1 shows index 2 is order-dependent; selection is by date and the check raises otherwise |
# | `SKIP_EXISTING = True` | removed; sidecar manifest per output, disagreement printed field by field |
# | band descriptions written, never read | `bands.json` written, read back, and used as the only band index |
# | `**/R10m/` pattern only | three-pattern lookup with a named error when nothing matches |
# | stacked file with `nodata=None` | explicit `nodata = -1.0`, outside the physical range of reflectance |
# | `(data >= 1) & (data <= 44)` | `labels.clean_labels` with the declared nodata and `valid_class_codes()` |
# | `MPLCONFIGDIR` set after the matplotlib import | set before it, pointing at scratch |
# | `try/except Exception: return None` everywhere | exceptions propagate; a missing band raises with the directory it searched |
# | `OUTPUT_DIR` hard-coded to `/p/project1` in one cell, env var in another | every path from `eo_course.paths` |
# | zero exercises, zero questions in the whole notebook | three graded predictions, one negative control, one gate board |
