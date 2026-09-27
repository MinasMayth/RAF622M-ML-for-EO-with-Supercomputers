# %% [markdown]
# # Lab 3.1 — Sentinel-2 acquisition for a CORINE-supervised task
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers · JURECA**
#
# This lab produces four Level-2A Sentinel-2 scenes from **one** MGRS tile, an
# extraction on disk in the layout Lab 4.1 actually reads, a `manifest.json` that
# records every choice you made, and a QC report that fails loudly when those
# choices are not what you claimed.
#
# It assumes Lab 1 (Judoor/JURECA access, `$SCRATCH`, `$PROJECT`) and Lab 2
# (Jupyter-JSC, the course virtualenv, git). It produces the input for Lab 4.1.
#
# **You must run every cell yourself.** Generated notebooks ship with no outputs,
# deliberately. The 2025/26 edition of this notebook was committed with
# `Authentication failed: Invalid credentials` in its output and all ten downstream
# cells unexecuted — the entire deliverable was undemonstrated, and nobody noticed
# because the cells above it looked fine. A green-looking notebook is not evidence.
#
# ## The assignment, stated exactly
#
# Each team picks **one Sentinel-2 MGRS tile in Europe** and retrieves **four
# Level-2A acquisitions** spread across **March–October 2018**, each with reported
# cloud cover **≤ 30%**, and relates them to **CORINE Land Cover 2018** labels.
# Teams claim their tile publicly so two teams do not collide.
#
# | Lab | Notebook | Status |
# |---|---|---|
# | 1 | `lab1_judoor_hpc_access` | done |
# | 2 | `lab2_jupyter_jsc_git` | done |
# | **3.1** | **`lab3_1_data_acquisition` (this one)** | **current** |
# | 4.1 | `lab4_1_data_preprocessing` | next |
# | 4.2 | `lab4_2_patch_extraction` | |
# | 5.1 | `lab5_1_cnn_training` | |
# | 5.2 | `lab5_2_pytorch_lightning` | |
# | 6 | `lab6_terratorch_finetuning` | |
# | 7 | `lab7_model_evaluation` | final deliverable |
#
# There is no "Lab 3.2" and no "Understanding Transformers" notebook. The 2025/26
# roadmap table in this cell listed seven labs, none of which matched a file in the
# repository, and pointed students at a notebook that did not exist.

# %% [markdown]
# ## The constraint the old lab never stated
#
# The date window **March–October 2018** is not a convenience. It is the **CORINE
# 2018 vintage**. CORINE Land Cover is a *dated* map: the 2018 edition describes the
# land surface as it was around 2018, produced from 2018 imagery by photo-interpretation
# and spot checks. Your imagery and your labels must be temporally coherent, or the
# label is not a label.
#
# A team that downloads 2024 imagery and supervises it with CORINE 2018 is training a
# model to predict **land-cover change**, and reporting it as land-cover
# classification. In a Bavarian or Iberian tile that error is small and invisible:
# most pixels did not change in six years, so accuracy looks fine and your rare-class
# numbers are quietly wrong. In a tile with active urban expansion, forestry felling,
# or post-fire regeneration, the mismatch is the dominant signal.
#
# You cannot see this error in a confusion matrix. It has to be prevented at
# acquisition time, which is why it is stated here and nowhere later.
#
# ```
# CORINE 2018  ──describes──▶  land surface circa 2018
# Sentinel-2   ──must image──▶  the same land surface, 2018
# ```

# %% [markdown]
# ## Why the tile must be in Europe (and why this repo says "Iceland")
#
# The repository is called `iceland-ml` and the notebooks live in
# `notebooks/iceland-ml/`, while Lab 4.1 states in its first line that **CORINE does
# not cover Iceland**. Both statements are true and the contradiction is resolved by
# the assignment itself: the course uses **European** tiles precisely *because*
# CORINE 2018 covers the EEA member states and not Iceland.
#
# The 2025/26 notebook did not resolve this. It set a Bavarian bounding box
# (`min_lon, min_lat = 8.0, 47.0`) inside a repo named for Iceland, used tile
# `T33UUP` downstream, and offered no explanation. Students could not tell whether
# Bavaria was a mistake or the plan.
#
# **The honest position:**
#
# * Course work: pick a tile in a CORINE-covered European country. Do this lab as written.
# * Iceland work: CORINE 2018 is unavailable. You need a different label source —
#   ESA WorldCover (10 m, 2020/21 vintage), a national land-cover map, or your own
#   labelled polygons. That is a **different project** with a different label-quality
#   story, a different vintage constraint, and no CORINE class taxonomy; it is not a
#   drop-in substitution, and it is not offered as an option in this course's
#   assessment. If you want to do it, talk to us before Lab 4.

# %% [markdown]
# ## Part 0 — Setup
#
# `MPLCONFIGDIR` is set **before** matplotlib is imported. The 2025/26 notebooks set
# it after the import, which does nothing, and shipped the warning
# `Matplotlib created a temporary cache directory at /tmp/matplotlib-_m96zrml` in
# committed output for three notebooks running.

# %%
from eo_course import paths

print(paths.describe())

# %%
import os
import re
import json
import copy
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = paths.matplotlib_cache_dir()

import matplotlib.pyplot as plt  # noqa: E402  (must follow MPLCONFIGDIR)
import rasterio  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
from rasterio.warp import transform  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402

from eo_course import radiometry, gates, results  # noqa: E402

SEED = 20180301
rng = np.random.default_rng(SEED)
print(f"seed = {SEED}  (one rng for the whole notebook; pass it down)")

# %% [markdown]
# ## Part 1 — Credentials, and why a default value is a bug
#
# Register an OAuth2 client once:
#
# 1. Open <https://browser.dataspace.copernicus.eu/> and sign in (create an account if needed).
# 2. Profile (top right) → **User settings** → **OAuth clients** → **Create new client**.
# 3. Name it (e.g. `raf622m-lab3`), set expiry to a year or more, confirm.
# 4. Copy **Client ID** and **Client secret**. The secret is shown **once**.
# 5. On JURECA, put them in `~/.bashrc` — *not* in a notebook cell:
#
#    ```bash
#    export COPERNICUS_CLIENT_ID="..."
#    export COPERNICUS_CLIENT_SECRET="..."
#    ```
#
#    Then start your Jupyter-JSC session **from a shell that has sourced that file**,
#    or the kernel will not see the variables.
#
# The 2025/26 notebook printed `See COPERNICUS_SETUP.md for detailed instructions`.
# That file was never in the student repository — the instructions existed only on the
# instructor's copy. They are in this cell now.
#
# ### Two credential defects that are worth understanding
#
# **Defect 1 — printing a credential fragment.** Cell 7 of the old notebook printed
# the first fifteen characters of the client secret into notebook output. That output
# is stored in the `.ipynb` JSON, committed, pushed, and shared with every student.
# A truncated secret is still a secret: it removes a large fraction of the search
# space and it tells an attacker the credential is real. This notebook prints `set`
# or `MISSING` and nothing else.
#
# **Defect 2 — a silent fallback.** The old cell read the client ID as
# `os.getenv('COPERNICUS_CLIENT_ID', <a literal id>)`, so a student with no
# credential silently ran against someone else's client. That is exactly why the
# notebook is committed in a failed state: the fallback produced a credential that was
# not the owner's, authentication returned `Invalid credentials`, and every cell after
# it was never run. **A missing credential should raise, not guess.** We read
# `os.environ["NAME"]`, which raises `KeyError`.

# %%
CRED_NAMES = ("COPERNICUS_CLIENT_ID", "COPERNICUS_CLIENT_SECRET")
present = {n: ("set" if os.environ.get(n) else "MISSING") for n in CRED_NAMES}
for name, state in present.items():
    print(f"  {name:<28} {state}")
missing = [n for n, s in present.items() if s == "MISSING"]

# %% [markdown]
# ### Live mode and demo mode — an explicit, labelled fork
#
# The old notebook hard-failed at authentication and left 10 cells unexecuted. That is
# the right *error* but the wrong *experience*: a student with no credential yet learns
# nothing about search, tiles, or QC.
#
# So this notebook forks **explicitly**:
#
# * **live** — real credentials present: query CDSE, download real products.
# * **demo** — credentials absent: build small *synthetic* products with the same
#   naming, structure and metadata fields as real ones, so every downstream cell
#   (tile grouping, selection, extraction, AOI cloud arithmetic, QC, manifest) runs on
#   data that is honestly labelled synthetic.
#
# Demo output is **not** course data. It is stamped `synthetic: true` in the manifest
# and the QC board prints a warning. You cannot submit demo output.

# %%
DEMO_OK = True  # set False to force a hard failure when credentials are missing
MODE = "demo" if (missing and DEMO_OK) else "live"
if missing and not DEMO_OK:
    raise KeyError(
        f"missing environment variables: {missing}. Set them in ~/.bashrc and restart "
        "the kernel, or set DEMO_OK = True to walk the pipeline on synthetic data."
    )
print(f"MODE = {MODE}")
if MODE == "demo":
    print("  Synthetic products will be generated locally. Nothing here is course data.")
    print("  To run for real: export the two variables above, restart the kernel, re-run.")

# %%
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products"


def get_access_token():
    """Client-credentials token. Raises if the credential is missing or rejected."""
    import requests

    resp = requests.post(
        AUTH_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["COPERNICUS_CLIENT_ID"],
            "client_secret": os.environ["COPERNICUS_CLIENT_SECRET"],
        },
        timeout=30,
    )
    if resp.status_code == 401:
        raise PermissionError(
            "CDSE rejected the credential (401). Check the client has not expired and "
            "that you copied the whole secret. No credential material is printed here "
            "on purpose -- look at the Dashboard, not at this notebook."
        )
    resp.raise_for_status()
    body = resp.json()
    print(f"  token obtained, valid {body.get('expires_in', 3600) // 60} min")
    return body["access_token"]


access_token = get_access_token() if MODE == "live" else None
print(f"  access_token = {'set' if access_token else 'absent (demo mode)'}")

# %% [markdown]
# ## Part 2 — What a Level-2A product actually stores
#
# This is the part the 2025/26 course never taught, anywhere. It belongs here, because
# this is the lab where the product specification is cited.
#
# Sentinel-2 L2A does **not** store reflectance. It stores an integer quantised
# reflectance:
#
# $$\text{DN} = \operatorname{round}\!\left(\frac{\rho}{\text{QUANTIFICATION\_VALUE}}\right),
# \qquad \rho = \text{surface reflectance},\quad \text{QUANTIFICATION\_VALUE} = 10000$$
#
# So DN 4000 is 40 % reflectance and DN 100 is 1 %. `QUANTIFICATION_VALUE` is not a
# normalisation trick; it is the product's unit definition, and it is in
# `eo_course.radiometry` as a named constant.
#
# Two consequences with numbers attached:
#
# 1. Any threshold you write in reflectance units (cloud, snow, water, NDVI) is wrong
#    by four orders of magnitude if you forget the division.
# 2. Geospatial foundation models were pretrained with published per-band mean/std in
#    reflectance or DN units. Feeding a percentile stretch in $[0,1]$ against means of
#    order 1000 is a ~1000× scale error. In 2025/26 Lab 6 that produced a test accuracy
#    of exactly `0.10000000149011612` — precisely $1/10$, chance for ten classes — and
#    the unit sheet was satisfied anyway, because "functioning TerraTorch workflow" is
#    also satisfied by a model that predicts one class.
#
# **L1C vs L2A**, also never stated in the course: L1C is top-of-atmosphere reflectance
# (aerosol, Rayleigh scattering, ozone still in it). L2A (`productType eq 'S2MSI2A'`)
# has been atmospherically corrected to surface reflectance by Sen2Cor, and carries a
# per-band `nodata` value. We use L2A because CORINE labels are about the *surface*, and
# because correcting an atmosphere is a whole lab of its own.

# %%
QV = radiometry.QUANTIFICATION_VALUE
demo_dn = np.array([[0, 100, 1500, 4000, 9000]], dtype=np.uint16)
demo_rho = radiometry.dn_to_reflectance(demo_dn)
print(f"QUANTIFICATION_VALUE = {QV:.0f}")
print(f"  DN          {demo_dn[0].tolist()}")
print(f"  reflectance {[round(float(v), 4) for v in demo_rho[0]]}")
radiometry.assert_reflectance_range(demo_rho, "demo_rho")
try:
    radiometry.assert_reflectance_range(demo_dn, "raw DN mislabelled as reflectance")
except AssertionError as exc:
    print(f"  caught, as intended: {str(exc).splitlines()[0]}")

# %% [markdown]
# ## Part 3 — Exercise 1: predict before you run (cloud cover)
#
# CDSE reports a `cloudCover` attribute for every product. The old notebook filtered on
# it server-side and then printed `✓ All products have <30% cloud cover` as if that
# settled the question.
#
# It does not. **`cloudCover` is a scene-level percentage**: the fraction of the whole
# 109 km × 109 km MGRS tile that is cloud, computed by the processor over the full
# footprint. Your area of interest is a small part of that footprint. A scene reported
# at 28 % can be 90 % cloudy over your AOI, and a scene reported at 45 % can be perfectly
# clear over your AOI. The filter is a coarse pre-screen, not a quality check.
#
# **Write your prediction down before running Part 6.**
#
# > **P1.** For the four scenes this lab ends up selecting, predict the **cloudy
# > fraction over your AOI** for each one, as a percentage. Then predict the single
# > largest absolute gap, in percentage points, between a scene's reported scene-level
# > `cloudCover` and its AOI fraction.
# >
# > Record it in `PREDICTIONS["cloud"]` below. Part 6 prints the actual numbers. You are
# > graded on the prediction and on your explanation of the disagreement — not on the
# > output.

# %%
PREDICTIONS = {
    # TODO(you): fill these in BEFORE running Part 6.
    # Falsifiable condition: `aoi` must be four numbers in [0, 100] and `max_gap_pp`
    # a number in [0, 100]. Leaving them as None makes the comparison cell report
    # "not recorded", which is scored as not attempted.
    "cloud": {"aoi": None, "max_gap_pp": None, "why": ""},
    "temporal": {"better_set": None, "why": ""},
    "bands": {"n_bands": None, "why": ""},
}

# %% [markdown]
# ## Part 4 — Search parameters: three decisions, not three defaults
#
# ### Decision A — the area of interest
#
# The AOI is a box *inside* one MGRS tile. It is deliberately much smaller than the
# 6° × 2.5° box the 2025/26 notebook used, because that box spanned **many** tiles, and
# the notebook then (i) truncated the result at 1000 products, (ii) counted scenes per
# tile on the truncated list, and (iii) picked the tile with the most scenes — which
# measures "largest overlap with a wide box", not "best tile for my area".
#
# ### Decision B — the date window
#
# Fixed by CORINE: `2018-03-01` to `2018-10-31`. Not a choice; a constraint.
#
# ### Decision C — the reported-cloud ceiling
#
# 30 %, per the assignment. Understand what it buys and what it costs: at 10 % the
# candidate pool in central Europe in spring collapses, and teams end up with four
# scenes clustered in August — which defeats the whole point of a March–October window.
# 30 % is a compromise that keeps the seasonal spread. Exercise 1 shows you why the
# number is only a pre-screen.

# %%
AOI_NAME = "Munich SE (demo AOI, inside one MGRS tile)"
min_lon, min_lat = 11.420, 48.060
max_lon, max_lat = 11.550, 48.180
AOI_WKT = (
    f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
    f"{min_lon} {max_lat},{min_lon} {min_lat}))"
)

START_DATE = "2018-03-01T00:00:00.000Z"
END_DATE = "2018-10-31T23:59:59.999Z"
MAX_CLOUD = 30.0
N_SCENES = 4

print(f"  AOI        : {AOI_NAME}")
print(f"  bbox       : lon [{min_lon}, {max_lon}]  lat [{min_lat}, {max_lat}]")
print(f"  bbox size  : {(max_lon - min_lon) * 75:.0f} km E-W x "
      f"{(max_lat - min_lat) * 111:.0f} km N-S at 48N")
print(f"  window     : {START_DATE[:10]} .. {END_DATE[:10]}  (CORINE 2018 vintage)")
print(f"  cloud      : reported scene-level cloudCover <= {MAX_CLOUD:.0f}%")
print(f"  scenes     : {N_SCENES}, all from one tile")

# %% [markdown]
# ### MGRS tiles: the unit of work is the tile, not the box
#
# Sentinel-2 divides the land surface into 10980 × 10980-pixel tiles on a Military
# Grid Reference System grid, each about 109 km × 109 km. Three facts that matter:
#
# * **A tile is not aligned to your AOI.** Most of a tile is usually somewhere else.
#   Statistics computed over a whole tile are not statistics about your AOI — which is
#   exactly the error behind Exercise 1.
# * **The MGRS band number is not the UTM zone number.** Tiles are gridded in a UTM
#   projection, and the projection a given tile uses is a property of the product, not
#   something you derive. Lab 4.1 in 2025/26 asserted "EPSG:32633 for zone 33N" for the
#   tile it actually used, and the tile's own GeoTIFFs declared a different EPSG. We
#   read the CRS out of the file, every time.
# * **One tile, four scenes.** Scenes from different tiles have different geometry,
#   different illumination geometry, and different amounts of their footprint outside
#   your AOI. Mixing them means your four "temporal" samples are also four different
#   spatial samples, and you cannot tell which variable moved.

# %% [markdown]
# ## Part 5 — Search, with pagination
#
# The OData filter is six clauses joined by `and`. Read them one at a time:
#
# | clause | meaning |
# |---|---|
# | `Collection/Name eq 'SENTINEL-2'` | restrict to the S2 mission |
# | `...productType... eq 'S2MSI2A'` | **Level-2A**, surface reflectance (see Part 2) |
# | `ContentDate/Start gt <start>` | sensing time after the window opens |
# | `ContentDate/Start lt <end>` | sensing time before the window closes |
# | `OData.CSC.Intersects(area=...)` | footprint intersects the AOI polygon |
# | `...cloudCover... lt <max>` | reported **scene-level** cloud below the ceiling |
#
# The old search asked for `"$top": 1000` and stopped. A six-month query over a 6° ×
# 2.5° box returns more than 1000 products, so the notebook reported "✓ Found 1000
# products" — a number that is the page size, not the result size — and every per-tile
# count computed from it was wrong. Here we follow `@odata.nextLink` until the server
# stops handing us one, and we report how many pages that took.

# %%
def build_filter(aoi_wkt: str, start: str, end: str, max_cloud: float) -> str:
    return " and ".join([
        "Collection/Name eq 'SENTINEL-2'",
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
        f"ContentDate/Start gt {start}",
        f"ContentDate/Start lt {end}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
        "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value lt {max_cloud})",
    ])


def search_sentinel2(aoi_wkt, start, end, max_cloud, page=200):
    """Return (products, n_pages). Complete: follows @odata.nextLink to exhaustion."""
    import requests

    params = {
        "$filter": build_filter(aoi_wkt, start, end, max_cloud),
        "$orderby": "ContentDate/Start asc",
        "$top": page,
        "$count": "true",
    }
    headers = {"Authorization": f"Bearer {access_token}"}
    url, products, pages = SEARCH_URL, [], 0
    while url:
        resp = requests.get(url, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        products.extend(body.get("value", []))
        pages += 1
        # The nextLink already carries the query string; do not re-send params.
        url = body.get("uri:nextLink") or body.get("@odata.nextLink")
        params = {}
    return products, pages


# %%
def synthetic_catalog(tile_dates: dict, cloud_by_scene: dict) -> list[dict]:
    """CDSE-shaped product dicts for demo mode. `synthetic` is stamped on each."""
    out = []
    for tile, dates in tile_dates.items():
        for i, iso in enumerate(dates):
            key = f"{tile}_{iso}"
            cc = cloud_by_scene[key]
            t = iso.replace("-", "").replace(":", "") + "00"
            name = (f"S2{'A' if i % 2 == 0 else 'B'}_MSIL2A_{t}_N0207_R108_"
                    f"{tile}_{t[:8]}T120000.SAFE")
            out.append({
                "Id": f"demo-{name}",
                "Name": name,
                "ContentDate": {"Start": iso + "T10:30:00.000Z"},
                "ContentLength": int(1.18e9),
                "Attributes": {"cloudCover": cc},
                "synthetic": True,
            })
    return sorted(out, key=lambda p: p["ContentDate"]["Start"])


# Demo tile/date layout: three candidate tiles, deliberately uneven, so the tile
# decision in Part 7 is a decision. Cloud values are scene-level percentages.
DEMO_TILE_DATES = {
    "T32UPD": ["2018-03-04", "2018-03-24", "2018-04-13", "2018-05-03", "2018-05-23",
               "2018-06-12", "2018-07-02", "2018-07-22", "2018-08-11", "2018-08-31",
               "2018-09-20", "2018-10-10", "2018-10-30"],
    "T32TMS": ["2018-04-05", "2018-05-18", "2018-06-25", "2018-08-02", "2018-09-14"],
    "T31TFI": ["2018-03-19", "2018-06-08", "2018-09-01"],
}
DEMO_CLOUD = {}
for _t, _ds in DEMO_TILE_DATES.items():
    for _d in _ds:
        DEMO_CLOUD[f"{_t}_{_d}"] = float(np.round(rng.uniform(2.0, 29.0), 1))

# %%
if MODE == "live":
    products, n_pages = search_sentinel2(AOI_WKT, START_DATE, END_DATE, MAX_CLOUD)
    synthetic = False
else:
    products, n_pages = synthetic_catalog(DEMO_TILE_DATES, DEMO_CLOUD), 1
    synthetic = True
    print("  [demo] synthetic catalog; no network call was made")

print(f"  products returned : {len(products)} over {n_pages} page(s) followed to exhaustion")
if len(products) and len(products) % 200 == 0 and n_pages == 1:
    print("  WARNING: count is an exact multiple of the page size on a single page -- "
          "the server may have truncated. Check the pagination loop.")
if products:
    print(f"  date span         : {products[0]['ContentDate']['Start'][:10]} .. "
          f"{products[-1]['ContentDate']['Start'][:10]}")

# %% [markdown]
# ### Grouping by tile: parse it, do not count underscores
#
# The old notebook read the tile as `product_name.split('_')[5]` guarded by
# `len(...) > 5`, so any name with a different field count was silently labelled
# `'Unknown'` and its scenes vanished from every tile's count. We match the tile token
# where it actually is — `T` + two digits + three letters, delimited by underscores —
# and **skip with a warning** when nothing matches, so a malformed name is visible
# rather than absorbed.

# %%
TILE_RE = re.compile(r"_(T\d{2}[A-Z]{3})_")


def group_by_tile(prods):
    tiles, skipped = defaultdict(list), []
    for p in prods:
        m = TILE_RE.search(p.get("Name", ""))
        if m:
            tiles[m.group(1)].append(p)
        else:
            skipped.append(p.get("Name", "<no name>"))
    return tiles, skipped


tiles, skipped = group_by_tile(products)
if skipped:
    print(f"  WARNING: {len(skipped)} product(s) had no MGRS tile token and were skipped:")
    for s in skipped[:5]:
        print(f"    {s}")
print(f"  {len(tiles)} tile(s), {len(products) - len(skipped)} products grouped\n")
print(f"  {'tile':<9} {'scenes':>6}  {'first':<12} {'last':<12} {'cloud range':>16}")
for tile, tp in sorted(tiles.items(), key=lambda kv: -len(kv[1])):
    ds = [p["ContentDate"]["Start"][:10] for p in tp]
    cs = [float(p["Attributes"]["cloudCover"]) for p in tp]
    print(f"  {tile:<9} {len(tp):>6}  {ds[0]:<12} {ds[-1]:<12} "
          f"{min(cs):>7.1f}-{max(cs):<6.1f}")

# %% [markdown]
# ## Part 6 — Decision D: which tile, and on what criterion
#
# The old notebook chose `max(tiles.items(), key=lambda x: len(x[1]))` — the tile with
# the most scenes — and left the manual alternative commented out, so no student ever
# chose. "Most scenes" is a proxy for "largest overlap with a wide bounding box". The
# winning tile can be one whose footprint only grazes your AOI, in which case most of
# its pixels are somewhere else entirely and your four scenes are four views of mostly
# the wrong place.
#
# **You choose, and you state the criterion.** Reasonable criteria, in rough order of
# how much they should matter here:
#
# 1. **Coverage of the AOI.** Does the tile contain your whole AOI, or clip it?
# 2. **Temporal spread available.** Can you get four scenes spanning March *and* October?
# 3. **Reported cloud of the scenes you would actually use**, not the tile average.
# 4. **Claimability.** Is the tile already taken by another team? Check the shared
#    claims list before you commit — this is a course rule, not a suggestion.
# 5. Number of scenes. Last, because it is the easiest to satisfy and the least
#    informative.
#
# Whatever you pick, the notebook records it. A decision that is not written down cannot
# be graded, and in 2025/26 nothing about tile choice was written down at all.

# %%
SELECTED_TILE = "T32UPD"  # TODO(you): change this; it must be a key of `tiles`
TILE_CRITERION = (
    "TODO(you): one or two sentences. Name the criterion (coverage / temporal spread / "
    "cloud / claimability) and the number that supports it."
)

assert SELECTED_TILE in tiles, (
    f"{SELECTED_TILE!r} is not among the tiles found: {sorted(tiles)}. "
    "Falsifiable condition: this cell must run without raising."
)
selected_products_pool = tiles[SELECTED_TILE]
print(f"  selected tile : {SELECTED_TILE}  ({len(selected_products_pool)} scenes available)")
print(f"  criterion     : {TILE_CRITERION}")

# %% [markdown]
# ### Choosing four dates: evenly spaced is a rule, not a guarantee
#
# Four scenes is the assignment's budget, and it is a real constraint: at 5-day
# revisit and a 30 % cloud ceiling you will not get a scene for every date you want.
# Even spacing across the window is the cheapest way to spend four samples on a year,
# but it is only defensible if the *achieved* spacing is reported. The old notebook
# printed "These acquisitions span the full date range" without checking.
#
# We select by **date target**, not by list index — indexing into a filtered list is
# sensitive to which scenes happened to survive the cloud filter.

# %%
def pick_evenly_spaced(pool, n, start_iso, end_iso):
    """For n evenly spaced target dates, take the nearest available scene to each.

    Returns (deviation_days, products) sorted by sensing date.
    """
    t0 = datetime.fromisoformat(start_iso[:10])
    t1 = datetime.fromisoformat(end_iso[:10])
    span = (t1 - t0).days
    targets = [t0.timestamp() + span * 86400 * i / (n - 1) for i in range(n)]
    dated = [(datetime.fromisoformat(p["ContentDate"]["Start"][:10]).timestamp(), p)
             for p in pool]
    chosen, used = [], set()
    for tgt in targets:
        dev, i = min(((abs(ts - tgt), j) for j, (ts, p) in enumerate(dated) if j not in used))
        used.add(i)
        chosen.append((dev / 86400, dated[i][1]))
    chosen.sort(key=lambda c: c[1]["ContentDate"]["Start"])
    return [c[0] for c in chosen], [c[1] for c in chosen]


offsets, selected_products = pick_evenly_spaced(
    selected_products_pool, N_SCENES, START_DATE, END_DATE)
dates = [p["ContentDate"]["Start"][:10] for p in selected_products]
print(f"  {'#':<3} {'date':<12} {'cloud%':>7} {'|offset-target| days':>22}  product")
for i, (dev, p) in enumerate(zip(offsets, selected_products)):
    print(f"  {i+1:<3} {dates[i]:<12} {float(p['Attributes']['cloudCover']):>7.1f} "
          f"{dev:>22.1f}  {p['Name'][:52]}")
gaps = [(datetime.fromisoformat(dates[i+1]) - datetime.fromisoformat(dates[i])).days
        for i in range(len(dates) - 1)]
print(f"  inter-scene gaps (days): {gaps}")

# %% [markdown]
# ## Part 7 — Exercise 2: predict before you run (temporal selection)
#
# Two candidate 4-scene sets, both legal under the rules (one tile, four scenes, all
# ≤ 30 % reported cloud):
#
# * **Set A** — evenly spaced across March–October (what Part 6 just did).
# * **Set B** — the four lowest-cloud scenes available in the window.
#
# Set B has better cloud numbers. It will almost certainly cluster in late summer,
# because that is when central Europe is clear.
#
# **Write your prediction down before running the next cell.**
#
# > **P2.** Which set better represents a year of surface state for supervised
# > land-cover classification against CORINE 2018 — A or B? Give your answer as
# > `"A"` or `"B"`, plus one sentence of reasoning. Reasoning that mentions only cloud
# > cover is incomplete: say what land-cover information each set can and cannot
# > express.
#
# The reveal is not "A wins because it is evenly spaced". The real content is *what the
# four dates let a classifier see*. March–October deliberately straddles three states
# that CORINE treats as one class: **snow season** (class 34 "Glaciers and perpetual
# snow" is mapped where snow is *perpetual*; seasonal snow is not a CORINE class, so a
# March scene shows bare soil and litter as "arable land" or "broad-leaved forest"),
# **growing season** (peak NDVI, classes spectrally separable), and **harvest**
# (stubble, bare soil, and post-harvest ploughing, where arable land becomes
# spectrally similar to bare rock and to construction sites). Four scenes that all sit
# in August show you only the middle state, and the model never learns that class 12
# "Non-irrigated arable land" also looks like soil.

# %%
def set_by_lowest_cloud(pool, n):
    ranked = sorted(pool, key=lambda p: float(p["Attributes"]["cloudCover"]))[:n]
    return sorted(ranked, key=lambda p: p["ContentDate"]["Start"])


set_b = set_by_lowest_cloud(selected_products_pool, N_SCENES)
b_dates = [p["ContentDate"]["Start"][:10] for p in set_b]
b_cloud = [float(p["Attributes"]["cloudCover"]) for p in set_b]
b_gaps = [(datetime.fromisoformat(b_dates[i+1]) - datetime.fromisoformat(b_dates[i])).days
          for i in range(len(b_dates) - 1)]

print(f"  Set A dates {dates}")
print(f"          cloud {[float(p['Attributes']['cloudCover']) for p in selected_products]}")
print(f"          span {(datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])).days} d, "
      f"gaps {[gaps]}")
print(f"  Set B dates {b_dates}")
print(f"          cloud {b_cloud}")
print(f"          span {(datetime.fromisoformat(b_dates[-1]) - datetime.fromisoformat(b_dates[0])).days} d, "
      f"gaps {b_gaps}")
print("\n  Record your answer in PREDICTIONS['temporal'] now, then run the next cell.")

# %%
# The comparison the grader reads: predicted vs actual, and an explanation.
_a = [datetime.fromisoformat(d) for d in dates]
_b = [datetime.fromisoformat(d) for d in b_dates]
actual = {
    "A_span_days": (_a[-1] - _a[0]).days,
    "B_span_days": (_b[-1] - _b[0]).days,
    "A_mean_gap_days": (_a[-1] - _a[0]).days / (N_SCENES - 1),
    "B_mean_gap_days": (_b[-1] - _b[0]).days / (N_SCENES - 1),
    "A_mean_cloud": float(np.mean([float(p["Attributes"]["cloudCover"]) for p in selected_products])),
    "B_mean_cloud": float(np.mean(b_cloud)),
    "A_months": sorted({d[5:7] for d in dates}),
    "B_months": sorted({d[5:7] for d in b_dates}),
}
print(json.dumps(actual, indent=2))
print(f"\n  your prediction : {PREDICTIONS['temporal']['better_set']!r}")
print(f"  your reasoning  : {PREDICTIONS['temporal']['why'] or '(not recorded)'}")
print("  Grading is on the reasoning. If you picked B, say what information you gave")
print("  up in exchange for the lower cloud, and whether it matters for CORINE classes.")

# %% [markdown]
# ## Part 8 — Decision E: 4 bands or 6, decided at acquisition time
#
# Sentinel-2 L2A ships 13 bands. This course uses either four or six. The choice is
# yours, and it is **cheapest to make here**, before four products are downloaded and
# re-projected — not in Lab 6, after a foundation model has already been fine-tuned on
# the wrong band set.
#
# | option | bands | consequence |
# |---|---|---|
# | 4 | B02, B03, B04, B08 | Blue, Green, Red, Narrow-NIR. Half the storage, good for urban/vegetation contrast. **No SWIR at all.** |
# | 6 | B02, B03, B04, B08, B11, B12 | Adds SWIR-1 and SWIR-2. |
#
# The reason to care about SWIR is specific to CORINE, not generic. **SWIR is what
# separates CORINE arable land from forest and from bare soil.** Moisture and lignin
# absorb strongly in SWIR-2 (B12); a crop canopy and a leaf-off deciduous forest can be
# close in Red and Narrow-NIR and far apart in B11/B12, and dry cropland versus bare
# rock — classes 12 and 31, both spectrally bright and "brown" — is largely a SWIR
# discrimination. Drop SWIR and those confusions become irreducible: no amount of model
# capacity recovers information the sensor did not record.
#
# There is a second, harder constraint downstream. **Prithvi-EO-2.0, used in Lab 6, was
# pretrained on exactly six bands** — `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2` —
# with published per-band mean/std. A 4-band configuration keeps the first four in
# pretrained order and drops both SWIR bands, which is legitimate but is a real loss,
# not a detail. Six bands keeps you aligned with the pretrained input.
#
# > **P3 / Decision.** Choose 4 or 6. Record it in `PREDICTIONS["bands"]` with the
# > reason and the downstream consequence you accepted. Falsifiable condition: the cell
# > below must print a `band decision` line whose reason is not `TODO`.

# %%
BANDS_4 = radiometry.S2_BANDS_4
BANDS_6 = radiometry.S2_BANDS_6
N_BANDS = 6  # TODO(you): 4 or 6
BAND_REASON = "TODO(you): why this choice, and what you gave up."
assert N_BANDS in (4, 6), "N_BANDS must be 4 or 6"
BANDS = BANDS_6 if N_BANDS == 6 else BANDS_4

mean, std = radiometry.prithvi_norm(bands=N_BANDS, dn_units=True)
print(f"  band decision : {N_BANDS} bands -> {BANDS}")
print(f"  reason        : {BAND_REASON}")
print(f"  dropped       : {sorted(set(BANDS_6) - set(BANDS)) or 'none'}")
print(f"  Prithvi-EO-2.0 pretrained mean (DN): {[f'{m:.0f}' for m in mean]}")
print(f"  Prithvi-EO-2.0 pretrained std  (DN): {[f'{s:.0f}' for s in std]}")
print("  Lab 6 must normalise to these numbers, in this band order. Not a percentile stretch.")

# %% [markdown]
# ## Part 9 — Download: this is a batch job, not a notebook cell
#
# A full Sentinel-2 L2A product is roughly **1.2 GB**. Four of them, downloaded
# serially in a Python `for` loop on a login node, is 20–60 minutes of wall clock
# depending on CDSE throttling — and it is exactly the sustained login-node I/O that
# gets a training project's login session throttled or disconnected. The 2025/26
# notebook did precisely this, and its own "claimed 120 min" was really 2.5–3 h.
#
# The correct shape is a CPU job. There is no download script in `slurm/` yet; copy
# `slurm/train_jureca.sbatch` and change the header and the body:
#
# ```bash
# #SBATCH --job-name=s2_fetch
# #SBATCH --account=training2600
# #SBATCH --partition=dc-cpu          # lower case; the upper-case name in the 2025/26 lab is not a partition
# #SBATCH --ntasks-per-node=4         # one scene per task, or use --array
# #SBATCH --cpus-per-task=1
# #SBATCH --mem-per-cpu=1000
# #SBATCH --time=01:00:00
# ```
#
# and fan the four products out with `srun --tasks-per-node=4` or
# `sbatch --array=0-3`. Four parallel streams finish in about the time one serial
# stream takes, and the login node stays responsive. Note that `module` is a shell
# function, so it cannot be called from a notebook `!` line — that is why the 2025/26
# notebooks used `!source ...`, which fails under `/bin/sh`.
#
# Also note `timeout=300` in a streaming download is a **per-read** timeout, not a
# total budget. It does not mean "abort after five minutes".
#
# **In this notebook** we keep the download small so the lab fits its session: in live
# mode `N_DOWNLOAD` defaults to one scene and the QC board then *fails* with a message
# telling you to run the rest as a job. Failing with an explanation is better than
# succeeding at less than you claimed.

# %%
def zip_name(product) -> str:
    """`<NAME>.SAFE.zip` -> `<NAME>.zip`.
    
    The 2025/26 downloader wrote `f"{product_name}.zip"` where `product_name` already
    ended in `.SAFE`, so files on disk were `....SAFE.zip`. That single naming bug is
    why the old extraction cell needed about ninety lines of `possible_paths` guessing
    and still could not be graded. Fix the name and the guessing collapses to three
    lines. `removesuffix` (Python 3.9+; the course pins 3.12) removes a suffix.
    """
    return product["Name"].removesuffix(".SAFE") + ".zip"


def download_product(product, zip_dir, session):
    """Stream one product to `zip_dir`. Reuses the caller's authenticated session."""
    import requests

    paths.ensure(zip_dir)
    out = Path(zip_dir) / zip_name(product)
    if out.exists() and out.stat().st_size == int(product.get("ContentLength", 0)):
        print(f"  already present: {out.name}")
        return out
    url = f"{DOWNLOAD_URL}({product['Id']})/$value"
    t0 = time.time()
    with requests.get(url, headers=session.headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        written = 0
        with open(out, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
                    written += len(chunk)
    print(f"  {out.name}: {written / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    return out


# %%
def zip_is_intact(path):
    """True if the archive opens and no member fails its CRC.

    A truncated download is the most common failure in this lab and the one the 2025/26
    notebook never checked: it printed the file size and moved on. `testzip` walks every
    member, so it catches a zip whose central directory survived but whose payload did
    not. It reads the whole archive, so run it in the batch job, not on 5 GB at a time.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def download_session():
    import requests

    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {access_token}"})
    return s


print("  download helpers defined")
print("  one token, one Session, reused for every product -- token requests are rate limited")

# %%
N_DOWNLOAD = 1 if MODE == "live" else N_SCENES  # live: keep the session short
ZIP_DIR = paths.data_root() / "zip"
paths.ensure(ZIP_DIR)
zip_paths = []
if MODE == "live":
    sess = download_session()
    for p in selected_products[:N_DOWNLOAD]:
        zip_paths.append(download_product(p, ZIP_DIR, sess))
    zip_paths += [None] * (len(selected_products) - len(zip_paths))
    print(f"\n  downloaded {N_DOWNLOAD} of {N_SCENES}. The QC board will fail on the")
    print("  missing scenes on purpose. Submit the rest as a dc-cpu job, then re-run.")
else:
    print("  [demo] synthetic products are generated locally in the next cell")
    print(f"  target dir: {ZIP_DIR}")

# %% [markdown]
# ### Demo product builder
#
# Builds a small SAFE-shaped product — same directory layout, band naming, `nodata`
# metadata and per-band quantisation as a real L2A granule — and zips it, so the
# download/extraction path exercised below is the real code path. The scene is a
# ~5 km stub, not a 109 km granule, and its cloud field is generated independently of
# the reported scene-level `cloudCover` so Exercise 1's discrepancy is *computed*, not
# asserted.

# %%
# A real granule is 10980 px at 10 m (109.8 km). The stub is 1536 px (15.4 km), sized
# so it fully contains the demo AOI, and its origin is chosen in the same UTM 32N
# frame the AOI transforms into. Real code reads the CRS out of the file.
DEMO_SIZE_10M = 1536
DEMO_ORIGIN_X, DEMO_ORIGIN_Y = 680000.0, 5340000.0
DEMO_EPSG = "EPSG:32632"


#: Reflectance by band and surface class: water, vegetation, soil/urban, dark.
#: Vegetation is low in Red, high in Narrow-NIR, and low in SWIR-2 (lignin and
#: moisture absorption) -- the contrast that makes B11/B12 do real work.
DEMO_SURFACE_REFLECTANCE = {
    "B02": (0.030, 0.110, 0.100, 0.020), "B03": (0.040, 0.190, 0.120, 0.030),
    "B04": (0.040, 0.160, 0.110, 0.040), "B08": (0.050, 0.480, 0.150, 0.060),
    "B11": (0.100, 0.220, 0.210, 0.090), "B12": (0.080, 0.160, 0.180, 0.070),
}
#: Cloud top reflectance by band. Thin cloud sits BELOW the 0.45 Red threshold used
#: by the bright-pixel proxy, so the proxy under-counts it. That gap is the lesson.
DEMO_CLOUD_REFLECTANCE = {"B02": 0.40, "B03": 0.42, "B04": 0.44, "B08": 0.46,
                          "B11": 0.38, "B12": 0.30}


def demo_band_values(band, h, w, surface, cloud):
    """Reflectance-like DN for one band, given a synthetic surface and cloud field."""
    base = DEMO_SURFACE_REFLECTANCE[band]
    dn = np.select([surface == 0, surface == 1, surface == 2],
                   [base[0], base[1], base[2]], default=base[3])
    dn = dn + rng.normal(0, 0.008, size=(h, w))
    dn = np.where(cloud > 0.5, DEMO_CLOUD_REFLECTANCE[band] * rng.uniform(0.85, 1.6, size=(h, w)), dn)
    dn = np.clip(dn, 0.0, 0.95) * radiometry.QUANTIFICATION_VALUE
    return np.rint(dn).astype("uint16")


# %%
def write_demo_bands(img, name, surface, cloud_mask, cloud):
    """Write the six band rasters and the cloud QI layer for one demo product."""
    tf10 = rasterio.transform.from_origin(DEMO_ORIGIN_X, DEMO_ORIGIN_Y, 10.0, 10.0)
    h10, w10 = cloud.shape
    for band, res in (("B02", 10), ("B03", 10), ("B04", 10), ("B08", 10),
                      ("B11", 20), ("B12", 20)):
        step = res // 10
        arr = demo_band_values(band, h10 // step, w10 // step,
                               surface[::step, ::step], cloud[::step, ::step])
        prof = {"driver": "JP2OpenJPEG", "height": arr.shape[0], "width": arr.shape[1],
                "count": 1, "dtype": "uint16", "crs": DEMO_EPSG, "nodata": 0,
                "transform": rasterio.transform.from_origin(
                    DEMO_ORIGIN_X, DEMO_ORIGIN_Y, float(res), float(res))}
        with rasterio.open(img / f"R{res}m" / f"{name[:-5]}_{band}_{res}m.jp2",
                           "w", **prof) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, band)
    with rasterio.open(img / "QI_DATA" / "MSK_CLOUDS_B00.tif", "w", driver="GTiff",
                       height=h10, width=w10, count=1, dtype="uint8", crs=DEMO_EPSG,
                       nodata=255, transform=tf10) as dst:
        dst.write(cloud_mask, 1)


def demo_surface_and_cloud(product, h=DEMO_SIZE_10M, w=DEMO_SIZE_10M):
    """Synthetic land-cover surface and a cloud field for one stub granule.

    The cloud field is drawn independently of the reported scene-level percentage and
    then rescaled so the whole stub averages to it. The AOI is a sub-window, so its
    fraction will NOT equal the reported number -- that is Exercise 1, computed.
    """
    surface = np.ones((h, w), dtype="uint8")     # 1 vegetation
    surface[:, w // 2:] = 2                      # 2 soil / urban
    surface[h * 3 // 4:, :] = 0                  # 0 water
    surface = surface + (rng.random((h, w)) < 0.05).astype("uint8") * 3

    reported = float(product["Attributes"]["cloudCover"]) / 100.0
    cloud = np.zeros((h, w), dtype="float32")
    yy, xx = np.ogrid[:h, :w]
    for _ in range(int(rng.integers(2, 6))):
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        r = float(rng.uniform(0.05, 0.28)) * h
        cloud = np.maximum(cloud, np.clip(1.0 - ((yy - cy) ** 2 + (xx - cx) ** 2) / r ** 2, 0, 1))
    cloud = np.clip(cloud * (reported / max(float(cloud.mean()), 1e-6)), 0, 1)
    return surface, cloud


# %%
def build_demo_product(product, out_root):
    """Write a SAFE-shaped directory and zip it. Returns the zip path."""
    name = product["Name"]
    safe = Path(out_root) / name
    img = safe / "GRANULE" / (name.replace("S2", "L1C", 1)[:50] + "0") / "IMG_DATA"
    for sub in ("R10m", "R20m", "QI_DATA"):
        (img / sub).mkdir(parents=True, exist_ok=True)

    surface, cloud = demo_surface_and_cloud(product)
    write_demo_bands(img, name, surface, (cloud > 0.5).astype("uint8"), cloud)

    zip_dir = paths.data_root() / "zip"
    paths.ensure(zip_dir)
    out = zip_dir / zip_name(product)
    if not out.exists():
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(safe.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(Path(out_root)))
    return out


if MODE == "demo":
    zip_paths = [build_demo_product(p, paths.data_root() / "staging")
                 for p in selected_products]
    print(f"  built {len(zip_paths)} synthetic products in {ZIP_DIR}")
    for z in zip_paths:
        print(f"    {z.name}  {z.stat().st_size / 1e6:.1f} MB (synthetic stub)")

# %% [markdown]
# ## Part 10 — Extraction is a step, not a side effect of plotting
#
# The chain between Lab 3 and Lab 4 was broken by construction: Lab 3.1 downloaded
# `.zip` files, and Lab 4.1 globs `*.SAFE` **directories**. The only extraction code in
# the whole course lived inside a function called `extract_and_visualize_sentinel2`
# buried in a *visualisation* cell, and it only ran if an earlier path-guessing cell
# happened to find a valid zip. The instructor's own committed output showed Lab 4.1
# skipping all four tiles because pre-extracted directories were already sitting on
# scratch — so the hand-off had never actually been exercised end to end.
#
# Extraction is therefore an explicit, asserted step here, and it produces the layout
# Lab 4.1 reads: **flat `*.SAFE` directories directly under `paths.data_root()`**.
# (The 2025/26 notebook printed a tidy nested tree, commented out the `makedirs` call,
# and told students to create it "before Lab 3.2" — a lab that does not exist, and a
# layout Lab 4.1 would not have found.)

# %%
SAFE_ROOT = paths.data_root()   # lab4_1 globs *.SAFE here


def extract_one(zip_path, dest_root):
    """Unpack one product zip so `<dest_root>/<NAME>.SAFE/` exists. Idempotent."""
    dest_root = Path(dest_root)
    paths.ensure(dest_root)
    with zipfile.ZipFile(zip_path) as zf:
        top = zf.namelist()[0].split("/")[0]
        target = dest_root / top
        if not target.is_dir():
            zf.extractall(dest_root)
    return target


safe_dirs = [extract_one(z, SAFE_ROOT) for z in zip_paths]
for d in safe_dirs:
    assert d.is_dir(), f"extraction produced no directory: {d}"
found = sorted(SAFE_ROOT.glob("*.SAFE"))
print(f"  extracted {len(safe_dirs)} -> {SAFE_ROOT}")
print(f"  `*.SAFE` visible to Lab 4.1: {len(found)}")
for d in found:
    n_bands = len(list(d.glob("**/IMG_DATA/R*m/*_*m.jp2")))
    print(f"    {d.name[:56]:<56} bands={n_bands}")

# %% [markdown]
# ## Part 11 — Exercise 1 revealed: cloud over *your* AOI
#
# Now compute the number you predicted. Two diagnostics, because they have different
# availability and different meanings:
#
# 1. **Cloud-mask fraction.** The product's own cloud/no-cloud mask, read over the AOI
#    window only. In the demo this is `QI_DATA/MSK_CLOUDS_B00.tif`. In a real L2A SAFE
#    the equivalent is `GRANULE/*/IMG_DATA/R20m/QI_DATA/MSK_CLOUDS_B00.gml` — a *vector*
#    mask, not a raster — and CDSE's Zarr/STAC rendition of the same product exposes a
#    raster quality layer. If no raster quality layer is present, this function returns
#    `None` and the QC board **fails that check with instructions**. It does not pass it
#    quietly, which is precisely the 2025/26 failure mode.
# 2. **Bright-pixel proxy.** Fraction of AOI pixels with B04 reflectance above 0.45.
#    Always computable from the bands alone. It is a *proxy*: it also fires on seasonal
#    snow, bright bare soil, and concrete roofs. That is not a reason to skip it — it is
#    a reason to read it alongside the mask, and a reason the mask matters.

# %%
def aoi_slices(src, aoi_wkt_bbox):
    """Row/col slices covering the AOI bbox, in the raster's own CRS."""
    x0, y0, x1, y1 = aoi_wkt_bbox
    xs, ys = transform("EPSG:4326", src.crs, [x0, x1], [y0, y1])
    lo_x, hi_x = min(xs), max(xs)
    lo_y, hi_y = min(ys), max(ys)
    w = from_bounds(lo_x, lo_y, hi_x, hi_y, src.transform)
    c0 = max(int(np.floor(w.col_off)), 0)
    r0 = max(int(np.floor(w.row_off)), 0)
    c1 = min(int(np.ceil(w.col_off + w.width)), src.width)
    r1 = min(int(np.ceil(w.row_off + w.height)), src.height)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("AOI falls outside this raster; check the tile you chose")
    return slice(r0, r1), slice(c0, c1)


def find_quality_raster(safe_dir):
    """First cloud/quality raster in the product, or None. Never guesses silently."""
    for pat in ("**/QI_DATA/MSK_CLOUDS_B00.tif", "**/*qi_info*.tif", "**/*_SCL*.*",
                "**/MSK_CLOUDS*"):
        hits = sorted(Path(safe_dir).glob(pat))
        if hits:
            return hits[0]
    return None


AOI_BBOX = (min_lon, min_lat, max_lon, max_lat)
print(f"  AOI bbox (EPSG:4326): {AOI_BBOX}")
print("  CRS is read from each raster, never derived from the MGRS band number.")

# %%
def aoi_cloud_fraction(safe_dir):
    """Cloud and no-data fractions over the AOI, from the product's own metadata.

    Returns (aoi_cloud_frac, aoi_bright_proxy_frac, aoi_nodata_frac, quality_path).
    The no-data fraction uses the band's *declared* nodata, which is the only correct
    sentinel: DN 0 is legitimate dark reflectance (deep water, shadow, asphalt).
    """
    q = find_quality_raster(safe_dir)
    red = sorted(Path(safe_dir).glob("**/R10m/*_B04_10m.jp2"))
    frac = None
    if q is not None:
        with rasterio.open(q) as src:
            rs, cs = aoi_slices(src, AOI_BBOX)
            arr = src.read(1, window=(rs, cs), masked=True)
            valid = ~arr.mask
            if int(valid.sum()) == 0:
                raise ValueError("quality layer is fully no-data over the AOI")
            frac = float((arr.data[valid] > 0).mean())
    proxy = nodata = None
    if red:
        with rasterio.open(red[0]) as src:
            rs, cs = aoi_slices(src, AOI_BBOX)
            dn = src.read(1, window=(rs, cs), masked=True)
            nodata = float(np.ma.getmaskarray(dn).mean())
            # Fill nodata with 0 so it counts as dark, not bright: the proxy measures
            # bright pixels, and a nodata pixel is neither bright nor a measurement.
            proxy = float((radiometry.dn_to_reflectance(dn.filled(0)) > 0.45).mean())
    return frac, proxy, nodata, q


CLOUD_ACTUAL = []
print(f"  {'date':<12} {'reported %':>11} {'AOI mask %':>11} {'AOI bright %':>13} "
      f"{'AOI nodata %':>13}  quality layer")
for p, d in zip(selected_products, safe_dirs):
    rep = float(p["Attributes"]["cloudCover"])
    frac, proxy, nodata, q = aoi_cloud_fraction(d)
    CLOUD_ACTUAL.append({"date": p["ContentDate"]["Start"][:10], "reported": rep,
                         "aoi_mask": frac, "aoi_bright_proxy": proxy,
                         "aoi_nodata": nodata})
    a = "n/a" if frac is None else f"{100 * frac:.1f}"
    b = "n/a" if proxy is None else f"{100 * proxy:.1f}"
    nod = "n/a" if nodata is None else f"{100 * nodata:.1f}"
    print(f"  {p['ContentDate']['Start'][:10]:<12} {rep:>11.1f} {a:>11} {b:>13} {nod:>13}  "
          f"{q.name if q else 'MISSING -> QC will fail'}")

# %%
gaps = [abs(c["reported"] - 100 * c["aoi_mask"]) for c in CLOUD_ACTUAL
        if c["aoi_mask"] is not None]
print("\n  PREDICTED vs ACTUAL — cloud over AOI")
print(f"    predicted AOI % : {PREDICTIONS['cloud']['aoi'] or '(not recorded)'}")
print(f"    actual AOI %    : {[None if c['aoi_mask'] is None else round(100 * c['aoi_mask'], 1) for c in CLOUD_ACTUAL]}")
print(f"    predicted max gap (pp): {PREDICTIONS['cloud']['max_gap_pp'] or '(not recorded)'}")
print(f"    actual max gap (pp)   : {round(max(gaps), 1) if gaps else 'n/a'}")
print("    Write one sentence on any disagreement. 'The reported number is scene-level'")
print("    is the mechanism; say which of your four scenes it hurt and by how much.")

# %% [markdown]
# ## Part 12 — Visualisation, done correctly
#
# Two defects in the 2025/26 stretch, both of which change what you can see:
#
# * **Pooled percentiles.** `np.percentile(rgb, (2, 98))` over a stacked `(H, W, 3)`
#   array gives Blue, Green and Red *one shared* pair of bounds. Narrow-NIR is far
#   brighter than Blue over vegetation, so a shared stretch drags the bounds up and
#   squeezes Blue toward a constant — destroying the band that most separates urban
#   fabric from vegetation. Fix: per-band bounds, which is what
#   `radiometry.per_band_percentiles` returns.
# * **`rgb > 0` as the nodata test.** In L2A, DN 0 is a *legitimate* dark reflectance:
#   deep open water, cast shadow, asphalt. Discarding it brightens exactly the classes
#   you are trying to separate. The sentinel is the band's declared `nodata` metadata,
#   which we read. (Lab 4.1 in 2025/26 used the same pooled stretch *without* the
#   `> 0` filter — two notebooks, two stretches, no explanation of the difference.)
#
# And the figure is written to `paths.results_dir()`, whose directory is created first.
# The old cell saved to `$PROJECT/<user>/results/S2_vis.png` without creating the
# directory, raising `FileNotFoundError` on any machine where Lab 2 had not happened to
# make it.

# %%
def read_band(safe_dir, band, res=10, factor=8):
    """AOI window of one band, decimated, masked at the band's declared nodata."""
    hits = sorted(Path(safe_dir).glob(f"**/R{res}m/*_{band}_{res}m.jp2"))
    if not hits:
        raise FileNotFoundError(f"{band} not found under {safe_dir}; check the SAFE layout")
    with rasterio.open(hits[0]) as src:
        rs, cs = aoi_slices(src, AOI_BBOX)
        h = max((rs.stop - rs.start) // factor, 1)
        w = max((cs.stop - cs.start) // factor, 1)
        arr = src.read(1, window=(rs, cs), out_shape=(h, w),
                       resampling=Resampling.average, masked=True)
        meta = {"crs": str(src.crs), "nodata": src.nodata, "dtype": src.dtypes[0],
                "full_shape": src.shape, "res": src.res}
    return arr, meta


arr_r, meta_r = read_band(safe_dirs[0], "B04")
print(f"  CRS from file : {meta_r['crs']}  (do not derive this from the tile name)")
print(f"  nodata        : {meta_r['nodata']}   dtype: {meta_r['dtype']}   "
      f"full granule: {meta_r['full_shape']} at {meta_r['res'][0]} m")
print(f"  AOI window    : {arr_r.shape[0]} x {arr_r.shape[1]} after 1/8 decimation")
print(f"  DN range      : {int(arr_r.min())} .. {int(arr_r.max())}")
print(f"  reflectance   : {radiometry.dn_to_reflectance(arr_r.min()):.3f} .. "
      f"{radiometry.dn_to_reflectance(arr_r.max()):.3f}")
print(f"  masked pixels : {int(np.ma.getmaskarray(arr_r).sum())} "
      f"(declared nodata only -- DN 0 is NOT masked)")

# %%
stack = np.ma.stack([read_band(safe_dirs[0], b)[0] for b in ("B04", "B03", "B02")])
stack = stack.astype("float32")                     # (3, H, W), masked where nodata
vis = radiometry.Norm.fit(np.ma.filled(stack, np.nan), mode="percentile",
                          channel_axis=0, low_pct=2.0, high_pct=98.0)
stretch = vis.apply(np.ma.filled(stack, np.nan))    # (3, H, W) in [0, 1]; nodata -> 0
nodata_mask = np.ma.getmaskarray(stack[0])
print(f"  {vis}")
print("  per-band 2/98 percentile bounds (DN):")
for band, lo, hi in zip(("B04", "B03", "B02"), vis.low_, vis.high_):
    print(f"    {band}: {lo:.0f} .. {hi:.0f}")
print("  A pooled stretch would have used one pair for all three; Blue's range is the")
print("  narrowest, so it is the band that loses contrast first.")
print("  This is the same Norm class Lab 4.2 uses -- and it is fitted on ONE scene,")
print("  for a picture. Fitting it on your whole dataset is a different, later decision.")

# %%
nodata_mask = np.ma.filled(stack[0].mask, False) if np.ma.getmask(stack[0]) is not np.ma.nomask else np.zeros(stack.shape[1:], dtype=bool)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
axes[0].imshow(np.transpose(stretch, (1, 2, 0)))
axes[0].set_title(f"RGB per-band stretch, {dates[0]}")
axes[0].axis("off")
axes[1].imshow(np.transpose(stretch, (1, 2, 0)))
axes[1].contour(nodata_mask.astype(float), levels=[0.5], colors="red", linewidths=0.6)
axes[1].set_title("declared nodata outlined (not DN==0)")
axes[1].axis("off")
for i, band in enumerate(("B04", "B03", "B02")):
    axes[2].hist(stack[i].compressed()[::37], bins=60, alpha=0.6, label=band)
axes[2].set_title("DN histograms (dark tail kept, nodata excluded)")
axes[2].set_xlabel("DN")
axes[2].set_yscale("log")
axes[2].legend()
fig.tight_layout()
fig_path = paths.results_dir() / "lab3_rgb_stretch.png"
paths.ensure(fig_path.parent)
fig.savefig(fig_path, dpi=110)
plt.show()
print(f"  saved {fig_path}")

# %% [markdown]
# ## Part 13 — The manifest: the artifact the old deliverable never had
#
# The 2025/26 deliverable was "downloaded scenes + metadata". There was no metadata
# artifact. `products` was a live Python list that died with the kernel, so nothing
# recorded which tile, which dates, which cloud covers, or which bands a team had
# actually chosen — and nothing could be graded.
#
# `manifest.json` is what Lab 4.1 reads and what the grader checks. It is written
# before you leave this notebook.

# %%
manifest = {
    "schema": "lab3-manifest/1",
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "user": paths.username(),
    "mode": MODE,
    "synthetic": bool(synthetic),
    "tile": SELECTED_TILE,
    "tile_criterion": TILE_CRITERION,
    "aoi": {"name": AOI_NAME, "bbox_epsg4326": list(AOI_BBOX)},
    "date_window": {"start": START_DATE[:10], "end": END_DATE[:10],
                    "reason": "CORINE Land Cover 2018 vintage; labels and imagery "
                              "must be temporally coherent"},
    "max_reported_cloud_pct": MAX_CLOUD,
    "n_scenes_required": N_SCENES,
    "bands": list(BANDS),
    "n_bands": N_BANDS,
    "band_reason": BAND_REASON,
    "quantification_value": radiometry.QUANTIFICATION_VALUE,
    "scenes": [
        {
            "product_id": p["Id"],
            "name": p["Name"],
            "sensing_date": p["ContentDate"]["Start"][:10],
            "reported_cloud_pct": float(p["Attributes"]["cloudCover"]),
            "aoi_cloud_mask_pct": c["aoi_mask"],
            "aoi_bright_proxy_pct": c["aoi_bright_proxy"],
            "aoi_nodata_pct": c["aoi_nodata"],
            "zip": str(z),
            "safe_dir": str(d),
            "zip_bytes": Path(z).stat().st_size if z else 0,
        }
        for p, c, z, d in zip(selected_products, CLOUD_ACTUAL, zip_paths, safe_dirs)
    ],
    "storage": {"zip_dir": str(ZIP_DIR), "safe_root": str(SAFE_ROOT),
                "results_dir": str(paths.results_dir())},
}
manifest_path = paths.results_dir() / "manifest.json"
paths.ensure(manifest_path.parent)
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"  wrote {manifest_path}  ({manifest_path.stat().st_size} bytes)")

# %%
back = json.loads(paths.require_existing(manifest_path, "lab3 manifest.json").read_text(encoding="utf-8"))
assert back["tile"] == SELECTED_TILE
assert len(back["scenes"]) == N_SCENES
assert {s["name"].split("_")[0] for s in back["scenes"]}
print(f"  read back: tile={back['tile']} scenes={len(back['scenes'])} "
      f"bands={back['bands']} synthetic={back['synthetic']}")
print("  Lab 4.1 should read this file instead of globbing scratch and hoping.")

# %% [markdown]
# ## Part 14 — QC board
#
# The old notebook printed `✓ Visualization complete!` from a fallback branch that
# matched **any** `.SAFE` directory in the data folder. A team that downloaded the wrong
# tile entirely, or the wrong dates, or one scene instead of four, still saw success.
#
# These checks are the replacement. They are written as gates, in the same style as
# `eo_course.gates`: a failure is a claim that is not supported, and it raises.

# %%
MIN_ZIP_BYTES_LIVE = int(0.8 * 1024 ** 3)  # a real L2A product is ~1.2 GB; below 0.8 GB = truncated
MIN_ZIP_BYTES_DEMO = 1 * 1024 ** 2         # synthetic stubs are small; scaled, and said so


def qc_metadata(man, n_expected=N_SCENES):
    """Claims about the *selection*: count, tile, window, cloud, uniqueness."""
    fails = []
    scenes = man["scenes"]
    if len(scenes) != n_expected:
        fails.append(f"scene count is {len(scenes)}, must be exactly {n_expected}")
    tiles = {TILE_RE.search(s["name"]).group(1) for s in scenes if TILE_RE.search(s["name"])}
    if tiles != {man["tile"]}:
        fails.append(f"scenes span more than one tile: {sorted(tiles)}")
    for s in scenes:
        d = s["sensing_date"]
        if not (START_DATE[:10] <= d <= END_DATE[:10]):
            fails.append(f"{d} is outside the CORINE-2018 window")
        if s["reported_cloud_pct"] > man["max_reported_cloud_pct"]:
            fails.append(f"{d} reported cloud {s['reported_cloud_pct']:.1f}% > "
                         f"{man['max_reported_cloud_pct']:.0f}%")
    if len({s["sensing_date"] for s in scenes}) != len(scenes):
        fails.append("duplicate sensing dates")
    return fails


# %%
def qc_files(man):
    """Claims about the *artifacts*: they exist, they are whole, they are usable."""
    fails, notes = [], []
    floor = MIN_ZIP_BYTES_DEMO if man["synthetic"] else MIN_ZIP_BYTES_LIVE
    if man["synthetic"]:
        notes.append(f"size floor scaled for synthetic stubs; real floor is "
                     f"{MIN_ZIP_BYTES_LIVE / 1e9:.2f} GB")
    for s in man["scenes"]:
        z = Path(s["zip"]) if s["zip"] else None
        if z is None or not z.exists():
            fails.append(f"missing zip for {s['sensing_date']} -- run the rest as a "
                         "dc-cpu job, do not download on the login node")
        else:
            size = z.stat().st_size
            if size < floor:
                fails.append(f"{z.name} is {size / 1e6:.1f} MB, below the "
                             f"{floor / 1e6:.0f} MB floor -- truncated download")
            elif not zip_is_intact(z):
                fails.append(f"{z.name} is a corrupt archive -- delete it and re-download")
        if not Path(s["safe_dir"]).is_dir():
            fails.append(f"{s['safe_dir']} was never extracted")
        if s["aoi_cloud_mask_pct"] is None:
            fails.append(f"{s['sensing_date']}: no cloud/quality raster found, so AOI "
                         "cloud cover is unverified. Obtain the QI layer (SAFE "
                         "MSK_CLOUDS_B00.gml, or the CDSE Zarr quality band).")
        elif s["aoi_cloud_mask_pct"] > 0.5:
            fails.append(f"{s['sensing_date']}: {100 * s['aoi_cloud_mask_pct']:.0f}% of "
                         "the AOI is cloud, whatever the scene-level number said")
        nd = s.get("aoi_nodata_pct")
        if nd is not None and nd > 0.2:
            fails.append(f"{s['sensing_date']}: {100 * nd:.0f}% of the AOI is declared "
                         "no-data in B04 -- the tile does not really cover your AOI")
    return fails, notes


def qc_checks(man, n_expected=N_SCENES):
    file_fails, notes = qc_files(man)
    return qc_metadata(man, n_expected) + file_fails, notes


qc_fails, qc_notes = qc_checks(back)
for nline in qc_notes:
    print(f"  note: {nline}")
print(f"  QC failures: {len(qc_fails)}")
for f in qc_fails:
    print(f"    - {f}")

# %% [markdown]
# ## Part 14b — Negative control: prove the QC board can fail
#
# A check you have never seen fail is a check you have not verified. The 2025/26
# notebook's `✓ Visualization complete!` fired from a fallback that matched *any*
# `.SAFE` directory, so it was green for the wrong tile, the wrong dates and the wrong
# scene count simultaneously — a check that cannot fail is not a check.
#
# Below, the same `qc_checks` runs against four deliberately broken copies of your own
# manifest. **Write down which of the four you expect to be caught before running it.**
#
# > **P4 (record in `PREDICTIONS["negative_control"]`).** How many of the four
# > corruptions produce at least one failure — 0, 1, 2, 3 or 4? Name the one you think
# > will slip through, and why.
#
# The interesting case is the one that survives: if a corruption you consider serious
# passes, the QC board has a hole, and that is a finding worth writing up.

# %%
corruptions = {
    "wrong tile": lambda m: m["scenes"].__setitem__(
        0, {**m["scenes"][0], "name": m["scenes"][0]["name"].replace(m["tile"], "T31TFI")}),
    "year 2024 imagery vs CORINE 2018": lambda m: m["scenes"].__setitem__(
        1, {**m["scenes"][1], "sensing_date": "2024-06-15"}),
    "reported cloud 44%": lambda m: m["scenes"].__setitem__(
        3, {**m["scenes"][3], "reported_cloud_pct": 44.0}),
    "manifest lists band B08A that is not on disk": lambda m: m["bands"].append("B8A"),
}
PREDICTIONS["negative_control"] = {"caught_of_4": None, "would_slip_through": "", "why": ""}

n_caught = 0
for label, corrupt in corruptions.items():
    bad = copy.deepcopy(back)
    corrupt(bad)
    found = qc_checks(bad)[0]
    n_caught += bool(found)
    print(f"  {label:<46} -> {len(found)} failure(s)"
          f"{': ' + found[0][:70] if found else '   NOT CAUGHT'}")
print(f"\n  caught {n_caught} of {len(corruptions)}")
print(f"  your prediction: {PREDICTIONS['negative_control']['caught_of_4']!r} of 4")
print("  The last one is a real hole: qc_checks reads the manifest and the zips, and")
print("  never opens a band file. A granule that unpacked with bands missing, or a")
print("  4-band download recorded as 6, passes every check above and fails in Lab 4.1.")
print("  Write one sentence on the cheapest assertion that would close it.")

# %% [markdown]
# ## Part 15 — Gate board
#
# `eo_course.gates` raises rather than warns, because a gate is a claim. The
# performance gates (`gate_beats_baselines`, `gate_above_chance`, …) belong to Labs 5–7;
# here the claims are about the *data*: that four scenes exist, from one tile, in the
# right window, over your AOI, in units the models downstream expect.

# %%
def gate(name, ok, detail):
    """One board line, in the same shape eo_course.gates renders."""
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def pct(v):
    return "n/a" if v is None else round(100 * v, 1)


def nodata_of(s):
    """Declared-nodata fraction over the AOI. None means unmeasured, and unmeasured
    must not pass -- that is the failure mode this whole lab is a correction of."""
    return s.get("aoi_nodata_pct")


x0, y0, x1, y1 = AOI_BBOX
xs, ys = transform("EPSG:4326", meta_r["crs"], [x0, x1], [y0, y1])
aoi_km2 = ((max(xs) - min(xs)) / 1e3) * ((max(ys) - min(ys)) / 1e3)
scene_tiles = {TILE_RE.search(s["name"]).group(1) for s in back["scenes"] if TILE_RE.search(s["name"])}
span_days = (datetime.fromisoformat(back["scenes"][-1]["sensing_date"])
             - datetime.fromisoformat(back["scenes"][0]["sensing_date"])).days
worst_aoi = max((s["aoi_cloud_mask_pct"] for s in back["scenes"]
                 if s["aoi_cloud_mask_pct"] is not None), default=1.0)

# Claims about the *selection* -- what you asked the catalog for.
board = [
    gate("scene_count", len(back["scenes"]) == N_SCENES,
         f"{len(back['scenes'])} scenes, need {N_SCENES}"),
    gate("single_tile", scene_tiles == {back["tile"]}, f"tile(s) present: {sorted(scene_tiles)}"),
    gate("corine_2018_window",
         all(START_DATE[:10] <= s["sensing_date"] <= END_DATE[:10] for s in back["scenes"]),
         f"{back['scenes'][0]['sensing_date']} .. {back['scenes'][-1]['sensing_date']}"),
    gate("reported_cloud_le_30",
         all(s["reported_cloud_pct"] <= MAX_CLOUD for s in back["scenes"]),
         f"max reported {max(s['reported_cloud_pct'] for s in back['scenes']):.1f}%"),
    gate("temporal_spread", span_days >= 150, f"span {span_days} days, need >= 150"),
    gate("aoi_inside_tile", 20 <= aoi_km2 <= 5000, f"AOI covers {aoi_km2:.0f} km^2"),
]

# %%
# Claims about the *data* -- what you actually got, and what you decided.
# NB `v or 1.0` would be wrong in aoi_cloud_under_50pct: a scene that is 0.0 cloudy
# over the AOI is falsy and would be scored as fully cloudy. Compare to None.
board += [
    gate("aoi_cloud_verified",
         all(s["aoi_cloud_mask_pct"] is not None for s in back["scenes"]),
         f"AOI mask % = {[pct(s['aoi_cloud_mask_pct']) for s in back['scenes']]}"),
    gate("aoi_cloud_under_50pct",
         all(s["aoi_cloud_mask_pct"] is not None and s["aoi_cloud_mask_pct"] <= 0.5
             for s in back["scenes"]),
         f"worst AOI scene = {worst_aoi:.0%}"),
    gate("aoi_nodata_under_20pct",
         all(nodata_of(s) is not None and nodata_of(s) <= 0.2 for s in back["scenes"]),
         f"worst AOI nodata = "
         f"{max((v for s in back['scenes'] if (v := nodata_of(s)) is not None), default=1.0):.0%} "
         f"(declared nodata, not DN==0)"),
    gate("files_on_disk",
         all(Path(s["zip"]).exists() and Path(s["safe_dir"]).is_dir() for s in back["scenes"]),
         f"{sum(Path(s['zip']).exists() for s in back['scenes'])}/{N_SCENES} zips, "
         f"{sum(Path(s['safe_dir']).is_dir() for s in back['scenes'])}/{N_SCENES} SAFE dirs"),
    gate("lab4_handoff", len(list(SAFE_ROOT.glob("*.SAFE"))) >= N_SCENES,
         f"{len(list(SAFE_ROOT.glob('*.SAFE')))} *.SAFE under {SAFE_ROOT}"),
    gate("manifest_written", manifest_path.exists(),
         f"{manifest_path.name}, {manifest_path.stat().st_size} bytes"),
    gate("band_set_recorded", "TODO" not in back["band_reason"],
         f"{back['n_bands']} bands {back['bands']}; reason = "
         f"{'still the TODO placeholder' if 'TODO' in back['band_reason'] else 'recorded'}"),
    gate("tile_criterion_recorded", "TODO" not in back["tile_criterion"],
         "criterion = " + ("still the TODO placeholder -- name the criterion and the "
                           "number behind it" if "TODO" in back["tile_criterion"]
                           else "recorded")),
]
try:
    radiometry.assert_reflectance_range(radiometry.dn_to_reflectance(arr_r), "B04 reflectance")
    board.append(gate("units_are_reflectance", True, "DN/10000 verified in [0,1]"))
except AssertionError as exc:
    board.append(gate("units_are_reflectance", False, str(exc).splitlines()[0]))

gates.print_gate_board(board)

# %% [markdown]
# ### Reading a failed board
#
# A red line here is not a formatting problem. `files_on_disk` failing means you
# downloaded on a login node and stopped; `aoi_cloud_verified` failing means you have
# four scenes and no evidence they are usable; `temporal_spread` failing means you
# picked Set B and gave up the season coverage you were supposed to buy. Report the
# failure and its cause; do not re-run until it is green by quietly loosening a
# threshold.
#
# Two gates are red **on first run by design**: `band_set_recorded` and
# `tile_criterion_recorded` fail while `BAND_REASON` and `TILE_CRITERION` still contain
# the `TODO` placeholder. They are not code bugs and no amount of re-running will turn
# them green. They are the notebook checking that you made the two decisions this lab
# exists to make, and wrote them down.

# %%
run_id = f"lab3_{back['tile']}_{back['scenes'][0]['sensing_date'].replace('-', '')}"
qc_pass = sum("[PASS]" in b for b in board)
n_checks = len(board)
test_metrics = {
    "n": int(sum(s["zip_bytes"] for s in back["scenes"])),
    "overall_acc": qc_pass / n_checks,
    "balanced_acc": qc_pass / n_checks,
    "macro_f1": qc_pass / n_checks,
}
notes = (
    f"Lab 3 acquisition, mode={back['mode']}, synthetic={back['synthetic']}. "
    f"QC {qc_pass}/{n_checks} gates passed. "
    f"AOI cloud {[None if s['aoi_cloud_mask_pct'] is None else round(100 * s['aoi_cloud_mask_pct'], 1) for s in back['scenes']]}% "
    f"vs reported {[s['reported_cloud_pct'] for s in back['scenes']]}%. "
    f"NOTE: this lab has no held-out split, so the required keys carry the acquisition "
    "QC pass fraction, not model metrics; n is total bytes acquired. The graded content "
    "is in config and extra."
)
rec_kwargs = dict(
    lab="lab3",
    config={
        "tile": back["tile"], "tile_criterion": back["tile_criterion"],
        "aoi_bbox": list(AOI_BBOX), "dates": [s["sensing_date"] for s in back["scenes"]],
        "bands": back["bands"], "n_bands": back["n_bands"],
        "band_reason": back["band_reason"],
        "max_reported_cloud_pct": MAX_CLOUD,
        "quantification_value": radiometry.QUANTIFICATION_VALUE,
        "mode": back["mode"], "synthetic": back["synthetic"], "seed": SEED,
    },
    split_manifest_hash=None,
    seed=SEED,
    test_metrics=test_metrics,
    notes=notes,
    extra={"cloud": CLOUD_ACTUAL, "gate_board": board, "predictions": PREDICTIONS,
           "manifest": str(manifest_path), "qc_failures": qc_fails},
)
try:
    rec = results.record_run(run_id, **rec_kwargs)
    print(f"  recorded run_id={rec['run_id']}")
except results.ResultsError as exc:
    print(f"  not recorded: {exc}")
    print("  results.json is append-only. Re-run with a new run_id (the tile+date "
          "suffix changes when you change your selection) so the earlier attempt stays.")

print()
print(results.summary_table("lab3"))

# %% [markdown]
# ## Submission checklist
#
# Everything here is a file the grader can open. Self-attestation is not an artifact.
#
# * `results/manifest.json` — tile, criterion, AOI bbox, four dates, product IDs,
#   reported **and** AOI cloud cover, band list with reason, storage paths,
#   `synthetic: false`.
# * `results/results.json` — one `lab3` record from `results.record_run`, with the
#   gate board and your three predictions inside `extra`.
# * `results/lab3_rgb_stretch.png` — per-band stretch, nodata from metadata.
# * `<scratch>/<user>/data/*.SAFE` — four extracted granules, flat, where Lab 4.1 looks.
# * `<scratch>/<user>/data/zip/*.zip` — four zips at full size.
# * In your write-up: the gate board printed, with every `FAIL` explained; the
#   predicted-vs-actual lines for cloud cover and temporal selection; and the band
#   decision with the SWIR consequence stated.
#
# **Before Lab 4.1**, confirm two things by hand: `mode` is `live` and `synthetic` is
# `false` in the manifest, and your tile is on the course claims list. Demo output does
# not count, and a tile two other teams already have is a collision you will discover in
# Lab 5, not Lab 3.
#
# ### What was wrong here in 2025/26, in one list
#
# | defect | consequence |
# |---|---|
# | client secret prefix printed to output | credential fragment in a committed, shared artifact |
# | `os.getenv(ID, <literal id>)` | ran against someone else's client; notebook committed broken, 10 cells never executed |
# | Bavarian bbox in an `iceland-ml` repo | nobody could tell if it was a mistake |
# | CORINE 2018 vintage never stated | imagery/label mismatch invisible in the confusion matrix |
# | `"$top": 1000`, no pagination | "Found 1000 products" was the page size; per-tile counts wrong |
# | `split('_')[5]` tile parse | malformed names silently became `Unknown` |
# | tile = most scenes | chose overlap with a wide box, not the AOI's tile |
# | `products[0]` downloaded after choosing a tile | downloaded the wrong tile |
# | `<name>.SAFE.zip` naming | forced ~90 lines of path guessing |
# | ROPC password-grant "alternate" download | deprecated grant, plaintext password, raised on import |
# | pooled percentile stretch, `> 0` as nodata | Blue's contrast crushed; dark water/shadow discarded |
# | `savefig` into a directory never created | `FileNotFoundError` |
# | extraction only inside a plotting helper | Lab 4.1 globs `*.SAFE`; the hand-off never ran |
# | `✓ Visualization complete!` on any `.SAFE` | wrong tile still looked like success |
# | no metadata artifact | the deliverable could not be graded |
# | `COPERNICUS_SETUP.md` referenced, never shipped | instructions existed only on the instructor's copy |
# | GEE cell importing `geemap` | not installed; `ee.Authenticate()` on a headless node needs the manual code flow, never described |
#
# The GEE path is not in this notebook at all. If you want it, install the optional
# extra (`uv sync --extra gee`) and use the service-account or `--manual`
# authentication flow — `ee.Authenticate()` opens a browser that does not exist on a
# login node.
