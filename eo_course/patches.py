"""Patch extraction that keeps provenance, and honest per-band normalisation.

Why provenance is mandatory
---------------------------
The 2025/26 pipeline threw away scene identity and patch position at the moment
of extraction: lab4_2 wrote ``np.savez(patches=..., labels=...)`` and
``combine_datasets`` then concatenated all scenes and shuffled. By the time a
model was trained, it was *impossible* to build a leakage-free split even if you
wanted one -- there was no group variable left. That is why every lab fell back
to a random permutation over spatially contiguous patches.

So :func:`extract_patches` returns, for every patch, its scene id and its row/col
within the scene, and :func:`save_patches` persists them. :mod:`eo_course.splits`
consumes those two arrays. This is the one interface change that makes the whole
back half of the course honest.

Other fixes relative to lab4_2
------------------------------
* ``nodata`` is read from the raster's own metadata instead of assuming ``0`` is
  no-data. DN 0 is a legitimate dark-target value (clear deep water, shadow,
  asphalt), so the old ``if np.any(patch == 0): continue`` discarded exactly the
  dark surfaces that make water and shadow classes hard -- and it was the *only*
  no-data guard, because lab4_1 never wrote a ``nodata`` value.
* Normalisation statistics are **not** computed here over the whole image. The
  old comment said "Apply normalization to entire image first (for consistent
  statistics)" and that is leakage: those statistics were then applied to the
  test split. You extract raw patches, split, then fit :class:`eo_course.radiometry.Norm`
  on the training patches only.
* Labels are stored ``int64``, because ``F.cross_entropy`` needs int64 and the old
  ``uint8`` storage forced a cast in every downstream notebook.
* ``MAX_PATCHES`` truncation in raster-scan order is removed. The old code kept
  the first 50 000 windows, i.e. the **north-west corner** of each scene -- a
  systematic spatial bias that was invisible. Subsampling is now explicitly
  random and seeded, or absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eo_course import labels as lab_mod

#: Filename pattern for one scene's archive. Deliberately does NOT end in
#: ``_data.npz``: lab5_2's ``glob("*_data.npz")`` matched both the per-scene files
#: and ``combined_training_data.npz``, double-loading every patch. See SCENE_SUFFIX.
SCENE_PREFIX = "patches_"
SCENE_SUFFIX = "_scene.npz"
SCENE_GLOB = f"{SCENE_PREFIX}*{SCENE_SUFFIX}"


@dataclass
class PatchSet:
    """Patches plus the provenance needed to split them honestly."""

    patches: np.ndarray          # (N, H, W, C) float32, raw units as read
    labels: np.ndarray           # (N,) int64 CORINE codes
    scene: np.ndarray            # (N,) object/str scene id
    row: np.ndarray              # (N,) int32 top-left row within scene
    col: np.ndarray              # (N,) int32 top-left col within scene
    bands: tuple[str, ...]
    patch_size: int
    stride: int
    stats: dict

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    @property
    def class_codes(self) -> tuple[int, ...]:
        return tuple(int(c) for c in np.unique(self.labels))

    def class_counts(self, codes=None) -> np.ndarray:
        return lab_mod.class_counts(self.labels, codes if codes is not None else self.class_codes)

    def block_ids(self, block: int = 10):
        """Spatial super-cell id per patch, for block-level splitting."""
        from eo_course.splits import block_ids

        n_rows = int(self.row.max()) + 1
        n_cols = int(self.col.max()) + 1
        return block_ids(self.row, self.col, n_rows, n_cols, block=block)

    def subset(self, idx) -> PatchSet:
        idx = np.asarray(idx)
        return PatchSet(
            patches=self.patches[idx],
            labels=self.labels[idx],
            scene=self.scene[idx],
            row=self.row[idx],
            col=self.col[idx],
            bands=self.bands,
            patch_size=self.patch_size,
            stride=self.stride,
            stats={**self.stats, "subset_of": self.stats.get("source", "unknown")},
        )

    def summary(self) -> str:
        codes = self.class_codes
        counts = self.class_counts(codes)
        lines = [
            f"  patches        : {self.patches.shape} dtype={self.patches.dtype}",
            f"  labels         : {self.labels.shape} dtype={self.labels.dtype}",
            f"  scenes         : {len(np.unique(self.scene))} -> {sorted(map(str, np.unique(self.scene)))[:6]}",
            f"  patch/stride   : {self.patch_size}/{self.stride}",
            f"  bands          : {list(self.bands)}",
            f"  classes        : {len(codes)} {list(codes)}",
            f"  imbalance      : {lab_mod.imbalance_ratio(counts):.1f}x (max/min)",
            f"  skipped        : {self.stats.get('skipped', {})}",
        ]
        return "\n".join(lines)


def extract_patches(
    image_path,
    label_path,
    scene_id: str,
    patch_size: int = 3,
    stride: int | None = None,
    valid_codes=None,
    nodata_label: int = lab_mod.CORINE_NODATA,
    max_patches: int | None = None,
    seed: int = 0,
    window_rows: int = 512,
    label_name: str = "center",
    verbose: bool = True,
) -> PatchSet:
    """Extract patches from one scene, keeping scene/row/col for every patch.

    Parameters
    ----------
    image_path : str | Path
        Band-stacked GeoTIFF from lab 4.1, ``(C, H, W)``, raw DN.
    label_path : str | Path
        CORINE raster reprojected onto the image grid (lab 4.1 output).
    patch_size : int
        Window edge in pixels. Odd values are strongly preferred: with an even
        size there is no centre pixel, and lab4_2's ``y + patch_size // 2`` label
        silently shifts half a pixel up-left.
    stride : int | None
        Step between windows. ``None`` means ``patch_size`` (non-overlapping).
        **Use stride > patch_size** (2-4x) if you intend to split randomly:
        adjacent windows share pixels and the same 100 m CORINE label support.
    max_patches : int | None
        Random subsample cap. Unlike the old raster-order truncation this is
        unbiased; the old one kept the north-west corner of every scene.
    label_name : {"center", "majority"}
        ``"center"`` uses the label under the patch centre. ``"majority"`` uses
        the modal label over the window, which is the better choice for
        ``patch_size >= 5`` since a 100 m label spans ~10x10 image pixels.

    Notes
    -----
    Reads are windowed. The old code did ``src.read()`` on a 10980x10980x4 uint16
    scene (~960 MB) and then made a float32 copy (~1.9 GB) on a *login node*.
    """
    import rasterio
    from rasterio.windows import Window

    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")
    if patch_size % 2 == 0 and label_name == "center":
        raise ValueError(
            f"patch_size={patch_size} is even, so there is no centre pixel. Use an odd size, "
            "or pass label_name='majority'."
        )
    stride = patch_size if stride is None else int(stride)
    if stride < 1:
        raise ValueError("stride must be >= 1")
    valid = {int(c) for c in (valid_codes if valid_codes is not None else lab_mod.valid_class_codes())}

    from eo_course import paths as _paths

    image_path = _paths.require_existing(image_path, "image (lab 4.1 stacked output)")
    label_path = _paths.require_existing(label_path, "label raster (lab 4.1 reprojected CORINE)")

    with rasterio.open(image_path) as src, rasterio.open(label_path) as lsrc:
        n_bands = src.count
        bands = tuple(src.descriptions[:n_bands]) if src.descriptions else tuple(f"B{i}" for i in range(n_bands))
        if any(b is None for b in bands):
            bands = tuple(f"B{i + 2}" for i in range(n_bands))
        H, W = src.height, src.width
        lh, lw = lsrc.height, lsrc.width
        if (lh, lw) != (H, W):
            raise ValueError(
                f"label raster is {lh}x{lw} but image is {H}x{W}. They must share a grid -- "
                "re-run the lab 4.1 reprojection. Reading them at mismatched shapes silently "
                "pairs pixels with the wrong label."
            )
        if src.transform != lsrc.transform:
            raise ValueError("image and label rasters have different geotransforms; not aligned")

        img_nodata = src.nodata
        label_nodata = lsrc.nodata if lsrc.nodata is not None else nodata_label
        if verbose:
            print(f"[extract] {scene_id}: {W}x{H}, {n_bands} bands {bands}, "
                  f"img nodata={img_nodata}, label nodata={label_nodata}")

        rows_out, cols_out, labels_out, patches_out = [], [], [], []
        skipped = {"nodata_image": 0, "nodata_label": 0, "invalid_class": 0, "mixed_label": 0}

        for y in range(0, H - patch_size + 1, stride):
            # Windowed read of a row-block of image bands + the matching label block.
            h_block = min(window_rows, H - y)
            win_img = Window(0, y, W, min(h_block, patch_size if stride >= patch_size else h_block))
            img_block = src.read(window=win_img, masked=False).astype(np.float32)  # (C, hb, W)
            lab_block = lsrc.read(1, window=Window(0, y, W, img_block.shape[1])).astype(np.int32)

            hb = img_block.shape[1]
            for ly in range(0, hb - patch_size + 1, stride):
                gy = y + ly
                for x in range(0, W - patch_size + 1, stride):
                    lab_win = lab_block[ly : ly + patch_size, x : x + patch_size]
                    img_win = img_block[:, ly : ly + patch_size, x : x + patch_size]

                    if img_nodata is not None and np.any(img_win == img_nodata):
                        skipped["nodata_image"] += 1
                        continue
                    if np.any(lab_win == label_nodata):
                        skipped["nodata_label"] += 1
                        continue

                    if label_name == "center":
                        cy = cx = patch_size // 2
                        code = int(lab_win[cy, cx])
                    else:
                        vals, cnts = np.unique(lab_win, return_counts=True)
                        if vals.size > 1 and cnts.max() / cnts.sum() < 0.5:
                            skipped["mixed_label"] += 1
                            continue
                        code = int(vals[int(np.argmax(cnts))])

                    if code not in valid:
                        skipped["invalid_class"] += 1
                        continue

                    patches_out.append(np.transpose(img_win, (1, 2, 0)))  # (H,W,C)
                    labels_out.append(code)
                    rows_out.append(gy)
                    cols_out.append(x)

            if verbose and y and y % (stride * 400) == 0:
                print(f"  ... row {y}/{H}, kept {len(labels_out)}")

    patches = np.asarray(patches_out, dtype=np.float32)
    labels_arr = np.asarray(labels_out, dtype=np.int64)
    rows_arr = np.asarray(rows_out, dtype=np.int32)
    cols_arr = np.asarray(cols_out, dtype=np.int32)

    if patches.size == 0:
        raise RuntimeError(
            f"scene {scene_id}: zero patches extracted. Skips: {skipped}. Check that the label "
            "raster is aligned and that its nodata value is what you think it is."
        )

    if max_patches is not None and len(labels_arr) > max_patches:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(labels_arr), size=int(max_patches), replace=False)
        keep.sort()
        patches, labels_arr = patches[keep], labels_arr[keep]
        rows_arr, cols_arr = rows_arr[keep], cols_arr[keep]
        skipped["subsampled_out"] = int(len(rows_out) - max_patches)

    ps = PatchSet(
        patches=patches,
        labels=labels_arr,
        scene=np.asarray([scene_id] * len(labels_arr), dtype=object),
        row=rows_arr,
        col=cols_arr,
        bands=tuple(bands),
        patch_size=int(patch_size),
        stride=int(stride),
        stats={
            "source": str(image_path),
            "label_source": str(label_path),
            "scene_id": scene_id,
            "skipped": skipped,
            "label_name": label_name,
            "image_nodata": None if img_nodata is None else float(img_nodata),
            "label_nodata": float(label_nodata),
            "seed": int(seed),
        },
    )
    if verbose:
        print(ps.summary())
    return ps


def save_patches(ps: PatchSet, out_dir) -> Path:
    """Persist one scene's archive with full provenance.

    Keys are explicit; readers must use them. ``np.savez(patches, labels)`` with
    positional args, or ``np.load(f).values()`` on read, is how band/label swaps
    become silent.
    """
    from eo_course import paths as _paths

    out = _paths.ensure(Path(out_dir))[0]
    p = out / f"{SCENE_PREFIX}{ps.stats['scene_id']}{SCENE_SUFFIX}"
    np.savez_compressed(
        p,
        patches=ps.patches.astype(np.float32),
        labels=ps.labels.astype(np.int64),
        scene=np.asarray(ps.scene, dtype=object),
        row=ps.row.astype(np.int32),
        col=ps.col.astype(np.int32),
        bands=np.asarray(ps.bands, dtype=object),
    )
    p.with_suffix(".json").write_text(
        json.dumps(dict(ps.stats.items()), indent=2, default=str), encoding="utf-8"
    )
    return p


def load_scene(path, require_provenance: bool = True) -> PatchSet:
    """Load one scene archive written by :func:`save_patches`.

    Loads **by key**. ``np.load(f).values()`` returns arrays in insertion order,
    so adding any metadata array silently swaps patches and labels -- that was
    lab5_2's ``_parse_npz_file``.
    """
    from eo_course import paths as _paths

    p = _paths.require_existing(path, "patch archive")
    with np.load(p, allow_pickle=True) as d:
        from eo_course.gates import gate_npz_loaded_by_key

        gate_npz_loaded_by_key(d, required=("patches", "labels", "scene", "row", "col"))
        meta = p.with_suffix(".json")
        stats = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        ps = PatchSet(
            patches=d["patches"].astype(np.float32),
            labels=d["labels"].astype(np.int64),
            scene=d["scene"],
            row=d["row"].astype(np.int32),
            col=d["col"].astype(np.int32),
            bands=tuple(str(b) for b in d["bands"]),
            patch_size=int(stats.get("patch_size", 0)) or _infer_patch_size(d["patches"]),
            stride=int(stats.get("stride", 0)) or 0,
            stats=stats,
        )
    if require_provenance and ps.scene is None:
        raise ValueError(f"{p}: archive has no scene provenance; cannot build a leakage-free split")
    return ps


def _infer_patch_size(patches: np.ndarray) -> int:
    return int(patches.shape[1]) if patches.ndim == 4 else 0


def load_all_scenes(directory, expect_scenes: int | None = None) -> PatchSet:
    """Load and concatenate every scene archive in ``directory``.

    Uses :data:`SCENE_GLOB` only. The 2025/26 ``glob("*_data.npz")`` also matched
    a combined file, so every patch was loaded twice and the random split then put
    duplicates in both train and test. There is deliberately no "combined" file in
    this design -- concatenation happens in memory, once.
    """
    from eo_course import paths as _paths

    d = _paths.require_existing(directory, "patch archive directory")
    files = sorted(d.glob(SCENE_GLOB))
    if not files:
        raise FileNotFoundError(
            f"no {SCENE_GLOB} archives in {d}. Run lab 4.2 (eo_course.patches.extract_patches + "
            "save_patches) first."
        )
    sets = [load_scene(f) for f in files]

    band_sets = {s.bands for s in sets}
    if len(band_sets) > 1:
        raise ValueError(
            f"scenes disagree on bands: {band_sets}. Concatenating them would mix band orders "
            "into one tensor; re-run lab 4.1 with a single band list."
        )
    size_sets = {s.patch_size for s in sets}
    if len(size_sets) > 1:
        raise ValueError(f"scenes disagree on patch_size: {size_sets}")

    if expect_scenes is not None and len(sets) != expect_scenes:
        raise ValueError(f"expected {expect_scenes} scene archives in {d}, found {len(sets)}: "
                         f"{[s.stats.get('scene_id') for s in sets]}")

    out = PatchSet(
        patches=np.concatenate([s.patches for s in sets], axis=0),
        labels=np.concatenate([s.labels for s in sets], axis=0),
        scene=np.concatenate([s.scene for s in sets], axis=0),
        row=np.concatenate([s.row for s in sets], axis=0),
        col=np.concatenate([s.col for s in sets], axis=0),
        bands=sets[0].bands,
        patch_size=sets[0].patch_size,
        stride=sets[0].stride,
        stats={
            "scenes": [s.stats.get("scene_id") for s in sets],
            "per_scene_n": [len(s) for s in sets],
            "skipped": {k: sum(s.stats.get("skipped", {}).get(k, 0) for s in sets)
                        for k in {"nodata_image", "nodata_label", "invalid_class", "mixed_label"}},
        },
    )
    return out


def as_chw(ps: PatchSet):
    """``(N,H,W,C)`` -> ``(N,C,H,W)`` float32, contiguous.

    The archives store channels-last (that is the rasterio/numpy reading order)
    while PyTorch wants channels-first. The old notebooks each did their own
    ``permute`` in a different cell, and lab5_2's ``TransformerModel`` used
    ``x.view(B, seq_len, C)`` on a CHW tensor -- a view, not a permute -- which
    scrambled band and position semantics without raising.
    """
    a = np.ascontiguousarray(np.transpose(ps.patches, (0, 3, 1, 2)))
    return a.astype(np.float32, copy=False)
