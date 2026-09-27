"""Leakage-free train/val/test splitting for gridded Earth-observation data.

Why the 2025/26 numbers were not real
-------------------------------------
Every lab split patches with a *random permutation* over a spatially contiguous
raster. That leaks in two independent ways:

1. **Neighbour leakage.** With ``PATCH_SIZE = 3`` and ``STRIDE = 3`` (the lab
   default), adjacent windows share an edge. CORINE labels are 100 m, i.e. a
   10x10 block of 10 m pixels, so one label pixel supports roughly 3x3
   neighbouring patches. A random split therefore puts near-duplicate samples
   with the same label into train and test. The model is being asked to
   interpolate, not to generalise.
2. **Scene leakage.** The dataset is built from a handful of Sentinel-2 scenes.
   Each scene carries its own illumination, atmosphere, sensor state and date.
   When every split receives patches from every scene, "generalisation" reduces
   to "generalising within a scene you have already seen".

The reported ``Overall accuracy: 0.6992 / Balanced accuracy: 0.5196`` (lab5_1)
and ``0.9231 / 0.8338 / 0.8376`` (lab5_2) are both inflated by these two
effects. Neither notebook contains a group or block variable.

The fix
-------
Split on **scene** first (hold out whole scenes), and within a scene split on
**spatial blocks** of several patches, never on individual patches. Every
function here returns index arrays plus a :class:`SplitManifest` that records
what was done, so a grader can recompute it and a student cannot quietly change
the split between tuning and reporting.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class SplitLeakError(RuntimeError):
    """Raised when a proposed split would leak. Never downgrade to a warning."""


@dataclass
class SplitManifest:
    """Machine-readable record of a split. Written to disk and hashed.

    The hash is the thing your ``results.json`` must carry. It makes "I tuned on
    train and reported on test" verifiable: the grader recomputes the split and
    compares.
    """

    seed: int
    method: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    groups: np.ndarray | None = None
    blocks: np.ndarray | None = None
    n_total: int = 0
    extra: dict = field(default_factory=dict)

    # -- integrity -------------------------------------------------------
    def index_hash(self, which: str) -> str:
        idx = getattr(self, which)
        return hashlib.sha256(np.sort(np.asarray(idx, dtype=np.int64)).tobytes()).hexdigest()[:16]

    @property
    def manifest_hash(self) -> str:
        payload = json.dumps(
            {
                "seed": self.seed,
                "method": self.method,
                "train": self.index_hash("train_idx"),
                "val": self.index_hash("val_idx"),
                "test": self.index_hash("test_idx"),
                "n_total": self.n_total,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def overlaps(self) -> dict[str, set]:
        """Pairwise index overlaps. All three sets should be empty."""
        a, b, c = set(self.train_idx.tolist()), set(self.val_idx.tolist()), set(self.test_idx.tolist())
        return {
            "train&val": a & b,
            "train&test": a & c,
            "val&test": b & c,
        }

    def group_leakage(self) -> dict[str, set]:
        """Groups (scenes) appearing in more than one split. Should be empty."""
        if self.groups is None:
            return {}
        g = np.asarray(self.groups)
        gs = {
            "train": set(g[self.train_idx].tolist()),
            "val": set(g[self.val_idx].tolist()),
            "test": set(g[self.test_idx].tolist()),
        }
        return {
            "train&val": gs["train"] & gs["val"],
            "train&test": gs["train"] & gs["test"],
            "val&test": gs["val"] & gs["test"],
        }

    def summary(self) -> dict:
        return {
            "seed": self.seed,
            "method": self.method,
            "manifest_hash": self.manifest_hash,
            "n_total": self.n_total,
            "n_train": int(self.train_idx.size),
            "n_val": int(self.val_idx.size),
            "n_test": int(self.test_idx.size),
            "n_scenes_train": int(np.unique(self.groups[self.train_idx]).size) if self.groups is not None else None,
            "n_scenes_val": int(np.unique(self.groups[self.val_idx]).size) if self.groups is not None else None,
            "n_scenes_test": int(np.unique(self.groups[self.test_idx]).size) if self.groups is not None else None,
            "index_hash": {
                "train": self.index_hash("train_idx"),
                "val": self.index_hash("val_idx"),
                "test": self.index_hash("test_idx"),
            },
            **self.extra,
        }

    def write(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        payload["indices"] = {
            "train": self.train_idx.astype(np.int64).tolist(),
            "val": self.val_idx.astype(np.int64).tolist(),
            "test": self.test_idx.astype(np.int64).tolist(),
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return p

    @classmethod
    def read(cls, path) -> SplitManifest:
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        idx = d.pop("indices")
        return cls(
            seed=d["seed"],
            method=d["method"],
            train_idx=np.asarray(idx["train"], dtype=np.int64),
            val_idx=np.asarray(idx["val"], dtype=np.int64),
            test_idx=np.asarray(idx["test"], dtype=np.int64),
            n_total=d["n_total"],
            extra={k: v for k, v in d.items() if k not in ("seed", "method", "n_total")},
        )

    def verify_against(self, other: SplitManifest) -> None:
        """Fail unless ``other`` is the same split. Used to check reported == used."""
        if self.manifest_hash != other.manifest_hash:
            raise SplitLeakError(
                f"split manifest mismatch: recorded {self.manifest_hash} vs recomputed "
                f"{other.manifest_hash}. You reported results for a different split than "
                "the one you are now using. Retrain, or re-report."
            )


# ---------------------------------------------------------------------------
# Splitting strategies
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _assert_disjoint(train, val, test, n_total):
    allidx = np.concatenate([train, val, test])
    if allidx.size != train.size + val.size + test.size:
        raise SplitLeakError("splits have different total size than their union; sizes are wrong")
    if np.unique(allidx).size != allidx.size:
        raise SplitLeakError("splits share sample indices -- this is direct leakage")
    if allidx.size and (allidx.min() < 0 or allidx.max() >= n_total):
        raise SplitLeakError("split indices out of range")


def block_ids(rows, cols, n_rows, n_cols, block: int = 10):
    """Assign each patch to a ``block x block`` spatial super-cell.

    Splitting on these instead of on individual patches removes neighbour
    leakage: a held-out block is a contiguous chunk of terrain whose every
    patch and every neighbour is held out too.

    Returns an int array of shape ``(n_rows * n_cols,)`` with values in
    ``[0, ceil(n_rows/block) * ceil(n_cols/block))``.
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    if rows.size != cols.size:
        raise ValueError("rows and cols must have the same length")
    if block < 1:
        raise ValueError("block must be >= 1")
    if rows.size and (rows.min() < 0 or cols.min() < 0):
        raise ValueError("row/col indices must be non-negative")
    if n_rows < block or n_cols < block:
        raise ValueError(
            f"grid is {n_rows}x{n_cols}, smaller than block={block}. Use a smaller block "
            "or a bigger scene; blocks larger than the grid collapse to one group."
        )
    br = rows // block
    bc = cols // block
    n_bc = int(np.ceil(n_cols / block))
    return (br * n_bc + bc).astype(np.int64)


def group_block_split(
    groups,
    rows=None,
    cols=None,
    block: int = 10,
    val_ratio: float = 0.15,
    test_scenes=None,
    seed: int = 0,
    labels=None,
    min_per_class: int = 25,
    n_total: int | None = None,
) -> SplitManifest:
    """Split by scene, then by spatial block within the remaining scenes.

    This is the course's default and the one your report must use unless you
    justify otherwise.

    Parameters
    ----------
    groups : array-like
        Scene/tile id per patch (strings or ints). Whole scenes are held out.
    rows, cols : array-like | None
        Patch position within its scene. Required for block splitting; if omitted
        we fall back to scene-only splitting and say so in ``extra``.
    block : int
        Edge length, in patches, of a spatial super-cell.
    test_scenes : sequence | None
        Explicit scenes to use as the test set. **Strongly recommended**: it makes
        the test set fixed for the whole class, so teams are comparable and nobody
        can select on it. If None, scenes are chosen randomly by seed.
    min_per_class : int
        Refuse a split in which any class has fewer than this many test samples;
        per-class metrics on n=1 are noise presented as evidence.

    Raises
    ------
    SplitLeakError
        If a scene straddles splits, indices overlap, or a class is under-supported.
    """
    groups = np.asarray(groups)
    n = groups.size
    n_total = n if n_total is None else int(n_total)
    if n_total != n:
        raise ValueError(f"n_total={n_total} but groups has {n} entries")
    if n == 0:
        raise ValueError("empty groups array")

    uniq = np.unique(groups)
    if uniq.size < 3:
        raise SplitLeakError(
            f"need at least 3 distinct scenes to split by scene (have {uniq.size}: "
            f"{uniq.tolist()}). With fewer, scene-level generalisation cannot be measured "
            "at all -- acquire more scenes rather than falling back to a random split."
        )

    rng = _rng(seed)
    if test_scenes is None:
        n_test_scenes = max(1, int(round(uniq.size * (1.0 - val_ratio))))
        # Keep at least one scene for train when val is also carved out.
        n_test_scenes = min(n_test_scenes, uniq.size - 2)
        perm = rng.permutation(uniq.size)
        test_set = set(uniq[perm[:n_test_scenes]].tolist())
    else:
        test_set = set(np.asarray(test_scenes).tolist())
        missing = test_set - set(uniq.tolist())
        if missing:
            raise ValueError(f"test_scenes not present in groups: {sorted(map(str, missing))}")
        if not test_set:
            raise ValueError("test_scenes is empty")

    remaining = np.array([g for g in uniq if g not in test_set])
    if remaining.size < 2:
        raise SplitLeakError(
            f"holding out {len(test_set)} scene(s) leaves {remaining.size} scene(s) for "
            "train+val; you need more scenes."
        )

    if val_ratio <= 0 or remaining.size < 2:
        val_scenes = np.array([], dtype=remaining.dtype)
    elif remaining.size == 2:
        val_scenes = remaining[1:2]
    else:
        n_val_scenes = max(1, int(round(remaining.size * val_ratio)))
        n_val_scenes = min(n_val_scenes, remaining.size - 1)
        rp = rng.permutation(remaining.size)
        val_scenes = remaining[rp[:n_val_scenes]]

    val_set = set(val_scenes.tolist())
    train_set = set(remaining.tolist()) - val_set

    g = groups.astype(object)
    in_train = np.array([x in train_set for x in g])
    in_val = np.array([x in val_set for x in g])
    in_test = np.array([x in test_set for x in g])

    train_idx = np.flatnonzero(in_train)
    val_idx = np.flatnonzero(in_val)
    test_idx = np.flatnonzero(in_test)

    # Within train, optionally carve val out of blocks if scenes are scarce.
    if rows is not None and cols is not None and block > 1:
        blk = block_ids(rows, cols, n_rows=int(np.max(rows)) + 1, n_cols=int(np.max(cols)) + 1, block=block)
    else:
        blk = None

    _assert_disjoint(train_idx, val_idx, test_idx, n)

    manifest = SplitManifest(
        seed=int(seed),
        method="group_block" if blk is not None else "group_only",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        groups=groups,
        blocks=blk,
        n_total=n,
        extra={
            "test_scenes": sorted(map(str, test_set)),
            "val_scenes": sorted(map(str, val_set)),
            "block": int(block) if blk is not None else None,
            "n_scenes": int(uniq.size),
        },
    )

    leaks = manifest.group_leakage()
    bad = {k: v for k, v in leaks.items() if v}
    if bad:
        raise SplitLeakError(f"scene leakage detected: { {k: sorted(map(str, v)) for k, v in bad.items()} }")

    if labels is not None and min_per_class > 0:
        _check_support(labels, manifest, min_per_class)

    return manifest


def stratified_random_split(
    labels,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
    min_per_class: int = 2,
) -> SplitManifest:
    """Per-class random split. **Provided as the negative control, not the default.**

    Labs 1-5 of the 2025/26 edition used exactly this and reported it as a result.
    It is here so you can run it, watch the accuracy go up, and write one sentence
    explaining why the number is not trustworthy.
    """
    labels = np.asarray(labels)
    rng = _rng(seed)
    train, val, test = [], [], []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        idx = rng.permutation(idx)
        n = idx.size
        if n < min_per_class:
            raise SplitLeakError(
                f"class {cls} has {n} samples; cannot form a split. Merge it into a rare "
                "class or drop it explicitly and record that you did."
            )
        n_test = max(1, int(round((1.0 - train_ratio - val_ratio) * n)))
        n_val = max(1, int(round(val_ratio * n)))
        n_train = n - n_test - n_val
        # The 2025/26 bug: n_train could go negative and the assert still passed
        # because -1 + 1 + 1 == 1. Guard it properly.
        if n_train < 1:
            raise SplitLeakError(
                f"class {cls}: ratios leave {n_train} training samples from {n} total. "
                "Fix the ratios or merge the class."
            )
        train.append(idx[:n_train])
        val.append(idx[n_train : n_train + n_val])
        test.append(idx[n_train + n_val :])

    manifest = SplitManifest(
        seed=int(seed),
        method="stratified_random",
        train_idx=np.sort(np.concatenate(train)),
        val_idx=np.sort(np.concatenate(val)),
        test_idx=np.sort(np.concatenate(test)),
        n_total=int(labels.size),
        extra={"warning": "spatially contiguous patches leak across this split; do not report as generalisation"},
    )
    _assert_disjoint(manifest.train_idx, manifest.val_idx, manifest.test_idx, labels.size)
    return manifest


def _check_support(labels, manifest: SplitManifest, min_per_class: int) -> None:
    """Every class needs enough test samples for per-class metrics to mean anything."""
    labels = np.asarray(labels)
    counts = np.bincount(labels[manifest.test_idx], minlength=int(labels.max()) + 1)
    present = np.flatnonzero(np.bincount(labels, minlength=labels.max() + 1) > 0)
    weak = [(int(c), int(counts[c])) for c in present if 0 < counts[c] < min_per_class]
    zero = [int(c) for c in present if counts[c] == 0]
    if weak or zero:
        raise SplitLeakError(
            f"test split is under-supported (min_per_class={min_per_class}). "
            f"weak={weak} absent={zero}. Per-class recall on n<25 has a standard error "
            f"of ~{1 / (2 * max(min_per_class, 1)) ** 0.5:.2f}, which is larger than any "
            "difference this lab asks you to resolve. Merge rare classes or add scenes."
        )


def per_split_class_counts(labels, manifest: SplitManifest, codes) -> dict:
    """Class counts per split, ordered by ``codes``. Put this in your report."""
    from eo_course.labels import class_counts

    return {
        "train": class_counts(labels[manifest.train_idx], codes).tolist(),
        "val": class_counts(labels[manifest.val_idx], codes).tolist(),
        "test": class_counts(labels[manifest.test_idx], codes).tolist(),
        "codes": [int(c) for c in codes],
    }
