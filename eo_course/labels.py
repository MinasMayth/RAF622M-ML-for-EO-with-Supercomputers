"""CORINE Land Cover taxonomy, valid-label handling, and label remapping.

The 2025/26 labs pasted a ~50-line ``CORINE_CLASSES`` dict into lab4_1, lab5_2
and lab6 independently. Three copies drifted: all three claimed
``48: "No data"`` and printed "Defined 45 CORINE land cover classes", but the
U2018 44-class raster actually stores no-data as **-128** in an int8 band, and
CORINE has 44 classes, not 45. A phantom class in the taxonomy is how you end up
reporting metrics over a class set that does not exist.

Two further hazards this module removes:

* **Water and sand as training classes.** The old filter was ``1 <= label <= 44``,
  which keeps 44 "Sea and ocean", 43 "Estuaries", 39 "Intertidal flats" and 30
  "Beaches, dunes, sands" as supervised classes. In a coastal tile those become
  real samples; in an inland tile they are zero-support classes that still
  occupy a row of your confusion matrix and a term in your macro-average.
* **Label space derived from all data.** ``np.unique(y)`` over train+val+test
  makes ``num_classes`` and the class set test-informed. :class:`LabelMap` is
  built from the training split only and then *applied* to val/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: CORINE Land Cover level-3 codes 1-44 with the names used throughout this
#: course. Kept verbatim from the 2025/26 notebooks so that names in your report
#: match names in the raster metadata. (Official EEA nomenclature is longer for
#: a few codes, e.g. 4 is "Road and rail networks and associated lines".)
CORINE_CLASSES: dict[int, str] = {
    # Artificial surfaces (1-11)
    1: "Continuous urban fabric",
    2: "Discontinuous urban fabric",
    3: "Industrial or commercial units",
    4: "Road and rail networks",
    5: "Port areas",
    6: "Airports",
    7: "Mineral extraction sites",
    8: "Dump sites",
    9: "Construction sites",
    10: "Green urban areas",
    11: "Sport and leisure facilities",
    # Agricultural areas (12-22)
    12: "Non-irrigated arable land",
    13: "Permanently irrigated land",
    14: "Rice fields",
    15: "Vineyards",
    16: "Fruit trees and berry plantations",
    17: "Olive groves",
    18: "Pastures",
    19: "Annual crops with permanent crops",
    20: "Complex cultivation patterns",
    21: "Agriculture with natural vegetation",
    22: "Agro-forestry areas",
    # Forest and semi-natural areas (23-34)
    23: "Broad-leaved forest",
    24: "Coniferous forest",
    25: "Mixed forest",
    26: "Natural grasslands",
    27: "Moors and heathland",
    28: "Sclerophyllous vegetation",
    29: "Transitional woodland-shrub",
    30: "Beaches, dunes, sands",
    31: "Bare rocks",
    32: "Sparsely vegetated areas",
    33: "Burnt areas",
    34: "Glaciers and perpetual snow",
    # Wetlands (35-39)
    35: "Inland marshes",
    36: "Peat bogs",
    37: "Salt marshes",
    38: "Salines",
    39: "Intertidal flats",
    # Water bodies (40-44)
    40: "Water courses",
    41: "Water bodies",
    42: "Coastal lagoons",
    43: "Estuaries",
    44: "Sea and ocean",
}

#: Value used for no-data in the EEA U2018 44-class raster (int8 band).
#: NOT 48, and NOT 0. Check your own file with
#: ``rasterio.open(path).nodata`` and assert it matches.
CORINE_NODATA = -128

#: Codes that are not land and should not be supervised classes by default.
#: Removing them is a decision you must record, not a silent filter.
WATER_AND_SEDIMENT_CODES: tuple[int, ...] = (30, 39, 42, 43, 44)

#: Level-1 grouping (5 classes), useful for a coarse auxiliary task and for
#: reading a confusion matrix without drowning in 44 rows.
CORINE_LEVEL1: dict[int, str] = {
    1: "Artificial surfaces",
    2: "Agricultural areas",
    3: "Forest and semi-natural areas",
    4: "Wetlands",
    5: "Water bodies",
}

#: code -> level-1 group, derived from the official code ranges.
def level1_of(code: int) -> str:
    """Map a level-3 CORINE code to its level-1 group name."""
    if 1 <= code <= 11:
        return CORINE_LEVEL1[1]
    if 12 <= code <= 22:
        return CORINE_LEVEL1[2]
    if 23 <= code <= 34:
        return CORINE_LEVEL1[3]
    if 35 <= code <= 39:
        return CORINE_LEVEL1[4]
    if 40 <= code <= 44:
        return CORINE_LEVEL1[5]
    raise ValueError(f"{code} is not a CORINE level-3 code (expected 1-44)")


#: Default display colours, from the course notebooks (EEA palette).
CORINE_COLORS: dict[int, str] = {
    1: "#E6004D", 2: "#FF0000", 3: "#CC4DF2", 4: "#CC0000", 5: "#E6CCCC",
    6: "#E6CCE6", 7: "#A600CC", 8: "#A64DCC", 9: "#FF4DFF", 10: "#FFA6FF",
    11: "#FFE6FF", 12: "#FFFFA8", 13: "#FFFF00", 14: "#E6E600", 15: "#E68000",
    16: "#F2A64D", 17: "#E6A600", 18: "#E6E64D", 19: "#FFE6A6", 20: "#FFE64D",
    21: "#E6CC4D", 22: "#F2CCA6", 23: "#80FF00", 24: "#00A600", 25: "#4DFF00",
    26: "#CCF24D", 27: "#A6FF80", 28: "#A6E64D", 29: "#A6F200", 30: "#E6E6E6",
    31: "#CCCCCC", 32: "#CCFFCC", 33: "#000000", 34: "#A6E6CC", 35: "#A6A6FF",
    36: "#4D4DFF", 37: "#CCCCFF", 38: "#E6E6FF", 39: "#A6A6E6", 40: "#00CCF2",
    41: "#80F2E6", 42: "#00FFA6", 43: "#A6FFE6", 44: "#E6F2FF",
}


def class_name(code: int) -> str:
    """Human-readable name for a CORINE code.

    Codes produced by :func:`eo_course.metrics.merge_rare_classes` are synthetic
    (``max(codes) + 1``) and are not CORINE classes, so they get a descriptive
    placeholder rather than ``UNKNOWN CORINE <n>`` -- that string in a gate
    message reads like a data error when it is a deliberate merge bucket.
    """
    code = int(code)
    if code in CORINE_CLASSES:
        return CORINE_CLASSES[code]
    if code > max(CORINE_CLASSES):
        return f"RARE-OTHER (merged bucket {code})"
    return f"UNKNOWN CORINE {code}"


def valid_class_codes(
    exclude: tuple[int, ...] | list[int] | None = None,
) -> tuple[int, ...]:
    """The CORINE codes eligible to be supervised classes.

    ``exclude=None`` means "exclude water and sediment" -- the course default.
    Pass ``exclude=()`` to keep everything, and say so in your report.
    """
    if exclude is None:
        exclude = WATER_AND_SEDIMENT_CODES
    bad = [c for c in exclude if c not in CORINE_CLASSES]
    if bad:
        raise ValueError(f"cannot exclude unknown CORINE codes {bad}")
    return tuple(c for c in sorted(CORINE_CLASSES) if c not in set(exclude))


def clean_labels(labels, nodata: int = CORINE_NODATA, allowed=None):
    """Boolean mask of pixels that are usable labels.

    Replaces the old ``if label < 1 or label > 44: continue``, which excluded the
    real no-data value only by luck and silently kept water.

    Parameters
    ----------
    labels : array-like
        Integer CORINE codes, any shape.
    nodata : int
        The raster's declared no-data value.
    allowed : iterable of int | None
        Permitted codes; default :func:`valid_class_codes`.

    Returns
    -------
    mask : ndarray of bool, same shape as ``labels``
    """
    arr = np.asarray(labels)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"labels must be integer, got {arr.dtype}")
    allowed = valid_class_codes() if allowed is None else list(allowed)
    mask = arr != nodata
    allowed_arr = np.asarray(allowed, dtype=arr.dtype)
    mask &= np.isin(arr, allowed_arr)
    return mask


@dataclass
class LabelMap:
    """Bidirectional map between CORINE codes and contiguous 0..K-1 targets.

    ``CrossEntropyLoss`` needs contiguous int64 targets, and the mapping must be
    established on the **training** split and then reused for val/test. Build it
    with :meth:`from_train`; ``encode`` then refuses codes the model has never
    seen rather than silently dropping them.
    """

    codes: tuple[int, ...]
    _to_idx: dict[int, int] = field(init=False, repr=False)
    _to_name: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self):
        codes = tuple(int(c) for c in self.codes)
        if len(set(codes)) != len(codes):
            raise ValueError(f"duplicate CORINE codes in LabelMap: {codes}")
        if not codes:
            raise ValueError("LabelMap needs at least one class")
        self.codes = codes
        self._to_idx = {c: i for i, c in enumerate(codes)}
        self._to_name = {c: class_name(c) for c in codes}

    # -- construction ----------------------------------------------------
    @classmethod
    def from_train(cls, train_labels, nodata: int = CORINE_NODATA, allowed=None) -> LabelMap:
        """Build the class set from TRAINING labels only."""
        arr = np.asarray(train_labels)
        mask = clean_labels(arr, nodata=nodata, allowed=allowed)
        codes = tuple(int(c) for c in np.unique(arr[mask]))
        if not codes:
            raise ValueError("no valid labels found in the training split")
        return cls(codes)

    # -- conversion ------------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.codes)

    @property
    def names(self) -> list[str]:
        """Class names ordered by model index 0..K-1."""
        return [self._to_name[c] for c in self.codes]

    @property
    def labels(self) -> np.ndarray:
        """CORINE codes ordered by model index 0..K-1."""
        return np.asarray(self.codes, dtype=np.int64)

    def index_of(self, code: int) -> int:
        return self._to_idx[int(code)]

    def code_of(self, index: int) -> int:
        return self.codes[int(index)]

    def encode(self, codes, strict: bool = True):
        """CORINE codes -> contiguous int64 targets.

        Unknown codes are a hard error by default. Silently dropping them is how
        a class disappears from your evaluation while your metric denominator
        stays the same.
        """
        arr = np.asarray(codes)
        out = np.full(arr.shape, -1, dtype=np.int64)
        for code, idx in self._to_idx.items():
            out[arr == code] = idx
        if strict:
            unknown = np.unique(arr[(out == -1)])
            if unknown.size:
                raise ValueError(
                    f"labels contain {unknown.tolist()} which are not in the LabelMap "
                    f"(built from train: {list(self.codes)}). Either they were excluded from "
                    "train (then drop those samples), or your class set is inconsistent."
                )
        return out

    def decode(self, indices):
        """Contiguous targets -> CORINE codes."""
        arr = np.asarray(indices, dtype=np.int64)
        out = np.full(arr.shape, -1, dtype=np.int64)
        for idx, code in enumerate(self.codes):
            out[arr == idx] = code
        return out

    def label_name(self, index: int) -> str:
        """Name of model class ``index``, e.g. ``'12 Non-irrigated arable land'``."""
        c = self.code_of(index)
        return f"{c} {self._to_name[c]}"

    def to_dict(self) -> dict:
        return {"codes": list(self.codes), "names": self.names}

    @classmethod
    def from_dict(cls, d: dict) -> LabelMap:
        return cls(codes=tuple(d["codes"]))

    def __repr__(self) -> str:
        return f"LabelMap(n_classes={self.n_classes}, codes={list(self.codes)})"


def class_counts(labels, codes) -> np.ndarray:
    """Counts per class, ordered by ``codes``, with zeros for absent classes.

    Always pass an explicit class set. ``np.bincount`` on raw data, or
    ``np.unique`` on the slice, both silently shrink the problem.
    """
    lm = LabelMap(tuple(int(c) for c in codes))
    enc = lm.encode(labels, strict=False)
    counts = np.bincount(enc[enc >= 0], minlength=lm.n_classes)
    return counts[: lm.n_classes].astype(np.int64)


def imbalance_ratio(counts) -> float:
    """max/min over classes with support >= 1.

    Defined here once because the labs used it three times with two different
    meanings (lab4_2 used ``counts[0]/counts[-1]``, i.e. positional, which is only
    max/min if the array happens to be sorted).
    """
    c = np.asarray(counts)
    c = c[c > 0]
    if c.size == 0:
        raise ValueError("no classes with support > 0")
    return float(c.max() / c.min())
