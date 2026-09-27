"""Course toolkit for RAF622M / TÖV606M — ML for Earth Observation.

One audited reference implementation shared by every lab, so that notebooks
*call* things instead of each re-deriving them. The 2025/26 series had the same
split logic, metric code and normalisation written out four times, and the four
copies disagreed with each other.

Modules
-------
``paths``
    Every filesystem path, derived from the environment. Nothing in a notebook
    hard-codes ``/p/...`` or a username again.
``radiometry``
    Digital numbers to reflectance, and fitted normalisations that refuse to run
    unfitted. ``Norm`` fits on the training split only, by construction.
``labels``
    The CORINE class table, and a ``LabelMap`` built from training data only.
``patches``
    Patch extraction that carries ``scene``/``row``/``col``. Provenance is what
    makes leakage detectable at all.
``splits``
    Group (scene + spatial block) splits with a ``manifest_hash``, plus a random
    split kept deliberately as a negative control.
``metrics``
    Evaluation on a fixed label set with scene-clustered bootstrap intervals.
``baselines``
    The trivial predictors a model must beat: majority, prior-matched random,
    per-scene majority, an NDVI rule, a linear probe.
``results``
    The append-only ``results.json`` record that makes a submission checkable.
``gates``
    Pass/fail checks that raise. A green board you cannot explain is worth
    nothing; a red board you have diagnosed is worth full marks.
``training``
    Lightning module and data module. Importing this package does **not** import
    torch; import ``eo_course.training`` when you need it.

Submodules are loaded lazily so that ``import eo_course`` stays cheap and does
not require torch, lightning or rasterio on a machine that only wants ``paths``.
"""
from __future__ import annotations

import importlib

__version__ = "0.2.0"

__all__ = [
    "baselines",
    "gates",
    "labels",
    "metrics",
    "patches",
    "paths",
    "radiometry",
    "results",
    "splits",
    "training",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
