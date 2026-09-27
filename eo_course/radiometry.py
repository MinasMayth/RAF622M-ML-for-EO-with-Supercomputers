"""Sentinel-2 radiometry and honest normalisation.

The thing the 2025/26 labs never taught
--------------------------------------
Sentinel-2 L2A products do **not** store reflectance. They store an integer
quantised reflectance::

    DN = round(surface_reflectance / QUANTIFICATION_VALUE)
    surface_reflectance = DN * QUANTIFICATION_VALUE,  QUANTIFICATION_VALUE = 10000

So a bright pixel with DN 4000 is 40 % reflectance. The old lab4_2 stated the
imagery was "12-bit, values 0-4095" (that is L1C's raw radiometric resolution,
not L2A storage) and then applied a percentile stretch, so the ÷10000 step
appeared nowhere in the course. It matters for two concrete reasons:

1. Any threshold you write in reflectance units (cloud, snow, water, NDVI) is
   wrong by four orders of magnitude if you forget it.
2. Geospatial foundation models were pretrained with stated per-band mean/std
   in *reflectance* units. Feeding them a [0, 1] percentile stretch is a ~1000x
   scale error, and it silently produces a model that predicts one class. That
   is exactly what happened in the 2025/26 Lab 6, where the reported test
   accuracy was 0.10000000149011612 -- i.e. exactly 1/10, chance for 10 classes.

The other thing this module enforces
-----------------------------------
Normalisation statistics must be **fitted on the training split only**. The old
lab4_2 computed percentile bounds over the whole raster, which later became
train + val + test, so every test patch contributed to the transform applied to
every training patch. :class:`Norm` makes the fit/apply separation structural:
you cannot apply statistics you never fitted, and ``fit`` takes only the train
array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Sentinel-2 L2A quantisation value (CDSE product specification).
QUANTIFICATION_VALUE = 10000.0

#: Sentinel-2 band names used by this course, in the order patches are stacked.
S2_BANDS_4 = ("B02", "B03", "B04", "B08")
S2_BANDS_6 = ("B02", "B03", "B04", "B08", "B11", "B12")

#: Band order Prithvi-EO-2.0 was pretrained on. Note it is Blue, **Green**, Red --
#: not the B02/B03/B04 numeric order a reader might assume, and not 6 optical
#: bands plus DEM. Verified in terratorch/models/backbones/prithvi_vit.py:
#: https://github.com/torchgeo/terratorch/blob/main/terratorch/models/backbones/prithvi_vit.py
PRITHVI_V2_BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")

#: Per-band mean/std used to pretrain Prithvi-EO-2.0, in DN units
#: (reflectance x 10000). Verified verbatim from ``PRITHVI_V2_MEAN`` /
#: ``PRITHVI_V2_STD`` in the file above.
PRITHVI_V2_MEAN = (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0)
PRITHVI_V2_STD = (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0)

#: The 4-band course configuration keeps B02/B03/B04/B08 = Blue, Green, Red,
#: Narrow-NIR, i.e. the first four pretrained bands in pretrained order. SWIR-1
#: and SWIR-2 are simply absent -- which is a real loss (SWIR is what separates
#: CORINE arable land from forest and from bare soil), not a detail.
PRITHVI_V2_BANDS_4B = PRITHVI_V2_BANDS[:4]
PRITHVI_V2_MEAN_4B = PRITHVI_V2_MEAN[:4]
PRITHVI_V2_STD_4B = PRITHVI_V2_STD[:4]


def prithvi_norm(bands: int = 4, dn_units: bool = True) -> tuple[list[float], list[float]]:
    """Mean/std for Prithvi, optionally rescaled for reflectance-in-[0,1] input.

    ``terratorch`` does **not** normalise your data for you -- these numbers are
    metadata about the pretraining distribution, and applying them is your job.
    If your patches are already percentile-stretched to [0, 1] you must either
    store raw DN instead or invert the stretch first; feeding [0, 1] values
    against means near 1000 is the ~1000x scale error that made the 2025/26 Lab 6
    model predict a single class.
    """
    if bands == 4:
        mean, std = list(PRITHVI_V2_MEAN_4B), list(PRITHVI_V2_STD_4B)
    elif bands == 6:
        mean, std = list(PRITHVI_V2_MEAN), list(PRITHVI_V2_STD)
    else:
        raise ValueError(f"Prithvi-EO-2.0 was pretrained on 6 bands; asked for {bands}")
    if not dn_units:
        mean = [m / QUANTIFICATION_VALUE for m in mean]
        std = [s / QUANTIFICATION_VALUE for s in std]
    return mean, std


def dn_to_reflectance(dn, quantization_value: float = QUANTIFICATION_VALUE) -> np.ndarray:
    """Convert L2A integer DNs to physical surface reflectance in [0, 1]."""
    return np.asarray(dn, dtype=np.float32) / float(quantization_value)


def reflectance_to_dn(reflectance, quantization_value: float = QUANTIFICATION_VALUE) -> np.ndarray:
    """Inverse of :func:`dn_to_reflectance`, rounded back to uint16 storage."""
    return np.rint(np.asarray(reflectance, dtype=np.float32) * float(quantization_value)).astype(np.uint16)


def per_band_percentiles(data, low_pct: float = 2.0, high_pct: float = 98.0, axis=None):
    """Per-band low/high percentiles.

    ``data`` is ``(C, H, W)`` or ``(N, H, W, C)``; pass ``axis=-1`` for the
    latter. The old code called ``np.percentile(data, 2)`` on the whole ``(4, H,
    W)`` stack, giving Blue/Green/Red/NIR one shared pair of bounds. NIR is far
    brighter than Blue over vegetation, so a pooled stretch squeezes Blue toward
    a constant -- destroying the band that most separates urban from vegetation.

    Returns
    -------
    low, high : ndarray
        One value per band, shape ``(C,)``.
    """
    data = np.asarray(data)
    if data.ndim == 3 and axis is None:
        axis = (1, 2)  # (C, H, W) -> per channel
    elif data.ndim == 4 and axis is None:
        axis = (1, 2)  # (N, H, W, C) would need axis=(0,1,2); caller must be explicit
        raise ValueError("For (N,H,W,C) input pass axis=(0, 1, 2) explicitly.")
    low = np.percentile(data, low_pct, axis=axis).astype(np.float32)
    high = np.percentile(data, high_pct, axis=axis).astype(np.float32)
    return low, high


#: Sentinel distinguishing "caller did not specify ``clip``" from an explicit
#: ``clip=None`` (which means: do not clip at all).
_UNSET: object = object()


@dataclass
class Norm:
    """A fitted normalisation transform.

    Build it with :meth:`fit` on **training data only**, then ``apply`` to
    train/val/test. ``fitted`` is False until then, and :meth:`apply` refuses to
    run -- which is the point.

    Modes
    -----
    ``"none"``
        Identity (still forces float32).
    ``"minmax"``
        Divide by ``max_val`` (default 10000). Physically meaningful: the result
        *is* surface reflectance. Use this when a pretrained model expects
        reflectance, or when you want thresholds to mean something.
    ``"percentile"``
        Linear stretch from the per-band low/high percentile to [0, 1], clipped.
        Improves contrast, but discards physical units -- a deliberate departure
        you must justify, not a default.
    ``"zscore"``
        Per-band (x - mean) / std.

    The ``clip`` default depends on the mode, and it matters:

    ==============  ==============================  ==========================
    mode            default ``clip``                why
    ==============  ==============================  ==========================
    minmax          ``(0.0, 1.0)``                  overshoot is a bad pixel
    percentile      ``(0.0, 1.0)``                  the stretch's whole point
    zscore          ``(-3.0, 3.0)``                 a z-score is signed
    none            ignored                         nothing to clip
    ==============  ==============================  ==========================

    Clipping a z-score to ``(0, 1)`` -- which is what a mode-independent default
    would do -- deletes every value below the mean. On a symmetric distribution
    that is half the data, silently replaced by 0.0. Pass ``clip=None`` for an
    unbounded z-score.
    """

    mode: str = "minmax"
    low_pct: float = 2.0
    high_pct: float = 98.0
    max_val: float = QUANTIFICATION_VALUE
    clip: tuple[float, float] | None | object = _UNSET
    low_: np.ndarray | None = field(default=None, repr=False)
    high_: np.ndarray | None = field(default=None, repr=False)
    mean_: np.ndarray | None = field(default=None, repr=False)
    std_: np.ndarray | None = field(default=None, repr=False)
    n_samples_seen: int = 0
    channel_axis: int = -1

    #: Per-mode default for ``clip``. Applied only when the caller left it unset.
    _DEFAULT_CLIP = {
        "minmax": (0.0, 1.0),
        "percentile": (0.0, 1.0),
        "zscore": (-3.0, 3.0),
        "none": None,
    }

    def __post_init__(self) -> None:
        if self.clip is _UNSET:
            object.__setattr__(self, "clip", self._DEFAULT_CLIP.get(self.mode, (0.0, 1.0)))

    @property
    def fitted(self) -> bool:
        if self.mode == "none":
            return True
        if self.mode == "minmax":
            return True
        if self.mode == "percentile":
            return self.low_ is not None
        if self.mode == "zscore":
            return self.mean_ is not None
        return False

    # -- construction ----------------------------------------------------
    @classmethod
    def fit(cls, train, mode: str = "minmax", channel_axis: int = -1, **kw) -> Norm:
        """Fit statistics from ``train`` only. Never pass val or test here."""
        arr = np.asarray(train)
        if arr.ndim < 2:
            raise ValueError(f"cannot fit on array of shape {arr.shape}")
        norm = cls(mode=mode, channel_axis=channel_axis, **kw)
        if mode == "none" or mode == "minmax":
            norm.n_samples_seen = int(arr.shape[0])
            return norm
        if mode not in ("percentile", "zscore"):
            raise ValueError(f"unknown mode {mode!r}; expected none/minmax/percentile/zscore")

        # Work with float32 and drop non-finite pixels from the statistics.
        finite = np.isfinite(arr)
        work = np.where(finite, arr, np.nan).astype(np.float32)

        if mode == "percentile":
            with np.errstate(invalid="ignore"):
                low = np.nanpercentile(work, norm.low_pct, axis=_reduce_axes(arr, channel_axis))
                high = np.nanpercentile(work, norm.high_pct, axis=_reduce_axes(arr, channel_axis))
            low = np.asarray(low, dtype=np.float32)
            high = np.asarray(high, dtype=np.float32)
            # A degenerate band (all-identical values) would divide by zero.
            span = high - low
            bad = ~np.isfinite(span) | (span <= 0)
            if np.any(bad):
                low = np.where(bad, 0.0, low).astype(np.float32)
                high = np.where(bad, 1.0, high).astype(np.float32)
            norm.low_, norm.high_ = low, high
        else:  # zscore
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(work, axis=_reduce_axes(arr, channel_axis))
                std = np.nanstd(work, axis=_reduce_axes(arr, channel_axis))
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
            mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
            norm.mean_, norm.std_ = mean, std

        norm.n_samples_seen = int(arr.shape[0])
        return norm

    # -- use -------------------------------------------------------------
    def apply(self, x) -> np.ndarray:
        """Transform ``x`` with the fitted statistics. Returns float32."""
        if not self.fitted:
            raise RuntimeError(
                f"Norm(mode={self.mode!r}) has not been fitted. Call Norm.fit(train_array, ...) "
                "on the TRAINING split only, then apply to train/val/test."
            )
        arr = np.asarray(x, dtype=np.float32)
        if self.mode == "none":
            return arr
        if self.mode == "minmax":
            out = arr / np.float32(self.max_val)
            return np.clip(out, *self.clip) if self.clip else out

        shape = _bcast_shape(arr, self.channel_axis)
        if self.mode == "percentile":
            low = self.low_.reshape(shape)
            high = self.high_.reshape(shape)
            out = (arr - low) / (high - low)
        else:
            out = (arr - self.mean_.reshape(shape)) / self.std_.reshape(shape)
        out = np.where(np.isfinite(out), out, np.float32(0.0))
        return np.clip(out, *self.clip) if self.clip else out.astype(np.float32)

    # -- bookkeeping -----------------------------------------------------
    def to_dict(self) -> dict:
        def _a(v):
            return None if v is None else np.asarray(v, dtype=np.float64).tolist()

        return {
            "mode": self.mode,
            "low_pct": self.low_pct,
            "high_pct": self.high_pct,
            "max_val": self.max_val,
            "clip": list(self.clip) if self.clip else None,
            "channel_axis": self.channel_axis,
            "low": _a(self.low_),
            "high": _a(self.high_),
            "mean": _a(self.mean_),
            "std": _a(self.std_),
            "n_samples_seen": self.n_samples_seen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Norm:
        def _a(v):
            return None if v is None else np.asarray(v, dtype=np.float32)

        return cls(
            mode=d["mode"],
            low_pct=d.get("low_pct", 2.0),
            high_pct=d.get("high_pct", 98.0),
            max_val=d.get("max_val", QUANTIFICATION_VALUE),
            clip=tuple(d["clip"]) if d.get("clip") else None,
            low_=_a(d.get("low")),
            high_=_a(d.get("high")),
            mean_=_a(d.get("mean")),
            std_=_a(d.get("std")),
            n_samples_seen=d.get("n_samples_seen", 0),
            channel_axis=d.get("channel_axis", -1),
        )

    def __repr__(self) -> str:
        return (
            f"Norm(mode={self.mode!r}, fitted={self.fitted}, "
            f"n_samples_seen={self.n_samples_seen}, clip={self.clip})"
        )


def _reduce_axes(arr: np.ndarray, channel_axis: int) -> tuple[int, ...]:
    """All axes except the channel axis."""
    nd = arr.ndim
    ca = channel_axis % nd
    return tuple(i for i in range(nd) if i != ca)


def _bcast_shape(arr: np.ndarray, channel_axis: int) -> tuple[int, ...]:
    """Shape that lets a per-band vector broadcast against ``arr``."""
    nd = arr.ndim
    ca = channel_axis % nd
    shape = [1] * nd
    shape[ca] = arr.shape[ca]
    return tuple(shape)


def assert_reflectance_range(x, name: str = "x", tol: float = 1e-3) -> None:
    """Fail loudly if data claimed to be reflectance is actually DN.

    The old lab5_1 multiplied an already-normalised array by 1e-4, feeding the
    network values of order 1e-5. BatchNorm hid it. This assert would not.
    """
    x = np.asarray(x)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError(f"{name}: all values are non-finite")
    lo, hi = float(finite.min()), float(finite.max())
    if hi > 1.0 + tol:
        raise AssertionError(
            f"{name}: max={hi:.4f} > 1, so this is not reflectance in [0,1]. "
            "If it is raw L2A DN, divide by QUANTIFICATION_VALUE (10000) or use "
            "Norm(mode='minmax'). Do not rescale by hand."
        )
    if lo < -tol:
        # Surface reflectance is non-negative. A negative band usually means a
        # z-scored array is being checked as if it were reflectance, or that
        # nodata was filled with a negative sentinel.
        raise AssertionError(
            f"{name}: min={lo:.4f} < 0, so this is not reflectance in [0,1]. "
            "Either this is a standardized array (check it with "
            "gates.gate_input_units(..., expect='standardized')) or nodata was "
            "filled with a negative value and survived into the training data."
        )
    if hi < 1e-2:
        raise AssertionError(
            f"{name}: max={hi:.6f} is implausibly small for reflectance -- this looks like "
            "data that was already normalised and then scaled again. Check for a stray "
            "* 0.0001 in your loading code."
        )
