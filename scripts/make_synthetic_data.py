"""Write a small synthetic patch dataset in the course's on-disk format.

Two uses
--------
1. **Integration test.** The whole pipeline (load -> split -> baselines -> gates ->
   results) can be exercised on a laptop in seconds, with no cluster, no CDSE
   account and no CORINE download. ``scripts/train_cnn.py --dry-run`` against this
   data is the fastest way to check the toolkit still hangs together.
2. **Lab fallback.** The instructor notes already say "provide a fallback dataset to
   bypass downloads when APIs are flaky". Lab 3 hard-fails at its authentication
   cell for anyone without credentials, and lab 4.2 has never been executed at all;
   a synthetic set keeps a blocked session from stalling the whole class.

The data is *not* realistic: labels are spatially smooth blobs and the bands are
noise around a per-class mean. It is structurally realistic (multiple scenes, a
spatial grid, per-scene patch archives with provenance, severe class imbalance),
which is what the pipeline cares about. Never report metrics from it as a result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo_course import patches as pat  # noqa: E402
from eo_course import paths  # noqa: E402
from eo_course.radiometry import S2_BANDS_4  # noqa: E402

#: (CORINE code, weight, band means in DN units for B02,B03,B04,B08)
DEFAULT_CLASSES = [
    (1, 0.42, (900, 1100, 1200, 1400)),      # continuous urban
    (12, 0.28, (800, 1000, 900, 3200)),      # arable
    (23, 0.18, (600, 900, 500, 3800)),       # broad-leaved forest
    (26, 0.06, (700, 1000, 800, 3400)),      # natural grassland
    (41, 0.04, (1200, 1100, 900, 600)),      # water bodies
    (36, 0.015, (700, 800, 700, 1100)),      # peat bogs  -- rare, Iceland-relevant
    (34, 0.005, (3000, 3200, 3100, 3400)),   # glacier/snow -- very rare
]


#: Per-patch white-noise sigma in DN units. Deliberately large: with small noise
#: the classes are linearly separable, the linear-probe baseline scores macro-F1
#: 1.0, and the "beat the baselines" gate becomes unsatisfiable. Real CORINE
#: patches overlap heavily in band space.
NOISE_SIGMA = 620.0

#: Amplitude and correlation length of each class's *spatial texture*, in DN and
#: pixels. This is what makes the dataset learnable by a convnet rather than by a
#: linear model alone.
#:
#: Without texture, every patch is i.i.d. noise around a class mean, so the
#: Bayes-optimal classifier is linear in the pixel means and a CNN can only ever
#: tie it -- the ablation "small CNN vs linear probe" would be decided by the data
#: generator, not by the models. Real land cover has structure: forest is fine
#: grained, urban is blocky, arable is medium and directional.
#:
#: (code, texture_amplitude, texture_correlation_length_pixels)
TEXTURE = {
    1: (700.0, 2.4),    # urban: blocky, large smooth regions
    12: (380.0, 1.1),   # arable: medium grain
    23: (520.0, 0.55),  # forest: fine, high-frequency canopy texture
    26: (260.0, 0.8),   # grassland: weak texture
    41: (180.0, 1.6),   # water: smooth with gentle variation
    36: (300.0, 0.7),   # peat bogs: mottled fine grain
    34: (240.0, 1.9),   # glacier/snow: smooth, bright
}


def _gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur, reflect-padded. Small and dependency-free."""
    if sigma <= 0.05:
        return a
    r = max(1, int(3 * sigma))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(t ** 2) / (2.0 * sigma ** 2))
    k /= k.sum()
    pad = np.pad(a, ((0, 0), (r, r), (r, r)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 1, pad)
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="valid"), 2, out)
    return out


def make_scene(scene_id: str, grid: int, patch_size: int, stride: int, seed: int, classes):
    """One scene as a smooth label field plus per-class band noise."""
    rng = np.random.default_rng(seed)
    codes = np.array([c for c, _, _ in classes])
    probs = np.array([w for _, w, _ in classes], dtype=np.float64)
    probs /= probs.sum()
    means = np.array([m for _, _, m in classes], dtype=np.float32)

    # Smooth label field: draw a coarse random field then upsample, so labels form
    # contiguous blobs. This is what makes a random split leak and a block split not.
    # The coarse draw respects `probs`, so the requested class imbalance survives.
    coarse_probs = rng.random((max(2, grid // 6),) * 2 + (len(codes),)) ** 2 * probs
    coarse = coarse_probs.argmax(axis=-1)
    fy = np.clip((np.arange(grid) / grid * coarse.shape[0]).astype(int), 0, coarse.shape[0] - 1)
    fx = np.clip((np.arange(grid) / grid * coarse.shape[1]).astype(int), 0, coarse.shape[1] - 1)
    field = coarse[np.ix_(fy, fx)]

    # Per-scene radiometric offset: each acquisition has its own illumination and
    # atmospheric state, so the same class sits at a different position in band
    # space in every scene. Without this, a model trained on two scenes transfers
    # to the third for free and the scene-held-out split measures nothing. This is
    # the single property that makes the 2025/26 random split misleading.
    offset = rng.normal(0, 260.0, size=4).astype(np.float32)

    # Build the whole scene as one (grid, grid, 4) image, then slice patches from
    # it. Doing it scene-wide rather than per patch is both much faster and the only
    # way texture is spatially *continuous* across patch borders -- which is exactly
    # the property that makes neighbouring patches leak across a random split.
    white = rng.normal(0.0, NOISE_SIGMA, size=(grid, grid, 4)).astype(np.float32)
    base = means[field].astype(np.float32) + offset[None, None, :]  # (grid,grid,4)

    tex = np.zeros((grid, grid, 4), dtype=np.float32)
    for k, code in enumerate(codes):
        amp, corr = TEXTURE.get(int(code), (300.0, 1.0))
        # One shared texture realisation per class per scene, blurred to the class's
        # correlation length, then scaled. Different classes get different spatial
        # scales, so a convnet can separate them where a linear model cannot.
        raw = rng.normal(0.0, 1.0, size=(grid, grid, 1))
        smooth = _gaussian_blur(raw, corr).astype(np.float32)
        s = float(smooth.std()) or 1.0
        tex += (smooth / s) * np.float32(amp) * np.float32(means[k] / max(means[k].mean(), 1.0))[None, None, :]

    image = np.clip(base + tex + white, 0, 10000).astype(np.float32)

    patches, labels, rows, cols = [], [], [], []
    for y in range(0, grid - patch_size + 1, stride):
        for x in range(0, grid - patch_size + 1, stride):
            # `field` holds class INDICES, not CORINE codes; convert before storing.
            win = field[y : y + patch_size, x : x + patch_size]
            vals, cnts = np.unique(win, return_counts=True)
            k = int(vals[int(np.argmax(cnts))])
            code = int(codes[k])
            patches.append(np.transpose(image[y : y + patch_size, x : x + patch_size], (0, 1, 2)))
            labels.append(code)
            rows.append(y)
            cols.append(x)

    return pat.PatchSet(
        patches=np.asarray(patches, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scene=np.asarray([scene_id] * len(labels), dtype=object),
        row=np.asarray(rows, dtype=np.int32),
        col=np.asarray(cols, dtype=np.int32),
        bands=S2_BANDS_4,
        patch_size=patch_size,
        stride=stride,
        stats={"scene_id": scene_id, "synthetic": True, "seed": seed, "skipped": {},
               "patch_size": patch_size, "stride": stride},
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: eo_course.paths.training_data_dir()")
    ap.add_argument("--scenes", default="S2A_20180315,S2A_20180602,S2A_20180811,S2A_20181004")
    ap.add_argument("--grid", type=int, default=120, help="scene side in pixels")
    ap.add_argument("--patch-size", type=int, default=5)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else paths.training_data_dir()
    paths.ensure(out)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    if len(scenes) < 3:
        ap.error("need >= 3 scenes; scene-level splitting is the point of this dataset")

    print(f"writing synthetic patches to {out}")
    for i, s in enumerate(scenes):
        ps = make_scene(s, args.grid, args.patch_size, args.stride, args.seed + i, DEFAULT_CLASSES)
        p = pat.save_patches(ps, out)
        print(f"  {s}: {len(ps)} patches -> {p.name}")
        print("   ", ps.summary().replace("\n", "\n    "))

    print(
        "\nNOTE: synthetic data. Structurally realistic (scenes, grid, imbalance), "
        "radiometrically fake.\nUse it to test the pipeline; never report its metrics "
        "as a result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
