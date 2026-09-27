"""Trivial classifiers every submitted model must beat.

The 2025/26 labs printed a majority-class count and called it a baseline, then
never scored it with the same code as the model. Lab 6's fine-tuned Prithvi
reported test accuracy 0.10000000149011612 -- exactly 1/10, chance for 10
classes -- while the majority-class baseline on that same 455-sample test set was
245/455 = 0.538. The model was worse than predicting one label, and nothing in
the notebook could have caught it.

These baselines exist so that fact is impossible to miss. Score them through
:func:`eo_course.metrics.evaluate` with the same label set and the same test
split as your model, and put the table in your report.

Reference values on the 2025/26 Lab 5.2 test set (455 samples, 10 classes,
majority class = 245): majority accuracy 0.538, uniform-random balanced accuracy
0.100. Any claim of "good performance" below those lines is a bug report, not a
result.
"""

from __future__ import annotations

import numpy as np


def majority_class(train_labels, labels, n: int) -> np.ndarray:
    """Always predict the most frequent *training* class.

    Its overall accuracy equals the test-set rate of that class; its balanced
    accuracy is exactly ``1/K``.
    """
    from eo_course.labels import LabelMap

    lm = LabelMap(tuple(int(c) for c in labels))
    enc = lm.encode(train_labels)
    counts = np.bincount(enc, minlength=lm.n_classes)
    best = int(np.argmax(counts))
    return np.full(int(n), best, dtype=np.int64)


def uniform_random(n: int, n_classes: int, seed: int = 0) -> np.ndarray:
    """Uniform random guess. Balanced accuracy ~ 1/K, macro-F1 ~ 1/K."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, int(n_classes), size=int(n), dtype=np.int64)


def prior_matched_random(train_labels, labels, n: int, seed: int = 0) -> np.ndarray:
    """Sample from the training class prior.

    Overall accuracy lands near ``sum(p_c^2)`` -- higher than uniform, and a
    useful reminder that a model scoring just above this has learned the marginal
    distribution and nothing else.
    """
    from eo_course.labels import LabelMap

    rng = np.random.default_rng(seed)
    lm = LabelMap(tuple(int(c) for c in labels))
    enc = lm.encode(train_labels)
    counts = np.bincount(enc, minlength=lm.n_classes).astype(np.float64)
    p = counts / counts.sum()
    return rng.choice(lm.n_classes, size=int(n), p=p).astype(np.int64)


def per_scene_majority(
    scene_ids, train_scene_ids, train_labels, labels, seed: int = 0, return_info: bool = False
):
    """Predict the most frequent class *within each scene*.

    This is the honest classical baseline for gridded EO: it uses no imagery at
    all, only the scene's own label histogram. A model that beats global majority
    but loses to per-scene majority has learned scene statistics, not land cover.
    Falls back to global majority for scenes unseen in training.

    With ``return_info=True`` also returns ``{"n_unseen_scenes": k}``. Unseen
    scenes are the interesting case -- the baseline is guessing there -- so the
    count must be visible rather than buried.
    """
    from eo_course.labels import LabelMap

    lm = LabelMap(tuple(int(c) for c in labels))
    scene_ids = np.asarray(scene_ids, dtype=object)
    train_scene_ids = np.asarray(train_scene_ids, dtype=object)
    enc_train = lm.encode(train_labels)

    global_best = int(np.argmax(np.bincount(enc_train, minlength=lm.n_classes)))
    table: dict = {}
    for s in np.unique(train_scene_ids):
        mask = train_scene_ids == s
        counts = np.bincount(enc_train[mask], minlength=lm.n_classes)
        table[s] = int(np.argmax(counts))

    out = np.empty(scene_ids.size, dtype=np.int64)
    unseen = []
    for i, s in enumerate(scene_ids):
        if s in table:
            out[i] = table[s]
        else:
            out[i] = global_best
            unseen.append(str(s))
    info = {"n_unseen_scenes": len(unseen), "unseen_examples": unseen[:5], "n_scenes_seen": len(table)}
    return (out, info) if return_info else out


def ndvi_rule(x, labels, nir_index: int = 3, red_index: int = 2, thresholds=(0.25, 0.6)) -> np.ndarray:
    """A physics baseline: two NDVI thresholds, mapped onto the class set.

    ``x`` is ``(N, C, H, W)`` or ``(N, H, W, C)`` reflectance in [0, 1].
    Low NDVI -> bare/urban, mid -> sparse/crop, high -> dense vegetation. The
    mapping onto the course's class set is deliberately crude; its value is that
    it is *interpretable* and *free*, so a learned model must beat it on evidence
    rather than on authority.

    Returns predictions in model-index space, choosing the class whose name best
    matches each NDVI band from the available classes.
    """
    from eo_course.labels import LabelMap

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"expected 4-D (N,C,H,W) or (N,H,W,C), got shape {x.shape}")
    # Detect channel axis: the channel count is small, spatial dims are large.
    if x.shape[1] <= 20 and min(x.shape[2], x.shape[3]) >= x.shape[1]:
        nir = x[:, nir_index]
        red = x[:, red_index]
    else:
        nir = x[..., nir_index]
        red = x[..., red_index]

    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / np.maximum(nir + red, 1e-6)
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    mean_ndvi = ndvi.reshape(ndvi.shape[0], -1).mean(axis=1)

    lm = LabelMap(tuple(int(c) for c in labels))
    names = [n.lower() for n in lm.names]

    def pick(*keywords, default: int = 0) -> int:
        for kw in keywords:
            for i, nm in enumerate(names):
                if kw in nm:
                    return i
        return default

    low = pick("urban", "construction", "industrial", "bare", "rock", "dump", default=0)
    mid = pick("arable", "pasture", "grassland", "vineyard", "fruit", default=low)
    high = pick("coniferous", "broad-leaved", "mixed forest", "forest", "tree", default=mid)

    lo, hi = thresholds
    out = np.full(mean_ndvi.shape, mid, dtype=np.int64)
    out[mean_ndvi < lo] = low
    out[mean_ndvi >= hi] = high
    return out


def linear_probe(x, y_train, labels, y_test=None, max_iter: int = 200, seed: int = 0):
    """Multinomial logistic regression on flattened patches.

    The cheap, embarrassingly strong baseline that was skipped. On small-patch
    data a linear model on raw reflectance is often within a few points of a CNN;
    if your CNN is not clearly ahead of this, the CNN is not doing the work you
    think it is.

    Returns
    -------
    (pred_fn, fitted_report) -- call ``pred_fn(x_test)`` for predictions.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    flat = x.reshape(n, -1)
    if flat.shape[1] > 4096:
        # 224x224x4 = 200704 features on ~8k samples is a bad idea; pool first.
        side = int(np.sqrt(flat.shape[1] / x.shape[-1] if x.ndim == 4 else flat.shape[1]))
        flat = _pool(x, max(1, side // 8))

    # sklearn >=1.7 removed the `multi_class` argument; softmax multinomial is now
    # the only behaviour, so we must not pass it.
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(flat, np.asarray(y_train, dtype=np.int64))
    report = {
        "type": "logistic_regression",
        "n_features": int(flat.shape[1]),
        "train_acc": float(clf.score(flat, np.asarray(y_train, dtype=np.int64))),
        "classes_seen": sorted(int(c) for c in clf.classes_),
    }

    def predict(x_new):
        f = np.asarray(x_new, dtype=np.float32)
        ff = f.reshape(f.shape[0], -1)
        if ff.shape[1] != flat.shape[1]:
            ff = _pool(f, max(1, int(np.sqrt(ff.shape[1] / (f.shape[-1] if f.ndim == 4 else 1))) // 8))
        return clf.predict(ff).astype(np.int64)

    return predict, report


def _pool(x, factor: int) -> np.ndarray:
    """Average-pool spatial dims by ``factor`` then flatten."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4:
        return x.reshape(x.shape[0], -1)
    chw = x.shape[1] <= 20 and min(x.shape[2], x.shape[3]) >= x.shape[1]
    if not chw:
        x = np.transpose(x, (0, 3, 1, 2))
    b, c, h, w = x.shape
    nh, nw = max(1, h // factor), max(1, w // factor)
    out = np.zeros((b, c, nh, nw), dtype=np.float32)
    for i in range(nh):
        for j in range(nw):
            out[:, :, i, j] = x[:, :, i * factor : (i + 1) * factor, j * factor : (j + 1) * factor].mean(
                axis=(2, 3)
            )
    return out.reshape(b, -1)


ALL_BASELINES = ("majority", "uniform_random", "prior_random", "per_scene_majority", "ndvi", "linear_probe")


def run_all(
    *,
    x_train,
    y_train_codes,
    x_test,
    y_test_codes,
    labels,
    codes,
    train_scene_ids,
    test_scene_ids,
    seed: int = 0,
    which=ALL_BASELINES,
) -> dict:
    """Score every requested baseline with the same metric code as the model.

    ``labels`` is the model-index set (``np.arange(K)``), ``codes`` the CORINE
    codes. Returns ``{name: Metrics.to_dict()}``.
    """
    from eo_course.labels import LabelMap
    from eo_course.metrics import evaluate

    lm = LabelMap(tuple(int(c) for c in codes))
    labels = np.asarray(labels, dtype=np.int64)
    y_test_enc = lm.encode(y_test_codes)
    n_test = int(np.asarray(y_test_codes).size)
    out: dict[str, dict] = {}

    preds: dict[str, np.ndarray] = {}
    if "majority" in which:
        preds["majority"] = majority_class(y_train_codes, codes, n_test)
    if "uniform_random" in which:
        preds["uniform_random"] = uniform_random(n_test, lm.n_classes, seed=seed)
    if "prior_random" in which:
        preds["prior_random"] = prior_matched_random(y_train_codes, codes, n_test, seed=seed)
    if "per_scene_majority" in which:
        preds["per_scene_majority"] = per_scene_majority(
            test_scene_ids, train_scene_ids, y_train_codes, codes, seed=seed
        )
    if "ndvi" in which and x_test is not None:
        preds["ndvi"] = ndvi_rule(x_test, codes)
    if "linear_probe" in which and x_test is not None:
        predict, _report = linear_probe(x_train, lm.encode(y_train_codes), codes, seed=seed)
        preds["linear_probe"] = predict(x_test)

    for name, p in preds.items():
        m = evaluate(
            y_test_enc, p, labels, codes=codes, names=lm.names,
            groups=test_scene_ids, n_boot=200, seed=seed,
        )
        out[name] = m.to_dict()
        out[name]["name"] = name
    return out
