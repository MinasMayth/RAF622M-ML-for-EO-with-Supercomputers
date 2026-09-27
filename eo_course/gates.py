"""Pass/fail gates. These raise; they do not warn.

Why gates exist
---------------
In 2025/26 a student could finish every lab, see green output, and submit without
ever having established that their model was better than guessing. Lab 6 shipped
a fine-tuned Prithvi with test accuracy exactly 1/10 -- chance -- and the unit
sheet's deliverable ("Functioning end-to-end TerraTorch workflow", "Fine-tuned
Prithvi-EO-2.0 checkpoint") was satisfied by it, because both are satisfied by a
random model.

Each gate below is a claim you are making when your code runs it. A gate failing
is not a formatting problem; it means a conclusion you would write down is not
supported. Read the message: it names the number and the threshold.

Thresholds are module-level constants so the whole class shares one definition
and the grader can quote them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eo_course import results as results_mod
from eo_course.metrics import Metrics

# --------------------------------------------------------------------------
# Course-wide thresholds. Justified in docs/units/*/README.md.
# --------------------------------------------------------------------------

#: A model must beat the best trivial baseline by this much macro-F1 on the held-out
#: split. Below this, seed noise alone can explain the gap (lab5_2's headline moved
#: ~0.05 from two misclassified samples).
MIN_MACRO_F1_OVER_BASELINE = 0.10

#: Balanced accuracy must exceed 1/K by this much. 1/K is what a constant
#: predictor scores; clearing it by <0.15 on <=10 classes is not evidence.
MIN_BALANCED_ACC_OVER_CHANCE = 0.15

#: Per-class metrics on fewer than this many samples are noise with a label on it.
#: (recall on n=25 has SE ~0.10; on n=1 it is literally 0 or 1.)
MIN_CLASS_SUPPORT = 25

#: Ablation claims need at least this many seeds, and a delta must exceed
#: MIN_SEED_MULTIPLIER * pooled std to be written as an improvement.
MIN_SEEDS = 3
MIN_SEED_MULTIPLIER = 2.0


class GateFailure(AssertionError):
    """A scientific claim is unsupported. Not recoverable by re-running."""


def _fmt(name: str, ok: bool, detail: str) -> str:
    return f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def report(name: str, ok: bool, detail: str) -> None:
    """Print a gate line. Used by notebooks to show the full gate board."""
    print(_fmt(name, ok, detail))


# --------------------------------------------------------------------------
# Data-integrity gates
# --------------------------------------------------------------------------


def gate_no_duplicate_patches(patches, groups=None, sample_hashes: int = 20000, seed: int = 0) -> None:
    """Fail if the same patch appears twice.

    Lab 5.2 did ``TRAINING_DATA_DIR.glob("*_data.npz")``, which matched both the
    per-scene files *and* ``combined_training_data.npz`` -- a permutation of the
    same patches. Every sample entered the array twice, and the random split then
    put duplicates in both train and test. This gate catches that class of bug
    regardless of how the duplication arose.
    """
    arr = np.asarray(patches)
    n = arr.shape[0]
    if n == 0:
        raise GateFailure("no patches supplied")
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= sample_hashes else rng.choice(n, sample_hashes, replace=False)
    # A cheap content fingerprint: per-sample sum + a position-weighted sum, which
    # collides far less than a plain sum.
    flat = arr.reshape(len(idx), -1).astype(np.float64)
    pos = np.arange(flat.shape[1], dtype=np.float64)
    keys = np.stack([flat.sum(1), (flat * pos[None, :]).sum(1), (flat * pos[None, :] ** 2).sum(1)], axis=1)
    uniq = np.unique(keys, axis=0)
    dups = len(idx) - len(uniq)
    if dups > 0:
        raise GateFailure(
            f"{dups} duplicate patches among {len(idx)} sampled (of {n} total). Almost always "
            "the loader is reading both per-scene files and a combined file. Glob the "
            "per-scene pattern only, and load npz entries by key, not .values()."
        )


def gate_npz_loaded_by_key(obj, required=("patches", "labels")) -> None:
    """Fail if an npz was read positionally.

    ``np.load(f).values()`` returns arrays in insertion order. Add one metadata
    array to the file and patches silently become labels.
    """
    keys = set(getattr(obj, "files", []) or [])
    missing = [k for k in required if k not in keys]
    if missing:
        raise GateFailure(
            f"npz is missing required keys {missing}; found {sorted(keys)}. Load with "
            "d['patches'] / d['labels'] -- never np.load(...).values()."
        )


def gate_input_units(x, name: str = "x", expect: str = "reflectance") -> None:
    """Fail if the input tensor is not in a unit system the model can consume.

    Lab 5.1 multiplied an already-percentile-normalised array by 1e-4, so the
    network saw values of order 1e-5. BatchNorm absorbed it and nothing failed --
    which is precisely why it survived a full academic year.

    ``expect`` names the unit system explicitly. The default is the strictest
    option, because the course convention is reflectance in [0, 1]:

    ``"reflectance"`` (default)
        Per-band values must lie in [0, 1] with a real spread. This is what
        ``radiometry.dn_to_reflectance`` produces.
    ``"standardized"``
        Per-band mean near 0 and std near 1, i.e. ``Norm(mode="zscore")`` or
        ``radiometry.prithvi_norm``. Ranges in [-3, 5] are expected and fine.
        Pass this explicitly -- it is what Prithvi wants.
    ``"dn"``
        Raw digital numbers, order 1e3--1e4.
    ``"auto"``
        Accept any sane scale, reject only the pathological ones.

    The point of the gate is not "is it in [0, 1]". It is "is this a scale a
    sensor produced, or an artifact of arithmetic". A tensor whose per-band
    maximum is 1e-5, or whose spread is 1e-9, is the latter whichever convention
    you intended, and that is what gets rejected.
    """
    import numpy as np

    from eo_course.radiometry import assert_reflectance_range

    if expect not in ("auto", "reflectance", "standardized", "dn"):
        raise ValueError(f"unknown expect={expect!r}")

    if expect == "reflectance":
        try:
            assert_reflectance_range(x, name)
        except (AssertionError, ValueError) as e:
            raise GateFailure(str(e)) from None
        return

    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        raise GateFailure(f"{name} is empty; nothing to check the units of")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        raise GateFailure(f"{name} contains no finite values")

    lo, hi = float(finite.min()), float(finite.max())
    spread = hi - lo

    # A double-normalized tensor: correct sign, absurd magnitude.
    if max(abs(lo), abs(hi)) < 1e-3:
        raise GateFailure(
            f"{name} spans [{lo:.3g}, {hi:.3g}] -- two orders of magnitude below "
            "reflectance. This is what double-normalization looks like: lab 5.1 "
            "divided an already-[0,1] array by 10000. BatchNorm hid it."
        )
    # A collapsed tensor: right range, no information.
    if spread < 1e-6:
        raise GateFailure(
            f"{name} has per-array spread {spread:.3g}; every value is effectively "
            "the same. Normalization cannot recover information that is not there."
        )

    if expect == "auto":
        return

    if expect == "standardized":
        mean, std = float(finite.mean()), float(finite.std())
        if not (-1.0 <= mean <= 1.0) or std < 0.2:
            raise GateFailure(
                f"{expect} requested but {name} has mean {mean:.3g}, std {std:.3g}. "
                "Apply radiometry.prithvi_norm or Norm(mode='zscore')."
            )
    elif expect == "dn":
        if hi < 50.0:
            raise GateFailure(
                f"{expect} requested but {name} maxes at {hi:.3g}; that is not "
                "digital numbers. Did you already divide by QUANTIFICATION_VALUE?"
            )


def gate_split_is_grouped(manifest) -> None:
    """Fail unless whole scenes were held out and no scene straddles splits."""
    if "stratified_random" in manifest.method:
        raise GateFailure(
            "method is stratified_random: adjacent patches from one scene share pixels and "
            "the same 100 m CORINE label, so this split measures interpolation. It is "
            "available as a negative control, not as a result."
        )
    if manifest.groups is None:
        raise GateFailure(
            "split has no scene/group variable, so it cannot be leakage-free. Use "
            "eo_course.splits.group_block_split with scene ids carried from lab 4.2."
        )
    leaks = {k: v for k, v in manifest.group_leakage().items() if v}
    if leaks:
        raise GateFailure(f"scene straddles splits: { {k: sorted(map(str, v)) for k, v in leaks.items()} }")


def gate_split_disjoint(manifest) -> None:
    bad = {k: v for k, v in manifest.overlaps().items() if v}
    if bad:
        raise GateFailure(
            f"splits share indices: { {k: len(v) for k, v in bad.items()} } samples overlap. "
            "This is direct leakage; the lab5_2 get_split() allowed num_train < 0, which put "
            "the same sample in val and test."
        )


def gate_class_support(metrics: Metrics, min_support: int = MIN_CLASS_SUPPORT) -> None:
    """Fail if any class the model is scored on has too little test support.

    The fix is decided on TRAIN support, never here. Choosing which classes to
    merge by looking at the test split is test-set selection: it lets the test
    set define the label space you are then scored on, and the reported
    macro-average stops being an estimate of anything. Decide the merged label
    space from ``train`` counts (``eo_course.metrics.merge_rare_classes`` on
    ``y[train_idx]``), rebuild the split-independent label map, and re-run.

    If a class is adequately supported in train but thin in test, that is a
    property of the split, not a licence to merge on test evidence. Either
    accept it and say which per-class numbers are unreliable, or change the
    split -- and record that you did.
    """
    weak = [
        (metrics.class_label(i), int(metrics.support[i]))
        for i in range(len(metrics.labels))
        if 0 < metrics.support[i] < min_support
    ]
    absent = [metrics.class_label(i) for i in range(len(metrics.labels)) if metrics.support[i] == 0]
    if weak or absent:
        raise GateFailure(
            f"test support below {min_support} -- weak={weak} absent={absent}. "
            "Per-class precision/recall/F1 for these classes is noise, and they "
            "drag the macro-average with them. Fix it on TRAIN evidence: merge "
            "under-supported classes with metrics.merge_rare_classes(y_train, ...) "
            "and report the merged class. Do NOT decide merges by looking at these "
            "test counts -- that is test-set selection. If these classes are "
            "adequately supported in train, the split is the problem, not the "
            "label space: say which numbers are unreliable, or change the split "
            "and record it. Do not average over what is left and call it macro."
        )


# --------------------------------------------------------------------------
# Performance gates
# --------------------------------------------------------------------------


def gate_beats_baselines(metrics: Metrics, baselines: dict) -> None:
    """The central gate of the course: your model must beat trivial predictors.

    ``baselines`` is the dict from :func:`eo_course.baselines.run_all` (or a
    ``{name: {macro_f1, balanced_acc, ...}}`` mapping), scored on the *same* split
    with the *same* metric code.
    """
    if not baselines:
        raise GateFailure(
            "no baselines supplied. Run eo_course.baselines.run_all(...) on the same test "
            "split. Lab 6 reported test accuracy 0.10000000149011612 -- exactly 1/10, chance "
            "-- against a majority-class baseline of 0.538, and nothing caught it."
        )
    best_name = max(baselines, key=lambda k: baselines[k].get("macro_f1", 0.0))
    best = baselines[best_name]
    need_f1 = best.get("macro_f1", 0.0) + MIN_MACRO_F1_OVER_BASELINE
    need_bal = best.get("balanced_acc", 0.0) + MIN_BALANCED_ACC_OVER_CHANCE

    fails = []
    if metrics.macro_f1 <= need_f1:
        fails.append(f"macro_F1 {metrics.macro_f1:.4f} <= {need_f1:.4f} (best baseline {best_name} "
                     f"{best.get('macro_f1', 0.0):.4f} + {MIN_MACRO_F1_OVER_BASELINE})")
    if metrics.balanced_acc <= need_bal:
        fails.append(f"balanced_acc {metrics.balanced_acc:.4f} <= {need_bal:.4f} (best baseline "
                     f"{best_name} {best.get('balanced_acc', 0.0):.4f} + {MIN_BALANCED_ACC_OVER_CHANCE})")
    if fails:
        raise GateFailure(
            "model does not beat trivial baselines by a meaningful margin:\n  - "
            + "\n  - ".join(fails)
            + "\nBefore assuming the model is bad, check the two usual causes: (1) input "
              "normalisation mismatched to what the model expects (gate_input_units), "
              "(2) a collapsed head -- look at `predicted` per class; if one class absorbs "
              "everything, the model learned the prior."
        )


def gate_not_collapsed(metrics: Metrics, max_share: float = 0.85) -> None:
    """Fail if one class receives most of the predictions.

    Lab 6's model predicted a single class: nine classes scored recall 0.0 and one
    scored 1.0. A confusion matrix shows this instantly, but only if someone looks.
    """
    total = int(metrics.predicted.sum())
    if total == 0:
        raise GateFailure("model produced no predictions")
    i = int(np.argmax(metrics.predicted))
    share = float(metrics.predicted[i] / total)
    if share > max_share:
        raise GateFailure(
            f"{share:.1%} of all predictions are class {metrics.class_label(i)!r} "
            f"({int(metrics.predicted[i])}/{total}). The model has collapsed to one class. "
            "Check input scaling, learning rate, and whether class weights/sampler were "
            "built from the wrong split."
        )


def gate_above_chance(metrics: Metrics) -> None:
    """Minimum bar: balanced accuracy must clear 1/K."""
    floor = metrics.floor_balanced_acc()
    if metrics.balanced_acc <= floor:
        raise GateFailure(
            f"balanced_acc {metrics.balanced_acc:.4f} <= 1/K = {floor:.4f}. A constant "
            "predictor scores this. Note that overall accuracy can look far higher and mean "
            "the same thing -- compare accuracy to the majority rate "
            f"({metrics.majority_rate():.4f}), not to 0."
        )


def gate_no_test_set_selection(record: dict) -> None:
    """Fail if the test split was used to choose anything."""
    if record.get("test_used_for_tuning"):
        raise GateFailure(
            f"run {record.get('run_id')!r} declares test_used_for_tuning=True. Any number "
            "selected using the test set is a training number with a test label on it. "
            "Re-run selection on validation, then report the untouched test result once."
        )


def gate_split_hash_matches(record: dict, manifest) -> None:
    """Fail if the reported split is not the split in use."""
    got = record.get("split_manifest_hash")
    want = manifest.manifest_hash
    if got != want:
        raise GateFailure(
            f"run {record.get('run_id')!r} recorded split_manifest_hash={got!r} but the "
            f"current split hashes to {want!r}. You are reporting numbers from a different "
            "split than the one you now have. Retrain or re-report."
        )


def gate_seeds_and_variance(runs: list[dict], metric: str = "balanced_acc") -> dict:
    """Require >= MIN_SEEDS seeds and return mean/std for the claim threshold.

    Returns ``{"mean": m, "std": s, "threshold": 2*s}``. Any ablation delta smaller
    than ``threshold`` must be written as "indistinguishable at n=<seeds>".
    """
    vals = []
    for r in runs:
        v = (r.get("test") or {}).get(metric)
        if v is not None:
            vals.append(float(v))
    if len(vals) < MIN_SEEDS:
        raise GateFailure(
            f"only {len(vals)} run(s) carry {metric!r}; need >= {MIN_SEEDS} seeds. Lab 5.2's "
            "headline moved ~0.05 from two misclassified samples, so a single run cannot "
            "resolve any difference this course asks about."
        )
    a = np.asarray(vals)
    std = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "std": std, "threshold": MIN_SEED_MULTIPLIER * std, "n": int(a.size)}


def gate_claim_supported(delta: float, runs: list[dict], metric: str = "balanced_acc", label: str = "") -> None:
    """Fail if an 'improvement' is inside seed noise."""
    stats = gate_seeds_and_variance(runs, metric=metric)
    if abs(delta) < stats["threshold"]:
        raise GateFailure(
            f"{label or metric}: delta {delta:+.4f} is below {MIN_SEED_MULTIPLIER} x pooled std "
            f"= {stats['threshold']:.4f} (std {stats['std']:.4f} over {stats['n']} seeds). "
            "Report it as 'no detectable effect at n="
            f"{stats['n']}', not as an improvement."
        )


# --------------------------------------------------------------------------
# Deliverable gates
# --------------------------------------------------------------------------


def gate_results_file(path=None, lab: str | None = None, require_baselines: bool = True) -> list[dict]:
    """Fail unless a well-formed results record exists for ``lab``."""
    runs = results_mod.load_results(path)["runs"]
    if lab:
        runs = [r for r in runs if r.get("lab") == lab]
    if not runs:
        raise GateFailure(
            f"no results recorded{' for lab ' + lab if lab else ''} at "
            f"{path or results_mod.default_results_path()}. Nothing was persisted in 2025/26 "
            "and therefore nothing could be graded; this gate is the fix. Call "
            "eo_course.results.record_run(...) after trainer.test()."
        )
    out = []
    for r in runs:
        for key in ("run_id", "config", "seed", "split_manifest_hash", "test"):
            if key not in r:
                raise GateFailure(f"run {r.get('run_id')!r} is missing required key {key!r}")
        if require_baselines and not r.get("baselines"):
            raise GateFailure(
                f"run {r['run_id']!r} has no baselines block. A number with nothing to "
                "compare against is not a result."
            )
        out.append(r)
    return out


def gate_predictions_saved(y_true, y_pred, path, split_manifest_hash: str | None = None) -> Path:
    """Persist raw predictions so a grader can recompute every metric.

    Metrics are derived; predictions are the evidence. Saving them makes every
    number in your report independently checkable.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
    )
    if split_manifest_hash:
        Path(str(p) + ".split").write_text(str(split_manifest_hash), encoding="utf-8")
    return p


def run_all_gates(
    metrics: Metrics,
    baselines: dict,
    manifest,
    record: dict,
    min_support: int = MIN_CLASS_SUPPORT,
) -> list[str]:
    """Run the full performance gate board and return the printed lines."""
    lines: list[str] = []

    def check(name, fn, *a, **kw):
        try:
            fn(*a, **kw)
            lines.append(_fmt(name, True, "ok"))
        except (GateFailure, AssertionError) as e:
            lines.append(_fmt(name, False, str(e).replace("\n", " ")))

    check("split_grouped", gate_split_is_grouped, manifest)
    check("split_disjoint", gate_split_disjoint, manifest)
    check("split_hash", gate_split_hash_matches, record, manifest)
    check("no_test_selection", gate_no_test_set_selection, record)
    check("class_support", gate_class_support, metrics, min_support)
    check("not_collapsed", gate_not_collapsed, metrics)
    check("above_chance", gate_above_chance, metrics)
    check("beats_baselines", gate_beats_baselines, metrics, baselines)
    return lines


def print_gate_board(lines: list[str]) -> bool:
    for ln in lines:
        print(ln)
    ok = all("[FAIL]" not in ln for ln in lines)
    print("\n" + ("ALL GATES PASSED" if ok else "GATES FAILED -- read each message; they name the number and the threshold."))
    return ok


def describe() -> str:
    return (
        "Course gates\n"
        f"  MIN_MACRO_F1_OVER_BASELINE   = {MIN_MACRO_F1_OVER_BASELINE}\n"
        f"  MIN_BALANCED_ACC_OVER_CHANCE = {MIN_BALANCED_ACC_OVER_CHANCE}\n"
        f"  MIN_CLASS_SUPPORT            = {MIN_CLASS_SUPPORT}\n"
        f"  MIN_SEEDS                    = {MIN_SEEDS}\n"
        f"  MIN_SEED_MULTIPLIER          = {MIN_SEED_MULTIPLIER}\n"
    )
