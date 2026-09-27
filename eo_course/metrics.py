"""Imbalance-aware metrics with a fixed label set and honest uncertainty.

What the 2025/26 notebooks got wrong, and what this module fixes
----------------------------------------------------------------
1. **Label set derived from the evaluation slice.**
   ``labels = np.unique(np.concatenate([y_true, y_pred]))`` then
   ``balanced_acc = recall.mean()``. A class absent from both arrays silently
   leaves the average, so balanced accuracy is computed over *fewer classes than
   exist* and comes out inflated. Here the label set is always the explicit
   training class set.
2. **Comparing incompatible quantities.**
   ``print(f"Model improvement: {balanced_acc - majority_baseline}")`` subtracts a
   majority-class *accuracy* from a *balanced* accuracy. A constant predictor's
   balanced accuracy is ``1/K``, not its majority accuracy. Both comparisons are
   reported separately and correctly here.
3. **Averaging over batches instead of samples.**
   ``self.log("val_acc", batch_acc)`` weights a 124-sample final batch the same as
   a 256-sample one. Metrics here are computed once from the full arrays.
4. **No uncertainty.**
   Lab 5.2's headline ``Balanced accuracy 0.8338`` came from 455 validation samples
   where five of ten classes had support <= 9 (two had support 1). Two
   misclassifications move it by ~0.05. Every number here carries a bootstrap
   confidence interval, resampled **by scene** because patches within a scene are
   correlated -- resampling patches would understate the interval.

Definitions you are accountable for
-----------------------------------
* overall accuracy = correct / total. On imbalanced data it is dominated by the
  majority class; its floor is the majority-class rate, not 0.
* per-class recall = TP / (TP + FN) for that class.
* **balanced accuracy** = unweighted mean of per-class recalls. Its floor is
  ``1/K`` for a constant predictor, so on 10 classes 0.10 is *chance*, not success.
* **macro-F1** = unweighted mean of per-class F1. Unlike balanced accuracy it
  includes precision, so it punishes a model that sprays predictions onto rare
  classes. Balanced accuracy and macro-F1 diverge for exactly that reason, and
  reading both is the point of the lab.
* weighted-F1 = F1 averaged with class-support weights; closer to overall accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def confusion(y_true, y_pred, labels) -> np.ndarray:
    """Confusion matrix with rows = true, cols = predicted, over a fixed ``labels``.

    Passing ``labels`` explicitly is what keeps absent classes as real zero rows
    instead of silently shrinking the problem.
    """
    labels = np.asarray(labels, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return np.asarray(cm, dtype=np.int64)


def _prf_from_cm(cm: np.ndarray):
    """Per-class precision, recall, F1 from a square confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)   # true count per class
    pred = cm.sum(axis=0).astype(np.float64)      # predicted count per class

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
        recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / np.maximum(precision + recall, 1e-12),
            0.0,
        )
    return precision, recall, f1, support.astype(np.int64)


@dataclass
class Metrics:
    """Everything you are allowed to claim, with the support it rests on.

    ``labels`` are the model-index class ids (0..K-1 after remapping). ``codes``
    are the corresponding CORINE codes, used for names in reports and plots.
    """

    labels: np.ndarray
    codes: np.ndarray | None
    overall_acc: float
    balanced_acc: float
    macro_f1: float
    weighted_f1: float
    micro_f1: float
    kappa: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    predicted: np.ndarray
    confusion: np.ndarray
    n: int
    ci: dict | None = None
    names: list[str] | None = None

    # -- presentation ----------------------------------------------------
    def floor_balanced_acc(self) -> float:
        """Balanced accuracy of a constant predictor: 1/K."""
        return 1.0 / len(self.labels)

    def majority_rate(self) -> float:
        """Overall accuracy of a constant majority predictor."""
        return float(self.support.max() / max(self.n, 1))

    def worst_class(self) -> str:
        i = int(np.argmin(np.where(self.support > 0, self.f1, np.inf)))
        return self.class_label(i)

    def class_label(self, i: int) -> str:
        if self.names:
            return self.names[i]
        if self.codes is not None:
            # Resolve CORINE codes to names here rather than returning the bare
            # integer, so gate messages and tables are readable without every
            # caller having to pass names= themselves. class_name() labels the
            # synthetic merge bucket instead of calling it "UNKNOWN CORINE".
            from eo_course.labels import class_name

            code = int(self.codes[i])
            name = class_name(code)
            return f"{code} {name}" if not name.startswith("UNKNOWN") else str(code)
        return f"class {int(self.labels[i])}"

    def table(self, min_support: int = 0) -> str:
        """Fixed-width per-class table. Classes below ``min_support`` are flagged."""
        head = (
            f"{'class':<38} {'sup':>5} {'pred':>5} {'prec':>6} {'rec':>6} {'F1':>6} {'note':>10}"
        )
        lines = [head, "-" * len(head)]
        for i in range(len(self.labels)):
            note = ""
            if self.support[i] == 0:
                note = "NO SUPPORT"
            elif self.support[i] < min_support:
                note = f"n<{min_support}"
            lines.append(
                f"{self.class_label(i):<38} {int(self.support[i]):>5} {int(self.predicted[i]):>5} "
                f"{self.precision[i]:>6.3f} {self.recall[i]:>6.3f} {self.f1[i]:>6.3f} {note:>10}"
            )
        lines.append("-" * len(head))
        lines.append(
            f"{'OVERALL':<38} {self.n:>5} {'':>5} {'':>6} {'':>6} {'':>6}"
            f"\n  acc={self.overall_acc:.4f}  balanced_acc={self.balanced_acc:.4f} (floor {self.floor_balanced_acc():.4f})"
            f"  macro-F1={self.macro_f1:.4f}  weighted-F1={self.weighted_f1:.4f}  kappa={self.kappa:.4f}"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "n": int(self.n),
            "overall_acc": float(self.overall_acc),
            "balanced_acc": float(self.balanced_acc),
            "macro_f1": float(self.macro_f1),
            "weighted_f1": float(self.weighted_f1),
            "micro_f1": float(self.micro_f1),
            "kappa": float(self.kappa),
            "floor_balanced_acc": self.floor_balanced_acc(),
            "majority_rate": self.majority_rate(),
            "n_classes": int(len(self.labels)),
            "labels": self.labels.tolist(),
            "codes": None if self.codes is None else self.codes.tolist(),
            "precision": self.precision.tolist(),
            "recall": self.recall.tolist(),
            "f1": self.f1.tolist(),
            "support": self.support.tolist(),
            "predicted": self.predicted.tolist(),
            "confusion": self.confusion.tolist(),
        }
        if self.ci:
            d["ci95"] = self.ci
        return d


def evaluate(
    y_true,
    y_pred,
    labels,
    codes=None,
    names=None,
    groups=None,
    n_boot: int = 1000,
    seed: int = 0,
) -> Metrics:
    """Full evaluation on a fixed label set, with scene-clustered bootstrap CIs.

    Parameters
    ----------
    y_true, y_pred : 1-D integer arrays of model-index class ids.
    labels : the fixed class set (usually ``np.arange(K)``). Never derived from
        the data being evaluated.
    codes : matching CORINE codes, for names.
    groups : scene/block id per sample. Bootstrap resamples these, not individual
        samples. If omitted, samples are resampled i.i.d., which understates
        uncertainty for spatially correlated patches -- we record that in ``ci``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("evaluate() called on empty arrays")

    stray = set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
    missing = stray - set(labels.tolist())
    if missing:
        raise ValueError(
            f"predictions/targets contain classes {sorted(missing)} absent from the fixed "
            f"label set {labels.tolist()}. Fix the label set at the source; do not let "
            "metrics silently widen or shrink the problem."
        )

    cm = confusion(y_true, y_pred, labels)
    precision, recall, f1, support = _prf_from_cm(cm)
    predicted = cm.sum(axis=0).astype(np.int64)
    n = int(y_true.size)

    overall = float(np.trace(cm) / n)
    present = support > 0
    # Balanced accuracy averages over classes that actually exist in the problem
    # (support > 0), not over classes that happen to appear in this slice.
    balanced = float(recall[present].mean()) if present.any() else 0.0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    w = support[present].astype(np.float64)
    weighted_f1 = float((f1[present] * w).sum() / w.sum()) if w.sum() > 0 else 0.0

    # micro-F1 == overall accuracy for single-label multiclass; computed explicitly
    # so the equality is visible rather than assumed.
    micro_f1 = overall

    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels)) if len(labels) > 1 else 0.0

    ci = None
    if n_boot and n_boot > 0:
        ci = bootstrap_ci(
            y_true, y_pred, labels, groups=groups, n_boot=n_boot, seed=seed,
            present_mask=present,
        )

    return Metrics(
        labels=labels,
        codes=None if codes is None else np.asarray(codes, dtype=np.int64),
        overall_acc=overall,
        balanced_acc=balanced,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        micro_f1=micro_f1,
        kappa=kappa,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        predicted=predicted,
        confusion=cm,
        n=n,
        ci=ci,
        names=list(names) if names else None,
    )


def bootstrap_ci(
    y_true,
    y_pred,
    labels,
    groups=None,
    n_boot: int = 1000,
    seed: int = 0,
    present_mask=None,
) -> dict:
    """Percentile bootstrap CIs for balanced accuracy and macro-F1.

    Resamples **groups** (scenes / spatial blocks) with replacement when given.
    Patches from one scene are strongly correlated, so an i.i.d. patch bootstrap
    gives a reassuringly narrow interval that is simply wrong.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    n = y_true.size

    if groups is None:
        clusters = np.arange(n)
        clustered = False
    else:
        clusters = np.asarray(groups).ravel()
        if clusters.size != n:
            raise ValueError(f"groups has {clusters.size} entries but {n} samples")
        clustered = True

    uniq = np.unique(clusters)
    bal, mf1 = [], []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in picked])
        if idx.size == 0:
            continue
        cm = confusion(y_true[idx], y_pred[idx], labels)
        p, r, f, s = _prf_from_cm(cm)
        pres = s > 0 if present_mask is None else (s > 0) & present_mask
        if not pres.any():
            continue
        bal.append(r[pres].mean())
        mf1.append(f[pres].mean())

    if not bal:
        return {"n_boot": 0, "clustered": clustered}

    bal_a, mf1_a = np.asarray(bal), np.asarray(mf1)
    return {
        "n_boot": int(len(bal)),
        "clustered": bool(clustered),
        "n_clusters": int(uniq.size),
        "balanced_acc": [float(np.percentile(bal_a, 2.5)), float(np.percentile(bal_a, 97.5))],
        "macro_f1": [float(np.percentile(mf1_a, 2.5)), float(np.percentile(mf1_a, 97.5))],
        "note": None if clustered else "resampled samples, not scenes: intervals are optimistic",
    }


def paired_bootstrap_delta(
    y_true,
    pred_a,
    pred_b,
    labels,
    groups=None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """CI for metric(A) - metric(B) on the *same* samples.

    Use this before writing the word "improvement". Two runs on one split differ
    by more than you think, and unpaired intervals overlap for a reason.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    n = y_true.size
    clusters = np.arange(n) if groups is None else np.asarray(groups).ravel()
    uniq = np.unique(clusters)

    def _bm(t, p, pres_hint):
        cm = confusion(t, p, labels)
        _, r, f, s = _prf_from_cm(cm)
        pres = s > 0
        return (r[pres].mean() if pres.any() else 0.0, f[pres].mean() if pres.any() else 0.0)

    db, df = [], []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == c) for c in picked])
        if idx.size == 0:
            continue
        ba, fa = _bm(y_true[idx], pred_a[idx], None)
        bb, fb = _bm(y_true[idx], pred_b[idx], None)
        db.append(ba - bb)
        df.append(fa - fb)

    if not db:
        return {"n_boot": 0}
    db_a, df_a = np.asarray(db), np.asarray(df)
    return {
        "n_boot": int(len(db)),
        "delta_balanced_acc": [float(np.percentile(db_a, 2.5)), float(np.percentile(db_a, 97.5))],
        "delta_macro_f1": [float(np.percentile(df_a, 2.5)), float(np.percentile(df_a, 97.5))],
        "mean_delta_balanced_acc": float(db_a.mean()),
        "mean_delta_macro_f1": float(df_a.mean()),
        "significant_balanced_acc": bool(np.percentile(db_a, 2.5) > 0),
        "significant_macro_f1": bool(np.percentile(df_a, 2.5) > 0),
    }


def merge_rare_classes(y, codes, min_support: int, keep=None):
    """Collapse under-supported CORINE codes into one ``rare-other`` bucket.

    Reporting ``recall = 0.000`` for a class with one test sample is not a
    finding, it is noise with a label attached. Merge, and report the merged
    class -- do not delete the row and quietly average over what remains.

    Returns
    -------
    y_new, new_codes, mapping
    """
    from eo_course.labels import class_counts

    y = np.asarray(y, dtype=np.int64)
    codes = [int(c) for c in codes]
    counts = class_counts(y, codes)
    keep = set(codes) if keep is None else {int(c) for c in keep}
    # strict=True: class_counts() must return one count per code. A shorter
    # result would silently leave classes unexamined for support.
    rare = [c for c, k in zip(codes, counts, strict=True) if k < min_support and c in keep]
    if not rare:
        return y, np.asarray(codes, dtype=np.int64), {"merged": [], "rare_other_code": None}

    rare_other = max(codes) + 1
    out = y.copy()
    for c in rare:
        out[y == c] = rare_other
    new_codes = [c for c in codes if c not in rare and c in keep] + [rare_other]
    mapping = {"merged": rare, "rare_other_code": rare_other, "min_support": int(min_support)}
    return out, np.asarray(new_codes, dtype=np.int64), mapping


def top_confusions(m: Metrics, k: int = 8) -> list[tuple[str, str, int]]:
    """The k largest off-diagonal cells: "class X is being called Y", with counts."""
    cm = m.confusion.astype(np.float64)
    np.fill_diagonal(cm, 0)
    order = np.argsort(cm.ravel())[::-1]
    K = len(m.labels)
    out = []
    for flat in order[:k]:
        i, j = divmod(int(flat), K)
        if cm[i, j] <= 0:
            break
        out.append((m.class_label(i), m.class_label(j), int(m.confusion[i, j])))
    return out
