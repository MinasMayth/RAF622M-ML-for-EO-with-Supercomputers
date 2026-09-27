"""Append-only results bookkeeping. This is what gets graded.

Nothing in the 2025/26 labs was persisted. ``grep`` for ``to_csv`` / ``json.dump``
/ ``torch.save`` in lab5_1 and lab5_2 returns nothing, so there was no artifact to
grade, no way to tell whether ``trainer.test()`` ever ran, and no way to detect a
student who reported a validation number as a test number.

The contract
------------
Every run appends one record to ``results.json`` under ``runs``, carrying:

* the config that produced it (so a run is re-executable),
* the split ``manifest_hash`` (so reported == used is checkable),
* the seed and git SHA (so it is attributable and reproducible),
* whether the split's test set was consulted during selection
  (``test_used_for_tuning``) -- set it truthfully; the gates fail it closed.

Writes are append-only and atomic. Overwriting a previous run destroys the
evidence that you did the work, so :func:`record_run` refuses to reuse a
``run_id``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from eo_course import paths

SCHEMA_VERSION = 2


class ResultsError(RuntimeError):
    """Raised on a malformed or dishonest results record."""


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays and Paths for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def git_sha() -> str | None:
    """Current commit, or ``None`` off-repo. Recorded so a run is attributable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=True
        )
        return bool(out.stdout.strip())
    except Exception:
        return None


def default_results_path() -> Path:
    return paths.results_dir() / "results.json"


def load_results(path=None) -> dict:
    """Read the results file, or an empty skeleton if absent."""
    p = Path(path) if path is not None else default_results_path()
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "runs": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        # Not fatal: old files stay readable, but the grader uses the new schema.
        data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("runs", [])
    return data


def save_results(data: dict, path=None) -> Path:
    p = Path(path) if path is not None else default_results_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")
    os.replace(tmp, p)  # atomic: a crash mid-write cannot corrupt the ledger
    return p


def record_run(
    run_id: str,
    *,
    lab: str,
    config: dict,
    split_manifest_hash: str | None,
    seed: int | None,
    test_metrics: dict,
    val_metrics: dict | None = None,
    train_metrics: dict | None = None,
    baselines: dict | None = None,
    test_used_for_tuning: bool = False,
    n_seeds: int = 1,
    notes: str = "",
    extra: dict | None = None,
    path=None,
) -> dict:
    """Append one graded run record. Returns the record.

    ``test_metrics`` is required and must come from ``eo_course.metrics.evaluate``
    on the held-out split. Passing validation numbers here is the single most
    common way to fail this course's gates -- they check the split hash, not your
    confidence.
    """
    data = load_results(path)
    if any(r.get("run_id") == run_id for r in data["runs"]):
        raise ResultsError(
            f"run_id {run_id!r} already recorded in {path or default_results_path()}. "
            "Results are append-only: pick a new run_id so the earlier attempt stays "
            "in the record."
        )
    if not test_metrics:
        raise ResultsError(
            "test_metrics is empty. trainer.test() output is not a substitute -- build it "
            "with eo_course.metrics.evaluate(y_true, y_pred, labels, codes=..., groups=...)."
        )
    for key in ("balanced_acc", "macro_f1", "overall_acc", "n"):
        if key not in test_metrics:
            raise ResultsError(f"test_metrics is missing required key {key!r}")

    rec = {
        "run_id": run_id,
        "lab": lab,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "user": paths.username(),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "seed": seed,
        "n_seeds": int(n_seeds),
        "split_manifest_hash": split_manifest_hash,
        "test_used_for_tuning": bool(test_used_for_tuning),
        "config": config,
        "test": test_metrics,
        "val": val_metrics,
        "train": train_metrics,
        "baselines": baselines,
        "notes": notes,
    }
    if extra:
        rec["extra"] = extra
    data["runs"].append(rec)
    save_results(data, path)
    return rec


def get_run(run_id: str, path=None) -> dict:
    data = load_results(path)
    for r in data["runs"]:
        if r.get("run_id") == run_id:
            return r
    raise ResultsError(f"no run_id {run_id!r} recorded")


def runs_for_lab(lab: str, path=None) -> list[dict]:
    return [r for r in load_results(path)["runs"] if r.get("lab") == lab]


def summary_table(lab: str | None = None, path=None) -> str:
    """The table your report must contain. Print it, do not retype it."""
    runs = load_results(path)["runs"]
    if lab:
        runs = [r for r in runs if r.get("lab") == lab]
    if not runs:
        return f"(no runs recorded{' for lab ' + lab if lab else ''})"

    head = (
        f"{'run_id':<26} {'seed':>4} {'acc':>7} {'bal_acc':>8} {'macroF1':>8} "
        f"{'minRec':>7} {'nMinCls':>8} {'dF1_vs_best_base':>17} {'params':>10} {'split':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in runs:
        t = r.get("test") or {}
        rec = t.get("recall") or []
        sup = t.get("support") or []
        # recall[] and support[] are written by metrics.evaluate() from the same
        # arrays, so unequal lengths mean a truncated or hand-edited record. Say
        # so loudly rather than silently scoring the shorter of the two.
        if len(rec) != len(sup):
            raise ResultsError(
                f"run {r.get('run_id')!r}: test.recall has {len(rec)} entries but "
                f"test.support has {len(sup)}. A results.json record is written by "
                "record_run() and must not be edited by hand."
            )
        present = [x for x, s in zip(rec, sup, strict=True) if s and s > 0]
        min_rec = min(present) if present else float("nan")
        n_min = min([s for s in sup if s > 0]) if sup else 0
        base = r.get("baselines") or {}
        best_base_f1 = max([b.get("macro_f1", 0.0) for b in base.values()], default=0.0)
        delta = t.get("macro_f1", 0.0) - best_base_f1
        params = (r.get("config") or {}).get("n_params")
        lines.append(
            f"{r['run_id']:<26} {str(r.get('seed')):>4} {t.get('overall_acc', float('nan')):>7.4f} "
            f"{t.get('balanced_acc', float('nan')):>8.4f} {t.get('macro_f1', float('nan')):>8.4f} "
            f"{min_rec:>7.3f} {n_min:>8} {delta:>+17.4f} "
            f"{(f'{params:,}' if params else '-'):>10} {str(r.get('split_manifest_hash'))[:7]:>7}"
        )
    return "\n".join(lines)


def write_csv(lab: str | None = None, path=None, csv_path=None) -> Path:
    """Flatten runs to CSV for spreadsheet comparison across a team."""
    import csv

    runs = load_results(path)["runs"]
    if lab:
        runs = [r for r in runs if r.get("lab") == lab]
    out = Path(csv_path) if csv_path else (paths.results_dir() / "results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "run_id", "lab", "seed", "n_seeds", "split_manifest_hash", "test_used_for_tuning",
        "acc", "balanced_acc", "macro_f1", "weighted_f1", "kappa", "n_test", "min_class_support",
        "best_baseline_macro_f1", "delta_macro_f1_vs_baseline", "n_params", "git_sha", "notes",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in runs:
            t = r.get("test") or {}
            sup = [s for s in (t.get("support") or []) if s]
            base = r.get("baselines") or {}
            bb = max([b.get("macro_f1", 0.0) for b in base.values()], default=0.0)
            w.writerow({
                "run_id": r["run_id"],
                "lab": r.get("lab"),
                "seed": r.get("seed"),
                "n_seeds": r.get("n_seeds"),
                "split_manifest_hash": r.get("split_manifest_hash"),
                "test_used_for_tuning": r.get("test_used_for_tuning"),
                "acc": t.get("overall_acc"),
                "balanced_acc": t.get("balanced_acc"),
                "macro_f1": t.get("macro_f1"),
                "weighted_f1": t.get("weighted_f1"),
                "kappa": t.get("kappa"),
                "n_test": t.get("n"),
                "min_class_support": min(sup) if sup else None,
                "best_baseline_macro_f1": bb,
                "delta_macro_f1_vs_baseline": (t.get("macro_f1", 0.0) - bb) if t else None,
                "n_params": (r.get("config") or {}).get("n_params"),
                "git_sha": r.get("git_sha"),
                "notes": r.get("notes"),
            })
    return out


def describe_contract() -> str:
    return (
        f"results.json contract (schema v{SCHEMA_VERSION})\n"
        "  one record per run, append-only, containing:\n"
        "    run_id, lab, seed, n_seeds, split_manifest_hash, test_used_for_tuning,\n"
        "    config (incl. n_params), test{...}, val{...}, baselines{...}, git_sha, notes\n"
        "  test{...} must contain: n, overall_acc, balanced_acc, macro_f1, weighted_f1,\n"
        "    micro_f1, kappa, precision[], recall[], f1[], support[], predicted[], confusion[]\n"
        "  produced by eo_course.metrics.evaluate() on the held-out split.\n"
        "  Write it with record_run(); the gates in eo_course.gates read it.\n"
    )
