"""Print the graded results table and run the deliverable gates.

    python scripts/summarize_results.py --lab lab5
    python scripts/summarize_results.py --lab lab5 --check

``--check`` is what the grader runs. It exits non-zero unless every recorded run
for the lab has baselines, >= 3 seeds per arm, a matching split hash, and a test
macro-F1 above the best baseline by the course margin.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo_course import gates, results  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lab", default=None)
    ap.add_argument("--check", action="store_true", help="run deliverable gates and exit non-zero on failure")
    ap.add_argument("--csv", action="store_true", help="also write results.csv")
    args = ap.parse_args(argv)

    print(gates.describe())
    print(results.summary_table(args.lab))
    if args.csv:
        p = results.write_csv(args.lab)
        print(f"\nwrote {p}")

    if not args.check:
        print("\n" + results.describe_contract())
        return 0

    print("\n=== deliverable gates ===")
    failures: list[str] = []
    try:
        runs = gates.gate_results_file(lab=args.lab)
        print(f"[PASS] results_file: {len(runs)} record(s)")
    except gates.GateFailure as e:
        print(f"[FAIL] results_file: {e}")
        return 1

    # Group by arm so the seed requirement is per arm, not per file.
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_arm[(r.get("config") or {}).get("arm", r["run_id"])].append(r)

    for arm, rs in sorted(by_arm.items()):
        try:
            st = gates.gate_seeds_and_variance(rs)
            print(f"[PASS] seeds[{arm}]: n={st['n']} mean_bal_acc={st['mean']:.4f} "
                  f"std={st['std']:.4f} (claim threshold {st['threshold']:.4f})")
        except gates.GateFailure as e:
            print(f"[FAIL] seeds[{arm}]: {e}")
            failures.append(arm)

        for r in rs:
            if r.get("test_used_for_tuning"):
                try:
                    gates.gate_no_test_set_selection(r)
                except gates.GateFailure as e:
                    print(f"[FAIL] test_selection[{r['run_id']}]: {e}")
                    failures.append(r["run_id"])

            t = r.get("test") or {}
            base = r.get("baselines") or {}
            if base:
                best = max(base.values(), key=lambda b: b.get("macro_f1", 0.0))
                need = best.get("macro_f1", 0.0) + gates.MIN_MACRO_F1_OVER_BASELINE
                if t.get("macro_f1", 0.0) <= need:
                    print(f"[FAIL] beats_baselines[{r['run_id']}]: macro_f1 "
                          f"{t.get('macro_f1', 0):.4f} <= {need:.4f}")
                    failures.append(r["run_id"])
                else:
                    print(f"[PASS] beats_baselines[{r['run_id']}]: macro_f1 "
                          f"{t['macro_f1']:.4f} > {need:.4f}")

    print("\n" + ("ALL DELIVERABLE GATES PASSED" if not failures else f"{len(set(failures))} item(s) failed"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
