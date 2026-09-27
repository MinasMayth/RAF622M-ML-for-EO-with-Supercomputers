"""Build course .ipynb files from readable Python source files.

Why generate notebooks instead of editing them
----------------------------------------------
Hand-editing 40 KB of notebook JSON is how the 2025/26 edition accumulated its
worst defects: stale outputs that contradict the code (lab5_2's cell 35 printed a
0.8/0.1/0.1 split while cell 34 declared 0.9/0.05/0.05), cells executed out of
order, a summary describing a model that was not the one evaluated, and two cells
committed in an error state.

With a generator:
* the source is diffable and reviewable in a normal code review;
* outputs are always empty, so nothing stale can contradict the code -- a student
  must actually run it, which is the entire point;
* house rules (no hard-coded usernames, no credential printing, no dead partition
  names) are enforced in every notebook at once by :func:`lint`.

Source file format
------------------
A plain ``.py`` file using VS Code / jupytext cell markers::

    # %% [markdown]
    # # Lab 5 -- Training
    # A markdown cell.

    # %%
    import eo_course          # a code cell

Markdown lines are comments after the marker; a code cell is ordinary code.

Usage
-----
    python scripts/nbbuild.py                 # build all
    python scripts/nbbuild.py lab5            # build matching sources
    python scripts/nbbuild.py --check         # fail if any .ipynb is stale or dirty
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "notebooks_src"
OUT_DIR = ROOT / "notebooks" / "iceland-ml"

#: Text that must never appear in a shipped notebook. Each entry encodes a real
#: defect from the 2025/26 edition, so this list is a regression test for the audit.
FORBIDDEN: list[tuple[str, str]] = [
    ("hashim1", "another user's home hard-coded as a data path"),
    ("maurogiovanni", "another user's home hard-coded as a data path"),
    ("/p/project/training2600", "wrong project root; JURECA uses /p/project1/training2600"),
    ("JURECA-DC_CPU", "not a real partition name; use dc-cpu"),
    ("JURECA-DC_GPU", "not a real partition name; use dc-gpu"),
    ("YOUR_ORG", "placeholder repo URL that does not exist"),
    ("iceland-ml-course.git", "repository name that does not exist"),
    ("CLIENT_SECRET[:", "printing a credential fragment into notebook output"),
    ("your_pswd", "plaintext password placeholder in an executable cell"),
    ("CUDA_LAUNCH_BLOCKING", "debug flag that serialises CUDA and slows training 2-5x"),
    ("multi_class=", "removed in scikit-learn >= 1.7"),
    ("rstrip('.SAFE')", "rstrip takes a char set, not a suffix; use removesuffix"),
    ("path/to/model.pt", "no notebook saves a .pt; use load_from_checkpoint"),
]

#: Per-code-cell heuristics that are warnings, not errors.
WARN_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnp\.random\.(rand|randn|permutation|choice|seed)\b"),
     "use np.random.default_rng(seed); the legacy global RNG is irreproducible"),
    (re.compile(r"np\.load\([^)]*\)\.values\(\)"),
     "load npz entries by key, not .values(); dict order is insertion order"),
    (re.compile(r"^!source\s", re.M),
     "`!` runs /bin/sh -c, where `source` is unavailable; use `!bash -c \"source ...\"`"),
    (re.compile(r"^!cd\s", re.M),
     "each `!` line is a new subprocess, so `!cd` cannot affect the next line"),
    (re.compile(r"T_max\s*=\s*self\.max_epochs"),
     "cosine scheduler needs interval='epoch' or T_max in steps"),
    (re.compile(r"\.test\(\s*(?:model\s*=\s*)?\w+\s*,\s*datamodule\s*=\s*\w+\s*\)"),
     "trainer.test(model, datamodule=...) with no ckpt_path evaluates the "
     "final-epoch in-memory weights, not the selected checkpoint; pass "
     "ckpt_path='best' (labs 5.1, 5.2 and 6 all shipped this bug)"),
]

_MD_MARKER = re.compile(r"^# %%\s*\[markdown\]\s*$")
_CODE_MARKER = re.compile(r"^# %%\s*$")


class Cell:
    def __init__(self, kind: str, source: str):
        self.kind = kind
        s = source.strip("\n")
        self.source = (s + "\n") if s else ""

    def to_json(self, idx: int) -> dict:
        lines = self.source.splitlines(keepends=True)
        cid = f"c{idx:03d}"
        if self.kind == "markdown":
            return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": lines}
        return {
            "cell_type": "code", "id": cid, "metadata": {},
            "execution_count": None, "outputs": [], "source": lines,
        }


def parse_source(text: str) -> list[Cell]:
    """Split a ``# %%``-marked source file into cells."""
    lines = text.splitlines(keepends=True)
    cells: list[Cell] = []
    kind: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal buf, kind
        body = "".join(buf)
        if kind == "markdown":
            # Strip the leading "# " from each comment line to recover markdown text.
            md = "\n".join(
                ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln
                for ln in body.splitlines()
            )
            if md.strip():
                cells.append(Cell("markdown", md))
        elif kind == "code" and body.strip():
            cells.append(Cell("code", body))
        buf = []

    for ln in lines:
        if _MD_MARKER.match(ln.rstrip("\n")):
            flush()
            kind = "markdown"
            continue
        if _CODE_MARKER.match(ln.rstrip("\n")):
            flush()
            kind = "code"
            continue
        if kind is None:
            # Preamble before the first marker: treat as code (imports, docstring).
            kind = "code"
        buf.append(ln)
    flush()
    return cells


def lint(cells: list[Cell], name: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for i, c in enumerate(cells):
        for needle, why in FORBIDDEN:
            if needle in c.source:
                errors.append(f"{name} cell {i}: contains {needle!r} -- {why}")
        if c.kind == "code":
            for pat, why in WARN_RULES:
                if pat.search(c.source):
                    warnings.append(f"{name} cell {i}: {why}")
    return errors, warnings


def normalise(cells: list[Cell]) -> list[str]:
    return [c.source.strip() for c in cells if c.source.strip()]


def read_source(src: Path) -> str:
    """Read a source file, stripping a UTF-8 BOM if present.

    Windows editors (and PowerShell's `Set-Content -Encoding utf8`) like to write a
    BOM. It is invisible, and it silently breaks the very first `# %%` marker, which
    turns the notebook's opening markdown cell into a code cell.
    """
    return src.read_text(encoding="utf-8-sig")


def build(src: Path, out: Path) -> tuple[list[str], list[str]]:
    cells = parse_source(read_source(src))
    errors, warnings = lint(cells, src.name)
    if errors:
        return errors, warnings

    nb = {
        "cells": [c.to_json(i) for i, c in enumerate(cells)],
        "metadata": {
            "kernelspec": {"display_name": "ML-EO Course", "language": "python", "name": "ml_eo_course"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return errors, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="*", help="source stem filter(s), e.g. lab5")
    ap.add_argument("--check", action="store_true", help="report staleness/dirt, write nothing")
    args = ap.parse_args(argv)

    if not SRC_DIR.is_dir():
        print(f"no source directory: {SRC_DIR}", file=sys.stderr)
        return 1

    sources = sorted(SRC_DIR.glob("*.py"))
    if args.which:
        sources = [s for s in sources if any(w in s.stem for w in args.which)]
    if not sources:
        print("no matching sources", file=sys.stderr)
        return 1

    problems: list[str] = []
    n_ok = 0
    for src in sources:
        out = OUT_DIR / (src.stem + ".ipynb")
        cells = parse_source(read_source(src))
        errors, warnings = lint(cells, src.name)
        problems.extend(errors)
        for w in warnings:
            print(f"  warn: {w}")

        if args.check:
            if not out.exists():
                problems.append(f"{out.name}: missing")
            else:
                have = [
                    "".join(c.get("source", []))
                    for c in json.loads(out.read_text(encoding="utf-8"))["cells"]
                ]
                if normalise(cells) != [h.strip() for h in have]:
                    problems.append(f"{out.name}: stale -- run scripts/nbbuild.py")
        elif not errors:
            build(src, out)
            n_ok += 1
            print(f"  built {out.name} ({len(cells)} cells)")

    for p in problems:
        print(f"  ERROR: {p}", file=sys.stderr)
    print(f"\n{'checked' if args.check else 'built'} {len(sources)} source(s), "
          f"{n_ok} written, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
