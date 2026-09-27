"""Compile every code cell of the generated notebooks.

``scripts/nbbuild.py`` lints notebook *sources* for known-bad patterns, but it
does not parse them: a cell with a stray bracket builds into a notebook that
only fails when a student runs it. This closes that gap by compiling each code
cell the way an interpreter would.

IPython line and cell magics (``%matplotlib inline``, ``%%time``) are legal in a
notebook and not legal Python, so they are commented out before compiling. That
is the only transformation applied -- no cell is skipped.

Usage
-----
    python scripts/check_notebooks.py                  # all notebooks
    python scripts/check_notebooks.py "notebooks/iceland-ml/lab5*.ipynb"

Exits 1 if any cell fails to compile.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = str(ROOT / "notebooks" / "iceland-ml" / "lab*.ipynb")

_MAGIC = re.compile(r"^(\s*)(%{1,2}\w+)")


def strip_magics(src: str) -> str:
    """Comment out IPython magics so the rest of the cell can be compiled."""
    out = []
    for line in src.splitlines():
        m = _MAGIC.match(line)
        out.append(m.group(1) + "#" + line[m.end(1):] if m else line)
    return "\n".join(out)


def check(pattern: str) -> tuple[int, int, list[str]]:
    files = sorted(glob.glob(pattern))
    n_cells = 0
    errors: list[str] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            nb = json.load(fh)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            n_cells += 1
            try:
                compile(strip_magics("".join(cell["source"])), f"cell {i}", "exec")
            except SyntaxError as e:
                errors.append(f"{Path(f).name} cell={i} line={e.lineno}: {e.msg}")
    return len(files), n_cells, errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default=DEFAULT_GLOB, help="glob over .ipynb files")
    args = ap.parse_args(argv)

    n_files, n_cells, errors = check(args.pattern)
    if not n_files:
        print(f"no notebooks matched {args.pattern}", file=sys.stderr)
        return 1
    for e in errors:
        print(f"SYNTAX {e}")
    print(f"\nchecked {n_files} notebook(s), {n_cells} code cell(s), "
          f"{len(errors)} syntax error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
