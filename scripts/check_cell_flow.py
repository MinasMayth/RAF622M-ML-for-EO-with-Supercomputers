"""Check that the generated notebooks actually run top-to-bottom.

``nbbuild.py`` lints and ``check_notebooks.py`` compiles, and both were green
while Lab 2 contained three bugs that only appeared when a student ran it in
order:

* ``git_out()`` defaults to ``cwd=REPO``, but section 4 called it and section 5
  assigned ``REPO``. A function body is not executed at ``def`` time, so the
  ``NameError`` happens at the *call*, which no compile can see.
* ``matplotlib.get_config_dir()`` -- the real name is ``get_configdir()``. Six
  call sites, all valid Python, all ``AttributeError``.
* ``ip.getoutput("pwd").strip()`` -- IPython returns an ``SList`` (a list
  subclass), not a string.

Two audits, both static:

1. **Name flow.** For each cell, every name it loads must already be defined --
   either directly, or as a free name of a function the cell calls. This is the
   part that needs no third-party packages.
2. **Attribute resolution** (``--api``). Every ``module.attr`` access is checked
   with ``hasattr`` against the *installed* package. Run this one with the
   course interpreter, or every check is a false positive.

Both over-approximate bindings, so they can hide a problem but cannot invent
one. Neither replaces running the notebooks; they catch what a compiler cannot.

Usage
-----
    python scripts/check_cell_flow.py                      # name flow, all labs
    python scripts/check_cell_flow.py --api                # + attributes (course venv)
    python scripts/check_cell_flow.py --api "notebooks/iceland-ml/lab5*.ipynb"

Exits 1 if any audit reports a problem.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import glob
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = str(ROOT / "notebooks" / "iceland-ml" / "lab*.ipynb")

BUILTIN_NAMES = set(dir(builtins)) | {
    "__name__", "__file__", "__doc__", "__spec__", "__package__", "__builtins__",
}
# Names IPython puts in every kernel, so they are always available.
IPYTHON_NAMES = {"get_ipython", "In", "Out", "exit", "quit", "getcwd", "cd", "ls",
                 "_", "__", "open"}

# Submodules that only exist once something else has imported them; resolving
# them here would report the parent package and then fail on a real attribute.
SKIP_ATTR_MODULES: set[str] = set()


class Scope(ast.NodeVisitor):
    """Names loaded and names bound anywhere inside a subtree.

    Binding is over-approximated across nested scopes: a nested function's local
    counts as bound here. That can only suppress a finding, never create one.
    """

    def __init__(self) -> None:
        self.loads: set[str] = set()
        self.binds: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.add(node.id)
        else:
            self.binds.add(node.id)

    def _args(self, a: ast.arguments) -> None:
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]:
            self.binds.add(arg.arg)
        if a.vararg:
            self.binds.add(a.vararg.arg)
        if a.kwarg:
            self.binds.add(a.kwarg.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.binds.add(node.name)
        self._args(node.args)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.binds.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.binds.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.binds.add((alias.asname or alias.name).split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.binds.add((alias.asname or alias.name).split(".")[0])

    def visit_Global(self, node: ast.Global) -> None:
        # `global X` means loads of X refer to module scope, not this subtree
        self.binds.update(node.names)
        self.loads.update(node.names)

    def free(self) -> set[str]:
        return self.loads - self.binds - BUILTIN_NAMES - IPYTHON_NAMES


def collect(tree: ast.Module) -> tuple[set[str], dict[str, set[str]], set[str]]:
    """Return (names this cell defines, {func: free names}, names this cell loads)."""
    defined: set[str] = set()
    funcs: dict[str, set[str]] = {}
    loads: set[str] = set()

    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = Scope()
            scope.visit(stmt)
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                funcs[stmt.name] = scope.free()
            defined.add(stmt.name)
            # decorators run now, so their loads are this cell's loads
            for deco in stmt.decorator_list:
                for n in ast.walk(deco):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        loads.add(n.id)
            continue
        scope = Scope()
        scope.visit(stmt)
        defined.update(scope.binds)
        loads.update(scope.loads - scope.binds)
    return defined, funcs, loads


def propagate(funcs: dict[str, set[str]]) -> dict[str, set[str]]:
    """A function's free names include those of the functions it calls."""
    out = {k: set(v) for k, v in funcs.items()}
    for _ in range(len(out) + 1):
        changed = False
        for name, free in out.items():
            for other, ofree in out.items():
                if other != name and other in free and not (ofree - {name}) <= free:
                    free.update(ofree - {name})
                    changed = True
        if not changed:
            break
    return out


def check_name_flow(cells: list[tuple[int, str]]) -> list[str]:
    problems: list[str] = []
    known: set[str] = set()
    funcs: dict[str, set[str]] = {}
    for idx, src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # check_notebooks.py reports this
        defined, cell_funcs, loads = collect(tree)
        available = known | defined | BUILTIN_NAMES | IPYTHON_NAMES
        for name in sorted(loads - available):
            problems.append(f"cell {idx}: loads {name!r}, which is not defined by then")
        for fname, free in cell_funcs.items():
            funcs.setdefault(fname, set()).update(free)
        propagated = propagate(funcs)
        for name in sorted(loads & set(funcs)):
            missing = propagated[name] - available - set(funcs)
            if missing:
                problems.append(f"cell {idx}: calls {name}() which needs "
                                f"{sorted(missing)} not defined by then")
        known |= defined
    return problems


# ---------------------------------------------------------------- attribute API

def module_bindings(tree: ast.Module) -> dict[str, str]:
    """Map a bound name to the dotted module it refers to.

    ``import a.b.c`` binds the single name ``a`` (bound to the package ``a``),
    while ``import a.b.c as x`` binds ``x`` to ``a.b.c``. Getting this wrong makes
    every dotted access look like a typo.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    out[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    out[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                dotted = f"{node.module}.{alias.name}"
                try:
                    child = importlib.import_module(dotted)
                except Exception:
                    continue
                if hasattr(child, "__path__") or hasattr(child, "__file__"):
                    out[alias.asname or alias.name] = dotted
    return out


def dotted_chain(node: ast.Attribute) -> list[str] | None:
    """Flatten ``a.b.c`` to ``['a','b','c']``, or None if it is not a pure chain."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None
    parts.append(cur.id)
    return list(reversed(parts))


def resolve_chain(parts: list[str], bound: dict[str, str]) -> tuple[str, list[str]] | None:
    """Longest importable module prefix of ``parts``, plus what follows it.

    ``urllib.request.Request`` under ``import urllib.request`` starts as module
    ``urllib`` with chain ``request.Request``; extending the prefix as far as the
    importable modules allow leaves ``urllib.request`` and ``['Request']``.
    """
    if parts[0] not in bound:
        return None
    dotted = bound[parts[0]]
    rest = list(parts[1:])
    while rest:
        try:
            importlib.import_module(f"{dotted}.{rest[0]}")
        except Exception:
            break
        dotted = f"{dotted}.{rest[0]}"
        rest = rest[1:]
    if not rest:
        return None  # the whole chain named modules; nothing to check
    return dotted, rest


def check_attribute_api(cells: list[tuple[int, str]]) -> list[str]:
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    bound: dict[str, str] = {}
    for idx, src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        bound.update(module_bindings(tree))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = dotted_chain(node)
            if not parts:
                continue
            resolved = resolve_chain(parts, bound)
            if resolved is None:
                continue
            dotted, rest = resolved
            if dotted in SKIP_ATTR_MODULES:
                continue
            try:
                target = importlib.import_module(dotted)
            except Exception:
                continue  # not installed here; not our business
            ok = True
            for attr in rest[:-1]:
                target = getattr(target, attr, None)
                if target is None:
                    ok = False
                    break
            if ok and not hasattr(target, rest[-1]):
                key = (dotted, ".".join(rest))
                if key not in seen:
                    seen.add(key)
                    problems.append(f"cell {idx}: {key[0]}.{key[1]} does not exist "
                                    f"(written as {'.'.join(parts)})")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pattern", nargs="?", default=DEFAULT_GLOB, help="glob over .ipynb files")
    ap.add_argument("--api", action="store_true",
                    help="also resolve module attributes against installed packages; "
                         "run this with the course interpreter")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"no notebooks matched {args.pattern}", file=sys.stderr)
        return 1

    total = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            nb = json.load(fh)
        cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
                 if c["cell_type"] == "code"]
        problems = check_name_flow(cells)
        if args.api:
            problems += check_attribute_api(cells)
        name = Path(f).name
        if problems:
            total += len(problems)
            print(f"{name}:")
            for p in problems:
                print(f"  {p}")
        else:
            print(f"{name}: clean")
    print(f"\n{len(files)} notebook(s), {total} problem(s)"
          + ("" if args.api else "  (pass --api to also check module attributes)"))
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
