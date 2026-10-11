"""Execute a notebook cell by cell in a chosen kernel and report what fails.

The obvious tool for this is ``nbconvert --execute`` or ``papermill``, and
neither is a course dependency -- so a checker that needed one could not run in
a student's environment. This uses only ``jupyter_client``, which ``ipykernel``
pulls in, and reports the *first* failing cell with its traceback instead of
dying with one giant notebook-wide error.

Run it on a JURECA login node, in the course venv, against the built notebook::

    $VENV/bin/python scripts/run_notebook_headless.py \\
        notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb --kernel ml_eo_course

The point is not to produce an executed notebook; it is to prove the lab runs
top-to-bottom in the kernel a student will actually use. Cells that ask the
student to fill in a ``PREDICTION`` will report as passing with ``None`` in the
values, which is correct -- the assertions downstream of them are written to
tolerate that.

Exits 0 if every executed cell succeeded, 1 otherwise, 2 if the kernel would
not start. A kernel that will not start is almost always the missing ``-E``
flag; see Lab 1 section 1.2.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def drain(kc, msg_id: str, timeout: float) -> tuple[str, str]:
    """Collect stream and error output for one execution until the kernel goes idle."""
    out, err = "", ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            m = kc.get_iopub_msg(timeout=5)
        except Exception:
            continue
        if m.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        if m["msg_type"] == "stream":
            out += m["content"]["text"]
        elif m["msg_type"] == "error":
            err = "\n".join(m["content"]["traceback"])
        elif m["msg_type"] == "status" and m["content"]["execution_state"] == "idle":
            return out, err
    return out, f"TIMEOUT after {timeout:.0f}s waiting for kernel"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--kernel", default="ml_eo_course",
                    help="jupyter kernel name; use python3 to test the "
                         "pre-environment sections")
    ap.add_argument("--only", default="", metavar="LO:HI",
                    help="run only notebook cells LO..HI (e.g. 3:10)")
    ap.add_argument("--skip", default="", metavar="I,J",
                    help="comma-separated cell indices to omit")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-cell wall clock; SLURM polls need minutes")
    ap.add_argument("--quiet", action="store_true", help="suppress cell output")
    ap.add_argument("--list-kernels", action="store_true")
    args = ap.parse_args(argv)

    if args.list_kernels:
        for name, spec in sorted(KernelSpecManager().find_kernel_specs().items()):
            print(f"  {name:24s} {spec}")
        return 0

    nb_path = Path(args.notebook).resolve()
    with nb_path.open(encoding="utf-8") as fh:
        nb = json.load(fh)
    cells = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
             if c["cell_type"] == "code"]
    if args.only:
        lo, hi = (int(x) for x in args.only.split(":"))
        cells = [(i, s) for i, s in cells if lo <= i < hi]
    if args.skip:
        drop = {int(x) for x in args.skip.split(",") if x.strip()}
        cells = [(i, s) for i, s in cells if i not in drop]
    if not cells:
        print("no code cells selected", file=sys.stderr)
        return 1
    idxs = [i for i, _ in cells]
    print(f"kernel={args.kernel}  cells={len(cells)}  indices={idxs[:3]}..{idxs[-3:]}")

    km = KernelManager(kernel_name=args.kernel)
    # Start in the notebook's own directory, exactly as JupyterLab would, so that
    # relative paths and `from eo_course import paths` behave the same way.
    km.start_kernel(cwd=str(nb_path.parent))
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=180)
    except RuntimeError as exc:
        print(f"KERNEL_START_FAILED ({args.kernel}): {exc}")
        print("If the kernel is the course one, this is the missing -E flag: "
              "re-run Lab 1 section 1.2.")
        km.shutdown_kernel()
        return 2

    mid = kc.execute("import sys; print('PREFIX', sys.prefix, sys.version.split()[0])")
    print(drain(kc, mid, 120)[0].strip())

    fails = 0
    for idx, src in cells:
        head = next((ln for ln in src.splitlines() if ln.strip()), "")[:76]
        t0 = time.time()
        mid = kc.execute(src)
        out, err = drain(kc, mid, args.timeout)
        dt = time.time() - t0
        if err:
            fails += 1
            print(f"\n### CELL {idx} FAILED after {dt:.1f}s: {head}")
            print(out[-2500:])
            print(err[-4000:])
            break
        print(f"cell {idx:3d} ok ({dt:5.1f}s)  {head}")
        if out.strip() and not args.quiet:
            print("    " + out.strip().replace("\n", "\n    ")[:1200])

    kc.stop_channels()
    km.shutdown_kernel()
    print(f"\nRESULT: {len(cells) - fails}/{len(cells)} clean -> "
          + ("NOTEBOOK_OK" if not fails else "NOTEBOOK_FAILED"))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
