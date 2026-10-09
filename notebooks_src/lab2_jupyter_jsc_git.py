# %% [markdown]
# # Lab 2 — Jupyter-JSC, the course environment, and Git
#
# **RAF622M / TÖV606M — ML for Earth Observation powered by Supercomputers**
#
# ## What this lab produces
#
# Your first graded deliverable is **a green test suite**, not a screenshot. The last
# cell runs `pytest tests -q` against your clone and refuses to let the notebook
# finish if it is red. Everything before it exists to make that run possible:
#
# 1. The course virtual environment, in **project** space, built by `uv` from
#    `pyproject.toml` + `uv.lock`, registered as a Jupyter kernel.
# 2. Proof that the kernel you are typing in *is* that environment — by comparing
#    `sys.prefix` to the path, not by trusting the name in the menu.
# 3. The course repository cloned to a path the later labs can find.
# 4. A Git identity that is not the literal string `"Your Name"`, verified.
# 5. A Git workflow that demonstrably changes branches and commits, asserted after
#    every step.
# 6. A `results.json` record carrying your predictions and their outcomes.
#
# ## What it assumes from Lab 1
#
# A Judoor account in `training2653`, a working SSH key, and the workspace layout
# (`repos/`, `envs/`, `scripts/`, `logs/`, `results/`) under your project space. If
# you skipped Lab 1's layout cell, the clone in section 5 will create the parents for
# you, but `envs/` is where your environment must live, so make sure it exists.
#
# ## You must run this yourself
#
# The 2025/26 edition of this notebook had 14 code cells, 8 of them executed, and
# **zero errors** — because the cells that could not work were written so that they
# appeared to succeed. Its clone cell did nothing, its `source activate` cell could
# not run, and its branching demo printed `Already up to date.` while demonstrating
# nothing. Eight executed cells and no error is not a passing notebook; it is a
# notebook that never actually did anything. Every claim below is behind an `assert`.

# %% [markdown]
# ## 1. Setup
#
# `eo_course` is the course toolkit, and it lives in the repository you are about to
# clone. So this first cell has a bootstrap problem: it needs `paths` before the
# environment exists. We resolve it by walking up from the notebook's directory until
# we find the package, and we say out loud that this is bootstrap-only — it is *not*
# the same thing as the package being installed, which is what section 8 checks.

# %%
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_NAME = "RAF622M-ML-for-EO-with-Supercomputers"
REPO_URL = f"https://github.com/MinasMayth/{REPO_NAME}"

_here = Path.cwd().resolve()
for _cand in [_here, *_here.parents]:
    if (_cand / "eo_course" / "paths.py").is_file():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from eo_course import gates, paths, results  # noqa: E402

SEED = 20260116
import numpy as np  # noqa: E402

rng = np.random.default_rng(SEED)

print(paths.describe())

# %%
def run(cmd, cwd=None, env=None):
    """Run a command with an explicit working directory; echo it and capture output.

    The `cwd` argument is the whole point of this helper. IPython's `!` operator has
    no equivalent: it hands the string to `/bin/sh -c` in a **new process for every
    line**, so nothing about that process survives to the next `!` line. Section 3
    proves it. Every Git operation in this lab goes through `run(..., cwd=REPO)`.
    """
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    printable = cmd if isinstance(cmd, str) else " ".join(map(str, cmd))
    print(f"$ {printable}" + (f"    [cwd={cwd}]" if cwd else ""))
    if proc.stdout.strip():
        print(proc.stdout.rstrip())
    if proc.stderr.strip():
        print("[stderr] " + proc.stderr.rstrip())
    if proc.returncode != 0:
        print(f"[returncode] {proc.returncode}")
    return proc


def git(*args, cwd=None):
    return run(["git", *args], cwd=cwd or REPO)


def git_out(*args, cwd=None) -> str:
    p = subprocess.run(["git", *args], cwd=cwd or REPO, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()

# %% [markdown]
# ## 2. Launching Jupyter-JSC, and which machine you are actually on
#
# 1. Open <https://jupyter-jsc.fz-juelich.de> and log in with **Judoor**.
# 2. In the job form choose **System: JURECA**, **Project: training2653**, and a
#    wall-clock time that covers the lab.
# 3. For the interactive partition, **read the form's own list** and pick the
#    JURECA-DC interactive entry it offers you. Do not type a name from memory.
# 4. Click **Start**, wait for the job to leave `PENDING`, then **Connect**.
#
# The 2025/26 notebook told students to set **Partition: `login`**. `login` is not a
# JURECA partition — login nodes are not scheduled by SLURM at all — and it directly
# contradicted Lab 1, which only ever named `dc-cpu` and `dc-gpu`. A student who typed
# it got a form validation error with no explanation.
#
# Then there is the other half of the problem. That notebook's architecture diagram
# said `Your Browser → Jupyter-JSC Portal → SLURM Scheduler → Compute Node`, and two
# cells later a cell titled **"HPC Environment Check"** printed
# `Hostname: jrlogin04.jureca`. A login node. Nobody noticed the contradiction, and
# labs 3 and 4 then did multi-gigabyte raster I/O on it.
#
# **Why heavy I/O on a login node is a problem for the whole training project**, not
# just you: the login nodes are shared by every member of `training2653` and by the
# Jupyter-JSC launcher that starts everyone's sessions. One student reading a 20 GB
# GeoTIFF through a notebook cell saturates the Lustre client and the CPU, and
# everyone else's terminal stalls and their session fails to launch. JSC will kill
# the process and, if it recurs, restrict the account. The rule is: login node for
# editing, `git`, `sbatch`, and small checks; anything that reads a raster or loops
# over patches goes in a job.

# %%
HOST = socket.gethostname()
print(f"hostname      : {HOST}")
print(f"user          : {paths.username()}")
print(f"on JURECA     : {paths.on_jureca()}")
print(f"login node    : {paths.on_login_node()}")
print(f"SLURM_JOB_ID  : {os.environ.get('SLURM_JOB_ID', 'unset -> not inside a SLURM job')}")

assert paths.on_jureca(), (
    "This notebook must run on JURECA. Open it in a Jupyter-JSC session "
    f"(System=JURECA, Project={paths.PROJECT_ACCOUNT})."
)
if paths.on_login_node():
    print(
        "\nNOTE: you are on a login node. Fine for this lab — nothing here is heavy. "
        "It is not fine for labs 3-5. If you want a compute node for this notebook, "
        "launch it through Jupyter-JSC with a GPU/CPU allocation rather than a "
        "login-node session, and check that SLURM_JOB_ID above is set."
    )
print("\nOK: hostname printed and recorded. Do not assert it is a compute node — "
      "assert that you know which one you are on.")

# %% [markdown]
# ## 3. The single most useful thing in this lab
#
# Every `!` line in a Jupyter cell is executed as its own `/bin/sh -c` process. The
# process exits. Its working directory, its environment variables, its activated
# virtualenv — all gone. The next `!` line starts a brand-new shell that knows nothing
# about the previous one.
#
# Here is what that cost the 2025/26 edition. Its clone cell was:
#
# ```python
# !cd $PROJECT_training2600
# !mkdir -p $USER
# !cd $USER
# !git clone <url>
# !cd <repo>
# !git status
# ```
#
# The three `cd` lines did nothing. `mkdir` and `git clone` ran **in the notebook's own
# directory**, and `git status` reported on whatever the notebook happened to live in.
# The cell produced no error, so it looked like it worked.
#
# The same mechanism killed two more cells in that notebook:
#
# * `!source ~/envs/.../bin/activate` — `source` is a **bash builtin**, and `/bin/sh`
#   on most Linux systems is `dash`, which has no `source`. That line could not have
#   worked on any machine.
# * the branching demo, `!git branch` / `!git checkout` / `!git merge` — no checkout
#   persisted, so the commit landed on whatever branch was already current, and
#   `git merge` printed `Already up to date.` The cell *appeared* to succeed while
#   demonstrating nothing at all.
#
# ### Exercise 1 — predict before you run
#
# The next cell runs `cd /tmp` and then, separately, `pwd`.
#
# 1. What will `pwd` print? Write the full path.
# 2. Will the `cd /tmp` line report success or failure? Answer "success" or "failure".
#
# Write them in the cell after this one, then run it.
#
# (We call `get_ipython().system(...)` rather than typing `!cd /tmp` because `!cmd` is
# exactly sugar for `get_ipython().system("cmd")` — same mechanism, and it keeps this
# cell valid Python. The build linter in `scripts/nbbuild.py` refuses a line starting
# `!cd` for precisely the reason you are about to watch.)

# %%
PREDICTION = {
    "pwd_after_cd": None,        # TODO(you): the full path you expect
    "cd_line_status": None,      # TODO(you): "success" | "failure"
}
print("Fill in PREDICTION, run this cell, then run the next one.")

# %%
ip = get_ipython()  # noqa: F821

# Exactly what `!cd /tmp` compiles to. The shell prints nothing and exits.
cd_status = ip.system("cd /tmp")
# Exactly what `!pwd` compiles to, in a NEW shell that never saw the cd.
printed_pwd = ip.system("pwd")
actual = ip.getoutput("pwd").strip()

print(f"\n`cd /tmp` exit status: {cd_status}   (it succeeded — in its own process)")
print(f"`pwd` exit status    : {printed_pwd}")
print(f"the shell's cwd was  : {actual}")
print(f"you predicted        : {PREDICTION['pwd_after_cd']!r}")
assert Path(actual) != Path("/tmp"), "the cd would have had to persist, and it cannot"
print(f"\nProof: the second shell started in {actual}, not /tmp. Two `!` lines, two "
      "processes, no shared state.")

# %%
correct_cwd = Path.cwd()
run(["bash", "-c", "cd /tmp && pwd"])
print("\none shell, one command: it moved to /tmp and reported it.")
print(f"our Python process is still in: {Path.cwd()}")
assert Path.cwd() == correct_cwd, "the notebook's own cwd must not move"

os.chdir(paths.user_scratch())
print(f"\nafter os.chdir(paths.user_scratch()): {Path.cwd()}")
run(["pwd"])
os.chdir(correct_cwd)
print(f"back to: {Path.cwd()}")
print(
    "\nRule: `os.chdir()` for the kernel, `run(..., cwd=...)` for one subprocess, or "
    "one `!bash -c \"cd X && Y && Z\"` per logical sequence. Never a bare `!cd`."
)

# %% [markdown]
# ## 4. Git identity — and the placeholder that shipped
#
# Git stamps an author name and email into every commit object, permanently, in every
# clone anyone ever makes of your history. The 2025/26 notebook contained:
#
# ```python
# !git config --global user.name "Your Name"
# !git config --global user.email "your.email@hi.is"
# ```
#
# committed verbatim. A student who ran it as written set their **global** identity to
# the literal string `Your Name`, and every commit they made for the rest of the course
# — including the ones carrying their Lab 5 results — carried it. There was no warning
# and no check.
#
# We set it from environment variables so there is no placeholder to copy, and then we
# **assert** the values are sane rather than assuming.

# %%
run(["git", "--version"])

name = os.environ.get("COURSE_GIT_NAME", "")
email = os.environ.get("COURSE_GIT_EMAIL", "")
if name:
    run(["git", "config", "--global", "user.name", name])
if email:
    run(["git", "config", "--global", "user.email", email])

cfg_name = git_out("config", "--global", "user.name") if subprocess.run(
    ["git", "config", "--global", "user.name"], capture_output=True, text=True).returncode == 0 else ""
cfg_email = git_out("config", "--global", "user.email") if subprocess.run(
    ["git", "config", "--global", "user.email"], capture_output=True, text=True).returncode == 0 else ""
print(f"\nglobal user.name  = {cfg_name!r}")
print(f"global user.email = {cfg_email!r}")
print(
    "\nIf either is empty or wrong, set it yourself in the JupyterLab terminal:\n"
    "  git config --global user.name \"Your Real Name\"\n"
    "  git config --global user.email \"you@your-institution\"\n"
    "then re-run this cell and the check in section 12."
)

# %% [markdown]
# ## 5. Cloning the course repository
#
# The 2025/26 clone cell pointed at a placeholder organisation slug and a repository
# name that has never existed on GitHub. No notebook, README, or docs page anywhere in
# that edition gave students the real URL, so Lab 2's core deliverable — "clone the
# course repository" — was **unachievable as written**. It was the single hardest
# blocker in the set, and it produced no error message that pointed at the cause: just
# `repository not found`.
#
# The real URL is the one below, taken from this repository's own `git remote -v`.
# Fork it on GitHub if you intend to push; clone the upstream if you only need to read
# and run.

# %%
REPO = paths.user_project("repos") / REPO_NAME
REPO.parent.mkdir(parents=True, exist_ok=True)
print(f"target path: {REPO}")

if (REPO / ".git").is_dir():
    print("already cloned; fetching instead of cloning.")
    git("fetch", "--all", "--prune", cwd=REPO)
else:
    p = run(["git", "clone", REPO_URL, str(REPO)])
    assert p.returncode == 0, (
        f"clone failed. URL: {REPO_URL}\n"
        "Check the URL against the one the instructor posted, and check you have "
        "access to the course GitHub organisation."
    )

assert (REPO / ".git").is_dir(), f"{REPO} is not a git repository"
assert (REPO / "eo_course" / "paths.py").is_file(), "cloned repo has no eo_course package"
assert (REPO / "pyproject.toml").is_file() and (REPO / "uv.lock").is_file(), \
    "cloned repo has no pyproject.toml/uv.lock — wrong repository?"
print(f"\nOK: repository at {REPO}")
print(f"origin = {git_out('remote', 'get-url', 'origin', cwd=REPO)}")
print(f"HEAD   = {git_out('rev-parse', '--short', 'HEAD', cwd=REPO)}  "
      f"branch {git_out('branch', '--show-current', cwd=REPO)}")

# %% [markdown]
# ## 6. Building the course environment with `uv`
#
# Three things the 2025/26 environment cell got wrong, each with a number attached:
#
# 1. **It put three virtualenvs in `$HOME`.** Home quota is ~50 GB and is backed up,
#    which means every one of those environments was copied into the nightly backup.
#    The same notebook's next-but-one cell printed a matplotlib warning proving home
#    was not writable, and nobody connected the two. Environments go in **project**
#    space: `paths.user_project("envs")`.
# 2. **It activated the environment with `!source`.** Cannot work — see section 3.
# 3. **It installed a hand-typed package list** (`pip install ipykernel numpy pandas
#    matplotlib jupyterlab rasterio earthengine-api torch`). That list omitted
#    `geemap`, which Lab 3 imported, and `lightning`, which Labs 5–7 import. It also
#    ignored `uv.lock`, so two students running the same lab got two different
#    dependency graphs.
#
# The correct path is `uv sync`, which reads `pyproject.toml` and installs exactly what
# `uv.lock` pins. `geemap` and `earthengine-api` are an **optional extra** (`gee`)
# because Lab 3 also works through the Copernicus Data Space API; install them only if
# you are taking that path.
#
# One more mechanism worth knowing: `module` is a **shell function** defined by
# modulecmd's shell hook, not a program on `PATH`. That is why `!module load Python`
# fails with `module: command not found` while the same line works in a terminal and in
# an `sbatch` script. Anything using `module` must run inside `bash -l -c`.

# %%
VENV = paths.user_project("envs") / "ml_eo_course"
print(f"expected environment: {VENV}")

which_uv = subprocess.run(["bash", "-lc", "command -v uv"], capture_output=True, text=True)
if which_uv.returncode != 0:
    print("`uv` not found. Install it (user-local, no root needed) with:\n"
          "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
          "then restart the kernel so PATH is refreshed.")
else:
    print(f"uv: {which_uv.stdout.strip()}")
    print(subprocess.run(["bash", "-lc", "uv --version"], capture_output=True, text=True).stdout.strip())

# %% [markdown]
# ### Why a uv-managed interpreter, and not `module load Python`
#
# The tempting recipe is `module load Stages/2026 GCCcore/14.3.0 Python/3.13.5` and
# then build the virtualenv on top of that. It builds, it even verifies, and then it
# breaks the moment you stop watching it, for two separate reasons:
#
# 1. A venv created from a module-backed CPython keeps a **dynamic dependency on
#    `libpython3.13.so.1.0` inside the module tree**. It runs in the shell that loaded
#    the module and fails with `cannot open shared object file` anywhere else -- which
#    includes the Jupyter kernel. Try it: build one this way, then run
#    `env -u LD_LIBRARY_PATH $VENV/bin/python -c "import sys"` and watch it die.
# 2. The module can be re-staged or retired underneath you. That is exactly how the
#    2025/26 environment died: it was built against a `Python/3.12` that later
#    disappeared, and a whole cohort inherited
#    `libpython3.12.so.1.0: cannot open shared object file`.
#
# `uv python install` puts a self-contained CPython in your scratch directory. The
# venv has no dependency on the module tree at all, so it survives module churn and
# works in a kernel. That is the point of `UV_PYTHON` below.
#
# Note this cell runs `uv sync` **here**, in the notebook. Compute nodes have no route
# to PyPI, so package installation belongs on a login node -- which is what a
# Jupyter-JSC session is. Do not move this cell into a `srun` wrapper.

# %%
# Scratch, not home: the interpreter is ~150 MB, home is quota-limited and backed up.
UV_PY_HOME = paths.user_scratch(".uv-python")
print(f"uv-managed interpreter dir: {UV_PY_HOME}")

SYNC = f"""
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_INSTALL_DIR="${{UV_PYTHON_INSTALL_DIR:-{UV_PY_HOME}}}"
export UV_PYTHON="${{UV_PYTHON:-3.12}}"
export UV_PROJECT_ENVIRONMENT={VENV}
uv python install "$UV_PYTHON"
uv sync --frozen
echo "--- python in the environment ---"
# -u LD_LIBRARY_PATH proves the interpreter is self-contained rather than quietly
# inheriting libpython from whatever modules this shell happens to have loaded.
env -u LD_LIBRARY_PATH "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import sys; print(sys.version); print(sys.prefix)"
"""
sync = run(["bash", "-lc", SYNC], cwd=REPO)
assert sync.returncode == 0, (
    "uv sync failed. Read the output above. The two common causes on JURECA are a full "
    "project quota (check Lab 1's quota cell) and running this from a compute node, "
    "which has no route to PyPI; the error names whichever it is."
)
assert (VENV / "bin" / "python").is_file(), f"{VENV} was not created"
print(f"\nOK: environment exists at {VENV}")

# %%
# The gate is not "the venv exists" but "the venv runs without help". A kernel does not
# inherit your login shell's LD_LIBRARY_PATH, so neither may the environment.
probe = run(["env", "-u", "LD_LIBRARY_PATH", str(VENV / "bin" / "python"),
             "-c", "import sys; print(sys.prefix)"])
assert probe.returncode == 0, (
    "The environment only works with module-provided libraries on LD_LIBRARY_PATH. "
    "It will fail in Jupyter. Delete it and re-run the cell above so it is rebuilt "
    "on the uv-managed interpreter."
)
print("OK: the interpreter is self-contained and will work in a kernel.")

# %% [markdown]
# Optional, only if you are taking the Google Earth Engine path in Lab 3:

# %%
gee = run(["bash", "-lc", f"cd {REPO} && UV_PROJECT_ENVIRONMENT={VENV} uv sync --frozen --extra gee"])
if gee.returncode == 0:
    check = run([str(VENV / "bin" / "python"), "-c", "import geemap, ee; print('geemap', geemap.__version__)"])
    assert check.returncode == 0, "gee extra installed but geemap still does not import"
else:
    print("gee extra skipped. Fine if you are using the Copernicus Data Space API in Lab 3.")

# %% [markdown]
# ## 7. Registering the kernel
#
# `python -m ipykernel install --user` writes a small JSON directory into
# `$HOME/.local/share/jupyter/kernels`. That is a few kilobytes, so home is the right
# place for it — unlike the environment itself, which is gigabytes. The 2025/26 lab
# blurred the two and put both in home.
#
# **After running the next cell you must switch kernels.** In JupyterLab:
# **Kernel → Change Kernel → ML-EO Course**, then re-run section 1 and continue. Until
# you do, you are still typing in whatever kernel the notebook started with, and the
# assertion in section 8 will fail. That failure is the gate; it is supposed to fail.

# %%
# `--user` refuses to overwrite, and a kernel spec left by an earlier course run may be
# a *symlink into a virtualenv that no longer exists*. ipykernel happily writes through
# such a symlink, and Jupyter then offers a kernel that dies on contact -- which is
# what the 2025/26 cohort saw. Clear the old spec first.
KDIR = Path.home() / ".local" / "share" / "jupyter" / "kernels" / "ml_eo_course"
if KDIR.is_symlink() or KDIR.exists():
    print(f"removing existing kernel spec: {KDIR}"
          + (f" -> {KDIR.resolve()}" if KDIR.is_symlink() else ""))
    if KDIR.is_dir() and not KDIR.is_symlink():
        shutil.rmtree(KDIR)
    else:
        KDIR.unlink()  # a real file, or a symlink pointing at nothing

reg = run([str(VENV / "bin" / "python"), "-m", "ipykernel", "install", "--user",
           "--name", "ml_eo_course", "--display-name", "ML-EO Course"])
assert reg.returncode == 0, "kernel registration failed"

assert KDIR.is_dir() and not KDIR.is_symlink(), f"{KDIR} is not a real directory"
assert (KDIR / "kernel.json").is_file(), "kernel.json was not written"
print((KDIR / "kernel.json").read_text())

kernels = run([str(VENV / "bin" / "python"), "-m", "jupyter", "kernelspec", "list"])
assert "ml_eo_course" in kernels.stdout, "kernel not in the kernelspec list"
print("\nKernel registered. Now switch this notebook to **ML-EO Course** and re-run "
      "section 1, then continue.")

# %% [markdown]
# ## 8. Are you in the course environment? (this is the gate)
#
# The notebook menu says a kernel name. Names are cheap. `sys.prefix` is the truth: it
# is the root of the interpreter that is actually executing this cell.

# %%
print(f"sys.executable = {sys.executable}")
print(f"sys.prefix     = {sys.prefix}")
print(f"expected       = {VENV}")

assert Path(sys.prefix).resolve() == VENV.resolve(), (
    "You are NOT in the course environment.\n"
    f"  running : {sys.prefix}\n"
    f"  expected: {VENV}\n"
    "In JupyterLab: Kernel -> Change Kernel -> ML-EO Course, then restart and re-run "
    "from section 1. Do not skip this; every later lab assumes this interpreter."
)
print("\nOK: the kernel you are typing in is the course environment.")

# %% [markdown]
# ### Installing anything else, correctly
#
# The 2025/26 notebook ran `!pip install lckr-jupyterlab-variableinspector` after
# spending twenty minutes arguing that the active kernel must be the isolated course
# venv. `pip` on `PATH` is whichever `pip` the *shell* resolves, which on JURECA is
# usually not the venv's. So the install landed in whatever interpreter `PATH` pointed
# at, and the notebook's own kernel never saw it.
#
# The notebook-idiomatic fix is `import sys` then
# `!{sys.executable} -m pip install <pkg>`: `sys.executable` is *this* interpreter, and
# `-m pip` makes pip run inside it rather than being found on `PATH`. We use the
# strictly equivalent `subprocess.run([sys.executable, "-m", "pip", ...])` so the cell
# stays valid Python.

# %%
print(f"pip that would run from PATH : {subprocess.run(['bash', '-lc', 'command -v pip'], capture_output=True, text=True).stdout.strip() or '<none>'}")
print(f"pip inside this kernel       : {sys.executable} -m pip")
print("\nAlways the second form. For course packages, prefer `uv add` / `uv sync` so the "
      "change lands in pyproject.toml and uv.lock and your partner gets it too.")

# %% [markdown]
# ## 9. Matplotlib, and the fix that was applied too late
#
# The 2025/26 notebook imported matplotlib at the top of a cell and set `MPLCONFIGDIR`
# further down the *same* cell. Its own stored output is the proof that this did
# nothing:
#
# > `Matplotlib created a temporary cache directory at /tmp/matplotlib-gdtdnool
# > because the default path (/p/home/jusers/.../.cache/matplotlib) is not a writable
# > directory`
#
# Matplotlib resolves its config directory during `import matplotlib`, so by the time
# the assignment ran the decision had already been made. The same bug appeared verbatim
# in two later labs. On a shared login node, `/tmp` is shared too: the fallback cache
# is world-readable, and on a busy node the fallback creation races and fails outright.
#
# The order is: set the variable, then import.

# %%
mpl_dir = paths.matplotlib_cache_dir()
os.environ["MPLCONFIGDIR"] = mpl_dir
print(f"MPLCONFIGDIR = {os.environ['MPLCONFIGDIR']}")

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

print(f"matplotlib   = {matplotlib.__version__}")
print(f"resolved cache dir = {matplotlib.get_config_dir()}")
assert Path(matplotlib.get_config_dir()).resolve() == Path(mpl_dir).resolve(), (
    "MPLCONFIGDIR was not honoured — it was set after the import, or is not writable"
)
assert "/tmp" not in matplotlib.get_config_dir(), "matplotlib fell back to /tmp"
print("OK: cache dir is in your scratch space, set before the import.")

# %%
# One figure, to prove the backend renders and writes where we expect. The course rule
# is savefig *and* show: show() alone leaves nothing behind on a headless node, and
# savefig alone leaves you guessing what the plot looked like.
fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(rng.normal(0, 1, 50).cumsum(), lw=1)
ax.set_title("matplotlib renders and writes")
ax.set_xlabel("step")
ax.set_ylabel("cumulative")
fig_path = paths.user_project("results") / "lab2_mpl_check.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(fig_path, dpi=90, bbox_inches="tight")
plt.show()
assert fig_path.stat().st_size > 1000, f"figure not written: {fig_path}"
print(f"wrote {fig_path} ({fig_path.stat().st_size:,} bytes)")

# %% [markdown]
# ## 10. A Git workflow that actually works
#
# Every step goes through `git(...)` with an explicit `cwd=REPO`, and every step is
# followed by an assertion that state really changed. Compare with the 2025/26 demo,
# which ran `!git branch`, `!git checkout`, `!git merge` as separate subprocesses and
# printed `Already up to date.` because no checkout had persisted.
#
# We also avoid `git add .`. On this course `git add .` is how `.npz` files reach a
# commit; section 11 makes that concrete.

# %%
BRANCH = f"lab2-{paths.username()}"
before_head = git_out("rev-parse", "HEAD")
before_branch = git_out("branch", "--show-current")
print(f"start: branch={before_branch!r} head={before_head[:8]}")

git("checkout", "-B", BRANCH)
after_branch = git_out("branch", "--show-current")
assert after_branch == BRANCH, f"checkout did not persist: still on {after_branch!r}"
print(f"OK: now on {after_branch!r}  (the 2025/26 cell could not show this)")

# %%
workfile = REPO / "notebooks" / "iceland-ml" / f"lab2_{paths.username()}.md"
workfile.write_text(
    f"# Lab 2 log — {paths.username()}\n\n"
    f"- kernel: `{sys.prefix}`\n- host: `{HOST}`\n- seed: `{SEED}`\n"
)
git("add", str(workfile.relative_to(REPO)))
staged = git_out("diff", "--cached", "--name-only").splitlines()
assert str(workfile.relative_to(REPO)) in staged, f"nothing staged: {staged}"
print(f"staged: {staged}")

msg = f"lab2: {paths.username()} environment log"
commit = git("commit", "-m", msg)
assert commit.returncode == 0, "commit failed — read stderr, it usually names the identity problem"
after_head = git_out("rev-parse", "HEAD")
assert after_head != before_head, "HEAD did not move: the commit did not happen"
print(f"OK: HEAD moved {before_head[:8]} -> {after_head[:8]}")

log = git_out("log", "--oneline", "-3")
print(f"\n{log}")
author = git_out("log", "-1", "--format=%an <%ae>")
print(f"author of that commit: {author}")

# %% [markdown]
# ## 11. Do not commit data artifacts
#
# This is the Git mistake that actually costs you on this course, and the 2025/26 lab
# never mentioned it. Labs 3 and 4 write `.tif`, `.npz`, `.npy` and `.pt` files, and
# the notebooks live inside the repository, so the artifacts land **next to** the code.
# A single Lab 4 patch archive is multi-gigabyte. One such file in history makes every
# future `git clone` by your partner take an hour, and it cannot be removed later
# without rewriting history.
#
# ### Exercise 2 — predict before you run
#
# The next cell writes two fake artifacts into your clone: a 5 MB `scene.tif` and a
# 2 MB `patches.npz`, in a `data/` directory inside the repository.
#
# 1. How many entries will `git status --porcelain` list? Give an integer.
# 2. Will `git check-ignore` report them as ignored, or not ignored?
# 3. If you then ran `git add . && git commit`, how many bytes would enter history?
#    Give a number in MB.
#
# Write them down, then run it.

# %%
PREDICTION["artifacts_porcelain_lines"] = None  # TODO(you): integer
PREDICTION["artifacts_ignored"] = None          # TODO(you): True | False
PREDICTION["bytes_into_history_MB"] = None      # TODO(you): number in MB
print(PREDICTION)

# %%
art_dir = REPO / "data"
art_dir.mkdir(exist_ok=True)
(art_dir / "scene.tif").write_bytes(b"\0" * 5_000_000)
(art_dir / "patches.npz").write_bytes(b"\0" * 2_000_000)

porcelain = git_out("status", "--porcelain", "--untracked-files=all")
print("git status --porcelain --untracked-files=all:")
print(porcelain or "(clean)")
art_lines = [ln for ln in porcelain.splitlines() if "data/" in ln]
size_mb = sum(p.stat().st_size for p in art_dir.iterdir()) / 1e6
print(f"\n{len(art_lines)} artifact entries; {size_mb:.1f} MB would enter history "
      f"if you ran `git add .`.")
print(f"you predicted {PREDICTION['artifacts_porcelain_lines']} entries and "
      f"ignored={PREDICTION['artifacts_ignored']}")
assert art_lines, "git status did not surface the artifacts — check you are in the right repo"

# %%
ci = subprocess.run(["git", "check-ignore", "-v", "data/scene.tif", "data/patches.npz"],
                    cwd=REPO, capture_output=True, text=True)
covered = ci.returncode == 0
print(f"git check-ignore exit={ci.returncode} -> patterns cover these artifacts: {covered}")
print(ci.stdout.strip() or "(no matching ignore pattern)")
print(f"you predicted ignored={PREDICTION['artifacts_ignored']}")

if not covered:
    print(
        "\nThe repository's .gitignore does NOT cover raster/array artifacts. The "
        "durable fix is a pull request adding the patterns to .gitignore, which is a "
        "real contribution — send one. The immediate fix is `.git/info/exclude`, a "
        "repo-local ignore file that is never committed and so cannot conflict with "
        "anyone else."
    )
    exclude = REPO / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude.read_text() if exclude.exists() else ""
    patterns = ["*.tif", "*.tiff", "*.npz", "*.npy", "*.pt", "*.ckpt", "*.h5", "data/"]
    missing = [p for p in patterns if p not in existing.split()]
    with exclude.open("a") as fh:
        fh.write("\n# RAF622M lab data artifacts (local only; propose a .gitignore PR)\n")
        fh.write("".join(p + "\n" for p in missing))
    print(f"added {len(missing)} patterns to {exclude}")

ci2 = subprocess.run(["git", "check-ignore", "-v", "data/scene.tif", "data/patches.npz"],
                     cwd=REPO, capture_output=True, text=True)
assert ci2.returncode == 0, "artifacts are still not ignored — the exclusion did not take"
print("\nnow ignored by:")
print(ci2.stdout.strip())

after = git_out("status", "--porcelain", "--untracked-files=all")
assert "data/scene.tif" not in after, "still visible to git status"
print(f"\ngit status after exclusion:\n{after or '(clean)'}")

# %%
for p in (art_dir / "scene.tif", art_dir / "patches.npz"):
    p.unlink(missing_ok=True)
art_dir.rmdir()
print(f"removed the fake artifacts. Working tree: {git_out('status', '--porcelain') or '(clean)'}")
print(
    "\nHabit: `git status` before every `git add`, and `git add <file>` rather than "
    "`git add .`. Check what you are about to send with `git diff --cached --stat`. "
    "If a large file is already in history, `git log --stat -- '*.npz' '*.tif'` finds it."
)

# %% [markdown]
# ## 12. Exercise 3 — predict the merge conflict
#
# The 2025/26 lab told students to provoke a conflict with `!git pull origin main`
# inside a notebook cell. Two problems: a non-interactive subprocess that makes git
# open `$EDITOR` for the merge message hangs forever, and "resolve it by editing the
# file" cannot be done from a cell that has already finished running.
#
# We do it properly, in a throwaway repository under scratch with a real local
# `origin`, so your course clone is never at risk. The scenario is the ordinary one:
# you and a partner edit the same line.
#
# **Predict, before running:**
#
# 1. Mid-conflict, `git status --porcelain` prints two characters per file. What are
#    they for a content conflict? Give the two characters.
# 2. How many conflict-marker lines (`<<<<<<<`, `=======`, `>>>>>>>`) will the file
#    contain? Give an integer.
# 3. After you resolve and `git add` the file, what will `git status --porcelain`
#    print for it — the same code, a different one, or nothing at all?
#
# Write the three answers in the next cell.

# %%
PREDICTION["conflict_status_code"] = None       # TODO(you): two characters
PREDICTION["conflict_marker_lines"] = None      # TODO(you): integer
PREDICTION["after_add_porcelain"] = None        # TODO(you): "same" | "different" | "nothing"
print(PREDICTION)

# %%
DEMO = paths.user_scratch("lab2_conflict_demo")
shutil.rmtree(DEMO, ignore_errors=True)
(DEMO / "origin.git").mkdir(parents=True)
run(["git", "-c", "init.defaultBranch=main", "init", "--bare", str(DEMO / "origin.git")])
A = DEMO / "cloneA"
run(["git", "clone", str(DEMO / "origin.git"), str(A)], cwd=DEMO)
# A clone of an empty repository has no branch at all, so create it explicitly rather
# than relying on init.defaultBranch, which differs between git versions and users.
run(["git", "checkout", "-B", "main"], cwd=A)
run(["git", "config", "user.name", "Partner"], cwd=A)
run(["git", "config", "user.email", "partner@example.invalid"], cwd=A)
(A / "protocol.md").write_text("cloud threshold: 20 percent\n")
run(["git", "add", "protocol.md"], cwd=A)
run(["git", "commit", "-m", "initial protocol"], cwd=A)
run(["git", "push", "-u", "origin", "main"], cwd=A)

B = DEMO / "cloneB"
run(["git", "clone", str(DEMO / "origin.git"), str(B)], cwd=DEMO)
run(["git", "config", "user.name", "You"], cwd=B)
run(["git", "config", "user.email", "you@example.invalid"], cwd=B)
(B / "protocol.md").write_text("cloud threshold: 10 percent\n")
run(["git", "add", "protocol.md"], cwd=B)
run(["git", "commit", "-m", "tighten cloud threshold to 10"], cwd=B)
run(["git", "push", "origin", "main"], cwd=B)
print("partner's clone pushed a different value for the same line.")

# %%
(A / "protocol.md").write_text("cloud threshold: 30 percent\n")
run(["git", "add", "protocol.md"], cwd=A)
run(["git", "commit", "-m", "relax cloud threshold to 30"], cwd=A)
pull = run(["git", "pull", "--no-rebase", "--no-edit", "origin", "main"], cwd=A)

mid = subprocess.run(["git", "status", "--porcelain"], cwd=A, capture_output=True, text=True).stdout
print(f"\ngit status --porcelain mid-conflict:\n{mid.rstrip() or '(clean)'}")
codes = [ln[:2] for ln in mid.splitlines()]
markers = sum(1 for ln in (A / "protocol.md").read_text().splitlines()
              if ln.startswith(("<<<<<<<", "=======", ">>>>>>>")))
print(f"status code(s): {codes}")
print(f"conflict-marker lines in protocol.md: {markers}")
print(f"\nyou predicted: code={PREDICTION['conflict_status_code']!r} "
      f"markers={PREDICTION['conflict_marker_lines']}")
assert "UU" in codes, f"expected an unmerged UU entry, got {codes}"
assert "<<<<<<<" in (A / "protocol.md").read_text(), "no conflict markers in the file"
print("\n`UU` = unmerged, both modified. This is the state the 2025/26 cell left "
      "students in with no way out.")

# %%
print((A / "protocol.md").read_text())
print(
    "You resolve it by deciding, not by deleting markers at random: open the file, "
    "keep the content you want, remove the `<<<<<<<` / `=======` / `>>>>>>>` lines, "
    "`git add` the file, then `git commit`. `git add` is what tells git the conflict "
    "is resolved. To abandon the whole merge instead: `git merge --abort`."
)

(A / "protocol.md").write_text("cloud threshold: 20 percent  # agreed 2026-01-15\n")
run(["git", "add", "protocol.md"], cwd=A)
post = subprocess.run(["git", "status", "--porcelain"], cwd=A, capture_output=True, text=True).stdout
print(f"after add, porcelain for the file: {post.rstrip() or '(clean)'}")
print(f"you predicted: {PREDICTION['after_add_porcelain']!r}")
run(["git", "commit", "--no-edit", "-m", "merge partner's threshold, agree on 20"], cwd=A)
clean = subprocess.run(["git", "status", "--porcelain"], cwd=A, capture_output=True, text=True).stdout
assert not clean, f"merge not finished: {clean}"
print(f"\nOK: resolved and committed. `git log --graph` in {A} shows the merge.")
run(["git", "log", "--graph", "--oneline", "--all"], cwd=A)
print(
    "\nNow do it in the JupyterLab **terminal**, on your own clone, so your hands know "
    "the sequence: `git pull`, open the file, edit, `git add`, `git commit`. A notebook "
    "cell cannot hold you in a conflicted state while you think."
)

# %% [markdown]
# ## 13. Exercise 4 — a real dataset, and no answer
#
# The 2025/26 lab stated a four-part task ("load a sample CSV, do statistics,
# visualise, save results") and then spent the next four cells supplying the complete,
# commented solution to all four. Worse, the supplied answer did not satisfy part 1: no
# CSV was ever read. The data came from `np.random.uniform`, the scene IDs were
# fabricated strings like `S2A_TILE_001`, and the cloud cover was a uniform random
# number — so the "21.7 % of scenes are clear" the notebook printed was a property of
# the random number generator, not of the atmosphere. The 20 % cloud threshold was
# never justified because there was nothing for it to be a threshold *on*.
#
# That is the pattern this course exists to break: an analysis that runs end to end,
# prints plausible numbers, and measures nothing.
#
# ### The data
#
# The next cell downloads a **real** dataset: NASA GISTEMP global monthly temperature
# anomalies, a CSV that has been published at the same URL for two decades. If the
# download fails (no network route from this node), the cell says so plainly and writes
# a clearly-labelled synthetic substitute instead — and then asks you what that
# substitution implies for every number you produce afterwards.
#
# ### Your task — write it yourself, in a new notebook or a `.py` file
#
# In `paths.user_project("results")`, produce four artifacts from that CSV:
#
# 1. `lab2_stats.csv` — one row per calendar year, with columns `year`, `mean_anomaly`,
#    `n_months`, `min_anomaly`, `max_anomaly`. Exactly 12 `n_months` for every complete
#    year.
# 2. `lab2_trend.png` — a figure of the annual mean anomaly against year, with axis
#    labels and a linear fit drawn on it.
# 3. `lab2_summary.json` — keys `source_url`, `n_rows_read`, `first_year`, `last_year`,
#    `slope_per_decade`, `warmest_year`, `coldest_year`.
# 4. `lab2_notes.md` — three sentences: what the linear fit does and does not claim,
#    and what would change if the data were synthetic.
#
# The checker cell after the download asserts these exist and have the right shape. It
# does **not** compute them for you. Budget 30–40 minutes.

# %%
import urllib.error
import urllib.request

CSV_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
data_dir = paths.user_scratch("lab2_data")
data_dir.mkdir(parents=True, exist_ok=True)
csv_path = data_dir / "GLB.Ts+dSST.csv"


def write_synthetic(target: Path) -> None:
    """Fallback data, clearly labelled. Every statistic from this is a property of
    `rng` and the formula below — nothing else."""
    months = np.arange(1880 * 12, 2026 * 12, dtype=int)
    base = -0.2 + 0.00013 * (months - months[0]) + 0.4 / (1 + np.exp(-(months - 1980 * 12) / 60))
    anomaly = base + rng.normal(0, 0.09, months.size)
    lines = ["Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec,J-D,D-N-D"]
    for y in range(1880, 2026):
        vals = anomaly[(y * 12):((y + 1) * 12)]
        lines.append(str(y) + "," + ",".join(f"{v:.2f}" for v in vals) + f",{vals.mean():.2f},")
    target.write_text("SYNTHETIC - NOT NASA DATA - source=synthetic\n" + "\n".join(lines) + "\n")


# %%
try:
    req = urllib.request.Request(CSV_URL, headers={"User-Agent": "RAF622M-lab2"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        csv_path.write_bytes(resp.read())
    SYNTHETIC = False
    print(f"downloaded {csv_path.stat().st_size:,} bytes from\n  {CSV_URL}")
except (urllib.error.URLError, TimeoutError, OSError) as exc:
    SYNTHETIC = True
    print(f"DOWNLOAD FAILED ({exc.__class__.__name__}: {exc})\n"
          "Writing a clearly-labelled SYNTHETIC substitute so the exercise can proceed.")
    write_synthetic(csv_path)

print(f"\nfile      : {csv_path}")
print(f"synthetic : {SYNTHETIC}")
print("first three lines of the file:")
print("\n".join(csv_path.read_text(errors="replace").splitlines()[:3]))
if SYNTHETIC:
    print(
        "\nWhat this implies, and you must write it in lab2_notes.md: every statistic "
        "you compute is a statistic of the random number generator and the formula in "
        "write_synthetic(). A trend you fit here is a restatement of the 0.00013 "
        "coefficient that generated it. That is exactly the mistake the 2025/26 lab "
        "made while presenting the result as satellite analysis."
    )
print(f"\nRESULTS_DIR for your artifacts: {paths.results_dir()}")

# %%
import json

RD = paths.results_dir()
problems = []
stats_path = RD / "lab2_stats.csv"
REQUIRED_COLS = {"year", "mean_anomaly", "n_months", "min_anomaly", "max_anomaly"}
REQUIRED_KEYS = {"source_url", "n_rows_read", "first_year", "last_year",
                 "slope_per_decade", "warmest_year", "coldest_year"}

if stats_path.exists():
    import csv as _csv

    with stats_path.open() as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        problems.append(f"{stats_path.name} has no data rows")
    elif not REQUIRED_COLS <= set(rows[0]):
        problems.append(f"{stats_path.name} missing columns: {sorted(REQUIRED_COLS - set(rows[0]))}")
    else:
        incomplete = [r["year"] for r in rows if r["n_months"] not in ("12", "12.0")]
        if len(incomplete) > 1:
            problems.append(f"{len(incomplete)} years do not have n_months=12: {incomplete[:5]}")
else:
    problems.append(f"missing {stats_path}")

# %%
for name in ("lab2_trend.png", "lab2_summary.json", "lab2_notes.md"):
    p = RD / name
    if not p.exists():
        problems.append(f"missing {p}")
    elif p.stat().st_size == 0:
        problems.append(f"{p} is empty")

summary = {}
if (RD / "lab2_summary.json").exists():
    summary = json.loads((RD / "lab2_summary.json").read_text())
    missing = REQUIRED_KEYS - set(summary)
    if missing:
        problems.append(f"lab2_summary.json missing keys: {sorted(missing)}")

print("EXERCISE CHECKER")
if problems:
    for p in problems:
        print(f"  [ ] {p}")
    print(f"\n{len(problems)} item(s) outstanding. This is expected before you have "
          "done the exercise; it is not a lab failure.")
else:
    print("  [x] all four artifacts present and well-formed")
    print(f"  slope per decade = {summary['slope_per_decade']}, "
          f"warmest = {summary['warmest_year']}, coldest = {summary['coldest_year']}")
EXERCISE_DONE = not problems

# %% [markdown]
# ## 14. Gate board
#
# `eo_course.gates` prints a pass/fail board. Lab 2's gates are environment gates:
# each one is something a student could previously claim without evidence.

# %%
lines = []


def check(name, ok, detail):
    """Print one gate line and keep it for the board and the deliverable.

    eo_course.gates.report owns the formatting; we mirror its line so
    print_gate_board can render the verdict.
    """
    gates.report(name, bool(ok), detail)
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


identity_ok = bool(cfg_name) and cfg_name.strip() not in {"Your Name", "your name", ""}
identity_ok = identity_ok and bool(cfg_email) and "@" in cfg_email and "your.email" not in cfg_email

check("kernel_is_course_venv", Path(sys.prefix).resolve() == VENV.resolve(),
      f"{sys.prefix} == {VENV}" if Path(sys.prefix).resolve() == VENV.resolve()
      else f"running {sys.prefix}, expected {VENV}")
check("repo_at_expected_path", (REPO / ".git").is_dir(), str(REPO))
check("git_identity_sane", identity_ok, f"{cfg_name!r} <{cfg_email!r}>")
check("commit_authored_by_you", identity_ok and author.endswith(f"<{cfg_email}>"),
      f"last commit author {author!r}")
check("mplconfigdir_before_import", "/tmp" not in matplotlib.get_config_dir(),
      matplotlib.get_config_dir())
check("artifacts_not_committable", ci2.returncode == 0, "tif/npz excluded from git status")
check("branch_demo_persisted", after_branch == BRANCH, f"branch {after_branch!r}")
check("conflict_exercised", "UU" in codes, f"status codes {codes}")
check("exercise_artifacts", EXERCISE_DONE, "four artifacts in results dir"
      if EXERCISE_DONE else f"{len(problems)} item(s) outstanding")

ok = gates.print_gate_board(lines)
n_pass = sum(1 for ln in lines if "[PASS]" in ln)
print(f"\n{n_pass}/{len(lines)} lab-2 gates passed.")

# %% [markdown]
# ## 15. Self-test — the gate for Lab 2
#
# Your deliverable is a **green test suite**. `tests/test_eo_course.py` covers the
# radiometry, label, split, metric, baseline and gate code that labs 3–7 depend on. If
# it is red, your environment is not the environment the rest of the course was written
# against, and continuing produces numbers you cannot trust.
#
# We run it as a subprocess with `cwd=REPO` and the venv's own interpreter — not
# `!pytest`, which would run whatever `pytest` is on `PATH` in a shell that never
# activated anything.

# %%
pytest = run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=REPO)
print(f"\npytest exit status: {pytest.returncode}")

import importlib

eo_ver = importlib.import_module("eo_course.paths").__file__
self_test = {
    "kernel_prefix": sys.prefix,
    "kernel_is_course_venv": Path(sys.prefix).resolve() == VENV.resolve(),
    "repo_path": str(REPO),
    "repo_head": git_out("rev-parse", "--short", "HEAD"),
    "git_name": cfg_name,
    "git_email": cfg_email,
    "eo_course_importable": Path(eo_ver).is_file(),
    "eo_course_file": eo_ver,
    "pytest_exit_status": pytest.returncode,
    "pytest_passed": pytest.returncode == 0,
}
for k, v in self_test.items():
    print(f"  {k:26s} {v}")

assert self_test["kernel_is_course_venv"], "not running in the course venv — section 8"
assert (REPO / ".git").is_dir(), "repository not cloned at the expected path — section 5"
assert identity_ok, f"git identity is a placeholder or unset: {cfg_name!r} <{cfg_email!r}>"
assert self_test["eo_course_importable"], "eo_course is not importable from this kernel"
assert self_test["pytest_passed"], (
    f"`pytest tests -q` exited {pytest.returncode}. This is the Lab 2 gate. Read the "
    "failures above; they name the module and the threshold. Do not proceed to Lab 3 "
    "with a red suite."
)
print("\nSELF-TEST PASSED — Lab 2 gate cleared.")

# %% [markdown]
# ## 16. Deliverable
#
# `results.json` is the course ledger. Lab 2 has no model, so there are no accuracy
# numbers; `record_run` requires four metric keys, so we put the honest thing there —
# the self-test pass rate — and say so in `notes` rather than inventing a score. Your
# predictions and their outcomes go in `config`, which is what gets graded.

# %%
pass_rate = n_pass / len(lines)
lab2_config = {
    "host": HOST,
    "on_login_node": paths.on_login_node(),
    "kernel_prefix": sys.prefix,
    "venv": str(VENV),
    "repo_path": str(REPO),
    "repo_head": self_test["repo_head"],
    "git_name": cfg_name,
    "git_email": cfg_email,
    "branch": BRANCH,
    "commit": after_head[:8],
    "mplconfigdir": matplotlib.get_config_dir(),
    "gee_extra_installed": gee.returncode == 0,
    "conflict_status_codes": codes,
    "data_synthetic": SYNTHETIC,
    "exercise_done": EXERCISE_DONE,
    "exercise_summary": summary or None,
    "predictions": PREDICTION,
}

# %%
record = results.record_run(
    run_id=f"lab2-{paths.username()}-{time.strftime('%Y%m%dT%H%M%S')}",
    lab="lab2",
    config=lab2_config,
    split_manifest_hash=None,
    seed=SEED,
    test_metrics={"n": len(lines), "overall_acc": pass_rate,
                  "balanced_acc": pass_rate, "macro_f1": pass_rate},
    notes=(
        f"Lab 2 environment + git check. No model: the four metric keys carry the gate "
        f"tally ({n_pass}/{len(lines)}), not accuracy. pytest exit "
        f"{self_test['pytest_exit_status']}. Graded content is in config."
    ),
)
out = results.default_results_path()
print(f"appended run_id={record['run_id']} to {out}")
assert out.exists(), f"record_run did not leave a file at {out}"
print(results.summary_table("lab2"))

# %% [markdown]
# ## 17. Submission checklist
#
# Concrete artifacts. Attach or paste each one.
#
# 1. **The executed notebook**, no red cells. Where a cell raised, keep the traceback
#    and write the diagnosis under it.
# 2. **The self-test cell passing**, showing `pytest_passed: True` and
#    `SELF-TEST PASSED`. Include the pytest summary line (`NN passed in X.XXs`).
# 3. **The kernel assertion passing**, showing `sys.prefix` equal to your project-space
#    `envs/ml_eo_course`.
# 4. **Your four prediction blocks** filled in, each with `predicted vs actual`, and
#    one sentence for every disagreement.
# 5. **The conflict exercise output**: the mid-conflict `git status --porcelain` line
#    and the `git log --graph` after resolution.
# 6. **The four exercise artifacts** in `paths.results_dir()`: `lab2_stats.csv`,
#    `lab2_trend.png`, `lab2_summary.json`, `lab2_notes.md` — and the checker cell
#    printing `all four artifacts present and well-formed`.
# 7. **`results.json`** containing your `lab2` record.
#
# ### What to bring to Lab 3
#
# Lab 3 acquires real Sentinel-2 imagery. It needs the kernel you just built, and it
# needs the habit from section 11: its outputs are gigabyte rasters that must go to
# `paths.data_root()` in scratch and must never reach a commit. If you took the Google
# Earth Engine path, you also need the `gee` extra from section 6 and your own
# Copernicus Data Space or GEE credentials in environment variables — read with
# `os.environ["NAME"]` so a missing one raises, never with a silent default, and never
# printed, not even truncated.
#
# ### Support
#
# * Course Slack first, with the exact error text and the cell that produced it.
# * JSC user support for account and filesystem issues: <https://judoor.fz-juelich.de>.
# * Jupyter-JSC documentation: <https://jupyter-jsc.fz-juelich.de/jupyter-jsc/docs/>.
