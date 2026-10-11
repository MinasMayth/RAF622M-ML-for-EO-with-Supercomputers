#!/bin/bash
# Course environment bootstrap -- run this in a JupyterLab terminal, once.
#
#     cd ~/repos/RAF622M-ML-for-EO-with-Supercomputers
#     bash scripts/bootstrap_course_env.sh
#
# It does the three things that need a real login shell and a login node, and that
# therefore cannot happen inside a notebook cell:
#
#   1. install `uv` for your user (no root, nothing outside $HOME)
#   2. install a self-contained CPython under your scratch directory
#   3. build the course virtualenv from pyproject.toml + uv.lock
#
# Registering the Jupyter kernel is deliberately NOT here -- that is the next step,
# and it is a notebook cell so you can see what it writes and why.
#
# Why a login node and a terminal:
#
#   * JURECA compute nodes have no route to PyPI. `uv sync` there dies with
#     "tcp connect error: No route to host (os error 113)". A Jupyter-JSC session
#     runs on a login node, so this works from that terminal.
#   * `module` is a shell function, not a program on PATH, so anything that needs it
#     must run in a login shell. A notebook `!` line is neither a login shell nor the
#     same shell twice in a row.
#
# Why a uv-managed interpreter instead of `module load Python`:
#
#   A virtualenv created on top of a module-backed CPython keeps a dynamic dependency
#   on libpython3.13.so.1.0 inside the module tree. It runs in the shell that loaded
#   the module and fails with "cannot open shared object file" anywhere else -- which
#   includes a Jupyter kernel, and includes next term when the site is re-staged.
#   That is exactly how the 2025/26 course environment died. The interpreter installed
#   below is self-contained and lives in your scratch, so the venv outlives any module
#   churn. Step 4 below proves it rather than asserting it.
set -uo pipefail

ACCT="${EO_COURSE_ACCOUNT:-training2653}"
ACCT_ENV="${ACCT//-/_}"
REPO_NAME="RAF622M-ML-for-EO-with-Supercomputers"

# printenv, not ${PROJECT_FOO:-}: bash 5.1 rejects nested expansion in the name
# position, and the account name is a variable here.
PROJ="$(printenv "PROJECT_${ACCT_ENV}")"; : "${PROJ:=/p/project1/${ACCT}}"
SCR="$(printenv "SCRATCH_${ACCT_ENV}")"; : "${SCR:=/p/scratch/${ACCT}}"

VENV="${UV_PROJECT_ENVIRONMENT:-${PROJ}/${USER}/envs/ml_eo_course}"

fail() { echo "!! $*"; echo "BOOTSTRAP_FAILED"; exit 1; }

echo "host     : $(hostname)"
echo "user     : ${USER}"
echo "account  : ${ACCT}"
echo "venv     : ${VENV}"

# ---------------------------------------------------------------- preflight
REPO="$(pwd -P)"
[ -f pyproject.toml ] && [ -f uv.lock ] || fail \
  "run this from the repository root. No pyproject.toml/uv.lock in ${REPO}.
   cd into your clone first:  cd ~/repos/${REPO_NAME}"
case "$(basename "${REPO}")" in
  "${REPO_NAME}") : ;;
  *) echo "NOTE: this does not look like the course repo ($(basename "${REPO}"))" ;;
esac

if [ -n "${SLURM_JOB_ID:-}" ] && [ ! -d /p/project1 ]; then
  fail "you appear to be on a compute node, which has no route to PyPI.
   Run this in the terminal of a Jupyter-JSC session, or over ssh on a login node."
fi

# ---------------------------------------------------- 1. uv, for your user
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo
  echo "=== 1/4 installing uv for your user ==="
  echo "  (the JURECA \`uv\` module cannot be loaded -- it pulls in \`make\`, which
   conflicts with the stage. The installer writes only into ~/.local, no root.)"
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv installer failed"
fi
command -v uv >/dev/null 2>&1 || fail "uv is still not on PATH after installing.
   Open a new terminal tab, or run:  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "uv       : $(uv --version 2>&1) at $(command -v uv)"

# ------------------------------------------- 2. a self-contained interpreter
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${SCR}/${USER}/.uv-python}"
export UV_PYTHON="${UV_PYTHON:-3.12}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCR}/${USER}/.cache/uv}"
mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" || fail "cannot write to ${SCR}"

echo
echo "=== 2/4 self-contained interpreter ==="
echo "  UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR}"
uv python install "${UV_PYTHON}" 2>&1 | tail -3 || fail "no usable Python ${UV_PYTHON}"
echo "  resolved: $(uv python find "${UV_PYTHON}" 2>&1)"

# --------------------------------------------------- 3. the course packages
export UV_PROJECT_ENVIRONMENT="${VENV}"

echo
echo "=== 3/4 uv sync ==="
# A lock out of step with pyproject.toml makes `uv sync --frozen` abort with
# "Could not find root package", and with a plain `set -e` removed script people
# still saw a success message. Detect it here, not three minutes later.
if ! uv lock --check >/dev/null 2>&1; then
  echo "  lock is out of date with pyproject.toml; re-resolving"
  uv lock || fail "uv lock failed"
fi
uv sync --frozen || fail "uv sync --frozen failed. Read the lines above: on JURECA the
   two causes are a full project quota and running this on a compute node."
[ -x "${VENV}/bin/python" ] || fail "no interpreter in ${VENV}"

# ------------------------------------------------- 4. prove it stands alone
echo
echo "=== 4/4 self-containment ==="
# A kernel does not inherit your login shell's LD_LIBRARY_PATH, so neither may the
# environment. This is the check whose absence cost the 2025/26 cohort a term.
env -u LD_LIBRARY_PATH "${VENV}/bin/python" -c \
  'import sys; print("  ok:", sys.version.split()[0], sys.prefix)' \
  || fail "${VENV} only works with module libraries on LD_LIBRARY_PATH.
   It will fail in Jupyter. Remove it and re-run:  rm -rf ${VENV}"

# Import from outside the repo, because a kernel starts in the notebook's own
# directory, not the repository root.
( cd /tmp && env -u LD_LIBRARY_PATH -u PYTHONPATH "${VENV}/bin/python" \
    -c 'import eo_course.paths as p; print("  eo_course ok, account", p.PROJECT_ACCOUNT)' ) \
  || fail "eo_course is not importable from outside the repo. A kernel starts in the
   notebook's directory, so this must work. Usually it means the project was not
   installed: check that pyproject.toml has a [build-system] section."

echo
echo "BOOTSTRAP_OK"
echo
echo "Next: back in JupyterLab, open notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb"
echo "and run the 'Register the kernel' cell. Then switch to the ML-EO Course kernel"
echo "with Kernel -> Change Kernel -> ML-EO Course."
