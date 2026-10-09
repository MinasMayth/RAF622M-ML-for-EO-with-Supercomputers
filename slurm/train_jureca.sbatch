#!/bin/bash
# Lab 5/6 training job for JURECA.
#
# Read this before submitting
# --------------------------
# * Account defaults to `training2653`. `#SBATCH` lines are parsed before the shell
#   runs, so the account cannot be a shell variable here: to run the labs under a
#   different project, override it at submit time and tell the code about it, e.g.
#       EO_COURSE_ACCOUNT=training2600 sbatch --account=training2600 slurm/train_jureca.sbatch
#   Everything below (env dir, scratch cache) is derived from EO_COURSE_ACCOUNT, so
#   the two lines above are the only place an account name appears.
# * JURECA-DC partitions are `dc-cpu` and `dc-gpu`
#   (lower case). The 2025/26 lab 1 taught `JURECA-DC_CPU` / `JURECA-DC_GPU`,
#   which are not partition names, and its example sbatch had no `--gres` line --
#   so a student who copied it asked for a GPU job and got one that could not see
#   a GPU. Confirm live names with `sinfo` before your first submission.
# * ONE ARM PER JOB. Do not loop over ablation arms inside one script: a job that
#   dies at arm 2 loses arm 1's wall clock and you learn nothing. Use
#   submit_sweep.sbatch, which fans out with `sbatch --array`.
# * Jupyter-JSC quietly caps you at one GPU even when it allocated four, printing
#   "Trainer will use only 1 of 4 GPUs because it is running inside an interactive
#   / notebook environment". That is what the 2025/26 Lab 6 output showed: a paid
#   4-GPU allocation running one process on one GPU, with logger=False so there
#   were no curves to inspect. Train from sbatch instead.
#
# Submit:     sbatch slurm/train_jureca.sbatch
# Named arm:  sbatch --export=ALL,ARM_TAG=weights_sqrt slurm/train_jureca.sbatch

#SBATCH --job-name=eo_train
#SBATCH --account=training2653
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---------------------------------------------------------------- environment
# `module` is a shell function, so it cannot be called from a notebook `!` cell.
# That is why the 2025/26 notebooks used `!source ...` (which fails under /bin/sh)
# and put virtualenvs in $HOME. In a real sbatch script it works normally.
#
# EO_COURSE_ACCOUNT is the single source of truth for the project id, and must
# match eo_course.paths.PROJECT_ACCOUNT. `module purge` would drop it if it were
# only ever set by a module, so re-export it explicitly.
export EO_COURSE_ACCOUNT="${EO_COURSE_ACCOUNT:-training2653}"
EO_ACCT_ENV="${EO_COURSE_ACCOUNT//-/_}"

# The course virtualenv is built by lab 2 on a uv-managed interpreter, so it is
# self-contained and needs no modules. Loading Python here is therefore optional --
# and it must stay optional: `set -euo pipefail` above would abort an otherwise
# healthy training job just because the site re-staged a module version.
# (Python/3.12.3, named in the 2025/26 recipe, does not exist on JURECA at all, and
# Python/3.13.5 is not loadable on its own -- Lmod refuses it unless its toolchain is
# present: `module load Stages/2026 GCCcore/14.3.0 Python/3.13.5`.)
# Override the list with EO_COURSE_MODULES if you genuinely need a module.
module purge
# shellcheck disable=SC2086
if ! module load ${EO_COURSE_MODULES:-Stages/2026 GCCcore/14.3.0 Python/3.13.5} 2>/dev/null; then
    echo "NOTE: Python modules unavailable; continuing on the self-contained venv." >&2
fi

# One virtualenv for the whole course, in PROJECT space, created by lab 2.
# JSC exports PROJECT_<account> / SCRATCH_<account> (dashes -> underscores).
# Bash cannot expand a variable name built at runtime inside ${...:-...}, so read
# them with printenv rather than eval.
_jsc_project() { printenv "PROJECT_${EO_ACCT_ENV}" 2>/dev/null || true; }
_jsc_scratch() { printenv "SCRATCH_${EO_ACCT_ENV}" 2>/dev/null || true; }

PROJECT_ROOT="$(_jsc_project)"
: "${PROJECT_ROOT:=/p/project1/${EO_COURSE_ACCOUNT}}"
SCRATCH_ROOT="$(_jsc_scratch)"
: "${SCRATCH_ROOT:=/p/scratch/${EO_COURSE_ACCOUNT}}"

ENV_DIR="${PROJECT_ROOT}/${USER}/envs/ml_eo_course"
if [[ ! -d "$ENV_DIR" ]]; then
    echo "ERROR: no virtualenv at $ENV_DIR" >&2
    echo "       Run lab 2 (environment setup) before submitting training jobs." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# Matplotlib on a shared node needs a writable cache dir BEFORE it is imported.
# The 2025/26 notebooks set this AFTER importing pyplot, which does nothing.
export MPLCONFIGDIR="${SCRATCH_ROOT}/${USER}/.cache/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# CUDA_LAUNCH_BLOCKING=1 serialises every kernel and slows training 2-5x. The
# 2025/26 notebooks set it in cell 2 and never unset it. Make sure it is off.
unset CUDA_LAUNCH_BLOCKING || true

# ------------------------------------------------------------------- identity
# ARM_TAG names the run so parallel array jobs do not collide.
# When submitted via submit_sweep.sbatch, ARRAY_CELLS holds a comma-separated
# list of "arm:seed" cells and this task's index selects one of them. Decoding it
# here keeps the worker ignorant of the grid layout.
if [[ -n "${ARRAY_CELLS:-}" ]]; then
    idx=$(( ${SLURM_ARRAY_TASK_ID:-1} - 1 ))
    IFS=',' read -r -a CELLS <<< "$ARRAY_CELLS"
    if (( idx < 0 || idx >= ${#CELLS[@]} )); then
        echo "ERROR: array index $idx out of range for ${#CELLS[@]} cells" >&2
        exit 1
    fi
    ARM_TAG="${CELLS[$idx]%%:*}"
    SEED="${CELLS[$idx]##*:}"
    echo "array cell $idx -> arm=$ARM_TAG seed=$SEED"
fi

ARM_TAG="${ARM_TAG:-default}"
RUN_ID="${ARM_TAG}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

echo "=== job start =========================================================="
echo "job_id     : ${SLURM_JOB_ID:-none}  array: ${SLURM_ARRAY_TASK_ID:-none}"
echo "host       : $(hostname)"
echo "user       : $USER"
echo "arm        : $ARM_TAG   run_id: $RUN_ID   seed: ${SEED:-0}"
echo "partition  : ${SLURM_PARTITION:-none}"
echo "gres       : ${SLURM_STEP_GPUS:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  (no GPU visible -- check --gres)"
echo "python     : $(command -v python) $(python -V 2>&1)"
echo

# ------------------------------------------------------------------ preflight
# Fail in 5 seconds rather than after 40 minutes of loading data.
python - <<'PY'
import sys

missing = []
for mod, imp in (("torch", "torch"), ("lightning", "lightning.pytorch"),
                 ("scikit-learn", "sklearn"), ("numpy", "numpy"), ("rasterio", "rasterio")):
    try:
        __import__(imp)
    except Exception as exc:  # noqa: BLE001
        missing.append(f"{mod}: {exc}")
if missing:
    print("ERROR missing packages:\n  " + "\n  ".join(missing), file=sys.stderr)
    sys.exit(1)

import torch

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("ERROR: no GPU visible. Use --partition=dc-gpu with --gres=gpu:1.", file=sys.stderr)
    sys.exit(1)
PY

# --------------------------------------------------------------------- train
# Heavy lifting lives in scripts/train_cnn.py so the same code runs in a notebook
# (few epochs) and here (full epochs) with no copy-paste drift.
rc=0
python scripts/train_cnn.py \
    --arm "$ARM_TAG" \
    --run-id "$RUN_ID" \
    --epochs "${EPOCHS:-50}" \
    --seed "${SEED:-0}" \
    --accelerator gpu \
    --devices 1 || rc=$?

echo "=== training exit code $rc ============================================="

# ------------------------------------------------------------------- evidence
# Record resource usage alongside results, so "it was slow" becomes a number.
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,ReqTRES,ExitCode 2>/dev/null || true

exit $rc
