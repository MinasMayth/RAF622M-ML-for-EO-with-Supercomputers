#!/bin/bash
# Fan an ablation grid out over GPUs with a SLURM job array.
#
# Why an array and not a loop
# ---------------------------
# The labs ask for ablations x seeds. Running those serially on one GPU is how
# students end up reporting one run per arm and then claiming differences that are
# seed noise. One array task per (arm, seed) cell means:
#   * every cell gets a full GPU and its own wall clock,
#   * one failing cell cannot kill the others,
#   * the whole grid lands in one results.json, so the seed-variance gate in
#     eo_course.gates.gate_seeds_and_variance can actually see >= 3 seeds.
#
# This is the shape the course's own Lab 6 needed and never had: it ran one
# 2-epoch fit interactively and reported a model that scored exactly 1/10.
#
# Usage
# -----
#   sbatch slurm/submit_sweep.sbatch
#   sbatch --export=ALL,GRID="none sampler focal" slurm/submit_sweep.sbatch
#   sbatch --export=ALL,SEEDS="0 1 2 3" slurm/submit_sweep.sbatch
#
# Then wait for the array and read the table:
#   squeue -u $USER
#   python scripts/summarize_results.py --lab lab5

#SBATCH --job-name=eo_sweep
#SBATCH --account=training2653
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=sweep_submit_%j.out

set -euo pipefail

# The grid. Each arm name must exist in scripts/train_cnn.py's ARMS dict.
GRID="${GRID:-none weights_sqrt weights_inv sampler focal}"
SEEDS="${SEEDS:-0 1 2}"
MAX_ARRAY="${MAX_ARRAY:-60}"

CELLS=()
for arm in $GRID; do
    for seed in $SEEDS; do
        CELLS+=("${arm}:${seed}")
    done
done

N=${#CELLS[@]}
if (( N > MAX_ARRAY )); then
    echo "ERROR: grid is $N cells but MAX_ARRAY=$MAX_ARRAY. Split the sweep or raise MAX_ARRAY." >&2
    exit 1
fi

echo "Submitting $N cells: arms=[$GRID] seeds=[$SEEDS]"
printf '  %s\n' "${CELLS[@]}"

# The worker maps its array index back to (arm, seed) via ARRAY_CELLS, so it needs
# no knowledge of the grid layout. --export=NONE clears the inherited environment,
# so the account (and any Python module override) must be passed explicitly or the
# worker silently falls back to the default project.
export ARRAY_CELLS="$(IFS=,; echo "${CELLS[*]}")"
EO_COURSE_ACCOUNT="${EO_COURSE_ACCOUNT:-training2653}"

sbatch --array=1-"${N}"%4 \
       --export=NONE,ARRAY_CELLS="${ARRAY_CELLS}",EO_COURSE_ACCOUNT="${EO_COURSE_ACCOUNT}" \
       --job-name=eo_cell \
       --account="${EO_COURSE_ACCOUNT}" \
       --time="${CELL_TIME:-02:00:00}" \
       slurm/train_jureca.sbatch

echo
echo "Done submitting. Monitor with: squeue -u \$USER"
echo "Read results with:             python scripts/summarize_results.py --lab lab5"
