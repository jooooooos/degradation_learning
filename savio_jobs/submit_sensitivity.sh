#!/bin/bash
# =============================================================================
# Sensitivity-analysis SLURM array job for the departure-threshold experiment.
#
# Each array task computes one ROW of one parameter pair's heatmap.
# 6 parameter pairs * N_GRID rows = 6 * N_GRID array tasks total.
#
# USAGE
#   1. Edit N_GRID below (number of grid points per axis). The --array
#      directive must be 0-(6*N_GRID - 1). Defaults: N_GRID=30, array=0-179.
#   2. Submit from repo root:
#        sbatch savio_jobs/submit_sensitivity.sh
#      (or override at submit time:
#        N_GRID=50 sbatch --array=0-299 savio_jobs/submit_sensitivity.sh )
#   3. After all tasks finish, merge per-row pickles into per-pair matrices
#      that are drop-in replacements for data/sensitivity_*.pkl:
#        python savio_jobs/merge_chunks.py \
#            --in-dir  /global/scratch/users/jooseunglee/raas_sensitivity_<DATE>/chunks \
#            --out-dir data
#      (Back up the existing data/sensitivity_*.pkl first if you want.)
# =============================================================================
#SBATCH --job-name=raas_sens
#SBATCH --account=fc_calfit
#SBATCH --partition=savio4_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-179
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jooseung_lee@berkeley.edu
#SBATCH --output=slurm_out/sens_%A_%a.out
#SBATCH --error=slurm_out/sens_%A_%a.err

set -euo pipefail

# --- Configuration --------------------------------------------------------- #
# Number of grid points per axis. Original sensitivity_analysis.ipynb used 15;
# 30 quadruples cell count per pair (4x finer heatmap). To change, also update
# the --array=0-(6*N_GRID - 1) directive above.
N_GRID="${N_GRID:-30}"

# VI hyperparameters (match original notebook defaults)
VI_ITERS="${VI_ITERS:-200}"
VI_SAMPLES="${VI_SAMPLES:-300000}"

# Output goes to GPFS scratch (faster, large quota) under a date-stamped run.
EXPDATE="$(date +%Y%m%d)"
SCRATCH_BASE="${SCRATCH_BASE:-/global/scratch/users/${USER}/raas_sensitivity_${EXPDATE}}"
OUT_DIR="${SCRATCH_BASE}/chunks"

# Repo root (this script lives in <repo>/savio_jobs/).
REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"

# --- Sanity check task-id range ------------------------------------------- #
EXPECTED_LAST=$((6 * N_GRID - 1))
if [[ "${SLURM_ARRAY_TASK_ID}" -gt "${EXPECTED_LAST}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} > expected last id ${EXPECTED_LAST}" >&2
    echo "       Set --array=0-${EXPECTED_LAST} for N_GRID=${N_GRID}." >&2
    exit 1
fi

# --- Environment ---------------------------------------------------------- #
mkdir -p slurm_out

module purge
module load anaconda3
source activate res

# Pin numba/OpenMP to the cores actually allocated to this task; matters because
# raas.optimized_discrete_policy uses parallel=True kernels that would otherwise
# spawn N_PHYS threads regardless of cgroup limits.
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_ROOT}"

echo "=================================================================="
echo "Job:        ${SLURM_JOB_ID}  array task ${SLURM_ARRAY_TASK_ID}"
echo "Host:       $(hostname)"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "Repo root:  ${REPO_ROOT}"
echo "Out dir:    ${OUT_DIR}"
echo "N_GRID=${N_GRID}  VI_ITERS=${VI_ITERS}  VI_SAMPLES=${VI_SAMPLES}"
echo "=================================================================="

python savio_jobs/run_sensitivity.py \
    --n-grid     "${N_GRID}" \
    --vi-iters   "${VI_ITERS}" \
    --vi-samples "${VI_SAMPLES}" \
    --out-dir    "${OUT_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} complete."
