#!/bin/bash
#SBATCH --job-name=thesis_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=2000-2025
#SBATCH --output=logs/swissdox_%A_%a.out
#SBATCH --error=logs/swissdox_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO"
cd "${REPO_DIR}"

YEAR=${SLURM_ARRAY_TASK_ID}
OUTDIR="data/input/year_${YEAR}"

mkdir -p logs "${OUTDIR}"

module purge
module load python/3.12.1

source .venv/bin/activate

export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

echo "=== Downloading year ${YEAR} ==="

python scripts/download.py \
  --start "${YEAR}-01-01" \
  --end   "${YEAR}-12-31" \
  --max-results 50000 \
  --outdir "${OUTDIR}"

echo "Job finished for year ${YEAR}."
