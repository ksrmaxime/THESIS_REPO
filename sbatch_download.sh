#!/bin/bash
#SBATCH --job-name=thesis_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/swissdox_%j.out
#SBATCH --error=logs/swissdox_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -euo pipefail

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO"
cd "${REPO_DIR}"

mkdir -p logs data/input

module purge
module load python/3.12.1

source .venv/bin/activate

python --version
which python

export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

python scripts/download.py \
  --start 2000-01-01 \
  --end   2025-12-31 \
  --max-results 500000 \
  --outdir data/input

echo "Job finished."
