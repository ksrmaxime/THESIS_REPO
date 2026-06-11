#!/bin/bash -l
#SBATCH --job-name=std_run5
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/std_run5_%j.out
#SBATCH --error=logs/std_run5_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch sbatch_standardize_run5.sh [<merged_parquet_path>]

dcsrsoft use 20241118

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

INPUT=${1:-"${WORKDIR}/data/output/run5_merged_61406751.parquet"}
OUTPUT_DIR="${WORKDIR}/data/output/run5_standardized_job${SLURM_JOB_ID}"

if [ ! -f "$INPUT" ]; then
    echo "[ERROR] Fichier introuvable : ${INPUT}"
    exit 1
fi

echo "=== STANDARDIZE run5 → job ${SLURM_JOB_ID} ==="
echo "DATE=$(date -Is)"
echo "INPUT=${INPUT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

python3 scripts/standardize_run5.py \
    --input      "$INPUT" \
    --output_dir "$OUTPUT_DIR"

echo "=== DONE ==="
echo "Output folder: ${OUTPUT_DIR}"
echo "  results.parquet"
echo "  results.csv"

# ── Auto-chain ─────────────────────────────────────────────────────────────────
STD_PARQUET="${OUTPUT_DIR}/results.parquet"
sbatch "${WORKDIR}/sbatch_run5eval_array.sh" "${STD_PARQUET}"
echo "[chain] → sbatch_run5eval_array.sh submitted (input: ${STD_PARQUET})"
