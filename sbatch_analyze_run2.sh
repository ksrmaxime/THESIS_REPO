#!/bin/bash
#SBATCH --job-name=analyze_run2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analyze_run2_%j.out
#SBATCH --error=logs/analyze_run2_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch sbatch_analyze_run2.sh <STD_JOB_ID>
#
# Chained après standardize :
#   STD_ID=$(sbatch --parsable --dependency=afterok:${MERGE_ID} sbatch_standardize_run2.sh ${MERGE_ID}) && \
#   sbatch --dependency=afterok:${STD_ID} sbatch_analyze_run2.sh ${STD_ID}

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

STD_JOB_ID=${1:-""}
if [ -z "$STD_JOB_ID" ]; then
    echo "[ERROR] Passer le STD_JOB_ID en argument: sbatch sbatch_analyze_run2.sh <STD_JOB_ID>"
    exit 1
fi

OUTDIR="${WORKDIR}/data/output"
INPUT="${OUTDIR}/run2_standardized_job${STD_JOB_ID}/results.parquet"
OUTPUT_DIR="${OUTDIR}/run2_analysis_job${SLURM_JOB_ID}"

echo "=== ANALYZE run2 (std job ${STD_JOB_ID}) → job ${SLURM_JOB_ID} ==="
echo "DATE=$(date -Is)"
echo "INPUT=${INPUT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

if [ ! -f "$INPUT" ]; then
    FLAT="${OUTDIR}/run2_standardized_job${STD_JOB_ID}.parquet"
    if [ -f "$FLAT" ]; then
        INPUT="$FLAT"
        echo "[warn] archive path not found, using flat parquet: ${INPUT}"
    else
        echo "[ERROR] Aucun fichier standardisé trouvé pour le job ${STD_JOB_ID}."
        echo "  Cherché : ${INPUT}"
        echo "  Cherché : ${FLAT}"
        exit 1
    fi
fi

echo "INPUT final : ${INPUT}"

python3 scripts/analyze_run2.py \
    --input      "$INPUT" \
    --output_dir "$OUTPUT_DIR"

echo "=== DONE ==="
echo "Output folder: ${OUTPUT_DIR}"
echo "  figures/   — 12 PNG plots"
echo "  tables/    — CSV tables"
echo "  report.txt — text summary"
