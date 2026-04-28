#!/bin/bash
#SBATCH --job-name=std_run2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/std_run2_%j.out
#SBATCH --error=logs/std_run2_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage (run after merge):
#   sbatch sbatch_standardize_run2.sh <ARRAY_JOB_ID>
#
# Or chained automatically after the merge job:
#   MERGE_ID=$(sbatch --parsable --dependency=afterok:${ARRAY_ID} sbatch_merge_run2.sh ${ARRAY_ID}) && \
#   sbatch --dependency=afterok:${MERGE_ID} sbatch_standardize_run2.sh ${ARRAY_ID}

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

ARRAY_JOB_ID=${1:-""}
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERROR] Passer le ARRAY_JOB_ID en argument: sbatch sbatch_standardize_run2.sh <ARRAY_JOB_ID>"
    exit 1
fi

OUTDIR="${WORKDIR}/data/output"
INPUT="${OUTDIR}/run2_merged_job${ARRAY_JOB_ID}/results.parquet"
OUTPUT="${OUTDIR}/run2_standardized_job${SLURM_JOB_ID}.parquet"

echo "=== STANDARDIZE run2 (array job ${ARRAY_JOB_ID}) ==="
echo "DATE=$(date -Is)"
echo "INPUT=${INPUT}"
echo "OUTPUT=${OUTPUT}"

if [ ! -f "$INPUT" ]; then
    FLAT="${OUTDIR}/run2_merged_job${ARRAY_JOB_ID}.parquet"
    if [ -f "$FLAT" ]; then
        INPUT="$FLAT"
        echo "[warn] archive path not found, using flat parquet: ${INPUT}"
    else
        echo "[ERROR] Aucun fichier d'entrée trouvé."
        echo "  Cherché : ${INPUT}"
        echo "  Cherché : ${FLAT}"
        echo "  Contenu de ${OUTDIR} :"
        ls -lh "${OUTDIR}" 2>/dev/null || echo "  (répertoire introuvable)"
        exit 1
    fi
fi

echo "INPUT final : ${INPUT}"

python3 scripts/standardize_run2.py \
    --input  "$INPUT" \
    --output "$OUTPUT"

echo "=== DONE ==="
echo "Output: ${OUTPUT}"
