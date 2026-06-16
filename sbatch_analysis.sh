#!/bin/bash -l
#SBATCH --job-name=analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

# Usage:
#   sbatch sbatch_analysis.sh [<run6_merged_results.parquet>] [<granularity>] [<crisis_k>] [<transition_window_months>]
#
# If the input path is omitted, falls back to the run6 archive pointer left
# by sbatch_merge_run6.sh (data/output/.last_run6_archive).
#
# Final, purely-CPU stage of the pipeline: descriptive stats, temporal
# evolution, source/target/content cross-tabs, partisan alignment (E3),
# minister party-change event study, and crisis/peak detection (E4).
# No GPU, no LLM calls.

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

INPUT=${1:-""}
GRANULARITY=${2:-"M"}
CRISIS_K=${3:-"2.0"}
TRANSITION_WINDOW_MONTHS=${4:-"6"}

if [[ -z "$INPUT" ]]; then
    POINTER="${WORKDIR}/data/output/.last_run6_archive"
    if [[ -f "$POINTER" ]]; then
        INPUT="$(cat "$POINTER")/results.parquet"
        echo "[INFO] Pas d'input fourni — utilisation du dernier run6 archivé: ${INPUT}"
    else
        echo "[ERROR] Pas d'input fourni et aucun pointeur ${POINTER} trouvé." >&2
        echo "[ERROR] Usage: sbatch sbatch_analysis.sh <run6_merged_results.parquet> [granularity] [crisis_k] [transition_window_months]" >&2
        exit 1
    fi
fi

if [[ ! -f "$INPUT" ]]; then
    echo "[ERROR] Fichier introuvable : ${INPUT}" >&2
    exit 1
fi

JOB_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${WORKDIR}/data/output/analysis_${JOB_ID}"

echo "=== ANALYSIS (job ${JOB_ID}) ==="
echo "DATE=$(date -Is)"
echo "INPUT=${INPUT}"
echo "OUTPUT_DIR=${OUT_DIR}"
echo "GRANULARITY=${GRANULARITY} | CRISIS_K=${CRISIS_K} | TRANSITION_WINDOW_MONTHS=${TRANSITION_WINDOW_MONTHS}"

python3 scripts/run_analysis.py \
    --input        "$INPUT" \
    --output_dir   "$OUT_DIR" \
    --granularity  "$GRANULARITY" \
    --crisis_k     "$CRISIS_K" \
    --transition_window_months "$TRANSITION_WINDOW_MONTHS"

cp "$0"               "${OUT_DIR}/sbatch_used.sh"            || true
cp "scripts/run_analysis.py" "${OUT_DIR}/run_analysis_used.py" || true
cp "src/analysis_config.py"  "${OUT_DIR}/analysis_config_used.py" || true

echo "=== DONE ==="
echo "Output folder: ${OUT_DIR}"
echo "  analysis_report.md"
echo "  descriptive/ temporal/ crosstabs/ partisan_alignment/ minister_transitions/ crisis/"

echo "${OUT_DIR}" > "${WORKDIR}/data/output/.last_analysis_archive"

# =============================================================================
# COPY TO NAS
# =============================================================================

NAS_BASE="/nas/FAC/FDCA/IDHEAP/mhinterl/parp/D2c/maxime/THESIS/output"
NAS_RUN_DIR="${NAS_BASE}/analysis_${JOB_ID}"

echo "=== COPY TO NAS ==="
echo "Destination : ${NAS_RUN_DIR}"

if mkdir -p "$NAS_RUN_DIR" 2>/dev/null; then
    cp -r "${OUT_DIR}/." "${NAS_RUN_DIR}/" && echo "[NAS] → ${NAS_RUN_DIR}" || true
    echo "=== NAS COPY DONE : ${NAS_RUN_DIR} ==="
else
    echo "[WARN] Impossible de créer ${NAS_RUN_DIR} — NAS non monté ?" >&2
fi

echo "=== PIPELINE COMPLET (analysis) ==="
