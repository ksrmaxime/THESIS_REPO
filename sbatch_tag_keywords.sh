#!/bin/bash -l
#SBATCH --job-name=thesis_tag_keywords
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tag_keywords_%j.out
#SBATCH --error=logs/tag_keywords_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

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

# Usage:
#   sbatch sbatch_tag_keywords.sh [<input_parquet_or_csv>]
#
# If the input path is omitted, falls back to the download pointer left by
# sbatch_download.sh (data/input/.last_download). Output is written next to
# the input with a "_tagged" suffix; --outdir redirects it if needed.
INPUT=${1:-""}
if [[ -z "$INPUT" ]]; then
    DOWNLOAD_POINTER="${REPO_DIR}/data/input/.last_download"
    if [[ -f "$DOWNLOAD_POINTER" ]]; then
        INPUT="$(cat "$DOWNLOAD_POINTER")"
        echo "[INFO] Pas d'input fourni — utilisation du dernier download: ${INPUT}"
    else
        echo "[ERROR] Pas d'input fourni et aucun pointeur ${DOWNLOAD_POINTER} trouvé." >&2
        echo "[ERROR] Usage: sbatch sbatch_tag_keywords.sh <input_parquet_or_csv>" >&2
        exit 1
    fi
fi

if [[ ! -f "$INPUT" ]]; then
    echo "[ERROR] Fichier introuvable : ${INPUT}" >&2
    exit 1
fi

python scripts/tag_keywords.py \
  --infile "$INPUT" \
  --outdir data/input

echo "Job finished."

# ── Auto-chain ─────────────────────────────────────────────────────────────────
TAGGED_POINTER="${REPO_DIR}/data/input/.last_tagged"
if [[ ! -f "$TAGGED_POINTER" ]]; then
    echo "[ERROR] Pointeur de tagging introuvable : ${TAGGED_POINTER}" >&2
    exit 1
fi
TAGGED_FILE="$(cat "$TAGGED_POINTER")"
sbatch "${REPO_DIR}/sbatch_run3_array.sh" "${TAGGED_FILE}"
echo "[chain] → sbatch_run3_array.sh submitted (input: ${TAGGED_FILE})"
