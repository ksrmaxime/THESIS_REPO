#!/bin/bash -l
#SBATCH --job-name=run5_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/run5_array_%A_%a.out
#SBATCH --error=logs/run5_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-8   # 9 tâches (0-8) — ajuster selon le volume

dcsrsoft use 20241118

# Trace immédiate avant tout — si cette ligne n'apparaît pas dans les logs,
# c'est que SLURM tue le job avant même que bash ne démarre.
echo "=== BASH STARTED — job=${SLURM_JOB_ID:-?} task=${SLURM_ARRAY_TASK_ID:-?} host=$(hostname) ===" 2>&1

set -euo pipefail

# =============================================================================
# USER VARIABLES — edit here, nowhere else
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO

# I/O
# Input = fichier merged produit par sbatch_merge_run4.sh  ← à adapter
INPUT=${1:-"${WORKDIR}/data/output/run4_merged_61312765.parquet"}
OUTPUT_BASE=${WORKDIR}/data/output/run5
TEXT_COL=${2:-"critic_answer"}   # critic_answer_final quand appelé depuis merge_run4arbitre
N_ROWS=0          # 0 = toutes les lignes ; mettre ex. 100 pour un test rapide

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference
# Les entrées sont courtes (critic_answer = quelques mots/phrases) → batch plus grand
BATCH_SIZE=8
MAX_NEW_TOKENS=50      # SOURCE: Category — Name/Details  (~20-30 tokens suffisent)
MAX_INPUT_TOKENS=2048  # system prompt (~1200 tokens) + user prompt (~100 tokens)
TEMPERATURE=0.0

NUM_TASKS=9   # doit correspondre au nombre de tâches dans --array (0-8 = 9 tâches)

# =============================================================================

module purge || true
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs data/output

# ── Auto-chain : task 0 soumet le merge dès le début (SLURM attend que TOUTES les tâches réussissent) ──
if [[ "${SLURM_ARRAY_TASK_ID:-}" == "0" ]]; then
    sbatch --dependency=afterok:${SLURM_ARRAY_JOB_ID} \
        "${WORKDIR}/sbatch_merge_run5.sh" "${SLURM_ARRAY_JOB_ID}"
    echo "[chain] Submitted sbatch_merge_run5.sh (dependency: afterok:${SLURM_ARRAY_JOB_ID})"
fi

echo "=== SLURM ARRAY ==="
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>} TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

echo "=== RUN CONFIG ==="
echo "INPUT=${INPUT} | OUTPUT_BASE=${OUTPUT_BASE}"
echo "TEXT_COL=${TEXT_COL} | NUM_TASKS=${NUM_TASKS} | TASK=${SLURM_ARRAY_TASK_ID}"
echo "MODEL=${MODEL_PATH} | DTYPE=${DTYPE} | BACKEND=${BACKEND}"
echo "BATCH=${BATCH_SIZE} | MAX_NEW_TOKENS=${MAX_NEW_TOKENS} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | TEMP=${TEMPERATURE}"

python scripts/run5_pipeline.py \
  --input             "$INPUT" \
  --output_base       "$OUTPUT_BASE" \
  --text_col          "$TEXT_COL" \
  --n_rows            "$N_ROWS" \
  --model_path        "$MODEL_PATH" \
  --dtype             "$DTYPE" \
  --backend           "$BACKEND" \
  --trust_remote_code \
  --batch_size        "$BATCH_SIZE" \
  --max_new_tokens    "$MAX_NEW_TOKENS" \
  --max_input_tokens  "$MAX_INPUT_TOKENS" \
  --temperature       "$TEMPERATURE" \
  --num_tasks         "$NUM_TASKS"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminée."
