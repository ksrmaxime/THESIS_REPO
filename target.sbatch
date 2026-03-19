#!/bin/bash
#SBATCH --job-name=target_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/target_run_%j.out
#SBATCH --error=logs/target_run_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# =============================================================================
# USER VARIABLES — edit here, nowhere else
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO

# I/O
INPUT=${WORKDIR}/data/input/target_input.csv
OUTPUT_BASE=${WORKDIR}/data/output/target
TEXT_COL=TARGETED_ENTITY_NAME

# Subset: number of rows to run (0 = full dataset)
N_ROWS=0

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference
BATCH_SIZE=8
MAX_NEW_TOKENS=32
MAX_INPUT_TOKENS=512
TEMPERATURE=0.0

# =============================================================================

module purge
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs data/output

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

echo "=== RUN CONFIG ==="
echo "INPUT=${INPUT}"
echo "OUTPUT_BASE=${OUTPUT_BASE}"
echo "TEXT_COL=${TEXT_COL} | N_ROWS=${N_ROWS}"
echo "MODEL=${MODEL_PATH} | DTYPE=${DTYPE} | BACKEND=${BACKEND}"
echo "BATCH=${BATCH_SIZE} | MAX_NEW_TOKENS=${MAX_NEW_TOKENS} | MAX_INPUT_TOKENS=${MAX_INPUT_TOKENS} | TEMP=${TEMPERATURE}"

python scripts/target_pipeline.py \
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
  --temperature       "$TEMPERATURE"

# =============================================================================
# ARCHIVE — result + prompts + this sbatch stored together under one run folder
# =============================================================================

PRED_CSV="${OUTPUT_BASE}_job${SLURM_JOB_ID}.csv"
RUN_DIR="${WORKDIR}/data/output/run_target_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$PRED_CSV"                "$RUN_DIR/results.csv"          || true
cp "src/target_prompt.py"     "$RUN_DIR/prompts_used.py"
cp "$0"                       "$RUN_DIR/sbatch_used.sbatch"

echo "=== ARCHIVED ==="
echo "Run folder : $RUN_DIR"
echo "  results.csv"
echo "  prompts_used.py"
echo "  sbatch_used.sbatch"
