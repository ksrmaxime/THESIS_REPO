#!/bin/bash
#SBATCH --job-name=thesis_run2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/thesis_run2_%j.out
#SBATCH --error=logs/thesis_run2_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# =============================================================================
# USER VARIABLES — edit here, nowhere else
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO

# I/O — point INPUT to the run1 output parquet (or csv)
INPUT=${WORKDIR}/data/output/output_job60427540.parquet
OUTPUT_BASE=${WORKDIR}/data/output/run2
TEXT_COL=CRITICISM_SUMMARY

# Subset: number of rows to run (0 = full dataset)
N_ROWS=0

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference
BATCH_SIZE=4
MAX_NEW_TOKENS=256
MAX_INPUT_TOKENS=2048
TEMPERATURE=0.0

# =============================================================================

module purge
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

export PYTORCH_ALLOC_CONF=expandable_segments:True

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

python scripts/run2_pipeline.py \
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
RUN_DIR="${WORKDIR}/data/output/run2_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$PRED_CSV"              "$RUN_DIR/results.csv"          || true
cp "src/run2_prompts.py"    "$RUN_DIR/prompts_used.py"      || true
cp "$0"                     "$RUN_DIR/sbatch_used.sbatch"   || true

echo "=== ARCHIVED ==="
echo "Run folder : $RUN_DIR"
echo "  results.csv"
echo "  prompts_used.py"
echo "  sbatch_used.sbatch"
