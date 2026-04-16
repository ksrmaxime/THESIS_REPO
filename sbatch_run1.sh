#!/bin/bash
#SBATCH --job-name=thesis_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/thesis_run_%j.out
#SBATCH --error=logs/thesis_run_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# =============================================================================
# USER VARIABLES — edit here, nowhere else
# =============================================================================

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO

# I/O
INPUT=${WORKDIR}/data/input/input.parquet
OUTPUT_BASE=${WORKDIR}/data/output
TEXT_COL=text
GOLD_CSV=${WORKDIR}/data/input/THESIS_RUN1_Max_Gold.csv

# Subset: number of rows to run (0 = full dataset)
N_ROWS=300

# Model
MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

# Inference
BATCH_SIZE=2
MAX_NEW_TOKENS=512
MAX_INPUT_TOKENS=16384
TEMPERATURE=0.0

# =============================================================================

module purge
module load python/3.12.1

cd "$WORKDIR"
source .venv/bin/activate

# Reduce GPU memory fragmentation (recommended when OOM on large models)
export PYTORCH_ALLOC_CONF=expandable_segments:True

mkdir -p logs data/processed

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

python scripts/run1_pipeline.py \
  --input             "$INPUT" \
  --output_base       "$OUTPUT_BASE" \
  --text_col          "$TEXT_COL" \
  --n_rows            "300" \
  --model_path        "$MODEL_PATH" \
  --dtype             "$DTYPE" \
  --backend           "$BACKEND" \
  --trust_remote_code \
  --batch_size        "$BATCH_SIZE" \
  --max_new_tokens    "$MAX_NEW_TOKENS" \
  --max_input_tokens  "$MAX_INPUT_TOKENS" \
  --temperature       "$TEMPERATURE"

# =============================================================================
# SCORING — disabled (new output format not yet aligned with gold data)
# =============================================================================

# PRED_CSV="${OUTPUT_BASE}_job${SLURM_JOB_ID}.csv"
# SCORE_LOG=$(python scripts/score.py \
#   --pred       "$PRED_CSV" \
#   --gold       "$GOLD_CSV" \
#   --id_col article_id \
#   --cols       "SWISS_CONTEXT,CRITICISM,TARGETED_ENTITY_TYPE,TARGETED_ENTITY_NAME,SOURCE_TYPE,SOURCE_NAME,CRITICISM_TOPIC,POPULIST_RHETORIC" \
#   --col_kinds  "TARGETED_ENTITY_NAME=text,SOURCE_NAME=text,CRITICISM_TOPIC=text" \
#   --extra_cols "$TEXT_COL" \
#   --report_dir "${WORKDIR}/data/output/run_job${SLURM_JOB_ID}/eval" \
#   --print_errors_head 10 \
#   --max_rows 300)
# echo "$SCORE_LOG"

# =============================================================================
# ARCHIVE — result + prompts + this sbatch stored together under one run folder
# =============================================================================

PRED_CSV="${OUTPUT_BASE}_job${SLURM_JOB_ID}.csv"
RUN_DIR="${WORKDIR}/data/output/run_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$PRED_CSV"            "$RUN_DIR/results.csv"       || true
cp "src/run1_prompts.py"  "$RUN_DIR/prompts_used.py"   || true
cp "$0"                   "$RUN_DIR/sbatch_used.sbatch" || true

echo "=== ARCHIVED ==="
echo "Run folder : $RUN_DIR"
echo "  results.csv"
echo "  prompts_used.py"
echo "  sbatch_used.sbatch"
