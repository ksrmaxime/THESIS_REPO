#!/bin/bash
#SBATCH --job-name=merge_run2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_run2_%j.out
#SBATCH --error=logs/merge_run2_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage:
#   ARRAY_ID=$(sbatch --parsable sbatch_run2_array.sh) && \
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_run2.sh ${ARRAY_ID}

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

ARRAY_JOB_ID=${1:-""}
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERROR] Passer le ARRAY_JOB_ID en argument: sbatch sbatch_merge_run2.sh <ARRAY_JOB_ID>"
    exit 1
fi

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/run2"
MERGED_PARQUET="${OUTDIR}/run2_merged_job${ARRAY_JOB_ID}.parquet"
MERGED_CSV="${OUTDIR}/run2_merged_job${ARRAY_JOB_ID}.csv"

echo "=== MERGE run2 array job ${ARRAY_JOB_ID} ==="
echo "DATE=$(date -Is)"

export OUTBASE ARRAY_JOB_ID MERGED_PARQUET MERGED_CSV

python3 - <<'PYEOF'
import sys, os, glob
import pandas as pd

outbase        = os.environ["OUTBASE"]
job_id         = os.environ["ARRAY_JOB_ID"]
merged_parquet = os.environ["MERGED_PARQUET"]
merged_csv     = os.environ["MERGED_CSV"]

pattern = f"{outbase}_task*_job{job_id}.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{outbase}_task*_job{job_id}.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] Aucun fichier trouvé avec le pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    dfs = [pd.read_csv(f) for f in files]
else:
    dfs = [pd.read_parquet(f) for f in files]

print(f"[merge] {len(files)} fichiers trouvés:")
for f, df_f in zip(files, dfs):
    print(f"  {f}  →  {len(df_f):,} lignes")

merged = pd.concat(dfs, ignore_index=True)

sort_col = "article_id" if "article_id" in merged.columns else merged.columns[0]
merged = merged.sort_values(sort_col).reset_index(drop=True)

merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes")
print(f"[merge] Sorted by: {sort_col}")
print(f"[merge] Saved → {merged_parquet}")
print(f"[merge] Saved → {merged_csv}")

for col in ["SOURCE", "TARGET", "WHAT"]:
    if col in merged.columns:
        filled = merged[col].notna().sum()
        print(f"[merge] {col} rempli: {filled:,} / {len(merged):,}")
PYEOF

# =============================================================================
# ARCHIVE
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/run2_merged_job${ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"                "${RUN_DIR}/results.csv"          || true
cp "$MERGED_PARQUET"            "${RUN_DIR}/results.parquet"      || true
cp "src/run2_prompts.py"        "${RUN_DIR}/prompts_used.py"      || true
cp "sbatch_run2_array.sh"       "${RUN_DIR}/sbatch_array_used.sh" || true
cp "$0"                         "${RUN_DIR}/sbatch_merge_used.sh" || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"
echo "  results.csv / results.parquet"
echo "  prompts_used.py"
echo "  sbatch_array_used.sh / sbatch_merge_used.sh"

echo "Merge terminé."
