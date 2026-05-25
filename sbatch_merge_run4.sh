#!/bin/bash
#SBATCH --job-name=merge_run4
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_run4_%j.out
#SBATCH --error=logs/merge_run4_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage:
#   sbatch sbatch_merge_run4.sh
#
# Ou avec dépendance sur le job array :
#   ARRAY_ID=$(sbatch --parsable sbatch_run4_array.sh) && \
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_run4.sh

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/run4"
MERGE_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
MERGED_PARQUET="${OUTDIR}/run4_merged_${MERGE_ID}.parquet"
MERGED_CSV="${OUTDIR}/run4_merged_${MERGE_ID}.csv"

echo "=== MERGE run4 (job ${MERGE_ID}) ==="
echo "DATE=$(date -Is)"

export OUTDIR OUTBASE MERGE_ID MERGED_PARQUET MERGED_CSV

python3 - <<'PYEOF'
import sys
import os
import glob
import pandas as pd

outbase        = os.environ["OUTBASE"]
merged_parquet = os.environ["MERGED_PARQUET"]
merged_csv     = os.environ["MERGED_CSV"]

# ---------------------------------------------------------------------------
# 1. Gather task outputs (YES rows only: one row per article+keyword)
# ---------------------------------------------------------------------------
pattern = f"{outbase}_task*.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{outbase}_task*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] Aucun fichier trouvé avec le pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
else:
    dfs = [pd.read_parquet(f) for f in files]

print(f"[merge] {len(files)} fichiers trouvés:")
for f, df_f in zip(files, dfs):
    print(f"  {f}  →  {len(df_f):,} lignes")

merged = pd.concat(dfs, ignore_index=True)

sort_cols = [c for c in ["article_id", "keyword"] if c in merged.columns]
if not sort_cols:
    sort_cols = [merged.columns[0]]
merged = merged.sort_values(sort_cols).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Save
# ---------------------------------------------------------------------------
merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes (keyword_answer=YES, une par article+keyword)")
print(f"[merge] Trié par: {sort_cols}")
print(f"[merge] → {merged_parquet}")
print(f"[merge] → {merged_csv}")

# Summary of critic_answer
if "critic_answer" in merged.columns:
    filled = merged["critic_answer"].notna().sum()
    total = len(merged)
    print(f"\n[merge] critic_answer rempli: {filled:,}/{total:,} ({100*filled/total:.1f}%)")

PYEOF

# =============================================================================
# ARCHIVE — merged results + prompts + sbatch under one run folder
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/run4_merged_${MERGE_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"           "${RUN_DIR}/results.csv"            || true
cp "$MERGED_PARQUET"       "${RUN_DIR}/results.parquet"        || true
cp "src/run4_prompts.py"   "${RUN_DIR}/prompts_used.py"        || true
cp "sbatch_run4_array.sh"  "${RUN_DIR}/sbatch_array_used.sh"   || true
cp "$0"                    "${RUN_DIR}/sbatch_merge_used.sh"   || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"
echo "  results.csv / results.parquet  (une ligne par article+keyword, keyword_answer=YES)"
echo "  prompts_used.py"
echo "  sbatch_array_used.sh / sbatch_merge_used.sh"

echo "Merge terminé."
