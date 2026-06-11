#!/bin/bash -l
#SBATCH --job-name=merge_run5eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_run5eval_%j.out
#SBATCH --error=logs/merge_run5eval_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

set -euo pipefail

module purge || true
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/run5eval"
MERGE_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
MERGED_PARQUET="${OUTDIR}/run5eval_merged_${MERGE_ID}.parquet"
MERGED_CSV="${OUTDIR}/run5eval_merged_${MERGE_ID}.csv"

echo "=== MERGE run5eval (job ${MERGE_ID}) ==="
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
# 1. Gather task outputs
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

print(f"\n[merge] Total: {len(merged):,} lignes")
print(f"[merge] Trié par: {sort_cols}")
print(f"[merge] → {merged_parquet}")
print(f"[merge] → {merged_csv}")

if "run5_valid" in merged.columns:
    yes  = (merged["run5_valid"] == "YES").sum()
    no   = (merged["run5_valid"] == "NO").sum()
    na   = merged["run5_valid"].isna().sum()
    total = len(merged)
    print(f"\n[merge] run5_valid: {yes:,} YES / {no:,} NO / {na:,} non-traité (total {total:,})")

PYEOF

# =============================================================================
# ARCHIVE
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/run5eval_merged_${MERGE_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"              "${RUN_DIR}/results.csv"            || true
cp "$MERGED_PARQUET"          "${RUN_DIR}/results.parquet"        || true
cp "src/run5eval_prompts.py"  "${RUN_DIR}/prompts_used.py"        || true
cp "sbatch_run5eval_array.sh" "${RUN_DIR}/sbatch_array_used.sh"   || true
cp "$0"                       "${RUN_DIR}/sbatch_merge_used.sh"   || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"

echo "Merge terminé."

# ── Auto-chain ─────────────────────────────────────────────────────────────────
sbatch "${WORKDIR}/sbatch_run6_array.sh" "${MERGED_PARQUET}"
echo "[chain] → sbatch_run6_array.sh submitted (input: ${MERGED_PARQUET})"
