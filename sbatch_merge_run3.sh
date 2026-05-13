#!/bin/bash
#SBATCH --job-name=merge_run3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_run3_%j.out
#SBATCH --error=logs/merge_run3_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage:
#   ARRAY_ID=$(sbatch --parsable sbatch_run3_array.sh)
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_run3.sh ${ARRAY_ID}
#
# Ou en une commande :
#   ARRAY_ID=$(sbatch --parsable sbatch_run3_array.sh) && \
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_run3.sh ${ARRAY_ID}

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

ARRAY_JOB_ID=${1:-""}
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERROR] Passer le ARRAY_JOB_ID en argument: sbatch sbatch_merge_run3.sh <ARRAY_JOB_ID>"
    exit 1
fi

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/run3"
MERGED_PARQUET="${OUTDIR}/run3_merged_job${ARRAY_JOB_ID}.parquet"
MERGED_CSV="${OUTDIR}/run3_merged_job${ARRAY_JOB_ID}.csv"
MERGED_WIDE_CSV="${OUTDIR}/run3_merged_wide_job${ARRAY_JOB_ID}.csv"

echo "=== MERGE run3 array job ${ARRAY_JOB_ID} ==="
echo "DATE=$(date -Is)"

export OUTDIR OUTBASE ARRAY_JOB_ID MERGED_PARQUET MERGED_CSV MERGED_WIDE_CSV

python3 - <<'PYEOF'
import sys
import os
import glob
import json
import pandas as pd

outbase        = os.environ["OUTBASE"]
job_id         = os.environ["ARRAY_JOB_ID"]
merged_parquet = os.environ["MERGED_PARQUET"]
merged_csv     = os.environ["MERGED_CSV"]
merged_wide_csv = os.environ["MERGED_WIDE_CSV"]

# ---------------------------------------------------------------------------
# 1. Gather task outputs
# ---------------------------------------------------------------------------
pattern = f"{outbase}_task*_job{job_id}.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{outbase}_task*_job{job_id}.csv"
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

sort_col = "article_id" if "article_id" in merged.columns else merged.columns[0]
merged = merged.sort_values(sort_col).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Save compact version (with JSON keyword_stances column)
# ---------------------------------------------------------------------------
merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes")
print(f"[merge] Sorted by: {sort_col}")
print(f"[merge] Compact → {merged_parquet}")
print(f"[merge] Compact → {merged_csv}")

filled = merged["keyword_stances"].notna().sum()
print(f"[merge] keyword_stances rempli: {filled:,} / {len(merged):,}")

# ---------------------------------------------------------------------------
# 3. Expand keyword_stances JSON into individual KW_<keyword> columns
#    Values: CRITICIZED | PRAISED | NEUTRAL | NaN (keyword absent from article)
# ---------------------------------------------------------------------------
def parse_stances(raw) -> dict:
    if pd.isna(raw) or not str(raw).strip():
        return {}
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}

all_stances = merged["keyword_stances"].apply(parse_stances)

# Collect the full keyword universe across all articles
all_keywords = sorted({kw for stances in all_stances for kw in stances})
print(f"[merge] {len(all_keywords)} keywords distincts trouvés dans les données")

# Build wide DataFrame: one column per keyword, prefixed with KW_
wide_cols = {}
for kw in all_keywords:
    col_name = f"KW_{kw}"
    wide_cols[col_name] = all_stances.apply(lambda d, k=kw: d.get(k, pd.NA))

wide_df = pd.DataFrame(wide_cols, index=merged.index)

# Concatenate with original columns (drop the JSON column to avoid duplication)
merged_wide = pd.concat([merged, wide_df], axis=1)
merged_wide.to_csv(merged_wide_csv, index=False, encoding="utf-8-sig")

print(f"[merge] Wide    → {merged_wide_csv}")
print(f"[merge] Colonnes wide: {list(wide_cols.keys())[:10]} {'...' if len(all_keywords) > 10 else ''}")

# Summary of stances distribution
stance_counts = {"CRITICIZED": 0, "PRAISED": 0, "NEUTRAL": 0}
for d in all_stances:
    for v in d.values():
        if v in stance_counts:
            stance_counts[v] += 1
total_classifications = sum(stance_counts.values())
print(f"\n[merge] Distribution des stances ({total_classifications:,} classifications):")
for stance, count in stance_counts.items():
    pct = 100 * count / total_classifications if total_classifications else 0
    print(f"  {stance}: {count:,} ({pct:.1f}%)")

PYEOF

# =============================================================================
# ARCHIVE — merged results + prompts + sbatch under one run folder
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/run3_merged_job${ARRAY_JOB_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"           "${RUN_DIR}/results_compact.csv"     || true
cp "$MERGED_PARQUET"       "${RUN_DIR}/results_compact.parquet" || true
cp "$MERGED_WIDE_CSV"      "${RUN_DIR}/results_wide.csv"        || true
cp "src/run3_prompts.py"   "${RUN_DIR}/prompts_used.py"         || true
cp "sbatch_run3_array.sh"  "${RUN_DIR}/sbatch_array_used.sh"    || true
cp "$0"                    "${RUN_DIR}/sbatch_merge_used.sh"    || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"
echo "  results_compact.csv / results_compact.parquet"
echo "  results_wide.csv  (une colonne KW_<keyword> par keyword)"
echo "  prompts_used.py"
echo "  sbatch_array_used.sh / sbatch_merge_used.sh"

echo "Merge terminé."
