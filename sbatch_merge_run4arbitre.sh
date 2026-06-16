#!/bin/bash -l
#SBATCH --job-name=merge_run4arbitre
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_run4arbitre_%j.out
#SBATCH --error=logs/merge_run4arbitre_%j.err
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

ARRAY_JOB_ID=${1:-""}
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERROR] Passer le ARRAY_JOB_ID en argument: sbatch sbatch_merge_run4arbitre.sh <ARRAY_JOB_ID>" >&2
    exit 1
fi

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/run4arbitre"
MERGE_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
MERGED_PARQUET="${OUTDIR}/run4arbitre_merged_${MERGE_ID}.parquet"
MERGED_CSV="${OUTDIR}/run4arbitre_merged_${MERGE_ID}.csv"

echo "=== MERGE run4arbitre (job ${MERGE_ID}) ==="
echo "DATE=$(date -Is)"

export OUTDIR OUTBASE ARRAY_JOB_ID MERGE_ID MERGED_PARQUET MERGED_CSV

python3 - <<'PYEOF'
import sys
import os
import glob
import pandas as pd

outbase        = os.environ["OUTBASE"]
array_job_id   = os.environ["ARRAY_JOB_ID"]
merged_parquet = os.environ["MERGED_PARQUET"]
merged_csv     = os.environ["MERGED_CSV"]

# ---------------------------------------------------------------------------
# 1. Gather task outputs
# ---------------------------------------------------------------------------
pattern = f"{outbase}_task*_job{array_job_id}.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{outbase}_task*_job{array_job_id}.csv"
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
# 2. Résoudre → critic_answer_final
#    critic_answer             = réponse run4 originale (jamais modifiée)
#    run4_eval_justification   = description indépendante de l'évaluateur (jamais modifiée)
#    arbiter_answer            = description synthétisée par l'arbitre (seulement si NO)
#    critic_answer_final       = colonne résolue transmise à run5
#      run4_valid == YES  → critic_answer (validé directement)
#      run4_valid == NO   → arbiter_answer (synthèse de l'arbitre)
#      fallback           → critic_answer
# ---------------------------------------------------------------------------
def resolve_final_answer(row):
    if str(row.get("run4_valid", "")).strip().upper() == "NO":
        val = row.get("arbiter_answer", None)
        if val is not None and not pd.isna(val) and str(val).strip():
            return str(val).strip()
    return row["critic_answer"]

merged["critic_answer_final"] = merged.apply(resolve_final_answer, axis=1)

# ---------------------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------------------
merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes")
print(f"[merge] Trié par: {sort_cols}")
print(f"[merge] → {merged_parquet}")
print(f"[merge] → {merged_csv}")

if "arbiter_answer" in merged.columns:
    filled   = merged["arbiter_answer"].notna().sum()
    na_count = merged["arbiter_answer"].isna().sum()
    total    = len(merged)
    print(f"\n[merge] arbiter_answer: {filled:,} synthèses générées / {na_count:,} non-arbitré (total {total:,})")

if "critic_answer_final" in merged.columns:
    filled = merged["critic_answer_final"].notna().sum()
    total  = len(merged)
    print(f"[merge] critic_answer_final rempli: {filled:,}/{total:,} ({100*filled/total:.1f}%)")

PYEOF

# =============================================================================
# ARCHIVE
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/run4arbitre_merged_${MERGE_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"                  "${RUN_DIR}/results.csv"            || true
cp "$MERGED_PARQUET"              "${RUN_DIR}/results.parquet"        || true
cp "src/run4arbitre_prompts.py"   "${RUN_DIR}/prompts_used.py"        || true
cp "sbatch_run4arbitre_array.sh"  "${RUN_DIR}/sbatch_array_used.sh"   || true
cp "$0"                           "${RUN_DIR}/sbatch_merge_used.sh"   || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"

echo "Merge terminé."
echo "${RUN_DIR}" > "${WORKDIR}/data/output/.last_run4arbitre_archive"

# ── Auto-chain ─────────────────────────────────────────────────────────────────
sbatch "${WORKDIR}/sbatch_run5_array.sh" "${MERGED_PARQUET}" "critic_answer_final"
echo "[chain] → sbatch_run5_array.sh submitted (input: ${MERGED_PARQUET}, text_col: critic_answer_final)"
