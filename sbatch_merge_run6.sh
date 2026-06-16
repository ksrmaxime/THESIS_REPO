#!/bin/bash -l
#SBATCH --job-name=merge_run6
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_run6_%j.out
#SBATCH --error=logs/merge_run6_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

dcsrsoft use 20241118

# Usage:
#   sbatch sbatch_merge_run6.sh <ARRAY_JOB_ID>
#
# Ou avec dépendance sur le job array :
#   ARRAY_ID=$(sbatch --parsable sbatch_run6_array.sh) && \
#   sbatch --dependency=afterok:${ARRAY_ID} sbatch_merge_run6.sh ${ARRAY_ID}

set -euo pipefail

# $1 = ARRAY_JOB_ID (obligatoire)
if [[ -z "${1:-}" ]]; then
    echo "[ERROR] Usage: sbatch sbatch_merge_run6.sh <ARRAY_JOB_ID>" >&2
    exit 1
fi
ARRAY_JOB_ID="$1"

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

OUTDIR="${WORKDIR}/data/output"
OUTBASE="${OUTDIR}/run6"
MERGE_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
MERGED_PARQUET="${OUTDIR}/run6_merged_${MERGE_ID}.parquet"
MERGED_CSV="${OUTDIR}/run6_merged_${MERGE_ID}.csv"

echo "=== MERGE run6 (job ${MERGE_ID}) — array ${ARRAY_JOB_ID} ==="
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
# 1. Gather task outputs — filtrer sur le job ID pour éviter de mélanger
#    les sorties de runs précédents
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
# 2. Save
# ---------------------------------------------------------------------------
merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes")
print(f"[merge] Trié par: {sort_cols}")
print(f"[merge] → {merged_parquet}")
print(f"[merge] → {merged_csv}")

if "criticism_target" in merged.columns:
    filled = merged["criticism_target"].notna().sum()
    total = len(merged)
    print(f"\n[merge] criticism_target rempli: {filled:,}/{total:,} ({100*filled/total:.1f}%)")
    print("\n[merge] Distribution criticism_target:")
    print(merged["criticism_target"].value_counts(dropna=False).to_string())

PYEOF

# =============================================================================
# ARCHIVE — merged results + prompts + sbatch under one run folder
# =============================================================================

RUN_DIR="${WORKDIR}/data/output/run6_merged_${MERGE_ID}"
mkdir -p "$RUN_DIR"

cp "$MERGED_CSV"           "${RUN_DIR}/results.csv"            || true
cp "$MERGED_PARQUET"       "${RUN_DIR}/results.parquet"        || true
cp "src/run6_prompts.py"   "${RUN_DIR}/prompts_used.py"        || true
cp "sbatch_run6_array.sh"  "${RUN_DIR}/sbatch_array_used.sh"   || true
cp "$0"                    "${RUN_DIR}/sbatch_merge_used.sh"   || true

echo "=== ARCHIVED ==="
echo "Run folder : ${RUN_DIR}"
echo "  results.csv / results.parquet"
echo "  prompts_used.py"
echo "  sbatch_array_used.sh / sbatch_merge_used.sh"

echo "Merge terminé."
echo "${RUN_DIR}" > "${WORKDIR}/data/output/.last_run6_archive"

# ── Auto-chain ─────────────────────────────────────────────────────────────────
sbatch "${WORKDIR}/sbatch_analysis.sh" "${MERGED_PARQUET}"
echo "[chain] → sbatch_analysis.sh submitted (input: ${MERGED_PARQUET})"

echo "=== PIPELINE COMPLET (run3-run6) ==="
echo "Résultat final : ${MERGED_CSV}"

# =============================================================================
# COPY TO NAS — un dossier horodaté avec les résultats et prompts de chaque run
# =============================================================================

NAS_BASE="/nas/FAC/FDCA/IDHEAP/mhinterl/parp/D2c/maxime/THESIS/output"
NAS_RUN_DIR="${NAS_BASE}/pipeline_$(date +%Y%m%d_%H%M%S)_job${ARRAY_JOB_ID}"

echo "=== COPY TO NAS ==="
echo "Destination : ${NAS_RUN_DIR}"

if mkdir -p "$NAS_RUN_DIR" 2>/dev/null; then
    for run_name in run3 run4 run4eval run5 run5eval run6; do
        pointer="${WORKDIR}/data/output/.last_${run_name}_archive"
        if [[ -f "$pointer" ]]; then
            archive_dir=$(cat "$pointer")
            echo "[NAS] ${run_name} ← ${archive_dir}"
            for ext in parquet csv; do
                src="${archive_dir}/results.${ext}"
                if [[ -f "$src" ]]; then
                    cp "$src" "${NAS_RUN_DIR}/${run_name}_results.${ext}" \
                        && echo "  → ${run_name}_results.${ext}" || true
                fi
            done
            src_prompts="${archive_dir}/prompts_used.py"
            if [[ -f "$src_prompts" ]]; then
                cp "$src_prompts" "${NAS_RUN_DIR}/${run_name}_prompts.py" \
                    && echo "  → ${run_name}_prompts.py" || true
            fi
        else
            echo "[NAS] WARN: pointeur manquant pour ${run_name} (${pointer})" >&2
        fi
    done
    echo "=== NAS COPY DONE : ${NAS_RUN_DIR} ==="
else
    echo "[WARN] Impossible de créer ${NAS_RUN_DIR} — NAS non monté ?" >&2
fi
