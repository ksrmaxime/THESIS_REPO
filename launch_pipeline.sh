#!/bin/bash
# =============================================================================
# launch_pipeline.sh — Lance le pipeline complet en une seule commande.
#
# Usage :
#   bash launch_pipeline.sh              # pipeline complet depuis le download
#   bash launch_pipeline.sh run3         # depuis run3 (fichier tagué par défaut)
#   bash launch_pipeline.sh run3 /path/to/tagged.csv
#   bash launch_pipeline.sh run4 /path/to/run3_merged.parquet
#   bash launch_pipeline.sh run5 /path/to/run4_merged.parquet
#   bash launch_pipeline.sh run6 /path/to/run5_standardized/results.parquet
#
# Chaque étape soumet automatiquement la suivante via --dependency=afterok.
# Un seul job actif à la fois — aucun superviseur, aucune limite de 3 jours.
# =============================================================================

set -euo pipefail

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO
STEP=${1:-"download"}
INPUT=${2:-""}

cd "$WORKDIR"

case "$STEP" in
  download)
    JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_download.sh")
    echo "Pipeline lancé depuis : download  (job ${JOB_ID})"
    ;;
  tag)
    JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_tag_keywords.sh")
    echo "Pipeline lancé depuis : tag_keywords  (job ${JOB_ID})"
    ;;
  run3)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run3_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run3_array.sh")
    fi
    echo "Pipeline lancé depuis : run3  (array job ${JOB_ID})"
    ;;
  run4)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run4_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run4_array.sh")
    fi
    echo "Pipeline lancé depuis : run4  (array job ${JOB_ID})"
    ;;
  run5)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run5_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run5_array.sh")
    fi
    echo "Pipeline lancé depuis : run5  (array job ${JOB_ID})"
    ;;
  standardize)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_standardize_run5.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_standardize_run5.sh")
    fi
    echo "Pipeline lancé depuis : standardize_run5  (job ${JOB_ID})"
    ;;
  run6)
    if [[ -n "$INPUT" ]]; then
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run6_array.sh" "$INPUT")
    else
        JOB_ID=$(sbatch --parsable "${WORKDIR}/sbatch_run6_array.sh")
    fi
    echo "Pipeline lancé depuis : run6  (array job ${JOB_ID})"
    ;;
  *)
    echo "Étape inconnue : '$STEP'"
    echo "Étapes valides : download | tag | run3 | run4 | run5 | standardize | run6"
    exit 1
    ;;
esac

echo ""
echo "Suivi : squeue -u \$USER"
echo "Logs  : ls ${WORKDIR}/logs/"
