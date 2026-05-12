#!/bin/bash
#SBATCH --job-name=thesis_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_downloads_%j.out
#SBATCH --error=logs/merge_downloads_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/THESIS_REPO"
cd "${REPO_DIR}"

mkdir -p logs

module purge
module load python/3.12.1

source .venv/bin/activate

python scripts/merge_downloads.py \
  --indir  data/input \
  --outdir data/input \
  --out-stem swissdox_all

echo "Merge finished."
