#!/bin/bash
#SBATCH --job-name=demucs-pretrained-eval
#SBATCH --output=logs/demucs-pretrained-eval-%j.out
#SBATCH --error=logs/demucs-pretrained-eval-%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G

set -euo pipefail

########################################
# User-configurable settings
########################################
# Path to DSD100 dataset root (containing Mixtures/ and Sources/)
DATASET_ROOT="data/DSD100subset"
# Subset to evaluate: "Dev" or "Test"
SUBSET="Dev"
# Demucs model: htdemucs, htdemucs_ft, mdx_extra, mdx_extra_q
MODEL_NAME="htdemucs"
########################################

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
METRICS_DIR="${PROJECT_DIR}/metrics"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${METRICS_DIR}"

cd "${PROJECT_DIR}"

# Install demucs if not present
pip show demucs > /dev/null 2>&1 || pip install demucs

# Generate timestamped output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="${METRICS_DIR}/demucs_pretrained_${MODEL_NAME}_${TIMESTAMP}.json"

echo "[$(date)] Evaluating pretrained Demucs model: ${MODEL_NAME}"
echo "[$(date)] Dataset: ${DATASET_ROOT}, Subset: ${SUBSET}"
echo "[$(date)] Metrics will be saved to: ${OUTPUT_JSON}"

python -u evaluation/evaluate_demucs_pretrained.py \
    --dataset-root "${DATASET_ROOT}" \
    --subset "${SUBSET}" \
    --model-name "${MODEL_NAME}" \
    --output-json "${OUTPUT_JSON}"

echo "[$(date)] Pretrained Demucs evaluation complete"
echo "[$(date)] Metrics saved to: ${OUTPUT_JSON}"
