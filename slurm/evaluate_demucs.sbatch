#!/bin/bash
#SBATCH --job-name=demucs-eval
#SBATCH --output=logs/demucs-eval-%j.out
#SBATCH --error=logs/demucs-eval-%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G

set -euo pipefail

########################################
# User-configurable evaluation settings
########################################
# Path to DSD100 dataset root
DATASET_ROOT="data/DSD100subset"
# Subset to evaluate: "Dev" or "Test"
SUBSET="Dev"
# Model architecture (must match training)
HIDDEN_CHANNELS=48
DEPTH=5
########################################

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
MODEL_DIR="${PROJECT_DIR}/model/demucs"
METRICS_DIR="${PROJECT_DIR}/metrics"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${METRICS_DIR}"

cd "${PROJECT_DIR}"

# Find latest checkpoint
LATEST_CKPT=$(ls -1 "${MODEL_DIR}"/demucs_final.pt 2>/dev/null || ls -1 "${MODEL_DIR}"/checkpoint_epoch_*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -z "${LATEST_CKPT}" ]; then
    echo "No Demucs checkpoints found in ${MODEL_DIR}."
    exit 1
fi

# Generate timestamped output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="${METRICS_DIR}/demucs_trained_${TIMESTAMP}.json"

echo "[$(date)] Evaluating trained Demucs checkpoint: ${LATEST_CKPT}"
echo "[$(date)] Dataset: ${DATASET_ROOT}, Subset: ${SUBSET}"
echo "[$(date)] Model config: hidden=${HIDDEN_CHANNELS}, depth=${DEPTH}"
echo "[$(date)] Metrics will be saved to: ${OUTPUT_JSON}"

python -u evaluation/evaluate_demucs.py \
    --dataset-root "${DATASET_ROOT}" \
    --subset "${SUBSET}" \
    --model-dir "${MODEL_DIR}" \
    --checkpoint "${LATEST_CKPT}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --depth "${DEPTH}" \
    --output-json "${OUTPUT_JSON}"

echo "[$(date)] Demucs evaluation complete"
echo "[$(date)] Metrics saved to: ${OUTPUT_JSON}"
