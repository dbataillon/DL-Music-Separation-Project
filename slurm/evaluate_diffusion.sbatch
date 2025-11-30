#!/bin/bash
#SBATCH --job-name=freqnet-diffusion-eval
#SBATCH --output=logs/freqnet-diffusion-eval-%j.out
#SBATCH --error=logs/freqnet-diffusion-eval-%j.err
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
# Path to DSD100 dataset root (containing Mixtures/ and Sources/)
DATASET_ROOT="data/DSD100subset"
# Subset to evaluate: "Dev" or "Test"
SUBSET="Dev"
# Number of diffusion timesteps
TIMESTEPS=200
########################################

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
MODEL_DIR="${PROJECT_DIR}/model/diffusion"
METRICS_DIR="${PROJECT_DIR}/metrics"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${METRICS_DIR}"

cd "${PROJECT_DIR}"

# Find latest checkpoint
LATEST_CKPT=$(ls -1 "${MODEL_DIR}"/checkpoint_epoch_*.pt 2>/dev/null | sort | tail -n 1 || true)
if [ -z "${LATEST_CKPT}" ]; then
    echo "No diffusion checkpoints found in ${MODEL_DIR}."
    exit 1
fi

# Generate timestamped output filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="${METRICS_DIR}/diffusion_eval_${TIMESTAMP}.json"

echo "[$(date)] Evaluating diffusion checkpoint: ${LATEST_CKPT}"
echo "[$(date)] Dataset: ${DATASET_ROOT}, Subset: ${SUBSET}"
echo "[$(date)] Metrics will be saved to: ${OUTPUT_JSON}"

python -u evaluation/evaluate_diffusion.py \
    --dataset-root "${DATASET_ROOT}" \
    --subset "${SUBSET}" \
    --model-dir "${MODEL_DIR}" \
    --checkpoint "${LATEST_CKPT}" \
    --timesteps "${TIMESTEPS}" \
    --output-json "${OUTPUT_JSON}"

echo "[$(date)] Diffusion evaluation complete"
echo "[$(date)] Metrics saved to: ${OUTPUT_JSON}"
