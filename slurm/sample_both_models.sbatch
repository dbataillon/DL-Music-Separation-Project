#!/bin/bash
#SBATCH --job-name=sample-both
#SBATCH --output=logs/sample-both-%j.out
#SBATCH --error=logs/sample-both-%j.err
#SBATCH --time=01:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G

# Sample from both epsilon-prediction and v-prediction diffusion models

set -euo pipefail

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
OUTPUT_DIR="${PROJECT_DIR}/samples_comparison"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

# Select a test track (can be changed)
TEST_TRACK="Spectrogram/055 - Angels In Amplifiers - I'm Alright.npz"

echo "[$(date)] Sampling from both diffusion models"
echo "[$(date)] Test track: ${TEST_TRACK}"
echo "[$(date)] Output directory: ${OUTPUT_DIR}"

python -u evaluation/sample_both_models.py \
    "${TEST_TRACK}" \
    --eps-model-dir "model/diffusion" \
    --vpred-model-dir "model/diffusion_vpred" \
    --output-dir "${OUTPUT_DIR}" \
    --timesteps 200

echo "[$(date)] Sampling complete"
echo "[$(date)] Results saved to: ${OUTPUT_DIR}"
