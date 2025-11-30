#!/bin/bash
#SBATCH --job-name=demucs-train
#SBATCH --output=logs/demucs-train-%j.out
#SBATCH --error=logs/demucs-train-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G

set -euo pipefail

########################################
# User-configurable training settings
########################################
# Paths to DSD100 dataset roots (space-separated for multiple)
# DSD100 has 50 Dev + 50 Test tracks
DATASET_ROOTS="data/DSD100"
# Dataset subset to train on (Dev = 50 tracks)
SUBSET="Dev"
# Number of epochs
EPOCHS=300
# Batch size
BATCH_SIZE=4
# Learning rate
LR=1e-4
# Audio chunk duration in seconds
CHUNK_DURATION=6.0
# Chunks sampled per track per epoch
CHUNKS_PER_TRACK=20
# Model hidden channels (start small for faster training)
HIDDEN_CHANNELS=48
# Model depth (encoder/decoder layers)
DEPTH=5
# Save checkpoint every N epochs
SAVE_EVERY=10
# Resume from latest checkpoint
RESUME=""  # Set to "--resume" to continue from checkpoint
########################################

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
MODEL_DIR="${PROJECT_DIR}/model/demucs"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${MODEL_DIR}"

cd "${PROJECT_DIR}"

echo "[$(date)] Starting Demucs waveform model training"
echo "[$(date)] Dataset roots: ${DATASET_ROOTS} / subset: ${SUBSET}"
echo "[$(date)] Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}, LR: ${LR}"
echo "[$(date)] Chunk: ${CHUNK_DURATION}s, Chunks/track: ${CHUNKS_PER_TRACK}"
echo "[$(date)] Hidden: ${HIDDEN_CHANNELS}, Depth: ${DEPTH}"
echo "[$(date)] Model directory: ${MODEL_DIR}"

python -u training/Training_demucs.py \
    --dataset-roots ${DATASET_ROOTS} \
    --subset "${SUBSET}" \
    --model-dir "${MODEL_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --chunk-duration "${CHUNK_DURATION}" \
    --chunks-per-track "${CHUNKS_PER_TRACK}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --depth "${DEPTH}" \
    --save-every "${SAVE_EVERY}" \
    ${RESUME}

echo "[$(date)] Demucs training complete"
echo "[$(date)] Checkpoints saved to: ${MODEL_DIR}"
