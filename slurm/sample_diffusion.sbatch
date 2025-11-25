#!/bin/bash
#SBATCH --job-name=freqnet-diffusion-sample
#SBATCH --output=logs/freqnet-diffusion-sample-%j.out
#SBATCH --error=logs/freqnet-diffusion-sample-%j.err
#SBATCH --time=02:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G

set -euo pipefail

########################
# User-configurable section
########################
# Path to spectrogram .npz inside the project (relative or absolute)
SPECTROGRAM_PATH="Spectrogram/055 - Angels In Amplifiers - I'm Alright.npz"

# Optional: path to the mixture WAV for writing audio stems (leave empty to skip audio output)
MIXTURE_AUDIO_PATH="data/DSD100subset/Mixtures/Dev/055 - Angels In Amplifiers - I'm Alright/mixture.wav"

# Optional: override checkpoint path (leave empty to use latest in MODEL_DIR)
CHECKPOINT_PATH=""

# Sampling hyperparameters and output locations
TIMESTEPS=200
OUTPUT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project/samples"
AUDIO_OUTPUT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project/samples/audio"
########################

if [ -z "${SPECTROGRAM_PATH}" ]; then
    echo "Edit slurm/sample_diffusion.sbatch and set SPECTROGRAM_PATH before submitting."
    exit 1
fi

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
MODEL_DIR="${PROJECT_DIR}/model/diffusion"

if [[ "${SPECTROGRAM_PATH}" != /* ]]; then
    SPECTROGRAM_PATH="${PROJECT_DIR}/${SPECTROGRAM_PATH}"
fi

if [ -n "${MIXTURE_AUDIO_PATH}" ] && [[ "${MIXTURE_AUDIO_PATH}" != /* ]]; then
    MIXTURE_AUDIO_PATH="${PROJECT_DIR}/${MIXTURE_AUDIO_PATH}"
fi

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${AUDIO_OUTPUT_DIR}"

cd "${PROJECT_DIR}"

if [ ! -f "${SPECTROGRAM_PATH}" ]; then
    echo "Spectrogram file ${SPECTROGRAM_PATH} not found."
    exit 1
fi

CMD=(
    python -u evaluation/sample_diffusion.py "${SPECTROGRAM_PATH}" \
        --model-dir "${MODEL_DIR}" \
        --timesteps "${TIMESTEPS}"
)

if [ -n "${CHECKPOINT_PATH}" ]; then
    CMD+=(--checkpoint "${CHECKPOINT_PATH}")
fi

BASENAME=$(basename "${SPECTROGRAM_PATH}" .npz)
OUTPUT_PATH="${OUTPUT_DIR}/${BASENAME}_pred.npz"
CMD+=(--output "${OUTPUT_PATH}")

if [ -n "${MIXTURE_AUDIO_PATH}" ]; then
    CMD+=(--mixture-audio "${MIXTURE_AUDIO_PATH}")
    CMD+=(--audio-out-dir "${AUDIO_OUTPUT_DIR}/${BASENAME}")
fi

echo "[$(date)] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[$(date)] Sampling finished. Output saved to ${OUTPUT_PATH}"
