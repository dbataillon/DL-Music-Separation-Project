#!/bin/bash
#SBATCH --job-name=freqnet-unet-train
#SBATCH --output=logs/freqnet-unet-train-%j.out
#SBATCH --error=logs/freqnet-unet-train-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=48G

set -euo pipefail

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"

mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

mkdir -p logs

spectro_count=$(find Spectrogram -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l || true)
if [ "$spectro_count" -eq 0 ]; then
    echo "Spectrogram cache missing. Run slurm/preprocess_spectrograms.sbatch first."
    exit 1
fi

# To use multiple GPUs, change --gres=gpu:V100:2 in the SBATCH header above
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[$(date)] Starting training with $NUM_GPUS GPU(s)"
python -u -m training.train_baseline 2>&1 | tee -a logs/freqnet-unet-train-progress.log
echo "[$(date)] Training finished"
