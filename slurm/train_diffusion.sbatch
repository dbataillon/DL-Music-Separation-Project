#!/bin/bash
#SBATCH --job-name=freqnet-diffusion-train
#SBATCH --output=logs/freqnet-diffusion-train-%j.out
#SBATCH --error=logs/freqnet-diffusion-train-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:V100:2
#SBATCH --mem=48G

set -euo pipefail

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/model/diffusion"

cd "${PROJECT_DIR}"

spectro_count=$(find Spectrogram -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l || true)
if [ "$spectro_count" -eq 0 ]; then
    echo "Spectrogram cache missing. Run slurm/preprocess_spectrograms.sbatch first."
    exit 1
fi

# To use multiple GPUs, change --gres=gpu:V100:2 in the SBATCH header above
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[$(date)] Starting diffusion training with $NUM_GPUS GPU(s)"
echo "[$(date)] Checkpoints will be saved to model/diffusion/"
echo "[$(date)] Training will resume from last checkpoint if available"

python -u -m training.Training_diffusion 2>&1 | tee -a logs/freqnet-diffusion-train-progress.log

echo "[$(date)] Diffusion training finished"
