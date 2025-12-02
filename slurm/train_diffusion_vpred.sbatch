#!/bin/bash
#SBATCH --job-name=diff-vpred-log
#SBATCH --output=logs/diffusion-vpred-log-%j.out
#SBATCH --error=logs/diffusion-vpred-log-%j.err
#SBATCH --time=08:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=48G

# V-prediction diffusion with log-magnitude spectrograms

set -euo pipefail

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/model/diffusion_vpred"

cd "${PROJECT_DIR}"

spectro_count=$(find Spectrogram -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l || true)
if [ "$spectro_count" -eq 0 ]; then
    echo "Spectrogram cache missing. Run slurm/preprocess_spectrograms.sbatch first."
    exit 1
fi

# To use multiple GPUs, change --gres=gpu:V100:2 in the SBATCH header above
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
echo "[$(date)] Starting V-PREDICTION diffusion training with LOG spectrograms"
echo "[$(date)] Using $NUM_GPUS GPU(s)"
echo "[$(date)] Checkpoints will be saved to model/diffusion_vpred/"
echo "[$(date)] Training will resume from last checkpoint if available"

python -u -m training.Training_diffusion_vpred 2>&1 | tee -a logs/diffusion-vpred-log-progress.log

echo "[$(date)] V-prediction diffusion training finished"
