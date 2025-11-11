#!/bin/bash
#SBATCH --job-name=freqnet-pre
#SBATCH --output=logs/freqnet-pre-%j.out
#SBATCH --error=logs/freqnet-pre-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=coc-ice
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
DATA_PARENT="${DATA_PARENT:-${PROJECT_DIR}/data}"

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

cd "${PROJECT_DIR}"

mkdir -p logs

echo "[$(date)] Starting spectrogram preprocessing"
python -m preprocessing.CCMixter_process --DATADIR "${DATA_PARENT}"
echo "[$(date)] Preprocessing finished"
