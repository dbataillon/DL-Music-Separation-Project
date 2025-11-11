#!/bin/bash
#SBATCH --job-name=freqnet-data
#SBATCH --output=logs/freqnet-data-%j.out
#SBATCH --error=logs/freqnet-data-%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=coc-ice
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

module load anaconda3
conda activate freqnet
PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
DATADIR="${PROJECT_DIR}/data"
ENV_NAME="freqnet"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${DATADIR}"

cd "${PROJECT_DIR}"

python download_data.py --DATADIR "${DATADIR}"
