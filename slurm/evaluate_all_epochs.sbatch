#!/bin/bash
#SBATCH --job-name=freqnet-unet-eval-all
#SBATCH --output=logs/freqnet-unet-eval-all-%j.out
#SBATCH --error=logs/freqnet-unet-eval-all-%j.err
#SBATCH --time=12:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=48G

set -euo pipefail

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/DSD100}"

mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

mkdir -p metrics

MODEL_DIR="./model/unet"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -1 $MODEL_DIR/checkpoint_epoch_*.pt 2>/dev/null)" ]; then
    echo "Error: No checkpoints found in $MODEL_DIR/"
    exit 1
fi

echo "[$(date)] Starting evaluation of all epochs with 1 parallel worker"
python -u -m evaluation.evaluate_all_epochs \
    --dataset-root "$DATA_ROOT" \
    --subset Dev \
    --model-dir "$MODEL_DIR" \
    --num-workers 1 \
    --output-json "metrics/all_epochs_eval_${SLURM_JOB_ID}.json" 2>&1 | tee -a logs/freqnet-unet-eval-all-progress.log

echo "[$(date)] Evaluation complete"

