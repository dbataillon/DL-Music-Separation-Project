#!/bin/bash
#SBATCH --job-name=freqnet-unet-eval
#SBATCH --output=logs/freqnet-unet-eval-%j.out
#SBATCH --error=logs/freqnet-unet-eval-%j.err
#SBATCH --time=04:00:00
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=16G

set -euo pipefail

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/DSD100}"
SAMPLE_WAV_PATH="${SAMPLE_WAV_PATH:-./sample.wav}"

mkdir -p "${PROJECT_DIR}/logs"

cd "${PROJECT_DIR}"

mkdir -p metrics

LATEST_CKPT=$(ls -1 models/unet/checkpoint_epoch_*.pt 2>/dev/null | sort | tail -n 1)
if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoints found in models/unet/ directory."
    exit 1
fi
echo "Using checkpoint: $LATEST_CKPT"

if [ -f "$SAMPLE_WAV_PATH" ]; then
    echo "[$(date)] Running sample inference on $SAMPLE_WAV_PATH"
    cp "$SAMPLE_WAV_PATH" ./sample.wav
    python -m evaluation.Test
else
    echo "Skipping Test because SAMPLE_WAV_PATH ($SAMPLE_WAV_PATH) not found."
fi

for SUBSET in Dev Test; do
    if [ -d "$DATA_ROOT/Mixtures/$SUBSET" ] && [ -d "$DATA_ROOT/Sources/$SUBSET" ]; then
        echo "[$(date)] Evaluating on $SUBSET subset"
        OUT_JSON="metrics/baseline_${SUBSET}_${SLURM_JOB_ID}.json"
        python -m evaluation.evaluate_baseline \
            --dataset-root "$DATA_ROOT" \
            --subset "$SUBSET" \
            --checkpoint "$LATEST_CKPT" \
            --output-json "$OUT_JSON"
    else
        echo "Subset $SUBSET not found under $DATA_ROOT; skipping."
    fi
done

echo "[$(date)] Evaluation complete"
