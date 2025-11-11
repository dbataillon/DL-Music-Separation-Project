#!/bin/bash
#SBATCH --job-name=freqnet-train
#SBATCH --partition=pace-gpu
#SBATCH --qos=pace-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/freqnet-train-%j.out
#SBATCH --error=logs/freqnet-train-%j.err

set -euo pipefail

PROJECT_DIR="/home/hice1/etarr7/scratch/DL-Music-Separation-Project"
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/data/DSD100}"
SAMPLE_WAV_PATH="${SAMPLE_WAV_PATH:-./sample.wav}"

module load anaconda3
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate freqnet

cd "${PROJECT_DIR}"

mkdir -p logs metrics

spectro_count=$(find Spectrogram -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l || true)
if [ "$spectro_count" -eq 0 ]; then
    echo "Spectrogram cache missing. Run slurm/preprocess_spectrograms.sbatch first."
    exit 1
fi

echo "[$(date)] Starting training"
python -m training.Training
echo "[$(date)] Training finished"

LATEST_CKPT=$(ls -1 model/checkpoint_epoch_*.pt | sort | tail -n 1)
echo "Latest checkpoint: $LATEST_CKPT"

if [ -f "$SAMPLE_WAV_PATH" ]; then
    echo "Running sample inference on $SAMPLE_WAV_PATH"
    cp "$SAMPLE_WAV_PATH" ./sample.wav
    python -m evaluation.Test
else
    echo "Skipping Test.py because SAMPLE_WAV_PATH ($SAMPLE_WAV_PATH) not found."
fi

for SUBSET in Dev Test; do
    if [ -d "$DATA_ROOT/Mixtures/$SUBSET" ] && [ -d "$DATA_ROOT/Sources/$SUBSET" ]; then
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

echo "[$(date)] Training job complete"
