#!/bin/bash
#SBATCH --job-name=freqnet_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# === User-configurable paths ===
# Point DATA_ROOT at the directory that contains the DSD100 download
# (with subdirectories Mixtures/ and Sources/).
: "${DATA_ROOT:?Set DATA_ROOT to your DSD100 root, e.g. export DATA_ROOT=/storage/DSD100}"

# Optional: supply a mix wav for Test.py (defaults to ./sample.wav if unset).
SAMPLE_WAV_PATH="${SAMPLE_WAV_PATH:-./sample.wav}"

# Create log directory if it does not exist (safe for shared filesystems).
mkdir -p logs

echo "Job $SLURM_JOB_ID launched on $(date)"
echo "Using DATA_ROOT=$DATA_ROOT"

# === Environment setup ===
module purge
module load python/3.9

VENV_DIR="${SCRATCH:-$PWD}/freqnet_env"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt

# === Data preparation (generate spectrogram cache if missing) ===
if [ ! -d Spectrogram ] || [ -z "$(find Spectrogram -maxdepth 1 -type f -name '*.npz' 2>/dev/null)" ]; then
    echo "Spectrogram cache missing; generating from DSD100 (this may take a while)..."
    python CCMixter_process.py --DATADIR "$DATA_ROOT"
else
    echo "Found existing Spectrogram cache; skipping generation."
fi

# === Training ===
echo "Starting training at $(date)"
python Training.py
echo "Training finished at $(date)"

# Capture latest checkpoint for evaluation and testing.
LATEST_CKPT=$(ls -1 model/checkpoint_epoch_*.pt | sort | tail -n 1)
echo "Latest checkpoint: $LATEST_CKPT"

# === Optional sample inference ===
if [ -f "$SAMPLE_WAV_PATH" ]; then
    echo "Running Test.py on $SAMPLE_WAV_PATH"
    cp "$SAMPLE_WAV_PATH" ./sample.wav
    python Test.py
else
    echo "Skipping Test.py because SAMPLE_WAV_PATH ($SAMPLE_WAV_PATH) not found."
fi

# === Evaluation metrics (Dev and Test splits) ===
METRIC_DIR="metrics"
mkdir -p "$METRIC_DIR"

for SUBSET in Dev Test; do
    if [ -d "$DATA_ROOT/Mixtures/$SUBSET" ] && [ -d "$DATA_ROOT/Sources/$SUBSET" ]; then
        OUT_JSON="$METRIC_DIR/baseline_${SUBSET}_${SLURM_JOB_ID}.json"
        echo "Evaluating subset $SUBSET -> $OUT_JSON"
        python evaluate_baseline.py \
            --dataset-root "$DATA_ROOT" \
            --subset "$SUBSET" \
            --checkpoint "$LATEST_CKPT" \
            --output-json "$OUT_JSON"
    else
        echo "Subset $SUBSET not found under $DATA_ROOT; skipping."
    fi
done

echo "Job $SLURM_JOB_ID completed at $(date)"
