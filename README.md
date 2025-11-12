# Music-Separation
This is an implementation of U-Net for vocal, bass, drums separation with tensorflow

## Requirement
- librosa==0.6.2
- numpy==1.14.3
- tensorflow==1.13.0
- python==3.6.5

## Download Dataset
I download [dsd100](https://sigsep.github.io/datasets/dsd100.html) dataset.
<pre><code>$ python -m preprocessing.download_data --DATADIR ./data </code></pre>

## Data
I prepare CCMixter datasets in "./data" and Each track consisted of Mixed, bass, drums, other, vocal version
<pre><code>$ python -m preprocessing.CCMixter_process --DATADIR ./data </code></pre>

## Usage
- Train (run everything from repo root so package imports resolve)
<pre><code>$ python -m training.train_baseline</code></pre>
- Test
<pre><code>$ python -m evaluation.Test</code></pre>

## SLURM Workflow for Training and Evaluation

To train the baseline U-Net model and evaluate it, follow these steps using the SLURM job submission system:

### **Step 1: Download and Preprocess Data**
```bash
sbatch slurm/download_data.sbatch          # Downloads DSD100 dataset (~100GB)
sbatch slurm/preprocess_spectrograms.sbatch # Creates spectrogram cache (required before training)
```

### **Step 2: Train the Baseline Model**
```bash
sbatch slurm/train_baseline.sbatch  # Trains U-Net on spectrograms
```
- **Duration**: ~12 hours for 300 epochs
- **Checkpoints**: Saved to `models/unet/checkpoint_epoch_XXXX.pt`
- **Loss log**: Saved to `models/unet/training_loss.json`
- **Resume**: If job times out, resubmit the same command to resume from latest checkpoint
- **Monitor progress**: `tail -f logs/freqnet-unet-train-progress.log`

### **Step 3: Evaluate Single Checkpoint**
```bash
# After training completes, evaluate the latest checkpoint on Dev subset
sbatch slurm/evaluate_baseline.sbatch
```
- Evaluates on Dev subset and outputs metrics to JSON
- Uses the most recent checkpoint automatically
- **Duration**: ~4 hours

### **Step 4 (Optional): Evaluate All Checkpoints**
```bash
# Evaluate all trained epochs in parallel (uses 2 GPUs)
sbatch slurm/evaluate_all_epochs.sbatch
```
- Evaluates all checkpoints across training history
- Shows progression of metrics over epochs
- Parallelized with 2 workers for faster evaluation (~6 hours instead of 12)
- Outputs comprehensive JSON with all epoch metrics and training loss

### **Complete Workflow Summary**
```
1. sbatch slurm/download_data.sbatch           (one time, 2-3 hrs)
   ↓ (wait for completion)
2. sbatch slurm/preprocess_spectrograms.sbatch (one time, pretty quick)
   ↓ (wait for completion)
3. sbatch slurm/train_baseline.sbatch          (12 hours)
   ↓ (can resubmit if interrupted)
4. sbatch slurm/evaluate_baseline.sbatch       (4 hours? (haven't tested yet))
   OR
   sbatch slurm/evaluate_all_epochs.sbatch     (6 hours? - parallel (haven't tested yet))
```

### **Output Files**
- **Checkpoints**: `models/unet/checkpoint_epoch_*.pt`
- **Training loss**: `models/unet/training_loss.json`
- **Evaluation metrics**: `metrics/baseline_*.json` or `metrics/all_epochs_eval_*.json`
- **Progress logs**: `logs/freqnet-unet-train-progress.log`, `logs/freqnet-unet-eval-all-progress.log`

## Paper
Jaehoon Oh et al. [spectrogram-channels u-net: a source separation model viewing each channel as the spectrogram of each source](https://arxiv.org/abs/1810.11520)

## Base Implementation
* https://github.com/Jeongseungwoo/Singing-Voice-Separation

## To Do List
* convert wav files to mp3 files
* make tfrecord format files
