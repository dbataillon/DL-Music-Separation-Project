# Training_demucs.py - Train waveform-domain Demucs model for music source separation
"""
Trains a simplified Demucs model on raw audio waveforms.
Uses DSD100 dataset structure with mixtures and source stems.
"""
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import BATCH, EPOCH, LEARNING_RATE, SR
from models.demucs import Demucs, DemucsLoss, get_demucs_model

DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "demucs"
STEM_NAMES = ["bass", "drums", "other", "vocals"]


class WaveformDataset(Dataset):
    """
    Dataset for loading raw audio waveforms from DSD100.
    
    Loads mixture and corresponding stem waveforms, returns random chunks.
    Samples multiple chunks per track per epoch for small datasets.
    Supports multiple dataset roots to combine DSD100 and DSD100subset.
    """
    
    def __init__(
        self,
        dataset_roots: List[str],  # Support multiple dataset roots
        subset: str = "Dev",
        sample_rate: int = 16000,
        chunk_duration: float = 6.0,  # seconds
        stems: List[str] = STEM_NAMES,
        chunks_per_track: int = 50,  # Sample multiple chunks per track per epoch
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.stems = stems
        self.chunks_per_track = chunks_per_track
        
        # Build track list from all dataset roots
        self.tracks: List[Tuple[str, str]] = []
        seen_tracks = set()  # Avoid duplicates
        
        for dataset_root in dataset_roots:
            mix_base = os.path.join(dataset_root, "Mixtures", subset)
            src_base = os.path.join(dataset_root, "Sources", subset)
            
            if os.path.isdir(mix_base) and os.path.isdir(src_base):
                for track in sorted(os.listdir(mix_base)):
                    if track in seen_tracks:
                        continue  # Skip duplicates
                    mix_path = os.path.join(mix_base, track, "mixture.wav")
                    stem_dir = os.path.join(src_base, track)
                    if os.path.isfile(mix_path) and os.path.isdir(stem_dir):
                        self.tracks.append((mix_path, stem_dir))
                        seen_tracks.add(track)
        
        if not self.tracks:
            raise RuntimeError(f"No tracks found in {dataset_roots}/{subset}")
        
        print(f"WaveformDataset: Found {len(self.tracks)} tracks in {subset}")
        print(f"WaveformDataset: {self.chunks_per_track} chunks per track = {len(self)} samples per epoch")
        
        # Cache loaded audio for efficiency
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_enabled = True
    
    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file, resample if needed, convert to mono."""
        if self._cache_enabled and path in self._cache:
            return self._cache[path]
        
        try:
            import torchaudio
            waveform, sr = torchaudio.load(path)
        except Exception:
            # Fallback to librosa
            import librosa
            waveform, sr = librosa.load(path, sr=None, mono=False)
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            waveform = torch.from_numpy(waveform)
        
        # Resample if needed
        if sr != self.sample_rate:
            import torchaudio.functional as F_audio
            waveform = F_audio.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if self._cache_enabled:
            self._cache[path] = waveform
        
        return waveform
    
    def __len__(self) -> int:
        # Return total number of chunks across all tracks
        return len(self.tracks) * self.chunks_per_track
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mix: (1, chunk_samples) mixture waveform
            stems: (num_stems, 1, chunk_samples) source waveforms
        """
        # Map idx to track (allows multiple chunks per track)
        track_idx = idx % len(self.tracks)
        mix_path, stem_dir = self.tracks[track_idx]
        
        # Load mixture
        mix = self._load_audio(mix_path)  # (1, total_samples)
        total_samples = mix.shape[-1]
        
        # Load stems
        stems_list = []
        for stem_name in self.stems:
            stem_path = os.path.join(stem_dir, f"{stem_name}.wav")
            if os.path.isfile(stem_path):
                stem = self._load_audio(stem_path)
            else:
                # Create silent stem if missing
                stem = torch.zeros_like(mix)
            stems_list.append(stem)
        
        # Stack stems: (num_stems, 1, total_samples)
        stems = torch.stack(stems_list, dim=0)
        
        # Ensure all have same length
        min_len = min(mix.shape[-1], stems.shape[-1])
        mix = mix[..., :min_len]
        stems = stems[..., :min_len]
        total_samples = min_len
        
        # Random chunk
        if total_samples > self.chunk_samples:
            start = random.randint(0, total_samples - self.chunk_samples)
            mix = mix[..., start:start + self.chunk_samples]
            stems = stems[..., start:start + self.chunk_samples]
        elif total_samples < self.chunk_samples:
            # Pad if too short
            pad_len = self.chunk_samples - total_samples
            mix = F.pad(mix, (0, pad_len))
            stems = F.pad(stems, (0, pad_len))
        
        return mix, stems  # (1, chunk), (4, 1, chunk)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_latest_checkpoint(model_dir: Path) -> Tuple[Optional[Path], int]:
    """Get the latest checkpoint from model_dir. Returns (path, epoch) or (None, 0)."""
    if not model_dir.exists():
        return None, 0

    checkpoint_files = sorted(model_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None, 0

    latest_ckpt = checkpoint_files[-1]
    match = re.search(r"checkpoint_epoch_(\d+)", latest_ckpt.name)
    if match:
        epoch = int(match.group(1))
        return latest_ckpt, epoch

    return None, 0


def load_checkpoint_for_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    """Load checkpoint for resuming training. Returns the epoch number to resume from."""
    state = torch.load(checkpoint_path, map_location=device)

    model_state = state["model_state_dict"]
    is_dataparallel = any(key.startswith("module.") for key in model_state.keys())

    if is_dataparallel and not isinstance(model, nn.DataParallel):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    elif not is_dataparallel and isinstance(model, nn.DataParallel):
        model_state = {f"module.{k}": v for k, v in model_state.items()}

    model.load_state_dict(model_state)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state.get("epoch", 0)
    print(f"Resumed from checkpoint: {checkpoint_path} (epoch {epoch})")
    return epoch


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: Optional[float] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
):
    """Save model checkpoint."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    
    model_state = model.state_dict()
    if isinstance(model, nn.DataParallel):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    print(f"Saved checkpoint: {path}")


def save_final_weights(model: nn.Module, model_dir: Path = DEFAULT_MODEL_DIR):
    """Save final model weights only (no optimizer state)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "demucs_final.pt"
    
    model_state = model.state_dict()
    if isinstance(model, nn.DataParallel):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    
    torch.save(model_state, path)
    print(f"Saved final weights: {path}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (mix, stems) in enumerate(dataloader):
        # mix: (B, 1, samples), stems: (B, num_stems, 1, samples)
        mix = mix.to(device)
        stems = stems.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(mix)  # (B, num_stems, 1, samples)
        
        # Compute loss
        loss = criterion(pred, stems)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    return total_loss / max(num_batches, 1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Demucs waveform separation model.")
    parser.add_argument("--dataset-roots", type=str, nargs="+", 
                        default=["data/DSD100"],
                        help="Root directories of DSD100 dataset(s). Can specify multiple.")
    parser.add_argument("--subset", type=str, default="Dev",
                        help="Dataset subset to train on.")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR),
                        help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=EPOCH,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH,
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--chunk-duration", type=float, default=6.0,
                        help="Audio chunk duration in seconds.")
    parser.add_argument("--chunks-per-track", type=int, default=50,
                        help="Number of random chunks to sample per track per epoch.")
    parser.add_argument("--hidden-channels", type=int, default=48,
                        help="Initial hidden channels in model.")
    parser.add_argument("--depth", type=int, default=5,
                        help="Number of encoder/decoder layers.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint.")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs.")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset and dataloader
    print("Loading dataset...")
    print(f"Dataset roots: {args.dataset_roots}")
    dataset = WaveformDataset(
        dataset_roots=args.dataset_roots,
        subset=args.subset,
        sample_rate=SR,
        chunk_duration=args.chunk_duration,
        stems=STEM_NAMES,
        chunks_per_track=args.chunks_per_track,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,  # Don't drop last batch for small datasets
    )
    
    # Create model
    print("Creating model...")
    model = get_demucs_model(
        in_channels=1,
        num_stems=4,
        hidden_channels=args.hidden_channels,
        depth=args.depth,
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = DemucsLoss(l1_weight=1.0, stft_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Resume from checkpoint if requested
    start_epoch = 1
    if args.resume:
        ckpt_path, last_epoch = get_latest_checkpoint(model_dir)
        if ckpt_path:
            start_epoch = load_checkpoint_for_resume(model, optimizer, ckpt_path, device) + 1
            # Advance scheduler
            for _ in range(start_epoch - 1):
                scheduler.step()
    
    # Training loop
    loss_log_path = model_dir / "loss_log.json"
    loss_history: List[Dict] = []
    
    # Load existing loss history if resuming
    if loss_log_path.exists():
        with open(loss_log_path, "r") as f:
            loss_history = json.load(f)
    
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Chunk duration: {args.chunk_duration}s, Sample rate: {SR}Hz")
    print("-" * 60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        avg_loss = train_one_epoch(
            model, dataloader, criterion, optimizer, device, epoch
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Log loss
        loss_history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "lr": current_lr,
            "time": epoch_time,
        })
        with open(loss_log_path, "w") as f:
            json.dump(loss_history, f, indent=2)
        
        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(model, optimizer, epoch, avg_loss, model_dir)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save final weights
    save_final_weights(model, model_dir)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
