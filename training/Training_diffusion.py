# Training_diffusion.py - Train diffusion model for music source separation
import gc
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from configs.config import BATCH, EPOCH, LEARNING_RATE
from models.diffusion_model import DiffusionUNet
from preprocessing.util import (
    LoadSpectrogram,
    Magnitude_phase_x,
    Magnitude_phase_y,
    sampling,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model" / "diffusion"


def make_cosine_schedule(T=100, s=0.008):
    """Create cosine noise schedule for diffusion."""
    steps = torch.arange(T + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(1e-6, 0.999)
    return betas  # (T,)


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # great for M1/M2 Macs
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_latest_checkpoint(model_dir: Path) -> tuple:
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
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    """Load checkpoint for resuming training. Returns the epoch number to resume from."""
    state = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel checkpoints - remove 'module.' prefix if present
    model_state = state["model_state_dict"]
    is_dataparallel = any(key.startswith("module.") for key in model_state.keys())

    if is_dataparallel and not isinstance(model, nn.DataParallel):
        # Checkpoint was from DataParallel, but model isn't - remove prefix
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    elif not is_dataparallel and isinstance(model, nn.DataParallel):
        # Checkpoint wasn't from DataParallel, but model is - add prefix
        model_state = {f"module.{k}": v for k, v in model_state.items()}

    model.load_state_dict(model_state)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch = state.get("epoch", 0)
    print(f"Resumed from checkpoint: {checkpoint_path} (epoch {epoch})")
    return epoch


def save_checkpoint(model, optimizer, epoch, loss=None, model_dir: Path = DEFAULT_MODEL_DIR):
    """Save model checkpoint under model/diffusion."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
    latest_alias = model_dir / "checkpoint_latest.pt"
    shutil.copy2(path, latest_alias)
    return path


def get_base_state_dict(model: torch.nn.Module):
    """Return base state_dict regardless of DataParallel wrapping."""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def train_one_epoch(model, loader, optimizer, device, betas, alphas_bar, T):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    n_samples = 0

    for mix_mag, y_clean in loader:
        mix_mag = mix_mag.to(device, non_blocking=True)
        y_clean = y_clean.to(device, non_blocking=True)

        # Sample random timesteps for each item in batch
        bsz = y_clean.size(0)
        t_int = torch.randint(0, T, (bsz,), device=device)
        a_bar = alphas_bar[t_int].view(bsz, 1, 1, 1)  # (B,1,1,1)

        # Add noise according to forward diffusion process
        eps = torch.randn_like(y_clean)
        y_noisy = a_bar.sqrt() * y_clean + (1.0 - a_bar).sqrt() * eps

        # Predict the noise
        eps_hat = model(y_noisy, mix_mag, t_int)  # (B,4,H,W)
        loss = F.mse_loss(eps_hat, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * bsz
        n_samples += bsz

    return running_loss / max(1, n_samples)


def train_diffusion():
    """Main training function for diffusion model."""
    device = get_device()
    print(f"Using device: {device}")

    # Diffusion hyperparameters
    T = 200
    betas = make_cosine_schedule(T=T).to(device)  # (T,)
    alphas = 1.0 - betas  # (T,)
    alphas_bar = torch.cumprod(alphas, dim=0)  # (T,)

    # Initialize model and optimizer
    net = DiffusionUNet().to(device)
    print("number of parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        net = nn.DataParallel(net)

    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCH, eta_min=1e-6)

    # Setup checkpoint directory
    model_dir = DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # File to track training loss over epochs
    loss_log_path = model_dir / "training_loss.json"

    # Try to resume from latest checkpoint
    start_epoch = 1
    latest_ckpt_path, latest_epoch = get_latest_checkpoint(model_dir)
    if latest_ckpt_path is not None:
        try:
            start_epoch = (
                load_checkpoint_for_resume(net, opt, str(latest_ckpt_path), device) + 1
            )
            # Load existing loss history
            if loss_log_path.exists():
                with open(loss_log_path, "r") as f:
                    loss_history = json.load(f)
            else:
                loss_history = {}
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Starting training from epoch 1")
            loss_history = {}
    else:
        print("No checkpoint found. Starting fresh training from epoch 1")
        loss_history = {}

    # Data prep (same utilities as baseline)
    print("Loading spectrograms...")
    X_list, Y_list = LoadSpectrogram()
    X_mag, _ = Magnitude_phase_x(X_list)
    Y_mag, _ = Magnitude_phase_y(Y_list)

    if not X_mag or not Y_mag:
        raise RuntimeError("Spectrogram cache is empty after preprocessing.")
    sample_mix_shape = X_mag[0].shape if hasattr(X_mag[0], "shape") else None
    sample_stem_shape = (
        len(Y_mag[0]),
        getattr(Y_mag[0][0], "shape", None),
    )

    # Clear what we don't need
    del X_list, Y_list
    gc.collect()

    print(
        "Loaded {n_mix} mixtures with mix shape {mix_shape} and stems {stem_shape}".format(
            n_mix=len(X_mag), mix_shape=sample_mix_shape, stem_shape=sample_stem_shape
        )
    )

    # Training loop
    for epoch in range(start_epoch, EPOCH + 1):
        t0 = time.time()

        # Sample data for this epoch (same as baseline approach)
        X_np, Y_np = sampling(X_mag, Y_mag)  # same sampling you used before

        # Convert sampled numpy arrays to torch tensors in NCHW format
        if X_np.ndim == 3:
            X_np = X_np[:, None, :, :]
        elif X_np.ndim == 4 and X_np.shape[-1] == 1:
            X_np = np.transpose(X_np, (0, 3, 1, 2))
        else:
            raise ValueError(f"Unexpected X_np shape {X_np.shape}")

        if Y_np.ndim == 4 and Y_np.shape[-1] == 4:
            Y_np = np.transpose(Y_np, (0, 3, 1, 2))
        elif Y_np.ndim == 4 and Y_np.shape[1] == 4:
            pass
        else:
            raise ValueError(f"Unexpected Y_np shape {Y_np.shape}")

        X_t = torch.from_numpy(X_np).float()
        Y_t = torch.from_numpy(Y_np).float()

        # Clear numpy arrays to free memory
        del X_np, Y_np
        gc.collect()

        # Create dataloader
        ds = TensorDataset(X_t, Y_t)
        dl = DataLoader(
            ds,
            batch_size=BATCH,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        # Train for one epoch
        avg_loss = train_one_epoch(net, dl, opt, device, betas, alphas_bar, T)

        # Clear tensors from this epoch
        del ds, dl, X_t, Y_t
        gc.collect()

        # Save checkpoint
        ckpt_path = save_checkpoint(net, opt, epoch, loss=avg_loss, model_dir=model_dir)

        # Save loss to JSON log
        loss_history[f"epoch_{epoch:04d}"] = {
            "epoch": epoch,
            "loss": float(avg_loss),
        }
        with open(loss_log_path, "w") as f:
            json.dump(loss_history, f, indent=2)

        # Step learning rate scheduler
        scheduler.step()

        dt = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"[Epoch {epoch:03d}/{EPOCH}] loss={avg_loss:.6f} | lr={current_lr:.2e} | saved: {ckpt_path} | {dt:.1f}s"
        )

    final_model_path = model_dir / f"diffusion_model_final_epoch_{EPOCH:04d}.pt"
    torch.save(get_base_state_dict(net), final_model_path)
    print(f"Saved final diffusion weights to {final_model_path}")

    print("Diffusion training complete!")


if __name__ == "__main__":
    train_diffusion()
