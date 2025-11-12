# train_baseline.py - Train baseline U-Net model for music source separation
import os
import time
import gc
import json
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Your utilities & config (same ones the TF script used)
from preprocessing.util import (
    LoadSpectrogram,
    Magnitude_phase_x,
    Magnitude_phase_y,
    sampling,
)
from configs.config import EPOCH, BATCH, LEARNING_RATE, image_width

# Your PyTorch U-Net (standard Conv->BN->ReLU order)
from models.U_net import UNetStandard

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")     # great for M1/M2 Macs
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_latest_checkpoint(model_dir: str) -> tuple:
    """Get the latest checkpoint from model_dir. Returns (path, epoch) or (None, 0)."""
    if not os.path.exists(model_dir):
        return None, 0
    
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith(".pt") and f.startswith("checkpoint_epoch_")]
    if not checkpoint_files:
        return None, 0
    
    # Sort by epoch number and get the latest
    latest_ckpt = sorted(checkpoint_files)[-1]
    ckpt_path = os.path.join(model_dir, latest_ckpt)
    
    # Extract epoch number from filename
    match = re.search(r"checkpoint_epoch_(\d+)", latest_ckpt)
    if match:
        epoch = int(match.group(1))
        return ckpt_path, epoch
    
    return None, 0


def load_checkpoint_for_resume(model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str, device: torch.device) -> int:
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

def numpy_to_torch_batch(X_np, Y_np):
    """
    TF code reshaped to [-1, image_width, 128, 1] (NHWC).
    Our model expects NCHW: (B, 1, image_width, 128).
    This function converts numpy arrays to torch tensors in the right shape.
    """
    # Expect X_np: (N, image_width, 128) or (N, image_width, 128, 1)
    if X_np.ndim == 3:
        X_np = X_np[:, None, :, :]  # (N, 1, H, W)
    elif X_np.ndim == 4 and X_np.shape[-1] == 1:
        X_np = np.transpose(X_np, (0, 3, 1, 2))  # NHWC -> NCHW
    else:
        raise ValueError(f"Unexpected X shape: {X_np.shape}")

    # Y_np: (N, image_width, 128, 4) in TF -> want (N, 4, H, W)
    if Y_np.ndim == 4 and Y_np.shape[-1] == 4:
        Y_np = np.transpose(Y_np, (0, 3, 1, 2))  # NHWC -> NCHW
    elif Y_np.ndim == 4 and Y_np.shape[1] == 4:
        # already NCHW
        pass
    else:
        raise ValueError(f"Unexpected Y shape: {Y_np.shape}")

    X = torch.from_numpy(X_np).float()
    Y = torch.from_numpy(Y_np).float()
    return X, Y

def save_checkpoint(model, optimizer, epoch, loss=None, model_dir="./models/unet"):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    return path

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.L1Loss()  # same as tf.losses.absolute_difference
    epoch_loss = 0.0
    n_samples = 0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        Y_batch = Y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        Y_hat = model(X_batch, training=True)  # keep dropout on during training
        loss = criterion(Y_hat, Y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        epoch_loss += loss.item() * batch_size
        n_samples += batch_size

    return epoch_loss / max(1, n_samples)

def main():
    device = get_device()
    print(f"Using device: {device}")

    # === Data prep (same utilities you used in TF) ===
    # X_list (mixture), Y_list (stems)
    X_list, Y_list = LoadSpectrogram()
    X_mag, X_phase = Magnitude_phase_x(X_list)
    Y_mag, _       = Magnitude_phase_y(Y_list)
    
    # Clear X_phase since we don't need it during training
    del X_list, X_phase
    gc.collect()

    # === Model ===
    # final_activation="relu" to match TF training on non-negative magnitudes.
    # (When you move to diffusion, set final_activation=None and change the loss/objective.)
    model = UNetStandard(num_outputs=4, dropout_p=0.4, final_activation="relu").to(device)
    
    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_dir = "./models/unet"
    os.makedirs(model_dir, exist_ok=True)
    
    # File to track training loss over epochs
    loss_log_path = os.path.join(model_dir, "training_loss.json")
    
    # Try to resume from latest checkpoint
    start_epoch = 1
    latest_ckpt_path, latest_epoch = get_latest_checkpoint(model_dir)
    if latest_ckpt_path is not None:
        try:
            start_epoch = load_checkpoint_for_resume(model, optimizer, latest_ckpt_path, device) + 1
            # Load existing loss history
            if os.path.exists(loss_log_path):
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

    # === Training loop (mirrors TF's epoch loop that calls sampling() per epoch) ===
    for epoch in range(start_epoch, EPOCH + 1):
        t0 = time.time()

        # TF code did: X, y = sampling(X_mag, Y_mag) per epoch, then 1 epoch over that sample
        X_np, Y_np = sampling(X_mag, Y_mag)  # expected numpy arrays
        # Convert to tensors with correct shape (NCHW)
        X_t, Y_t = numpy_to_torch_batch(X_np, Y_np)
        
        # Clear numpy arrays to free memory
        del X_np, Y_np
        gc.collect()

        dataset = TensorDataset(X_t, Y_t)
        # Disable pin_memory to reduce RAM usage
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, pin_memory=False)

        avg_loss = train_one_epoch(model, loader, optimizer, device)
        
        # Clear tensors from this epoch
        del dataset, loader, X_t, Y_t
        gc.collect()

        ckpt_path = save_checkpoint(model, optimizer, epoch, loss=avg_loss, model_dir=model_dir)
        
        # Save loss to JSON log
        loss_history[f"epoch_{epoch:04d}"] = {
            "epoch": epoch,
            "loss": float(avg_loss),
        }
        with open(loss_log_path, "w") as f:
            json.dump(loss_history, f, indent=2)
        
        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f} | saved: {ckpt_path} | {dt:.1f}s")

    print("Training Complete!!")

if __name__ == "__main__":
    main()
