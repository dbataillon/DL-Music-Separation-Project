# train_pytorch.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Your utilities & config (same ones the TF script used)
from data.util import LoadSpectrogram, Magnitude_phase_x, Magnitude_phase_y, sampling
from configs.config import EPOCH, BATCH, LEARNING_RATE, image_width

# Your PyTorch U-Net (standard Conv->BN->ReLU order)
from models.U_net import UNetStandard

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")     # great for M1/M2 Macs
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

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

def save_checkpoint(model, optimizer, epoch, model_dir="./model"):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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

    # === Model ===
    # final_activation="relu" to match TF training on non-negative magnitudes.
    # (When you move to diffusion, set final_activation=None and change the loss/objective.)
    model = UNetStandard(num_outputs=4, dropout_p=0.4, final_activation="relu").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model_dir = "./model"
    os.makedirs(model_dir, exist_ok=True)

    # === Training loop (mirrors TF's epoch loop that calls sampling() per epoch) ===
    for epoch in range(1, EPOCH + 1):
        t0 = time.time()

        # TF code did: X, y = sampling(X_mag, Y_mag) per epoch, then 1 epoch over that sample
        X_np, Y_np = sampling(X_mag, Y_mag)  # expected numpy arrays
        # Convert to tensors with correct shape (NCHW)
        X_t, Y_t = numpy_to_torch_batch(X_np, Y_np)

        dataset = TensorDataset(X_t, Y_t)
        loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, pin_memory=(device.type != "cpu"))

        avg_loss = train_one_epoch(model, loader, optimizer, device)

        ckpt_path = save_checkpoint(model, optimizer, epoch, model_dir=model_dir)
        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.6f} | saved: {ckpt_path} | {dt:.1f}s")

    print("Training Complete!!")

if __name__ == "__main__":
    main()
