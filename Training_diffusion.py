import torch, math, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from diffusion_model import DiffusionUNet
from util import LoadSpectrogram, Magnitude_phase_x, Magnitude_phase_y, sampling
from configs.config import EPOCH, BATCH, LEARNING_RATE

def make_cosine_schedule(T=1000, s=0.008):
    steps = torch.arange(T+1, dtype=torch.float32)
    alphas_cumprod = torch.cos(( (steps / T) + s) / (1+s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(1e-6, 0.999)
    return betas  # (T,)

def train_diffusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 1000
    betas = make_cosine_schedule(T=T).to(device)             # (T,)
    alphas = 1.0 - betas                                     # (T,)
    alphas_bar = torch.cumprod(alphas, dim=0)                # (T,)

    net = DiffusionUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Data prep (same utilities)
    X_list, Y_list = LoadSpectrogram()
    X_mag, _ = Magnitude_phase_x(X_list)    # (N, H, W)
    Y_mag, _ = Magnitude_phase_y(Y_list)    # (N, H, W, 4)

    for epoch in range(1, EPOCH+1):
        X_np, Y_np = sampling(X_mag, Y_mag)  # same sampling you used before
        # Shapes to tensors
        # X_np: (B,H,W) —> (B,1,H,W)
        # Y_np: (B,H,W,4) —> (B,4,H,W)
        X_t = torch.from_numpy(X_np[:, None, :, :]).float().to(device)
        Y_t = torch.from_numpy(np.transpose(Y_np, (0,3,1,2))).float().to(device)

        ds = TensorDataset(X_t, Y_t)
        dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

        net.train()
        running = 0.0
        for mix_mag, y_clean in dl:
            # Sample random t for each item
            bsz = y_clean.size(0)
            t_int = torch.randint(0, T, (bsz,), device=device)
            a_bar = alphas_bar[t_int].view(bsz, 1, 1, 1)     # (B,1,1,1)
            # Add noise
            eps = torch.randn_like(y_clean)
            y_noisy = a_bar.sqrt() * y_clean + (1.0 - a_bar).sqrt() * eps

            # Predict eps
            eps_hat = net(y_noisy, mix_mag, t_int)           # (B,4,H,W)
            loss = F.mse_loss(eps_hat, eps)

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item() * bsz

        print(f"[{epoch}] loss={running / len(ds):.6f}")

if __name__ == "__main__":
    train_diffusion()
