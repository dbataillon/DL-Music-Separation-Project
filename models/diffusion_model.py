# diffusion_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.U_net import UNetStandard  # uses Conv->BN->ReLU order

# ---------------------------
# Utilities: timestep embeds
# ---------------------------
def sinusoidal_time_embed(timesteps, dim):
    """
    timesteps: (B,) int or float in [0, T-1]
    returns: (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / (half - 1))
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class TimeMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------------------
# Diffusion wrapper around your U-Net
# ---------------------------------------
class DiffusionUNet(nn.Module):
    """
    - Input: concat([y_noisy(4ch), mix_mag(1ch)]) => 5 channels
    - Output: epsilon hat for the 4 stem channels
    - Timestep embedding injected via small FiLM-like heads
    """
    def __init__(self, base_channels=32, t_dim=256):
        super().__init__()
        # Reuse your UNetStandard but change the first conv's in_channels
        # Easiest way: create a UNet that *expects* 5 inputs by replacing enc1's first conv.
        # Simpler: wrap UNetStandard and prepend a 1x1 "adapter" that maps 5->1 channels,
        # but better is to make UNetStandard configurable. Here we do a quick adapter:
        self.adapter = nn.Conv2d(5, 1, kernel_size=1, bias=True)
        self.backbone = UNetStandard(num_outputs=4, dropout_p=0.0, final_activation=None)

        # Timestep embedding (sinusoidal -> MLP)
        self.t_embed_dim = t_dim
        self.time_mlp = TimeMLP(in_dim=t_dim, hidden_dim=t_dim, out_dim=t_dim)

        # Light FiLM: project t-embed to a few layers’ channel dims and add as bias
        # We tap into key stages by storing small 1x1 convs to produce per-channel bias.
        # Choose points: after enc1, enc2, enc3, enc4, bottleneck, dec blocks.
        # For simplicity and to avoid modifying backbone internals, we add a few
        # t-projected biases and add them around those blocks' outputs via hooks here.
        # Minimal explicit set:
        self.film_enc1 = nn.Linear(t_dim, 32)    # matches backbone.enc1 out-ch
        self.film_enc2 = nn.Linear(t_dim, 64)
        self.film_enc3 = nn.Linear(t_dim, 128)
        self.film_enc4 = nn.Linear(t_dim, 256)
        self.film_bot  = nn.Linear(t_dim, 512)
        self.film_dec8 = nn.Linear(t_dim, 32)    # final decoder block out-ch

    def apply_film(self, x, film_proj):
        b, c, h, w = x.shape
        bias = film_proj(self._t)[:, :, None, None]  # (B,C,1,1)
        # broadcast add:
        return x + bias

    def forward(self, y_noisy_4ch, mix_mag_1ch, t_index):
        """
        y_noisy_4ch: (B,4,H,W)
        mix_mag_1ch: (B,1,H,W) magnitude spectrogram of mixture
        t_index: (B,) integer or float in [0..T-1]
        returns eps_hat: (B,4,H,W)
        """
        # Build & cache time embedding
        t_emb = sinusoidal_time_embed(t_index, self.t_embed_dim)  # (B,t_dim)
        self._t = self.time_mlp(t_emb)  # (B,t_dim)

        x_in = torch.cat([y_noisy_4ch, mix_mag_1ch], dim=1)  # (B,5,H,W)
        x_in = self.adapter(x_in)  # (B,1,H,W) -> “drop-in” to your backbone

        # ---- Encoder 1 ----
        c1 = self.backbone.enc1(x_in)       # (B,32,H,W)
        c1 = self.apply_film(c1, self.film_enc1)
        p1 = self.backbone.pool(c1)

        # ---- Encoder 2 ----
        c2 = self.backbone.enc2(p1)         # (B,64,H/2,W/2)
        c2 = self.apply_film(c2, self.film_enc2)
        p2 = self.backbone.pool(c2)

        # ---- Encoder 3 ----
        c3 = self.backbone.enc3(p2)         # (B,128,H/4,W/4)
        c3 = self.apply_film(c3, self.film_enc3)
        p3 = self.backbone.pool(c3)

        # ---- Encoder 4 ----
        c4 = self.backbone.enc4(p3)         # (B,256,H/8,W/8)
        c4 = self.apply_film(c4, self.film_enc4)
        p4 = self.backbone.pool(c4)

        # ---- Bottleneck ----
        b  = self.backbone.bottleneck(p4)   # (B,512,H/16,W/16)
        b  = self.apply_film(b, self.film_bot)

        # ---- Decoder (reuse your modules) ----
        u1 = self.backbone.up1(b)
        d2 = self.backbone.dec2(torch.cat([u1, c4], dim=1))

        u3 = self.backbone.up3(d2)
        d4 = self.backbone.dec4(torch.cat([u3, c3], dim=1))

        u5 = self.backbone.up5(d4)
        d6 = self.backbone.dec6(torch.cat([u5, c2], dim=1))

        u7 = self.backbone.up7(d6)
        d8 = self.backbone.dec8(torch.cat([u7, c1], dim=1))  # (B,32,H,W)
        d8 = self.apply_film(d8, self.film_dec8)

        out = self.backbone.head_bn(self.backbone.head(d8))  # (B,4,H,W), linear
        return out  # eps_hat
