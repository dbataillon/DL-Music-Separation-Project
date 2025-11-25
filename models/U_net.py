import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNRelu(nn.Module):
    """Standard: Conv2d → BN → ReLU."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class DeconvBNRelu(nn.Module):
    """ConvTranspose2d → BN → ReLU."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, output_padding=0):
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s,
                               padding=p, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class UNetStandard(nn.Module):
    """
    Standard-order PyTorch port of U_net.py (Conv→BN→ReLU).
    Input:  (B, in_channels, H, 128)   # magnitude spectrograms
    Output: (B, num_outputs, H, 128)
    """
    def __init__(
        self,
        num_outputs=4,
        dropout_p=0.4,
        final_activation=None,
        in_channels=1,
    ):
        super().__init__()
        self.final_activation = final_activation  # e.g. 'relu' for TF parity, None for diffusion

        # Encoder
        self.enc1 = nn.Sequential(
            ConvBNRelu(in_channels, 32),
            ConvBNRelu(32, 32),
        )
        self.enc2 = nn.Sequential(
            ConvBNRelu(32, 64), ConvBNRelu(64, 64)
        )
        self.enc3 = nn.Sequential(
            ConvBNRelu(64, 128), ConvBNRelu(128, 128)
        )
        self.enc4 = nn.Sequential(
            ConvBNRelu(128, 256), ConvBNRelu(256, 256)
        )
        self.bottleneck = nn.Sequential(
            ConvBNRelu(256, 512), ConvBNRelu(512, 512)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(p=dropout_p)

        # Decoder
        self.up1 = DeconvBNRelu(512, 256, k=5, s=2, p=2, output_padding=1)
        self.dec2 = nn.Sequential(
            DeconvBNRelu(512, 256), DeconvBNRelu(256, 256)
        )

        self.up3 = DeconvBNRelu(256, 128, k=5, s=2, p=2, output_padding=1)
        self.dec4 = nn.Sequential(
            DeconvBNRelu(256, 128), DeconvBNRelu(128, 128)
        )

        self.up5 = DeconvBNRelu(128, 64, k=5, s=2, p=2, output_padding=1)
        self.dec6 = nn.Sequential(
            DeconvBNRelu(128, 64), DeconvBNRelu(64, 64)
        )

        self.up7 = DeconvBNRelu(64, 32, k=5, s=2, p=2, output_padding=1)
        self.dec8 = nn.Sequential(
            DeconvBNRelu(64, 32), DeconvBNRelu(32, 32)
        )

        self.head = nn.Conv2d(32, num_outputs, kernel_size=1, stride=1, padding=0, bias=True)
        self.head_bn = nn.BatchNorm2d(num_outputs)

    def forward(self, x, training=True):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool(c1)

        c2 = self.enc2(p1)
        p2 = self.pool(c2)

        c3 = self.enc3(p2)
        p3 = self.pool(c3)

        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        b = self.bottleneck(p4)

        # Decoder
        u1 = self.up1(b)
        u1 = F.dropout(u1, p=self.drop.p, training=training)
        d2 = self.dec2(torch.cat([u1, c4], dim=1))

        u3 = self.up3(d2)
        u3 = F.dropout(u3, p=self.drop.p, training=training)
        d4 = self.dec4(torch.cat([u3, c3], dim=1))

        u5 = self.up5(d4)
        u5 = F.dropout(u5, p=self.drop.p, training=training)
        d6 = self.dec6(torch.cat([u5, c2], dim=1))

        u7 = self.up7(d6)
        u7 = F.dropout(u7, p=self.drop.p, training=training)
        d8 = self.dec8(torch.cat([u7, c1], dim=1))

        out = self.head_bn(self.head(d8))

        if self.final_activation == "relu":
            out = F.relu(out)
        return out
