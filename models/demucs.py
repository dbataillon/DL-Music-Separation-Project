"""
Simplified Demucs-style model for waveform-domain music source separation.

This is a simplified implementation inspired by the original Demucs architecture:
- 1D convolutional encoder-decoder
- Skip connections between encoder and decoder
- Bidirectional LSTM at the bottleneck
- Direct waveform in/out (no spectrogram)

Reference: Défossez et al., "Music Source Separation in the Waveform Domain", 2019
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """1D Conv block: Conv1d -> BatchNorm1d -> ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 4,
        padding: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class DeconvBlock1D(nn.Module):
    """1D Deconv block: ConvTranspose1d -> BatchNorm1d -> ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 8,
        stride: int = 4,
        padding: int = 2,
        output_padding: int = 0,
    ):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.deconv(x)))


class Demucs(nn.Module):
    """
    Simplified Demucs model for waveform source separation.
    
    Architecture:
    - Encoder: 6 conv layers with stride=4, doubling channels each layer
    - Bottleneck: Bidirectional LSTM
    - Decoder: 6 deconv layers with stride=4, halving channels
    - Skip connections from encoder to decoder
    
    Args:
        in_channels: Number of input audio channels (1=mono, 2=stereo)
        out_channels: Number of output audio channels per stem
        num_stems: Number of stems to separate (default 4: drums, bass, other, vocals)
        hidden_channels: Initial hidden channel count (doubles each encoder layer)
        depth: Number of encoder/decoder layers
        lstm_layers: Number of LSTM layers at bottleneck
        kernel_size: Convolution kernel size
        stride: Convolution stride (downsampling factor per layer)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_stems: int = 4,
        hidden_channels: int = 48,
        depth: int = 5,
        lstm_layers: int = 2,
        kernel_size: int = 8,
        stride: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stems = num_stems
        self.depth = depth
        self.stride = stride
        
        # Calculate channel progression
        channels = [hidden_channels * (2 ** i) for i in range(depth + 1)]
        # channels = [48, 96, 192, 384, 768, 1536] for depth=5, hidden=48
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        self.encoder.append(
            ConvBlock1D(in_channels, channels[0], kernel_size, stride, padding=kernel_size // 2 - 1)
        )
        for i in range(1, depth):
            self.encoder.append(
                ConvBlock1D(channels[i - 1], channels[i], kernel_size, stride, padding=kernel_size // 2 - 1)
            )
        
        # Bottleneck LSTM
        lstm_channels = channels[depth - 1]
        self.lstm = nn.LSTM(
            lstm_channels,
            lstm_channels // 2,  # Bidirectional doubles this
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        
        # Decoder layers
        # Skip connections: we concatenate encoder output with decoder output
        # First decoder takes LSTM output + skip from last encoder layer
        # Subsequent decoders take prev decoder output + skip from corresponding encoder
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            # All decoder layers receive concatenated input (decoder + skip)
            dec_in = channels[i] * 2  # Always concat with skip
            dec_out = channels[i - 1]
            self.decoder.append(
                DeconvBlock1D(dec_in, dec_out, kernel_size, stride, padding=kernel_size // 2 - 1)
            )
        
        # Final output layer
        self.final_deconv = nn.ConvTranspose1d(
            channels[0] * 2,  # Skip connection from first encoder
            out_channels * num_stems,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2 - 1,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input waveform, shape (B, in_channels, samples)
            
        Returns:
            Separated stems, shape (B, num_stems, out_channels, samples)
        """
        batch_size = x.shape[0]
        original_length = x.shape[-1]
        
        # Pad to ensure proper reconstruction
        # Total downsampling factor = stride^depth
        factor = self.stride ** self.depth
        padded_length = math.ceil(original_length / factor) * factor
        if padded_length > original_length:
            x = F.pad(x, (0, padded_length - original_length))
        
        # Encoder with skip connections
        skips: List[torch.Tensor] = []
        out = x
        for enc_layer in self.encoder:
            out = enc_layer(out)
            skips.append(out)
        
        # Bottleneck LSTM
        # Reshape for LSTM: (B, C, T) -> (B, T, C)
        out = out.permute(0, 2, 1)
        out, _ = self.lstm(out)
        out = out.permute(0, 2, 1)  # Back to (B, C, T)
        
        # Decoder with skip connections
        # skips = [enc0_out, enc1_out, enc2_out, enc3_out, enc4_out] for depth=5
        # We use skips in reverse: enc4 with dec0, enc3 with dec1, etc.
        for i, dec_layer in enumerate(self.decoder):
            # Use skip from encoder at same level (in reverse order)
            # i=0: use skips[depth-1] = skips[4] (last encoder output)
            # i=1: use skips[depth-2] = skips[3], etc.
            skip_idx = self.depth - 1 - i
            skip = skips[skip_idx]
            
            # Match temporal dimension if needed
            if out.shape[-1] != skip.shape[-1]:
                out = F.interpolate(out, size=skip.shape[-1], mode="linear", align_corners=False)
            
            out = torch.cat([out, skip], dim=1)
            out = dec_layer(out)
        
        # Final layer with first skip connection
        if out.shape[-1] != skips[0].shape[-1]:
            out = F.interpolate(out, size=skips[0].shape[-1], mode="linear", align_corners=False)
        out = torch.cat([out, skips[0]], dim=1)
        out = self.final_deconv(out)
        
        # Match original length
        if out.shape[-1] != original_length:
            out = F.interpolate(out, size=original_length, mode="linear", align_corners=False)
        
        # Reshape to (B, num_stems, out_channels, samples)
        out = out.view(batch_size, self.num_stems, self.out_channels, original_length)
        
        return out


class DemucsLoss(nn.Module):
    """
    Combined loss for Demucs training.
    
    Uses L1 loss on waveforms with optional multi-resolution STFT loss.
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        stft_weight: float = 0.0,
        stft_fft_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.stft_fft_sizes = stft_fft_sizes or [512, 1024, 2048]
        
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred: Predicted stems (B, num_stems, channels, samples)
            target: Ground truth stems (B, num_stems, channels, samples)
            
        Returns:
            Combined loss scalar
        """
        # L1 loss on waveforms
        l1_loss = F.l1_loss(pred, target)
        
        if self.stft_weight <= 0:
            return self.l1_weight * l1_loss
        
        # Multi-resolution STFT loss
        stft_loss = 0.0
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        
        for fft_size in self.stft_fft_sizes:
            hop_size = fft_size // 4
            window = torch.hann_window(fft_size, device=pred.device)
            
            pred_stft = torch.stft(
                pred_flat, fft_size, hop_size, window=window, return_complex=True
            )
            target_stft = torch.stft(
                target_flat, fft_size, hop_size, window=window, return_complex=True
            )
            
            # Magnitude loss
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            stft_loss += F.l1_loss(pred_mag, target_mag)
            
            # Log magnitude loss for low-energy components
            stft_loss += F.l1_loss(
                torch.log1p(pred_mag), torch.log1p(target_mag)
            )
        
        stft_loss /= len(self.stft_fft_sizes) * 2
        
        return self.l1_weight * l1_loss + self.stft_weight * stft_loss


def get_demucs_model(
    in_channels: int = 1,
    num_stems: int = 4,
    hidden_channels: int = 48,
    depth: int = 5,
) -> Demucs:
    """Factory function to create Demucs model with sensible defaults."""
    return Demucs(
        in_channels=in_channels,
        out_channels=in_channels,
        num_stems=num_stems,
        hidden_channels=hidden_channels,
        depth=depth,
        lstm_layers=2,
        kernel_size=8,
        stride=4,
    )


if __name__ == "__main__":
    # Quick test
    model = get_demucs_model(in_channels=1, num_stems=4, hidden_channels=48, depth=5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 16000 * 10)  # 2 batches, mono, 10 seconds at 16kHz
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")  # Should be (2, 4, 1, 160000)
