"""Sample stems from the diffusion model for a single spectrogram."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from configs.config import patch_size
from models.diffusion_model import DiffusionUNet
from preprocessing.util import LoadAudio, SaveAudio

STEM_NAMES = ["vocal", "bass", "drums", "other"]


def make_cosine_schedule(T=200, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(1e-6, 0.999)
    return betas

def load_mix(path: Path) -> np.ndarray:
    data = np.load(path)
    if "mix" not in data:
        raise ValueError(f"{path} does not contain 'mix'")
    return data["mix"].astype(np.float32)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_latest_checkpoint(model_dir: Path) -> Path:
    ckpts = sorted(model_dir.glob("checkpoint_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return ckpts[-1]


def load_model(checkpoint: Path, device: torch.device) -> DiffusionUNet:
    model = DiffusionUNet().to(device)
    state = torch.load(checkpoint, map_location=device)
    model_state = state.get("model_state_dict", state)
    has_module = any(k.startswith("module.") for k in model_state.keys())
    target_state = model.state_dict()
    if has_module:
        # checkpoint from DataParallel
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        print(f"Warning: missing keys in checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    model.eval()
    return model


def prepare_mix_batches(mix: np.ndarray, stride: int) -> tuple:
    height, width = mix.shape
    usable = height - 1  # drop DC row to match training patches
    mix_no_dc = mix[1:, :]
    num_segments = math.ceil(width / stride)

    patches = []
    segments = []
    for idx in range(num_segments):
        start = idx * stride
        end = min(start + stride, width)
        patch = mix_no_dc[:, start:end]
        if patch.shape[1] < stride:
            pad = np.zeros((usable, stride - patch.shape[1]), dtype=patch.dtype)
            patch = np.concatenate([patch, pad], axis=1)
        patches.append(patch[None, None, :, :])  # (1,1,512,stride)
        segments.append((start, end))
    batch = np.concatenate(patches, axis=0)
    return batch, segments, width


def ddim_step(model, x, mix, t, betas, alphas, alphas_bar):
    beta_t = betas[t]
    alpha_t = alphas[t]
    alpha_bar_t = alphas_bar[t]
    t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)

    eps_theta = model(x, mix, t_tensor)
    coeff1 = 1.0 / torch.sqrt(alpha_t)
    coeff2 = beta_t / torch.sqrt(1 - alpha_bar_t)
    mean = coeff1 * (x - coeff2 * eps_theta)
    if t > 0:
        noise = torch.randn_like(x)
        sigma = torch.sqrt(beta_t)
        x = mean + sigma * noise
    else:
        x = mean
    return x


def sample_stems(model, mix_batch, betas):
    device = next(model.parameters()).device
    mix_t = torch.from_numpy(mix_batch).float().to(device)
    B, _, H, W = mix_t.shape

    betas = betas.to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    x = torch.randn((B, 4, H, W), device=device)
    T = betas.shape[0]
    model.eval()
    with torch.no_grad():
        for t in reversed(range(T)):
            x = ddim_step(model, x, mix_t, t, betas, alphas, alphas_bar)
    return x.cpu().numpy()


def stitch_predictions(pred, segments, width):
    B = pred.shape[0]
    result = np.zeros((4, 513, width), dtype=np.float32)
    for i in range(B):
        start, end = segments[i]
        span = end - start
        result[:, 1:, start:end] = pred[i, :, :, :span]
    return result


def save_audio_predictions(pred, mix_audio_path: Path, out_dir: Path):
    if not mix_audio_path.exists():
        raise FileNotFoundError(f"Mixture audio file {mix_audio_path} not found")
    mix_mag, mix_phase = LoadAudio(str(mix_audio_path))
    min_width = min(mix_phase.shape[1], pred.shape[2])
    if min_width != pred.shape[2]:
        pred = pred[:, :, :min_width]
        print(
            f"Warning: mixture phase shorter than prediction; truncating predictions to {min_width} frames"
        )
    mix_phase = mix_phase[:, :min_width]
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, stem in enumerate(STEM_NAMES):
        mag = pred[idx]
        SaveAudio(str(out_dir / f"{stem}.wav"), mag, mix_phase)


def main():
    parser = argparse.ArgumentParser(description="Run diffusion sampling for one spectrogram")
    parser.add_argument("spectrogram", type=str, help="Path to .npz spectrogram (with 'mix')")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to diffusion checkpoint")
    parser.add_argument("--model-dir", type=str, default="model/diffusion", help="Directory with checkpoints")
    parser.add_argument("--output", type=str, default=None, help="Path to save predicted stems (.npz)")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of diffusion steps (T)")
    parser.add_argument(
        "--mixture-audio",
        type=str,
        default=None,
        help="Optional path to mixture WAV for saving stems as audio",
    )
    parser.add_argument(
        "--audio-out-dir",
        type=str,
        default=None,
        help="Directory to write WAV stems (requires --mixture-audio)",
    )
    args = parser.parse_args()

    spec_path = Path(args.spectrogram)
    mix = load_mix(spec_path)
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = Path(args.checkpoint) if args.checkpoint else get_latest_checkpoint(Path(args.model_dir))
    print(f"Loading checkpoint: {checkpoint}")

    model = load_model(checkpoint, device)

    betas = make_cosine_schedule(T=args.timesteps)
    mix_batch, segments, width = prepare_mix_batches(mix, patch_size)
    preds = sample_stems(model, mix_batch, betas)
    full = stitch_predictions(preds, segments, width)

    output_path = Path(args.output) if args.output else spec_path.with_name(spec_path.stem + "_diffusion_pred.npz")
    np.savez_compressed(output_path, prediction=full, mix=mix)
    print(f"Saved diffusion prediction to {output_path}")

    if args.mixture_audio:
        mix_audio_path = Path(args.mixture_audio)
        audio_dir = Path(args.audio_out_dir) if args.audio_out_dir else output_path.parent / f"{output_path.stem}_audio"
        print(f"Saving audio stems to {audio_dir}")
        save_audio_predictions(full, mix_audio_path, audio_dir)
        print("Audio stems written.")


if __name__ == "__main__":
    main()
