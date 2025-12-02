"""Convert predicted spectrograms to audio files for both v-pred and eps-pred models."""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.util import LoadAudio, SaveAudio

# Stem order matches training
STEM_NAMES = ["vocal", "bass", "drums", "other"]


def convert_prediction_to_audio(pred_npz_path: Path, mixture_wav_path: Path, output_dir: Path):
    """Convert a predicted spectrogram .npz file to audio .wav files."""
    # Load prediction
    data = np.load(pred_npz_path)
    pred = data["prediction"]  # Shape: (4, 513, width)
    
    # Load mixture audio to get phase information
    mix_mag, mix_phase = LoadAudio(str(mixture_wav_path))
    
    # Align lengths
    min_width = min(mix_phase.shape[1], pred.shape[2])
    pred = pred[:, :, :min_width]
    mix_phase = mix_phase[:, :min_width]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {pred_npz_path.name} to audio...")
    print(f"  Prediction shape: {pred.shape}")
    print(f"  Phase shape: {mix_phase.shape}")
    
    for idx, stem in enumerate(STEM_NAMES):
        mag = pred[idx]
        out_path = output_dir / f"{stem}.wav"
        SaveAudio(str(out_path), mag, mix_phase)
    
    print(f"  Saved to: {output_dir}")


def main():
    # Paths
    samples_dir = PROJECT_ROOT / "samples_comparison"
    track_name = "055 - Angels In Amplifiers - I'm Alright"
    
    eps_pred_path = samples_dir / f"{track_name}_eps_pred.npz"
    vpred_path = samples_dir / f"{track_name}_vpred.npz"
    
    mixture_wav = PROJECT_ROOT / "data" / "DSD100subset" / "Mixtures" / "Dev" / track_name / "mixture.wav"
    
    # Output directories
    eps_audio_dir = samples_dir / "audio_eps"
    vpred_audio_dir = samples_dir / "audio_vpred"
    
    # Also save ground truth audio stems for comparison
    gt_sources_dir = PROJECT_ROOT / "data" / "DSD100subset" / "Sources" / "Dev" / track_name
    
    if not mixture_wav.exists():
        print(f"Error: Mixture audio not found at {mixture_wav}")
        return
    
    # Convert epsilon prediction
    if eps_pred_path.exists():
        convert_prediction_to_audio(eps_pred_path, mixture_wav, eps_audio_dir)
    else:
        print(f"Warning: {eps_pred_path} not found")
    
    # Convert v-prediction
    if vpred_path.exists():
        convert_prediction_to_audio(vpred_path, mixture_wav, vpred_audio_dir)
    else:
        print(f"Warning: {vpred_path} not found")
    
    print("\nDone! Audio files saved to:")
    print(f"  Epsilon prediction: {eps_audio_dir}")
    print(f"  V-prediction: {vpred_audio_dir}")
    print(f"\nGround truth stems available at:")
    print(f"  {gt_sources_dir}")


if __name__ == "__main__":
    main()
