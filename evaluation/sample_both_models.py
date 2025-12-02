#!/usr/bin/env python
"""
Sample from both epsilon-prediction and v-prediction diffusion models for comparison.
Saves predictions and optionally generates audio.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Import from both sampling modules
from evaluation.sample_diffusion import (
    load_mix,
    get_device,
    get_latest_checkpoint as get_latest_eps,
    load_model,
    make_cosine_schedule,
    prepare_mix_batches,
    sample_stems as sample_stems_eps,
    stitch_predictions,
    STEM_NAMES,
)
from evaluation.sample_diffusion_vpred import (
    sample_stems as sample_stems_vpred,
    DEFAULT_VPRED_MODEL_DIR,
)
from configs.config import patch_size


def get_latest_checkpoint(model_dir: Path) -> Path:
    """Get the latest checkpoint from a directory."""
    ckpts = sorted(model_dir.glob("checkpoint_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return ckpts[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Sample from both eps-pred and v-pred diffusion models"
    )
    parser.add_argument(
        "spectrogram",
        type=str,
        help="Path to .npz spectrogram file (with 'mix' key)",
    )
    parser.add_argument(
        "--eps-model-dir",
        type=str,
        default="model/diffusion",
        help="Directory with eps-prediction checkpoints",
    )
    parser.add_argument(
        "--vpred-model-dir",
        type=str,
        default=DEFAULT_VPRED_MODEL_DIR,
        help="Directory with v-prediction checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples_comparison",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200,
        help="Number of diffusion timesteps",
    )
    args = parser.parse_args()

    spec_path = Path(args.spectrogram)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    track_name = spec_path.stem

    # Load mixture
    print(f"Loading spectrogram: {spec_path}")
    mix = load_mix(spec_path)

    device = get_device()
    print(f"Using device: {device}")

    # Prepare batches (same for both models)
    betas = make_cosine_schedule(T=args.timesteps)
    mix_batch, segments, width = prepare_mix_batches(mix, patch_size)

    # === Epsilon-prediction model ===
    eps_model_dir = Path(args.eps_model_dir)
    try:
        eps_ckpt = get_latest_checkpoint(eps_model_dir)
        print(f"\n=== Epsilon-Prediction Model ===")
        print(f"Checkpoint: {eps_ckpt}")

        eps_model = load_model(eps_ckpt, device)
        eps_preds = sample_stems_eps(eps_model, mix_batch, betas)
        eps_full = stitch_predictions(eps_preds, segments, width)

        # Save
        eps_output = output_dir / f"{track_name}_eps_pred.npz"
        np.savez_compressed(eps_output, prediction=eps_full, mix=mix)
        print(f"Saved eps-prediction to: {eps_output}")

        # Print stats
        print(f"  Output range: [{eps_full.min():.4f}, {eps_full.max():.4f}]")
        print(f"  Output mean: {eps_full.mean():.4f}")
        for i, stem in enumerate(STEM_NAMES):
            print(f"  {stem}: mean={eps_full[i].mean():.4f}, std={eps_full[i].std():.4f}")

    except FileNotFoundError as e:
        print(f"Skipping eps-prediction: {e}")
        eps_full = None

    # === V-prediction model ===
    vpred_model_dir = Path(args.vpred_model_dir)
    try:
        vpred_ckpt = get_latest_checkpoint(vpred_model_dir)
        print(f"\n=== V-Prediction Model ===")
        print(f"Checkpoint: {vpred_ckpt}")

        vpred_model = load_model(vpred_ckpt, device)
        vpred_preds = sample_stems_vpred(vpred_model, mix_batch, betas)
        vpred_full = stitch_predictions(vpred_preds, segments, width)

        # Save
        vpred_output = output_dir / f"{track_name}_vpred.npz"
        np.savez_compressed(vpred_output, prediction=vpred_full, mix=mix)
        print(f"Saved v-prediction to: {vpred_output}")

        # Print stats
        print(f"  Output range: [{vpred_full.min():.4f}, {vpred_full.max():.4f}]")
        print(f"  Output mean: {vpred_full.mean():.4f}")
        for i, stem in enumerate(STEM_NAMES):
            print(f"  {stem}: mean={vpred_full[i].mean():.4f}, std={vpred_full[i].std():.4f}")

    except FileNotFoundError as e:
        print(f"Skipping v-prediction: {e}")
        vpred_full = None

    # === Load ground truth for comparison ===
    print(f"\n=== Ground Truth ===")
    gt_data = np.load(spec_path)
    for stem in STEM_NAMES:
        if stem in gt_data:
            gt_stem = gt_data[stem]
            print(f"  {stem}: mean={gt_stem.mean():.4f}, std={gt_stem.std():.4f}")

    print(f"\n=== Summary ===")
    print(f"Output directory: {output_dir}")
    if eps_full is not None:
        print(f"  - {track_name}_eps_pred.npz")
    if vpred_full is not None:
        print(f"  - {track_name}_vpred.npz")


if __name__ == "__main__":
    main()
