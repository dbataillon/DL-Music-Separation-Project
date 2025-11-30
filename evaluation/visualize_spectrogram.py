"""
Visualize predicted spectrograms from diffusion model sampling.

Loads a predicted .npz file (and optionally ground truth) and plots spectrograms
for each stem side-by-side.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

STEM_NAMES = ["bass", "drums", "other", "vocals"]


def load_npz(path: str, load_stems: bool = False) -> np.ndarray:
    """Load spectrogram from .npz file."""
    data = np.load(path)
    
    # If loading stems, try to stack them
    if load_stems:
        stem_keys = ["bass", "drums", "other", "vocals", "vocal"]
        stems = []
        for key in ["bass", "drums", "other"]:
            if key in data:
                stems.append(data[key])
        # Handle vocal vs vocals
        if "vocals" in data:
            stems.append(data["vocals"])
        elif "vocal" in data:
            stems.append(data["vocal"])
        
        if stems:
            return np.stack(stems, axis=0)
    
    # Try common keys
    for key in ["predicted", "prediction", "arr_0", "mag", "magnitude"]:
        if key in data:
            return data[key]
    # Just return the first array
    return data[list(data.keys())[0]]


def plot_spectrogram(ax, spec: np.ndarray, title: str, vmin: float = None, vmax: float = None):
    """Plot a single spectrogram on the given axis."""
    # Convert to dB scale for better visualization
    spec_db = 20 * np.log10(np.abs(spec) + 1e-8)
    
    if vmin is None:
        vmin = np.percentile(spec_db, 5)
    if vmax is None:
        vmax = np.percentile(spec_db, 95)
    
    im = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    return im


def visualize_prediction(
    pred_path: str,
    gt_path: str = None,
    output_path: str = None,
    show: bool = True,
):
    """
    Visualize predicted spectrograms.
    
    Args:
        pred_path: Path to predicted .npz file (output from sample_diffusion.py)
        gt_path: Optional path to ground truth .npz file
        output_path: Optional path to save the figure
        show: Whether to display the figure
    """
    # Load prediction
    pred = load_npz(pred_path)
    print(f"Predicted shape: {pred.shape}")
    
    # Handle different shapes
    if pred.ndim == 2:
        # Single spectrogram
        pred = pred[np.newaxis, ...]
    elif pred.ndim == 3 and pred.shape[0] != 4:
        # Might be (H, W, C) instead of (C, H, W)
        if pred.shape[-1] == 4:
            pred = np.transpose(pred, (2, 0, 1))
    
    num_stems = min(pred.shape[0], 4)
    
    # Load ground truth if provided
    gt = None
    if gt_path and os.path.isfile(gt_path):
        gt = load_npz(gt_path, load_stems=True)
        print(f"Ground truth shape: {gt.shape}")
    
    # Create figure
    if gt is not None:
        fig, axes = plt.subplots(num_stems, 3, figsize=(15, 3 * num_stems))
        cols = ["Predicted", "Ground Truth", "Difference"]
    else:
        fig, axes = plt.subplots(num_stems, 1, figsize=(8, 3 * num_stems))
        axes = axes.reshape(-1, 1) if num_stems > 1 else np.array([[axes]])
        cols = ["Predicted"]
    
    if num_stems == 1:
        axes = axes.reshape(1, -1)
    
    # Get global vmin/vmax for consistent coloring
    pred_db = 20 * np.log10(np.abs(pred) + 1e-8)
    vmin = np.percentile(pred_db, 5)
    vmax = np.percentile(pred_db, 95)
    
    for i in range(num_stems):
        stem_name = STEM_NAMES[i] if i < len(STEM_NAMES) else f"Stem {i}"
        
        # Plot prediction
        plot_spectrogram(axes[i, 0], pred[i], f"{stem_name} - Predicted", vmin, vmax)
        
        if gt is not None and i < gt.shape[0]:
            # Plot ground truth
            plot_spectrogram(axes[i, 1], gt[i], f"{stem_name} - Ground Truth", vmin, vmax)
            
            # Plot difference
            diff = pred[i] - gt[i]
            diff_db = 20 * np.log10(np.abs(diff) + 1e-8)
            im = axes[i, 2].imshow(
                diff_db,
                aspect="auto",
                origin="lower",
                cmap="RdBu_r",
                vmin=-30,
                vmax=30,
            )
            axes[i, 2].set_title(f"{stem_name} - Difference (dB)", fontsize=10)
            axes[i, 2].set_xlabel("Time")
            axes[i, 2].set_ylabel("Frequency")
    
    plt.tight_layout()
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    plt.colorbar(plt.cm.ScalarMappable(cmap="magma"), cax=cbar_ax, label="dB")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_statistics(pred_path: str, gt_path: str = None):
    """Print statistics about the spectrograms."""
    pred = load_npz(pred_path)
    
    print("\n=== Prediction Statistics ===")
    print(f"Shape: {pred.shape}")
    print(f"Min: {pred.min():.4f}, Max: {pred.max():.4f}")
    print(f"Mean: {pred.mean():.4f}, Std: {pred.std():.4f}")
    print(f"% zeros: {100 * (pred == 0).sum() / pred.size:.2f}%")
    print(f"% negative: {100 * (pred < 0).sum() / pred.size:.2f}%")
    
    if gt_path and os.path.isfile(gt_path):
        gt_data = np.load(gt_path)
        for key in ["y", "stems", "sources", "arr_0"]:
            if key in gt_data:
                gt = gt_data[key]
                break
        else:
            gt = gt_data[list(gt_data.keys())[0]]
        
        print("\n=== Ground Truth Statistics ===")
        print(f"Shape: {gt.shape}")
        print(f"Min: {gt.min():.4f}, Max: {gt.max():.4f}")
        print(f"Mean: {gt.mean():.4f}, Std: {gt.std():.4f}")
        
        # Compute error metrics
        if pred.shape == gt.shape:
            mse = np.mean((pred - gt) ** 2)
            mae = np.mean(np.abs(pred - gt))
            print(f"\n=== Error Metrics ===")
            print(f"MSE: {mse:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"RMSE: {np.sqrt(mse):.6f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize diffusion model predictions.")
    parser.add_argument("prediction", help="Path to predicted .npz file")
    parser.add_argument("--ground-truth", "-gt", help="Path to ground truth .npz file")
    parser.add_argument("--output", "-o", help="Path to save output figure (PNG/PDF)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the figure")
    parser.add_argument("--stats", action="store_true", help="Print statistics only")
    args = parser.parse_args()
    
    if args.stats:
        plot_statistics(args.prediction, args.ground_truth)
    else:
        plot_statistics(args.prediction, args.ground_truth)
        visualize_prediction(
            args.prediction,
            args.ground_truth,
            args.output,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
