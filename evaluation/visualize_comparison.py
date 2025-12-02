"""
Visualization script to create spectrogram comparison figures.
Uses pre-computed predictions from .npz files (no model inference needed).
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Output directory for figures
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Data paths
SPECTROGRAM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Spectrogram")
DIFFUSION_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")

# Stems to visualize
STEMS = ["vocal", "bass", "drums", "other"]

def load_ground_truth(track_name):
    """Load ground truth spectrograms from preprocessed data."""
    spec_path = os.path.join(SPECTROGRAM_DIR, f"{track_name}.npz")
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Ground truth not found: {spec_path}")
    
    data = np.load(spec_path)
    return {
        "mix": data["mix"],
        "vocal": data["vocal"],
        "bass": data["bass"],
        "drums": data["drums"],
        "other": data["other"]
    }

def load_diffusion_predictions(track_name):
    """Load diffusion model predictions.
    
    Predictions are stored as (4, 513, T) where:
    - 0: vocal, 1: bass, 2: drums, 3: other
    """
    pred_path = os.path.join(DIFFUSION_SAMPLES_DIR, f"{track_name}_pred.npz")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Diffusion predictions not found: {pred_path}")
    
    data = np.load(pred_path)
    prediction = data["prediction"]  # Shape: (4, 513, T)
    
    return {
        "vocal": prediction[0],
        "bass": prediction[1],
        "drums": prediction[2],
        "other": prediction[3]
    }

def create_comparison_figure(track_name, stem="vocal", time_slice=None, output_path=None):
    """
    Create a comparison figure showing mixture, ground truth, and predictions.
    
    Args:
        track_name: Name of the track (without extension)
        stem: Which stem to visualize (vocal, bass, drums, other)
        time_slice: Tuple (start, end) for time axis, or None for full
        output_path: Where to save the figure
    """
    print(f"Loading data for {track_name}...")
    
    # Load data
    gt_data = load_ground_truth(track_name)
    diff_data = load_diffusion_predictions(track_name)
    
    # Get spectrograms
    mixture = gt_data["mix"]
    gt_stem = gt_data[stem]
    diff_pred = diff_data.get(stem, diff_data.get("vocal"))  # fallback
    
    # Spectrograms are already 2D (513, T)
    if len(mixture.shape) > 2:
        mixture = mixture[0] if len(mixture.shape) == 3 else mixture[0, 0]
    if len(gt_stem.shape) > 2:
        gt_stem = gt_stem[0] if len(gt_stem.shape) == 3 else gt_stem[0, 0]
    if len(diff_pred.shape) > 2:
        diff_pred = diff_pred[0] if len(diff_pred.shape) == 3 else diff_pred[0, 0]
    
    # Apply time slice if specified
    if time_slice:
        mixture = mixture[:, time_slice[0]:time_slice[1]]
        gt_stem = gt_stem[:, time_slice[0]:time_slice[1]]
        diff_pred = diff_pred[:, time_slice[0]:time_slice[1]]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Common colorbar limits based on ground truth
    vmin, vmax = 0, 1
    
    # Plot mixture
    im0 = axes[0].imshow(mixture, aspect='auto', origin='lower', 
                         cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Mixture', fontsize=12)
    axes[0].set_ylabel('Frequency Bin')
    axes[0].set_xlabel('Time Frame')
    
    # Plot ground truth
    im1 = axes[1].imshow(gt_stem, aspect='auto', origin='lower', 
                         cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Ground Truth ({stem})', fontsize=12)
    axes[1].set_xlabel('Time Frame')
    
    # Plot diffusion prediction
    # Clip to valid range for visualization
    diff_pred_clipped = np.clip(diff_pred, vmin, vmax)
    im2 = axes[2].imshow(diff_pred_clipped, aspect='auto', origin='lower', 
                         cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Diffusion Prediction', fontsize=12)
    axes[2].set_xlabel('Time Frame')
    
    # Add colorbar
    fig.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.8, label='Magnitude')
    
    plt.suptitle(f'Spectrogram Comparison: {track_name}', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, f"spectrogram_comparison_{stem}.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path

def create_all_stems_figure(track_name, time_slice=None, output_path=None):
    """
    Create a figure comparing all stems: mixture + 4 ground truths + 4 diffusion predictions.
    Layout: 3 rows x 4 columns (mixture repeated, or top row for reference)
    """
    print(f"Loading data for {track_name}...")
    
    # Load data
    gt_data = load_ground_truth(track_name)
    diff_data = load_diffusion_predictions(track_name)
    
    # Get mixture
    mixture = gt_data["mix"]
    if len(mixture.shape) > 2:
        mixture = mixture[0] if len(mixture.shape) == 3 else mixture[0, 0]
    
    # Apply time slice
    if time_slice:
        mixture = mixture[:, time_slice[0]:time_slice[1]]
    
    # Create figure: 2 rows (GT, Prediction) x 4 columns (stems)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    
    vmin, vmax = 0, 1
    
    for i, stem in enumerate(STEMS):
        # Get ground truth
        gt_stem = gt_data[stem]
        if len(gt_stem.shape) > 2:
            gt_stem = gt_stem[0] if len(gt_stem.shape) == 3 else gt_stem[0, 0]
        
        # Get diffusion prediction
        diff_pred = diff_data.get(stem, np.zeros_like(gt_stem))
        if len(diff_pred.shape) > 2:
            diff_pred = diff_pred[0] if len(diff_pred.shape) == 3 else diff_pred[0, 0]
        
        # Apply time slice
        if time_slice:
            gt_stem = gt_stem[:, time_slice[0]:time_slice[1]]
            diff_pred = diff_pred[:, time_slice[0]:time_slice[1]]
        
        # Clip diffusion for visualization
        diff_pred_clipped = np.clip(diff_pred, vmin, vmax)
        
        # Plot ground truth (top row)
        axes[0, i].imshow(gt_stem, aspect='auto', origin='lower', 
                          cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f'GT: {stem}', fontsize=11)
        if i == 0:
            axes[0, i].set_ylabel('Ground Truth\nFreq Bin')
        
        # Plot diffusion prediction (bottom row)
        im = axes[1, i].imshow(diff_pred_clipped, aspect='auto', origin='lower', 
                               cmap='magma', vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f'Pred: {stem}', fontsize=11)
        axes[1, i].set_xlabel('Time Frame')
        if i == 0:
            axes[1, i].set_ylabel('Diffusion\nFreq Bin')
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Magnitude')
    
    plt.suptitle(f'Source Separation Comparison: {track_name}', fontsize=14)
    plt.tight_layout()
    
    # Save
    if output_path is None:
        output_path = os.path.join(FIGURES_DIR, "spectrogram_comparison_all_stems.png")
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path

def analyze_predictions():
    """Analyze the diffusion predictions to understand their distribution."""
    track_name = "055 - Angels In Amplifiers - I'm Alright"
    
    gt_data = load_ground_truth(track_name)
    diff_data = load_diffusion_predictions(track_name)
    
    print("\n=== Analysis of Diffusion Predictions ===\n")
    
    for stem in STEMS:
        gt = gt_data[stem].flatten()
        pred = diff_data.get(stem, np.array([0])).flatten()
        
        print(f"{stem}:")
        print(f"  GT range: [{gt.min():.4f}, {gt.max():.4f}], mean: {gt.mean():.4f}")
        print(f"  Pred range: [{pred.min():.4f}, {pred.max():.4f}], mean: {pred.mean():.4f}")
        
        # Count values at boundaries (after clamping)
        pred_clamped = np.clip(pred, 0, 1)
        near_zero = (pred_clamped < 0.01).sum() / len(pred) * 100
        near_one = (pred_clamped > 0.99).sum() / len(pred) * 100
        print(f"  Pred near 0 (<0.01): {near_zero:.1f}%")
        print(f"  Pred near 1 (>0.99): {near_one:.1f}%")
        print()

if __name__ == "__main__":
    # Track to visualize (from samples directory)
    track_name = "055 - Angels In Amplifiers - I'm Alright"
    
    # Analyze predictions first
    analyze_predictions()
    
    # Create single stem comparison
    create_comparison_figure(track_name, stem="vocal", time_slice=(0, 200))
    
    # Create all stems comparison
    create_all_stems_figure(track_name, time_slice=(0, 200))
    
    print("\nDone!")
