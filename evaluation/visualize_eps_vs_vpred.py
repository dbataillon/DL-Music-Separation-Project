"""
Visualization comparing epsilon-prediction vs v-prediction diffusion outputs.
Creates a figure with ground truth, eps-pred, and v-pred side by side.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
SPECTROGRAM_DIR = PROJECT_ROOT / "Spectrogram"
SAMPLES_DIR = PROJECT_ROOT / "samples_comparison"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STEMS = ["vocal", "bass", "drums", "other"]


def load_data(track_name):
    """Load ground truth and predictions."""
    # Ground truth
    gt_path = SPECTROGRAM_DIR / f"{track_name}.npz"
    gt_data = np.load(gt_path)
    
    # Epsilon prediction
    eps_path = SAMPLES_DIR / f"{track_name}_eps_pred.npz"
    eps_data = np.load(eps_path)
    
    # V-prediction
    vpred_path = SAMPLES_DIR / f"{track_name}_vpred.npz"
    vpred_data = np.load(vpred_path)
    
    return gt_data, eps_data, vpred_data


def create_comparison_figure(track_name, time_slice=(0, 200), output_path=None):
    """
    Create a 3-row x 4-column comparison figure.
    
    Rows: Ground Truth, Epsilon-Prediction, V-Prediction
    Columns: vocal, bass, drums, other
    """
    gt_data, eps_data, vpred_data = load_data(track_name)
    
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    
    # Use log scale for better visibility of sparse data
    # We'll use a small offset to avoid log(0)
    def log_transform(x):
        return np.log1p(x * 100) / np.log1p(100)
    
    vmin, vmax = 0, 1
    
    row_labels = ['Ground Truth', 'ε-Prediction', 'V-Prediction']
    
    for col, stem in enumerate(STEMS):
        # Ground truth
        gt_stem = gt_data[stem]
        if time_slice:
            gt_stem = gt_stem[:, time_slice[0]:time_slice[1]]
        
        # Epsilon prediction (shape: 4, 513, T)
        eps_pred = eps_data['prediction'][col]
        if time_slice:
            eps_pred = eps_pred[:, time_slice[0]:time_slice[1]]
        
        # V-prediction
        vpred = vpred_data['prediction'][col]
        if time_slice:
            vpred = vpred[:, time_slice[0]:time_slice[1]]
        
        # Apply log transform for better visibility
        gt_vis = log_transform(gt_stem)
        eps_vis = log_transform(np.clip(eps_pred, 0, 1))
        vpred_vis = log_transform(np.clip(vpred, 0, 1))
        
        # Plot ground truth (row 0)
        axes[0, col].imshow(gt_vis, aspect='auto', origin='lower', 
                            cmap='magma', vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f'{stem.capitalize()}', fontsize=12, fontweight='bold')
        if col == 0:
            axes[0, col].set_ylabel(row_labels[0], fontsize=11)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        
        # Plot epsilon prediction (row 1)
        axes[1, col].imshow(eps_vis, aspect='auto', origin='lower', 
                            cmap='magma', vmin=vmin, vmax=vmax)
        if col == 0:
            axes[1, col].set_ylabel(row_labels[1], fontsize=11)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        
        # Plot v-prediction (row 2)
        im = axes[2, col].imshow(vpred_vis, aspect='auto', origin='lower', 
                                  cmap='magma', vmin=vmin, vmax=vmax)
        if col == 0:
            axes[2, col].set_ylabel(row_labels[2], fontsize=11)
        axes[2, col].set_xlabel('Time', fontsize=10)
        axes[2, col].set_yticks([])
    
    # Add colorbar on the right side, outside the plots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Log Magnitude', fontsize=10)
    
    plt.suptitle(f'Diffusion Model Comparison', fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(left=0.06, right=0.90, top=0.92, bottom=0.08, wspace=0.1, hspace=0.15)
    
    # Save
    if output_path is None:
        output_path = FIGURES_DIR / "diffusion_comparison_eps_vs_vpred.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path


def create_single_stem_detailed(track_name, stem="vocal", time_slice=(0, 300), output_path=None):
    """
    Create a detailed single-stem comparison with statistics.
    """
    gt_data, eps_data, vpred_data = load_data(track_name)
    
    stem_idx = STEMS.index(stem)
    
    gt_stem = gt_data[stem]
    eps_pred = eps_data['prediction'][stem_idx]
    vpred = vpred_data['prediction'][stem_idx]
    
    if time_slice:
        gt_stem = gt_stem[:, time_slice[0]:time_slice[1]]
        eps_pred = eps_pred[:, time_slice[0]:time_slice[1]]
        vpred = vpred[:, time_slice[0]:time_slice[1]]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    vmin, vmax = 0, 0.5  # Tighter range to show detail
    
    # Ground truth
    axes[0].imshow(gt_stem, aspect='auto', origin='lower', cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Ground Truth\nmean={gt_stem.mean():.4f}', fontsize=11)
    axes[0].set_ylabel('Frequency Bin')
    axes[0].set_xlabel('Time Frame')
    
    # Epsilon prediction
    axes[1].imshow(np.clip(eps_pred, vmin, vmax), aspect='auto', origin='lower', 
                   cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'ε-Prediction\nmean={eps_pred.mean():.4f}', fontsize=11)
    axes[1].set_xlabel('Time Frame')
    
    # V-prediction
    im = axes[2].imshow(np.clip(vpred, vmin, vmax), aspect='auto', origin='lower', 
                        cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(f'V-Prediction\nmean={vpred.mean():.4f}', fontsize=11)
    axes[2].set_xlabel('Time Frame')
    
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Magnitude')
    
    plt.suptitle(f'{stem.capitalize()} Stem Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if output_path is None:
        output_path = FIGURES_DIR / f"diffusion_comparison_{stem}_detailed.png"
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()
    
    return output_path


def print_statistics(track_name):
    """Print detailed statistics comparing all three."""
    gt_data, eps_data, vpred_data = load_data(track_name)
    
    print("\n" + "="*70)
    print(f"Statistics for: {track_name}")
    print("="*70)
    
    print(f"\n{'Stem':<10} {'Source':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*70)
    
    for i, stem in enumerate(STEMS):
        gt = gt_data[stem].flatten()
        eps = eps_data['prediction'][i].flatten()
        vpred = vpred_data['prediction'][i].flatten()
        
        print(f"{stem:<10} {'Ground Truth':<15} {gt.mean():>10.4f} {gt.std():>10.4f} {gt.min():>10.4f} {gt.max():>10.4f}")
        print(f"{'':<10} {'ε-Prediction':<15} {eps.mean():>10.4f} {eps.std():>10.4f} {eps.min():>10.4f} {eps.max():>10.4f}")
        print(f"{'':<10} {'V-Prediction':<15} {vpred.mean():>10.4f} {vpred.std():>10.4f} {vpred.min():>10.4f} {vpred.max():>10.4f}")
        print()


if __name__ == "__main__":
    track_name = "055 - Angels In Amplifiers - I'm Alright"
    
    # Print statistics
    print_statistics(track_name)
    
    # Create main comparison figure
    create_comparison_figure(track_name, time_slice=(0, 200))
    
    # Create detailed single-stem comparisons
    create_single_stem_detailed(track_name, stem="vocal", time_slice=(0, 300))
    
    print("\nDone!")
