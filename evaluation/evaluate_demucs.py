"""Evaluate our trained Demucs model on DSD100 subsets with SDR / SI-SDR metrics."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn.functional as F

try:
    import torchaudio
except ImportError as exc:
    raise ImportError("torchaudio is required.") from exc

try:
    from museval.metrics import bss_eval_sources
except ImportError as exc:
    raise ImportError(
        "museval is required for evaluation. Install with: pip install museval"
    ) from exc

from configs.config import SR
from models.demucs import get_demucs_model

STEM_NAMES = ["bass", "drums", "other", "vocals"]


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    """Compute Scale-Invariant SDR."""
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    ref_energy = np.sum(reference ** 2)
    if ref_energy <= eps:
        return float("-inf")
    projection = np.sum(estimate * reference) / (ref_energy + eps) * reference
    noise = estimate - projection
    numerator = np.sum(projection ** 2)
    denominator = np.sum(noise ** 2) + eps
    return 10 * np.log10((numerator + eps) / (denominator + eps))


def aggregate_metrics(
    track_metrics: List[Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Aggregate per-track metrics into mean/std/ci95."""
    summary: Dict[str, Dict[str, List[float]]] = {}
    for metrics in track_metrics:
        for stem, values in metrics.items():
            summary.setdefault(stem, {}).setdefault("SDR", []).append(values["SDR"])
            summary.setdefault(stem, {}).setdefault("SI_SDR", []).append(values["SI_SDR"])

    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stem, metric_dict in summary.items():
        aggregated[stem] = {}
        for metric_name, values in metric_dict.items():
            arr = np.asarray(values, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            ci = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
            aggregated[stem][metric_name] = {
                "mean": mean,
                "std": std,
                "ci95": float(ci),
            }
    return aggregated


def build_track_list(dataset_root: str, subset: str) -> List[Tuple[str, str]]:
    """Build list of (mix_path, stem_dir) for DSD100 subset."""
    mix_base = os.path.join(dataset_root, "Mixtures", subset)
    src_base = os.path.join(dataset_root, "Sources", subset)
    if not os.path.isdir(mix_base) or not os.path.isdir(src_base):
        raise FileNotFoundError(
            f"Expected DSD100 structure under '{dataset_root}'. "
            f"Missing Mixtures/{subset} or Sources/{subset}."
        )

    tracks: List[Tuple[str, str]] = []
    for track in sorted(os.listdir(mix_base)):
        mix_path = os.path.join(mix_base, track, "mixture.wav")
        stem_dir = os.path.join(src_base, track)
        if os.path.isfile(mix_path) and os.path.isdir(stem_dir):
            tracks.append((mix_path, stem_dir))
    if not tracks:
        raise RuntimeError(f"No tracks found for subset '{subset}' under {dataset_root}")
    return tracks


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load audio file and resample if needed. Returns (1, samples) mono."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def load_model(checkpoint_path: Path, device: torch.device, hidden_channels: int = 48, depth: int = 5):
    """Load trained Demucs model from checkpoint."""
    model = get_demucs_model(
        in_channels=1,
        num_stems=4,
        hidden_channels=hidden_channels,
        depth=depth,
    )
    
    state = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "model_state_dict" in state:
        model_state = state["model_state_dict"]
    else:
        model_state = state
    
    # Remove module. prefix if present
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model


def get_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """Find latest checkpoint in model directory."""
    if not model_dir.exists():
        return None
    
    # Try final weights first
    final_path = model_dir / "demucs_final.pt"
    if final_path.exists():
        return final_path
    
    # Otherwise find latest epoch checkpoint
    ckpts = sorted(model_dir.glob("checkpoint_epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def separate_with_model(
    model: torch.nn.Module,
    mix_waveform: torch.Tensor,
    device: torch.device,
    chunk_size: int = 16000 * 30,  # 30 seconds at 16kHz
    overlap: int = 16000 * 2,  # 2 seconds overlap
) -> Dict[str, np.ndarray]:
    """
    Separate mixture using our trained Demucs model.
    Uses chunked processing for long audio.
    """
    mix = mix_waveform.unsqueeze(0).to(device)  # (1, 1, samples)
    total_samples = mix.shape[-1]
    
    if total_samples <= chunk_size:
        # Process entire track at once
        with torch.no_grad():
            pred = model(mix)  # (1, num_stems, 1, samples)
        pred = pred[0].cpu().numpy()  # (num_stems, 1, samples)
    else:
        # Chunked processing with overlap-add
        num_stems = 4
        pred = np.zeros((num_stems, 1, total_samples), dtype=np.float32)
        weights = np.zeros((1, 1, total_samples), dtype=np.float32)
        
        step = chunk_size - overlap
        for start in range(0, total_samples, step):
            end = min(start + chunk_size, total_samples)
            chunk = mix[..., start:end]
            
            # Pad if chunk is too small
            if chunk.shape[-1] < chunk_size:
                pad_len = chunk_size - chunk.shape[-1]
                chunk = F.pad(chunk, (0, pad_len))
            
            with torch.no_grad():
                chunk_pred = model(chunk)  # (1, num_stems, 1, chunk_size)
            
            chunk_pred = chunk_pred[0].cpu().numpy()
            actual_len = end - start
            
            # Create triangular window for overlap-add
            window = np.ones(actual_len)
            if start > 0:
                fade_in = min(overlap, actual_len)
                window[:fade_in] = np.linspace(0, 1, fade_in)
            if end < total_samples:
                fade_out = min(overlap, actual_len)
                window[-fade_out:] = np.linspace(1, 0, fade_out)
            
            window = window.reshape(1, 1, -1)
            pred[:, :, start:end] += chunk_pred[:, :, :actual_len] * window
            weights[:, :, start:end] += window
        
        # Normalize by weights
        pred = pred / np.maximum(weights, 1e-8)
    
    result = {}
    for idx, stem in enumerate(STEM_NAMES):
        result[stem] = pred[idx, 0].astype(np.float32)
    return result


def evaluate_track(
    model: torch.nn.Module,
    mix_path: str,
    stem_dir: str,
    device: torch.device,
    sample_rate: int,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on a single track."""
    # Load mixture
    mix_waveform = load_audio(mix_path, sample_rate)
    
    # Separate
    estimates = separate_with_model(model, mix_waveform, device)
    
    metrics: Dict[str, Dict[str, float]] = {}
    references = []
    estimate_list = []

    for stem in STEM_NAMES:
        stem_path = os.path.join(stem_dir, f"{stem}.wav")
        if not os.path.isfile(stem_path):
            print(f"Warning: {stem_path} not found, skipping stem.")
            continue
        
        ref_waveform = load_audio(stem_path, sample_rate)
        ref_mono = ref_waveform[0].numpy()
        
        est_mono = estimates[stem]
        
        # Align lengths
        min_len = min(len(ref_mono), len(est_mono))
        ref_mono = ref_mono[:min_len]
        est_mono = est_mono[:min_len]
        
        references.append(ref_mono)
        estimate_list.append(est_mono)
        
        metrics[stem] = {
            "SI_SDR": si_sdr(ref_mono, est_mono),
            "SDR": 0.0,
        }

    # Compute SDR using museval
    if references and estimate_list:
        references_np = np.stack(references)
        estimates_np = np.stack(estimate_list)
        bss_outputs = bss_eval_sources(references_np, estimates_np)
        if len(bss_outputs) >= 4:
            sdr = bss_outputs[0]
            for idx, stem in enumerate(STEM_NAMES):
                if stem in metrics:
                    metrics[stem]["SDR"] = float(np.nanmean(sdr[idx]))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Demucs model on DSD100 subset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory containing Mixtures/ and Sources/.",
    )
    parser.add_argument(
        "--subset",
        choices=["Dev", "Test"],
        default="Dev",
        help="Subset to evaluate.",
    )
    parser.add_argument(
        "--model-dir",
        default="model/demucs",
        help="Directory with Demucs checkpoints.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint (defaults to latest in model-dir).",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=48,
        help="Model hidden channels (must match training).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="Model depth (must match training).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write aggregated metrics to JSON.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find checkpoint
    model_dir = Path(args.model_dir)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = get_latest_checkpoint(model_dir)
        if not checkpoint_path:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    print(f"Using checkpoint: {checkpoint_path}")

    # Load model
    model = load_model(checkpoint_path, device, args.hidden_channels, args.depth)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build track list
    tracks = build_track_list(args.dataset_root, args.subset)
    print(f"Found {len(tracks)} tracks in {args.subset} subset.")

    per_track_metrics: List[Dict[str, Dict[str, float]]] = []
    timings: List[float] = []

    for idx, (mix_path, stem_dir) in enumerate(tracks, start=1):
        track_name = os.path.basename(os.path.dirname(mix_path))
        start_time = time.time()
        
        try:
            metrics = evaluate_track(model, mix_path, stem_dir, device, SR)
            dt = time.time() - start_time
            per_track_metrics.append(metrics)
            timings.append(dt)
            
            print(
                f"[{idx:03d}/{len(tracks):03d}] {track_name}: "
                + ", ".join(
                    f"{stem} SDR={metrics[stem]['SDR']:.2f}dB SI-SDR={metrics[stem]['SI_SDR']:.2f}dB"
                    for stem in STEM_NAMES
                    if stem in metrics
                )
                + f" | {dt:.1f}s"
            )
        except Exception as e:
            print(f"[{idx:03d}/{len(tracks):03d}] {track_name}: ERROR - {e}")
            continue

    if not per_track_metrics:
        print("No tracks evaluated successfully.")
        return

    # Aggregate metrics
    summary = aggregate_metrics(per_track_metrics)
    avg_time = float(np.mean(timings))

    print("\n=== Aggregated Metrics (Trained Demucs) ===")
    print(f"Checkpoint: {checkpoint_path}")
    for stem in STEM_NAMES:
        if stem in summary:
            stem_metrics = summary[stem]
            print(
                f"{stem}: SDR {stem_metrics['SDR']['mean']:.2f}±{stem_metrics['SDR']['ci95']:.2f} dB | "
                f"SI-SDR {stem_metrics['SI_SDR']['mean']:.2f}±{stem_metrics['SI_SDR']['ci95']:.2f} dB"
            )
    print(f"Average inference time per track: {avg_time:.1f}s")

    # Save to JSON
    if args.output_json:
        payload = {
            "checkpoint": str(checkpoint_path),
            "dataset_root": args.dataset_root,
            "subset": args.subset,
            "hidden_channels": args.hidden_channels,
            "depth": args.depth,
            "metrics": summary,
            "average_inference_seconds": avg_time,
            "num_tracks": len(per_track_metrics),
        }
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
