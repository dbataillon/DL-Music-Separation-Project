"""Evaluate pretrained Demucs on DSD100 subsets with SDR / SI-SDR metrics."""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import torch
    import torchaudio
except ImportError as exc:
    raise ImportError("torch and torchaudio are required.") from exc

try:
    from demucs import pretrained
    from demucs.apply import apply_model
except ImportError as exc:
    raise ImportError(
        "demucs is required. Install with: pip install demucs"
    ) from exc

try:
    from museval.metrics import bss_eval_sources
except ImportError as exc:
    raise ImportError(
        "museval is required for evaluation. Install with: pip install museval"
    ) from exc

# Demucs stem order: drums, bass, other, vocals
DEMUCS_STEMS = ["drums", "bass", "other", "vocals"]
# DSD100 stem filenames
DSD100_STEMS = ["drums", "bass", "other", "vocals"]


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


def load_audio(path: str, target_sr: int = 44100) -> torch.Tensor:
    """Load audio file and resample if needed. Returns (channels, samples)."""
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform


def separate_with_demucs(
    model, mix_waveform: torch.Tensor, device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Run Demucs separation on mixture waveform.
    Returns dict mapping stem name to numpy array (samples,) mono.
    """
    # Demucs expects (batch, channels, samples)
    mix = mix_waveform.unsqueeze(0).to(device)
    
    with torch.no_grad():
        sources = apply_model(model, mix, device=device, progress=False)
    
    # sources shape: (batch, num_sources, channels, samples)
    sources = sources[0].cpu().numpy()  # (num_sources, channels, samples)
    
    result = {}
    for idx, stem in enumerate(DEMUCS_STEMS):
        # Convert to mono by averaging channels
        stem_audio = sources[idx].mean(axis=0)
        result[stem] = stem_audio.astype(np.float32)
    return result


def evaluate_track(
    model,
    mix_path: str,
    stem_dir: str,
    device: torch.device,
    target_sr: int = 44100,
) -> Dict[str, Dict[str, float]]:
    """Evaluate Demucs on a single track."""
    # Load mixture
    mix_waveform = load_audio(mix_path, target_sr)
    
    # Separate
    estimates = separate_with_demucs(model, mix_waveform, device)
    
    metrics: Dict[str, Dict[str, float]] = {}
    references = []
    estimate_list = []

    for stem in DSD100_STEMS:
        # Load ground truth
        stem_path = os.path.join(stem_dir, f"{stem}.wav")
        if not os.path.isfile(stem_path):
            print(f"Warning: {stem_path} not found, skipping stem.")
            continue
        
        ref_waveform = load_audio(stem_path, target_sr)
        ref_mono = ref_waveform.mean(dim=0).numpy()  # Convert to mono
        
        est_mono = estimates[stem]
        
        # Align lengths
        min_len = min(len(ref_mono), len(est_mono))
        ref_mono = ref_mono[:min_len]
        est_mono = est_mono[:min_len]
        
        references.append(ref_mono)
        estimate_list.append(est_mono)
        
        # Compute SI-SDR per stem
        metrics[stem] = {
            "SI_SDR": si_sdr(ref_mono, est_mono),
            "SDR": 0.0,  # Filled after bss_eval
        }

    # Compute SDR using museval
    if references and estimate_list:
        references_np = np.stack(references)
        estimates_np = np.stack(estimate_list)
        bss_outputs = bss_eval_sources(references_np, estimates_np)
        if len(bss_outputs) >= 4:
            sdr = bss_outputs[0]
            for idx, stem in enumerate(DSD100_STEMS):
                if stem in metrics:
                    metrics[stem]["SDR"] = float(np.nanmean(sdr[idx]))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained Demucs on DSD100 subset."
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
        "--model-name",
        default="htdemucs",
        help="Demucs model name (htdemucs, htdemucs_ft, mdx_extra, etc.).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write aggregated metrics to JSON.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda, cpu). Auto-detected if not specified.",
    )
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained Demucs
    print(f"Loading pretrained Demucs model: {args.model_name}")
    model = pretrained.get_model(args.model_name)
    model.to(device)
    model.eval()
    print(f"Model loaded. Sources: {model.sources}")

    # Build track list
    tracks = build_track_list(args.dataset_root, args.subset)
    print(f"Found {len(tracks)} tracks in {args.subset} subset.")

    per_track_metrics: List[Dict[str, Dict[str, float]]] = []
    timings: List[float] = []

    for idx, (mix_path, stem_dir) in enumerate(tracks, start=1):
        track_name = os.path.basename(os.path.dirname(mix_path))
        start_time = time.time()
        
        try:
            metrics = evaluate_track(model, mix_path, stem_dir, device)
            dt = time.time() - start_time
            per_track_metrics.append(metrics)
            timings.append(dt)
            
            print(
                f"[{idx:03d}/{len(tracks):03d}] {track_name}: "
                + ", ".join(
                    f"{stem} SDR={metrics[stem]['SDR']:.2f}dB SI-SDR={metrics[stem]['SI_SDR']:.2f}dB"
                    for stem in DSD100_STEMS
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

    print("\n=== Aggregated Metrics (Pretrained Demucs) ===")
    print(f"Model: {args.model_name}")
    for stem in DSD100_STEMS:
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
            "model": args.model_name,
            "dataset_root": args.dataset_root,
            "subset": args.subset,
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
