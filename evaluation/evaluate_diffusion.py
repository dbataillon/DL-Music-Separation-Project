"""Evaluate diffusion model on DSD100 subsets with SDR / SI-SDR metrics."""
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch

from configs.config import SR, window_size, hop_length, patch_size
from evaluation.sample_diffusion import (
    STEM_NAMES,
    load_model,
    make_cosine_schedule,
    prepare_mix_batches,
    sample_stems,
    stitch_predictions,
)

try:
    from museval.metrics import bss_eval_sources
except ImportError as exc:  # pragma: no cover
    raise ImportError("museval is required for evaluation. Install with `pip install museval`.") from exc


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
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


def aggregate_metrics(track_metrics: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
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
    mix_base = os.path.join(dataset_root, "Mixtures", subset)
    src_base = os.path.join(dataset_root, "Sources", subset)
    if not os.path.isdir(mix_base) or not os.path.isdir(src_base):
        raise FileNotFoundError(
            f"Expected DSD100 structure under '{dataset_root}'. Missing Mixtures/{subset} or Sources/{subset}."
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


def diffusion_forward(model, betas, mix_mag: np.ndarray) -> np.ndarray:
    mix_batch, segments, width = prepare_mix_batches(mix_mag, patch_size)
    preds = sample_stems(model, mix_batch, betas)
    stitched = stitch_predictions(preds, segments, width)
    return stitched


def reconstruct_audio(predicted_mag: np.ndarray, phase: np.ndarray, mix_length: int) -> np.ndarray:
    complex_spec = predicted_mag * phase
    audio = librosa.istft(
        complex_spec,
        hop_length=hop_length,
        win_length=window_size,
        length=mix_length,
    )
    return audio.astype(np.float32)


def evaluate_track(model, betas, mix_path: str, stem_dir: str) -> Dict[str, Dict[str, float]]:
    mix_audio, _ = librosa.load(mix_path, sr=SR)
    mix_audio = mix_audio.astype(np.float32)
    stft = librosa.stft(mix_audio, n_fft=window_size, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    predicted_mag = diffusion_forward(model, betas, mag)

    metrics: Dict[str, Dict[str, float]] = {}
    references = []
    estimates = []

    for idx, stem in enumerate(STEM_NAMES):
        stem_path = os.path.join(stem_dir, f"{stem}.wav")
        target_audio, _ = librosa.load(stem_path, sr=SR)
        target_audio = target_audio.astype(np.float32)

        estimate_audio = reconstruct_audio(predicted_mag[idx], phase, len(mix_audio))
        min_len = min(len(target_audio), len(estimate_audio))
        target_audio = target_audio[:min_len]
        estimate_audio = estimate_audio[:min_len]

        references.append(target_audio)
        estimates.append(estimate_audio)
        metrics[stem] = {
            "SI_SDR": si_sdr(target_audio, estimate_audio),
            "SDR": 0.0,  # placeholder, fill after bss_eval
        }

    references_np = np.stack(references)
    estimates_np = np.stack(estimates)
    bss_outputs = bss_eval_sources(references_np, estimates_np)
    if len(bss_outputs) == 4:
        sdr, _, _, _ = bss_outputs
    elif len(bss_outputs) == 5:
        sdr, _, _, _, _ = bss_outputs
    else:
        raise ValueError(f"Unexpected museval output length: {len(bss_outputs)}")

    for idx, stem in enumerate(STEM_NAMES):
        metrics[stem]["SDR"] = float(np.nanmean(sdr[idx]))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion model on DSD100 subset.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing Mixtures/ and Sources/.")
    parser.add_argument("--subset", choices=["Dev", "Test"], default="Dev", help="Subset to evaluate.")
    parser.add_argument("--model-dir", default="model/diffusion", help="Directory with diffusion checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (defaults to latest in model-dir).")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of diffusion steps (T).")
    parser.add_argument(
        "--output-json", default=None, help="Optional path to write aggregated metrics to JSON.")
    args = parser.parse_args()

    model_dir = args.model_dir
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpts = sorted(
            [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        )
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {model_dir}")
        checkpoint_path = ckpts[-1]

    print(f"Using checkpoint: {checkpoint_path}")
    betas = make_cosine_schedule(T=args.timesteps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(checkpoint_path), device)
    betas = betas.to(device)

    tracks = build_track_list(args.dataset_root, args.subset)
    per_track_metrics: List[Dict[str, Dict[str, float]]] = []
    timings: List[float] = []

    for idx, (mix_path, stem_dir) in enumerate(tracks, start=1):
        start_time = time.time()
        metrics = evaluate_track(model, betas, mix_path, stem_dir)
        dt = time.time() - start_time
        per_track_metrics.append(metrics)
        timings.append(dt)
        print(
            f"[{idx:03d}/{len(tracks):03d}] {os.path.basename(os.path.dirname(mix_path))}: "
            + ", ".join(f"{stem} SDR={metrics[stem]['SDR']:.2f}dB SI-SDR={metrics[stem]['SI_SDR']:.2f}dB" for stem in STEM_NAMES)
            + f" | {dt:.1f}s"
        )

    summary = aggregate_metrics(per_track_metrics)
    avg_time = float(np.mean(timings))

    print("\n=== Aggregated Metrics ===")
    for stem in STEM_NAMES:
        stem_metrics = summary[stem]
        print(
            f"{stem}: SDR {stem_metrics['SDR']['mean']:.2f}±{stem_metrics['SDR']['ci95']:.2f} dB | "
            f"SI-SDR {stem_metrics['SI_SDR']['mean']:.2f}±{stem_metrics['SI_SDR']['ci95']:.2f} dB"
        )
    print(f"Average inference time per track: {avg_time:.1f}s")

    if args.output_json:
        payload = {
            "dataset_root": args.dataset_root,
            "subset": args.subset,
            "checkpoint": checkpoint_path,
            "timesteps": args.timesteps,
            "metrics": summary,
            "average_inference_seconds": avg_time,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
