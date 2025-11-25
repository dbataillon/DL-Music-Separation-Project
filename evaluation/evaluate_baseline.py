import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch

try:
    from museval.metrics import bss_eval_sources
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError("museval is required for evaluation. Install with `pip install museval`.") from exc

from configs.config import SR, window_size, hop_length, patch_size
from models.U_net import UNetStandard


STEM_ORDER = ("vocal", "bass", "drums", "other")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])


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
    return 10 * np.log10((numerator + eps) / denominator)


def chunk_spectrogram(mag_wo_dc: np.ndarray) -> Tuple[np.ndarray, int]:
    _, total_frames = mag_wo_dc.shape
    num_patches = int(np.ceil(total_frames / patch_size))
    padded_frames = num_patches * patch_size
    padded = np.zeros((mag_wo_dc.shape[0], padded_frames), dtype=np.float32)
    padded[:, :total_frames] = mag_wo_dc
    return padded, total_frames


def run_model_on_mix(
    model: torch.nn.Module,
    device: torch.device,
    mix_audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(mix_audio, n_fft=window_size, hop_length=hop_length)
    mag, phase = librosa.magphase(stft)
    mag_wo_dc = mag[1:, :]  # drop DC bin to match training

    padded_mag, original_frames = chunk_spectrogram(mag_wo_dc)
    num_frames = padded_mag.shape[1]
    pred_mag = np.zeros((len(STEM_ORDER), mag_wo_dc.shape[0], num_frames), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, num_frames, patch_size):
            end = start + patch_size
            patch = padded_mag[:, start:end]
            patch_tensor = torch.from_numpy(patch[None, None, :, :]).to(device)
            preds = model(patch_tensor, training=False).cpu().numpy()[0]  # (4, 512, patch_size)
            pred_mag[:, :, start:end] = preds

    pred_mag = pred_mag[:, :, :original_frames]
    return pred_mag, phase[:, :original_frames]


def reconstruct_audio(
    predicted_mag: np.ndarray,
    phase: np.ndarray,
    mix_length: int,
) -> np.ndarray:
    full_mag = np.vstack(
        (np.zeros((1, predicted_mag.shape[1]), dtype=np.float32), predicted_mag)
    )
    complex_spec = full_mag * phase
    audio = librosa.istft(
        complex_spec, hop_length=hop_length, win_length=window_size, length=mix_length
    )
    return audio.astype(np.float32)


def evaluate_track(
    model: torch.nn.Module,
    device: torch.device,
    mix_path: str,
    stem_dir: str,
) -> Dict[str, Dict[str, float]]:
    mix_audio, _ = librosa.load(mix_path, sr=SR)
    mix_audio = mix_audio.astype(np.float32)
    predicted_mag, phase = run_model_on_mix(model, device, mix_audio)

    references = []
    estimates = []
    si_sdr_scores = {}

    for idx, stem in enumerate(STEM_ORDER):
        stem_path = os.path.join(stem_dir, f"{stem}.wav")
        target_audio, _ = librosa.load(stem_path, sr=SR)
        target_audio = target_audio.astype(np.float32)

        estimate_audio = reconstruct_audio(predicted_mag[idx], phase, len(mix_audio))

        min_len = min(len(target_audio), len(estimate_audio))
        target_audio = target_audio[:min_len]
        estimate_audio = estimate_audio[:min_len]

        references.append(target_audio)
        estimates.append(estimate_audio)
        si_sdr_scores[stem] = si_sdr(target_audio, estimate_audio)

    references = np.stack(references)
    estimates = np.stack(estimates)
    bss_outputs = bss_eval_sources(references, estimates)
    if len(bss_outputs) == 4:
        sdr, _, _, _ = bss_outputs
    elif len(bss_outputs) == 5:
        sdr, _, _, _, _ = bss_outputs
    else:  # pragma: no cover - defensive for future API changes
        raise ValueError(f"Unexpected number of outputs from bss_eval_sources: {len(bss_outputs)}")

    metrics = {}
    for idx, stem in enumerate(STEM_ORDER):
        metrics[stem] = {
            "SDR": float(np.nanmean(sdr[idx])),
            "SI_SDR": float(si_sdr_scores[stem]),
        }
    return metrics


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
            arr = np.array(values, dtype=np.float32)
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
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
            f"Expected DSD100 structure under '{dataset_root}'. "
            "Directories 'Mixtures/{subset}' and 'Sources/{subset}' must exist."
        )

    tracks = []
    for track in sorted(os.listdir(mix_base)):
        mix_path = os.path.join(mix_base, track, "mixture.wav")
        stem_dir = os.path.join(src_base, track)
        if os.path.isfile(mix_path) and os.path.isdir(stem_dir):
            tracks.append((mix_path, stem_dir))
    if not tracks:
        raise RuntimeError(f"No tracks found for subset '{subset}' in {dataset_root}")
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline U-Net on DSD100.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory of DSD100 (contains Mixtures/ and Sources/).",
    )
    parser.add_argument(
        "--subset",
        default="Dev",
        choices=["Dev", "Test"],
        help="Subset to evaluate (Dev or Test).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained U-Net checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write metrics summary as JSON.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluating on device: {device}")

    model = UNetStandard(num_outputs=len(STEM_ORDER), dropout_p=0.4, final_activation="relu").to(device)
    load_checkpoint(model, args.checkpoint, device)

    print('nb parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    # tracks = build_track_list(args.dataset_root, args.subset)
    # per_track_metrics: List[Dict[str, Dict[str, float]]] = []
    # timings: List[float] = []

    # for idx, (mix_path, stem_dir) in enumerate(tracks, start=1):
    #     track_name = os.path.basename(os.path.dirname(mix_path))
    #     t0 = time.time()
    #     metrics = evaluate_track(model, device, mix_path, stem_dir)
    #     dt = time.time() - t0
    #     per_track_metrics.append(metrics)
    #     timings.append(dt)
    #     print(f"[{idx:03d}/{len(tracks):03d}] {track_name}: "
    #           + ", ".join(f"{stem} SDR={metrics[stem]['SDR']:.2f} dB" for stem in STEM_ORDER)
    #           + f" | {dt:.1f}s")

    # summary = aggregate_metrics(per_track_metrics)
    # avg_time = float(np.mean(timings))
    # print("\n=== Aggregated Metrics ===")
    # for stem in STEM_ORDER:
    #     stem_metrics = summary[stem]
    #     print(
    #         f"{stem}: SDR {stem_metrics['SDR']['mean']:.2f}±{stem_metrics['SDR']['ci95']:.2f} dB "
    #         f"| SI-SDR {stem_metrics['SI_SDR']['mean']:.2f}±{stem_metrics['SI_SDR']['ci95']:.2f} dB"
    #     )
    # print(f"Average inference time per track: {avg_time:.1f}s")

    # if args.output_json:
    #     payload = {
    #         "dataset_root": args.dataset_root,
    #         "subset": args.subset,
    #         "checkpoint": args.checkpoint,
    #         "metrics": summary,l
    #         "average_inference_seconds": avg_time,
    #     }
    #     with open(args.output_json, "w", encoding="utf-8") as f:
    #         json.dump(payload, f, indent=2)
    #     print(f"Wrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
