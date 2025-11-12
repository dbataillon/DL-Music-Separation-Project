import argparse
import json
import os
import time
import re
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> Tuple[int, float]:
    """Load checkpoint and return epoch number and training loss if available."""
    state = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel checkpoints - remove 'module.' prefix if present
    model_state = state["model_state_dict"]
    is_dataparallel = any(key.startswith("module.") for key in model_state.keys())
    
    if is_dataparallel:
        # Checkpoint was from DataParallel, remove prefix for single-GPU eval
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    
    model.load_state_dict(model_state)
    epoch = state.get("epoch", -1)
    training_loss = state.get("loss", None)  # Try to get loss if saved
    return epoch, training_loss


def load_training_loss_from_log(model_dir: str, epoch: int) -> float:
    """Load training loss from the training_loss.json log file."""
    loss_log_path = os.path.join(model_dir, "training_loss.json")
    if os.path.exists(loss_log_path):
        with open(loss_log_path, "r") as f:
            loss_history = json.load(f)
        epoch_key = f"epoch_{epoch:04d}"
        if epoch_key in loss_history:
            return loss_history[epoch_key].get("loss", None)
    return None


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


def get_checkpoints_sorted(model_dir: str) -> List[Tuple[str, int]]:
    """Get all checkpoint files sorted by epoch number."""
    checkpoints = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".pt") and filename.startswith("checkpoint_epoch_"):
            # Extract epoch number from filename: checkpoint_epoch_XXXX.pt
            match = re.search(r"checkpoint_epoch_(\d+)", filename)
            if match:
                epoch = int(match.group(1))
                path = os.path.join(model_dir, filename)
                checkpoints.append((path, epoch))
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


def evaluate_single_epoch(args_tuple: Tuple) -> Dict:
    """Evaluate a single checkpoint epoch. Used for parallel processing."""
    ckpt_path, epoch, dataset_root, subset, model_dir = args_tuple
    
    device = get_device()
    
    # Load model
    model = UNetStandard(num_outputs=len(STEM_ORDER), dropout_p=0.4, final_activation="relu").to(device)
    loaded_epoch, checkpoint_loss = load_checkpoint(model, ckpt_path, device)
    
    # Try to load loss from JSON log file (preferred)
    training_loss = load_training_loss_from_log(model_dir, epoch)
    if training_loss is None:
        training_loss = checkpoint_loss  # fallback to checkpoint loss

    # Load tracks once per epoch
    tracks = build_track_list(dataset_root, subset)

    # Evaluate on all tracks
    per_track_metrics: List[Dict[str, Dict[str, float]]] = []
    timings: List[float] = []

    for idx, (mix_path, stem_dir) in enumerate(tracks, start=1):
        t0 = time.time()
        metrics = evaluate_track(model, device, mix_path, stem_dir)
        dt = time.time() - t0
        per_track_metrics.append(metrics)
        timings.append(dt)

    # Aggregate metrics for this epoch
    summary = aggregate_metrics(per_track_metrics)
    avg_time = float(np.mean(timings))

    # Return results
    return {
        "epoch": epoch,
        "checkpoint": ckpt_path,
        "training_loss": training_loss,
        "metrics": summary,
        "average_inference_seconds": avg_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoint epochs on DSD100.")
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
        "--model-dir",
        default="./models/unet",
        help="Directory containing checkpoint files.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write metrics summary as JSON.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers for evaluation. Each worker uses 1 GPU if available.",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluating on device: {device}")

    # Get all checkpoints
    checkpoints = get_checkpoints_sorted(args.model_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {args.model_dir}")
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate")

    # Store results for all epochs
    all_results = {}

    if args.num_workers > 1:
        print(f"Using {args.num_workers} parallel workers")
        # Prepare arguments for parallel processing
        eval_args = [
            (ckpt_path, epoch, args.dataset_root, args.subset, args.model_dir)
            for ckpt_path, epoch in checkpoints
        ]
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(evaluate_single_epoch, arg): arg[1] for arg in eval_args}
            
            for future in as_completed(futures):
                epoch = futures[future]
                try:
                    result = future.result()
                    all_results[f"epoch_{result['epoch']:04d}"] = result
                    
                    # Print summary for this epoch
                    print(f"\n=== Evaluated Epoch {epoch:04d} ===")
                    metrics = result["metrics"]
                    loss = result["training_loss"] if result["training_loss"] is not None else "N/A"
                    print(f"  Training Loss: {loss}")
                    for stem in STEM_ORDER:
                        stem_metrics = metrics[stem]
                        print(
                            f"  {stem}: SDR {stem_metrics['SDR']['mean']:.2f}±{stem_metrics['SDR']['ci95']:.2f} dB "
                            f"| SI-SDR {stem_metrics['SI_SDR']['mean']:.2f}±{stem_metrics['SI_SDR']['ci95']:.2f} dB"
                        )
                    print(f"  Average inference time per track: {result['average_inference_seconds']:.1f}s")
                except Exception as e:
                    print(f"Error evaluating epoch {epoch}: {e}")
    else:
        print("Using 1 worker (sequential evaluation)")
        # Sequential evaluation (original behavior)
        for ckpt_path, epoch in checkpoints:
            print(f"\n=== Evaluating Epoch {epoch:04d} ===")
            
            result = evaluate_single_epoch((ckpt_path, epoch, args.dataset_root, args.subset, args.model_dir))
            all_results[f"epoch_{result['epoch']:04d}"] = result
            
            # Print summary for this epoch
            metrics = result["metrics"]
            loss = result["training_loss"] if result["training_loss"] is not None else "N/A"
            print(f"  Training Loss: {loss}")
            for stem in STEM_ORDER:
                stem_metrics = metrics[stem]
                print(
                    f"  {stem}: SDR {stem_metrics['SDR']['mean']:.2f}±{stem_metrics['SDR']['ci95']:.2f} dB "
                    f"| SI-SDR {stem_metrics['SI_SDR']['mean']:.2f}±{stem_metrics['SI_SDR']['ci95']:.2f} dB"
                )
            print(f"  Average inference time per track: {result['average_inference_seconds']:.1f}s")

    # Save all results to JSON
    if args.output_json:
        output_data = {
            "dataset_root": args.dataset_root,
            "subset": args.subset,
            "model_dir": args.model_dir,
            "num_epochs_evaluated": len(checkpoints),
            "epochs": all_results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Wrote evaluation results to {args.output_json}")

        # Also print a summary table
        print("\n=== Summary Across All Epochs ===")
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Vocal SDR':<12} {'Bass SDR':<12} {'Drums SDR':<12} {'Other SDR':<12}")
        print("-" * 85)
        for epoch_key in sorted(all_results.keys()):
            result = all_results[epoch_key]
            epoch = result["epoch"]
            loss = result["training_loss"] if result["training_loss"] is not None else "N/A"
            loss_str = f"{loss:.6f}" if isinstance(loss, float) else str(loss)
            
            metrics = result["metrics"]
            vocal_sdr = metrics["vocal"]["SDR"]["mean"]
            bass_sdr = metrics["bass"]["SDR"]["mean"]
            drums_sdr = metrics["drums"]["SDR"]["mean"]
            other_sdr = metrics["other"]["SDR"]["mean"]
            
            print(f"{epoch:<8} {loss_str:<15} {vocal_sdr:<12.2f} {bass_sdr:<12.2f} {drums_sdr:<12.2f} {other_sdr:<12.2f}")


if __name__ == "__main__":
    main()
