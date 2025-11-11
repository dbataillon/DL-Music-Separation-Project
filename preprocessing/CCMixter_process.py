from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import librosa

from preprocessing.util import SaveSpectrogram
from preprocessing.input_parameter import get_args

SPECTRO_DIR = Path("./Spectrogram")
TRAIN_SPLITS = ("Dev",)  # keep Test unseen
DATASET_NAMES = ("DSD100", "DSD100subset")


def find_dataset_roots(base_dir: Path) -> list[Path]:
    """Return dataset roots supporting both parent and direct dataset paths."""
    roots: list[Path] = []

    def _is_dataset_root(path: Path) -> bool:
        return (path / "Mixtures").is_dir() and (path / "Sources").is_dir()

    if _is_dataset_root(base_dir):
        roots.append(base_dir)

    for name in DATASET_NAMES:
        candidate = base_dir / name
        if candidate.is_dir() and _is_dataset_root(candidate):
            roots.append(candidate)

    if not roots:
        raise FileNotFoundError(
            f"No dataset directories found under {base_dir}. "
            f"Expected one of {', '.join(DATASET_NAMES)} "
            "or a directory containing Mixtures/ and Sources/."
        )

    # Remove duplicates while preserving order.
    seen = set()
    deduped: list[Path] = []
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def iter_tracks(dataset_root: Path) -> Iterable[tuple[str, Path, Path]]:
    for split in TRAIN_SPLITS:
        mix_dir = dataset_root / "Mixtures" / split
        sources_dir = dataset_root / "Sources" / split
        if not mix_dir.is_dir() or not sources_dir.is_dir():
            continue
        for track in sorted(os.listdir(sources_dir)):
            mix_track = mix_dir / track
            source_track = sources_dir / track
            if mix_track.is_dir() and source_track.is_dir():
                yield track, mix_track, source_track


def process_track(track_name: str, mix_dir: Path, source_dir: Path) -> bool:
    out_file = SPECTRO_DIR / f"{track_name}.npz"
    if out_file.exists():
        print(f"[skip] {track_name} already cached")
        return False

    def _load(filename: str):
        return librosa.load(source_dir / filename, sr=None)[0]

    bass = _load("bass.wav")
    drums = _load("drums.wav")
    other = _load("other.wav")
    vocal = _load("vocals.wav")
    mix = librosa.load(mix_dir / "mixture.wav", sr=None)[0]

    print(f"[save] {track_name}")
    SaveSpectrogram(
        mix,
        bass,
        drums,
        other,
        vocal,
        track_name,
        out_dir=str(SPECTRO_DIR),
    )
    return True


def main() -> None:
    args = get_args()
    base_dir = Path(args.DATADIR).expanduser().resolve()

    SPECTRO_DIR.mkdir(parents=True, exist_ok=True)

    dataset_roots = find_dataset_roots(base_dir)
    print(f"Found datasets: {[str(p) for p in dataset_roots]}")

    total = 0
    failures: list[tuple[str, Exception]] = []
    for root in dataset_roots:
        for track_name, mix_dir, source_dir in iter_tracks(root):
            try:
                if process_track(track_name, mix_dir, source_dir):
                    total += 1
            except Exception as exc:  # keep going; report later
                failures.append((track_name, exc))

    if total == 0:
        print("No tracks were processed; check that Mixtures/Sources contain audio.")
    else:
        print(f"Complete! Cached {total} tracks in {SPECTRO_DIR}.")

    if failures:
        print("\nThe following tracks failed:")
        for name, exc in failures:
            print(f" - {name}: {exc}")
        raise RuntimeError(f"{len(failures)} track(s) failed; rerun after fixing the issue.")


if __name__ == "__main__":
    main()
