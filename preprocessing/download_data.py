"""Dataset download helper (DSD100 + optional subset)."""

from __future__ import annotations

import os
import zipfile
from typing import Iterable, Sequence
from urllib.error import URLError

import wget

from preprocessing.input_parameter import get_args

DSD100_URLS: Sequence[str] = ("http://liutkus.net/DSD100.zip",)
DSD100_SUBSET_URLS: Sequence[str] = (
    "https://www.loria.fr/~aliutkus/DSD100subset.zip",
    "https://web.archive.org/web/20240602013333if_/https://members.loria.fr/ALiutkus/DSD100subset.zip",
)


def download_and_extract(urls: Iterable[str], data_dir: str, label: str) -> None:
    """Download from the first working URL and extract into ``data_dir``."""
    os.makedirs(data_dir, exist_ok=True)
    last_error: Exception | None = None

    for url in urls:
        archive_path: str | None = None
        try:
            print(f"Downloading {label} from {url}")
            archive_path = wget.download(url, out=data_dir)
            print()  # newline after wget progress
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            os.remove(archive_path)
            print(f"Finished downloading {label}")
            return
        except (URLError, OSError, zipfile.BadZipFile) as exc:
            last_error = exc
            print(f"Failed to download {label} from {url}: {exc}")
            if archive_path and os.path.exists(archive_path):
                os.remove(archive_path)

    raise RuntimeError(f"Unable to download {label} from provided URLs.") from last_error


def main() -> None:
    args = get_args()
    data_dir = os.path.abspath(args.DATADIR)
    os.makedirs(data_dir, exist_ok=True)

    dsd_root = os.path.join(data_dir, "DSD100")
    subset_root = os.path.join(data_dir, "DSD100subset")

    if not os.path.isdir(dsd_root):
        download_and_extract(DSD100_URLS, data_dir, "DSD100")
    else:
        print("DSD100 already downloaded.")

    if not os.path.isdir(subset_root):
        download_and_extract(DSD100_SUBSET_URLS, data_dir, "DSD100subset")
    else:
        print("DSD100subset already downloaded.")


if __name__ == "__main__":
    main()
