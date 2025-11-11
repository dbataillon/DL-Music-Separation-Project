import os
import shutil
import zipfile
from typing import Iterable, List, Union
from urllib.error import URLError

import wget

from input_parameter import get_args


def download_unzip(
    urls: Union[str, Iterable[str]],
    data_dir: str,
    label: str,
) -> None:
    """
    Attempt to download/extract ``label`` from the list of URLs until one succeeds.
    """
    url_candidates: List[str] = (
        list(urls) if isinstance(urls, (list, tuple)) else [str(urls)]
    )
    last_error: Exception | None = None

    for url in url_candidates:
        archive_path: str | None = None
        try:
            print(f"Downloading {label} from {url}")
            archive_path = wget.download(url, out=data_dir)
            print()  # ensure the next log starts on a new line
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Finished downloading {label}")
            os.remove(archive_path)
            return
        except (URLError, OSError, zipfile.BadZipFile) as err:
            last_error = err
            print(f"Failed to download {label} from {url}: {err}")
            if archive_path and os.path.exists(archive_path):
                os.remove(archive_path)

    raise RuntimeError(f"Unable to download {label} from the provided URLs") from last_error

args = get_args()

if not os.path.isdir(args.DATADIR):
    os.mkdir(args.DATADIR)

data_dir = args.DATADIR

DSD100_url = ['http://liutkus.net/DSD100.zip']
DSD100subset_urls = [
    'https://www.loria.fr/~aliutkus/DSD100subset.zip',
    'https://web.archive.org/web/20240602013333if_/https://members.loria.fr/ALiutkus/DSD100subset.zip',
]

DSD100_dir = os.path.join(data_dir, 'DSD100')
DSD100subset = os.path.join(data_dir, 'DSD100subset')

if not os.path.isdir(DSD100_dir):
    download_unzip(DSD100_url, data_dir, 'DSD100')
else:
    print("DSD100 already downlaoded!!")

if not os.path.isdir(DSD100subset):
    download_unzip(DSD100subset_urls, data_dir, 'DSD100subset')
else:
    print("DSD100subset already downlaoded!!")
