"""Download and prepare SMD, SMAP, and MSL without Unix shell scripts."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
from pathlib import Path
from zipfile import ZipFile


SMD_COMMIT = "7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4"
SMD_URL = (
    "https://github.com/NetManAIOps/OmniAnomaly/archive/"
    f"{SMD_COMMIT}.zip"
)
NASA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "patrickfleith/nasa-anomaly-detection-dataset-smap-msl"
    "?datasetVersionNumber=1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"reuse {destination} sha256={_sha256(destination)}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "FL-Anomaly-Dataset-Prep/1.0"}
    )
    print(f"download {url}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    temporary.replace(destination)
    print(
        f"saved {destination} bytes={destination.stat().st_size} "
        f"sha256={_sha256(destination)}"
    )


def _extract_smd(archive: Path, datasets_root: Path) -> None:
    output = datasets_root / "SMD"
    counts = {"train": 0, "test": 0, "test_label": 0}
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = Path(member.filename).parts
            if "ServerMachineDataset" not in parts or member.is_dir():
                continue
            base_index = parts.index("ServerMachineDataset")
            relative = parts[base_index + 1 :]
            if relative == ("LICENSE",):
                target = output / "LICENSE"
            elif len(relative) == 2 and relative[0] in counts:
                target = output.joinpath(*relative)
                counts[relative[0]] += 1
            else:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    if counts != {"train": 28, "test": 28, "test_label": 28}:
        raise RuntimeError(f"Unexpected SMD extraction counts: {counts}")
    print(f"prepared SMD at {output}: {counts}")


def _extract_telemanom(archive: Path, datasets_root: Path) -> None:
    output = datasets_root / "telemanom"
    counts = {"train": 0, "test": 0, "labels": 0}
    with ZipFile(archive) as bundle:
        for member in bundle.infolist():
            parts = Path(member.filename).parts
            relative: tuple[str, ...] | None = None
            counter: str | None = None
            if member.filename == "labeled_anomalies.csv":
                relative = ("labeled_anomalies.csv",)
                counter = "labels"
            elif (
                len(parts) == 4
                and parts[:2] == ("data", "data")
                and parts[2] in {"train", "test"}
                and parts[3].endswith(".npy")
            ):
                relative = (parts[2], parts[3])
                counter = parts[2]
            if relative is None or counter is None:
                continue
            target = output.joinpath(*relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            counts[counter] += 1
    if counts != {"train": 82, "test": 82, "labels": 1}:
        raise RuntimeError(f"Unexpected Telemanom extraction counts: {counts}")
    print(f"prepared SMAP/MSL at {output}: {counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--dataset", choices=("all", "smd", "nasa"), default="all"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract archives already present in datasets/_downloads",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloads = args.datasets_root / "_downloads"
    smd_archive = downloads / "omni_anomaly.zip"
    nasa_archive = downloads / "nasa_smap_msl.zip"
    if not args.skip_download:
        if args.dataset in {"all", "smd"}:
            _download(SMD_URL, smd_archive)
        if args.dataset in {"all", "nasa"}:
            _download(NASA_URL, nasa_archive)
    if args.dataset in {"all", "smd"}:
        _extract_smd(smd_archive, args.datasets_root)
    if args.dataset in {"all", "nasa"}:
        _extract_telemanom(nasa_archive, args.datasets_root)


if __name__ == "__main__":
    main()
