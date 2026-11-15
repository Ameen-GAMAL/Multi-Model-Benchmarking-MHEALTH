"""Integrity-checked acquisition of the pinned UCI MHEALTH archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo

DATA_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
DATA_DOI = "https://doi.org/10.24432/C5TW22"
ARCHIVE_NAME = "mhealth+dataset.zip"
ARCHIVE_SHA256 = "16ad0ce709f3f00df18f348610d15bce0884b79e2143f57f446493673f02b8e0"
ARCHIVE_BYTES = 75_567_983
COLLECTION_DIR = "MHEALTHDATASET"
SOURCE_MANIFEST = ".dataset-provenance.json"
# These fingerprints come from the verified archive above. Cache verification
# uses this source-controlled reference rather than trusting a writable marker.
_PINNED_MEMBERS = {
    "README.txt": (5_197, "70305a3d49cd2f319b539568a1b7fdfc6232c881d457464259e39cbea79ba8ad"),
    "mHealth_subject1.log": (30_019_680, "da8d1bbf644a206886333f945a1dd7fe5081382b3bb3ee42821f08ad765732e4"),
    "mHealth_subject2.log": (24_446_574, "20e288750fe9f4f0495686fabf1500bfe655a549f0d2c3a9be5aa6456461b8f1"),
    "mHealth_subject3.log": (22_788_438, "90ead135c9fa9f1188e3b1f7afc0c95899a0614a15917cf3590f52923d6e03fd"),
    "mHealth_subject4.log": (21_833_916, "3b66f87a292f4cc457aea18cb7e15a10c18055c803df27f5284c6a1c5751a469"),
    "mHealth_subject5.log": (22_285_024, "380e526a85bde64baebd1d1a2478c4cfad947a4ab994c8b39b8dcda195f78a30"),
    "mHealth_subject6.log": (18_243_582, "4b8bc81186229e096937f37f9ae070017ee653005b21f88fcd0504e9195f8926"),
    "mHealth_subject7.log": (19_460_454, "5e372b2fad64a3ebaadd103ec09cde68d448ef5b707b010d02f438b8b9203b1f"),
    "mHealth_subject8.log": (24_087_426, "c6ddb8926be4c193890a28cc84cafe35c16a4bc14f23c3e696d504643652a61b"),
    "mHealth_subject9.log": (25_148_835, "fa68f90a9dfeda47abce41ebc1dec8ce55f66dc56d513ab90a6ae20ad045f2c9"),
    "mHealth_subject10.log": (18_500_501, "382d3d7b58e1fe24679fb241fd7e48ce1992757f6b1b237a56cd660e8f1fabda"),
}
PINNED_SOURCE = {
    "collection": "UCI MHEALTH (dataset 319, published 2014)",
    "doi": DATA_DOI,
    "license": "CC BY 4.0",
    "url": DATA_URL,
    "archive_sha256": ARCHIVE_SHA256,
    "archive_bytes": ARCHIVE_BYTES,
    "files": {name: {"bytes": size, "sha256": digest} for name, (size, digest) in _PINNED_MEMBERS.items()},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(path: Path) -> None:
    if path.stat().st_size != ARCHIVE_BYTES or sha256_file(path) != ARCHIVE_SHA256:
        raise ValueError(f"Archive size or SHA-256 did not match the pinned UCI dataset: {path}")


def validate_members(infos: list[ZipInfo]) -> dict[str, ZipInfo]:
    """Check every member before writing any of the exact expected leaf names."""
    files = {}
    seen = set()
    for info in infos:
        name = info.filename
        parts = PurePosixPath(name).parts
        if name in seen or "\\" in name or ".." in parts or PurePosixPath(name).is_absolute():
            raise ValueError(f"Invalid or duplicate archive member: {name!r}")
        seen.add(name)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if name == COLLECTION_DIR + "/" and info.is_dir() and kind in (0, stat.S_IFDIR):
            continue
        if (
            len(parts) != 2
            or parts[0] != COLLECTION_DIR
            or parts[1] not in _PINNED_MEMBERS
            or name != f"{COLLECTION_DIR}/{parts[1]}"
            or info.is_dir()
            or kind not in (0, stat.S_IFREG)
        ):
            raise ValueError(f"Unexpected archive member: {name!r}")
        files[parts[1]] = info
    if files.keys() != _PINNED_MEMBERS.keys() or any(
        info.file_size != _PINNED_MEMBERS[name][0] for name, info in files.items()
    ):
        raise ValueError("Archive contents do not match the pinned UCI dataset.")
    return files


def _download(archive: Path) -> None:
    print(f"Downloading {DATA_URL}", flush=True)
    with tempfile.TemporaryDirectory(prefix="mhealth-download-", dir=archive.parent) as staging:
        partial = Path(staging) / "download.zip"
        received = 0
        with urlopen(DATA_URL, timeout=120) as response, partial.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                received += len(block)
                if received > ARCHIVE_BYTES:
                    raise ValueError("The response exceeded the pinned archive size.")
                output.write(block)
        validate_archive(partial)
        os.replace(partial, archive)


def read_verified_source(target: Path) -> dict | None:
    """Return local extraction provenance if it identifies this pinned release."""
    marker = target / SOURCE_MANIFEST
    if not marker.is_file():
        return None
    source = json.loads(marker.read_text(encoding="utf-8"))
    if source != PINNED_SOURCE:
        raise ValueError(f"Extraction record differs from the pinned UCI dataset: {marker}.")
    return source


def ensure_dataset(raw_dir: Path, archive: Path | None = None) -> Path:
    target = raw_dir / COLLECTION_DIR
    if target.exists():
        source = read_verified_source(target)
        if source is None:
            raise ValueError(f"Existing directory lacks provenance: {target}. Choose another --raw-dir.")
        for name, record in source["files"].items():
            path = target / name
            if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Existing extracted file differs from its recorded checksum: {path}")
        return target

    raw_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = raw_dir / ARCHIVE_NAME
        if not archive.exists():
            _download(archive)
    validate_archive(archive)

    with tempfile.TemporaryDirectory(prefix="mhealth-extract-", dir=raw_dir) as staging, ZipFile(archive) as bundle:
        files = validate_members(bundle.infolist())
        staged_target = Path(staging) / COLLECTION_DIR
        staged_target.mkdir()
        for name, member in sorted(files.items()):
            output = staged_target / name
            with bundle.open(member) as original, output.open("xb") as copied:
                shutil.copyfileobj(original, copied, length=1024 * 1024)
            if sha256_file(output) != _PINNED_MEMBERS[name][1]:
                raise ValueError(f"Extracted file hash differs from the pinned UCI dataset: {name}.")
        (staged_target / SOURCE_MANIFEST).write_text(json.dumps(PINNED_SOURCE, indent=2) + "\n", encoding="utf-8")
        os.replace(staged_target, target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--archive", type=Path, help="Use a separately downloaded copy of the pinned archive.")
    args = parser.parse_args(argv)
    try:
        path = ensure_dataset(args.raw_dir, args.archive)
    except (OSError, ValueError, BadZipFile) as exc:
        parser.exit(1, f"Dataset setup failed. {exc}\n")
    print(f"Verified UCI MHEALTH data at {path}")
    return 0
