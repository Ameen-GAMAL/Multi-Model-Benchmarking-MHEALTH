"""Download and extract the UCI MHEALTH dataset into ``data/raw/``.

Usage
-----
    python scripts/download_data.py

The dataset (~75 MB zipped) is fetched from the UCI Machine Learning Repository
and unpacked to ``data/raw/MHEALTHDATASET/``.  Re-running is a no-op once the
log files are present.
"""
from __future__ import annotations

import io
import os
import sys
import urllib.request
import zipfile

DATA_URL = "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip"
RAW_DIR = os.path.join("data", "raw")
MARKER = os.path.join(RAW_DIR, "MHEALTHDATASET", "mHealth_subject1.log")


def main() -> int:
    if os.path.exists(MARKER):
        print(f"Dataset already present at {os.path.dirname(MARKER)} -- nothing to do.")
        return 0

    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"Downloading MHEALTH dataset from:\n  {DATA_URL}")
    try:
        with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
            payload = resp.read()
    except Exception as exc:  # noqa: BLE001 - surface any network error clearly
        print(f"ERROR: download failed ({exc}).", file=sys.stderr)
        print(
            "You can download the zip manually and extract it so that\n"
            f"  {MARKER}\nexists, then re-run the benchmark.",
            file=sys.stderr,
        )
        return 1

    print(f"Downloaded {len(payload) / 1e6:.1f} MB. Extracting...")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(RAW_DIR)

    if os.path.exists(MARKER):
        print(f"Done. Data ready at {os.path.dirname(MARKER)}")
        return 0
    print("ERROR: extraction finished but expected files are missing.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
