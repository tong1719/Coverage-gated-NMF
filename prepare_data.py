"""Place verified public source files in the experiment directories."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE = ROOT / "candidates" / "c530_complementary_bin_operator_nmf" / "phase_b"
RAMAN_SHA256 = "c426967faac3005b1145385d5f8da5bfa85078084f7c5931b30a9bf9fcb11758"
MASSBANK_SHA = {
    "AC000008": "5ea4261226469f664cdf58476971f337d7fd585f",
    "AC000009": "39d556e0de17c6861875696810e86847e05c40ae",
    "AC000010": "cb15328cbbdc77cdb4d5bf204a3feca5b06a9154",
    "AC000015": "fbb0772e824a9dca2b1fb60c95246db51c44d613",
    "AC000020": "e529c33d47a6a9091479201636a825d7d8adbf02",
    "AC000025": "51a2fd5c28db19d943397d36182164ea482b80bb",
    "AC000030": "301d9d1d298b4d76e315209a609d7d9b3530c94d",
    "AC000035": "3978c9e83fc72e140bdf35750a1ea8bca1088fff",
    "AC000040": "a6dd5e7f68037fe5961e1d085ddcc289c3e6f02e",
    "AC000045": "67fa613cd2d464a68112996588a9546b45be2cc2",
    "AC000050": "896beeff94ae38b338a529d80b1f50134e60ca3a",
    "AC000055": "9d1397d0a6d6763014edd0577dccd71c8d2ae7ee",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def fetch_massbank() -> None:
    target_dir = PHASE / "massbank_2026_03_fixed"
    target_dir.mkdir(parents=True, exist_ok=True)
    for accession, expected_sha in MASSBANK_SHA.items():
        target = target_dir / f"{accession}.json"
        if target.is_file():
            response = json.loads(target.read_text(encoding="utf-8"))
        else:
            url = (
                "https://api.github.com/repos/MassBank/MassBank-data/contents/"
                f"AAFC/MSBNK-AAFC-{accession}.txt?ref=2026.03"
            )
            request = urllib.request.Request(
                url, headers={"Accept": "application/vnd.github+json", "User-Agent": "coverage-gated-nmf"}
            )
            with urllib.request.urlopen(request, timeout=30) as stream:
                response = json.load(stream)
        payload = base64.b64decode(response["content"])
        if response["sha"] != expected_sha or git_blob_sha(payload) != expected_sha:
            raise ValueError(f"MassBank source mismatch: {accession}")
        if not target.is_file():
            target.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        print(f"MassBank {accession}: verified")


def place_raman(source: Path) -> None:
    if not source.is_file() or sha256(source) != RAMAN_SHA256:
        raise ValueError("Raman CSV does not match the recorded source file")
    target_dir = PHASE / "real_raman_oil_data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "raman-oils-raw.csv"
    if target.exists():
        if sha256(target) != RAMAN_SHA256:
            raise ValueError("Existing Raman CSV differs from the recorded source file")
    else:
        shutil.copyfile(source, target)
    print("Raman CSV: verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raman", type=Path, required=True, help="Path to the original downloaded Raman CSV")
    args = parser.parse_args()
    place_raman(args.raman)
    fetch_massbank()


if __name__ == "__main__":
    main()
