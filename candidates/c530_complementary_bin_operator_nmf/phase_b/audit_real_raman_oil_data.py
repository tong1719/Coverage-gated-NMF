"""Blind structural and quality audit for the public Raman oil-mixture data.

This script profiles only the raw data and official split labels.  It does not
fit, initialize, or score any factorization model, so it is safe to run before
the real-mixture protocol is locked.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "real_raman_oil_data" / "raman-oils-raw.csv"
OUT = HERE / "real_raman_oil_data" / "blind_data_quality_audit.json"
META = [
    "Date",
    "Obs / day",
    "Spectra / Mixture",
    "EVOO (%)",
    "Sunflower (%)",
    "Tps(ms)",
    "Scans",
    "P(mW)",
    "Type",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.rstrip("%"), errors="coerce")


def main() -> int:
    frame = pd.read_csv(DATA, sep=";", dtype=str, keep_default_na=False)
    missing_meta = [name for name in META if name not in frame.columns]
    if missing_meta:
        raise ValueError(f"Missing documented metadata columns: {missing_meta}")

    spectral_columns = [name for name in frame.columns if name not in META]
    wavelengths = np.asarray(
        [float(name.replace(",", ".")) for name in spectral_columns], dtype=float
    )
    spectra = frame[spectral_columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    evoo = parse_percent(frame["EVOO (%)"])
    sunflower = parse_percent(frame["Sunflower (%)"])
    split = frame["Type"].str.strip().str.upper()
    dates = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce")

    composite_key = frame[["Date", "Obs / day", "Spectra / Mixture"]].astype(str)
    row_spectral_hashes = [
        hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
        for row in spectra
    ]
    hash_counts = Counter(row_spectral_hashes)
    repeated_hash_rows = sum(count for count in hash_counts.values() if count > 1)

    concentration_counts = (
        pd.DataFrame({"split": split, "evoo": evoo})
        .groupby(["split", "evoo"], dropna=False)
        .size()
    )
    concentration_by_split = {
        label: {
            str(int(level) if float(level).is_integer() else float(level)): int(count)
            for (_, level), count in concentration_counts.items()
            if _ == label and pd.notna(level)
        }
        for label in sorted(split.unique())
    }
    dates_by_split = {
        label: sorted(
            value.strftime("%Y-%m-%d")
            for value in dates[split == label].dropna().drop_duplicates()
        )
        for label in sorted(split.unique())
    }

    per_group = (
        pd.DataFrame({"date": frame["Date"], "evoo": evoo, "split": split})
        .groupby(["date", "evoo", "split"], dropna=False)
        .size()
    )
    repeated_split_sets = []
    for value, count in hash_counts.items():
        if count > 1:
            repeated_split_sets.append(sorted(set(split[np.asarray(row_spectral_hashes) == value])))
    cross_split_repeated_groups = sum(len(values) > 1 for values in repeated_split_sets)

    finite = np.isfinite(spectra)
    report = {
        "audit_scope": "blind structural/data-quality audit; no factorization fitted",
        "source_sha256": sha256(DATA),
        "shape": {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "metadata_columns": len(META),
            "spectral_channels": len(spectral_columns),
        },
        "grain_and_uniqueness": {
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "duplicate_composite_keys": int(composite_key.duplicated().sum()),
            "repeated_spectral_hash_groups": int(sum(v > 1 for v in hash_counts.values())),
            "rows_in_repeated_spectral_hash_groups": int(repeated_hash_rows),
            "duplicate_rows_beyond_first": int(
                sum(max(0, value - 1) for value in hash_counts.values())
            ),
            "cross_split_repeated_spectral_hash_groups": int(cross_split_repeated_groups),
            "replicate_count_min": int(per_group.min()),
            "replicate_count_max": int(per_group.max()),
            "replicate_count_distribution": {
                str(int(key)): int(value)
                for key, value in per_group.value_counts().sort_index().items()
            },
        },
        "completeness_and_validity": {
            "empty_cells_total": int((frame == "").to_numpy().sum()),
            "nonfinite_spectral_cells": int((~finite).sum()),
            "rows_with_nonfinite_spectral_cells": int((~finite).any(axis=1).sum()),
            "negative_spectral_cells": int((spectra[finite] < 0).sum()),
            "zero_spectral_cells": int((spectra[finite] == 0).sum()),
            "invalid_dates": int(dates.isna().sum()),
            "invalid_evoo_percent": int(evoo.isna().sum()),
            "invalid_sunflower_percent": int(sunflower.isna().sum()),
            "percent_sum_not_100": int((~np.isclose(evoo + sunflower, 100.0)).sum()),
            "unexpected_split_labels": sorted(
                set(split.unique()) - {"CALIBRATION", "VALIDATION", "PREDICTION"}
            ),
            "documentation_spelling_note": (
                "The PDF says VALISATION, whereas the CSV consistently uses VALIDATION. "
                "The CSV label is used verbatim."
            ),
            "unexpected_acquisition_time_values": sorted(frame["Tps(ms)"].unique()),
            "unexpected_scan_values": sorted(frame["Scans"].unique()),
            "unexpected_power_values": sorted(frame["P(mW)"].unique()),
        },
        "spectral_axis": {
            "minimum_cm_inverse": float(wavelengths.min()),
            "maximum_cm_inverse": float(wavelengths.max()),
            "strictly_increasing": bool(np.all(np.diff(wavelengths) > 0)),
            "unique": bool(len(np.unique(wavelengths)) == len(wavelengths)),
            "spacing_min": float(np.diff(wavelengths).min()),
            "spacing_median": float(np.median(np.diff(wavelengths))),
            "spacing_max": float(np.diff(wavelengths).max()),
        },
        "intensity_profile": {
            "minimum": float(np.nanmin(spectra)),
            "q01": float(np.nanquantile(spectra, 0.01)),
            "median": float(np.nanmedian(spectra)),
            "q99": float(np.nanquantile(spectra, 0.99)),
            "maximum": float(np.nanmax(spectra)),
            "row_total_minimum": float(np.nansum(spectra, axis=1).min()),
            "row_total_median": float(np.median(np.nansum(spectra, axis=1))),
            "row_total_maximum": float(np.nansum(spectra, axis=1).max()),
        },
        "official_split": {
            "row_counts": {str(k): int(v) for k, v in split.value_counts().items()},
            "concentration_counts": concentration_by_split,
            "dates": dates_by_split,
            "date_overlap": {
                f"{left}__{right}": sorted(
                    set(dates_by_split[left]).intersection(dates_by_split[right])
                )
                for i, left in enumerate(sorted(dates_by_split))
                for right in sorted(dates_by_split)[i + 1 :]
            },
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
