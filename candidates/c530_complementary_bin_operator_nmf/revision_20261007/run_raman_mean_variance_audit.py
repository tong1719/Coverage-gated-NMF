"""Descriptive replicate mean--variance check for the public Raman file."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run_raman_loss_iteration_sensitivity import RA

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_raman_mean_variance_audit_mixture_blocks"

def summary(values):
    x = np.asarray(values, dtype=float)
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def describe_pairs(means, variances):
    mean = np.concatenate(means)
    var = np.concatenate(variances)
    positive = (mean > 0) & (var > 0)
    slope, intercept = np.polyfit(np.log(mean[positive]), np.log(var[positive]), 1)
    return {
        "n_group_channels": int(len(mean)),
        "n_positive_pairs": int(positive.sum()),
        "variance_over_mean": summary(var[positive] / mean[positive]),
        "log_variance_on_log_mean_slope": float(slope),
        "log_variance_on_log_mean_intercept": float(intercept),
    }


def main():
    original = pd.read_csv(RA.DATA, sep=";", dtype=str, keep_default_na=False)
    assert len(original) == 605 and len(original) % 5 == 0
    valid_original_blocks = 0
    replicate_label_anomalies = []
    for start in range(0, len(original), 5):
        block = original.iloc[start:start + 5]
        assert all(block[column].nunique() == 1 for column in ("Date", "EVOO (%)", "Type"))
        observation_numbers = block["Obs / day"].astype(int).to_numpy()
        assert np.array_equal(np.diff(observation_numbers), np.ones(4, dtype=int))
        if sorted(block["Spectra / Mixture"]) != [f"S{i}" for i in range(1, 6)]:
            replicate_label_anomalies.append({
                "first_original_row_zero_based": start,
                "labels": list(block["Spectra / Mixture"]),
            })
        valid_original_blocks += 1
    _, subsets, duplicates_removed = RA.load_data()
    raw_means, raw_variances = [], []
    normalized_means, normalized_variances = [], []
    group_sizes = Counter()
    row_total_cvs = []
    used_rows = 0
    groups = 0
    for split, (frame, raw) in subsets.items():
        group_id = frame["_original_index"].to_numpy(dtype=int) // 5
        for _, indices in frame.groupby(group_id, sort=True).groups.items():
            loc = np.asarray(list(indices), dtype=int)
            if len(loc) < 3:
                continue
            assert (frame.iloc[loc]["_split"] == split).all()
            assert frame.iloc[loc]["Date"].nunique() == 1
            assert frame.iloc[loc]["_evoo"].nunique() == 1
            x = raw[loc]
            z = RA.normalize_fine(x)
            raw_means.append(x.mean(axis=0))
            raw_variances.append(x.var(axis=0, ddof=1))
            normalized_means.append(z.mean(axis=0))
            normalized_variances.append(z.var(axis=0, ddof=1))
            totals = x.sum(axis=1)
            row_total_cvs.append(float(totals.std(ddof=1) / totals.mean()))
            group_sizes[str(len(loc))] += 1
            groups += 1
            used_rows += len(loc)
    report = {
        "protocol": "PROTOCOL_RAMAN_MEAN_VARIANCE_AUDIT.md",
        "duplicates_removed": duplicates_removed,
        "verified_original_five_spectrum_blocks": valid_original_blocks,
        "replicate_label_anomalies": replicate_label_anomalies,
        "retained_rows": int(sum(len(raw) for _, raw in subsets.values())),
        "groups_with_three_or_more_rows": groups,
        "rows_in_groups": used_rows,
        "group_size_counts": dict(sorted(group_sizes.items())),
        "raw_intensity": describe_pairs(raw_means, raw_variances),
        "total_normalized_intensity": describe_pairs(normalized_means, normalized_variances),
        "within_group_raw_row_total_cv": summary(row_total_cvs),
        "interpretation_boundary": (
            "Mixture replicates and processed intensity units cannot establish "
            "a conditional photon-count Poisson law."
        ),
        "checks": {
            "duplicate_count_matches_submission": duplicates_removed == 14,
            "row_count_matches_submission": sum(len(raw) for _, raw in subsets.values()) == 591,
            "spectral_channel_count_matches_submission": all(raw.shape[1] == 710 for _, raw in subsets.values()),
            "original_mixture_blocks_valid": valid_original_blocks == 121,
            "metadata_anomaly_count_expected": len(replicate_label_anomalies) == 2,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
