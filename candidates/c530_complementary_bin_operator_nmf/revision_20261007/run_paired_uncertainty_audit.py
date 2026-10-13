"""Exact four-seed-block bootstrap for paired Poisson/HALS differences."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HALS = HERE / "results_operator_coupled_hals" / "records.json"
STOP = HERE / "results_full_poisson_stopping_audit"
FACTORIAL = HERE.parent / "phase_b" / "results_acquisition_factorial" / "records.json"
OUT = HERE / "results_paired_uncertainty_audit"


def record_key(record):
    if record["dataset"] == "controlled":
        return ("controlled", record["regime"], int(record["seed"]))
    return ("massbank", int(record["panel"]), record["depth"], int(record["seed"]))


def load():
    hals = json.loads(HALS.read_text(encoding="utf-8"))
    extended = {}
    for path in STOP.glob("*.json"):
        if path.name == "report.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        key = tuple([record["dataset"], *record["key"]])
        assert key not in extended, key
        extended[key] = record
    assert len(hals) == len(extended) == 32
    rows = []
    for record in hals:
        key = record_key(record)
        assert key in extended, key
        later = extended[key]
        assert np.isclose(record["poisson_heldout_nmse"],
                          later["metrics_420"]["heldout_nmse"],
                          rtol=0, atol=1e-12), key
        assert np.isclose(record["poisson_basis_cosine"],
                          later["metrics_420"]["matched_basis_cosine"],
                          rtol=0, atol=1e-12), key
        for checkpoint, nmse, cosine in (
            ("420", float(record["poisson_heldout_nmse"]),
             float(record["poisson_basis_cosine"])),
            ("extended", float(later["metrics_extended"]["heldout_nmse"]),
             float(later["metrics_extended"]["matched_basis_cosine"])),
        ):
            hals_nmse = float(record["heldout_nmse"])
            hals_cosine = float(record["basis_cosine"])
            assert hals_nmse > 0
            values = np.array([hals_nmse, hals_cosine, nmse, cosine])
            assert np.isfinite(values).all()
            rows.append({
                "dataset": record["dataset"],
                "stratum": (record["regime"] if record["dataset"] == "controlled"
                            else f"panel{record['panel']}_{record['depth']}"),
                "seed": int(record["seed"]),
                "checkpoint": checkpoint,
                "hals_nmse": hals_nmse,
                "poisson_nmse": nmse,
                "relative_nmse_reduction": (hals_nmse - nmse) / hals_nmse,
                "hals_basis_cosine": hals_cosine,
                "poisson_basis_cosine": cosine,
                "basis_cosine_difference": cosine - hals_cosine,
                "poisson_best_converged": bool(later["best_extended_converged"])
                if checkpoint == "extended" else False,
            })
    return rows


def summarize(rows):
    result = {}
    for dataset in ("controlled", "massbank"):
        for checkpoint in ("420", "extended"):
            selected = [row for row in rows if row["dataset"] == dataset
                        and row["checkpoint"] == checkpoint]
            assert len(selected) == 16
            strata = sorted({row["stratum"] for row in selected})
            seeds = sorted({row["seed"] for row in selected})
            assert len(strata) == len(seeds) == 4
            assert all(sum(row["stratum"] == stratum and row["seed"] == seed
                           for row in selected) == 1
                       for stratum in strata for seed in seeds)
            data = {}
            for metric in ("relative_nmse_reduction", "basis_cosine_difference"):
                matrix = np.array([
                    [next(row[metric] for row in selected
                          if row["stratum"] == stratum and row["seed"] == seed)
                     for stratum in strata]
                    for seed in seeds
                ], dtype=float)
                assert matrix.shape == (4, 4)
                estimates = []
                for indices in itertools.product(range(4), repeat=4):
                    estimates.append(float(np.median(matrix[list(indices)].ravel())))
                pooled = matrix.ravel()
                data[metric] = {
                    "median": float(np.median(pooled)),
                    "q25": float(np.percentile(pooled, 25)),
                    "q75": float(np.percentile(pooled, 75)),
                    "seed_block_bootstrap_95": [
                        float(np.percentile(estimates, 2.5)),
                        float(np.percentile(estimates, 97.5)),
                    ],
                    "wins": int(np.sum(pooled > 0)),
                    "n_cells": len(pooled),
                    "enumerated_bootstrap_resamples": len(estimates),
                }
            result[f"{dataset}_{checkpoint}"] = {
                "strata": strata,
                "seed_ids": seeds,
                "metrics": data,
                "n_best_converged": sum(row["poisson_best_converged"]
                                        for row in selected) if checkpoint == "extended" else None,
            }
    return result


def factorial_pairs():
    records = json.loads(FACTORIAL.read_text(encoding="utf-8"))
    assert len(records) == 80
    arm_means = {
        arm: float(np.mean([record["heldout_nmse"] for record in records
                            if record["arm"] == arm]))
        for arm in ("adjacent_incomplete", "alternating_incomplete", "repeated")
    }
    assert arm_means["alternating_incomplete"] == min(arm_means.values())
    index = {(record["regime"], int(record["seed"]), record["arm"]): record
             for record in records}
    assert len(index) == 80
    strata = sorted({record["regime"] for record in records})
    seeds = sorted({int(record["seed"]) for record in records})
    assert len(strata) == len(seeds) == 4
    rows = []
    for seed in seeds:
        for stratum in strata:
            full = index[(stratum, seed, "complete")]
            incomplete = index[(stratum, seed, "alternating_incomplete")]
            full_nmse = float(full["heldout_nmse"])
            incomplete_nmse = float(incomplete["heldout_nmse"])
            full_basis = float(full["basis_cosine"])
            incomplete_basis = float(incomplete["basis_cosine"])
            assert incomplete_nmse > 0
            assert np.isfinite([full_nmse, incomplete_nmse,
                                full_basis, incomplete_basis]).all()
            rows.append({
                "regime": stratum,
                "seed": seed,
                "complete_nmse": full_nmse,
                "alternating_incomplete_nmse": incomplete_nmse,
                "relative_nmse_reduction": (incomplete_nmse - full_nmse) / incomplete_nmse,
                "complete_basis_cosine": full_basis,
                "alternating_incomplete_basis_cosine": incomplete_basis,
                "basis_cosine_gain": full_basis - incomplete_basis,
            })
    summary = {"strata": strata, "seed_ids": seeds,
               "fixed_incomplete_arm": "alternating_incomplete",
               "arm_mean_heldout_nmse": arm_means, "metrics": {}}
    for metric in ("relative_nmse_reduction", "basis_cosine_gain"):
        matrix = np.array([
            [next(row[metric] for row in rows
                  if row["seed"] == seed and row["regime"] == stratum)
             for stratum in strata]
            for seed in seeds
        ])
        estimates = [float(np.median(matrix[list(indices)].ravel()))
                     for indices in itertools.product(range(4), repeat=4)]
        values = matrix.ravel()
        summary["metrics"][metric] = {
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "seed_block_bootstrap_95": [
                float(np.percentile(estimates, 2.5)),
                float(np.percentile(estimates, 97.5)),
            ],
            "wins": int(np.sum(values > 0)),
            "n_cells": len(values),
            "enumerated_bootstrap_resamples": len(estimates),
        }
    assert len(rows) == 16
    return rows, summary


def main():
    rows = load()
    summary = summarize(rows)
    factorial_rows, factorial_summary = factorial_pairs()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "paired_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "factorial_paired_cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(factorial_rows[0]))
        writer.writeheader()
        writer.writerows(factorial_rows)
    report = {
        "protocol": "PROTOCOL_PAIRED_UNCERTAINTY_AUDIT.md",
        "n_paired_cell_checkpoint_rows": len(rows),
        "conditional_on_fixed_scenarios_and_profiles": True,
        "no_physical_sample_inference": True,
        "summary": summary,
        "factorial_complete_vs_alternating": factorial_summary,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert len(rows) == 64
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
