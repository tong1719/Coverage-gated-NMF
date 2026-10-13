"""Independent integrity and locked-gate verifier for the Raman oil experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "real_raman_oil_data" / "raman-oils-raw.csv"
RESULTS = HERE / "results_real_raman_oil"
EXPECTED_HASH = "c426967faac3005b1145385d5f8da5bfa85078084f7c5931b30a9bf9fcb11758"
METHODS = {
    "c530_complete_operator",
    "c530_complete_operator_rank3_sensitivity",
    "c530_incomplete_operator",
    "fine_grid_oracle",
    "uniform_backproject_kl_nmf",
    "uniform_backproject_mcr_als",
    "wrong_operator",
}
SHIFTS = tuple(range(5))
EXPECTED_SEEDS = {shift: {53800 + 100 * shift, 53801 + 100 * shift} for shift in SHIFTS}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def operator(p: int, width: int, origin: int) -> np.ndarray:
    starts = list(range(-origin, p, width))
    matrix = np.zeros((p, len(starts)))
    for column, start in enumerate(starts):
        matrix[max(0, start) : min(p, start + width), column] = 1.0
    return matrix[:, matrix.sum(axis=0) > 0]


def close(left, right, tolerance=1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=tolerance))


def main() -> int:
    records = json.loads((RESULTS / "records.json").read_text(encoding="utf-8"))
    report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))
    checks = {}
    checks["source_hash"] = digest(DATA) == EXPECTED_HASH

    raw = pd.read_csv(DATA, sep=";", dtype=str, keep_default_na=False)
    spectral = raw.iloc[:, 9:].astype(float)
    duplicate = spectral.duplicated(keep="first")
    retained = raw.loc[~duplicate].copy()
    counts = retained["Type"].str.upper().value_counts().to_dict()
    checks["deduplication"] = int(duplicate.sum()) == 14 and counts == {
        "CALIBRATION": 341,
        "VALIDATION": 161,
        "PREDICTION": 89,
    }

    keys = [(row["method"], row["shift"], row["seed"]) for row in records]
    checks["record_count"] = len(records) == 70
    checks["unique_record_keys"] = len(keys) == len(set(keys))
    checks["method_set"] = {row["method"] for row in records} == METHODS
    checks["ten_cells_per_method"] = all(
        sum(row["method"] == method for row in records) == 10 for method in METHODS
    )
    checks["locked_shifts_and_seeds"] = all(
        row["shift"] in SHIFTS and row["seed"] in EXPECTED_SEEDS[row["shift"]]
        for row in records
    )
    numeric_values = []
    for row in records:
        numeric_values.extend(
            [
                row["runtime_seconds"],
                row["endpoint_basis_cosine"],
                row["validation"]["nmse"],
                row["validation"]["cosine"],
                row["validation"]["evoo_spearman"],
                row["prediction"]["nmse"],
                row["prediction"]["cosine"],
                row["prediction"]["evoo_spearman"],
            ]
        )
        if row["top_two_cosine"] is not None:
            numeric_values.append(row["top_two_cosine"])
        if row["start_objectives"] is not None:
            numeric_values.extend(row["start_objectives"])
    checks["finite_metrics"] = bool(np.isfinite(numeric_values).all())
    checks["metric_ranges"] = all(
        row["prediction"]["nmse"] >= 0
        and -1 <= row["prediction"]["cosine"] <= 1 + 1e-12
        and -1 <= row["prediction"]["evoo_spearman"] <= 1
        for row in records
    )

    candidate = [row for row in records if row["method"] == "c530_complete_operator"]
    means = {
        method: float(
            np.mean(
                [row["prediction"]["nmse"] for row in records if row["method"] == method]
            )
        )
        for method in METHODS
    }
    candidate_nmse = means["c530_complete_operator"]
    candidate_cosine = float(np.mean([row["prediction"]["cosine"] for row in candidate]))
    candidate_rho = float(np.mean([row["prediction"]["evoo_spearman"] for row in candidate]))
    endpoint_cosine = float(np.mean([row["endpoint_basis_cosine"] for row in candidate]))
    nonoperators = ["uniform_backproject_kl_nmf", "uniform_backproject_mcr_als"]
    best_nonoperator = min(nonoperators, key=means.get)
    gain = 1 - candidate_nmse / means[best_nonoperator]
    shift_wins = 0
    for shift in SHIFTS:
        c_value = np.mean(
            [row["prediction"]["nmse"] for row in candidate if row["shift"] == shift]
        )
        b_value = np.mean(
            [
                row["prediction"]["nmse"]
                for row in records
                if row["method"] == best_nonoperator and row["shift"] == shift
            ]
        )
        shift_wins += int(c_value < b_value)
    incomplete = {
        (row["shift"], row["seed"]): row
        for row in records
        if row["method"] == "c530_incomplete_operator"
    }
    complete_wins = sum(
        row["prediction"]["nmse"]
        < incomplete[(row["shift"], row["seed"])]["prediction"]["nmse"]
        for row in candidate
    )
    basin_cells = sum(row["top_two_cosine"] >= 0.95 for row in candidate)

    complete_matrix = np.concatenate([operator(710, 8, value) for value in range(8)], axis=1)
    incomplete_matrix = np.concatenate(
        [operator(710, 8, value) for value in (0, 0, 1, 1, 2, 2, 3, 3)], axis=1
    )
    complete_rank = int(np.linalg.matrix_rank(complete_matrix))
    incomplete_rank = int(np.linalg.matrix_rank(incomplete_matrix))
    checks["coverage_recomputed"] = complete_rank == 710 and incomplete_rank == 354

    gates = {
        "g1_coverage": complete_rank == 710 and incomplete_rank < 710,
        "g2_prediction_nmse": candidate_nmse <= 0.100,
        "g3_prediction_cosine": candidate_cosine >= 0.950,
        "g4_nonoperator_gain": gain >= 0.15,
        "g5_shift_wins": shift_wins >= 4,
        "g6_endpoint_basis": endpoint_cosine >= 0.900,
        "g7_evoo_spearman": abs(candidate_rho) >= 0.900,
        "g8_top_two_basin": basin_cells >= 8,
        "g9_complete_over_incomplete": complete_wins >= 8,
    }
    checks["gates_match"] = gates == report["gates"]
    checks["decision_match"] = report["decision"] == "NOT_SUPPORTIVE"
    checks["headline_nmse"] = close(
        candidate_nmse, report["summary"]["candidate_prediction_nmse"]
    )
    checks["headline_gain"] = close(
        gain, report["summary"]["relative_gain_over_best_nonoperator"]
    )
    checks["headline_rho"] = close(
        candidate_rho, report["summary"]["candidate_prediction_evoo_spearman"]
    )
    checks["paired_counts"] = (
        shift_wins == report["summary"]["shift_wins"] == 5
        and complete_wins == report["summary"]["complete_over_incomplete_wins"] == 10
        and basin_cells == report["summary"]["top_two_pass_cells"] == 0
    )
    checks["failure_retained"] = (
        sum(gates.values()) == 6
        and not gates["g4_nonoperator_gain"]
        and not gates["g7_evoo_spearman"]
        and not gates["g8_top_two_basin"]
    )

    verification = {
        "checks": checks,
        "passed": int(sum(checks.values())),
        "total": len(checks),
        "all_passed": bool(all(checks.values())),
        "independent_recalculation": {
            "candidate_prediction_nmse": candidate_nmse,
            "candidate_prediction_cosine": candidate_cosine,
            "candidate_prediction_evoo_spearman": candidate_rho,
            "endpoint_basis_cosine": endpoint_cosine,
            "best_nonoperator": best_nonoperator,
            "relative_gain": gain,
            "shift_wins": shift_wins,
            "complete_over_incomplete_wins": complete_wins,
            "top_two_pass_cells": basin_cells,
            "complete_rank": complete_rank,
            "incomplete_rank": incomplete_rank,
            "gates": gates,
        },
    }
    (RESULTS / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2))
    return 0 if verification["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
