"""Locked per-spectrum AsLS Raman preprocessing sensitivity (R1.9)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import run_poisson_multistart_audit as audit
from run_raman_loss_iteration_sensitivity import euclidean_evaluation

RA = audit.ra
HERE = Path(__file__).resolve().parent
OUT = HERE / "results_raman_baseline_correction_sensitivity"
ORIGINAL = HERE.parent / "phase_b" / "results_real_raman_oil" / "records.json"
SMOOTHNESS = 100_000.0
ASYMMETRY = 0.01
WEIGHT_STEPS = 10


def baseline_correct(x: np.ndarray) -> np.ndarray:
    """Asymmetric LS baseline on each row, followed by nonnegative clipping."""
    p = x.shape[1]
    second_difference = diags(
        (np.ones(p - 2), -2.0 * np.ones(p - 2), np.ones(p - 2)),
        (0, 1, 2), shape=(p - 2, p), format="csc"
    )
    penalty = (SMOOTHNESS * (second_difference.T @ second_difference)).tocsc()
    corrected = np.empty_like(x, dtype=float)
    for row_index, row in enumerate(x):
        weights = np.ones(p, dtype=float)
        for _ in range(WEIGHT_STEPS):
            baseline = spsolve(penalty + diags(weights, format="csc"), weights * row)
            weights = np.where(row > baseline, ASYMMETRY, 1.0 - ASYMMETRY)
        corrected[row_index] = np.maximum(row - baseline, 0.0)
    return corrected


def summarize(values):
    x = np.asarray(list(values), dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, subsets, duplicate_count = RA.load_data()
    (calibration, calibration_raw) = subsets["CALIBRATION"]
    (validation, validation_raw) = subsets["VALIDATION"]
    (prediction, prediction_raw) = subsets["PREDICTION"]
    raw = np.vstack((calibration_raw, validation_raw, prediction_raw))
    corrected = baseline_correct(raw)
    slices = np.cumsum((len(calibration_raw), len(validation_raw)))
    calibration_bc, validation_bc, prediction_bc = np.split(corrected, slices)
    retained = corrected.sum(axis=1) / raw.sum(axis=1)
    if np.any(corrected.sum(axis=1) <= 0):
        raise ValueError("Baseline correction produced a zero-signal row")
    if not np.isfinite(corrected).all():
        raise ValueError("Nonfinite corrected spectrum")

    endpoints = []
    for percentage in (0.0, 100.0):
        mask = calibration["_evoo"].to_numpy(dtype=float) == percentage
        endpoints.append(RA.normalize_fine(calibration_bc[mask]).mean(axis=0))
    endpoints = np.vstack(endpoints)
    p = raw.shape[1]
    mats = [RA.partition(p, RA.WIDTH, origin) for origin in RA.COMPLETE_ORIGINS]
    heldout = RA.partition(p, RA.HELDOUT_WIDTH, RA.HELDOUT_ORIGIN)
    old = {
        (record["shift"], record["seed"]): record
        for record in json.loads(ORIGINAL.read_text(encoding="utf-8"))
        if record["method"] == "c530_complete_operator"
    }
    rows = []
    for shift in RA.SHIFTS:
        cohorts = RA.allocate(calibration, calibration_bc, shift)
        ys = [RA.observe_coarse(cohort, mat) for cohort, mat in zip(cohorts, mats)]
        for seed in RA.SEEDS_BY_SHIFT[shift]:
            bp_w, bp_h, _ = RA.backproject_fit(ys, mats, 2, seed + 10)
            _, bp_val, bp_pred = RA.evaluate_basis(
                bp_h, validation_bc, validation["_evoo"].to_numpy(dtype=float),
                prediction_bc, prediction["_evoo"].to_numpy(dtype=float),
                heldout, endpoints
            )
            (best, top_two, objectives) = RA.multistart_operator(
                ys, mats, 2, seed + 100, bp_fit=(bp_w, bp_h)
            )
            _, h, iterations, objective = best
            endpoint, val, pred = RA.evaluate_basis(
                h, validation_bc, validation["_evoo"].to_numpy(dtype=float),
                prediction_bc, prediction["_evoo"].to_numpy(dtype=float),
                heldout, endpoints
            )
            x_backprojected = np.vstack([
                RA.backproject(y, a) for y, a in zip(ys, mats)
            ])
            _, mcr_h, mcr_iterations = RA.mcr_als(
                x_backprojected, 2, seed + 200
            )
            mcr = euclidean_evaluation(
                mcr_h, validation_bc, validation["_evoo"].to_numpy(dtype=float),
                prediction_bc, prediction["_evoo"].to_numpy(dtype=float),
                endpoints, heldout
            )
            reference = old[(shift, seed)]
            rows.append({
                "shift": shift, "seed": seed,
                "iterations": iterations,
                "training_kl": objective,
                "top_two_basis_cosine": top_two,
                "endpoint_basis_cosine": endpoint,
                "validation_corrected_nmse": val["nmse"],
                "prediction_corrected_nmse": pred["nmse"],
                "prediction_corrected_cosine": pred["cosine"],
                "prediction_corrected_evoo_spearman": pred["evoo_spearman"],
                "backprojection_validation_corrected_nmse": bp_val["nmse"],
                "backprojection_prediction_corrected_nmse": bp_pred["nmse"],
                "mcr_als_iterations": mcr_iterations,
                "mcr_als_validation_corrected_nmse": mcr["validation"]["nmse"],
                "mcr_als_prediction_corrected_nmse": mcr["prediction"]["nmse"],
                "prediction_original_nmse": reference["prediction"]["nmse"],
                "prediction_original_cosine": reference["prediction"]["cosine"],
                "start_objectives": json.dumps(objectives),
            })
            print(f"shift={shift} seed={seed} operator={pred['nmse']:.8g} "
                  f"backprojection={bp_pred['nmse']:.8g} "
                  f"MCR={mcr['prediction']['nmse']:.8g}", flush=True)
    with (OUT / "cell_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": {
            "asls_lambda": SMOOTHNESS, "asls_p": ASYMMETRY,
            "asls_iterations": WEIGHT_STEPS, "rank": 2, "starts": 6,
            "fit_iterations": 420, "shifts": list(RA.SHIFTS),
            "seeds_per_shift": 2, "duplicates_removed": duplicate_count,
            "spectra_by_split": {"calibration": len(calibration_bc),
                                 "validation": len(validation_bc),
                                 "prediction": len(prediction_bc)},
        },
        "retained_positive_signal_fraction": summarize(retained),
        "summary": {key: summarize(row[key] for row in rows)
                    for key in ("validation_corrected_nmse",
                                "prediction_corrected_nmse",
                                "prediction_corrected_cosine",
                                "backprojection_prediction_corrected_nmse",
                                "mcr_als_prediction_corrected_nmse",
                                "prediction_original_nmse",
                                "prediction_original_cosine",
                                "top_two_basis_cosine")},
        "corrected_nmse_lower_than_original_cells": sum(
            row["prediction_corrected_nmse"] < row["prediction_original_nmse"]
            for row in rows
        ),
        "operator_better_than_corrected_backprojection_cells": sum(
            row["prediction_corrected_nmse"]
            < row["backprojection_prediction_corrected_nmse"] for row in rows
        ),
        "operator_better_than_corrected_mcr_cells": sum(
            row["prediction_corrected_nmse"]
            < row["mcr_als_prediction_corrected_nmse"] for row in rows
        ),
        "checks": {
            "expected_cells": len(rows) == 10,
            "no_zero_corrected_rows": bool(np.all(corrected.sum(axis=1) > 0)),
            "finite": bool(np.isfinite(corrected).all() and all(
                np.isfinite(row[key]) for row in rows
                for key in ("training_kl", "prediction_corrected_nmse",
                            "prediction_corrected_cosine", "prediction_original_nmse")
            )),
            "all_checkpoint_420": all(row["iterations"] == 420 for row in rows),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
