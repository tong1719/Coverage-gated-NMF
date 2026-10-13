"""Ten-cell retrospective Raman loss and iteration sensitivity audit."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

import run_operator_aware_frobenius as fro
import run_poisson_multistart_audit as audit

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_raman_loss_iteration_sensitivity"
RA = audit.ra
EPS = 1e-12


def euclidean_evaluation(h, raw_validation, validation_percent, raw_prediction, prediction_percent, endpoints, heldout):
    order, endpoint_cosine = RA.endpoint_alignment(h, endpoints)

    def evaluate(raw, percentages):
        fine = RA.normalize_fine(raw)
        y = RA.observe_coarse(raw, heldout)
        basis = h @ heldout
        w = np.vstack([nnls(basis.T, row)[0] for row in y])
        reconstruction = w @ h
        row_nmse = np.sum((reconstruction - fine) ** 2, axis=1) / np.maximum(np.sum(fine**2, axis=1), EPS)
        row_cosine = np.sum(reconstruction * fine, axis=1) / np.maximum(
            np.linalg.norm(reconstruction, axis=1) * np.linalg.norm(fine, axis=1), EPS
        )
        oriented = w[:, order]
        evoo_fraction = oriented[:, 1] / np.maximum(oriented[:, :2].sum(axis=1), EPS)
        return {
            "nmse": float(np.mean(row_nmse)),
            "cosine": float(np.mean(row_cosine)),
            "evoo_spearman": float(spearmanr(evoo_fraction, percentages).statistic),
            "n": int(len(raw)),
        }

    return {
        "endpoint_cosine": endpoint_cosine,
        "validation": evaluate(raw_validation, validation_percent),
        "prediction": evaluate(raw_prediction, prediction_percent),
    }


def summarize(values):
    x = np.asarray(list(values), dtype=float)
    return {
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def run_cell(shift, seed, calibration, calibration_raw, validation, validation_raw, prediction, prediction_raw, endpoints):
    started = time.perf_counter()
    cohorts = RA.allocate(calibration, calibration_raw, shift)
    mats = [RA.partition(calibration_raw.shape[1], RA.WIDTH, origin) for origin in RA.COMPLETE_ORIGINS]
    mappings = [audit.observed_mapping(mat) for mat in mats]
    ys = [RA.observe_coarse(raw, a) for raw, a in zip(cohorts, mats)]
    heldout = RA.partition(calibration_raw.shape[1], RA.HELDOUT_WIDTH, RA.HELDOUT_ORIGIN)
    bp_w, bp_h, _ = RA.backproject_fit(ys, mats, 2, seed + 10)
    specs = [(seed + 100, bp_h, bp_w)] + [(seed + 101 + index, None, None) for index in range(5)]
    results = {"poisson_420": [], "poisson_2000": [], "frobenius_420": [], "frobenius_2000": []}
    audit.MAX_ITER = 2000
    fro.REL_TOL = 0.0
    for start, (start_seed, h0, w0) in enumerate(specs):
        cp, ext, _, max_inc = audit.fit(ys, mappings, 2, start_seed, h0=h0, w0=w0)
        results["poisson_420"].append({"h": cp["h"], "objective": cp["objective"], "iteration": cp["iteration"], "converged": False, "max_increase": max_inc, "start": start})
        results["poisson_2000"].append({"h": ext["h"], "objective": ext["objective"], "iteration": ext["iteration"], "converged": ext["converged"], "max_increase": max_inc, "start": start})
        for cap in (420, 2000):
            fro.MAX_ITER = cap
            _, h, trace, iteration, converged, max_inc_fro = fro.fit_frobenius(
                ys, mappings, 2, start_seed, h0=h0, w0=w0
            )
            results[f"frobenius_{cap}"].append({"h": h, "objective": trace[-1], "iteration": iteration, "converged": converged, "max_increase": max_inc_fro, "start": start})
    output = {"shift": shift, "seed": seed, "methods": {}}
    for method, fits in results.items():
        ranked = sorted(fits, key=lambda fit: fit["objective"])
        best, second = ranked[:2]
        if method.startswith("poisson"):
            endpoint, validation_metrics, prediction_metrics = RA.evaluate_basis(
                best["h"],
                validation_raw, validation["_evoo"].to_numpy(dtype=float),
                prediction_raw, prediction["_evoo"].to_numpy(dtype=float),
                heldout, endpoints,
            )
            evaluation = {
                "endpoint_cosine": endpoint,
                "validation": validation_metrics,
                "prediction": prediction_metrics,
            }
        else:
            evaluation = euclidean_evaluation(
                best["h"], validation_raw, validation["_evoo"].to_numpy(dtype=float),
                prediction_raw, prediction["_evoo"].to_numpy(dtype=float), endpoints, heldout
            )
        output["methods"][method] = {
            "best_start": best["start"],
            "best_objective": best["objective"],
            "second_relative_objective_gap": (second["objective"] - best["objective"]) / max(abs(best["objective"]), 1.0),
            "top_two_cosine": audit.basis_similarity(best["h"], second["h"]),
            "best_iterations": best["iteration"],
            "converged_starts": int(sum(fit["converged"] for fit in fits)),
            "max_objective_increase": float(max(fit["max_increase"] for fit in fits)),
            "evaluation": evaluation,
            "all_start_objectives": [fit["objective"] for fit in fits],
        }
    output["runtime_seconds"] = time.perf_counter() - started
    (OUT / f"shift{shift}_seed{seed}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"shift={shift} seed={seed} elapsed={output['runtime_seconds']:.1f}s", flush=True)
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, subsets, duplicate_count = RA.load_data()
    calibration, calibration_raw = subsets["CALIBRATION"]
    validation, validation_raw = subsets["VALIDATION"]
    prediction, prediction_raw = subsets["PREDICTION"]
    endpoints = []
    for percentage in (0.0, 100.0):
        mask = calibration["_evoo"].to_numpy() == percentage
        endpoints.append(RA.normalize_fine(calibration_raw[mask]).mean(axis=0))
    endpoints = np.vstack(endpoints)
    records = [
        run_cell(shift, seed, calibration, calibration_raw, validation, validation_raw, prediction, prediction_raw, endpoints)
        for shift in RA.SHIFTS
        for seed in RA.SEEDS_BY_SHIFT[shift]
    ]
    methods = ("poisson_420", "poisson_2000", "frobenius_420", "frobenius_2000")
    summary = {}
    for method in methods:
        rows = [record["methods"][method] for record in records]
        summary[method] = {
            "prediction_nmse": summarize(row["evaluation"]["prediction"]["nmse"] for row in rows),
            "validation_nmse": summarize(row["evaluation"]["validation"]["nmse"] for row in rows),
            "prediction_cosine": summarize(row["evaluation"]["prediction"]["cosine"] for row in rows),
            "endpoint_cosine": summarize(row["evaluation"]["endpoint_cosine"] for row in rows),
            "top_two_cosine": summarize(row["top_two_cosine"] for row in rows),
            "converged_start_total": int(sum(row["converged_starts"] for row in rows)),
            "best_start_backprojection_count": int(sum(row["best_start"] == 0 for row in rows)),
        }
    report = {
        "protocol": {"cells": 10, "starts_per_cell": 6, "rank": 2, "checkpoints": [420, 2000], "duplicate_rows_removed": duplicate_count},
        "summary": summary,
        "paired_direction_counts": {
            "poisson_2000_better_prediction_than_420": int(sum(
                record["methods"]["poisson_2000"]["evaluation"]["prediction"]["nmse"]
                < record["methods"]["poisson_420"]["evaluation"]["prediction"]["nmse"]
                for record in records
            )),
            "frobenius_2000_better_prediction_than_420": int(sum(
                record["methods"]["frobenius_2000"]["evaluation"]["prediction"]["nmse"]
                < record["methods"]["frobenius_420"]["evaluation"]["prediction"]["nmse"]
                for record in records
            )),
            "frobenius_2000_better_prediction_than_poisson_2000": int(sum(
                record["methods"]["frobenius_2000"]["evaluation"]["prediction"]["nmse"]
                < record["methods"]["poisson_2000"]["evaluation"]["prediction"]["nmse"]
                for record in records
            )),
        },
        "checks": {
            "cell_count": len(records) == 10,
            "finite": all(np.isfinite(record["methods"][method]["evaluation"]["prediction"]["nmse"]) for record in records for method in methods),
            "objective_monotone": all(record["methods"][method]["max_objective_increase"] <= 1e-7 for record in records for method in methods),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
