"""Re-evaluate saved-protocol Raman MCR bases with NNLS held-out coefficients."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

import run_poisson_multistart_audit as audit

RA = audit.ra
HERE = Path(__file__).resolve().parent
OUT = HERE / "results_raman_mcr_heldout_loss_match"
ORIGINAL = HERE.parent / "phase_b" / "results_real_raman_oil" / "records.json"
EPS = 1e-12


def evaluate(h: np.ndarray, raw: np.ndarray, operator: np.ndarray) -> dict:
    y = RA.observe_coarse(raw, operator)
    fine = RA.normalize_fine(raw)
    coarse_basis = h @ operator
    w_kl = RA.fixed_w(y, coarse_basis, max_iter=400)
    w_nnls = np.vstack([nnls(coarse_basis.T, row)[0] for row in y])
    output = {}
    for name, w in (("kl", w_kl), ("nnls", w_nnls)):
        recon = w @ h
        per_row = np.sum((recon - fine) ** 2, axis=1) / np.maximum(
            np.sum(fine ** 2, axis=1), EPS
        )
        output[name] = float(np.mean(per_row))
    return output


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, subsets, duplicates = RA.load_data()
    calibration, calibration_raw = subsets["CALIBRATION"]
    _, validation_raw = subsets["VALIDATION"]
    _, prediction_raw = subsets["PREDICTION"]
    p = calibration_raw.shape[1]
    mats = [RA.partition(p, RA.WIDTH, origin) for origin in RA.COMPLETE_ORIGINS]
    heldout = RA.partition(p, RA.HELDOUT_WIDTH, RA.HELDOUT_ORIGIN)
    original_records = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    prior = {(record["shift"], record["seed"], record["method"]): record
             for record in original_records}
    rows = []
    for shift in RA.SHIFTS:
        cohorts = RA.allocate(calibration, calibration_raw, shift)
        ys = [RA.observe_coarse(raw, a) for raw, a in zip(cohorts, mats)]
        x = np.vstack([RA.backproject(y, a) for y, a in zip(ys, mats)])
        for seed in RA.SEEDS_BY_SHIFT[shift]:
            _, h, iterations = RA.mcr_als(x, 2, seed + 200)
            val = evaluate(h, validation_raw, heldout)
            pred = evaluate(h, prediction_raw, heldout)
            saved = prior[(shift, seed, "uniform_backproject_mcr_als")]
            operator = prior[(shift, seed, "c530_complete_operator")]
            row = {
                "shift": shift,
                "seed": seed,
                "als_iterations": iterations,
                "validation_nmse_original_kl_coeff": val["kl"],
                "validation_nmse_loss_matched_nnls": val["nnls"],
                "prediction_nmse_original_kl_coeff": pred["kl"],
                "prediction_nmse_loss_matched_nnls": pred["nnls"],
                "saved_prediction_nmse": saved["prediction"]["nmse"],
                "operator_poisson_prediction_nmse": operator["prediction"]["nmse"],
                "replay_error": max(
                    abs(val["kl"] - saved["validation"]["nmse"]),
                    abs(pred["kl"] - saved["prediction"]["nmse"]),
                ),
            }
            rows.append(row)
            print(f"shift={shift} seed={seed} KL={pred['kl']:.9g} "
                  f"NNLS={pred['nnls']:.9g} replay={row['replay_error']:.3g}",
                  flush=True)
    with (OUT / "cell_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ("prediction_nmse_original_kl_coeff", "prediction_nmse_loss_matched_nnls",
               "validation_nmse_original_kl_coeff", "validation_nmse_loss_matched_nnls",
               "operator_poisson_prediction_nmse")
    summary = {
        metric: {"mean": float(np.mean([r[metric] for r in rows])),
                 "median": float(np.median([r[metric] for r in rows]))}
        for metric in metrics
    }
    mcr_mean = summary["prediction_nmse_loss_matched_nnls"]["mean"]
    operator_mean = summary["operator_poisson_prediction_nmse"]["mean"]
    report = {
        "protocol": {"cells": len(rows), "shifts": list(RA.SHIFTS),
                     "seeds_per_shift": 2, "rank": 2,
                     "duplicates_removed": duplicates},
        "summary": summary,
        "mean_relative_operator_advantage_vs_nnls_mcr":
            float((mcr_mean - operator_mean) / mcr_mean),
        "nnls_better_than_kl_coefficient_cells": int(sum(
            r["prediction_nmse_loss_matched_nnls"]
            < r["prediction_nmse_original_kl_coeff"] for r in rows)),
        "checks": {
            "all_cells": len(rows) == 10,
            "finite": all(np.isfinite(v) for r in rows for v in r.values()),
            "original_replay": max(r["replay_error"] for r in rows) < 1e-8,
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
