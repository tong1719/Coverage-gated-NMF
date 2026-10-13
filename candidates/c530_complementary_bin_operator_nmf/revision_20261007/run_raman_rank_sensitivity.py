"""Predeclared Raman K=1--4 sensitivity at three retrospective allocations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_poisson_multistart_audit as audit

RA = audit.ra
OUT = Path(__file__).resolve().parent / "results_raman_rank_sensitivity"
RANKS = (1, 2, 3, 4)
SHIFTS = (0, 2, 4)
ITERATIONS = 420
EPS = 1e-12


def kl_per_count(y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.maximum(pred, EPS)
    positive = y > 0
    loss = float(np.sum(pred - y))
    loss += float(np.sum(y[positive] * np.log(y[positive] / pred[positive])))
    return loss / max(float(y.sum()), EPS)


def coarse_cv(h: np.ndarray, y: np.ndarray, operator: np.ndarray) -> float:
    basis = h @ operator
    fold_scores = []
    for train_start in (0, 1):
        train_cols = np.arange(train_start, y.shape[1], 2)
        test_cols = np.arange(1 - train_start, y.shape[1], 2)
        w = RA.fixed_w(y[:, train_cols], basis[:, train_cols], max_iter=400)
        fold_scores.append(kl_per_count(y[:, test_cols], w @ basis[:, test_cols]))
    return float(np.mean(fold_scores))


def fine_nmse(h: np.ndarray, raw: np.ndarray, operator: np.ndarray) -> float:
    y = RA.observe_coarse(raw, operator)
    w = RA.fixed_w(y, h @ operator, max_iter=400)
    truth = RA.normalize_fine(raw)
    recon = w @ h
    per_row = np.sum((recon - truth) ** 2, axis=1) / np.maximum(np.sum(truth ** 2, axis=1), EPS)
    return float(np.mean(per_row))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _, subsets, duplicates = RA.load_data()
    calibration, calibration_raw = subsets["CALIBRATION"]
    _, validation_raw = subsets["VALIDATION"]
    _, prediction_raw = subsets["PREDICTION"]
    p = calibration_raw.shape[1]
    mats = [RA.partition(p, RA.WIDTH, o) for o in RA.COMPLETE_ORIGINS]
    mappings = [audit.observed_mapping(a) for a in mats]
    heldout = RA.partition(p, RA.HELDOUT_WIDTH, RA.HELDOUT_ORIGIN)
    y_validation = RA.observe_coarse(validation_raw, heldout)
    audit.MAX_ITER = ITERATIONS
    rows = []
    for shift in SHIFTS:
        seed = RA.SEEDS_BY_SHIFT[shift][0]
        cohorts = RA.allocate(calibration, calibration_raw, shift)
        ys = [RA.observe_coarse(raw, a) for raw, a in zip(cohorts, mats)]
        for rank in RANKS:
            bp_w, bp_h, _ = RA.backproject_fit(ys, mats, rank, seed + 10)
            starts = [(seed + 100, bp_h, bp_w)]
            starts.extend((seed + 101 + i, None, None) for i in range(5))
            fits = []
            for start_id, (start_seed, h0, w0) in enumerate(starts):
                checkpoint, _, _, max_increase = audit.fit(
                    ys, mappings, rank, start_seed, h0=h0, w0=w0
                )
                fits.append((checkpoint["objective"], start_id, checkpoint["h"], max_increase))
            fits.sort(key=lambda item: item[0])
            best = fits[0]
            h = best[2]
            row = {
                "shift": shift,
                "seed": seed,
                "rank": rank,
                "best_start": best[1],
                "training_kl": best[0],
                "second_gap": (fits[1][0] - best[0]) / max(abs(best[0]), EPS),
                "top_two_cosine": audit.basis_similarity(h, fits[1][2]),
                "coarse_cv_kl_per_count": coarse_cv(h, y_validation, heldout),
                "validation_fine_nmse": fine_nmse(h, validation_raw, heldout),
                "prediction_fine_nmse": fine_nmse(h, prediction_raw, heldout),
                "max_objective_increase": max(fit[3] for fit in fits),
            }
            rows.append(row)
            print(f"shift={shift} rank={rank} cv={row['coarse_cv_kl_per_count']:.7g} "
                  f"prediction_nmse={row['prediction_fine_nmse']:.7g}", flush=True)
    with (OUT / "cell_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected = {
        str(shift): min((r for r in rows if r["shift"] == shift),
                        key=lambda r: (r["coarse_cv_kl_per_count"], r["rank"]))["rank"]
        for shift in SHIFTS
    }
    report = {
        "protocol": {"ranks": RANKS, "shifts": SHIFTS, "starts_per_cell": 6,
                     "iterations": ITERATIONS, "duplicates_removed": duplicates,
                     "calibration_n": len(calibration_raw),
                     "validation_n": len(validation_raw),
                     "prediction_n": len(prediction_raw)},
        "selected_rank_by_shift": selected,
        "median_by_rank": {
            str(rank): {
                metric: float(np.median([r[metric] for r in rows if r["rank"] == rank]))
                for metric in ("coarse_cv_kl_per_count", "validation_fine_nmse",
                               "prediction_fine_nmse", "top_two_cosine")
            }
            for rank in RANKS
        },
        "checks": {
            "expected_rows": len(rows) == len(RANKS) * len(SHIFTS),
            "finite": all(np.isfinite(v) for row in rows for v in row.values()),
            "objective_monotone": all(r["max_objective_increase"] <= 1e-7 for r in rows),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
