"""Check Raman optimization beyond the 2000-iteration audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_poisson_multistart_audit as audit

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_raman_long_run_check"
SEEDS = (53900, 53939)


def evaluation(h):
    _, subsets, _ = audit.ra.load_data()
    calibration, calibration_raw = subsets["CALIBRATION"]
    validation, validation_raw = subsets["VALIDATION"]
    prediction, prediction_raw = subsets["PREDICTION"]
    endpoints = []
    for percentage in (0.0, 100.0):
        mask = calibration["_evoo"].to_numpy() == percentage
        endpoints.append(audit.ra.normalize_fine(calibration_raw[mask]).mean(axis=0))
    heldout = audit.ra.partition(h.shape[1], audit.ra.HELDOUT_WIDTH, audit.ra.HELDOUT_ORIGIN)
    endpoint, val, pred = audit.ra.evaluate_basis(
        h,
        validation_raw,
        validation["_evoo"].to_numpy(dtype=float),
        prediction_raw,
        prediction["_evoo"].to_numpy(dtype=float),
        heldout,
        np.vstack(endpoints),
    )
    return {"endpoint_cosine": endpoint, "validation": val, "prediction": pred}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((HERE / "results_poisson_multistart_audit" / "raman_retrospective_shift0_53800.json").read_text(encoding="utf-8"))
    case = audit.raman_case()
    audit.MAX_ITER = 10000
    rows = []
    bases = []
    for index in (0, 39):
        seed, h0, w0 = case["starts"][index]
        cp420, final, trace, max_increase = audit.fit(
            case["ys"], case["mappings"], case["rank"], seed, h0=h0, w0=w0
        )
        comparison = old["records"][index]["deviance_extended"]
        dev2000 = dict(trace)[2000]
        row = {
            "start": index,
            "seed": seed,
            "objective_420": cp420["objective"],
            "objective_2000": dev2000,
            "objective_2000_prior": comparison,
            "objective_2000_reproduction_error": abs(dev2000 - comparison),
            "objective_final": final["objective"],
            "iteration_final": final["iteration"],
            "converged": final["converged"],
            "maximum_relative_objective_increase": max_increase,
            "evaluation_420": evaluation(cp420["h"]),
            "evaluation_final": evaluation(final["h"]),
            "trace": trace,
        }
        rows.append(row)
        bases.append(final["h"])
        print(f"seed={seed} iteration={final['iteration']} final_deviance={final['objective']:.6f}", flush=True)
    gap = abs(rows[1]["objective_final"] - rows[0]["objective_final"]) / max(min(row["objective_final"] for row in rows), 1.0)
    report = {
        "case": case["name"],
        "rows": rows,
        "final_relative_objective_gap": gap,
        "final_basis_cosine": audit.basis_similarity(bases[0], bases[1]),
        "checks": {
            "reproduced_2000": all(row["objective_2000_reproduction_error"] <= 1e-6 for row in rows),
            "finite": all(np.isfinite(row["objective_final"]) for row in rows),
            "monotone": all(row["maximum_relative_objective_increase"] <= 1e-8 for row in rows),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "final_bases.npz", bases=np.stack(bases))
    print(json.dumps({"gap": gap, "basis_cosine": report["final_basis_cosine"], "checks": report["checks"]}, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
