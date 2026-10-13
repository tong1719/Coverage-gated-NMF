"""Representative mean/sum/L2 row-gauge audit for R1.m7."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_poisson_multistart_audit as base

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_normalization_gauge_audit"
EPS = 1e-12
GAUGES = ("mean", "sum", "l2")


def fit420(case: dict, start: tuple, gauge: str) -> dict:
    seed, h0, w0 = start
    rng = np.random.default_rng(seed)
    ys, mappings, rank = case["ys"], case["mappings"], case["rank"]
    p = len(mappings[0])
    h = np.maximum(np.asarray(h0, dtype=float).copy(), EPS) if h0 is not None else np.maximum(rng.gamma(1, 1, (rank, p)), EPS)
    ws = [np.maximum(np.asarray(w, dtype=float).copy(), EPS) for w in w0] if w0 is not None else [rng.gamma(1, 1, (len(y), rank)) for y in ys]
    for iteration in range(1, 421):
        for c, (y, mapping) in enumerate(zip(ys, mappings)):
            basis = base.aggregate_basis(h, mapping, y.shape[1])
            pred = np.maximum(ws[c] @ basis, EPS)
            ws[c] *= ((y / pred) @ basis.T) / np.maximum(basis.sum(axis=1)[None, :], EPS)
            ws[c] = np.maximum(ws[c], EPS)
        numerator_h = np.zeros_like(h)
        denominator_h = np.zeros((rank, 1))
        for y, mapping, w in zip(ys, mappings, ws):
            basis = base.aggregate_basis(h, mapping, y.shape[1])
            pred = np.maximum(w @ basis, EPS)
            numerator_h += (w.T @ (y / pred))[:, mapping]
            denominator_h += w.sum(axis=0)[:, None]
        h *= numerator_h / np.maximum(denominator_h, EPS)
        h = np.maximum(h, EPS)
        if iteration % 25 == 0:
            if gauge == "mean":
                scale = h.mean(axis=1)
            elif gauge == "sum":
                scale = h.sum(axis=1)
            elif gauge == "l2":
                scale = np.linalg.norm(h, axis=1)
            else:
                raise ValueError(gauge)
            scale = np.maximum(scale, EPS)
            h /= scale[:, None]
            ws = [w * scale[None, :] for w in ws]
    objective = base.kl_divergence(ys, mappings, ws, h)
    metric = base.fit_metrics(case, {"h": h})
    return {"objective": objective, "heldout_nmse": metric["heldout_nmse"], "h": h, "ws": ws}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [base.controlled_case("nominal", 53400), base.massbank_case()]
    rows = []
    checks = {}
    selected_fit_comparisons = {}
    for case in cases:
        results = {gauge: [fit420(case, start, gauge) for start in case["starts"][:6]] for gauge in GAUGES}
        selected = {gauge: int(np.argmin([fit["objective"] for fit in fits])) for gauge, fits in results.items()}
        checks[case["name"] + "_selected_same"] = len(set(selected.values())) == 1
        chosen = selected["mean"]
        selected_fit_comparisons[case["name"]] = {
            gauge: {
                "selected_start": selected[gauge],
                "objective": results[gauge][selected[gauge]]["objective"],
                "heldout_nmse": results[gauge][selected[gauge]]["heldout_nmse"],
                "cosine_to_mean_selected": base.basis_similarity(results["mean"][chosen]["h"], results[gauge][selected[gauge]]["h"]),
            }
            for gauge in GAUGES
        }
        for start_index in range(6):
            reference = results["mean"][start_index]
            for gauge in GAUGES[1:]:
                trial = results[gauge][start_index]
                row = {
                    "case": case["name"], "start": start_index, "gauge": gauge,
                    "mean_objective": reference["objective"], "trial_objective": trial["objective"],
                    "absolute_objective_difference": abs(reference["objective"] - trial["objective"]),
                    "relative_objective_difference": abs(reference["objective"] - trial["objective"]) / max(abs(reference["objective"]), EPS),
                    "mean_heldout_nmse": reference["heldout_nmse"], "trial_heldout_nmse": trial["heldout_nmse"],
                    "absolute_heldout_nmse_difference": abs(reference["heldout_nmse"] - trial["heldout_nmse"]),
                    "aligned_basis_cosine": base.basis_similarity(reference["h"], trial["h"]),
                }
                rows.append(row)
        print(case["name"], selected, flush=True)
    with (OUT / "paired_starts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": {"cases": [case["name"] for case in cases], "starts_per_case": 6, "iterations": 420, "gauges": GAUGES},
        "max_relative_objective_difference": max(row["relative_objective_difference"] for row in rows),
        "max_absolute_heldout_nmse_difference": max(row["absolute_heldout_nmse_difference"] for row in rows),
        "min_aligned_basis_cosine": min(row["aligned_basis_cosine"] for row in rows),
        "selected_start_checks": checks,
        "selected_fit_comparisons": selected_fit_comparisons,
        "finite": all(np.isfinite(value) for row in rows for value in row.values() if isinstance(value, float)),
    }
    (OUT / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
