"""Replay all submitted count-data cells at 420 and extended stopping."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

import run_operator_coupled_hals as hals
import run_poisson_multistart_audit as audit

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_full_poisson_stopping_audit"
AF = audit.af
MB = audit.mb
EPS = 1e-12
N_STARTS = 6


def controlled_cases():
    original = hals.baseline_index("controlled")
    for regime, seed in itertools.product(AF.REGIMES, AF.SEEDS):
        fine, _, true_h, held_y, held_a, held_rate = AF.simulate_fine(regime, seed)
        mats = [AF.c528.partition(AF.WIDTH, origin) for origin in range(AF.WIDTH)]
        mappings = [audit.observed_mapping(a) for a in mats]
        ys = [x @ a for x, a in zip(fine, mats)]
        bp = AF.c528.backproject_fit(ys, mats, seed + 1)
        bases = AF.c529.aligned_backprojected_bases(ys, mats, seed + 10)
        h_sep = np.mean(bases, axis=0)
        w_sep = AF.c529.fit_w_all(ys, mats, h_sep)
        starts = [(seed + 20, bp[1], bp[0]), (seed + 21, h_sep, w_sep)]
        starts.extend((seed + 22 + start, None, None) for start in range(4))
        yield {
            "name": f"controlled_{regime}_{seed}", "dataset": "controlled",
            "key": (regime, seed), "ys": ys, "mappings": mappings,
            "rank": AF.K, "starts": starts, "true_h": true_h,
            "held_y": held_y, "held_a": held_a,
            "held_rate": held_rate, "original": original[(regime, seed)],
        }


def massbank_cases():
    original = hals.baseline_index("massbank")
    panels, _ = MB.panels_and_provenance()
    for panel_index, panel in enumerate(panels):
        for depth_name, seed in itertools.product(MB.DEPTHS, MB.SEEDS):
            rng = np.random.default_rng(seed + 100 * panel_index)
            true_h = np.vstack([row["vector"] for row in panel])
            true_h /= np.maximum(true_h.mean(axis=1, keepdims=True), EPS)
            mappings = [MB.bin_map(4, origin) for origin in range(4)]
            ys = []
            for mapping in mappings:
                w = rng.gamma(1.7, 0.7, (40, MB.K))
                w /= np.maximum(w.mean(axis=0, keepdims=True), EPS)
                w *= MB.DEPTHS[depth_name]
                rate = 0.002 + w @ true_h
                ys.append(MB.aggregate(rng.poisson(rate).astype(float), mapping))
            wt = rng.gamma(1.7, 0.7, (48, MB.K))
            wt /= np.maximum(wt.mean(axis=0, keepdims=True), EPS)
            wt *= MB.DEPTHS[depth_name]
            held_rate = 0.002 + wt @ true_h
            held_map = MB.bin_map(8, 3)
            held_y = MB.aggregate(rng.poisson(held_rate).astype(float), held_map)
            bp = MB.backproject_fit(ys, mappings, seed + 1)
            sep = MB.separate_fit(ys, mappings, seed + 10)
            starts = [(seed + 40, bp[1], bp[0]), (seed + 41, sep[1], sep[0])]
            starts.extend((seed + 42 + start, None, None) for start in range(4))
            key = (panel_index, depth_name, seed)
            yield {
                "name": f"massbank_panel{panel_index}_{depth_name}_{seed}",
                "dataset": "massbank", "key": key, "ys": ys,
                "mappings": mappings, "rank": MB.K, "starts": starts,
                "true_h": true_h, "held_y": held_y,
                "held_map": held_map, "held_rate": held_rate,
                "original": original[key],
            }


def metrics(case, fit):
    h = fit["h"]
    if case["dataset"] == "controlled":
        heldout = AF.c528.heldout(h, case["held_y"], case["held_a"], case["held_rate"])
    else:
        heldout = MB.heldout(h, case["held_y"], case["held_map"], case["held_rate"])
    return {
        "matched_basis_cosine": audit.basis_similarity(h, case["true_h"]),
        "heldout_nmse": heldout,
    }


def run_cell(case):
    started = time.perf_counter()
    starts = []
    fits420 = []
    fitsext = []
    for index, (seed, h0, w0) in enumerate(case["starts"]):
        cp, ext, trace, max_increase = audit.fit(
            case["ys"], case["mappings"], case["rank"], seed, h0=h0, w0=w0
        )
        fits420.append(cp)
        fitsext.append(ext)
        starts.append({
            "start": index, "seed": seed,
            "objective_420": cp["objective"], "objective_extended": ext["objective"],
            "iteration_extended": ext["iteration"], "converged": ext["converged"],
            "max_relative_objective_increase": max_increase,
            "objective_trace": trace,
        })
    index420 = int(np.argmin([fit["objective"] for fit in fits420]))
    indexext = int(np.argmin([fit["objective"] for fit in fitsext]))
    m420 = metrics(case, fits420[index420])
    mext = metrics(case, fitsext[indexext])
    original = case["original"]
    result = {
        "name": case["name"], "dataset": case["dataset"], "key": case["key"],
        "best_start_420": index420, "best_start_extended": indexext,
        "objective_420": fits420[index420]["objective"],
        "objective_extended": fitsext[indexext]["objective"],
        "metrics_420": m420, "metrics_extended": mext,
        "n_converged_starts": int(sum(row["converged"] for row in starts)),
        "best_extended_converged": bool(fitsext[indexext]["converged"]),
        "best_extended_iterations": int(fitsext[indexext]["iteration"]),
        "max_relative_objective_increase": float(max(row["max_relative_objective_increase"] for row in starts)),
        "submitted_metric_reproduction_error": {
            "basis_cosine": abs(m420["matched_basis_cosine"] - original["basis_cosine"]),
            "heldout_nmse": abs(m420["heldout_nmse"] - original["heldout_nmse"]),
        },
        "runtime_seconds": time.perf_counter() - started,
        "starts": starts,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{case['name']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{case['name']} best_iter={result['best_extended_iterations']} converged={result['best_extended_converged']} elapsed={result['runtime_seconds']:.1f}s", flush=True)
    return result


def qsummary(values):
    x = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))}


def main():
    audit.MAX_ITER = 5000
    audit.TOL = 1e-6
    started = time.perf_counter()
    records = [run_cell(case) for case in itertools.chain(controlled_cases(), massbank_cases())]
    hals_rows = json.loads((HERE / "results_operator_coupled_hals" / "records.json").read_text(encoding="utf-8"))
    hals_index = {
        (row["regime"], row["seed"]) if row["dataset"] == "controlled" else (row["panel"], row["depth"], row["seed"]): row
        for row in hals_rows
    }
    sections = {}
    for dataset in ("controlled", "massbank"):
        rows = [row for row in records if row["dataset"] == dataset]
        sections[dataset] = {
            "n_cells": len(rows),
            "basis_420": qsummary(row["metrics_420"]["matched_basis_cosine"] for row in rows),
            "basis_extended": qsummary(row["metrics_extended"]["matched_basis_cosine"] for row in rows),
            "heldout_420": qsummary(row["metrics_420"]["heldout_nmse"] for row in rows),
            "heldout_extended": qsummary(row["metrics_extended"]["heldout_nmse"] for row in rows),
            "relative_objective_improvement": qsummary(
                (row["objective_420"] - row["objective_extended"]) / max(row["objective_420"], 1.0)
                for row in rows
            ),
            "n_best_converged": int(sum(row["best_extended_converged"] for row in rows)),
            "n_all_six_converged": int(sum(row["n_converged_starts"] == N_STARTS for row in rows)),
            "extended_better_heldout_cells": int(sum(row["metrics_extended"]["heldout_nmse"] < row["metrics_420"]["heldout_nmse"] for row in rows)),
            "extended_better_basis_cells": int(sum(row["metrics_extended"]["matched_basis_cosine"] > row["metrics_420"]["matched_basis_cosine"] for row in rows)),
            "extended_poisson_wins_vs_hals": int(sum(row["metrics_extended"]["heldout_nmse"] < hals_index[tuple(row["key"])]["heldout_nmse"] for row in rows)),
            "max_submitted_metric_reproduction_error": {
                "basis_cosine": max(row["submitted_metric_reproduction_error"]["basis_cosine"] for row in rows),
                "heldout_nmse": max(row["submitted_metric_reproduction_error"]["heldout_nmse"] for row in rows),
            },
        }
    checks = {
        "record_count": len(records) == 32,
        "finite": all(np.isfinite(row["objective_extended"]) and np.isfinite(row["metrics_extended"]["heldout_nmse"]) for row in records),
        "objective_monotonicity": all(row["max_relative_objective_increase"] <= 1e-8 for row in records),
        "submitted_reproduction": all(max(row["submitted_metric_reproduction_error"].values()) <= 1e-9 for row in records),
    }
    report = {
        "protocol": {"starts": N_STARTS, "original_iterations": 420, "extended_cap": audit.MAX_ITER, "relative_tolerance": audit.TOL},
        "datasets": sections, "checks": checks,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
