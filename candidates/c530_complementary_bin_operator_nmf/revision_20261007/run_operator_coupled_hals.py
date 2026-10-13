"""Operator-coupled nonnegative coordinate descent for the C530 revision."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

import run_operator_aware_frobenius as eu

HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b"
OUT = HERE / "results_operator_coupled_hals"
EPS = 1e-12
MAX_SWEEPS = 1000
CHECK_EVERY = 10
REL_TOL = 1e-7
PATIENCE = 5
N_STARTS = 6


def build_residuals(ys, mappings, ws, h):
    return [
        y - w @ eu.aggregate_basis(h, mapping)
        for y, mapping, w in zip(ys, mappings, ws)
    ]


def fit_hals(ys, mappings, rank, width, seed, h0=None, w0=None):
    rng = np.random.default_rng(seed)
    p = len(mappings[0])
    groups = [np.arange(residue, p, width) for residue in range(width)]
    for mapping in mappings:
        for group in groups:
            if len(np.unique(mapping[group])) != len(group):
                raise ValueError("Coordinates within a group share a coarse bin")
    h = np.maximum(np.asarray(h0).copy(), EPS) if h0 is not None else np.maximum(rng.gamma(1, 1, (rank, p)), EPS)
    ws = [np.maximum(np.asarray(w).copy(), EPS) for w in w0] if w0 is not None else [rng.gamma(1, 1, (len(y), rank)) for y in ys]
    residuals = build_residuals(ys, mappings, ws, h)
    trace = [float(0.5 * sum(np.sum(r * r) for r in residuals))]
    stable = 0
    converged = False
    max_relative_increase = 0.0
    for sweep in range(1, MAX_SWEEPS + 1):
        for c, mapping in enumerate(mappings):
            basis = eu.aggregate_basis(h, mapping)
            for k in range(rank):
                bk = basis[k]
                denominator = max(float(bk @ bk), EPS)
                old = ws[c][:, k].copy()
                new = np.maximum(old + residuals[c] @ bk / denominator, 0.0)
                delta = new - old
                ws[c][:, k] = new
                residuals[c] -= np.outer(delta, bk)
        for k in range(rank):
            w_columns = [w[:, k].copy() for w in ws]
            denominator = max(float(sum(column @ column for column in w_columns)), EPS)
            for group in groups:
                numerator = np.zeros(len(group))
                bins_by_block = [mapping[group] for mapping in mappings]
                for residual, column, bins in zip(residuals, w_columns, bins_by_block):
                    numerator += column @ residual[:, bins]
                old = h[k, group].copy()
                new = np.maximum(old + numerator / denominator, 0.0)
                delta = new - old
                h[k, group] = new
                for residual, column, bins in zip(residuals, w_columns, bins_by_block):
                    residual[:, bins] -= np.outer(column, delta)
        if sweep % CHECK_EVERY == 0:
            scale = np.maximum(h.mean(axis=1), EPS)
            h /= scale[:, None]
            ws = [w * scale[None, :] for w in ws]
            residuals = build_residuals(ys, mappings, ws, h)
            current = float(0.5 * sum(np.sum(r * r) for r in residuals))
            relative = (trace[-1] - current) / max(abs(trace[-1]), EPS)
            max_relative_increase = max(max_relative_increase, -relative)
            trace.append(current)
            stable = stable + 1 if 0.0 <= relative < REL_TOL else 0
            if stable >= PATIENCE:
                converged = True
                break
    return ws, h, trace, sweep, converged, max_relative_increase


def qsummary(values):
    arr = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(arr)), "q25": float(np.quantile(arr, 0.25)), "q75": float(np.quantile(arr, 0.75))}


def solve_case(ys, mappings, width, rank, seed, bp, sep):
    starts = [(bp[1], bp[0]), (sep[1], sep[0])] + [(None, None)] * 4
    outputs = []
    started = time.perf_counter()
    for index, (h0, w0) in enumerate(starts):
        ws, h, trace, sweeps, converged, max_increase = fit_hals(
            ys, mappings, rank, width, seed + index, h0=h0, w0=w0
        )
        outputs.append({
            "ws": ws, "h": h, "objective": trace[-1], "trace": trace,
            "sweeps": sweeps, "converged": converged,
            "max_relative_increase": max_increase,
        })
    best = outputs[int(np.argmin([row["objective"] for row in outputs]))]
    return {
        "best": best,
        "runtime": time.perf_counter() - started,
        "converged_starts": int(sum(row["converged"] for row in outputs)),
        "six_start_agreement": float(np.mean([eu.align_similarity(row["h"], best["h"]) for row in outputs])),
        "minimum_start_agreement": float(np.min([eu.align_similarity(row["h"], best["h"]) for row in outputs])),
        "max_relative_increase": float(max(row["max_relative_increase"] for row in outputs)),
        "start_objectives": [row["objective"] for row in outputs],
        "start_sweeps": [row["sweeps"] for row in outputs],
        "start_traces": [row["trace"] for row in outputs],
    }


def baseline_index(dataset):
    path = PHASE_B / ("results_acquisition_factorial" if dataset == "controlled" else "results_massbank") / "records.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if dataset == "controlled":
        return {(row["regime"], row["seed"]): row for row in rows if row["arm"] == "complete"}
    return {(row["panel"], row["depth"], row["seed"]): row for row in rows if row["method"] == "c530_certified_operator"}


def euclidean_mu_index(dataset):
    path = HERE / "results_operator_aware_frobenius_extension_5000" / "records.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if dataset == "controlled":
        return {(row["regime"], row["seed"]): row for row in rows if row["dataset"] == "controlled"}
    return {(row["panel"], row["depth"], row["seed"]): row for row in rows if row["dataset"] == "massbank"}


def controlled_records():
    af = eu.af
    poisson = baseline_index("controlled")
    mu = euclidean_mu_index("controlled")
    records = []
    for regime, seed in itertools.product(af.REGIMES, af.SEEDS):
        fine, true_ws, true_h, held_y, held_a, held_rate = af.simulate_fine(regime, seed)
        mats = [af.c528.partition(af.WIDTH, origin) for origin in range(af.WIDTH)]
        mappings = [np.argmax(a, axis=1) for a in mats]
        ys = [fine_block @ mat for fine_block, mat in zip(fine, mats)]
        bp = af.c528.backproject_fit(ys, mats, seed + 1)
        bases = af.c529.aligned_backprojected_bases(ys, mats, seed + 10)
        h_sep = np.mean(bases, axis=0)
        sep = (af.c529.fit_w_all(ys, mats, h_sep), h_sep)
        output = solve_case(ys, mappings, af.WIDTH, af.K, seed + 7000, bp, sep)
        fit = output["best"]
        truth = np.vstack([0.015 + w @ true_h for w in true_ws])
        estimate = np.vstack([w @ fit["h"] for w in fit["ws"]])
        key = (regime, seed)
        records.append({
            "dataset": "controlled", "regime": regime, "seed": seed,
            "basis_cosine": af.c528.align_basis(fit["h"], true_h),
            "fine_nmse": float(np.sum((estimate - truth) ** 2) / max(np.sum(truth ** 2), EPS)),
            "heldout_nmse": eu.heldout_nnls(fit["h"], held_y, np.argmax(held_a, axis=1), held_rate),
            "objective": fit["objective"], "sweeps": fit["sweeps"],
            "converged_best": fit["converged"], "converged_starts": output["converged_starts"],
            "six_start_agreement": output["six_start_agreement"],
            "minimum_start_agreement": output["minimum_start_agreement"],
            "max_relative_increase": output["max_relative_increase"],
            "runtime_seconds": output["runtime"], "start_objectives": output["start_objectives"],
            "start_sweeps": output["start_sweeps"], "start_traces": output["start_traces"],
            "poisson_basis_cosine": poisson[key]["basis_cosine"],
            "poisson_heldout_nmse": poisson[key]["heldout_nmse"],
            "euclidean_mu_basis_cosine": mu[key]["basis_cosine"],
            "euclidean_mu_heldout_nmse": mu[key]["heldout_nmse"],
            "euclidean_mu_objective": mu[key]["training_objective"],
        })
        print(f"controlled {regime} {seed} sweeps={fit['sweeps']} elapsed={output['runtime']:.2f}s", flush=True)
    return records


def massbank_records():
    mb = eu.mb
    poisson = baseline_index("massbank")
    mu = euclidean_mu_index("massbank")
    panels, _ = mb.panels_and_provenance()
    records = []
    for panel_index, panel in enumerate(panels):
        for depth_name, seed in itertools.product(mb.DEPTHS, mb.SEEDS):
            rng = np.random.default_rng(seed + 100 * panel_index)
            true_h = np.vstack([row["vector"] for row in panel])
            true_h /= np.maximum(true_h.mean(axis=1, keepdims=True), EPS)
            mappings = [mb.bin_map(4, origin) for origin in range(4)]
            ys, true_ws = [], []
            for mapping in mappings:
                w = rng.gamma(1.7, 0.7, (40, mb.K))
                w /= np.maximum(w.mean(axis=0, keepdims=True), EPS)
                w *= mb.DEPTHS[depth_name]
                rate = 0.002 + w @ true_h
                ys.append(mb.aggregate(rng.poisson(rate).astype(float), mapping))
                true_ws.append(w)
            wt = rng.gamma(1.7, 0.7, (48, mb.K))
            wt /= np.maximum(wt.mean(axis=0, keepdims=True), EPS)
            wt *= mb.DEPTHS[depth_name]
            held_rate = 0.002 + wt @ true_h
            held_mapping = mb.bin_map(8, 3)
            held_y = mb.aggregate(rng.poisson(held_rate).astype(float), held_mapping)
            bp = mb.backproject_fit(ys, mappings, seed + 1)
            sep = mb.separate_fit(ys, mappings, seed + 10)
            output = solve_case(ys, mappings, 4, mb.K, seed + 8000, bp, sep)
            fit = output["best"]
            truth = np.vstack([0.002 + w @ true_h for w in true_ws])
            estimate = np.vstack([w @ fit["h"] for w in fit["ws"]])
            key = (panel_index, depth_name, seed)
            records.append({
                "dataset": "massbank", "panel": panel_index, "depth": depth_name, "seed": seed,
                "basis_cosine": mb.basis_cosine(fit["h"], true_h),
                "fine_nmse": float(np.sum((estimate - truth) ** 2) / max(np.sum(truth ** 2), EPS)),
                "heldout_nmse": eu.heldout_nnls(fit["h"], held_y, held_mapping, held_rate),
                "objective": fit["objective"], "sweeps": fit["sweeps"],
                "converged_best": fit["converged"], "converged_starts": output["converged_starts"],
                "six_start_agreement": output["six_start_agreement"],
                "minimum_start_agreement": output["minimum_start_agreement"],
                "max_relative_increase": output["max_relative_increase"],
                "runtime_seconds": output["runtime"], "start_objectives": output["start_objectives"],
                "start_sweeps": output["start_sweeps"], "start_traces": output["start_traces"],
                "poisson_basis_cosine": poisson[key]["basis_cosine"],
                "poisson_heldout_nmse": poisson[key]["heldout_nmse"],
                "euclidean_mu_basis_cosine": mu[key]["basis_cosine"],
                "euclidean_mu_heldout_nmse": mu[key]["heldout_nmse"],
                "euclidean_mu_objective": mu[key]["training_objective"],
            })
            print(f"massbank {panel_index} {depth_name} {seed} sweeps={fit['sweeps']} elapsed={output['runtime']:.2f}s", flush=True)
    return records


def report_summary(records):
    sections = {}
    for dataset in ("controlled", "massbank"):
        rows = [row for row in records if row["dataset"] == dataset]
        sections[dataset] = {
            "n_cells": len(rows),
            "hals_basis_cosine": qsummary(row["basis_cosine"] for row in rows),
            "poisson_basis_cosine": qsummary(row["poisson_basis_cosine"] for row in rows),
            "hals_heldout_nmse": qsummary(row["heldout_nmse"] for row in rows),
            "poisson_heldout_nmse": qsummary(row["poisson_heldout_nmse"] for row in rows),
            "euclidean_mu_heldout_nmse": qsummary(row["euclidean_mu_heldout_nmse"] for row in rows),
            "hals_runtime": qsummary(row["runtime_seconds"] for row in rows),
            "hals_sweeps": qsummary(row["sweeps"] for row in rows),
            "hals_converged_best": int(sum(row["converged_best"] for row in rows)),
            "hals_all_starts_converged_cells": int(sum(row["converged_starts"] == N_STARTS for row in rows)),
            "hals_better_training_sse_than_mu": int(sum(row["objective"] < row["euclidean_mu_objective"] for row in rows)),
            "hals_heldout_wins_over_poisson": int(sum(row["heldout_nmse"] < row["poisson_heldout_nmse"] for row in rows)),
            "hals_heldout_wins_over_euclidean_mu": int(sum(row["heldout_nmse"] < row["euclidean_mu_heldout_nmse"] for row in rows)),
            "max_relative_objective_increase": float(max(row["max_relative_increase"] for row in rows)),
        }
    checks = {
        "record_count": len(records) == 32,
        "finite": all(np.isfinite(row[field]) for row in records for field in ("basis_cosine", "fine_nmse", "heldout_nmse", "objective", "runtime_seconds")),
        "objective_monotonicity": all(row["max_relative_increase"] <= 1e-8 for row in records),
    }
    return {"protocol": {"starts": N_STARTS, "max_sweeps": MAX_SWEEPS, "relative_tolerance": REL_TOL}, "datasets": sections, "checks": checks}


def main():
    started = time.perf_counter()
    records = controlled_records() + massbank_records()
    report = report_summary(records)
    report["total_runtime_seconds"] = time.perf_counter() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
