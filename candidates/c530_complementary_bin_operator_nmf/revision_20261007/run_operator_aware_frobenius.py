"""Operator-aware Euclidean NMF audit for the C530 major revision."""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, nnls

HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b"
OUT = HERE / "results_operator_aware_frobenius"
EPS = 1e-12
MAX_ITER = 1000
CHECK_EVERY = 10
REL_TOL = 1e-7
PATIENCE = 5
N_STARTS = 6


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


af = load("c530_acquisition_factorial", PHASE_B / "run_acquisition_factorial.py")
mb = load("c530_massbank", PHASE_B / "run_massbank_public.py")


def aggregate_basis(h: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    bins = int(mapping.max() + 1)
    return np.vstack(
        [np.bincount(mapping, weights=row, minlength=bins) for row in h]
    )


def sse_objective(ys, mappings, ws, h) -> float:
    return float(
        0.5
        * sum(
            np.sum((y - w @ aggregate_basis(h, mapping)) ** 2)
            for y, mapping, w in zip(ys, mappings, ws)
        )
    )


def normalize(h: np.ndarray, ws: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    scale = np.maximum(np.linalg.norm(h, axis=1), EPS)
    return h / scale[:, None], [w * scale[None, :] for w in ws]


def fit_frobenius(
    ys,
    mappings,
    rank: int,
    seed: int,
    h0: np.ndarray | None = None,
    w0: list[np.ndarray] | None = None,
):
    rng = np.random.default_rng(seed)
    p = len(mappings[0])
    h = np.maximum(h0.copy(), EPS) if h0 is not None else rng.gamma(1.0, 1.0, (rank, p))
    if w0 is None:
        ws = [np.maximum(rng.gamma(1.0, 1.0, (len(y), rank)), EPS) for y in ys]
    else:
        ws = [np.maximum(w.copy(), EPS) for w in w0]
    h, ws = normalize(h, ws)
    trace = [sse_objective(ys, mappings, ws, h)]
    stable_checks = 0
    converged = False
    max_increase = 0.0
    for iteration in range(1, MAX_ITER + 1):
        for c, (y, mapping) in enumerate(zip(ys, mappings)):
            b = aggregate_basis(h, mapping)
            numerator = y @ b.T
            denominator = (ws[c] @ b) @ b.T
            ws[c] *= numerator / np.maximum(denominator, EPS)
            ws[c] = np.maximum(ws[c], EPS)
        numerator_h = np.zeros_like(h)
        denominator_h = np.zeros_like(h)
        for y, mapping, w in zip(ys, mappings, ws):
            b = aggregate_basis(h, mapping)
            numerator_h += (w.T @ y)[:, mapping]
            denominator_h += (w.T @ (w @ b))[:, mapping]
        h *= numerator_h / np.maximum(denominator_h, EPS)
        h = np.maximum(h, EPS)
        if iteration % CHECK_EVERY == 0:
            h, ws = normalize(h, ws)
            current = sse_objective(ys, mappings, ws, h)
            max_increase = max(max_increase, current - trace[-1])
            rel = (trace[-1] - current) / max(abs(trace[-1]), EPS)
            trace.append(current)
            stable_checks = stable_checks + 1 if 0.0 <= rel < REL_TOL else 0
            if stable_checks >= PATIENCE:
                converged = True
                break
    return ws, h, trace, iteration, converged, max_increase


def align_similarity(h: np.ndarray, ref: np.ndarray) -> float:
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    rn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), EPS)
    rows, cols = linear_sum_assignment(-(hn @ rn.T))
    return float(np.mean((hn @ rn.T)[rows, cols]))


def multistart_frobenius(ys, mappings, rank: int, seed: int, bp, separate):
    specs = [(bp[1], bp[0]), (separate[1], separate[0])] + [(None, None)] * 4
    fits = []
    for start, (h0, w0) in enumerate(specs):
        ws, h, trace, iterations, converged, max_increase = fit_frobenius(
            ys, mappings, rank, seed + start, h0=h0, w0=w0
        )
        fits.append(
            {
                "ws": ws,
                "h": h,
                "trace": trace,
                "iterations": iterations,
                "converged": converged,
                "max_increase": max_increase,
                "objective": trace[-1],
            }
        )
    best_index = int(np.argmin([fit["objective"] for fit in fits]))
    best = fits[best_index]
    similarities = [align_similarity(fit["h"], best["h"]) for fit in fits]
    best["start_index"] = best_index
    best["six_start_agreement"] = float(np.mean(similarities))
    best["minimum_start_agreement"] = float(np.min(similarities))
    best["n_converged_starts"] = int(sum(fit["converged"] for fit in fits))
    best["all_max_increase"] = float(max(fit["max_increase"] for fit in fits))
    best["last_relative_decrease"] = float(
        (best["trace"][-2] - best["trace"][-1]) / max(abs(best["trace"][-2]), EPS)
    )
    return best


def heldout_nnls(h: np.ndarray, y: np.ndarray, mapping: np.ndarray, truth_rate: np.ndarray) -> float:
    observed_basis = aggregate_basis(h, mapping)
    w = np.vstack([nnls(observed_basis.T, row)[0] for row in y])
    estimate = w @ h
    return float(np.sum((estimate - truth_rate) ** 2) / max(np.sum(truth_rate**2), EPS))


def qsummary(values) -> dict:
    arr = np.asarray(list(values), dtype=float)
    return {
        "median": float(np.median(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def synthetic_records() -> list[dict]:
    submitted = json.loads((PHASE_B / "results_acquisition_factorial" / "records.json").read_text())
    submitted = {
        (row["regime"], row["seed"]): row
        for row in submitted
        if row["arm"] == "complete"
    }
    records = []
    for regime, seed in itertools.product(af.REGIMES, af.SEEDS):
        fine, true_ws, true_h, yt, at, rt = af.simulate_fine(regime, seed)
        mappings = [np.argmax(af.c528.partition(af.WIDTH, origin), axis=1) for origin in range(af.WIDTH)]
        ys = [x @ af.c528.partition(af.WIDTH, origin) for origin, x in enumerate(fine)]
        bp = af.c528.backproject_fit(ys, [af.c528.partition(af.WIDTH, o) for o in range(af.WIDTH)], seed + 1)
        bases = af.c529.aligned_backprojected_bases(
            ys, [af.c528.partition(af.WIDTH, o) for o in range(af.WIDTH)], seed + 10
        )
        h_sep = np.mean(bases, axis=0)
        w_sep = af.c529.fit_w_all(
            ys, [af.c528.partition(af.WIDTH, o) for o in range(af.WIDTH)], h_sep
        )
        started = time.perf_counter()
        fit = multistart_frobenius(ys, mappings, af.K, seed + 7000, bp, (w_sep, h_sep))
        runtime = time.perf_counter() - started
        truth = np.vstack([0.015 + w @ true_h for w in true_ws])
        estimate = np.vstack([w @ fit["h"] for w in fit["ws"]])
        records.append(
            {
                "dataset": "controlled",
                "regime": regime,
                "seed": seed,
                "method": "operator_aware_frobenius",
                "basis_cosine": af.c528.align_basis(fit["h"], true_h),
                "fine_nmse": float(np.sum((estimate - truth) ** 2) / max(np.sum(truth**2), EPS)),
                "heldout_nmse": heldout_nnls(
                    fit["h"], yt, np.argmax(at, axis=1), rt
                ),
                "training_objective": fit["objective"],
                "iterations": fit["iterations"],
                "converged": fit["converged"],
                "runtime": runtime,
                "six_start_agreement": fit["six_start_agreement"],
                "minimum_start_agreement": fit["minimum_start_agreement"],
                "n_converged_starts": fit["n_converged_starts"],
                "max_objective_increase": fit["all_max_increase"],
                "last_relative_decrease": fit["last_relative_decrease"],
                "poisson_basis_cosine": submitted[(regime, seed)]["basis_cosine"],
                "poisson_heldout_nmse": submitted[(regime, seed)]["heldout_nmse"],
            }
        )
    return records


def massbank_records() -> list[dict]:
    submitted_rows = json.loads((PHASE_B / "results_massbank" / "records.json").read_text())
    submitted = {
        (row["panel"], row["depth"], row["seed"]): row
        for row in submitted_rows
        if row["method"] == "c530_certified_operator"
    }
    panels, _ = mb.panels_and_provenance()
    records = []
    for panel_index, panel in enumerate(panels):
        for depth_name, seed in itertools.product(mb.DEPTHS, mb.SEEDS):
            rng = np.random.default_rng(seed + 100 * panel_index)
            true_h = np.vstack([row["vector"] for row in panel])
            true_h /= np.maximum(true_h.mean(1, keepdims=True), EPS)
            mappings = [mb.bin_map(4, origin) for origin in range(4)]
            ys, true_ws = [], []
            scale = mb.DEPTHS[depth_name]
            for mapping in mappings:
                w = rng.gamma(1.7, 0.7, (40, mb.K))
                w /= np.maximum(w.mean(0, keepdims=True), EPS)
                w *= scale
                rate = 0.002 + w @ true_h
                fine = rng.poisson(rate).astype(float)
                ys.append(mb.aggregate(fine, mapping))
                true_ws.append(w)
            wt = rng.gamma(1.7, 0.7, (48, mb.K))
            wt /= np.maximum(wt.mean(0, keepdims=True), EPS)
            wt *= scale
            rt = 0.002 + wt @ true_h
            held_mapping = mb.bin_map(8, 3)
            yt = mb.aggregate(rng.poisson(rt).astype(float), held_mapping)
            bp = mb.backproject_fit(ys, mappings, seed + 1)
            separate = mb.separate_fit(ys, mappings, seed + 10)
            started = time.perf_counter()
            fit = multistart_frobenius(ys, mappings, mb.K, seed + 8000, bp, separate)
            runtime = time.perf_counter() - started
            truth = np.vstack([0.002 + w @ true_h for w in true_ws])
            estimate = np.vstack([w @ fit["h"] for w in fit["ws"]])
            key = (panel_index, depth_name, seed)
            records.append(
                {
                    "dataset": "massbank",
                    "panel": panel_index,
                    "depth": depth_name,
                    "seed": seed,
                    "method": "operator_aware_frobenius",
                    "basis_cosine": mb.basis_cosine(fit["h"], true_h),
                    "fine_nmse": float(np.sum((estimate - truth) ** 2) / max(np.sum(truth**2), EPS)),
                    "heldout_nmse": heldout_nnls(fit["h"], yt, held_mapping, rt),
                    "training_objective": fit["objective"],
                    "iterations": fit["iterations"],
                    "converged": fit["converged"],
                    "runtime": runtime,
                    "six_start_agreement": fit["six_start_agreement"],
                    "minimum_start_agreement": fit["minimum_start_agreement"],
                    "n_converged_starts": fit["n_converged_starts"],
                    "max_objective_increase": fit["all_max_increase"],
                    "last_relative_decrease": fit["last_relative_decrease"],
                    "poisson_basis_cosine": submitted[key]["basis_cosine"],
                    "poisson_heldout_nmse": submitted[key]["heldout_nmse"],
                }
            )
    return records


def summarize(records: list[dict]) -> dict:
    report = {}
    for dataset in ("controlled", "massbank"):
        rows = [row for row in records if row["dataset"] == dataset]
        report[dataset] = {
            "n_cells": len(rows),
            "frobenius_basis_cosine": qsummary(row["basis_cosine"] for row in rows),
            "poisson_basis_cosine": qsummary(row["poisson_basis_cosine"] for row in rows),
            "frobenius_heldout_nmse": qsummary(row["heldout_nmse"] for row in rows),
            "poisson_heldout_nmse": qsummary(row["poisson_heldout_nmse"] for row in rows),
            "frobenius_fine_nmse": qsummary(row["fine_nmse"] for row in rows),
            "frobenius_runtime": qsummary(row["runtime"] for row in rows),
            "six_start_agreement": qsummary(row["six_start_agreement"] for row in rows),
            "converged_best_fits": int(sum(row["converged"] for row in rows)),
            "all_starts_converged_cells": int(sum(row["n_converged_starts"] == N_STARTS for row in rows)),
            "frobenius_basis_wins": int(
                sum(row["basis_cosine"] > row["poisson_basis_cosine"] for row in rows)
            ),
            "frobenius_heldout_wins": int(
                sum(row["heldout_nmse"] < row["poisson_heldout_nmse"] for row in rows)
            ),
            "maximum_objective_increase": float(max(row["max_objective_increase"] for row in rows)),
            "last_relative_decrease": qsummary(row["last_relative_decrease"] for row in rows),
        }
    checks = {
        "record_count": len(records) == 32,
        "finite": all(
            np.isfinite(row[field])
            for row in records
            for field in (
                "basis_cosine",
                "fine_nmse",
                "heldout_nmse",
                "training_objective",
                "runtime",
                "six_start_agreement",
            )
        ),
        "monotone_with_tolerance": all(row["max_objective_increase"] <= 1e-7 for row in records),
        "best_fit_convergence": all(row["converged"] for row in records),
    }
    return {"protocol": {"starts": N_STARTS, "max_iter": MAX_ITER, "relative_tolerance": REL_TOL}, "datasets": report, "checks": checks}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--rel-tol", type=float, default=REL_TOL)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main() -> int:
    global MAX_ITER, REL_TOL, OUT
    args = parse_args()
    MAX_ITER = args.max_iter
    REL_TOL = args.rel_tol
    if args.label:
        if not args.label.replace("_", "").isalnum():
            raise ValueError("label must contain only letters, digits, and underscores")
        OUT = HERE / f"results_operator_aware_frobenius_{args.label}"
    started = time.perf_counter()
    records = synthetic_records() + massbank_records()
    report = summarize(records)
    report["total_runtime_seconds"] = time.perf_counter() - started
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
