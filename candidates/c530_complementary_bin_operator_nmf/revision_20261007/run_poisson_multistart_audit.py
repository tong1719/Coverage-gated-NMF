"""Nested 6/20/50-start and 420/extended-iteration C530 audit."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
PHASE_B = HERE.parent / "phase_b"
OUT = HERE / "results_poisson_multistart_audit"
EPS = 1e-12
N_STARTS = 50
MAX_ITER = 2000
ORIGINAL_ITER = 420
TOL = 1e-7
PATIENCE = 5
PREFIXES = (6, 20, 50)
NEAR_TOLERANCES = (1e-4, 1e-3)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


af = load("c530_audit_factorial", PHASE_B / "run_acquisition_factorial.py")
mb = load("c530_audit_massbank", PHASE_B / "run_massbank_public.py")
ra = load("c530_audit_raman", PHASE_B / "run_real_raman_oil_validation.py")


def aggregate_basis(h: np.ndarray, mapping: np.ndarray, bins: int) -> np.ndarray:
    return np.vstack([np.bincount(mapping, weights=row, minlength=bins) for row in h])


def kl_divergence(ys, mappings, ws, h) -> float:
    value = 0.0
    for y, mapping, w in zip(ys, mappings, ws):
        pred = np.maximum(w @ aggregate_basis(h, mapping, y.shape[1]), EPS)
        positive = y > 0
        value += float(np.sum(pred - y))
        value += float(np.sum(y[positive] * np.log(y[positive] / pred[positive])))
    return value


def fit(ys, mappings, rank, seed, h0=None, w0=None):
    rng = np.random.default_rng(seed)
    p = len(mappings[0])
    h = np.maximum(np.asarray(h0).copy(), EPS) if h0 is not None else np.maximum(rng.gamma(1, 1, (rank, p)), EPS)
    ws = [np.maximum(np.asarray(w).copy(), EPS) for w in w0] if w0 is not None else [rng.gamma(1, 1, (len(y), rank)) for y in ys]
    trace = []
    checkpoint = None
    stable = 0
    max_relative_increase = 0.0
    started = time.perf_counter()
    for iteration in range(1, MAX_ITER + 1):
        for c, (y, mapping) in enumerate(zip(ys, mappings)):
            basis = aggregate_basis(h, mapping, y.shape[1])
            pred = np.maximum(ws[c] @ basis, EPS)
            numerator = (y / pred) @ basis.T
            denominator = basis.sum(axis=1)[None, :]
            ws[c] *= numerator / np.maximum(denominator, EPS)
            ws[c] = np.maximum(ws[c], EPS)
        numerator_h = np.zeros_like(h)
        denominator_h = np.zeros((rank, 1))
        for y, mapping, w in zip(ys, mappings, ws):
            basis = aggregate_basis(h, mapping, y.shape[1])
            pred = np.maximum(w @ basis, EPS)
            numerator_h += (w.T @ (y / pred))[:, mapping]
            denominator_h += w.sum(axis=0)[:, None]
        h *= numerator_h / np.maximum(denominator_h, EPS)
        h = np.maximum(h, EPS)
        if iteration % 25 == 0:
            scale = np.maximum(h.mean(axis=1), EPS)
            h /= scale[:, None]
            ws = [w * scale[None, :] for w in ws]
        if iteration % 10 == 0:
            current = kl_divergence(ys, mappings, ws, h)
            if trace:
                previous = trace[-1][1]
                change = (previous - current) / max(abs(previous), EPS)
                max_relative_increase = max(max_relative_increase, -change)
                stable = stable + 1 if 0.0 <= change < TOL else 0
            trace.append((iteration, current))
            if iteration == ORIGINAL_ITER:
                checkpoint = {
                    "iteration": iteration,
                    "objective": current,
                    "h": h.copy(),
                    "ws": [w.copy() for w in ws],
                    "runtime": time.perf_counter() - started,
                }
            if iteration > ORIGINAL_ITER and stable >= PATIENCE:
                break
    assert checkpoint is not None
    extended = {
        "iteration": iteration,
        "objective": trace[-1][1],
        "h": h.copy(),
        "ws": [w.copy() for w in ws],
        "runtime": time.perf_counter() - started,
        "converged": stable >= PATIENCE,
    }
    return checkpoint, extended, trace, max_relative_increase


def basis_similarity(left, right) -> float:
    l = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    r = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    similarities = l @ r.T
    rows, cols = linear_sum_assignment(-similarities)
    return float(np.mean(similarities[rows, cols]))


def observed_mapping(a: np.ndarray) -> np.ndarray:
    if not np.all(a.sum(axis=1) == 1.0):
        raise ValueError("Expected a one-hot additive partition operator")
    return np.argmax(a, axis=1)


def controlled_case(regime: str, seed: int) -> dict:
    fine, _, true_h, yt, at, truth_rate = af.simulate_fine(regime, seed)
    mats = [af.c528.partition(af.WIDTH, o) for o in range(af.WIDTH)]
    mappings = [observed_mapping(a) for a in mats]
    ys = [x @ a for x, a in zip(fine, mats)]
    bp = af.c528.backproject_fit(ys, mats, seed + 1)
    bases = af.c529.aligned_backprojected_bases(ys, mats, seed + 10)
    h_sep = np.mean(bases, axis=0)
    w_sep = af.c529.fit_w_all(ys, mats, h_sep)
    starts = [(seed + 20, bp[1], bp[0]), (seed + 21, h_sep, w_sep)]
    starts.extend((seed + 22 + index, None, None) for index in range(N_STARTS - 2))
    return {
        "name": f"controlled_{regime}_{seed}",
        "kind": "controlled",
        "ys": ys,
        "mappings": mappings,
        "rank": af.K,
        "starts": starts,
        "true_h": true_h,
        "heldout_y": yt,
        "heldout_a": at,
        "heldout_rate": truth_rate,
    }


def massbank_case() -> dict:
    panel_index, depth_name, seed = 0, "standard", 53300
    panels, _ = mb.panels_and_provenance()
    panel = panels[panel_index]
    rng = np.random.default_rng(seed + 100 * panel_index)
    true_h = np.vstack([r["vector"] for r in panel])
    true_h /= np.maximum(true_h.mean(axis=1, keepdims=True), EPS)
    mappings = [mb.bin_map(4, o) for o in range(4)]
    ys = []
    for mapping in mappings:
        w = rng.gamma(1.7, 0.7, (40, mb.K))
        w /= np.maximum(w.mean(axis=0, keepdims=True), EPS)
        w *= mb.DEPTHS[depth_name]
        rate = 0.002 + w @ true_h
        ys.append(mb.aggregate(rng.poisson(rate).astype(float), mapping))
    wt = rng.gamma(1.7, 0.7, (48, mb.K))
    wt /= np.maximum(wt.mean(axis=0, keepdims=True), EPS)
    wt *= mb.DEPTHS[depth_name]
    heldout_rate = 0.002 + wt @ true_h
    heldout_mapping = mb.bin_map(8, 3)
    heldout_y = mb.aggregate(rng.poisson(heldout_rate).astype(float), heldout_mapping)
    bp = mb.backproject_fit(ys, mappings, seed + 1)
    sep = mb.separate_fit(ys, mappings, seed + 10)
    starts = [(seed + 40, bp[1], bp[0]), (seed + 41, sep[1], sep[0])]
    starts.extend((seed + 42 + index, None, None) for index in range(N_STARTS - 2))
    return {
        "name": "massbank_panel0_standard_53300",
        "kind": "massbank",
        "ys": ys,
        "mappings": mappings,
        "rank": mb.K,
        "starts": starts,
        "true_h": true_h,
        "heldout_y": heldout_y,
        "heldout_mapping": heldout_mapping,
        "heldout_rate": heldout_rate,
    }


def raman_case() -> dict:
    _, subsets, _ = ra.load_data()
    calibration, raw = subsets["CALIBRATION"]
    cohorts = ra.allocate(calibration, raw, 0)
    mats = [ra.partition(raw.shape[1], ra.WIDTH, o) for o in ra.COMPLETE_ORIGINS]
    ys = [ra.observe_coarse(x, a) for x, a in zip(cohorts, mats)]
    mappings = [observed_mapping(a) for a in mats]
    bp_w, bp_h, _ = ra.backproject_fit(ys, mats, 2, 53800 + 10)
    starts = [(53800 + 100, bp_h, bp_w)]
    starts.extend((53800 + 101 + index, None, None) for index in range(N_STARTS - 1))
    return {
        "name": "raman_retrospective_shift0_53800",
        "kind": "raman",
        "ys": ys,
        "mappings": mappings,
        "rank": 2,
        "starts": starts,
    }


def fit_metrics(case, snapshot):
    if case["kind"] == "raman":
        return {}
    h = snapshot["h"]
    if case["kind"] == "controlled":
        heldout = af.c528.heldout(h, case["heldout_y"], case["heldout_a"], case["heldout_rate"])
    else:
        heldout = mb.heldout(h, case["heldout_y"], case["heldout_mapping"], case["heldout_rate"])
    return {"matched_basis_cosine": basis_similarity(h, case["true_h"]), "heldout_nmse": heldout}


def summarize_snapshot(snapshots, prefix, tolerance):
    pool = snapshots[:prefix]
    ordered = sorted(range(prefix), key=lambda index: pool[index]["objective"])
    best = pool[ordered[0]]["objective"]
    gaps = [max(0.0, (snapshot["objective"] - best) / max(abs(best), 1.0)) for snapshot in pool]
    near = [index for index, gap in enumerate(gaps) if gap <= tolerance + 1e-14]
    similarities = [
        basis_similarity(pool[index]["h"], pool[ordered[0]]["h"])
        for index in range(prefix)
    ]
    pairwise = [
        basis_similarity(pool[i]["h"], pool[j]["h"])
        for ix, i in enumerate(near)
        for j in near[ix + 1 :]
    ]
    top_two = basis_similarity(pool[ordered[0]]["h"], pool[ordered[1]]["h"])
    return {
        "prefix": prefix,
        "tolerance": tolerance,
        "best_start": ordered[0],
        "best_deviance": float(best),
        "second_gap": float(gaps[ordered[1]]),
        "top_two_cosine": top_two,
        "near_optimal_size": len(near),
        "near_optimal_start_indices": near,
        "near_pairwise_median": float(np.median(pairwise)) if pairwise else None,
        "near_pairwise_min": float(np.min(pairwise)) if pairwise else None,
        "all_start_gap_median": float(np.median(gaps)),
        "all_start_gap_q75": float(np.quantile(gaps, 0.75)),
        "all_start_similarity_median": float(np.median(similarities)),
        "all_start_similarity_min": float(np.min(similarities)),
        "n_near_pairs_below_095": int(sum(value < 0.95 for value in pairwise)),
        "points": [
            {"start": index, "relative_gap": float(gaps[index]), "similarity_to_best": similarities[index]}
            for index in range(prefix)
        ],
    }


def run_case(case):
    started = time.perf_counter()
    records = []
    snapshots_420 = []
    snapshots_extended = []
    traces = []
    for index, (seed, h0, w0) in enumerate(case["starts"]):
        snap420, extended, trace, max_increase = fit(
            case["ys"], case["mappings"], case["rank"], seed, h0=h0, w0=w0
        )
        snapshots_420.append(snap420)
        snapshots_extended.append(extended)
        records.append(
            {
                "start": index,
                "seed": seed,
                "initialization": "backprojection" if index == 0 else ("separate" if index == 1 and case["kind"] != "raman" else "random"),
                "deviance_420": snap420["objective"],
                "deviance_extended": extended["objective"],
                "iteration_extended": extended["iteration"],
                "converged_extended": extended["converged"],
                "max_relative_objective_increase": max_increase,
                "runtime_seconds": extended["runtime"],
                "metrics_420": fit_metrics(case, snap420),
                "metrics_extended": fit_metrics(case, extended),
            }
        )
        traces.append([[int(it), float(value)] for it, value in trace])
        if (index + 1) in PREFIXES:
            print(f"{case['name']}: {index + 1}/{N_STARTS} starts, {time.perf_counter()-started:.1f}s", flush=True)
    np.savez_compressed(
        OUT / f"{case['name']}_bases.npz",
        h420=np.stack([row["h"] for row in snapshots_420]),
        hextended=np.stack([row["h"] for row in snapshots_extended]),
    )
    summaries = []
    for label, snapshots in (("420", snapshots_420), ("extended", snapshots_extended)):
        for prefix in PREFIXES:
            for tolerance in NEAR_TOLERANCES:
                summary = summarize_snapshot(snapshots, prefix, tolerance)
                summary["checkpoint"] = label
                summary["best_metrics"] = records[summary["best_start"]]["metrics_420" if label == "420" else "metrics_extended"]
                summaries.append(summary)
    output = {
        "case": case["name"],
        "kind": case["kind"],
        "n_starts": N_STARTS,
        "n_converged_extended": int(sum(row["converged_extended"] for row in records)),
        "runtime_seconds": time.perf_counter() - started,
        "records": records,
        "traces": traces,
        "summaries": summaries,
    }
    (OUT / f"{case['name']}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [controlled_case("nominal", 53400), controlled_case("low_count", 53400), massbank_case(), raman_case()]
    outputs = [run_case(case) for case in cases]
    report = {
        "protocol": {
            "starts": N_STARTS,
            "prefixes": PREFIXES,
            "original_checkpoint": ORIGINAL_ITER,
            "extended_cap": MAX_ITER,
            "relative_tolerance": TOL,
            "near_tolerances": NEAR_TOLERANCES,
        },
        "cases": [
            {
                "name": case["case"],
                "n_converged_extended": case["n_converged_extended"],
                "runtime_seconds": case["runtime_seconds"],
                "summaries": [
                    {k: v for k, v in row.items() if k != "points"}
                    for row in case["summaries"]
                ],
            }
            for case in outputs
        ],
        "checks": {
            "case_count": len(outputs) == 4,
            "start_count": all(len(output["records"]) == N_STARTS for output in outputs),
            "finite_deviance": all(np.isfinite(row["deviance_extended"]) for output in outputs for row in output["records"]),
            "objective_monotonicity": all(row["max_relative_objective_increase"] <= 1e-8 for output in outputs for row in output["records"]),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "cases": [(row["name"], row["n_converged_extended"], round(row["runtime_seconds"],1)) for row in report["cases"]]}, indent=2), flush=True)
    return 0 if all(report["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
