"""Independent-block acquisition and model-mismatch stress test."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

import run_operator_coupled_hals as hals
import run_poisson_multistart_audit as audit

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_model_mismatch_stress"
EPS = 1e-12
P = 96
K = 4
SEEDS = tuple(range(63000, 63004))
SCENARIOS = (
    "exact", "block_shift", "block_broadening", "boundary_misregistration",
    "smooth_background", "negative_binomial", "combined",
)
AF = audit.af
POISSON_MAX_ITER = 2000
POISSON_TOL = 1e-7


def shifted_basis(h: np.ndarray, amount: float) -> np.ndarray:
    locations = np.arange(h.shape[1])
    shifted = np.vstack([
        np.interp(locations - amount, locations, row, left=0.0, right=0.0)
        for row in h
    ])
    shifted = np.maximum(shifted, EPS)
    shifted /= np.maximum(shifted.mean(axis=1, keepdims=True), EPS)
    return shifted


def broadened_basis(h: np.ndarray, sigma: float) -> np.ndarray:
    result = gaussian_filter1d(h, sigma=sigma, axis=1, mode="nearest")
    result /= np.maximum(result.mean(axis=1, keepdims=True), EPS)
    return result


def true_operator(nominal: np.ndarray, amount: float) -> np.ndarray:
    if amount >= 0:
        adjusted = (1 - amount) * nominal + amount * np.roll(nominal, 1, axis=0)
    else:
        adjusted = (1 + amount) * nominal - amount * np.roll(nominal, -1, axis=0)
    if not np.allclose(adjusted.sum(axis=1), 1.0):
        raise AssertionError("Perturbed operator is not additive")
    return adjusted


def coefficient_blocks(seed: int):
    rng = np.random.default_rng(seed)
    output = []
    for n in (48, 48, 48, 48, 52):
        w = rng.gamma(1.7, 0.7, (n, K))
        w /= np.maximum(w.mean(axis=0, keepdims=True), EPS)
        output.append(w * 0.50)
    return output


def scenario_basis_and_operator(scenario, block, base_h, nominal):
    h = base_h.copy()
    actual = nominal.copy()
    background = np.zeros(P)
    shifts = (-0.35, -0.15, 0.15, 0.35, 0.25)
    widths = (0.20, 0.35, 0.50, 0.65, 0.40)
    edges = (-0.10, -0.05, 0.05, 0.10, 0.075)
    backgrounds = (0.025, 0.040, 0.055, 0.070, 0.048)
    if scenario == "block_shift":
        h = shifted_basis(h, shifts[block])
    elif scenario == "block_broadening":
        h = broadened_basis(h, widths[block])
    elif scenario == "boundary_misregistration":
        actual = true_operator(nominal, edges[block])
    elif scenario == "smooth_background":
        background = backgrounds[block] * (0.75 + 0.50 * np.arange(P) / (P - 1))
    elif scenario == "combined":
        h = shifted_basis(h, 0.5 * shifts[block])
        actual = true_operator(nominal, 0.5 * edges[block])
        background = 0.025 * (0.75 + 0.50 * np.arange(P) / (P - 1))
    return h, actual, background


def observe(rng, mean, scenario):
    if scenario == "negative_binomial":
        size = 20.0
    elif scenario == "combined":
        size = 40.0
    else:
        return rng.poisson(mean).astype(float)
    return rng.negative_binomial(size, size / (size + mean)).astype(float)


def simulate(scenario: str, seed: int):
    base_h = AF.c529.line_basis("nominal")
    ws = coefficient_blocks(seed)
    mats = [AF.c528.partition(4, origin) for origin in range(4)]
    held_a = AF.c528.partition(8, 3)
    all_nominal = mats + [held_a]
    rng = np.random.default_rng(seed + 100000 + 1000 * SCENARIOS.index(scenario))
    ys = []
    fine_rates = []
    for block, (w, nominal) in enumerate(zip(ws, all_nominal)):
        h, actual, background = scenario_basis_and_operator(scenario, block, base_h, nominal)
        rate = 0.015 + w @ h + background[None, :]
        y = observe(rng, rate @ actual, scenario)
        ys.append(y)
        fine_rates.append(rate)
    return ys[:4], mats, ws[:4], fine_rates[:4], base_h, ys[4], held_a, fine_rates[4]


def qsummary(values):
    x = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75))}


def run_cell(scenario, seed):
    started = time.perf_counter()
    ys, mats, _, _, base_h, held_y, held_a, held_rate = simulate(scenario, seed)
    mappings = [audit.observed_mapping(a) for a in mats]
    bp = AF.c528.backproject_fit(ys, mats, seed + 1)
    bases = AF.c529.aligned_backprojected_bases(ys, mats, seed + 10)
    h_sep = np.mean(bases, axis=0)
    sep = (AF.c529.fit_w_all(ys, mats, h_sep), h_sep)
    audit.MAX_ITER = POISSON_MAX_ITER
    audit.TOL = POISSON_TOL
    poisson_fits = []
    specs = [(seed + 20, bp[1], bp[0]), (seed + 21, sep[1], sep[0])]
    specs.extend((seed + 22 + index, None, None) for index in range(4))
    for fit_seed, h0, w0 in specs:
        _, extended, _, max_increase = audit.fit(ys, mappings, K, fit_seed, h0=h0, w0=w0)
        poisson_fits.append({"fit": extended, "max_increase": max_increase})
    best_poisson = min(poisson_fits, key=lambda row: row["fit"]["objective"])
    hals.MAX_SWEEPS = 1000
    hals_result = hals.solve_case(ys, mappings, 4, K, seed + 7000, bp, sep)
    best_hals = hals_result["best"]
    bp_h = bp[1]
    rows = {
        "poisson_operator": {
            "training_objective": best_poisson["fit"]["objective"],
            "template_basis_cosine": audit.basis_similarity(best_poisson["fit"]["h"], base_h),
            "heldout_nmse": AF.c528.heldout(best_poisson["fit"]["h"], held_y, held_a, held_rate),
            "iterations": best_poisson["fit"]["iteration"],
            "converged_starts": int(sum(row["fit"]["converged"] for row in poisson_fits)),
            "max_relative_objective_increase": max(row["max_increase"] for row in poisson_fits),
        },
        "frobenius_hals_operator": {
            "training_objective": best_hals["objective"],
            "template_basis_cosine": audit.basis_similarity(best_hals["h"], base_h),
            "heldout_nmse": hals.eu.heldout_nnls(best_hals["h"], held_y, np.argmax(held_a, axis=1), held_rate),
            "sweeps": best_hals["sweeps"],
            "converged_starts": hals_result["converged_starts"],
            "max_relative_objective_increase": hals_result["max_relative_increase"],
        },
        "shared_h_backprojection": {
            "template_basis_cosine": audit.basis_similarity(bp_h, base_h),
            "heldout_nmse": AF.c528.heldout(bp_h, held_y, held_a, held_rate),
        },
    }
    output = {"scenario": scenario, "seed": seed, "methods": rows, "runtime_seconds": time.perf_counter() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{scenario}_{seed}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"{scenario} {seed} elapsed={output['runtime_seconds']:.1f}s", flush=True)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poisson-max-iter", type=int, default=POISSON_MAX_ITER)
    parser.add_argument("--poisson-tol", type=float, default=POISSON_TOL)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def main():
    global POISSON_MAX_ITER, POISSON_TOL, OUT
    args = parse_args()
    POISSON_MAX_ITER = args.poisson_max_iter
    POISSON_TOL = args.poisson_tol
    if args.label:
        if not args.label.replace("_", "").isalnum():
            raise ValueError("label must contain only letters, digits, and underscores")
        OUT = HERE / f"results_model_mismatch_stress_{args.label}"
    started = time.perf_counter()
    records = [run_cell(scenario, seed) for scenario, seed in itertools.product(SCENARIOS, SEEDS)]
    scenarios = {}
    for scenario in SCENARIOS:
        cells = [row for row in records if row["scenario"] == scenario]
        methods = {}
        for name in ("poisson_operator", "frobenius_hals_operator", "shared_h_backprojection"):
            outputs = [cell["methods"][name] for cell in cells]
            methods[name] = {
                "heldout_nmse": qsummary(row["heldout_nmse"] for row in outputs),
                "template_basis_cosine": qsummary(row["template_basis_cosine"] for row in outputs),
            }
            if name != "shared_h_backprojection":
                methods[name]["converged_start_total"] = int(sum(row["converged_starts"] for row in outputs))
        scenarios[scenario] = {
            "n_cells": len(cells),
            "methods": methods,
            "poisson_wins_vs_hals": int(sum(cell["methods"]["poisson_operator"]["heldout_nmse"] < cell["methods"]["frobenius_hals_operator"]["heldout_nmse"] for cell in cells)),
            "poisson_wins_vs_backprojection": int(sum(cell["methods"]["poisson_operator"]["heldout_nmse"] < cell["methods"]["shared_h_backprojection"]["heldout_nmse"] for cell in cells)),
        }
    checks = {
        "record_count": len(records) == len(SCENARIOS) * len(SEEDS),
        "finite": all(np.isfinite(value["heldout_nmse"]) for row in records for value in row["methods"].values()),
        "objective_monotonicity": all(
            row["methods"][method]["max_relative_objective_increase"] <= 1e-8
            for row in records for method in ("poisson_operator", "frobenius_hals_operator")
        ),
    }
    report = {
        "protocol": {"seeds": SEEDS, "scenarios": SCENARIOS, "training_blocks": 4, "heldout_blocks": 1, "poisson_max_iter": POISSON_MAX_ITER, "poisson_relative_tolerance": POISSON_TOL},
        "scenarios": scenarios,
        "checks": checks,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
