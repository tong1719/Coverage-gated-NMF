"""Locked noncyclic partial-edge validation for C530."""
from __future__ import annotations

import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results_noncyclic_edges"
P, K, WIDTH = 96, 4, 4
SEEDS = tuple(range(53700, 53705))
REGIMES = ("nominal", "low_count", "overlap_lines", "edge_lines")
ARMS = {
    "complete": (0, 1, 2, 3),
    "adjacent": (0, 0, 1, 1),
    "alternating": (0, 0, 2, 2),
    "repeated": (0, 0, 0, 0),
}
EPS = 1e-12


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


c528 = load(
    "c528_noncyclic",
    ROOT / "screening" / "c528_complementary_partition_operator_nmf" / "run_synthetic_screen.py",
)
c529 = load(
    "c529_noncyclic",
    ROOT / "screening" / "c529_partition_intersection_line_nmf" / "run_synthetic_screen.py",
)
c530 = load(
    "c530_noncyclic",
    ROOT / "screening" / "c530_confirmatory_line_operator_nmf" / "run_confirmatory_screen.py",
)


def partition(width: int, origin: int) -> np.ndarray:
    columns = []
    for start in range(-origin, P, width):
        column = np.zeros(P)
        column[max(0, start) : min(P, start + width)] = 1.0
        if column.sum() > 0:
            columns.append(column)
    return np.column_stack(columns)


def coverage(offsets: tuple[int, ...]) -> tuple[float, int, int]:
    actual = np.concatenate([partition(WIDTH, o) for o in offsets], axis=1)
    complete = np.concatenate([partition(WIDTH, o) for o in range(WIDTH)], axis=1)
    actual_rank = int(np.linalg.matrix_rank(actual))
    complete_rank = int(np.linalg.matrix_rank(complete))
    return actual_rank / complete_rank, actual_rank, complete_rank


def basis(regime: str) -> np.ndarray:
    if regime != "edge_lines":
        return c529.line_basis("overlap_lines" if regime == "overlap_lines" else "nominal")
    positions = (
        (0, 4, 17, 41, 70, 95),
        (1, 12, 29, 53, 82, 94),
        (2, 21, 37, 62, 88, 93),
        (3, 9, 34, 58, 77, 92),
    )
    weights = np.array([1.00, 0.72, 0.55, 0.90, 0.64, 0.82])
    h = np.full((K, P), 0.003)
    for k, locations in enumerate(positions):
        for j, value in zip(locations, weights):
            h[k, j] += value * (1 + 0.08 * k)
    return h / np.maximum(h.mean(axis=1, keepdims=True), EPS)


def simulate(regime: str, seed: int):
    rng = np.random.default_rng(seed)
    h = basis(regime)
    scale = 0.20 if regime == "low_count" else 0.50
    ws, fines = [], []
    for _ in range(4):
        w = rng.gamma(1.7, 0.7, (48, K))
        w /= np.maximum(w.mean(axis=0, keepdims=True), EPS)
        w *= scale
        rate = 0.015 + w @ h
        ws.append(w)
        fines.append(rng.poisson(rate).astype(float))
    wt = rng.gamma(1.7, 0.7, (60, K))
    wt /= np.maximum(wt.mean(axis=0, keepdims=True), EPS)
    wt *= scale
    heldout_rate = 0.015 + wt @ h
    heldout_operator = partition(7, 3)
    heldout_y = rng.poisson(heldout_rate).astype(float) @ heldout_operator
    return ws, fines, h, heldout_y, heldout_operator, heldout_rate


def initialize(ys, mats, seed):
    backprojected = c528.backproject_fit(ys, mats, seed + 1)
    bases = c529.aligned_backprojected_bases(ys, mats, seed + 10)
    arithmetic = np.mean(bases, axis=0)
    intersection = c529.intersection_basis(bases)
    intersection_w = c529.fit_w_all(ys, mats, intersection)
    return backprojected, arithmetic, intersection, intersection_w


def multistart(ys, mats, seed, bp, intersection, intersection_w):
    fits = [
        c528.fit_operator(ys, mats, seed, h0=bp[1], w0=bp[0]),
        c528.fit_operator(ys, mats, seed + 1, h0=intersection, w0=intersection_w),
    ]
    fits.extend(c528.fit_operator(ys, mats, seed + 2 + i) for i in range(4))
    losses = np.array([c530.deviance(ys, mats, fit) for fit in fits])
    order = np.argsort(losses)
    best, second = fits[int(order[0])], fits[int(order[1])]
    top_two = c530.align_similarity(best[1], second[1])
    relative_gap = float((losses[order[1]] - losses[order[0]]) / max(abs(losses[order[0]]), EPS))
    return best, float(top_two), relative_gap


def metrics(fit, true_ws, true_h, heldout_y, heldout_operator, heldout_rate):
    ws, h = fit
    true_rate = np.vstack([0.015 + w @ true_h for w in true_ws])
    fitted_rate = np.vstack([w @ h for w in ws])
    return {
        "basis_cosine": c528.align_basis(h, true_h),
        "fine_nmse": float(
            np.sum((fitted_rate - true_rate) ** 2) / max(np.sum(true_rate**2), EPS)
        ),
        "heldout_nmse": c528.heldout(h, heldout_y, heldout_operator, heldout_rate),
    }


def run_cell(regime: str, seed: int):
    true_ws, fines, true_h, heldout_y, heldout_operator, heldout_rate = simulate(regime, seed)
    rows = []
    complete_fit = None
    complete_top_two = None
    for arm_index, (arm, offsets) in enumerate(ARMS.items()):
        mats = [partition(WIDTH, o) for o in offsets]
        ys = [fine @ matrix for fine, matrix in zip(fines, mats)]
        bp, arithmetic, intersection, intersection_w = initialize(ys, mats, seed + 1000 * arm_index)
        fit, top_two, relative_gap = multistart(
            ys, mats, seed + 100 + 1000 * arm_index, bp, intersection, intersection_w
        )
        ratio, operator_rank, full_rank = coverage(offsets)
        row = {
            "regime": regime,
            "seed": seed,
            "arm": arm,
            "method": "c530_operator",
            "coverage_ratio": ratio,
            "operator_rank": operator_rank,
            "complete_rank": full_rank,
            "top_two_cosine": top_two,
            "top_two_relative_gap": relative_gap,
            **metrics(fit, true_ws, true_h, heldout_y, heldout_operator, heldout_rate),
        }
        rows.append(row)
        if arm == "complete":
            complete_fit, complete_top_two = fit, top_two
            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "arm": arm,
                    "method": "uniform_backproject",
                    "coverage_ratio": ratio,
                    "operator_rank": operator_rank,
                    "complete_rank": full_rank,
                    "top_two_cosine": None,
                    "top_two_relative_gap": None,
                    **metrics(bp, true_ws, true_h, heldout_y, heldout_operator, heldout_rate),
                }
            )
            arithmetic_fit = (c529.fit_w_all(ys, mats, arithmetic), arithmetic)
            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "arm": arm,
                    "method": "separate_arithmetic",
                    "coverage_ratio": ratio,
                    "operator_rank": operator_rank,
                    "complete_rank": full_rank,
                    "top_two_cosine": None,
                    "top_two_relative_gap": None,
                    **metrics(arithmetic_fit, true_ws, true_h, heldout_y, heldout_operator, heldout_rate),
                }
            )
            single = c528.fit_operator(ys, mats, seed + 700)
            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "arm": arm,
                    "method": "single_correct_operator",
                    "coverage_ratio": ratio,
                    "operator_rank": operator_rank,
                    "complete_rank": full_rank,
                    "top_two_cosine": None,
                    "top_two_relative_gap": None,
                    **metrics(single, true_ws, true_h, heldout_y, heldout_operator, heldout_rate),
                }
            )
            wrong_mats = [mats[0], mats[2], mats[3], mats[1]]
            wrong = c528.fit_operator(ys, wrong_mats, seed + 800)
            rows.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "arm": arm,
                    "method": "wrong_operator",
                    "coverage_ratio": ratio,
                    "operator_rank": operator_rank,
                    "complete_rank": full_rank,
                    "top_two_cosine": None,
                    "top_two_relative_gap": None,
                    **metrics(wrong, true_ws, true_h, heldout_y, heldout_operator, heldout_rate),
                }
            )

    oracle_w, oracle_h = c528.nmf(np.vstack(fines), seed + 900)
    cuts = np.cumsum([len(x) for x in fines])[:-1]
    oracle = (list(np.split(oracle_w, cuts)), oracle_h)
    ratio, operator_rank, full_rank = coverage(ARMS["complete"])
    rows.append(
        {
            "regime": regime,
            "seed": seed,
            "arm": "fine_count",
            "method": "fine_count_oracle",
            "coverage_ratio": ratio,
            "operator_rank": operator_rank,
            "complete_rank": full_rank,
            "top_two_cosine": None,
            "top_two_relative_gap": None,
            **metrics(oracle, true_ws, true_h, heldout_y, heldout_operator, heldout_rate),
        }
    )
    assert complete_fit is not None and complete_top_two is not None
    return rows


def summarize(rows, total_runtime):
    def subset(method=None, arm=None, regime=None):
        return [
            r
            for r in rows
            if (method is None or r["method"] == method)
            and (arm is None or r["arm"] == arm)
            and (regime is None or r["regime"] == regime)
        ]

    def mean(method, field, arm=None, regime=None):
        return float(np.mean([r[field] for r in subset(method, arm, regime)]))

    candidate = subset("c530_operator", "complete")
    comparators = ("uniform_backproject", "separate_arithmetic", "wrong_operator")
    incomplete_arms = ("adjacent", "alternating", "repeated")
    basis_mean = mean("c530_operator", "basis_cosine", "complete")
    heldout_mean = mean("c530_operator", "heldout_nmse", "complete")
    strongest_nonoperator = min(mean(m, "heldout_nmse", "complete") for m in comparators)
    oracle_mean = mean("fine_count_oracle", "heldout_nmse", "fine_count")
    nonoperator_wins = 0
    incomplete_wins = 0
    for row in candidate:
        regime, seed = row["regime"], row["seed"]
        comparison_values = [
            next(r for r in subset(m, "complete", regime) if r["seed"] == seed)["heldout_nmse"]
            for m in comparators
        ]
        if row["heldout_nmse"] < min(comparison_values):
            nonoperator_wins += 1
        incomplete_values = [
            next(
                r
                for r in subset("c530_operator", arm, regime)
                if r["seed"] == seed
            )["heldout_nmse"]
            for arm in incomplete_arms
        ]
        if row["heldout_nmse"] < min(incomplete_values):
            incomplete_wins += 1
    regime_advantage = {}
    for regime in REGIMES:
        gains = []
        for row in subset("c530_operator", "complete", regime):
            incomplete_values = [
                next(
                    r
                    for r in subset("c530_operator", arm, regime)
                    if r["seed"] == row["seed"]
                )["heldout_nmse"]
                for arm in incomplete_arms
            ]
            gains.append(min(incomplete_values) - row["heldout_nmse"])
        regime_advantage[regime] = float(np.mean(gains))
    coverage_diagnostics = {
        arm: {
            "offsets": list(offsets),
            "ratio": coverage(offsets)[0],
            "rank": coverage(offsets)[1],
            "complete_rank": coverage(offsets)[2],
        }
        for arm, offsets in ARMS.items()
    }
    finite = all(
        np.isfinite(r[field])
        for r in rows
        for field in ("basis_cosine", "fine_nmse", "heldout_nmse", "coverage_ratio")
    )
    unique = len({(r["regime"], r["seed"], r["arm"], r["method"]) for r in rows}) == len(rows)
    top_two_all = all(r["top_two_cosine"] >= 0.95 for r in candidate)
    gates = {
        "g1_basis": basis_mean >= 0.985,
        "g2_heldout": heldout_mean <= 0.080,
        "g3_nonoperator_gain": heldout_mean <= 0.35 * strongest_nonoperator,
        "g4_nonoperator_wins": nonoperator_wins == 20,
        "g5_incomplete_wins": incomplete_wins >= 18,
        "g6_regime_advantage": all(value > 0 for value in regime_advantage.values()),
        "g7_oracle_ratio": heldout_mean <= 1.50 * oracle_mean,
        "g8_edge_lines": mean("c530_operator", "basis_cosine", "complete", "edge_lines") >= 0.980
        and mean("c530_operator", "heldout_nmse", "complete", "edge_lines") <= 0.100,
        "g9_coverage_and_certificate": coverage_diagnostics["complete"]["rank"] == P
        and coverage_diagnostics["complete"]["ratio"] == 1.0
        and all(coverage_diagnostics[arm]["ratio"] < 1.0 for arm in incomplete_arms)
        and top_two_all,
        "g10_integrity_runtime": finite and unique and len(rows) == 180 and total_runtime <= 240,
    }
    return {
        "summary": {
            "basis_cosine": basis_mean,
            "heldout_nmse": heldout_mean,
            "strongest_nonoperator": strongest_nonoperator,
            "oracle_heldout": oracle_mean,
            "nonoperator_wins": nonoperator_wins,
            "incomplete_wins": incomplete_wins,
            "regime_advantage": regime_advantage,
            "edge_basis": mean("c530_operator", "basis_cosine", "complete", "edge_lines"),
            "edge_heldout": mean("c530_operator", "heldout_nmse", "complete", "edge_lines"),
            "mean_top_two_cosine": float(np.mean([r["top_two_cosine"] for r in candidate])),
            "total_runtime": total_runtime,
        },
        "coverage_diagnostics": coverage_diagnostics,
        "method_summary": {
            method: {
                "basis_cosine": mean(method, "basis_cosine", "complete"),
                "heldout_nmse": mean(method, "heldout_nmse", "complete"),
            }
            for method in ("c530_operator",) + comparators + ("single_correct_operator",)
        },
        "gates": gates,
        "passed": int(sum(gates.values())),
        "total": 10,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "n_records": len(rows),
    }


def main() -> int:
    start = time.perf_counter()
    rows = []
    for regime, seed in itertools.product(REGIMES, SEEDS):
        rows.extend(run_cell(regime, seed))
    report = summarize(rows, time.perf_counter() - start)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
