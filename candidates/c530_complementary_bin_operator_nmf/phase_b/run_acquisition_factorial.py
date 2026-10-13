"""Locked equal-budget acquisition-origin factorial for C530."""
from __future__ import annotations

import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results_acquisition_factorial"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c528 = load(
    "c528_acquisition",
    ROOT / "screening" / "c528_complementary_partition_operator_nmf" / "run_synthetic_screen.py",
)
c529 = load(
    "c529_acquisition",
    ROOT / "screening" / "c529_partition_intersection_line_nmf" / "run_synthetic_screen.py",
)

EPS = c528.EPS
P = c528.P
K = c528.K
WIDTH = 4
SEEDS = tuple(range(53400, 53404))
REGIMES = ("nominal", "low_count", "overlap_lines", "weak_component")
ARMS = {
    "complete": (0, 1, 2, 3),
    "adjacent_incomplete": (0, 0, 1, 1),
    "alternating_incomplete": (0, 0, 2, 2),
    "repeated": (0, 0, 0, 0),
}


def deviance(ys, mats, fit):
    ws, h = fit
    return float(
        sum(
            np.sum(np.maximum(w @ h @ a, EPS) - y * np.log(np.maximum(w @ h @ a, EPS)))
            for y, a, w in zip(ys, mats, ws)
        )
    )


def align_similarity(h, ref):
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    rn = ref / np.maximum(np.linalg.norm(ref, axis=1, keepdims=True), EPS)
    rr, cc = linear_sum_assignment(-(hn @ rn.T))
    return float(np.mean((hn @ rn.T)[rr, cc]))


def multistart(ys, mats, seed):
    bp = c528.backproject_fit(ys, mats, seed + 1)
    bases = c529.aligned_backprojected_bases(ys, mats, seed + 10)
    h_sep = np.mean(bases, axis=0)
    w_sep = c529.fit_w_all(ys, mats, h_sep)
    fits = [
        c528.fit_operator(ys, mats, seed + 20, h0=bp[1], w0=bp[0]),
        c528.fit_operator(ys, mats, seed + 21, h0=h_sep, w0=w_sep),
    ]
    fits.extend(c528.fit_operator(ys, mats, seed + 22 + i) for i in range(4))
    losses = [deviance(ys, mats, fit) for fit in fits]
    best = fits[int(np.argmin(losses))]
    stability = float(np.mean([align_similarity(fit[1], best[1]) for fit in fits]))
    return best, stability


def simulate_fine(regime, seed):
    rng = np.random.default_rng(seed)
    true_h = c529.line_basis(regime)
    scale = 0.20 if regime == "low_count" else 0.50
    fine, true_ws = [], []
    for _ in range(4):
        w = rng.gamma(1.7, 0.7, (48, K))
        w /= np.maximum(w.mean(0, keepdims=True), EPS)
        w *= scale
        if regime == "weak_component":
            w[:, 3] *= 0.20
        rate = 0.015 + w @ true_h
        fine.append(rng.poisson(rate).astype(float))
        true_ws.append(w)
    wt = rng.gamma(1.7, 0.7, (52, K))
    wt /= np.maximum(wt.mean(0, keepdims=True), EPS)
    wt *= scale
    if regime == "weak_component":
        wt[:, 3] *= 0.20
    rt = 0.015 + wt @ true_h
    at = c528.partition(8, 3)
    yt = rng.poisson(rt).astype(float) @ at
    return fine, true_ws, true_h, yt, at, rt


def code_cardinality(offsets):
    labels = [np.argmax(c528.partition(WIDTH, o), axis=1) for o in offsets]
    return len({tuple(label[j] for label in labels) for j in range(P)})


def relative_rank(offsets):
    actual = np.concatenate([c528.partition(WIDTH, o) for o in offsets], axis=1)
    complete = np.concatenate([c528.partition(WIDTH, o) for o in range(WIDTH)], axis=1)
    return float(np.linalg.matrix_rank(actual) / np.linalg.matrix_rank(complete))


def run_case(regime, seed):
    fine, true_ws, true_h, yt, at, rt = simulate_fine(regime, seed)
    rows = []
    for arm, offsets in ARMS.items():
        mats = [c528.partition(WIDTH, o) for o in offsets]
        ys = [x @ a for x, a in zip(fine, mats)]
        started = time.perf_counter()
        (ws, h), stability = multistart(ys, mats, seed + 100 * list(ARMS).index(arm))
        runtime = time.perf_counter() - started
        truth = np.vstack([0.015 + w @ true_h for w in true_ws])
        estimate = np.vstack([w @ h for w in ws])
        rows.append(
            {
                "regime": regime,
                "seed": seed,
                "arm": arm,
                "offsets": list(offsets),
                "basis_cosine": c528.align_basis(h, true_h),
                "fine_nmse": float(np.sum((estimate - truth) ** 2) / max(np.sum(truth**2), EPS)),
                "heldout_nmse": c528.heldout(h, yt, at, rt),
                "stability": stability,
                "runtime": runtime,
                "code_cardinality": code_cardinality(offsets),
                "relative_rank": relative_rank(offsets),
            }
        )
    oracle_started = time.perf_counter()
    ow, oh = c528.nmf(np.vstack(fine), seed + 900)
    oracle_runtime = time.perf_counter() - oracle_started
    rows.append(
        {
            "regime": regime,
            "seed": seed,
            "arm": "fine_count_oracle",
            "offsets": [],
            "basis_cosine": c528.align_basis(oh, true_h),
            "fine_nmse": float("nan"),
            "heldout_nmse": c528.heldout(oh, yt, at, rt),
            "stability": float("nan"),
            "runtime": oracle_runtime,
            "code_cardinality": P,
            "relative_rank": 1.0,
        }
    )
    return rows


def summarize(rows, total_runtime):
    def sub(arm, regime=None):
        return [r for r in rows if r["arm"] == arm and (regime is None or r["regime"] == regime)]

    def avg(arm, field, regime=None):
        return float(np.mean([r[field] for r in sub(arm, regime)]))

    incomplete = tuple(a for a in ARMS if a != "complete")
    complete_h = avg("complete", "heldout_nmse")
    best_incomplete_h = min(avg(a, "heldout_nmse") for a in incomplete)
    oracle_h = avg("fine_count_oracle", "heldout_nmse")
    cells = list(itertools.product(REGIMES, SEEDS))
    wins = 0
    for regime, seed in cells:
        current = next(r for r in rows if r["arm"] == "complete" and r["regime"] == regime and r["seed"] == seed)
        competitors = [next(r for r in rows if r["arm"] == a and r["regime"] == regime and r["seed"] == seed) for a in incomplete]
        wins += current["heldout_nmse"] < min(r["heldout_nmse"] for r in competitors)
    regime_gains = {
        regime: min(avg(a, "heldout_nmse", regime) for a in incomplete) - avg("complete", "heldout_nmse", regime)
        for regime in REGIMES
    }
    codes = {arm: code_cardinality(offsets) for arm, offsets in ARMS.items()}
    ranks = {arm: relative_rank(offsets) for arm, offsets in ARMS.items()}
    numeric_fields = ("basis_cosine", "heldout_nmse", "runtime", "code_cardinality", "relative_rank")
    finite = all(np.isfinite(r[field]) for r in rows for field in numeric_fields)
    gates = {
        "g1_complete_basis": avg("complete", "basis_cosine") >= 0.99,
        "g2_complete_heldout": complete_h <= 0.04,
        "g3_incomplete_gain": complete_h <= 0.35 * best_incomplete_h,
        "g4_cell_wins": wins >= 14,
        "g5_regime_gains": all(value > 0 for value in regime_gains.values()),
        "g6_oracle_ratio": complete_h <= 1.50 * oracle_h,
        "g7_code_order": codes["complete"] > max(codes["adjacent_incomplete"], codes["alternating_incomplete"]) > codes["repeated"],
        "g8_rank_order": abs(ranks["complete"] - 1.0) <= 1e-12 and all(ranks[a] < 1.0 for a in incomplete),
        "g9_integrity": finite and len(rows) == 80 and len({(r["regime"], r["seed"], r["arm"]) for r in rows}) == 80,
        "g10_runtime": total_runtime <= 180,
    }
    arm_summary = {
        arm: {
            "basis_cosine": avg(arm, "basis_cosine"),
            "heldout_nmse": avg(arm, "heldout_nmse"),
            "stability": avg(arm, "stability"),
            "runtime": avg(arm, "runtime"),
            "code_cardinality": codes[arm],
            "relative_rank": ranks[arm],
        }
        for arm in ARMS
    }
    arm_summary["fine_count_oracle"] = {
        "basis_cosine": avg("fine_count_oracle", "basis_cosine"),
        "heldout_nmse": oracle_h,
        "runtime": avg("fine_count_oracle", "runtime"),
    }
    return {
        "summary": {
            "complete_basis_cosine": avg("complete", "basis_cosine"),
            "complete_heldout_nmse": complete_h,
            "best_incomplete_heldout_nmse": best_incomplete_h,
            "oracle_heldout_nmse": oracle_h,
            "cell_wins": int(wins),
            "regime_gain": regime_gains,
            "total_runtime": total_runtime,
        },
        "arm_summary": arm_summary,
        "gates": gates,
        "passed": int(sum(gates.values())),
        "total": 10,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "n_records": len(rows),
    }


def main():
    started = time.perf_counter()
    rows = []
    for regime, seed in itertools.product(REGIMES, SEEDS):
        rows.extend(run_case(regime, seed))
    report = summarize(rows, time.perf_counter() - started)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(rows, indent=2, allow_nan=True))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
