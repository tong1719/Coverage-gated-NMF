"""Locked rank-input sensitivity audit for C530."""
from __future__ import annotations

import importlib.util
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results_rank_sensitivity"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c528 = load(
    "c528_rank",
    ROOT / "screening" / "c528_complementary_partition_operator_nmf" / "run_synthetic_screen.py",
)
c529 = load(
    "c529_rank",
    ROOT / "screening" / "c529_partition_intersection_line_nmf" / "run_synthetic_screen.py",
)

EPS = c528.EPS
P = c528.P
RANKS = (3, 4, 5, 6)
REGIMES = ("nominal", "low_count", "overlap_lines", "weak_component")
SEEDS = tuple(range(53500, 53504))


def kl_nmf(x, rank, seed, iterations=500):
    model = NMF(
        n_components=rank,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=iterations,
        tol=1e-5,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        w = model.fit_transform(np.maximum(x, EPS))
    return np.maximum(w, EPS), np.maximum(model.components_, EPS)


def fixed_w(y, basis, w, iterations=300):
    denominator = np.ones_like(y) @ basis.T
    for _ in range(iterations):
        w *= ((y / np.maximum(w @ basis, EPS)) @ basis.T) / np.maximum(denominator, EPS)
        w = np.maximum(w, EPS)
    return w


def backproject_start(ys, mats, rank, seed):
    fine = [y @ a.T / np.maximum(a.sum(0)[None, :] @ a.T, EPS) for y, a in zip(ys, mats)]
    w, h = kl_nmf(np.vstack(fine), rank, seed)
    cuts = np.cumsum([len(x) for x in fine])[:-1]
    return list(np.split(w, cuts)), h


def fit_operator(ys, mats, rank, seed, h0=None, w0=None, iterations=420):
    rng = np.random.default_rng(seed)
    h = np.maximum(h0.copy() if h0 is not None else rng.gamma(1, 1, (rank, P)), EPS)
    ws = [np.maximum(w.copy(), EPS) for w in w0] if w0 is not None else [rng.gamma(1, 1, (len(y), rank)) for y in ys]
    for step in range(iterations):
        for c, (y, a) in enumerate(zip(ys, mats)):
            ws[c] = fixed_w(y, h @ a, ws[c], iterations=1)
        numerator = np.zeros_like(h)
        denominator = np.zeros_like(h)
        for y, a, w in zip(ys, mats, ws):
            prediction = np.maximum(w @ h @ a, EPS)
            numerator += w.T @ (y / prediction) @ a.T
            denominator += w.T @ np.ones_like(y) @ a.T
        h *= numerator / np.maximum(denominator, EPS)
        h = np.maximum(h, EPS)
        if step % 25 == 24:
            scale = np.maximum(h.mean(1), EPS)
            h /= scale[:, None]
            ws = [w * scale[None, :] for w in ws]
    return ws, h


def deviance(ys, mats, fit):
    ws, h = fit
    return float(
        sum(
            np.sum(np.maximum(w @ h @ a, EPS) - y * np.log(np.maximum(w @ h @ a, EPS)))
            for y, a, w in zip(ys, mats, ws)
        )
    )


def fit_multistart(ys, mats, rank, seed):
    bp = backproject_start(ys, mats, rank, seed + 1)
    fits = [fit_operator(ys, mats, rank, seed + 10, h0=bp[1], w0=bp[0])]
    fits.extend(fit_operator(ys, mats, rank, seed + 11 + i) for i in range(5))
    losses = [deviance(ys, mats, fit) for fit in fits]
    return fits[int(np.argmin(losses))], float(min(losses))


def component_scores(h, true_h):
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    tn = true_h / np.maximum(np.linalg.norm(true_h, axis=1, keepdims=True), EPS)
    similarities = tn @ hn.T
    recall = float(np.mean(np.max(similarities, axis=1)))
    precision = float(np.mean(np.max(similarities, axis=0)))
    harmonic = float(2 * recall * precision / max(recall + precision, EPS))
    return recall, precision, harmonic


def heldout(h, yt, at, rt):
    w = fixed_w(yt, h @ at, np.ones((len(yt), len(h))), iterations=400)
    return float(np.sum((w @ h - rt) ** 2) / max(np.sum(rt**2), EPS))


def run_case(regime, seed, rank):
    ys, mats, _, _, true_h, yt, at, _, rt = c529.simulate(regime, seed)
    started = time.perf_counter()
    (_, h), observed_deviance = fit_multistart(ys, mats, rank, seed + 100 * rank)
    runtime = time.perf_counter() - started
    recall, precision, harmonic = component_scores(h, true_h)
    return {
        "regime": regime,
        "seed": seed,
        "fitted_rank": rank,
        "heldout_nmse": heldout(h, yt, at, rt),
        "true_component_recall": recall,
        "estimated_component_precision": precision,
        "component_harmonic": harmonic,
        "observed_deviance": observed_deviance,
        "runtime": runtime,
    }


def summarize(rows, total_runtime):
    def sub(rank, regime=None):
        return [r for r in rows if r["fitted_rank"] == rank and (regime is None or r["regime"] == regime)]

    def avg(rank, field, regime=None):
        return float(np.mean([r[field] for r in sub(rank, regime)]))

    heldout = {rank: avg(rank, "heldout_nmse") for rank in RANKS}
    recall = {rank: avg(rank, "true_component_recall") for rank in RANKS}
    rank4 = heldout[4]
    overfit_heldout = float(np.mean([heldout[5], heldout[6]]))
    overfit_recall = float(np.mean([recall[5], recall[6]]))
    finite = all(np.isfinite(value) for r in rows for value in r.values() if isinstance(value, (int, float)))
    gates = {
        "g1_rank4_recall": recall[4] >= 0.99,
        "g2_rank4_heldout": rank4 <= 0.04,
        "g3_near_best_rank": rank4 <= 1.10 * min(heldout.values()),
        "g4_underfit_error": heldout[3] >= 1.20 * rank4,
        "g5_underfit_recall": recall[3] <= recall[4] - 0.05,
        "g6_overfit_error": overfit_heldout <= 1.50 * rank4,
        "g7_overfit_recall": overfit_recall >= 0.95,
        "g8_regime_bound": all(avg(4, "heldout_nmse", regime) <= 0.06 for regime in REGIMES),
        "g9_integrity": finite and len(rows) == 64 and len({(r["regime"], r["seed"], r["fitted_rank"]) for r in rows}) == 64,
        "g10_runtime": total_runtime <= 180,
    }
    rank_summary = {
        str(rank): {
            field: avg(rank, field)
            for field in (
                "heldout_nmse",
                "true_component_recall",
                "estimated_component_precision",
                "component_harmonic",
                "observed_deviance",
                "runtime",
            )
        }
        for rank in RANKS
    }
    return {
        "summary": {
            "rank4_heldout_nmse": rank4,
            "rank4_true_component_recall": recall[4],
            "best_rank_by_heldout": int(min(heldout, key=heldout.get)),
            "underfit_error_ratio": heldout[3] / max(rank4, EPS),
            "overfit_error_ratio": overfit_heldout / max(rank4, EPS),
            "total_runtime": total_runtime,
        },
        "rank_summary": rank_summary,
        "gates": gates,
        "passed": int(sum(gates.values())),
        "total": 10,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "n_records": len(rows),
    }


def main():
    started = time.perf_counter()
    rows = [run_case(regime, seed, rank) for regime, seed, rank in itertools.product(REGIMES, SEEDS, RANKS)]
    report = summarize(rows, time.perf_counter() - started)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(rows, indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
