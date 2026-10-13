"""Locked CILS-native MCR-ALS external-baseline audit for C530."""
from __future__ import annotations

import importlib.util
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "results_mcr_als_baseline"
EPS = 1e-12
K = 4


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c528 = load(
    "c528_mcr",
    ROOT / "screening" / "c528_complementary_partition_operator_nmf" / "run_synthetic_screen.py",
)
c529 = load(
    "c529_mcr",
    ROOT / "screening" / "c529_partition_intersection_line_nmf" / "run_synthetic_screen.py",
)
massbank = load("massbank_mcr", HERE / "run_massbank_public.py")


def backproject(y, a):
    return y @ a.T / np.maximum(a.sum(0)[None, :] @ a.T, EPS)


def mcr_als(x, seed, max_iterations=60, tolerance=1e-7):
    initializer = NMF(
        n_components=K,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=300,
        tol=1e-6,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        c = initializer.fit_transform(np.maximum(x, 0))
    st = initializer.components_
    previous = float(np.sum((x - c @ st) ** 2))
    iterations = 0
    for iteration in range(max_iterations):
        for j in range(x.shape[1]):
            st[:, j] = nnls(c, x[:, j])[0]
        for i in range(x.shape[0]):
            c[i, :] = nnls(st.T, x[i, :])[0]
        scale = np.maximum(st.mean(1), EPS)
        st /= scale[:, None]
        c *= scale[None, :]
        current = float(np.sum((x - c @ st) ** 2))
        iterations = iteration + 1
        if abs(previous - current) <= tolerance * max(previous, EPS):
            previous = current
            break
        previous = current
    return c, np.maximum(st, EPS), previous, iterations


def fit_heldout_nnls(y, coarse_basis):
    return np.vstack([nnls(coarse_basis.T, row)[0] for row in y])


def evaluate_mcr(ys, mats, true_h, yt, at, rt, seed):
    x = np.vstack([backproject(y, a) for y, a in zip(ys, mats)])
    started = time.perf_counter()
    _, h, sse, iterations = mcr_als(x, seed)
    runtime = time.perf_counter() - started
    wt = fit_heldout_nnls(yt, h @ at)
    heldout = float(np.sum((wt @ h - rt) ** 2) / max(np.sum(rt**2), EPS))
    return {
        "basis_cosine": c528.align_basis(h, true_h),
        "heldout_nmse": heldout,
        "runtime": runtime,
        "backproject_sse": sse,
        "iterations": iterations,
    }


def synthetic_rows():
    locked = json.loads(
        (ROOT / "screening" / "c530_confirmatory_line_operator_nmf" / "results" / "records.json").read_text()
    )
    index = {(r["regime"], r["seed"], r["method"]): r for r in locked}
    rows = []
    intended = c529.REGIMES[:-1]
    for regime, seed in itertools.product(intended, range(53000, 53008)):
        ys, mats, _, _, true_h, yt, at, _, rt = c529.simulate(regime, seed)
        mcr = evaluate_mcr(ys, mats, true_h, yt, at, rt, seed + 700)
        candidate = index[(regime, seed, "c530_multistart_operator")]
        kl = index[(regime, seed, "uniform_backproject")]
        rows.append(
            {
                "source": "synthetic",
                "regime": regime,
                "seed": seed,
                "mcr_als": mcr,
                "candidate": {k: candidate[k] for k in ("basis_cosine", "heldout_nmse", "runtime")},
                "kl_backproject": {k: kl[k] for k in ("basis_cosine", "heldout_nmse", "runtime")},
            }
        )
    return rows


def public_case(panel_index, panel, depth_name, seed):
    rng = np.random.default_rng(seed + 100 * panel_index)
    true_h = np.vstack([r["vector"] for r in panel])
    true_h /= np.maximum(true_h.mean(1, keepdims=True), EPS)
    maps = [massbank.bin_map(4, o) for o in range(4)]
    ys = []
    scale = massbank.DEPTHS[depth_name]
    for mapping in maps:
        w = rng.gamma(1.7, 0.7, (40, K))
        w /= np.maximum(w.mean(0, keepdims=True), EPS)
        w *= scale
        rate = 0.002 + w @ true_h
        ys.append(massbank.aggregate(rng.poisson(rate).astype(float), mapping))
    wt = rng.gamma(1.7, 0.7, (48, K))
    wt /= np.maximum(wt.mean(0, keepdims=True), EPS)
    wt *= scale
    rt = 0.002 + wt @ true_h
    mt = massbank.bin_map(8, 3)
    yt = massbank.aggregate(rng.poisson(rt).astype(float), mt)
    # Convert index maps to explicit aggregation matrices for the common MCR code.
    mats = []
    for mapping in maps:
        a = np.zeros((massbank.P, int(mapping.max() + 1)))
        a[np.arange(massbank.P), mapping] = 1.0
        mats.append(a)
    at = np.zeros((massbank.P, int(mt.max() + 1)))
    at[np.arange(massbank.P), mt] = 1.0
    return ys, mats, true_h, yt, at, rt


def public_rows():
    locked = json.loads((HERE / "results_massbank" / "records.json").read_text())
    index = {(r["panel"], r["depth"], r["seed"], r["method"]): r for r in locked}
    panels, _ = massbank.panels_and_provenance()
    rows = []
    for panel_index, depth, seed in itertools.product(range(len(panels)), massbank.DEPTHS, massbank.SEEDS):
        ys, mats, true_h, yt, at, rt = public_case(panel_index, panels[panel_index], depth, seed)
        mcr = evaluate_mcr(ys, mats, true_h, yt, at, rt, seed + 800 + 10 * panel_index)
        candidate = index[(panel_index, depth, seed, "c530_certified_operator")]
        kl = index[(panel_index, depth, seed, "uniform_backproject")]
        rows.append(
            {
                "source": "massbank_derived",
                "panel": panel_index,
                "depth": depth,
                "seed": seed,
                "mcr_als": mcr,
                "candidate": {k: candidate[k] for k in ("basis_cosine", "heldout_nmse", "runtime")},
                "kl_backproject": {k: kl[k] for k in ("basis_cosine", "heldout_nmse", "runtime")},
            }
        )
    return rows


def summarize(rows, total_runtime):
    synthetic = [r for r in rows if r["source"] == "synthetic"]
    public = [r for r in rows if r["source"] == "massbank_derived"]

    def avg(group, method, field):
        return float(np.mean([r[method][field] for r in group]))

    def stronger_external(r):
        return min(r["mcr_als"]["heldout_nmse"], r["kl_backproject"]["heldout_nmse"])

    synthetic_wins = sum(r["candidate"]["heldout_nmse"] < stronger_external(r) for r in synthetic)
    public_wins = sum(r["candidate"]["heldout_nmse"] < stronger_external(r) for r in public)
    regime_gains = {
        regime: float(
            np.mean(
                [
                    stronger_external(r) - r["candidate"]["heldout_nmse"]
                    for r in synthetic
                    if r["regime"] == regime
                ]
            )
        )
        for regime in sorted({r["regime"] for r in synthetic})
    }
    stratum_gains = {
        f"panel{panel}_{depth}": float(
            np.mean(
                [
                    stronger_external(r) - r["candidate"]["heldout_nmse"]
                    for r in public
                    if r["panel"] == panel and r["depth"] == depth
                ]
            )
        )
        for panel in sorted({r["panel"] for r in public})
        for depth in sorted({r["depth"] for r in public})
    }
    finite = all(
        np.isfinite(value)
        for r in rows
        for method in ("mcr_als", "candidate", "kl_backproject")
        for value in r[method].values()
    )
    gates = {
        "g1_finite": finite and len(rows) == 56,
        "g2_synthetic_basis": avg(synthetic, "candidate", "basis_cosine") > avg(synthetic, "mcr_als", "basis_cosine"),
        "g3_synthetic_heldout": avg(synthetic, "candidate", "heldout_nmse") < avg(synthetic, "mcr_als", "heldout_nmse"),
        "g4_synthetic_wins": synthetic_wins >= 36,
        "g5_synthetic_regimes": all(value > 0 for value in regime_gains.values()),
        "g6_public_basis": avg(public, "candidate", "basis_cosine") > avg(public, "mcr_als", "basis_cosine"),
        "g7_public_heldout": avg(public, "candidate", "heldout_nmse") < avg(public, "mcr_als", "heldout_nmse"),
        "g8_public_wins_strata": public_wins >= 12 and all(value > 0 for value in stratum_gains.values()),
        "g9_integrity": len(synthetic) == 40
        and len(public) == 16
        and len({(r["regime"], r["seed"]) for r in synthetic}) == 40
        and len({(r["panel"], r["depth"], r["seed"]) for r in public}) == 16,
        "g10_runtime": total_runtime <= 300,
    }
    summary = {
        "synthetic": {
            "candidate_basis": avg(synthetic, "candidate", "basis_cosine"),
            "mcr_basis": avg(synthetic, "mcr_als", "basis_cosine"),
            "kl_backproject_basis": avg(synthetic, "kl_backproject", "basis_cosine"),
            "candidate_heldout": avg(synthetic, "candidate", "heldout_nmse"),
            "mcr_heldout": avg(synthetic, "mcr_als", "heldout_nmse"),
            "kl_backproject_heldout": avg(synthetic, "kl_backproject", "heldout_nmse"),
            "wins": int(synthetic_wins),
            "regime_gain": regime_gains,
        },
        "massbank_derived": {
            "candidate_basis": avg(public, "candidate", "basis_cosine"),
            "mcr_basis": avg(public, "mcr_als", "basis_cosine"),
            "kl_backproject_basis": avg(public, "kl_backproject", "basis_cosine"),
            "candidate_heldout": avg(public, "candidate", "heldout_nmse"),
            "mcr_heldout": avg(public, "mcr_als", "heldout_nmse"),
            "kl_backproject_heldout": avg(public, "kl_backproject", "heldout_nmse"),
            "wins": int(public_wins),
            "stratum_gain": stratum_gains,
        },
        "total_runtime": total_runtime,
    }
    return {
        "summary": summary,
        "gates": gates,
        "passed": int(sum(gates.values())),
        "total": 10,
        "decision": "PASS" if all(gates.values()) else "FAIL",
        "n_records": len(rows),
    }


def main():
    started = time.perf_counter()
    rows = synthetic_rows() + public_rows()
    report = summarize(rows, time.perf_counter() - started)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(rows, indent=2))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
