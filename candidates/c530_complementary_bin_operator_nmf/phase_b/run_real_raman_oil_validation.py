"""Locked external real-mixture Raman validation for C530.

Protocol: LOCKED_REAL_RAMAN_OIL_PROTOCOL.md
The source contains real oil-mixture spectra; only the coarse acquisitions are
retrospective deterministic aggregations.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from scipy.stats import spearmanr
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


HERE = Path(__file__).resolve().parent
DATA = HERE / "real_raman_oil_data" / "raman-oils-raw.csv"
OUT = HERE / "results_real_raman_oil"
EPS = 1e-12
TOTAL = 1_000_000.0
WIDTH = 8
COMPLETE_ORIGINS = tuple(range(8))
INCOMPLETE_ORIGINS = (0, 0, 1, 1, 2, 2, 3, 3)
WRONG_ORIGINS = (1, 2, 0, 4, 5, 6, 7, 3)
SHIFTS = tuple(range(5))
SEEDS_BY_SHIFT = {
    shift: (53800 + 100 * shift, 53801 + 100 * shift) for shift in SHIFTS
}
HELDOUT_WIDTH = 11
HELDOUT_ORIGIN = 5


def partition(p: int, width: int, origin: int) -> np.ndarray:
    columns = []
    for start in range(-origin, p, width):
        column = np.zeros(p)
        column[max(0, start) : min(p, start + width)] = 1.0
        if column.sum() > 0:
            columns.append(column)
    return np.column_stack(columns)


def coverage(p: int, origins: tuple[int, ...]) -> tuple[int, int, float]:
    actual = np.concatenate([partition(p, WIDTH, origin) for origin in origins], axis=1)
    reference = np.concatenate(
        [partition(p, WIDTH, origin) for origin in COMPLETE_ORIGINS], axis=1
    )
    actual_rank = int(np.linalg.matrix_rank(actual))
    reference_rank = int(np.linalg.matrix_rank(reference))
    return actual_rank, reference_rank, actual_rank / reference_rank


def normalize_fine(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) / np.maximum(x.sum(axis=1, keepdims=True), EPS) * TOTAL


def observe_coarse(raw: np.ndarray, operator: np.ndarray) -> np.ndarray:
    y = np.maximum(raw, 0.0) @ operator
    return y / np.maximum(y.sum(axis=1, keepdims=True), EPS) * TOTAL


def kl_objective(ys, mats, fit) -> float:
    ws, h = fit
    total = 0.0
    for y, a, w in zip(ys, mats, ws):
        pred = np.maximum(w @ h @ a, EPS)
        positive = y > 0
        total += float(np.sum(pred - y))
        total += float(np.sum(y[positive] * np.log(y[positive] / pred[positive])))
    return total


def fixed_w(y: np.ndarray, basis: np.ndarray, w=None, max_iter: int = 400) -> np.ndarray:
    if w is None:
        w = np.ones((len(y), basis.shape[0]))
    w = np.maximum(np.asarray(w, dtype=float).copy(), EPS)
    denominator = np.ones_like(y) @ basis.T
    for _ in range(max_iter):
        pred = np.maximum(w @ basis, EPS)
        w *= ((y / pred) @ basis.T) / np.maximum(denominator, EPS)
        w = np.maximum(w, EPS)
    return w


def operator_fit(
    ys,
    mats,
    rank: int,
    seed: int,
    h0=None,
    w0=None,
    max_iter: int = 420,
    tolerance: float = 1e-7,
):
    rng = np.random.default_rng(seed)
    p = mats[0].shape[0]
    h = np.maximum(
        np.asarray(h0, dtype=float).copy()
        if h0 is not None
        else rng.gamma(1.0, 1.0, (rank, p)),
        EPS,
    )
    if w0 is None:
        ws = [rng.gamma(1.0, 1.0, (len(y), rank)) for y in ys]
    else:
        ws = [np.maximum(np.asarray(w, dtype=float).copy(), EPS) for w in w0]
    previous = np.inf
    iteration_count = 0
    for iteration in range(max_iter):
        for cohort, (y, a) in enumerate(zip(ys, mats)):
            ws[cohort] = fixed_w(y, h @ a, ws[cohort], max_iter=1)
        numerator = np.zeros_like(h)
        denominator = np.zeros_like(h)
        for y, a, w in zip(ys, mats, ws):
            pred = np.maximum(w @ h @ a, EPS)
            numerator += w.T @ (y / pred) @ a.T
            denominator += w.T @ np.ones_like(y) @ a.T
        h *= numerator / np.maximum(denominator, EPS)
        h = np.maximum(h, EPS)
        if iteration % 25 == 24:
            scale = np.maximum(h.mean(axis=1), EPS)
            h /= scale[:, None]
            ws = [w * scale[None, :] for w in ws]
        iteration_count = iteration + 1
        if iteration % 10 == 9:
            current = kl_objective(ys, mats, (ws, h))
            if np.isfinite(previous) and abs(previous - current) <= tolerance * max(
                abs(previous), EPS
            ):
                break
            previous = current
    scale = np.maximum(h.mean(axis=1), EPS)
    h /= scale[:, None]
    ws = [w * scale[None, :] for w in ws]
    return ws, h, iteration_count, kl_objective(ys, mats, (ws, h))


def kl_nmf(x: np.ndarray, rank: int, seed: int):
    model = NMF(
        n_components=rank,
        init="nndsvda",
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=420,
        tol=1e-7,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        w = model.fit_transform(np.maximum(x, EPS))
    h = np.maximum(model.components_, EPS)
    scale = np.maximum(h.mean(axis=1), EPS)
    return w * scale[None, :], h / scale[:, None], int(model.n_iter_)


def backproject(y: np.ndarray, operator: np.ndarray) -> np.ndarray:
    bin_sizes_at_channels = operator.sum(axis=0)[None, :] @ operator.T
    return y @ operator.T / np.maximum(bin_sizes_at_channels, EPS)


def backproject_fit(ys, mats, rank: int, seed: int):
    x = np.vstack([backproject(y, a) for y, a in zip(ys, mats)])
    w, h, iterations = kl_nmf(x, rank, seed)
    cuts = np.cumsum([len(y) for y in ys])[:-1]
    return list(np.split(w, cuts)), h, iterations


def mcr_als(x: np.ndarray, rank: int, seed: int, max_iter: int = 100):
    initializer = NMF(
        n_components=rank,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=300,
        tol=1e-6,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        c = initializer.fit_transform(np.maximum(x, 0.0))
    st = np.maximum(initializer.components_, EPS)
    previous = float(np.sum((x - c @ st) ** 2))
    iterations = 0
    for iteration in range(max_iter):
        for channel in range(x.shape[1]):
            st[:, channel] = nnls(c, x[:, channel])[0]
        for row in range(x.shape[0]):
            c[row, :] = nnls(st.T, x[row, :])[0]
        scale = np.maximum(st.mean(axis=1), EPS)
        st /= scale[:, None]
        c *= scale[None, :]
        current = float(np.sum((x - c @ st) ** 2))
        iterations = iteration + 1
        if abs(previous - current) <= 1e-7 * max(previous, EPS):
            break
        previous = current
    return c, np.maximum(st, EPS), iterations


def align_cosine(left: np.ndarray, right: np.ndarray) -> float:
    ln = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    rn = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    rows, cols = linear_sum_assignment(-(ln @ rn.T))
    return float(np.mean((ln @ rn.T)[rows, cols]))


def multistart_operator(ys, mats, rank: int, seed: int, bp_fit=None):
    fits = []
    if bp_fit is not None:
        bp_ws, bp_h = bp_fit
        fits.append(operator_fit(ys, mats, rank, seed, h0=bp_h, w0=bp_ws))
        random_count = 5
    else:
        random_count = 6
    for index in range(random_count):
        fits.append(operator_fit(ys, mats, rank, seed + 1 + index))
    order = np.argsort([fit[3] for fit in fits])
    best = fits[int(order[0])]
    second = fits[int(order[1])]
    return best, align_cosine(best[1], second[1]), [float(fit[3]) for fit in fits]


def endpoint_alignment(h: np.ndarray, endpoints: np.ndarray):
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)
    en = endpoints / np.maximum(np.linalg.norm(endpoints, axis=1, keepdims=True), EPS)
    endpoint_rows, learned_cols = linear_sum_assignment(-(en @ hn.T))
    order = learned_cols[np.argsort(endpoint_rows)]
    score = float(np.mean((en @ hn.T)[endpoint_rows, learned_cols]))
    return order, score


def evaluate_basis(
    h: np.ndarray,
    raw_validation: np.ndarray,
    validation_percent: np.ndarray,
    raw_prediction: np.ndarray,
    prediction_percent: np.ndarray,
    heldout_operator: np.ndarray,
    endpoints: np.ndarray,
):
    order, endpoint_cosine = endpoint_alignment(h, endpoints)

    def evaluate(raw, percentages):
        fine = normalize_fine(raw)
        y = observe_coarse(raw, heldout_operator)
        w = fixed_w(y, h @ heldout_operator, max_iter=400)
        reconstruction = w @ h
        row_nmse = np.sum((reconstruction - fine) ** 2, axis=1) / np.maximum(
            np.sum(fine**2, axis=1), EPS
        )
        row_cosine = np.sum(reconstruction * fine, axis=1) / np.maximum(
            np.linalg.norm(reconstruction, axis=1) * np.linalg.norm(fine, axis=1), EPS
        )
        oriented = w[:, order]
        evoo_fraction = oriented[:, 1] / np.maximum(oriented[:, :2].sum(axis=1), EPS)
        rho = float(spearmanr(evoo_fraction, percentages).statistic)
        return {
            "nmse": float(np.mean(row_nmse)),
            "cosine": float(np.mean(row_cosine)),
            "evoo_spearman": rho,
            "n": int(len(raw)),
        }

    return endpoint_cosine, evaluate(raw_validation, validation_percent), evaluate(
        raw_prediction, prediction_percent
    )


def load_data():
    frame = pd.read_csv(DATA, sep=";", dtype=str, keep_default_na=False)
    spectral_columns = list(frame.columns[9:])
    raw = frame[spectral_columns].astype(float).to_numpy()
    duplicate = pd.DataFrame(raw).duplicated(keep="first").to_numpy()
    retained = frame.loc[~duplicate].copy()
    retained_raw = raw[~duplicate]
    retained["_original_index"] = np.flatnonzero(~duplicate)
    retained["_evoo"] = retained["EVOO (%)"].str.rstrip("%").astype(float)
    retained["_split"] = retained["Type"].str.strip().str.upper()
    subsets = {}
    for label in ("CALIBRATION", "VALIDATION", "PREDICTION"):
        mask = retained["_split"].to_numpy() == label
        subsets[label] = (retained.loc[mask].reset_index(drop=True), retained_raw[mask])
    return spectral_columns, subsets, int(duplicate.sum())


def allocate(calibration: pd.DataFrame, raw: np.ndarray, shift: int):
    assignments = np.empty(len(calibration), dtype=int)
    for _, indices in calibration.groupby("_evoo", sort=True).groups.items():
        ordered = sorted(indices, key=lambda value: int(calibration.loc[value, "_original_index"]))
        for position, index in enumerate(ordered):
            assignments[index] = (position + shift) % len(COMPLETE_ORIGINS)
    return [raw[assignments == origin] for origin in COMPLETE_ORIGINS]


def method_record(
    method: str,
    shift: int,
    seed: int,
    rank: int,
    h: np.ndarray,
    runtime: float,
    endpoints: np.ndarray,
    validation_raw: np.ndarray,
    validation_percent: np.ndarray,
    prediction_raw: np.ndarray,
    prediction_percent: np.ndarray,
    heldout_operator: np.ndarray,
    iterations,
    top_two=None,
    objectives=None,
):
    endpoint_cosine, validation, prediction = evaluate_basis(
        h,
        validation_raw,
        validation_percent,
        prediction_raw,
        prediction_percent,
        heldout_operator,
        endpoints,
    )
    return {
        "method": method,
        "shift": shift,
        "seed": seed,
        "rank": rank,
        "runtime_seconds": runtime,
        "iterations": iterations,
        "top_two_cosine": top_two,
        "start_objectives": objectives,
        "endpoint_basis_cosine": endpoint_cosine,
        "validation": validation,
        "prediction": prediction,
    }


def run_cell(
    shift,
    seed,
    calibration,
    calibration_raw,
    validation_raw,
    validation_percent,
    prediction_raw,
    prediction_percent,
    endpoints,
    p,
):
    cohort_raw = allocate(calibration, calibration_raw, shift)
    complete_mats = [partition(p, WIDTH, origin) for origin in COMPLETE_ORIGINS]
    complete_ys = [observe_coarse(x, a) for x, a in zip(cohort_raw, complete_mats)]
    heldout_operator = partition(p, HELDOUT_WIDTH, HELDOUT_ORIGIN)
    rows = []

    started = time.perf_counter()
    bp_ws, bp_h, bp_iterations = backproject_fit(complete_ys, complete_mats, 2, seed + 10)
    rows.append(
        method_record(
            "uniform_backproject_kl_nmf",
            shift,
            seed,
            2,
            bp_h,
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            bp_iterations,
        )
    )

    started = time.perf_counter()
    complete_fit, complete_top_two, complete_objectives = multistart_operator(
        complete_ys, complete_mats, 2, seed + 100, bp_fit=(bp_ws, bp_h)
    )
    rows.append(
        method_record(
            "c530_complete_operator",
            shift,
            seed,
            2,
            complete_fit[1],
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            complete_fit[2],
            complete_top_two,
            complete_objectives,
        )
    )

    started = time.perf_counter()
    x_backprojected = np.vstack(
        [backproject(y, a) for y, a in zip(complete_ys, complete_mats)]
    )
    _, mcr_h, mcr_iterations = mcr_als(x_backprojected, 2, seed + 200)
    rows.append(
        method_record(
            "uniform_backproject_mcr_als",
            shift,
            seed,
            2,
            mcr_h,
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            mcr_iterations,
        )
    )

    wrong_mats = [partition(p, WIDTH, origin) for origin in WRONG_ORIGINS]
    assert all(y.shape[1] == a.shape[1] for y, a in zip(complete_ys, wrong_mats))
    wrong_bp_ws = [fixed_w(y, bp_h @ a) for y, a in zip(complete_ys, wrong_mats)]
    started = time.perf_counter()
    wrong_fit, wrong_top_two, wrong_objectives = multistart_operator(
        complete_ys, wrong_mats, 2, seed + 300, bp_fit=(wrong_bp_ws, bp_h)
    )
    rows.append(
        method_record(
            "wrong_operator",
            shift,
            seed,
            2,
            wrong_fit[1],
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            wrong_fit[2],
            wrong_top_two,
            wrong_objectives,
        )
    )

    incomplete_mats = [partition(p, WIDTH, origin) for origin in INCOMPLETE_ORIGINS]
    incomplete_ys = [observe_coarse(x, a) for x, a in zip(cohort_raw, incomplete_mats)]
    incomplete_bp_ws, incomplete_bp_h, _ = backproject_fit(
        incomplete_ys, incomplete_mats, 2, seed + 400
    )
    started = time.perf_counter()
    incomplete_fit, incomplete_top_two, incomplete_objectives = multistart_operator(
        incomplete_ys,
        incomplete_mats,
        2,
        seed + 500,
        bp_fit=(incomplete_bp_ws, incomplete_bp_h),
    )
    rows.append(
        method_record(
            "c530_incomplete_operator",
            shift,
            seed,
            2,
            incomplete_fit[1],
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            incomplete_fit[2],
            incomplete_top_two,
            incomplete_objectives,
        )
    )

    started = time.perf_counter()
    _, oracle_h, oracle_iterations = kl_nmf(normalize_fine(calibration_raw), 2, seed + 600)
    rows.append(
        method_record(
            "fine_grid_oracle",
            shift,
            seed,
            2,
            oracle_h,
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            oracle_iterations,
        )
    )

    started = time.perf_counter()
    rank3_fit, rank3_top_two, rank3_objectives = multistart_operator(
        complete_ys, complete_mats, 3, seed + 700
    )
    rows.append(
        method_record(
            "c530_complete_operator_rank3_sensitivity",
            shift,
            seed,
            3,
            rank3_fit[1],
            time.perf_counter() - started,
            endpoints,
            validation_raw,
            validation_percent,
            prediction_raw,
            prediction_percent,
            heldout_operator,
            rank3_fit[2],
            rank3_top_two,
            rank3_objectives,
        )
    )
    return rows


def summarize(rows, p, counts, total_runtime):
    def values(method, metric):
        return [row["prediction"][metric] for row in rows if row["method"] == method]

    def mean(method, metric):
        return float(np.mean(values(method, metric)))

    methods = sorted(set(row["method"] for row in rows))
    method_summary = {
        method: {
            "prediction_nmse": mean(method, "nmse"),
            "prediction_cosine": mean(method, "cosine"),
            "prediction_evoo_spearman": mean(method, "evoo_spearman"),
            "endpoint_basis_cosine": float(
                np.mean(
                    [row["endpoint_basis_cosine"] for row in rows if row["method"] == method]
                )
            ),
            "validation_nmse": float(
                np.mean([row["validation"]["nmse"] for row in rows if row["method"] == method])
            ),
            "runtime_seconds": float(
                np.mean([row["runtime_seconds"] for row in rows if row["method"] == method])
            ),
        }
        for method in methods
    }
    candidate = "c530_complete_operator"
    nonoperators = ("uniform_backproject_kl_nmf", "uniform_backproject_mcr_als")
    best_nonoperator = min(nonoperators, key=lambda method: mean(method, "nmse"))
    candidate_nmse = mean(candidate, "nmse")
    baseline_nmse = mean(best_nonoperator, "nmse")
    gain = 1.0 - candidate_nmse / max(baseline_nmse, EPS)

    shift_wins = 0
    shift_details = {}
    for shift in SHIFTS:
        candidate_shift = np.mean(
            [
                row["prediction"]["nmse"]
                for row in rows
                if row["method"] == candidate and row["shift"] == shift
            ]
        )
        baseline_shift = np.mean(
            [
                row["prediction"]["nmse"]
                for row in rows
                if row["method"] == best_nonoperator and row["shift"] == shift
            ]
        )
        shift_wins += int(candidate_shift < baseline_shift)
        shift_details[str(shift)] = {
            "candidate_nmse": float(candidate_shift),
            "best_nonoperator_nmse": float(baseline_shift),
        }

    candidate_rows = [row for row in rows if row["method"] == candidate]
    incomplete_index = {
        (row["shift"], row["seed"]): row
        for row in rows
        if row["method"] == "c530_incomplete_operator"
    }
    complete_incomplete_wins = sum(
        row["prediction"]["nmse"]
        < incomplete_index[(row["shift"], row["seed"])]["prediction"]["nmse"]
        for row in candidate_rows
    )
    top_two_pass_cells = sum(row["top_two_cosine"] >= 0.950 for row in candidate_rows)

    complete_rank, reference_rank, complete_ratio = coverage(p, COMPLETE_ORIGINS)
    incomplete_rank, _, incomplete_ratio = coverage(p, INCOMPLETE_ORIGINS)
    gates = {
        "g1_coverage": complete_rank == reference_rank == p and incomplete_rank < p,
        "g2_prediction_nmse": candidate_nmse <= 0.100,
        "g3_prediction_cosine": mean(candidate, "cosine") >= 0.950,
        "g4_nonoperator_gain": gain >= 0.15,
        "g5_shift_wins": shift_wins >= 4,
        "g6_endpoint_basis": method_summary[candidate]["endpoint_basis_cosine"] >= 0.900,
        "g7_evoo_spearman": abs(mean(candidate, "evoo_spearman")) >= 0.900,
        "g8_top_two_basin": top_two_pass_cells >= 8,
        "g9_complete_over_incomplete": complete_incomplete_wins >= 8,
    }
    primary = [gates["g1_coverage"], gates["g2_prediction_nmse"], gates["g3_prediction_cosine"]]
    secondary_passes = sum(gates[key] for key in list(gates)[3:])
    if all(gates.values()):
        decision = "STRONGLY_SUPPORTIVE"
    elif all(primary) and secondary_passes >= 4:
        decision = "PARTIALLY_SUPPORTIVE"
    else:
        decision = "NOT_SUPPORTIVE"
    return {
        "decision": decision,
        "gates": gates,
        "passed_gates": int(sum(gates.values())),
        "total_gates": len(gates),
        "summary": {
            "candidate_prediction_nmse": candidate_nmse,
            "candidate_prediction_cosine": mean(candidate, "cosine"),
            "candidate_prediction_evoo_spearman": mean(candidate, "evoo_spearman"),
            "candidate_endpoint_basis_cosine": method_summary[candidate][
                "endpoint_basis_cosine"
            ],
            "best_nonoperator": best_nonoperator,
            "best_nonoperator_prediction_nmse": baseline_nmse,
            "relative_gain_over_best_nonoperator": gain,
            "shift_wins": shift_wins,
            "top_two_pass_cells": top_two_pass_cells,
            "complete_over_incomplete_wins": int(complete_incomplete_wins),
            "complete_operator_rank": complete_rank,
            "incomplete_operator_rank": incomplete_rank,
            "reference_rank": reference_rank,
            "complete_coverage_ratio": complete_ratio,
            "incomplete_coverage_ratio": incomplete_ratio,
            "shift_details": shift_details,
            "total_runtime_seconds": total_runtime,
        },
        "method_summary": method_summary,
        "sample_counts_after_deduplication": counts,
        "n_records": len(rows),
    }


def main() -> int:
    started = time.perf_counter()
    spectral_columns, subsets, removed_duplicates = load_data()
    calibration, calibration_raw = subsets["CALIBRATION"]
    validation, validation_raw = subsets["VALIDATION"]
    prediction, prediction_raw = subsets["PREDICTION"]
    p = len(spectral_columns)
    endpoint_rows = []
    for percentage in (0.0, 100.0):
        mask = calibration["_evoo"].to_numpy() == percentage
        endpoint_rows.append(normalize_fine(calibration_raw[mask]).mean(axis=0))
    endpoints = np.vstack(endpoint_rows)
    rows = []
    for shift in SHIFTS:
        for seed in SEEDS_BY_SHIFT[shift]:
            cell_started = time.perf_counter()
            rows.extend(
                run_cell(
                    shift,
                    seed,
                    calibration,
                    calibration_raw,
                    validation_raw,
                    validation["_evoo"].to_numpy(dtype=float),
                    prediction_raw,
                    prediction["_evoo"].to_numpy(dtype=float),
                    endpoints,
                    p,
                )
            )
            print(
                f"completed shift={shift} seed={seed} "
                f"elapsed={time.perf_counter() - cell_started:.2f}s",
                flush=True,
            )
    total_runtime = time.perf_counter() - started
    counts = {
        "raw_total": 605,
        "duplicates_removed": removed_duplicates,
        "calibration": int(len(calibration)),
        "validation": int(len(validation)),
        "prediction": int(len(prediction)),
    }
    report = summarize(rows, p, counts, total_runtime)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "method": row["method"],
                "shift": row["shift"],
                "seed": row["seed"],
                "rank": row["rank"],
                "prediction_nmse": row["prediction"]["nmse"],
                "prediction_cosine": row["prediction"]["cosine"],
                "prediction_evoo_spearman": row["prediction"]["evoo_spearman"],
                "validation_nmse": row["validation"]["nmse"],
                "endpoint_basis_cosine": row["endpoint_basis_cosine"],
                "top_two_cosine": row["top_two_cosine"],
                "runtime_seconds": row["runtime_seconds"],
            }
            for row in rows
        ]
    ).to_csv(OUT / "cell_metrics.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
