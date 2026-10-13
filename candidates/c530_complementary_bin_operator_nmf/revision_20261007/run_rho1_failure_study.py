"""Stress-test rho=1 under factor, morphology, count, and rank failures."""
from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_rho1_failure_study"
P = 96
K_TRUE = 4
WIDTH = 4
ORIGINS = (0, 1, 2, 3)
SEEDS = tuple(range(61000, 61005))
N_STARTS = 20
EPS = 1e-12
SCENARIOS = (
    "nominal_sparse",
    "coefficient_deficient",
    "high_component_collinearity",
    "dense_broad",
    "low_count",
    "rank_underfit",
    "rank_overfit",
)


def partition(p: int, width: int, origin: int) -> np.ndarray:
    columns = []
    for start in range(-origin, p, width):
        column = np.zeros(p, dtype=float)
        column[max(0, start) : min(p, start + width)] = 1.0
        if column.sum() > 0:
            columns.append(column)
    return np.column_stack(columns)


def normalize_rows(h: np.ndarray) -> np.ndarray:
    return h / np.maximum(h.mean(axis=1, keepdims=True), EPS)


def sparse_basis() -> np.ndarray:
    positions = (
        (5, 17, 31, 49, 70, 89),
        (9, 22, 38, 56, 75, 92),
        (2, 27, 44, 63, 80, 94),
        (13, 34, 52, 67, 84, 95),
    )
    weights = np.array([1.0, 0.75, 0.55, 0.9, 0.65, 0.8])
    h = np.full((K_TRUE, P), 0.003)
    for component, locations in enumerate(positions):
        for location, weight in zip(locations, weights):
            h[component, location] += weight * (1 + 0.07 * component)
    return normalize_rows(h)


def broad_basis(collinear: bool = False) -> np.ndarray:
    x = np.linspace(0.0, 1.0, P, endpoint=False)
    common = 0.4 + np.exp(-0.5 * ((x - 0.48) / 0.23) ** 2)
    rows = []
    for component in range(K_TRUE):
        if collinear:
            center = 0.30 + 0.13 * component
            row = common + 0.025 * np.exp(-0.5 * ((x - center) / 0.055) ** 2)
        else:
            center1 = 0.20 + 0.16 * component
            center2 = 0.63 + 0.05 * component
            row = (
                0.08
                + np.exp(-0.5 * ((x - center1) / 0.12) ** 2)
                + 0.75 * np.exp(-0.5 * ((x - center2) / 0.14) ** 2)
            )
        rows.append(row)
    return normalize_rows(np.asarray(rows))


def coefficient_block(rng: np.random.Generator, n: int, scale: float, deficient: bool) -> np.ndarray:
    if deficient:
        latent = rng.gamma(1.7, 0.7, (n, 1))
        proportions = np.array([[1.0, 0.75, 1.25, 0.55]])
        w = latent @ proportions
    else:
        w = rng.gamma(1.7, 0.7, (n, K_TRUE))
    w /= np.maximum(w.mean(axis=0, keepdims=True), EPS)
    return w * scale


def scenario_spec(scenario: str) -> tuple[np.ndarray, float, bool, int]:
    if scenario == "high_component_collinearity":
        return broad_basis(collinear=True), 0.50, False, K_TRUE
    if scenario == "dense_broad":
        return broad_basis(collinear=False), 0.50, False, K_TRUE
    if scenario == "coefficient_deficient":
        return sparse_basis(), 0.50, True, K_TRUE
    if scenario == "low_count":
        return sparse_basis(), 0.025, False, K_TRUE
    if scenario == "rank_underfit":
        return sparse_basis(), 0.50, False, 3
    if scenario == "rank_overfit":
        return sparse_basis(), 0.50, False, 5
    return sparse_basis(), 0.50, False, K_TRUE


def simulate(scenario: str, seed: int):
    rng = np.random.default_rng(seed)
    true_h, scale, deficient, k_fit = scenario_spec(scenario)
    operators = [partition(P, WIDTH, origin) for origin in ORIGINS]
    ys, true_ws, fine_rates = [], [], []
    for operator in operators:
        w = coefficient_block(rng, 48, scale, deficient)
        rate = np.maximum(w @ true_h, EPS)
        y = rng.poisson(rate @ operator).astype(float)
        ys.append(y)
        true_ws.append(w)
        fine_rates.append(rate)
    heldout_operator = partition(P, 7, 3)
    heldout_w = coefficient_block(rng, 60, scale, deficient)
    heldout_rate = np.maximum(heldout_w @ true_h, EPS)
    heldout_y = rng.poisson(heldout_rate @ heldout_operator).astype(float)
    return ys, operators, true_ws, fine_rates, true_h, heldout_y, heldout_operator, heldout_rate, k_fit


def objective(ys, operators, ws, h) -> float:
    value = 0.0
    for y, operator, w in zip(ys, operators, ws):
        pred = np.maximum(w @ h @ operator, EPS)
        value += float(np.sum(pred - y * np.log(pred)))
    return value


def fit_operator(ys, operators, k_fit: int, seed: int):
    rng = np.random.default_rng(seed)
    h = np.maximum(rng.gamma(1.0, 1.0, (k_fit, P)), EPS)
    ws = [np.maximum(rng.gamma(1.0, 1.0, (len(y), k_fit)), EPS) for y in ys]
    trace = []
    stable_checks = 0
    max_relative_increase = 0.0
    previous = None
    for iteration in range(1, 2001):
        for block, (y, operator) in enumerate(zip(ys, operators)):
            basis = h @ operator
            pred = np.maximum(ws[block] @ basis, EPS)
            numerator = (y / pred) @ basis.T
            denominator = np.ones_like(y) @ basis.T
            ws[block] *= numerator / np.maximum(denominator, EPS)
            ws[block] = np.maximum(ws[block], EPS)
        numerator_h = np.zeros_like(h)
        denominator_h = np.zeros_like(h)
        for y, operator, w in zip(ys, operators, ws):
            pred = np.maximum(w @ h @ operator, EPS)
            numerator_h += w.T @ (y / pred) @ operator.T
            denominator_h += w.T @ np.ones_like(y) @ operator.T
        h *= numerator_h / np.maximum(denominator_h, EPS)
        h = np.maximum(h, EPS)
        if iteration % 25 == 0:
            scale = np.maximum(h.mean(axis=1), EPS)
            h /= scale[:, None]
            ws = [w * scale[None, :] for w in ws]
        if iteration % 10 == 0:
            current = objective(ys, operators, ws, h)
            trace.append(current)
            if previous is not None:
                relative_change = abs(previous - current) / max(abs(previous), EPS)
                relative_increase = (current - previous) / max(abs(previous), EPS)
                max_relative_increase = max(max_relative_increase, relative_increase)
                if iteration >= 100 and relative_change < 1e-7:
                    stable_checks += 1
                else:
                    stable_checks = 0
            previous = current
            if stable_checks >= 5:
                break
    return ws, h, np.asarray(trace), iteration, max_relative_increase


def infer_coefficients(y: np.ndarray, basis: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(9921)
    w = np.maximum(rng.gamma(1.0, 1.0, (len(y), len(basis))), EPS)
    projected = basis
    for _ in range(1500):
        pred = np.maximum(w @ projected, EPS)
        w *= ((y / pred) @ projected.T) / np.maximum(np.ones_like(y) @ projected.T, EPS)
        w = np.maximum(w, EPS)
    return w


def normalized_rows(h: np.ndarray) -> np.ndarray:
    return h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), EPS)


def basis_metrics(h: np.ndarray, true_h: np.ndarray) -> tuple[float, float, float]:
    fitted = normalized_rows(h)
    truth = normalized_rows(true_h)
    similarities = fitted @ truth.T
    rows, cols = linear_sum_assignment(-similarities)
    matched = float(np.mean(similarities[rows, cols]))
    recall = float(np.mean(np.max(similarities, axis=0)))
    precision = float(np.mean(np.max(similarities, axis=1)))
    return matched, recall, precision


def solution_similarity(h_a: np.ndarray, h_b: np.ndarray) -> float:
    a = normalized_rows(h_a)
    b = normalized_rows(h_b)
    similarities = a @ b.T
    rows, cols = linear_sum_assignment(-similarities)
    return float(np.mean(similarities[rows, cols]))


def run_cell(scenario: str, seed: int) -> dict:
    ys, operators, true_ws, fine_rates, true_h, held_y, held_operator, held_rate, k_fit = simulate(scenario, seed)
    started = time.perf_counter()
    fits = []
    for start in range(N_STARTS):
        ws, h, trace, iterations, max_increase = fit_operator(ys, operators, k_fit, seed + 1000 + start)
        loss = float(trace[-1])
        matched, recall, precision = basis_metrics(h, true_h)
        held_w = infer_coefficients(held_y, h @ held_operator)
        held_estimate = held_w @ h
        held_nmse = float(np.sum((held_estimate - held_rate) ** 2) / np.sum(held_rate**2))
        fits.append(
            {
                "start": start,
                "loss": loss,
                "matched_basis_cosine": matched,
                "true_basis_recall": recall,
                "fitted_basis_precision": precision,
                "heldout_nmse": held_nmse,
                "iterations": iterations,
                "max_relative_objective_increase": max_increase,
                "h": h,
                "trace": trace,
            }
        )
    order = np.argsort([fit["loss"] for fit in fits])
    best = fits[int(order[0])]
    gaps = np.array([(fit["loss"] - best["loss"]) / max(abs(best["loss"]), EPS) for fit in fits])
    near = [index for index, gap in enumerate(gaps) if gap <= 1e-4]
    all_similarities = [solution_similarity(best["h"], fit["h"]) for fit in fits]
    near_similarities = [all_similarities[index] for index in near]
    complete = np.concatenate(operators, axis=1)
    rho = float(np.linalg.matrix_rank(complete) / np.linalg.matrix_rank(complete))
    coefficient_rank = int(np.linalg.matrix_rank(np.vstack(true_ws)))
    truth_norm = normalized_rows(true_h)
    truth_gram = truth_norm @ truth_norm.T
    np.fill_diagonal(truth_gram, -np.inf)
    result = {
        "scenario": scenario,
        "seed": seed,
        "k_true": K_TRUE,
        "k_fit": k_fit,
        "rho": rho,
        "operator_rank": int(np.linalg.matrix_rank(complete)),
        "coefficient_rank": coefficient_rank,
        "true_basis_max_offdiag_cosine": float(np.max(truth_gram)),
        "best_start": int(best["start"]),
        "best_loss": float(best["loss"]),
        "best_matched_basis_cosine": float(best["matched_basis_cosine"]),
        "best_true_basis_recall": float(best["true_basis_recall"]),
        "best_fitted_basis_precision": float(best["fitted_basis_precision"]),
        "best_heldout_nmse": float(best["heldout_nmse"]),
        "near_optimal_count": len(near),
        "near_optimal_mean_similarity": float(np.mean(near_similarities)),
        "near_optimal_min_similarity": float(np.min(near_similarities)),
        "all_start_mean_similarity": float(np.mean(all_similarities)),
        "all_start_min_similarity": float(np.min(all_similarities)),
        "median_iterations": float(np.median([fit["iterations"] for fit in fits])),
        "max_relative_objective_increase": float(max(fit["max_relative_objective_increase"] for fit in fits)),
        "relative_objective_gaps": gaps.tolist(),
        "start_metrics": [
            {key: value for key, value in fit.items() if key not in {"h", "trace"}}
            | {"relative_objective_gap": float(gaps[index]), "objective_trace": fit["trace"].tolist()}
            for index, fit in enumerate(fits)
        ],
        "runtime_seconds": time.perf_counter() - started,
    }
    return result


def summary(records: list[dict]) -> dict:
    metrics = (
        "best_matched_basis_cosine",
        "best_true_basis_recall",
        "best_fitted_basis_precision",
        "best_heldout_nmse",
        "near_optimal_mean_similarity",
        "near_optimal_min_similarity",
        "all_start_mean_similarity",
        "all_start_min_similarity",
        "near_optimal_count",
        "median_iterations",
    )
    by_scenario = {}
    for scenario in SCENARIOS:
        cells = [record for record in records if record["scenario"] == scenario]
        by_scenario[scenario] = {
            metric: {
                "median": float(np.median([cell[metric] for cell in cells])),
                "q25": float(np.quantile([cell[metric] for cell in cells], 0.25)),
                "q75": float(np.quantile([cell[metric] for cell in cells], 0.75)),
                "min": float(np.min([cell[metric] for cell in cells])),
                "max": float(np.max([cell[metric] for cell in cells])),
            }
            for metric in metrics
        }
        med = by_scenario[scenario]
        by_scenario[scenario]["empirical_failure"] = bool(
            med["best_matched_basis_cosine"]["median"] < 0.90
            or med["best_heldout_nmse"]["median"] > 0.15
            or med["near_optimal_min_similarity"]["median"] < 0.95
        )
    nominal = by_scenario["nominal_sparse"]
    stress_failures = sum(by_scenario[scenario]["empirical_failure"] for scenario in SCENARIOS[1:])
    checks = {
        "all_rho_one": all(record["rho"] == 1.0 for record in records),
        "all_operator_rank_96": all(record["operator_rank"] == P for record in records),
        "nominal_basis": nominal["best_matched_basis_cosine"]["median"] >= 0.95,
        "nominal_heldout": nominal["best_heldout_nmse"]["median"] <= 0.10,
        "at_least_three_stress_failures": stress_failures >= 3,
        "objective_monotonic": all(record["max_relative_objective_increase"] <= 1e-8 for record in records),
        "finite": all(
            np.isfinite(record[key])
            for record in records
            for key in (
                "best_loss",
                "best_matched_basis_cosine",
                "best_heldout_nmse",
                "near_optimal_mean_similarity",
            )
        ),
        "record_count": len(records) == len(SCENARIOS) * len(SEEDS),
    }
    return {"by_scenario": by_scenario, "stress_failure_count": stress_failures, "checks": checks, "passed": all(checks.values())}


def main() -> int:
    records = [run_cell(scenario, seed) for scenario, seed in itertools.product(SCENARIOS, SEEDS)]
    report = {
        "protocol": "PROTOCOL_RHO1_FAILURE_STUDY.md",
        "scenarios": list(SCENARIOS),
        "seeds": list(SEEDS),
        "n_starts": N_STARTS,
        "summary": summary(records),
        "records": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for record in records:
        rows.append({key: value for key, value in record.items() if key not in {"relative_objective_gaps", "start_metrics"}})
    with (OUT / "cell_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    compact = {
        scenario: {
            "basis": values["best_matched_basis_cosine"]["median"],
            "heldout_nmse": values["best_heldout_nmse"]["median"],
            "near_min_similarity": values["near_optimal_min_similarity"]["median"],
            "failure": values["empirical_failure"],
        }
        for scenario, values in report["summary"]["by_scenario"].items()
    }
    print(json.dumps({"passed": report["summary"]["passed"], "checks": report["summary"]["checks"], "scenarios": compact}, indent=2))
    return 0 if report["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
