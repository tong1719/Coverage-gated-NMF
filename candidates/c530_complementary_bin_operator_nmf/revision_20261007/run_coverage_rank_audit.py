"""Revision audit for operator rank, conditioning, and rho=1 counterexamples."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_coverage_rank_audit"
EPS = np.finfo(float).eps
RELATIVE_TOLS = (1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


def cyclic_partition(p: int, q: int, origin: int) -> np.ndarray:
    if p % q:
        raise ValueError("cyclic audit requires p divisible by q")
    operator = np.zeros((p, p // q), dtype=float)
    for channel in range(p):
        operator[channel, ((channel - origin) % p) // q] = 1.0
    return operator


def noncyclic_partition(p: int, q: int, origin: int) -> np.ndarray:
    columns = []
    for start in range(-origin, p, q):
        column = np.zeros(p, dtype=float)
        column[max(0, start) : min(p, start + q)] = 1.0
        if column.sum() > 0:
            columns.append(column)
    return np.column_stack(columns)


def layout(p: int, q: int, origins: tuple[int, ...], boundary: str) -> np.ndarray:
    fn = cyclic_partition if boundary == "cyclic" else noncyclic_partition
    return np.concatenate([fn(p, q, origin) for origin in origins], axis=1)


def row_geometry(matrix: np.ndarray) -> tuple[int, float]:
    rounded = np.round(matrix, 14)
    distinct = int(np.unique(rounded, axis=0).shape[0])
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > 0
    normalized = matrix[keep] / norms[keep, None]
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    coherence = float(np.max(gram)) if len(gram) > 1 else 0.0
    return distinct, coherence


def rank_record(
    p: int,
    q: int,
    origins: tuple[int, ...],
    boundary: str,
    label: str,
) -> dict:
    matrix = layout(p, q, origins, boundary)
    complete = layout(p, q, tuple(range(q)), boundary)
    singular = np.linalg.svd(matrix, compute_uv=False)
    complete_singular = np.linalg.svd(complete, compute_uv=False)
    default_rank = int(np.linalg.matrix_rank(matrix))
    complete_default_rank = int(np.linalg.matrix_rank(complete))
    explicit_tol = float(singular[0] * max(matrix.shape) * EPS)
    complete_tol = float(complete_singular[0] * max(complete.shape) * EPS)
    explicit_rank = int(np.sum(singular > explicit_tol))
    complete_explicit_rank = int(np.sum(complete_singular > complete_tol))
    smallest_retained = float(singular[explicit_rank - 1]) if explicit_rank else 0.0
    condition_index = float(singular[0] / smallest_retained) if smallest_retained else float("inf")
    distinct_rows, row_coherence = row_geometry(matrix)
    tolerance_ranks = {
        f"{relative:.0e}": int(np.sum(singular > relative * singular[0]))
        for relative in RELATIVE_TOLS
    }
    return {
        "label": label,
        "boundary": boundary,
        "p": p,
        "q": q,
        "origins": list(origins),
        "shape": list(matrix.shape),
        "default_rank": default_rank,
        "explicit_machine_rank": explicit_rank,
        "complete_default_rank": complete_default_rank,
        "complete_explicit_machine_rank": complete_explicit_rank,
        "relative_rank": float(explicit_rank / complete_explicit_rank),
        "explicit_tolerance": explicit_tol,
        "sigma_max": float(singular[0]),
        "smallest_retained_sigma": smallest_retained,
        "retained_condition_index": condition_index,
        "distinct_row_codes": distinct_rows,
        "max_distinct_row_cosine": row_coherence,
        "relative_tolerance_ranks": tolerance_ranks,
        "default_matches_explicit": default_rank == explicit_rank,
    }


def cyclic_dense_counterexample() -> dict:
    p, q = 96, 4
    matrix = layout(p, q, tuple(range(q)), "cyclic")
    u, singular, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = int(np.linalg.matrix_rank(matrix))
    null_vector = u[:, rank]
    null_vector /= np.max(np.abs(null_vector))
    spectrum_a = 2.0 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, p, endpoint=False))
    max_step = np.min(spectrum_a[np.abs(null_vector) > 1e-14] / np.abs(null_vector[np.abs(null_vector) > 1e-14]))
    step = 0.45 * max_step
    spectrum_b = spectrum_a + step * null_vector
    forward_error = float(np.max(np.abs(spectrum_a @ matrix - spectrum_b @ matrix)))
    return {
        "p": p,
        "q": q,
        "rank": rank,
        "nullity": int(p - rank),
        "rho": 1.0,
        "minimum_spectrum_a": float(spectrum_a.min()),
        "minimum_spectrum_b": float(spectrum_b.min()),
        "spectrum_l2_difference": float(np.linalg.norm(spectrum_a - spectrum_b)),
        "max_forward_difference": forward_error,
        "passed": bool(
            spectrum_b.min() > 1e-10
            and np.linalg.norm(spectrum_a - spectrum_b) > 1e-6
            and forward_error < 1e-10
        ),
    }


def nmf_nonuniqueness_counterexample() -> dict:
    w = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.2], [0.8, 1.7]])
    grid = np.linspace(0.0, 1.0, 24)
    h = np.vstack([2.0 + 0.3 * np.cos(2 * np.pi * grid), 0.8 + 0.2 * np.sin(2 * np.pi * grid)])
    alpha = 0.25
    qmat = np.array([[1.0, alpha], [0.0, 1.0]])
    qinv = np.linalg.inv(qmat)
    w_alt = w @ qmat
    h_alt = qinv @ h
    product_error = float(np.max(np.abs(w @ h - w_alt @ h_alt)))
    return {
        "minimum_w": float(w.min()),
        "minimum_h": float(h.min()),
        "minimum_w_alt": float(w_alt.min()),
        "minimum_h_alt": float(h_alt.min()),
        "w_l2_difference": float(np.linalg.norm(w - w_alt)),
        "h_l2_difference": float(np.linalg.norm(h - h_alt)),
        "max_product_difference": product_error,
        "passed": bool(
            w_alt.min() > 1e-10
            and h_alt.min() > 1e-10
            and np.linalg.norm(h - h_alt) > 1e-6
            and product_error < 1e-10
        ),
    }


def main() -> int:
    designs: list[tuple[int, int, tuple[int, ...], str, str]] = []
    for q in (2, 3, 4, 6):
        designs.append((96, q, tuple(range(q)), "cyclic", f"cyclic_q{q}_complete"))
        designs.append((96, q, tuple([0] * q), "cyclic", f"cyclic_q{q}_repeated"))
        reduced = tuple((index % max(1, q // 2)) for index in range(q))
        designs.append((96, q, reduced, "cyclic", f"cyclic_q{q}_reduced"))
    for q in (3, 4, 6):
        designs.append((96, q, tuple(range(q)), "noncyclic", f"noncyclic_q{q}_complete"))
        designs.append((96, q, tuple([0] * q), "noncyclic", f"noncyclic_q{q}_repeated"))
        reduced = tuple((index % max(1, q // 2)) for index in range(q))
        designs.append((96, q, reduced, "noncyclic", f"noncyclic_q{q}_reduced"))
    designs.extend(
        [
            (710, 8, tuple(range(8)), "noncyclic", "raman_noncyclic_complete"),
            (710, 8, (0, 0, 1, 1, 2, 2, 3, 3), "noncyclic", "raman_noncyclic_incomplete"),
        ]
    )

    records = [rank_record(*design) for design in designs]
    dense = cyclic_dense_counterexample()
    nmf = nmf_nonuniqueness_counterexample()
    raman_complete = next(r for r in records if r["label"] == "raman_noncyclic_complete")
    raman_incomplete = next(r for r in records if r["label"] == "raman_noncyclic_incomplete")
    checks = {
        "default_matches_explicit_all": all(r["default_matches_explicit"] for r in records),
        "raman_complete_rank_710": raman_complete["explicit_machine_rank"] == 710,
        "raman_complete_rho_one": raman_complete["relative_rank"] == 1.0,
        "raman_incomplete_rank_354": raman_incomplete["explicit_machine_rank"] == 354,
        "raman_incomplete_below_one": raman_incomplete["relative_rank"] < 1.0,
        "dense_counterexample": dense["passed"],
        "nmf_counterexample": nmf["passed"],
        "finite_records": all(
            np.isfinite(r[key])
            for r in records
            for key in (
                "relative_rank",
                "explicit_tolerance",
                "sigma_max",
                "smallest_retained_sigma",
                "retained_condition_index",
                "max_distinct_row_cosine",
            )
        ),
    }
    report = {
        "protocol": "PROTOCOL_COVERAGE_RANK_AUDIT.md",
        "numpy_version": np.__version__,
        "machine_epsilon": EPS,
        "n_designs": len(records),
        "checks": checks,
        "passed": all(checks.values()),
        "dense_spectrum_counterexample": dense,
        "nmf_nonuniqueness_counterexample": nmf,
        "records": records,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    flat_rows = []
    for record in records:
        flat = {key: value for key, value in record.items() if key != "relative_tolerance_ranks"}
        flat["origins"] = json.dumps(flat["origins"])
        flat["shape"] = json.dumps(flat["shape"])
        flat.update({f"rank_at_{key}": value for key, value in record["relative_tolerance_ranks"].items()})
        flat_rows.append(flat)
    with (OUT / "operator_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"checks": checks, "passed": report["passed"], "counterexamples": {"dense": dense, "nmf": nmf}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
