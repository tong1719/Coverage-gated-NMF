"""Verify the bounded cyclic/noncyclic coverage propositions for C530."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = HERE / "results_coverage_theory"


def cyclic_partition(p: int, q: int, origin: int) -> np.ndarray:
    operator = np.zeros((p, p // q))
    for j in range(p):
        operator[j, ((j - origin) % p) // q] = 1.0
    return operator


def cyclic_windows(p: int, q: int) -> np.ndarray:
    columns = []
    for start in range(p):
        column = np.zeros(p)
        column[(start + np.arange(q)) % p] = 1.0
        columns.append(column)
    return np.column_stack(columns)


def noncyclic_partition(p: int, q: int, origin: int) -> np.ndarray:
    columns = []
    for start in range(-origin, p, q):
        column = np.zeros(p)
        column[max(0, start) : min(p, start + q)] = 1.0
        if column.sum() > 0:
            columns.append(column)
    return np.column_stack(columns)


def column_multiset(matrix: np.ndarray) -> list[bytes]:
    return sorted(np.ascontiguousarray(matrix[:, j]).tobytes() for j in range(matrix.shape[1]))


def main() -> int:
    records = []
    for p in range(12, 193):
        for q in range(2, min(12, p - 1) + 1):
            if p % q:
                continue
            cyclic = np.concatenate([cyclic_partition(p, q, o) for o in range(q)], axis=1)
            windows = cyclic_windows(p, q)
            noncyclic = np.concatenate(
                [noncyclic_partition(p, q, o) for o in range(q)], axis=1
            )
            cyclic_rank = int(np.linalg.matrix_rank(cyclic))
            noncyclic_rank = int(np.linalg.matrix_rank(noncyclic))
            distinct_cyclic = len({tuple(row) for row in cyclic})
            distinct_noncyclic = len({tuple(row) for row in noncyclic})
            bound_applies = p > q * (q - 1)
            proper_subset_column_bound = (q - 1) * (p // q)
            records.append(
                {
                    "p": p,
                    "q": q,
                    "cyclic_is_all_windows": column_multiset(cyclic)
                    == column_multiset(windows),
                    "cyclic_rank": cyclic_rank,
                    "expected_cyclic_rank": p - q + 1,
                    "cyclic_distinct_line_codes": distinct_cyclic,
                    "proper_subset_bound_applies": bound_applies,
                    "proper_subset_max_columns": proper_subset_column_bound,
                    "proper_subset_below_complete_rank": (
                        not bound_applies
                        or proper_subset_column_bound < p - q + 1
                    ),
                    "noncyclic_rank": noncyclic_rank,
                    "noncyclic_distinct_line_codes": distinct_noncyclic,
                }
            )

    checks = {
        "cyclic_is_all_windows": all(r["cyclic_is_all_windows"] for r in records),
        "cyclic_rank_formula": all(
            r["cyclic_rank"] == r["expected_cyclic_rank"] for r in records
        ),
        "cyclic_line_code_injective": all(
            r["cyclic_distinct_line_codes"] == r["p"] for r in records
        ),
        "proper_subset_column_bound": all(
            r["proper_subset_below_complete_rank"] for r in records
        ),
        "noncyclic_full_rank": all(r["noncyclic_rank"] == r["p"] for r in records),
        "noncyclic_line_code_injective": all(
            r["noncyclic_distinct_line_codes"] == r["p"] for r in records
        ),
    }
    report = {
        "n_cases": len(records),
        "p_range": [12, 192],
        "q_range": [2, 12],
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "enumeration supports the bounded propositions; it is not a general NMF identifiability proof",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
