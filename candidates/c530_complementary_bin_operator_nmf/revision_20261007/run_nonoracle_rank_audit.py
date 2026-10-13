"""Coarse-bin cross-validation for the submitted rank-sensitivity cells."""
from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
SOURCE = PARENT / "phase_b" / "run_rank_sensitivity.py"
OUT = HERE / "results_nonoracle_rank_audit"


def load_source():
    spec = importlib.util.spec_from_file_location("c530_rank_source_for_revision", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def poisson_deviance(y: np.ndarray, m: np.ndarray) -> float:
    m = np.maximum(m, 1e-12)
    terms = m - y
    positive = y > 0
    terms[positive] += y[positive] * np.log(y[positive] / m[positive])
    return float(np.sum(terms))


def split_prediction_score(source, y, a, h) -> float:
    scores = []
    for first in (0, 1):
        train = np.arange(y.shape[1]) % 2 == first
        test = ~train
        w = source.fixed_w(
            y[:, train], h @ a[:, train],
            np.ones((len(y), len(h))), iterations=400,
        )
        prediction = w @ h @ a[:, test]
        scores.append(poisson_deviance(y[:, test], prediction) / y[:, test].size)
    return float(np.mean(scores))


def main():
    source = load_source()
    rows = []
    for regime in source.REGIMES:
        for seed in source.SEEDS:
            ys, mats, _, _, _, yt, at, _, rt = source.c529.simulate(regime, seed)
            for rank in source.RANKS:
                (_, h), training_score = source.fit_multistart(
                    ys, mats, rank, seed + 100 * rank
                )
                rows.append({
                    "regime": regime,
                    "seed": int(seed),
                    "rank": int(rank),
                    "coarse_validation_deviance_per_cell": split_prediction_score(
                        source, yt, at, h
                    ),
                    "fine_grid_nmse_audit_only": source.heldout(h, yt, at, rt),
                    "training_score": training_score,
                })

    choices = []
    by_case = defaultdict(list)
    for row in rows:
        by_case[(row["regime"], row["seed"])].append(row)
    for (regime, seed), candidates in sorted(by_case.items()):
        choice = min(
            candidates,
            key=lambda r: (
                round(r["coarse_validation_deviance_per_cell"], 10),
                r["rank"],
            ),
        )
        true_rank = next(r for r in candidates if r["rank"] == 4)
        rank5 = next(r for r in candidates if r["rank"] == 5)
        choices.append({
            "regime": regime,
            "seed": seed,
            "selected_rank": choice["rank"],
            "selected_coarse_validation_deviance_per_cell": choice[
                "coarse_validation_deviance_per_cell"
            ],
            "rank5_minus_rank4_validation_deviance_per_cell":
                rank5["coarse_validation_deviance_per_cell"]
                - true_rank["coarse_validation_deviance_per_cell"],
            "selected_fine_grid_nmse_audit_only": choice[
                "fine_grid_nmse_audit_only"
            ],
            "rank4_fine_grid_nmse_audit_only": true_rank[
                "fine_grid_nmse_audit_only"
            ],
        })

    report = {
        "source": str(SOURCE),
        "protocol": {
            "ranks": list(source.RANKS),
            "regimes": list(source.REGIMES),
            "seeds": list(source.SEEDS),
            "validation": "two-way alternating held-out coarse-bin deviance",
            "tie_rule": "rounded 1e-10 then smaller rank",
        },
        "selection_counts": dict(sorted(Counter(
            str(c["selected_rank"]) for c in choices
        ).items())),
        "selection_by_regime": {
            regime: dict(sorted(Counter(
                str(c["selected_rank"])
                for c in choices if c["regime"] == regime
            ).items()))
            for regime in source.REGIMES
        },
        "rank4_selected": sum(c["selected_rank"] == 4 for c in choices),
        "n_cases": len(choices),
        "median_selected_fine_grid_nmse_audit_only": float(np.median([
            c["selected_fine_grid_nmse_audit_only"] for c in choices
        ])),
        "median_rank4_fine_grid_nmse_audit_only": float(np.median([
            c["rank4_fine_grid_nmse_audit_only"] for c in choices
        ])),
        "fine_grid_nmse_audit_only_worse_than_rank4_cases": sum(
            c["selected_fine_grid_nmse_audit_only"]
            > c["rank4_fine_grid_nmse_audit_only"] + 1e-12
            for c in choices
        ),
        "all_finite": all(
            np.isfinite(value)
            for row in rows for value in row.values()
            if isinstance(value, (float, int))
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, records in (("candidate_scores.csv", rows), ("selected_ranks.csv", choices)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
