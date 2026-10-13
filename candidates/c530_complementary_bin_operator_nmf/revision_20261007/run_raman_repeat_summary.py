"""Summarize saved Raman repeated-allocation outcomes for R1.m8/m17."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "phase_b" / "results_real_raman_oil"
OUT = HERE / "results_raman_repeat_summary"
METHODS = ("c530_complete_operator", "c530_complete_operator_rank3_sensitivity")


def describe(values):
    x = np.asarray(list(values), dtype=float)
    return {"mean": float(x.mean()), "median": float(np.median(x)),
            "q25": float(np.quantile(x, 0.25)), "q75": float(np.quantile(x, 0.75)),
            "min": float(x.min()), "max": float(x.max())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((SOURCE / "records.json").read_text(encoding="utf-8"))
    original = json.loads((SOURCE / "report.json").read_text(encoding="utf-8"))
    rows = []
    summary = {}
    for method in METHODS:
        selected = sorted((row for row in records if row["method"] == method), key=lambda row: (row["shift"], row["seed"]))
        assert len(selected) == 10 and {(r["shift"], r["seed"]) for r in selected} == {(shift, 53800+100*shift+j) for shift in range(5) for j in range(2)}
        for record in selected:
            rows.append({"method": method, "shift": record["shift"], "seed": record["seed"],
                         "rank": record["rank"], "prediction_evoo_spearman": record["prediction"]["evoo_spearman"],
                         "prediction_nmse": record["prediction"]["nmse"],
                         "endpoint_basis_cosine": record["endpoint_basis_cosine"]})
        summary[method] = {
            "prediction_evoo_spearman": describe(r["prediction"]["evoo_spearman"] for r in selected),
            "prediction_nmse": describe(r["prediction"]["nmse"] for r in selected),
            "endpoint_basis_cosine": describe(r["endpoint_basis_cosine"] for r in selected),
        }
        for key, report_key in (("prediction_evoo_spearman", "prediction_evoo_spearman"), ("prediction_nmse", "prediction_nmse")):
            target = original["method_summary"][method][report_key]
            assert abs(summary[method][key]["mean"] - target) < 1e-12, (method, key)
    with (OUT / "cell_values.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"source": str(SOURCE / "records.json"), "n_cells_per_method": 10,
              "interpretation": "descriptive_repeated_allocation_and_seed_spread_not_laboratory_CI",
              "summary": summary}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
