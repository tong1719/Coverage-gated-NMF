"""Frozen-record provenance and four-annotation supplementary MassBank panel."""
from __future__ import annotations

import base64
import csv
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PHASE = HERE.parent / "phase_b"
sys.path.insert(0, str(PHASE))
import run_massbank_public as mb  # noqa: E402

OUT = HERE / "results_massbank_distinct_panel_audit"
SELECTED = ("AC000020", "AC000040", "AC000045", "AC000055")


def source_metadata(path, record):
    frozen = json.loads(path.read_text(encoding="utf-8"))
    content = base64.b64decode(frozen["content"]).decode("utf-8")
    def field(pattern):
        match = re.search(pattern, content, re.MULTILINE)
        return match.group(1).strip() if match else ""
    mode = field(r"AC\$MASS_SPECTROMETRY: ION_MODE (.+)")
    ms_type = field(r"AC\$MASS_SPECTROMETRY: MS_TYPE (.+)")
    reasons = []
    if "DEPRECATED:" in content:
        reasons.append("deprecated")
    if mode != "POSITIVE":
        reasons.append("not_positive")
    if ms_type != "MS2":
        reasons.append("not_MS2")
    if record["occupied"] < 5:
        reasons.append("fewer_than_five_occupied_channels")
    return {
        "source_file": path.name,
        "accession": record["accession"],
        "title": record["title"],
        "sha": record["sha"],
        "license": record["license"],
        "ion_mode": mode,
        "ms_type": ms_type,
        "deprecated": "DEPRECATED:" in content,
        "occupied_channels": record["occupied"],
        "eligible": record["eligible"],
        "exclusion_reasons": ";".join(reasons),
    }


def pairwise_cosines(panel):
    h = np.vstack([row["vector"] for row in panel])
    h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), mb.EPS)
    values = h @ h.T
    return {
        f"{panel[i]['accession'].split('-')[-1]}__{panel[j]['accession'].split('-')[-1]}":
            float(values[i, j])
        for i in range(len(panel)) for j in range(i + 1, len(panel))
    }


def main():
    original_panels, _ = mb.panels_and_provenance()
    parsed = [
        (path, mb.parse_record(path))
        for path in sorted(mb.DATA.glob("*.json"))
    ]
    source_rows = [source_metadata(path, record) for path, record in parsed]
    lookup = {record["accession"].split("-")[-1]: record for _, record in parsed}
    panel = [lookup[name] for name in SELECTED]
    assert all(row["eligible"] for row in panel)
    assert len({row["title"].split(";")[0] for row in panel}) == 4
    cosines = pairwise_cosines(panel)
    assert max(cosines.values()) < 0.98
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "all_source_records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    cell_rows = []
    for depth, seed in itertools.product(mb.DEPTHS, mb.SEEDS):
        rows = mb.run_cell(2, panel, depth, seed)
        cell_rows.extend(rows)
        (OUT / f"panel_distinct_{depth}_{seed}.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        print(f"completed {depth} seed={seed}", flush=True)
    methods = list(mb.METHODS)
    comparison = []
    for depth, seed in itertools.product(mb.DEPTHS, mb.SEEDS):
        subset = [r for r in cell_rows if r["depth"] == depth and r["seed"] == seed]
        operator = next(r for r in subset if r["method"] == "c530_certified_operator")
        backprojection = next(r for r in subset if r["method"] == "uniform_backproject")
        comparison.append({
            "depth": depth,
            "seed": seed,
            "operator_basis_cosine": operator["basis_cosine"],
            "operator_heldout_nmse": operator["heldout_nmse"],
            "backprojection_heldout_nmse": backprojection["heldout_nmse"],
            "operator_wins_vs_backprojection": operator["heldout_nmse"] < backprojection["heldout_nmse"],
        })
    with (OUT / "cell_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    report = {
        "protocol": "PROTOCOL_MASSBANK_PROVENANCE_AND_DISTINCT_PANEL.md",
        "original_panel_accessions": [
            [row["accession"] for row in group] for group in original_panels
        ],
        "supplementary_panel_accessions": [row["accession"] for row in panel],
        "source_record_count": len(source_rows),
        "eligible_count": sum(row["eligible"] for row in source_rows),
        "excluded": [row for row in source_rows if not row["eligible"]],
        "unused_eligible_by_original_rule": [
            row["accession"] for row in source_rows
            if row["eligible"] and not any(
                row["accession"] == member["accession"]
                for group in original_panels for member in group
            )
        ],
        "supplementary_panel_pairwise_cosine": cosines,
        "supplementary_panel_max_pairwise_cosine": max(cosines.values()),
        "n_cells": len(comparison),
        "operator_median_basis_cosine": float(np.median([
            row["operator_basis_cosine"] for row in comparison
        ])),
        "operator_median_heldout_nmse": float(np.median([
            row["operator_heldout_nmse"] for row in comparison
        ])),
        "backprojection_median_heldout_nmse": float(np.median([
            row["backprojection_heldout_nmse"] for row in comparison
        ])),
        "operator_wins_vs_backprojection": sum(
            row["operator_wins_vs_backprojection"] for row in comparison
        ),
        "checks": {
            "source_count": len(source_rows) == 12,
            "eligible_count": sum(row["eligible"] for row in source_rows) == 9,
            "cell_count": len(comparison) == 8,
            "method_rows": len(cell_rows) == 8 * len(methods),
            "all_finite": all(
                np.isfinite(r[field])
                for r in cell_rows for field in ("basis_cosine", "heldout_nmse")
            ),
        },
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
