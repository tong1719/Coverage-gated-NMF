"""Run the published experiment scripts in dependency order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = Path("candidates/c530_complementary_bin_operator_nmf")
CORE = [
    Path("screening/c530_confirmatory_line_operator_nmf/run_confirmatory_screen.py"),
    BASE / "phase_b/run_robustness.py",
    BASE / "phase_b/run_coverage_certificate.py",
    BASE / "phase_b/run_acquisition_factorial.py",
    BASE / "phase_b/run_massbank_public.py",
    BASE / "phase_b/run_optimization_certificate.py",
    BASE / "phase_b/run_rank_sensitivity.py",
    BASE / "phase_b/run_mcr_als_baseline.py",
    BASE / "phase_b/verify_coverage_theory.py",
    BASE / "phase_b/run_noncyclic_edge_validation.py",
    BASE / "phase_b/audit_real_raman_oil_data.py",
    BASE / "phase_b/run_real_raman_oil_validation.py",
    BASE / "phase_b/verify_real_raman_oil_results.py",
]
REVISION = [
    (BASE / "revision_20261007/run_coverage_rank_audit.py",),
    (BASE / "revision_20261007/run_rho1_failure_study.py",),
    (BASE / "revision_20261007/run_operator_aware_frobenius.py",),
    (BASE / "revision_20261007/run_operator_aware_frobenius.py", "--max-iter", "5000", "--label", "extension_5000"),
    (BASE / "revision_20261007/run_operator_coupled_hals.py",),
    (BASE / "revision_20261007/run_poisson_multistart_audit.py",),
    (BASE / "revision_20261007/run_full_poisson_stopping_audit.py",),
    (BASE / "revision_20261007/run_model_mismatch_stress.py",),
    (BASE / "revision_20261007/run_model_mismatch_stress.py", "--poisson-max-iter", "5000", "--label", "extension_5000"),
    (BASE / "revision_20261007/run_massbank_distinct_panel_audit.py",),
    (BASE / "revision_20261007/run_raman_loss_iteration_sensitivity.py",),
    (BASE / "revision_20261007/run_raman_mcr_heldout_loss_match.py",),
    (BASE / "revision_20261007/run_raman_baseline_correction_sensitivity.py",),
    (BASE / "revision_20261007/run_raman_rank_sensitivity.py",),
    (BASE / "revision_20261007/run_nonoracle_rank_audit.py",),
    (BASE / "revision_20261007/run_paired_uncertainty_audit.py",),
    (BASE / "revision_20261007/run_raman_long_run_check.py",),
    (BASE / "revision_20261007/run_raman_repeat_summary.py",),
    (BASE / "revision_20261007/run_raman_mean_variance_audit.py",),
    (BASE / "revision_20261007/run_normalization_gauge_audit.py",),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("core", "revision", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = []
    if args.group in ("core", "all"):
        commands.extend((path,) for path in CORE)
    if args.group in ("revision", "all"):
        commands.extend(REVISION)
    for index, (path, *options) in enumerate(commands, start=1):
        script = ROOT / path
        if not script.is_file():
            raise FileNotFoundError(script)
        command = [sys.executable, str(script), *options]
        print(f"[{index}/{len(commands)}] {path} {' '.join(options)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
