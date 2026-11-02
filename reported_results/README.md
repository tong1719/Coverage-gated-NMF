# Numerical results for the manuscript

These files contain numerical outcomes, not third-party source spectra or generated figures. The experiment scripts and data-preparation instructions are in the parent repository. Script paths in the table below are relative to `candidates/c530_complementary_bin_operator_nmf/`. The files correspond to the version of the manuscript titled *Operator-coupled Poisson NMF with acquisition-coverage diagnostics for unpaired binned spectra*.

| File | Manuscript use | Producing script |
|---|---|---|
| \`acquisition_factorial_cells.json\` | equal-budget origin comparison | \`phase_b/run_acquisition_factorial.py\` |
| \`massbank_cells.json\` | semi-synthetic MS/MS comparisons | \`phase_b/run_massbank_public.py\` |
| \`raman_cells.csv\` | retrospectively binned Raman comparisons | \`phase_b/run_real_raman_oil_validation.py\` |
| \`operator_loss_paired_cells.csv\` | paired Poisson/HALS effects and intervals | \`revision_20261007/run_paired_uncertainty_audit.py\` |
| \`multistart_cases.json\` | Appendix A: four representative 50-start cases | \`revision_20261007/run_poisson_multistart_audit.py\` |
| \`full_rank_difficult_cells.csv\` | Appendix B: seven full-row-rank regimes | \`revision_20261007/run_rho1_failure_study.py\` |
| \`model_mismatch_cells.csv\` | independent-block mismatch comparison | \`revision_20261007/run_model_mismatch_stress.py\` |
| \`raman_baseline_cells.csv\` | Raman preprocessing sensitivity | \`revision_20261007/run_raman_baseline_correction_sensitivity.py\` |

The full-row-rank study used a 96-channel fine grid, noncyclic width-4 bins at origins 0, 1, 2 and 3, four training blocks of 48 mixtures, and 60 held-out mixtures aggregated with width 7, origin 3. The true rank was four. Five seeds (61000–61004) and 20 starts per cell were used. The nominal sparse setting had count scale 0.5. Other settings changed coefficient rank, spectral shape, count scale or fitted rank as identified in \`full_rank_difficult_cells.csv\` and the source script.

The mismatch study used four training blocks of 48 rank-4 mixtures on 96 channels and a held-out block of 52 mixtures. Training used noncyclic width-4 origins 0–3; the held-out operator had width 8, origin 3. Its seven independently generated scenarios and four seeds (63000–63003) are identified in \`model_mismatch_cells.csv\`. The tabulated Poisson outcomes are from continuation to at most 5000 iterations with the recorded stopping rule. The corresponding script specifies the spectral shifts, broadening, boundary changes, background and noise model. The CSV was assembled from the 28 per-cell JSON outputs without recomputing fits.

The multistart file records both the 420-iteration checkpoints and the extended fits, including objective-gap tolerances of $10^{-4}$ and $10^{-3}$. A singleton near-optimal set has no pairwise similarity value. The Raman rows are retrospective allocations of one measured dataset, not independent chemical batches.

