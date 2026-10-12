# paper_scripts

Analysis scripts for the paper; not part of the `stppy` package. `../stppy_quickstart.ipynb` runs
the pipeline (detection -> distances -> clusters -> back-fitted sequences) and then every script
below. All locations are variables in **`paper_config.py`**, set by an environment variable of the
same name, an untracked `paper_config_local.py` (machine-specific paths), or the defaults
(folders inside the repository):

| variable | contents |
|---|---|
| `RAW_DIR` | preprocessed PsiConnect EEGLAB `.set` files (notebook input) |
| `RUN_DIR` | pipeline products written by the notebook, read by every script |
| `DISTANCE_DIR` | UOT + subDTW distances: loaded by the notebook if present, otherwise computed and saved here |
| `RESULTS_DIR` | outputs, one subfolder per analysis |
| `SURROGATE_DIR` | IAAFT pipeline outputs (`surrogates/README.md`) |
| `TEST_RETEST_DIR` | test-retest (ds004148) run (`test_retest/README.md`) |

| paper | script (run from `paper_scripts/`) | output in `RESULTS_DIR` |
|---|---|---|
| Figure 1, Section 2.1 counts, travelling/standing | notebook, steps 2 and 4 | `figure1/` |
| Figure 2, Section 2.2.1 | `cluster_verification.py` | `figure2/` |
| Table 1 | `python -m participant_stats.table1_composition` | `table1/` |
| Table 4 | `python -m participant_stats.build_recordings`, `... table4_recordings` | `participant_level/`, `table4/` |
| Figure 3, Table 2, Sections 2.2.2-2.3 | `main_analysis.py`, `participant_stats.participant_counts`, `participant_stats.participant_level` | `figure3/`, `participant_counts/`, `participant_level/` |
| Table 3 | `travelling_polarity/travelling_polarity.py` | `travelling_polarity/` |
| Figure 4, Section 2.2.4 | `spectral/compute_recording_alpha.py`, `compute_sequence_spectra.py`, `plot_sequence_spectra.py`, `plot_alpha_vs_polarity.py` | `spectral/` |
| Figures 9, 10, Section 2.1 surrogates | `surrogates/plot_iaaft_clusters.py`, `plot_durations.py`, `surrogate_numbers.py` | `surrogates/` |
| Figure 8, Appendix A | `test_retest/percentile_cut.py`, `test_retest/plot_test_retest_clusters.py` | `test_retest/` |

Figures 5-7 are schematic illustrations. Other files:

| path | what |
|---|---|
| `psiconnect_permutation.py`, `psiconnect_utils.py` | permutation tests and shared helpers |
| `_validate_permutation.py` | checks of the permutation implementation |

Each subfolder has its own README.
