# Changelog

## 0.2.0

- **Fix (present since 0.1.0): sequence chaining in `stppy.detection.detect_sequences_in_clusters`.**
  The rule (a candidate overlapping the current window is a duplicate; the first candidate
  starting 0..`tolerance_samples` frames after the window's end is appended) was evaluated in
  float seconds, so a candidate starting exactly at, or one frame after, the end of the window
  was accepted or rejected depending on floating-point rounding of its absolute position (zero
  gap rejected at ~18 %, one-frame gap at ~9 % of positions for an 11-sample template). The rule
  is now evaluated on integer sample indices; sequences also carry `start_sample` / `end_sample`.
  On the paper data: 3657 sequences instead of 3494 (3045 instead of 2915 after the overlap
  removal); the conclusions are unchanged.
- `paper_scripts/main_analysis.remove_overlapping_bursts`: the 44 ms cross-cluster overlap is
  measured in integer samples (an overlap of exactly 11 samples was missed at about half of the
  positions in float seconds).
- `paper_scripts/participant_stats/paths.py`: `PAPER_DIR` and `RESULTS_DIR` can be set from the
  environment.
- Tests: `tests/test_sequence_chaining.py`, `paper_scripts/participant_stats/tests/test_overlap_removal.py`.
- **Paper reproduction**: `stppy_quickstart.ipynb` runs the complete analysis of the paper (pipeline,
  then every script in `paper_scripts/`); all paths are variables in `paper_scripts/paper_config.py`.
  New scripts for Table 1 (incl. tightness), Table 4, Figure 9, the Section 2.1 surrogate numbers and
  the test-retest cut; the IAAFT surrogate pipeline (`paper_scripts/surrogates/`) and the
  test-retest analysis (`paper_scripts/test_retest/`) are documented as sequences of scripts.
- `environment.yml`: the conda environment that reproduces the paper bit for bit; `LICENSE` (MIT);
  `pytest` runs the package and paper-script tests from the repository root.
- **New `stppy.jacobian`** — Hilbert phase gradient on irregular electrode layouts
  (`build_gradient_operators`, `compute_phase_gradient`, `propagation_field`) and
  per-polarity movement arrows (`pole_arrows`, `PoleArrow`). Ported from stppy 1.0.0 /
  complete_analysis_stppy100 so no cross-package import is needed.
  - Fix: the gradient operator now includes the `-f_i` term (the 1.0.0 operator omitted
    it). Phase gradients are unchanged (the omitted term is real and `Im()` drops it);
    gradients of any other field are now correct.
- **New `stppy.visualization.plot_clusters_with_arrows`** — all clusters in one figure
  ("Discovered Clusters"), `plot_cluster_average` topomap settings, one time-averaged
  movement arrow per pole (red = +, blue = −), dot only for standing poles, and a
  per-cluster travelling/standing label (`stppy.jacobian.classify_propagation`: travelling
  iff the stronger pole's net arrow reaches 75% of the longest arrow in the figure;
  independent of the 25% dot-vs-arrow cut).
- `stppy_quickstart.ipynb`: average reference OFF (5108 psiconnect avalanches; ON gives
  5333), relative paths, new all-clusters-with-arrows cell.
- `paper_scripts/regen/render_clusters_bothpol.py`: renders the psiconnect figure from
  the avg-ref-OFF regeneration via the new API, validating 10216 = 5108 x 2 avalanches
  and the paper's 12 cluster sizes.
- `tests/`: phase-gradient direction recovery, operator exactness, standing-dipole and
  absent-pole arrows, figure layout.
