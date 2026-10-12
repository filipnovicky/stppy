# test_retest: Appendix A (Figure 8)

The paper's 12 templates recovered in an independent dataset: Wang, Duan, Dong, Ding & Lei (2022),
*A test-retest resting, and cognitive state EEG dataset during multiple subject-driven states*,
Scientific Data 9, 566, doi:10.1038/s41597-022-01607-9 (OpenNeuro ds004148), eyes-closed resting
recordings (60 participants x 3 sessions = 180 recordings).

## Sequence (original code, `original/`)

The analysis was run with the scripts in `original/`, copied unchanged from the analysis
directory (including its absolute paths). They import stppy 0.1.0 (pinned in `original/config.py`),
whose detection, alignment and clustering code is byte-identical to this repository's (no
back-fitting is involved, so the 0.2.0 chaining fix does not apply). To rerun them, set the paths
in `original/config.py` and at the top of `original/preprocess_ds004148.py`.
Every step runs with `RUN=ds4148_EC`.

1. **Preprocessing** - `preprocess_ds004148.py` (`run_preprocess.pbs`): restore FCz, average
   reference, 250 Hz, bad channels interpolated, 1-40 Hz, ICA + ICLabel; one `.npz` per recording.
2. **Electrode positions** - `positions_run.pbs`:
   `python -c "import config as c; c.write_positions()"` -> `electrode_positions.json` (mm);
   the figure uses the same positions in metres (`electrode_positions_m.json`, every coordinate / 1000).
3. **Detection** - `detect_run.pbs` (`qsub -J 1-180 -v RUN=ds4148_EC`): +-3 SD, >= 11 samples,
   5/8-neighbour filter, sign encoding, per recording (`detect.py`).
4. **Pool** - `pool_run.pbs` (`pool.py`): all 2,767 avalanches (below the 5,000 cap) + polarity copies.
5. **Distances** - `matrix_run.pbs` (`matrix_block.py`, 16 blocks of UOT + subDTW), then
   `merge.py` (first line of `finish_run.pbs`) -> `similarity.npz`.
6. **Cut** - `python test_retest/percentile_cut.py` (in stppy020): the paper's cut 3.5 is the 1.83 %
   quantile of its distances; the same quantile here is 3.84.
7. **Candidate clusters** - `RUN=ds4148_EC CLUSTER_THRESHOLD=3.84 MIN_SIZE=10 python candidates.py`:
   average linkage at 3.84, minimum size 10, DTW consensus, polarity de-duplication (30 clusters).
8. **Figure 8 and the matching numbers** - `python test_retest/plot_test_retest_clusters.py` (in
   stppy020; same recipe as `original/match_paper.py` with `T=3.84 MIN_SIZE=10`, only the title
   differs): Hungarian matching on the 60 shared channels, null with permuted electrode labels
   (50 permutations), travelling at 0.7 of the PsiConnect reference arrow.

Steps 6 and 8 read `TEST_RETEST_DIR` (paper_config.py) and write to `RESULTS_DIR/test_retest`.
Below the title the figure is pixel-identical to the submitted one when rendered in the `base`
conda env (numpy 2.3.3, matplotlib 3.10.6).
