# test_retest: Appendix A (Figure 8)

The paper's 12 templates recovered in an independent dataset: Wang, Duan, Dong, Ding & Lei (2022),
*A test-retest resting, and cognitive state EEG dataset during multiple subject-driven states*,
Scientific Data 9, 566, doi:10.1038/s41597-022-01607-9 (OpenNeuro ds004148), eyes-closed resting
recordings (60 participants x 3 sessions = 180 recordings).

## Sequence (original code, `original/`)

The analysis was run with the scripts in `original/`, copied unchanged from the analysis
directory (including its absolute paths); the cluster job files, which only ran the commands
below, are left out. The scripts import stppy 0.1.0 (pinned in `original/config.py`), whose
detection, alignment and clustering code is the same as this repository's (no
back-fitting is involved, so the 0.2.0 chaining fix does not apply). To rerun them, set the paths
in `original/config.py` and at the top of `original/preprocess_ds004148.py`; steps 1-5 and 7 run
from `original/` with `RUN=ds4148_EC`.

1. **Preprocessing** - `python preprocess_ds004148.py <i>`, i = 1..360 (60 participants x
   3 sessions x eyes closed/open): restore FCz, average reference, 250 Hz, bad channels
   interpolated, 1-40 Hz, ICA + ICLabel; one `.npz` per recording.
2. **Electrode positions** - `python -c "import config as c; c.write_positions()"` ->
   `electrode_positions.json` (mm); the figure uses the same positions in metres
   (`electrode_positions_m.json`, every coordinate / 1000).
3. **Detection** - `REC_INDEX=<i> python detect.py`, i = 1..180 (one per eyes-closed recording):
   +-3 SD, >= 11 samples, 5/8-neighbour filter, sign encoding.
4. **Pool** - `python pool.py`: all 2,767 avalanches (below the 5,000 cap) + polarity copies.
5. **Distances** - `NBLOCKS=16 BLOCK=<b> NCPUS=32 python matrix_block.py`, b = 1..16 (UOT +
   subDTW), then `NBLOCKS=16 python merge.py` -> `similarity.npz`.
6. **Cut** - `python test_retest/percentile_cut.py` (from `paper_scripts/`): the paper's cut 3.5 is
   the 1.83 % quantile of its distances; the same quantile here is 3.84.
7. **Candidate clusters** - `RUN=ds4148_EC CLUSTER_THRESHOLD=3.84 MIN_SIZE=10 python candidates.py`:
   average linkage at 3.84, minimum size 10, DTW consensus, polarity de-duplication (30 clusters).
8. **Figure 8 and the matching numbers** - `python test_retest/plot_test_retest_clusters.py` (from
   `paper_scripts/`): Hungarian matching on the 60 shared channels, null with permuted electrode
   labels (50 permutations), travelling at 0.7 of the PsiConnect reference arrow.

Steps 6 and 8 read `TEST_RETEST_DIR` (paper_config.py) and write to `RESULTS_DIR/test_retest`.
