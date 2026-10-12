# surrogates: Section 2.1 surrogate comparison, Figures 9 and 10

Multivariate IAAFT surrogates (`surrogates.py`, tests in `test_surrogates.py`), one per
recording, run through the paper's pipeline. Paths come from `../paper_config.py`
(`RUN_DIR` for the recordings and the paper's templates, `SURROGATE_DIR` for the pipeline outputs,
`RESULTS_DIR/surrogates` for the paper outputs). Run from `paper_scripts/`.

## Sequence

1. **Pools** - `ARMS=real,iaaft NCPUS=32 python surrogates/pools.py`: one surrogate per recording
   (seed 20260925 + recording index) and the paper's detection (+-3 SD, >= 11 samples, 5/8
   filter) -> `SURROGATE_DIR/pools/<arm>_peak_avalanches.pkl`, and `<arm>_durations.npz` (every
   supra-threshold run, for Figure 10). IAAFT: ~35 min on 32 CPUs. Arm `real` reproduces the
   paper's 5108 avalanches.
2. **Distances** - `NBLOCKS=5 BLOCK=<1..5> NCPUS=32 python surrogates/full_route.py blocks iaaft`
   (one call per block, e.g. as a job array): UOT + subDTW, ~100 CPU-hours ->
   `SURROGATE_DIR/full/iaaft/block_*of05.npz`.
3. **Clusters** - `NBLOCKS=5 python surrogates/full_route.py cluster iaaft`: threshold 3.5, average
   linkage, minimum 30, polarity de-duplication, DTW consensus -> `SURROGATE_DIR/full/iaaft/clusters.npz`
   (18 clusters). `full_route.py cluster real` runs the same on the paper's own matrix and checks that
   the consensus templates equal the paper's.
4. **Paper outputs**:
   - `python surrogates/plot_iaaft_clusters.py` - Figure 9, matched |r| and the permutation null;
   - `python surrogates/plot_durations.py` - Figure 10;
   - `python surrogates/surrogate_numbers.py` - durations, pairs within the threshold, share of
     events in clusters (Section 2.1).

`compare.py` (template matching, permutation null, pole-arrow strengths) and `linear_distance.py`
(the back-fit similarity between two templates) are also used by `../test_retest/`.
