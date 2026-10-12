# Participant-level statistics

Every analysis here uses the **participant**, not the sequence, as the unit. The inputs are the
notebook's run (`RUN_DIR`: avalanches, clusters, back-fitted sequences) and the preprocessed
recordings (`RAW_DIR`, for the recording durations); see `../paper_config.py`.

## Run

From `paper_scripts/` (the scripts are modules of this package):

```bash
python -m participant_stats.build_recordings     # recordings.csv (durations, ~1 min)
python -m participant_stats.participant_counts   # N for Table 2 / Figure 3 / text
python -m participant_stats.participant_level    # proportions, rates, models (~1 min)
python -m participant_stats.table1_composition   # Table 1 incl. tightness, and its test (~1 min)
python -m participant_stats.table4_recordings    # Table 4 (needs recordings.csv)
python main_analysis.py                          # Figure 3 incl. panel e
python -m pytest participant_stats                # tests
```

## What each script answers

| script | reviewer point | output (`RESULTS_DIR/`, see `../paper_config.py`) | manuscript |
|---|---|---|---|
| `build_recordings.py` | - | `participant_level/recordings.csv` | input for the rates |
| `participant_counts.py` | R2 minor 2 | `participant_counts/participant_counts.csv` | Table 2 participant column, N in Figure 3b caption, N of the cluster-8 test |
| `participant_level.py` | R2.7 | `participant_level/` (`paired_tests.csv`, `models.csv`, `participant_level.png`) | mixed model OR / CI / N, rates per minute |
| `table1_composition.py` | R2 minor 3 | `table1/` (`composition_tests.csv`, `composition_check.csv`) | Table 1 test and correction family |
| `panel_e.py` | R2.7 | drawn by `main_analysis.py` into `figure3/main_results.pdf` | Figure 3e |

Shared code: `paths.py` (all locations), `data.py` (loaders), `stats.py` (sign-flip, bootstrap,
rank-biserial, chi-square).

## Methods in one line each

- **Participant counts.** Participants with any binary (stable/oscillating) sequence in the slice,
  per session; *paired* = present in both sessions = the N of the within-subject permutation test
  in `main_analysis.py` (single-session participants are dropped there).
- **Proportions.** P(stable) per participant and session; paired Wilcoxon and sign-flip permutation
  on the unweighted mean change (40 paired participants). Binomial GEE (participant cluster,
  task covariate) and binomial mixed GLM (random intercept + random drug slope, variational Bayes).
- **Rates.** Sequences per minute of retained recording; task-matched paired contrast (tasks a
  participant has in both sessions) and Poisson GEE / mixed GLM with log duration.
- **Table 1.** Cluster x level Pearson chi-square for drug, task and training on the 577 clustered
  events (each event once; its polarity copy sits in the mirror cluster k+12). Null by permutation
  respecting the design: per-participant session swap (drug), task shuffle within a
  participant-session (task), label shuffle across participants (training). BH over the 3 factors.
- **Figure 3e.** P(stable) per paired participant, dot area proportional to the number of
  sequences, median bars, sign-flip p with 100,000 permutations (same value as `paired_tests.csv`).

The variational mixed GLM is deterministic only up to ~1e-4 relative between runs.
