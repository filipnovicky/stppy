# Travelling/standing x stable/oscillating (R1.3)

Reviewer 1 (point 3) asked whether the two cluster classifications map onto each other, and
suggested treating them as orthogonal and reporting the full 2x2.

## The two axes

| axis | what it describes | timescale | source |
|---|---|---|---|
| travelling / standing | spatial evolution **within** one template | 40-44 ms | paper classification: travelling = clusters 0, 2, 3, 4, 5 (reproduced by the stppy020 pole-arrow measure, stronger pole >= 0.75 of the longest) |
| stable / oscillating | polarity relation **between** successive back-fitted occurrences of a template (a sequence) | sequence | the paper's back-fitted sequences after overlap removal, as in Table 2 |

## What the script does

`travelling_polarity.py` loads the paper's sequences (`sequences_by_cluster.pkl`), applies
the paper's overlap removal (`main_analysis.remove_overlapping_bursts`) and

1. counts stable/oscillating sequences per cluster and session, and **checks them against the
   paper's Table 2** (must print "Counts match paper Table 2: yes");
2. builds the **cluster-level 2x2** per session: each cluster placed by its predominant
   (majority) polarity in that session;
3. builds the **sequence-level 2x2** per session (descriptive only);
4. runs a **Fisher exact test on the cluster-level 2x2** (two-sided, and one-sided for
   travelling -> oscillating).

Cluster 3 is excluded (a single sequence).

**Why the test is on clusters, not sequences.** Travelling/standing is a property of a
cluster, so every sequence of a cluster carries the same label. A test on the sequence
table would treat ~2,000 sequences as independent evidence for an association that is
really observed on 11 clusters, and would be grossly anti-conservative.

## Results (paper data, avg-ref off)

Cluster-level, baseline:

|            | oscillating | stable |
|---|---|---|
| travelling | 0, 2, 4, 5  | -        |
| standing   | 1, 6, 8, 10 | 7, 9, 11 |

Fisher exact p = 0.24 (two-sided), 0.21 (one-sided). Under psilocybin only cluster 8 moves
(standing: oscillating -> stable); p = 0.19 / 0.11.

Sequence-level (baseline / psilocybin):

|            | oscillating | stable |
|---|---|---|
| travelling | 1464 / 433 | 1 / 0     |
| standing   | 332 / 100  | 300 / 284 |

**Reading.** The relation is asymmetric: every travelling cluster is oscillating, but four
standing clusters are oscillating too (1 and 10 near-exclusively, 6 and 8 mixed). With 11
clusters the association is not statistically significant, so the two axes should be
reported as distinct properties, not as one classification.

## Run

```bash
python travelling_polarity.py        # seconds; writes RESULTS_DIR/travelling_polarity/travelling_polarity_2x2.txt
pytest test_travelling_polarity.py
```

Paths (`RUN_DIR`, `RESULTS_DIR`) come from `../paper_config.py`.
