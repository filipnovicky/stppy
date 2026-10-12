# Travelling/standing x stable/oscillating (Table 3)

Do the two cluster classifications map onto each other? They are treated as separate properties
and reported as a full 2x2.

| axis | what it describes | timescale |
|---|---|---|
| travelling / standing | spatial evolution **within** one template; travelling = clusters 0, 2, 3, 4, 5 | 40-44 ms |
| stable / oscillating | polarity relation **between** successive back-fitted occurrences of a template (a sequence) | sequence |

`travelling_polarity.py` loads the back-fitted sequences (`sequences_by_cluster.pkl`), applies the
overlap removal of `main_analysis.remove_overlapping_bursts` and

1. counts stable/oscillating sequences per cluster and session and checks them against Table 2
   (prints "Counts match paper Table 2: yes");
2. builds the cluster-level 2x2 per session, each cluster placed by its predominant polarity;
3. builds the sequence-level 2x2 per session (Table 3, descriptive);
4. runs a Fisher exact test on the cluster-level 2x2.

Cluster 3 is excluded (a single sequence). The test is on clusters, not sequences: travelling/standing
is a property of a cluster, so its sequences are not independent evidence (thousands of sequences,
11 labelled clusters).

```bash
# from paper_scripts/; paths from paper_config.py
python travelling_polarity/travelling_polarity.py   # -> RESULTS_DIR/travelling_polarity/
pytest travelling_polarity
```
