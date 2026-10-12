# Spectral analysis of the detected sequences

Tests whether the recurring sequences coincide with alpha activity in the EEG (response to
reviewers, R3.3c). The spectrum of the original EEG around each sequence is compared with random
windows from the same recording, and the height of the alpha peak is compared between oscillating
and stable sequences, per cluster.

## Method

- **Sequences.** The paper's back-fitted sequences (`sequences_by_cluster.pkl`) after overlap removal
  (`main_analysis.remove_overlapping_bursts`), stable and oscillating only.
- **EEG.** The paper's preprocessed recordings (`datasets.pkl`: 64 channels, 250 Hz).
- **Sequence windows.** 1.024 s centred on the sequence midpoint; Hann periodogram per channel,
  averaged over channels. Morlet time-frequency power (3-40 Hz, f/2 cycles, at least 2) over ±1 s.
- **Random windows.** 20 per sequence, same recording and length, centres at least 0.5 s from every
  detected sequence of any cluster (5 of them for the time-frequency power).
- **Alpha peak height.** Maximum of the spectrum above its aperiodic (1/f) fit in 8-13 Hz, in dB.
  The fit is a straight line in log-log coordinates over 2-40 Hz, excluding 5-15 Hz.
- **Peak frequency** (statistics file only). Maximum of the same corrected spectrum in 5-15 Hz,
  counted only if it stands at least 1 dB above the fit.
- **Statistics.** Participant = unit: spectra are averaged over a participant's sequences (at least
  3) before features are computed. Paired sign-flip permutation tests (10,000 permutations) over
  participants: oscillating vs random windows, oscillating vs stable (pooled, and within every
  cluster with both polarities in at least 5 participants). Asterisks as in the paper:
  \* p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001.

## Run

```bash
# from paper_scripts/ (the notebook runs these four in this order)
python spectral/compute_recording_alpha.py   # whole-recording Welch spectra (alpha change per session)
python spectral/compute_sequence_spectra.py  # spectra around every sequence (~15 min, 16 GB)
python spectral/plot_sequence_spectra.py     # four-panel figure + sequence_spectra_stats.txt
python spectral/plot_alpha_vs_polarity.py    # Figure 4 + sequence_spectra_alpha_polarity_stats.txt
pytest spectral/test_spectral_tools.py
```

Inputs are read from `RUN_DIR`, outputs go to `RESULTS_DIR/spectral` (`../paper_config.py`).

| File | Contents |
|---|---|
| `spectral_tools.py` | windows, spectra, random-window sampling, alpha features, permutation test |
| `compute_sequence_spectra.py` | spectra around every sequence and its random windows |
| `plot_sequence_spectra.py` | four-panel figure and the per-cluster statistics |
| `compute_recording_alpha.py` | whole-recording spectra (Welch), alpha per recording |
| `plot_alpha_vs_polarity.py` | Figure 4 and its statistics (alpha change vs polarity change) |
| `sequence_spectra.npz` | output: spectra, time-frequency power, sequence metadata |
| `sequence_spectra.png/.pdf` | output: figure (a spectrum, b-c time-frequency, d alpha peak height per cluster) |
| `sequence_spectra_stats.txt` | output: per-group medians and all tests, for both features |
