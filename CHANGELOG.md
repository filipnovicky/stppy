# Changelog

## 0.2.0

- **Fix: sequence chaining** in `stppy.detection.detect_sequences_in_clusters` (present since 0.1.0).
  The chaining rule was evaluated in float seconds, so a match starting exactly at, or one frame
  after, the end of the previous window was accepted or rejected depending on rounding. It is now
  evaluated on integer samples, and sequences also carry `start_sample` / `end_sample`. On the paper
  data: 3657 instead of 3494 sequences (3045 instead of 2915 after overlap removal); the conclusions
  are unchanged.
- **Fix: overlap removal** in `paper_scripts/main_analysis.py` is measured in integer samples too.
- **New `stppy.jacobian`**: Hilbert phase gradient on irregular electrode layouts, one movement arrow
  per pole (`pole_arrows`) and the travelling/standing label (`classify_propagation`).
- **New `stppy.visualization.plot_clusters_with_arrows`**: all clusters in one figure with their
  movement arrows.
- **Paper reproduction**: `stppy_quickstart.ipynb` runs the whole analysis of the paper and then every
  script in `paper_scripts/`; locations are set in `paper_scripts/paper_config.py`, and
  `environment.yml` reproduces the numbers bit for bit.
- Tests (`pytest`), `LICENSE` (MIT); dependencies reduced to those actually imported.
