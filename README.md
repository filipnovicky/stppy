# STPPY - Spatiotemporal Pattern Processing for Python

Code for the paper:

> **Psilocybin Silences Rotating Spatiotemporal Patterns in Human Cortical Avalanches**
> *Uncovering the spatiotemporal structure of neural avalanches through optimal transport and dynamic time warping*

A Python package for analyzing EEG avalanches with spatial and temporal alignment methods.

**Author:** Filip Novicky (novicky@cs.cas.cz)
**Repository:** https://github.com/filipnovicky/stppy

## Features

- **Data I/O**: Load EEG data in various formats (.set, pickle) with flexible directory searching
- **Preprocessing**: Electrode interpolation and resampling
- **Avalanche Detection**:
  - Threshold-based detection with spatial filtering (de novo event detection)
  - Correlation-based sequence detection (pattern matching against templates)
- **Spatial Alignment**: Optimal transport for spatial pattern matching
- **Temporal Alignment**: (Subsequence) Dynamic Time Warping (DTW) for temporal alignment
- **Clustering**: Hierarchical clustering with consensus building
- **Visualization**: Spatiotemporal patterns as figures and videos
  - Cluster spatiotemporal patterns (consensus averages)
  - Single binary avalanches
  - Original data of backfitted clusters

## Installation

### From source

**Important:** Run these commands in your **terminal/command line**
```bash
git clone https://github.com/filipnovicky/stppy.git
cd stppy
pip install -e .
```

## Usage

> **[paper_scripts/stppy_quickstart.ipynb](paper_scripts/stppy_quickstart.ipynb)** — complete end-to-end walkthrough of the analysis pipeline.

## Pipeline Overview

```
EEG data (.set / pickle)
        │
        ▼
   stppy.io           load_eeg_files, load_eeg_raw, create_electrode_positions_file
        │
        ▼
stppy.preprocessing   interpolate_missing_electrodes, resample_data
        │
        ▼
stppy.detection       create_avalanche_filter → detect_avalanches
        │             → filter_avalanche_propagation → convert_to_peaks
        ▼
stppy.utils           create_eeg_ground_metric, normalize_distance_matrix
        │
        ▼
stppy.alignment       compute_full_similarity_matrix
        │             (large datasets) generate_block_plan → compute_similarity_block
        │                              → merge_blocks → reconstruct_full_matrices
        ▼
stppy.clustering      simple_hierarchical_clustering → build_consensus_alignment
        │             → remove_duplicate_polarities
        ▼
stppy.detection       detect_sequences_by_correlation, detect_sequences_in_clusters
        │
        ▼
stppy.visualization   plot_cluster_average, plot_avalanche_topomaps,
                      plot_sequence_topomaps, create_cluster_video,
                      create_avalanche_video, create_cluster_sequences_video
```

## Package Structure

```
stppy/
├── io/              # Data loading and saving
├── preprocessing/   # Interpolation and resampling
├── detection/       # Avalanche detection and filtering
├── alignment/       # Spatial and temporal alignment
├── clustering/      # Hierarchical clustering and consensus
├── visualization/   # Topomaps and videos
└── utils/           # Distance metrics and utilities
```

| Module | Functions |
|--------|-----------|
| `stppy.io` | `load_eeg_files`, `find_eeg_files`, `load_eeg_raw`, `load_avalanche_data`, `save_pickle`, `load_pickle`, `create_electrode_positions_file` |
| `stppy.preprocessing` | `interpolate_missing_electrodes`, `resample_data` |
| `stppy.detection` | `create_avalanche_filter`, `detect_avalanches`, `convert_to_peaks`, `filter_avalanche_propagation`, `detect_sequences_by_correlation`, `detect_sequences_in_clusters` |
| `stppy.alignment` | `compute_full_similarity_matrix`, `compute_similarity_block`, `generate_block_plan`, `merge_blocks`, `reconstruct_full_matrices`, `spatial_alignment_distance`, `dtw_on_similarity_matrix` |
| `stppy.clustering` | `simple_hierarchical_clustering`, `build_consensus_alignment`, `remove_duplicate_polarities` |
| `stppy.visualization` | `plot_cluster_average`, `plot_avalanche_topomaps`, `plot_sequence_topomaps`, `create_cluster_video`, `create_avalanche_video`, `create_cluster_sequences_video` |
| `stppy.utils` | `create_eeg_ground_metric`, `normalize_distance_matrix` |

## Paper Scripts

Scripts in `paper_scripts/` reproduce the figures from the published paper and are not part of the package:

- `main_analysis.py` — polarity distribution figures (drug/task/experience level comparisons, chi-square tests)
- `cluster_verification.py` — cluster spatial/temporal coherence verification figures
- `blocks/` — standalone scripts for parallel similarity matrix computation on large datasets

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- mne >= 1.0.0
- scikit-learn >= 1.0.0
- opencv-python >= 4.5.0

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

## Citation

If you use this package in your research, please cite:

```
Psilocybin Silences Rotating Spatiotemporal Patterns in Human Cortical Avalanches
```

## License

MIT License

## Contact

For questions or issues, please contact: novicky@cs.cas.cz
