# STPPY - Spatiotemporal Pattern Processing for Python

Code for the paper:

> **Uncovering the spatiotemporal structure of neural avalanches through optimal transport and dynamic time warping (https://www.biorxiv.org/content/10.64898/2026.07.10.737743v1)**

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
- **Propagation**: Phase-gradient movement arrows per pole, travelling vs standing patterns
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
pip install -e .            # the package
pip install -e ".[paper]"   # + what the notebook and paper_scripts/ need (pandas, statsmodels, nbclient)
```

To reproduce the paper's numbers bit for bit, use the conda environment it was computed with
(linux-64):

```bash
conda env create -f environment.yml
conda activate stppy-paper
pip install -e .
```

## Usage

> **[stppy_quickstart.ipynb](stppy_quickstart.ipynb)** — complete end-to-end walkthrough of the analysis pipeline, which also runs every analysis of the paper (see [Reproducing the Paper](#reproducing-the-paper)).

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
stppy.jacobian        pole_arrows → classify_propagation (travelling vs standing templates)
        │
        ▼
stppy.detection       detect_sequences_by_correlation, detect_sequences_in_clusters
        │
        ▼
stppy.visualization   plot_cluster_average, plot_clusters_with_arrows, plot_avalanche_topomaps,
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
├── jacobian/        # Phase gradient, per-pole movement arrows
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
| `stppy.jacobian` | `compute_phase_gradient`, `build_gradient_operators`, `propagation_field`, `pole_arrows`, `classify_propagation` |
| `stppy.visualization` | `plot_cluster_average`, `plot_clusters_with_arrows`, `plot_avalanche_topomaps`, `plot_sequence_topomaps`, `create_cluster_video`, `create_avalanche_video`, `create_cluster_sequences_video` |
| `stppy.utils` | `create_eeg_ground_metric`, `normalize_distance_matrix` |

## Reproducing the Paper

`stppy_quickstart.ipynb` runs the whole analysis of the paper: detection, distances, clusters
(Figure 1), back-fitted sequences, and then every script in `paper_scripts/` (Figures 2-4, 8-10,
Tables 1-4 and all statistics); `paper_scripts/README.md` lists which script makes which figure.
The scripts in `paper_scripts/` are not part of the package.

1. **Data.** The PsiConnect EEG (OpenNeuro [ds006110](https://openneuro.org/datasets/ds006110)),
   preprocessed in EEGLAB as described in the paper (one `.set` file per recording).
2. **Paths.** Set `RAW_DIR` (the `.set` files) and, if needed, the other locations in
   `paper_scripts/paper_config.py`: as environment variables, in an untracked
   `paper_scripts/paper_config_local.py`, or by keeping the defaults (`data/`, `runs/` inside the
   repository).
3. **Run** the notebook from the repository root. Steps 1-5 need ~100 GB of RAM; the distance matrix
   costs ~370 CPU-hours (set `N_WORKERS`) unless precomputed distances are found in `DISTANCE_DIR`.
   The surrogate (Figures 9-10) and test-retest (Figure 8) analyses are sequences of scripts run on a
   cluster, documented in `paper_scripts/surrogates/` and `paper_scripts/test_retest/`.

The paper's numbers were produced in the environment of `environment.yml` (Python 3.11, numpy 2.0.1
built against MKL, scipy 1.16.0, mne 1.10.1). Other builds change the preprocessed signals in the
last bits, which can move single threshold crossings (a pip install of the same versions, with
OpenBLAS, finds 5107 instead of 5108 avalanches).

## Dependencies

- Python >= 3.8 (tested with 3.8, 3.10, 3.11, 3.13)
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- mne >= 1.0.0
- scikit-learn >= 1.0.0
- opencv-python >= 4.5.0

## Tests

```bash
pip install -e ".[dev]"
pytest          # package and paper scripts, synthetic data only
```

## Citation

If you use this package in your research, please cite:

```
Uncovering the spatiotemporal structure of neural avalanches through optimal transport and dynamic time warping
(https://www.biorxiv.org/content/10.64898/2026.07.10.737743v1)
```

## License

MIT License, see `LICENSE`.

## Contact

For questions or issues, please contact: novicky@cs.cas.cz
