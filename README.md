# STPPY - Spatiotemporal Pattern Processing for Python

A Python package for analyzing EEG avalanches with spatial and temporal alignment methods.

**Author:** Filip Novicky (novicky@cs.cas.cz)
**Repository:** https://github.com/filipnovicky/stppy

## Features

- **Data I/O**: Load EEG data in various formats (.set, pickle) with flexible directory searching
- **Preprocessing**: Electrode interpolation and resampling
- **Avalanche Detection**:
  - Threshold-based detection with spatial filtering (de novo event detection)
  - Correlation-based burst detection (pattern matching against templates)
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

## Quick Start

**Prerequisites:** This package assumes your EEG data has been preprocessed (artifact removal, ICA, bandpass filtering, etc.) before analysis. 
The following steps are a step by step example of how to acquire results from the original paper.

### 1. Load and Preprocess EEG Data

```python
import stppy
import mne

# Find all EEG files (recursively searches directories)
files = stppy.io.find_eeg_files('/path/to/yourdata')

# Create the positions file (use a file with all electrodes present)
# Note: Some files may have electrodes removed, so find one with complete montage
raw_complete = stppy.io.load_eeg_raw(files[0], preload=True)  # This file contains all electrodes after cleaning
stppy.io.create_electrode_positions_file(raw_complete, './electrode_positions.json')

# Process all files
datasets = []
for i, file_path in enumerate(files):
    print(f"Processing {i+1}/{len(files)}: {file_path}")

    # Load raw data
    raw = stppy.io.load_eeg_raw(file_path, preload=True)

    # Interpolate missing electrodes (with optional average reference)
    raw_interp, interp_channels = stppy.preprocessing.interpolate_missing_electrodes(
        raw,
        positions_file='./electrode_positions.json',
        apply_average_reference=True,  # Optional: set to True to apply average reference
        verbose=False
    )

    # Resample to 250 Hz (Optional)
    raw_250 = stppy.preprocessing.resample_data(raw_interp, 250)

    # Get data array and add to datasets
    data = raw_250.get_data()
    datasets.append(data)
```

### 2. Detect Avalanches

```python
# Create avalanche filter
crossover_thresholds = [-3, 3]
filter_masks = stppy.detection.create_avalanche_filter(datasets, crossover_thresholds)

# Detect avalanches
all_avalanches, all_avalanche_times = stppy.detection.detect_avalanches(
    datasets,
    filter_masks,
    crossover_thresholds,
    min_duration=11
)

# Apply spatial filtering
K, channel_names = stppy.utils.create_eeg_ground_metric('./electrode_positions.json')

filtered_avalanches, filtered_times = stppy.detection.filter_avalanche_propagation(
    all_avalanches,
    all_avalanche_times,
    K,
    './electrode_positions.json',
    min_active_neighbors=5
)

# Convert to peak representation
peak_avalanches = stppy.detection.convert_to_peaks(filtered_avalanches)
stppy.io.save_pickle(peak_avalanches, './peak_avalanches.pkl')
stppy.io.save_pickle(filtered_times, './peak_avalanche_times.pkl')
```

### 3. Compute Similarity Matrix (All Avalanches)

#### Option A: Serial Computation (Small Datasets < 5000 avalanches)

```python
# Load avalanche data (include flipped versions to double dataset size)
# Optionally load timing info alongside arrays
avalanches, avalanche_times = stppy.io.load_avalanche_data(
    './peak_avalanches.pkl',
    min_time_length=11,
    include_flipped=True,  # Adds polarity-flipped versions (positive ↔ negative)
    times_filepath='./peak_avalanche_times.pkl'  # Optional: load timing info
)

# Create distance matrix
K, channel_names = stppy.utils.create_eeg_ground_metric('./electrode_positions.json')
K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

# Compute full pairwise similarity matrix
similarity_matrix, path_matrix = stppy.alignment.compute_full_similarity_matrix(
    avalanches,
    K_norm,
    show_progress=True,
    save_interval=100,  # Save checkpoint every 100 pairs
    checkpoint_file='./similarity_checkpoint.pkl'
)

# Save results
stppy.io.save_pickle(similarity_matrix, './similarity_matrix.pkl')
stppy.io.save_pickle(path_matrix, './path_matrix.pkl')
```

#### Option B: Parallel Block Computation (Large Datasets > 5000 avalanches)

```python
# Step 1: Generate block plan
plan = stppy.alignment.generate_block_plan(
    n_avalanches=len(avalanches),
    block_size=2000
)
# Outputs: "Need to compute 15 blocks"

# Step 2: Create a script for each block (RUN IN PARALLEL on different machines/cores)
# Example: block_000.py
```

```python
#!/usr/bin/env python3
"""Block 000: diagonal block (0, 0)"""

import stppy
import os

# Ensure blocks directory exists
os.makedirs('./blocks', exist_ok=True)

# Load avalanche data (include flipped versions)
print("Loading avalanche data...")
avalanches, avalanche_times = stppy.io.load_avalanche_data(
    './peak_avalanches.pkl',
    min_time_length=11,
    include_flipped=True,
    times_filepath='./peak_avalanche_times.pkl'
)
print(f"Loaded {len(avalanches)} avalanches (including flipped)")

# Create normalized distance matrix
print("Creating normalized distance matrix...")
K, channel_names = stppy.utils.create_eeg_ground_metric('./electrode_positions.json')
K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

# Compute block 000 (diagonal block)
print("Computing block 000...")
block_0 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=0, row_end=2000,
    col_start=0, col_end=2000,
    is_diagonal=True
)
stppy.io.save_pickle(block_0, './blocks/block_000.pkl')
print("Block 000 complete")
```

```python
# Similarly, create block_001.py, block_002.py, etc. for other blocks
# Job 1: block_001.py (off-diagonal block)
block_1 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=2000, row_end=4000,
    col_start=0, col_end=2000,
    is_diagonal=False
)
stppy.io.save_pickle(block_1, './blocks/block_001.pkl')
# ... (create one script per block and run in parallel)

# Step 3: Merge all blocks
import glob
block_files = sorted(glob.glob('./blocks/block_*.pkl'))
merged = stppy.alignment.merge_blocks(
    block_files,
    output_file='./similarity_merged.pkl'
)

# Step 4: Reconstruct full matrices
similarity_matrix, path_matrix = stppy.alignment.reconstruct_full_matrices(
    merged,
    n_avalanches=len(avalanches)
)

# Save
stppy.io.save_pickle(similarity_matrix, './similarity_matrix.pkl')
stppy.io.save_pickle(path_matrix, './path_matrix.pkl')
```

### 4. Cluster Avalanches

```python
# Load similarity matrix
similarity_matrix = stppy.io.load_pickle('./similarity_matrix.pkl')
path_matrix = stppy.io.load_pickle('./path_matrix.pkl')

# Hierarchical clustering
clusters, cluster_info = stppy.clustering.simple_hierarchical_clustering(
    similarity_matrix,
    threshold=3.5, # Arbitrary free parameter
    method='average',
    min_cluster_size=30 # Arbitrary free parameter
)

# Build consensus alignments for all clusters
cluster_averages = {}
for cluster_id in range(len(cluster_info)):
    aligned, average, info, medoid_idx, counts = stppy.clustering.build_consensus_alignment(
        cluster_id=cluster_id,
        clusters=clusters,
        avalanche_arrays=avalanches,
        avalanche_paths=path_matrix,
        similarity_clean=similarity_matrix
    )
    cluster_averages[cluster_id] = average

# Remove duplicate polarities (if using include_flipped=True in Step 3)
# Clusters with correlation ≤ -0.999 are polarity-flipped duplicates
clusters_to_keep = stppy.clustering.remove_duplicate_polarities(
    cluster_averages,
    threshold=-0.999
)
```

### 5. Detect Cluster Bursts in Continuous Data

Detect when cluster patterns appear in the continuous recordings and identify
bursts (2+ adjacent activations). An activation is grouped into a burst if it
starts within `tolerance_samples` frames after the previous activation ends.

```python
import numpy as np

# Build spatial smoothing matrix from electrode positions
positions = stppy.io.load_electrode_positions('./electrode_positions.json')
position_array = np.array([[positions[ch]['x'], positions[ch]['y'], positions[ch]['z']]
                           for ch in sorted(positions.keys())])

W_smooth, distances, mean_dist = stppy.detection.build_spatial_smoothing_matrix(
    position_array,
    k=10.0  # Spatial decay parameter (higher = less smoothing)
)


# Collect correlation timeseries for each cluster
# Using return_all=True gives correlations at EVERY timestep,
# allowing downstream analysis to apply any threshold as a free parameter
cluster_activation_times = {}
cluster_durations = {}

for cluster_id in clusters_to_keep:
    cluster_template = cluster_averages[cluster_id]
    cluster_duration = cluster_template.shape[1] / 250.0  # seconds
    cluster_durations[cluster_id] = cluster_duration

    activations = []
    for dataset_idx, continuous_data in enumerate(datasets):
        # Get correlation at every sliding window position (step=1)
        result = stppy.detection.detect_bursts_by_correlation(
            continuous_data,
            cluster_template,
            W_smooth,
            sfreq=250.0,
            return_all=True  # Return ALL correlations (no threshold filtering)
        )
        activations.append((dataset_idx, result['times'],
                          result['polarities'], result['correlations']))

    cluster_activation_times[cluster_id] = activations

# Save full correlation data (threshold can be applied later as free parameter)
stppy.io.save_pickle(
    {'cluster_activation_times': cluster_activation_times},
    './cluster_analysis.pkl'
)
stppy.io.save_pickle(cluster_durations, './cluster_durations.pkl')

# Optionally, detect bursts with a specific threshold using the library
# (threshold is applied inside detect_bursts_in_clusters via correlation_threshold)
bursts_by_cluster = stppy.detection.detect_bursts_in_clusters(
    cluster_activation_times,
    cluster_durations,
    dataset_names=[os.path.basename(f) for f in files],
    tolerance_samples=2,  # Max samples after end of previous activation (2 samples = 8ms at 250Hz)
    sfreq=250.0,
    min_burst_length=2,
    correlation_threshold=0.825  # Applied here as a free parameter
)

# View burst locations for each cluster
for cluster_id, bursts in bursts_by_cluster.items():
    print(f"\nCluster {cluster_id}: {len(bursts)} bursts")
    for burst in bursts[:5]:  # Show first 5
        print(f"  {burst['dataset']}: "
              f"{burst['start_time']:.3f}s - {burst['end_time']:.3f}s "
              f"({burst['length']} activations)")

stppy.io.save_pickle(bursts_by_cluster, './bursts_by_cluster.pkl')
```

Project-specific analysis scripts (polarity profiles, chi-square tests, per-participant breakdowns, etc.) are available in the `scripts/` folder. Point them to your data directory with:

```bash
export PSICONNECT_DATA_DIR=/path/to/your/data
python3 scripts/main_analysis.py
```

### 6. Visualize Spatiotemporal Patterns

```python
# Plot cluster spatiotemporal pattern (consensus average)
fig = stppy.visualization.plot_cluster_average(
    average,
    './electrode_positions.json',
    cluster_id=0
)

# Sample from ALL avalanches (no clustering filter)
fig = stppy.visualization.plot_avalanche_topomaps(
    avalanches,
    './electrode_positions.json',
    n_samples=10,
    random_seed=42
)

# Sample 5 avalanches from cluster 0
fig = stppy.visualization.plot_avalanche_topomaps(
    avalanches,
    './electrode_positions.json',
    clusters=clusters,
    cluster_id=0,
    n_samples=5
)

# Sample from noise/discarded avalanches (cluster = -1)
fig = stppy.visualization.plot_avalanche_topomaps(
    avalanches,
    './electrode_positions.json',
    clusters=clusters,
    cluster_id=-1,
    n_samples=10
)

# Manual indices (backwards compatible)
fig = stppy.visualization.plot_avalanche_topomaps(
    avalanches,
    './electrode_positions.json',
    indices=[0, 5, 10]
)

# Create video from cluster spatiotemporal pattern
stppy.visualization.create_cluster_video(
    average,
    './electrode_positions.json',
    output_file='./cluster_0.mp4',
    cluster_id=0,
    fps=5
)

# Create video from single avalanche
stppy.visualization.create_avalanche_video(
    avalanche_arrays[0],
    './electrode_positions.json',
    output_file='./avalanche_0.mp4'
)

# Create video showing bursts from a cluster (with optional filtering)
stppy.visualization.create_cluster_bursts_video(
    bursts_by_cluster,
    datasets,
    dataset_names=[os.path.basename(f) for f in files],
    './electrode_positions.json',
    output_file='./cluster_0_bursts.mp4',
    cluster_id=0,            # Which cluster to show
    polarity='stable',       # Optional: filter by polarity ('stable', 'switching', ...)
    sort_by_length=True,     # Optional: show longest bursts first
    n_bursts=10,             # How many bursts to include
    fps=10
)

# Plot burst topomaps (static figure of original EEG data during bursts)
fig = stppy.visualization.plot_burst_topomaps(
    bursts_by_cluster,
    datasets,
    dataset_names=[os.path.basename(f) for f in files],
    './electrode_positions.json',
    cluster_id=0,
    polarity='stable',
    sort_by_length=True,
    n_bursts=3
)
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
Recurring Spatiotemporal Patterns in Human EEG Reveal Psilocybin-Induced Shift from Rotating to Stable Propagation
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please contact: novicky@cs.cas.cz
