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
- **Analysis**:
  - Universal metadata system (user-defined fields)
  - Flexible querying and grouping

## Installation

### From source

```bash
git clone https://github.com/filipnovicky/stppy.git
cd stppy
pip install -e .
```

### Using pip (when published)

```bash
pip install stppy
```

## Quick Start

### 1. Load and Preprocess EEG Data

```python
import stppy
import mne

# Create electrode positions file (one-time setup)
raw = mne.io.read_raw_eeglab('your_data.set', preload=True)
raw.set_montage('standard_1020')
stppy.io.create_electrode_positions_file(raw, 'electrode_positions.json')

# Find all EEG files (recursively searches directories)
files = stppy.io.find_eeg_files('/path/to/yourdata')

# Load a single file
raw = stppy.io.load_eeg_raw(files[0], preload=True)

# Interpolate missing electrodes (with optional average reference)
raw_interp, interp_channels = stppy.preprocessing.interpolate_missing_electrodes(
    raw,
    positions_file='electrode_positions.json',
    apply_average_reference=True  # Optional: set to True to apply average reference
)

# Resample to 250 Hz
raw_250 = stppy.preprocessing.resample_data(raw_interp, 250)
```

### 2. Detect Avalanches

```python
# Get data arrays
datasets = [raw.get_data() for raw in processed_raws]

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
K, channel_names = stppy.utils.create_eeg_ground_metric('electrode_positions.json')

filtered_avalanches, filtered_times = stppy.detection.filter_avalanche_propagation(
    all_avalanches,
    all_avalanche_times,
    K,
    'electrode_positions.json',
    min_active_neighbors=5
)

# Convert to peak representation
peak_avalanches = stppy.detection.convert_to_peaks(filtered_avalanches)
```

### 3. Compute Similarity Matrix (All Avalanches)

#### Option A: Serial Computation (Small Datasets < 1000 avalanches)

```python
# Load avalanche data
avalanches = stppy.io.load_avalanche_data('avalanches.pkl', min_time_length=11)

# Create distance matrix
K, channel_names = stppy.utils.create_eeg_ground_metric('electrode_positions.json')
K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

# Compute full pairwise similarity matrix
similarity_matrix, path_matrix = stppy.alignment.compute_full_similarity_matrix(
    avalanches,
    K_norm,
    show_progress=True,
    save_interval=100,  # Save checkpoint every 100 pairs
    checkpoint_file='similarity_checkpoint.pkl'
)

# Save results
stppy.io.save_pickle(similarity_matrix, 'similarity_matrix.pkl')
stppy.io.save_pickle(path_matrix, 'path_matrix.pkl')
```

#### Option B: Parallel Block Computation (Large Datasets > 1000 avalanches)

```python
# Step 1: Generate block plan
plan = stppy.alignment.generate_block_plan(
    n_avalanches=len(avalanches),
    block_size=500
)
# Outputs: "Need to compute 15 blocks"

# Step 2: Compute each block (RUN IN PARALLEL on different machines/cores)
# Job 0:
block_0 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=0, row_end=500,
    col_start=0, col_end=500,
    is_diagonal=True
)
stppy.io.save_pickle(block_0, 'blocks/block_000.pkl')

# Job 1:
block_1 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=500, row_end=1000,
    col_start=0, col_end=500,
    is_diagonal=False
)
stppy.io.save_pickle(block_1, 'blocks/block_001.pkl')
# ... (repeat for all blocks in parallel)

# Step 3: Merge all blocks
import glob
block_files = sorted(glob.glob('blocks/block_*.pkl'))
merged = stppy.alignment.merge_blocks(
    block_files,
    output_file='similarity_merged.pkl'
)

# Step 4: Reconstruct full matrices
similarity_matrix, path_matrix = stppy.alignment.reconstruct_full_matrices(
    merged,
    n_avalanches=len(avalanches)
)

# Save
stppy.io.save_pickle(similarity_matrix, 'similarity_matrix.pkl')
stppy.io.save_pickle(path_matrix, 'path_matrix.pkl')
```

See `examples/compute_similarity_parallel.py` for a complete parallel workflow script.

### 4. Cluster Avalanches

```python
# Load similarity matrix
similarity_matrix = stppy.io.load_pickle('similarity_matrix.pkl')
path_matrix = stppy.io.load_pickle('path_matrix.pkl')

# Hierarchical clustering
clusters, cluster_info = stppy.clustering.simple_hierarchical_clustering(
    similarity_matrix,
    threshold=3.5,
    method='average',
    min_cluster_size=30
)

# Build consensus alignment for a cluster
aligned, average, info, medoid_idx, counts = stppy.clustering.build_consensus_alignment(
    cluster_id=0,
    clusters=clusters,
    avalanche_arrays=avalanches,
    avalanche_paths=path_matrix,
    similarity_clean=similarity_matrix
)
```

### 5. Correlation-Based Burst Detection (Pattern Matching)

The goal is to search for cluster patterns in the **entire continuous dataset**,
independently of threshold-based detection. This validates whether cluster patterns
actually appear in the raw data and measures their occurrence rates.

```python
# PREREQUISITE: You must have cluster templates from clustering
# cluster_averages = result from consensus alignment or cluster averaging

# Load continuous EEG data (entire recording, not just detected avalanches)
continuous_data = raw_250.get_data()  # Shape: (n_electrodes, n_samples)

# Build spatial smoothing matrix from electrode positions
positions = stppy.io.load_electrode_positions('electrode_positions.json')
position_array = np.array([positions[ch] for ch in sorted(positions.keys())])

W_smooth, distances, mean_dist = stppy.detection.build_spatial_smoothing_matrix(
    position_array,
    k=3.0  # Spatial decay parameter (higher = less smoothing)
)

# For each cluster template, detect when it appears in continuous data
for cluster_id, cluster_template in cluster_averages.items():
    burst_info = stppy.detection.detect_bursts_by_correlation(
        continuous_data,      # ENTIRE continuous recording
        cluster_template,     # Template from clustering
        W_smooth,
        threshold=0.7,        # Correlation threshold
        sfreq=250.0,
        return_correlations=True
    )

    print(f"Cluster {cluster_id}:")
    print(f"  Positive bursts: {burst_info['n_positive']}")
    print(f"  Negative bursts: {burst_info['n_negative']}")
    print(f"  Rate: {burst_info['rate_positive']:.2f} bursts/sec")

    # Compute detailed temporal statistics
    if len(burst_info['correlations']) > 0:
        stats = stppy.detection.compute_temporal_statistics(
            burst_info['correlations'],
            burst_info['positions'],
            threshold=0.7,
            sfreq=250.0
        )
        print(f"  Mean inter-burst interval: {stats['mean_lag_above']:.2f} sec")
```

**Key Point:** This detects cluster patterns ANYWHERE in the continuous data,
not just at threshold-detected avalanche times. It's independent validation
that the cluster patterns are real phenomena in your recordings.

See `examples/detect_bursts_correlation.py` for complete workflow.

### 6. Visualize Spatiotemporal Patterns

```python
# Plot cluster spatiotemporal pattern (consensus average)
fig = stppy.visualization.plot_cluster_average(
    average,
    'electrode_positions.json',
    cluster_id=0
)

# Plot single binary avalanches
fig = stppy.visualization.plot_avalanche_topomaps(
    avalanches,
    'electrode_positions.json',
    indices=[0, 5, 10]
)

# Create video from cluster spatiotemporal pattern
stppy.visualization.create_cluster_video(
    average,
    'electrode_positions.json',
    output_file='cluster_0.mp4',
    cluster_id=0,
    fps=5
)

# Create video from single avalanche
stppy.visualization.create_avalanche_video(
    avalanche_arrays[0],
    'electrode_positions.json',
    output_file='avalanche_0.mp4'
)

# Combine multiple cluster videos
stppy.visualization.combine_videos(
    ['cluster_0.mp4', 'cluster_1.mp4', 'cluster_2.mp4'],
    output_file='all_clusters.mp4',
    titles=['Cluster 0', 'Cluster 1', 'Cluster 2']
)
```

### 7. Universal Metadata System

```python
# Generic metadata (user defines any fields)
dataset_metadata = [
    {'subject': 'S01', 'condition': 'rest', 'age': 25, 'session': 1},
    {'subject': 'S02', 'condition': 'task', 'age': 30, 'session': 1},
    {'subject': 'S01', 'condition': 'rest', 'age': 25, 'session': 2}
]

# Create universal index (links avalanches to datasets and times)
index = stppy.analysis.create_avalanche_index(
    all_avalanches,
    all_avalanche_times,
    dataset_metadata
)

# Query avalanches by any field
s01_avalanches = stppy.analysis.query_avalanches(
    index,
    field='subject',
    value='S01'
)

# Complex queries with custom functions
rest_young = stppy.analysis.query_avalanches(
    index,
    filter_func=lambda x: x['metadata']['condition'] == 'rest' and
                          x['metadata']['age'] < 30
)

# Group by any field
groups = stppy.analysis.group_by_metadata_field(index, 'condition')
# Result: {'rest': [0, 2, 5, ...], 'task': [1, 3, 4, ...]}
```

## Package Structure

```
stppy/
├── io/              # Data loading and saving
├── preprocessing/   # Interpolation and resampling
├── detection/       # Avalanche detection and filtering
├── alignment/       # Spatial and temporal alignment
├── clustering/      # Hierarchical clustering and consensus
├── analysis/        # Metadata and statistical analysis
├── visualization/   # Topomap plotting
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
[Your citation here]
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please contact: novicky@cs.cas.cz
