# STPPY - Spatiotemporal Pattern Processing for Python

A Python package for analyzing EEG avalanches with spatial and temporal alignment methods.

## Features

- **Data I/O**: Load EEG data in various formats (.set, pickle) with flexible directory searching
- **Preprocessing**: Electrode interpolation and resampling
- **Avalanche Detection**: Threshold-based detection with spatial filtering
- **Spatial Alignment**: Optimal transport for spatial pattern matching
- **Temporal Alignment**: Dynamic Time Warping (DTW) for temporal alignment
- **Clustering**: Hierarchical clustering with consensus building
- **Visualization**:
  - Topomap plotting using MNE
  - Video creation from avalanche sequences
  - Timeline plots with cluster overlays
  - Correlation matrices between clusters
  - Raster plots of avalanche occurrences
- **Analysis**:
  - Universal metadata system (user-defined fields)
  - Flexible querying and grouping
  - Statistical analysis

## Installation

### From source

```bash
git clone https://github.com/yourusername/stppy.git
cd stppy
pip install -e .
```

### With video support (optional)

```bash
pip install -e ".[video]"
```

### Using pip (when published)

```bash
# Basic installation
pip install stppy

# With video support
pip install stppy[video]
```

## Quick Start

### 1. Load and Preprocess EEG Data

```python
import stppy

# NEW: Simple universal file loading (recursively finds all .set files)
files = stppy.io.find_eeg_files([
    '/path/to/control/data',
    '/path/to/drug/data'
])

# OLD: Specific directory structure (still supported)
base_dirs = ['/path/to/Control', '/path/to/Psilocybin']
conditions = ['Meditation', 'Music', 'Resting state', 'Video']
files = stppy.io.load_eeg_files(base_dirs, conditions)

# Load a single file
raw = stppy.io.load_eeg_raw(files[0], preload=True)

# Interpolate missing electrodes
raw_interp, interp_channels = stppy.preprocessing.interpolate_missing_electrodes(
    raw,
    positions_file='electrode_positions.json'
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

### 3. Compute Similarity and Cluster

```python
from stppy.alignment import spatial, temporal

# Load avalanche data
avalanches = stppy.io.load_avalanche_data('avalanches.pkl', min_time_length=11)

# Create distance matrix
K = stppy.utils.create_eeg_ground_metric('electrode_positions.json')[0]
K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

# Compute similarity between two avalanches
sim_matrix = spatial.create_temporal_similarity_matrix(
    avalanches[0], avalanches[1], K_norm
)

# Apply DTW
dtw_cost, paths, transposed = temporal.dtw_on_similarity_matrix(sim_matrix)

# Cluster avalanches
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

### 4. Visualize Results

```python
# Plot individual avalanches
fig = stppy.visualization.plot_avalanche_topomaps(
    avalanches,
    'electrode_positions.json',
    indices=[0, 5, 10]
)

# Plot cluster average
fig = stppy.visualization.plot_cluster_average(
    average,
    'electrode_positions.json',
    cluster_id=0
)

# NEW: Plot avalanche timeline (raw data with cluster overlays)
fig = stppy.visualization.plot_avalanche_timeline(
    raw_data,
    avalanche_times,
    cluster_assignments,
    channel_idx=0,
    title="Avalanches Over Time"
)

# Plot correlation matrix between clusters
fig = stppy.visualization.plot_cluster_correlations(
    cluster_averages_dict,
    method='pearson'
)

# Create raster plot
fig = stppy.visualization.plot_avalanche_raster(
    avalanche_times,
    cluster_assignments
)
```

### 5. Create Videos (Optional - requires opencv-python)

```python
# Create video from cluster consensus average
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

# Combine multiple videos with titles
stppy.visualization.combine_videos(
    ['cluster_0.mp4', 'cluster_1.mp4', 'cluster_2.mp4'],
    output_file='all_clusters.mp4',
    titles=['Cluster 0', 'Cluster 1', 'Cluster 2']
)
```

### 6. Universal Metadata System

```python
# NEW: Generic metadata (user defines any fields)
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

### 7. Legacy Metadata (Specific Fields - Still Supported)

```python
# Link avalanches to metadata
links = stppy.analysis.link_avalanches_to_datasets(
    all_avalanches,
    all_avalanche_times,
    metadata
)

# Analyze cluster composition
composition = stppy.analysis.analyze_clusters_by_avalanche_metadata(
    clusters,
    links,
    metadata
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

For questions or issues, please contact: your.email@example.com
