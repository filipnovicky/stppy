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

**Important:** Run these commands in your **terminal/command line**, NOT in a Python interpreter or Jupyter notebook.

```bash
git clone https://github.com/filipnovicky/stppy.git
cd stppy
pip install -e .
```

**If you're using Jupyter notebook/lab:**
```python
# Option 1: Run with ! prefix in a Jupyter cell
!git clone https://github.com/filipnovicky/stppy.git
%cd stppy
!pip install -e .
```

```python
# Option 2: Exit Python/Jupyter and run in terminal
# Press Ctrl+D or type exit() to leave Python, then run the bash commands above
```

### Using pip (when published)

```bash
pip install stppy
```

### Troubleshooting

**Error: `SyntaxError: invalid syntax` when running installation commands**

This means you're trying to run shell commands inside Python. These are bash/terminal commands, not Python code.

Solutions:
1. **Exit Python first**: If you're in a Python interpreter (prompt shows `>>>`), type `exit()` or press `Ctrl+D`, then run the commands in your terminal
2. **Use Jupyter magic**: If you're in Jupyter, prefix commands with `!` (e.g., `!pip install -e .`)
3. **Open a terminal**: Use a proper terminal/command prompt window instead of Python

## Quick Start

**Prerequisites:** This package assumes your EEG data has been preprocessed (artifact removal, ICA, bandpass filtering, etc.) before analysis.

### 1. Load and Preprocess EEG Data

```python
import stppy
import mne

# Find all EEG files (recursively searches directories)
files = stppy.io.find_eeg_files('/path/to/yourdata')

# Create the positions file (use a file with all electrodes present)
# Note: Some files may have electrodes removed, so find one with complete montage
raw_complete = stppy.io.load_eeg_raw(files[0], preload=True)  # Check this has all electrodes
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

    # Resample to 250 Hz
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
```

### 3. Compute Similarity Matrix (All Avalanches)

#### Option A: Serial Computation (Small Datasets < 5000 avalanches)

```python
# Load avalanche data (include flipped versions to double dataset size)
avalanches = stppy.io.load_avalanche_data(
    './peak_avalanches.pkl',
    min_time_length=11,
    include_flipped=True  # Adds polarity-flipped versions (positive ↔ negative)
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
stppy.io.save_pickle(block_0, './blocks/block_000.pkl')

# Job 1:
block_1 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=500, row_end=1000,
    col_start=0, col_end=500,
    is_diagonal=False
)
stppy.io.save_pickle(block_1, './blocks/block_001.pkl')
# ... (repeat for all blocks in parallel)

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
    threshold=3.5,
    method='average',
    min_cluster_size=30
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
# Output: "Removing cluster 5 (duplicate polarity of cluster 2, corr=-0.9998)"
#         "Keeping: 8 unique clusters"

# Analyze cluster composition (which datasets contribute avalanches to clusters)
dataset_names = [os.path.basename(f) for f in files]  # ['S01_rest.set', 'S02_task.set', ...]

avalanche_index = stppy.analysis.create_avalanche_index(
    all_avalanches,        # From Step 2
    all_avalanche_times,   # From Step 2
    dataset_names          # Simple filenames
)

composition = stppy.analysis.analyze_cluster_composition(
    clusters,
    avalanche_index
)
# Output: "Cluster 0: 150 avalanches"
#         "  Datasets: S01_rest.set: 90, S02_task.set: 60"
```

### 5. Correlation-Based Burst Detection (Pattern Matching)

The goal is to search for cluster patterns in the **entire continuous dataset**,
independently of threshold-based detection. This validates whether cluster patterns
actually appear in the raw data and measures their occurrence rates.

```python
# Build spatial smoothing matrix from electrode positions
positions = stppy.io.load_electrode_positions('./electrode_positions.json')
position_array = np.array([positions[ch] for ch in sorted(positions.keys())])

W_smooth, distances, mean_dist = stppy.detection.build_spatial_smoothing_matrix(
    position_array,
    k=3.0  # Spatial decay parameter (higher = less smoothing)
)

# Detect bursts for each unique cluster (from clusters_to_keep in Step 4)
for cluster_id in clusters_to_keep:
    cluster_template = cluster_averages[cluster_id]
    print(f"\n=== Analyzing Cluster {cluster_id} ===")

    burst_stats = []
    for dataset_idx, continuous_data in enumerate(datasets):
        # Detect when cluster template appears in continuous data
        burst_info = stppy.detection.detect_bursts_by_correlation(
            continuous_data,      # ENTIRE continuous recording
            cluster_template,     # Template from Step 4
            W_smooth,
            threshold=0.7,        # Correlation threshold
            sfreq=250.0,
            return_correlations=True
        )

        # Collect statistics with dataset name
        burst_stats.append({
            'dataset_name': dataset_names[dataset_idx],
            'n_positive': burst_info['n_positive'],
            'n_negative': burst_info['n_negative'],
            'rate': burst_info['rate_positive']
        })

    # Summarize burst statistics by dataset
    summary = stppy.analysis.summarize_burst_statistics(burst_stats)
    # Output: "S01_rest.set: 10 bursts (rate: 0.50 bursts/sec)"
    #         "S02_task.set: 8 bursts (rate: 0.30 bursts/sec)"
```

**Key Point:** This detects cluster patterns ANYWHERE in the continuous data,
not just at threshold-detected avalanche times. It's independent validation
that the cluster patterns are real phenomena in your recordings.

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

# Combine multiple cluster videos
stppy.visualization.combine_videos(
    ['./cluster_0.mp4', './cluster_1.mp4', './cluster_2.mp4'],
    output_file='./all_clusters.mp4',
    titles=['Cluster 0', 'Cluster 1', 'Cluster 2']
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
