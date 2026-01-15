#!/usr/bin/env python3
"""
Example: Detect recurring patterns using correlation-based burst detection.

This approach is complementary to threshold-based detection:
- Threshold-based: Find avalanches de novo based on signal strength
- Correlation-based: Find when cluster patterns recur in continuous data

TYPICAL WORKFLOW:
-----------------
1. Use threshold-based detection to find initial avalanches
2. Cluster avalanches to identify distinct patterns
3. Use correlation-based detection to find when these patterns recur
4. Analyze burst statistics (rates, intervals, polarity switching)

USAGE:
------
python detect_bursts_correlation.py
"""

import stppy
import numpy as np
import matplotlib.pyplot as plt

# Configuration
CLUSTER_AVERAGES_FILE = 'cluster_averages.pkl'  # From clustering
CONTINUOUS_DATA_FILE = 'continuous_eeg.pkl'  # Continuous recording
ELECTRODE_POSITIONS = 'electrode_positions.json'
CORRELATION_THRESHOLD = 0.7  # Correlation threshold for burst detection
SPATIAL_SMOOTHING_K = 3.0  # Spatial smoothing parameter (higher = less smoothing)
SFREQ = 250.0  # Sampling frequency

print("=" * 70)
print("CORRELATION-BASED BURST DETECTION")
print("=" * 70)

# Step 1: Load data
print("\n1. Loading data...")
cluster_averages = stppy.io.load_pickle(CLUSTER_AVERAGES_FILE)
print(f"   Loaded {len(cluster_averages)} cluster templates")

continuous_data = stppy.io.load_pickle(CONTINUOUS_DATA_FILE)
n_electrodes, n_samples = continuous_data.shape
print(f"   Loaded continuous data: {n_electrodes} channels, {n_samples} samples")
print(f"   Duration: {n_samples/SFREQ:.1f} seconds")

# Step 2: Build spatial smoothing matrix
print("\n2. Building spatial smoothing matrix...")
positions = stppy.io.load_electrode_positions(ELECTRODE_POSITIONS)
position_array = np.array([positions[ch] for ch in sorted(positions.keys())])

W_smooth, distances, mean_distance = stppy.detection.build_spatial_smoothing_matrix(
    position_array,
    k=SPATIAL_SMOOTHING_K
)
print(f"   Matrix shape: {W_smooth.shape}")
print(f"   Mean electrode distance: {mean_distance:.2f}")
print(f"   Spatial decay parameter (k): {SPATIAL_SMOOTHING_K}")

# Step 3: Detect bursts for each cluster pattern
print("\n3. Detecting bursts for each cluster pattern...")
print(f"   Correlation threshold: ±{CORRELATION_THRESHOLD}")
print()

all_burst_results = {}

for cluster_id, pattern in cluster_averages.items():
    print(f"   Cluster {cluster_id}:")
    print(f"     Pattern shape: {pattern.shape}")

    # Detect bursts by correlation
    burst_info = stppy.detection.detect_bursts_by_correlation(
        continuous_data,
        pattern,
        W_smooth,
        threshold=CORRELATION_THRESHOLD,
        sfreq=SFREQ,
        return_correlations=True
    )

    all_burst_results[cluster_id] = burst_info

    print(f"     Positive bursts: {burst_info['n_positive']} "
          f"({burst_info['rate_positive']:.2f}/sec)")
    print(f"     Negative bursts: {burst_info['n_negative']} "
          f"({burst_info['rate_negative']:.2f}/sec)")

# Step 4: Compute detailed temporal statistics
print("\n4. Computing temporal statistics...")

for cluster_id, burst_info in all_burst_results.items():
    if len(burst_info['correlations']) == 0:
        continue

    stats = stppy.detection.compute_temporal_statistics(
        burst_info['correlations'],
        burst_info['positions'],
        threshold=CORRELATION_THRESHOLD,
        sfreq=SFREQ
    )

    print(f"\n   Cluster {cluster_id}:")

    if stats['n_above_threshold'] > 0:
        print(f"     Positive bursts:")
        print(f"       Count: {stats['n_above_threshold']}")
        print(f"       Rate: {stats['rate_above']:.3f}/sec")
        if not np.isnan(stats['mean_lag_above']):
            print(f"       Mean interval: {stats['mean_lag_above']:.2f} ± "
                  f"{stats['std_lag_above']:.2f} sec")
            print(f"       Median interval: {stats['median_lag_above']:.2f} sec")

    if stats['n_below_threshold'] > 0:
        print(f"     Negative bursts:")
        print(f"       Count: {stats['n_below_threshold']}")
        print(f"       Rate: {stats['rate_below']:.3f}/sec")
        if not np.isnan(stats['mean_lag_below']):
            print(f"       Mean interval: {stats['mean_lag_below']:.2f} ± "
                  f"{stats['std_lag_below']:.2f} sec")
            print(f"       Median interval: {stats['median_lag_below']:.2f} sec")

# Step 5: Visualize correlation timeseries
print("\n5. Creating visualization...")

fig, axes = plt.subplots(len(all_burst_results), 1, figsize=(15, 3*len(all_burst_results)))
if len(all_burst_results) == 1:
    axes = [axes]

for idx, (cluster_id, burst_info) in enumerate(all_burst_results.items()):
    ax = axes[idx]

    # Plot correlation timeseries
    times = burst_info['positions'] / SFREQ
    correlations = burst_info['correlations']

    ax.plot(times, correlations, 'k-', alpha=0.5, linewidth=0.5)

    # Highlight positive bursts
    if burst_info['n_positive'] > 0:
        for start, end, peak_corr in burst_info['positive_bursts']:
            ax.axvspan(start/SFREQ, end/SFREQ, alpha=0.3, color='red')

    # Highlight negative bursts
    if burst_info['n_negative'] > 0:
        for start, end, peak_corr in burst_info['negative_bursts']:
            ax.axvspan(start/SFREQ, end/SFREQ, alpha=0.3, color='blue')

    # Threshold lines
    ax.axhline(CORRELATION_THRESHOLD, color='red', linestyle='--',
               alpha=0.7, label=f'Threshold: ±{CORRELATION_THRESHOLD}')
    ax.axhline(-CORRELATION_THRESHOLD, color='red', linestyle='--', alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    ax.set_ylabel(f'Cluster {cluster_id}\nCorrelation')
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if idx == len(all_burst_results) - 1:
        ax.set_xlabel('Time (seconds)')

plt.tight_layout()
plt.savefig('correlation_bursts.png', dpi=150, bbox_inches='tight')
print(f"   Saved: correlation_bursts.png")

# Step 6: Save results
print("\n6. Saving results...")
output_file = 'burst_detection_results.pkl'
stppy.io.save_pickle({
    'burst_results': all_burst_results,
    'parameters': {
        'correlation_threshold': CORRELATION_THRESHOLD,
        'spatial_smoothing_k': SPATIAL_SMOOTHING_K,
        'sfreq': SFREQ
    }
}, output_file)
print(f"   Saved: {output_file}")

print("\n" + "=" * 70)
print("✓ COMPLETE")
print("=" * 70)
print("\nNext steps:")
print("  - Analyze polarity switching patterns")
print("  - Compare burst rates across conditions")
print("  - Examine lead-lag relationships between clusters")
print("  - Correlate burst timing with behavioral events")
