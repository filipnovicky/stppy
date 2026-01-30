#!/usr/bin/env python3
"""
Full EEG analysis pipeline: preprocessing, correlation detection, and statistics.

Uses stppy library for optimized spatiotemporal correlation detection.
Pre-smoothing the data once and using vectorized frame correlations gives
a large speedup over the naive per-window approach.

Usage:
    python run_analysis.py                # Run both preprocessing and analysis
    python run_analysis.py --preprocess   # Step 1 only: load .set files, interpolate, save
    python run_analysis.py --analyze      # Step 2 only: load saved data, run correlations
"""

import argparse
import os
import numpy as np
from collections import defaultdict

import stppy


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIRS = [
    '/home/novicky/Documents/psiconnect/Control',
    '/home/novicky/Documents/psiconnect/Psilocybin'
]
CONDITIONS = ['Meditation', 'Music', 'Resting state', 'Video']
POSITIONS_FILE = '/home/novicky/Documents/stppy-work/electrode_positions.json'
OUTPUT_DIR = '/home/novicky/Documents/stppy-work'

# Analysis parameters
K_VALUES = [10]
SFREQ = 250.0
CO_ACTIVATION_THRESHOLD = 0.7
CO_ACTIVATION_WINDOW = 0.05   # 50 ms
LEAD_LAG_MAX_SEC = 5.0


# ============================================================================
# STEP 1: PREPROCESSING
# ============================================================================

def preprocess():
    """Load .set files, interpolate missing electrodes, resample, save."""

    file_paths = stppy.io.load_eeg_files(DATA_DIRS, CONDITIONS)

    datasets = []
    for i, file_path in enumerate(file_paths):
        print(f"Processing {i+1}/{len(file_paths)}: {os.path.basename(file_path)}")

        raw = stppy.io.load_eeg_raw(file_path, preload=True)

        raw_interp, _ = stppy.preprocessing.interpolate_missing_electrodes(
            raw,
            positions_file=POSITIONS_FILE,
            verbose=False
        )

        raw_resampled = stppy.preprocessing.resample_data(raw_interp, int(SFREQ))
        datasets.append(raw_resampled.get_data())

    # Parse metadata from file paths
    file_metadata = stppy.analysis.parse_dataset_metadata(file_paths)

    # Save
    stppy.io.save_pickle(datasets, os.path.join(OUTPUT_DIR, 'datasets.pkl'))
    stppy.io.save_pickle(file_metadata, os.path.join(OUTPUT_DIR, 'metadata.pkl'))
    print(f"Saved {len(datasets)} datasets and metadata")


# ============================================================================
# STEP 2: CORRELATION ANALYSIS
# ============================================================================

def run_analysis_for_k(k_value):
    """
    Run the full analysis pipeline for a specific k value.
    NO correlation threshold applied -- all correlations are saved.

    Uses stppy.detection.detect_bursts_by_correlation with return_all=True
    which pre-smooths the data once (W_smooth @ data) before the sliding
    window loop, giving a large speedup over the naive approach.

    Parameters
    ----------
    k_value : float
        Spatial decay parameter for smoothing

    Returns
    -------
    results : dict
        Complete results dictionary
    """

    print(f"\n{'='*80}")
    print(f"RUNNING ANALYSIS WITH k={k_value} (NO CORRELATION THRESHOLD)")
    print(f"{'='*80}")

    # Load pre-saved data
    print("Loading datasets...")
    datasets = stppy.io.load_pickle(os.path.join(OUTPUT_DIR, 'datasets.pkl'))

    print("Loading metadata...")
    file_metadata = stppy.io.load_pickle(os.path.join(OUTPUT_DIR, 'metadata.pkl'))

    print("Loading cluster averages...")
    cluster_averages = stppy.io.load_pickle(
        os.path.join(OUTPUT_DIR, 'cluster_averages_kept.pkl')
    )

    print("Loading electrode positions...")
    positions_dict = stppy.io.load_electrode_positions(POSITIONS_FILE)
    electrode_positions = np.array([
        [positions_dict[ch]['x'], positions_dict[ch]['y'], positions_dict[ch]['z']]
        for ch in sorted(positions_dict.keys())
    ])

    print(f"\nLoaded {len(datasets)} datasets")
    print(f"Loaded {len(cluster_averages)} clusters")

    # Build spatial smoothing matrix with specified k
    print(f"\nBuilding spatial smoothing matrix with k={k_value}...")
    W_smooth, distances, mean_distance = stppy.detection.build_spatial_smoothing_matrix(
        electrode_positions, k=k_value
    )

    print(f"Mean electrode distance: {mean_distance:.2f} mm")
    print(f"Smoothing matrix built: {W_smooth.shape}")

    # Use the actual number of clusters available
    top_clusters = len(cluster_averages)
    print(f"\nAnalyzing {top_clusters} clusters with spatiotemporal correlation")
    print("NOTE: Saving ALL correlations (no threshold)")

    # Track cluster-specific statistics
    cluster_stats = {}
    for cluster_id in range(top_clusters):
        cluster_stats[cluster_id] = {
            'drug': {'Control': [], 'Psilocybin': []},
            'task': {'Meditation': [], 'Music': [], 'Resting state': [], 'Video': []},
            'training': {'Trained': [], 'Non-trained': []}
        }

    # Store ALL sliding window correlations
    cluster_activation_times = {}
    for cluster_id in range(top_clusters):
        cluster_activation_times[cluster_id] = []

    # Co-activation and lead-lag matrices
    co_activation_matrix = np.zeros((top_clusters, top_clusters))
    lead_lag_matrix = np.zeros((top_clusters, top_clusters))
    lead_lag_counts = np.zeros((top_clusters, top_clusters))

    # Store all intervals for each cluster
    cluster_intervals = {}
    for cluster_id in range(top_clusters):
        cluster_intervals[cluster_id] = []

    print(f"\nAnalyzing datasets against clusters (k={k_value}, saving ALL correlations)...")
    for dataset_idx, (data, metadata) in enumerate(zip(datasets, file_metadata)):
        if dataset_idx % 10 == 0:
            print(f"  Processing dataset {dataset_idx}/{len(datasets)}...")

        dataset_activations = []  # (cluster_id, time_sec) for co-activation

        for cluster_id in range(top_clusters):
            cluster_pattern = cluster_averages[cluster_id]

            dataset_channels, dataset_timepoints = data.shape
            cluster_channels, cluster_timepoints = cluster_pattern.shape

            if dataset_channels != cluster_channels:
                continue

            # ----------------------------------------------------------
            # OPTIMIZED: uses pre-smoothing (W_smooth @ data computed
            # once) and vectorized frame correlations. step=1 gives one
            # correlation value per sample position.
            # ----------------------------------------------------------
            result = stppy.detection.detect_bursts_by_correlation(
                data,
                cluster_pattern,
                W_smooth,
                sfreq=SFREQ,
                return_all=True,   # all correlations, no threshold
                use_optimized=True
            )

            correlations = result['correlations']
            times_sec = result['times']
            polarities = result['polarities']

            if len(times_sec) > 0:
                # Store ALL activation times with their correlations
                cluster_activation_times[cluster_id].append((
                    dataset_idx,
                    times_sec,
                    polarities,
                    correlations
                ))

                # For co-activation analysis, use positive correlations > threshold
                positive_mask = correlations > CO_ACTIVATION_THRESHOLD
                positive_times_sec = times_sec[positive_mask]
                for time_sec in positive_times_sec:
                    dataset_activations.append((cluster_id, time_sec))

                # Calculate intervals between ALL timepoints
                if len(times_sec) > 1:
                    intervals = np.diff(times_sec)
                    cluster_intervals[cluster_id].extend(intervals)

            # Count positive activations for statistics
            positive_mask = correlations > CO_ACTIVATION_THRESHOLD
            activations = np.sum(positive_mask)

            # Normalize by dataset duration (activations per minute)
            dataset_duration_minutes = dataset_timepoints / (SFREQ * 60.0)
            normalized_activations = (
                activations / dataset_duration_minutes
                if dataset_duration_minutes > 0 else 0
            )

            # Store by condition
            cluster_stats[cluster_id]['drug'][metadata['drug_condition']].append(
                normalized_activations
            )
            if metadata['task']:
                cluster_stats[cluster_id]['task'][metadata['task']].append(
                    normalized_activations
                )
            if metadata['is_trained'] is not None:
                training_key = 'Trained' if metadata['is_trained'] else 'Non-trained'
                cluster_stats[cluster_id]['training'][training_key].append(
                    normalized_activations
                )

        # Co-activation and lead-lag analysis for this dataset
        if len(dataset_activations) > 1:
            dataset_activations.sort(key=lambda x: x[1])

            # Co-activation (50 ms window)
            for i, (cluster_a, time_a) in enumerate(dataset_activations):
                for j, (cluster_b, time_b) in enumerate(dataset_activations):
                    if i != j and cluster_a != cluster_b:
                        if abs(time_a - time_b) <= CO_ACTIVATION_WINDOW:
                            co_activation_matrix[cluster_a, cluster_b] += 1

            # Lead-lag analysis
            for i, (cluster_a, time_a) in enumerate(dataset_activations):
                for j, (cluster_b, time_b) in enumerate(dataset_activations):
                    if cluster_a != cluster_b:
                        time_diff = time_b - time_a
                        if 0 < time_diff <= LEAD_LAG_MAX_SEC:
                            lead_lag_matrix[cluster_a, cluster_b] += time_diff
                            lead_lag_counts[cluster_a, cluster_b] += 1

    print("Computing statistics...")

    # Average lead-lag times
    average_lead_lag_matrix = np.zeros((top_clusters, top_clusters))
    for i in range(top_clusters):
        for j in range(top_clusters):
            if lead_lag_counts[i, j] > 0:
                average_lead_lag_matrix[i, j] = (
                    lead_lag_matrix[i, j] / lead_lag_counts[i, j]
                )

    # Normalize co-activation matrix
    total_activations_per_cluster = np.zeros(top_clusters)
    for cluster_id in range(top_clusters):
        for _, times, _, _ in cluster_activation_times[cluster_id]:
            total_activations_per_cluster[cluster_id] += len(times)

    co_activation_probability = np.zeros((top_clusters, top_clusters))
    for i in range(top_clusters):
        for j in range(top_clusters):
            if i != j and total_activations_per_cluster[i] > 0:
                co_activation_probability[i, j] = (
                    co_activation_matrix[i, j] / total_activations_per_cluster[i]
                )

    # Polarity switching
    print("Analyzing polarity switches...")
    polarity_switch_stats = []

    for cluster_id in range(top_clusters):
        all_switches = []
        total_positive = 0
        total_negative = 0

        for dataset_idx, times, polarities, correlations in cluster_activation_times[cluster_id]:
            total_positive += np.sum(polarities > 0)
            total_negative += np.sum(polarities < 0)

            if len(polarities) > 1:
                polarity_diffs = np.diff(polarities)
                switch_indices = np.where(polarity_diffs != 0)[0]

                if len(switch_indices) > 0:
                    switch_times = times[switch_indices]
                    if len(switch_times) > 1:
                        switch_intervals = np.diff(switch_times)
                        all_switches.extend(switch_intervals)

        if len(all_switches) > 0:
            all_switches = np.array(all_switches)
            mean_interval = np.mean(all_switches)
            std_interval = np.std(all_switches)
            cv = std_interval / mean_interval if mean_interval > 0 else np.inf

            polarity_switch_stats.append({
                'cluster_id': cluster_id,
                'n_switches': len(all_switches),
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'cv': cv,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'polarity_ratio': (
                    total_positive / total_negative if total_negative > 0 else np.inf
                )
            })

    polarity_switch_stats_sorted = sorted(
        polarity_switch_stats, key=lambda x: x['n_switches'], reverse=True
    )

    regular_switchers = sorted(
        [s for s in polarity_switch_stats if s['cv'] < 1.0 and s['n_switches'] >= 10],
        key=lambda x: x['cv']
    )[:10]

    # Heatmap matrices
    drug_matrix = np.zeros((top_clusters, 2))
    drug_labels = ['Control', 'Psilocybin']
    task_matrix = np.zeros((top_clusters, 4))
    task_labels = ['Meditation', 'Music', 'Resting state', 'Video']
    training_matrix = np.zeros((top_clusters, 2))
    training_labels = ['Non-trained', 'Trained']

    for cluster_id in range(top_clusters):
        for i, drug in enumerate(['Control', 'Psilocybin']):
            vals = cluster_stats[cluster_id]['drug'][drug]
            if vals:
                drug_matrix[cluster_id, i] = np.mean(vals)

        for i, task in enumerate(['Meditation', 'Music', 'Resting state', 'Video']):
            vals = cluster_stats[cluster_id]['task'][task]
            if vals:
                task_matrix[cluster_id, i] = np.mean(vals)

        for i, training in enumerate(['Non-trained', 'Trained']):
            vals = cluster_stats[cluster_id]['training'][training]
            if vals:
                training_matrix[cluster_id, i] = np.mean(vals)

    # Drug preference ratios
    drug_ratios = []
    for cluster_id in range(top_clusters):
        control_rate = drug_matrix[cluster_id, 0]
        psilocybin_rate = drug_matrix[cluster_id, 1]
        if control_rate > 0 and psilocybin_rate > 0:
            ratio = psilocybin_rate / control_rate
            drug_ratios.append((cluster_id, ratio, psilocybin_rate, control_rate))

    psilocybin_preferred = sorted(
        [x for x in drug_ratios if x[1] > 1.2], key=lambda x: x[1], reverse=True
    )[:10]
    control_preferred = sorted(
        [x for x in drug_ratios if x[1] < 0.8], key=lambda x: x[1]
    )[:10]

    # Most active clusters
    overall_activity = []
    for cluster_id in range(top_clusters):
        mean_activity = np.mean([
            drug_matrix[cluster_id, 0], drug_matrix[cluster_id, 1]
        ])
        overall_activity.append((cluster_id, mean_activity))

    most_active = sorted(
        overall_activity, key=lambda x: x[1], reverse=True
    )[:min(15, top_clusters)]

    # Interval statistics
    interval_stats = []
    for cluster_id in range(top_clusters):
        if len(cluster_intervals[cluster_id]) > 2:
            intervals = np.array(cluster_intervals[cluster_id])
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            cv = std_interval / mean_interval if mean_interval > 0 else np.inf
            interval_stats.append(
                (cluster_id, mean_interval, std_interval, cv, len(intervals))
            )

    regular_clusters = sorted(
        [x for x in interval_stats if x[3] < 1.0], key=lambda x: x[3]
    )[:10]

    # Top co-activating pairs
    co_activation_pairs = []
    for i in range(top_clusters):
        for j in range(i + 1, top_clusters):
            prob = (
                co_activation_probability[i, j] + co_activation_probability[j, i]
            ) / 2
            if prob > 0.01:
                co_activation_pairs.append((
                    i, j, prob,
                    co_activation_matrix[i, j] + co_activation_matrix[j, i]
                ))

    top_coactivations = sorted(
        co_activation_pairs, key=lambda x: x[2], reverse=True
    )[:10]

    # Top lead-lag relationships
    lead_lag_relationships = []
    for i in range(top_clusters):
        for j in range(top_clusters):
            if i != j and lead_lag_counts[i, j] >= 5:
                avg_lag = average_lead_lag_matrix[i, j]
                count = lead_lag_counts[i, j]
                lead_lag_relationships.append((i, j, avg_lag, count))

    strong_relationships = sorted(
        lead_lag_relationships, key=lambda x: x[3], reverse=True
    )[:10]

    # Assemble results
    results = {
        'k_value': k_value,
        'correlation_threshold': None,
        'cluster_stats': cluster_stats,
        'drug_matrix': drug_matrix,
        'task_matrix': task_matrix,
        'training_matrix': training_matrix,
        'drug_labels': drug_labels,
        'task_labels': task_labels,
        'training_labels': training_labels,
        'psilocybin_preferred': psilocybin_preferred,
        'control_preferred': control_preferred,
        'most_active': most_active,
        'drug_ratios': drug_ratios,
        'overall_activity': overall_activity,
        # Temporal analysis
        'cluster_intervals': cluster_intervals,
        'cluster_activation_times': cluster_activation_times,
        'co_activation_matrix': co_activation_matrix,
        'co_activation_probability': co_activation_probability,
        'lead_lag_matrix': lead_lag_matrix,
        'average_lead_lag_matrix': average_lead_lag_matrix,
        'lead_lag_counts': lead_lag_counts,
        'interval_stats': interval_stats,
        'regular_clusters': regular_clusters,
        'top_coactivations': top_coactivations,
        'strong_relationships': strong_relationships,
        # Polarity switching
        'polarity_switch_stats': polarity_switch_stats,
        'regular_switchers': regular_switchers,
        # Spatial info
        'W_smooth': W_smooth,
        'electrode_positions': electrode_positions,
        'mean_distance': mean_distance,
        'spatial_k': k_value
    }

    print(f"\nAnalysis complete for k={k_value}")
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EEG correlation analysis pipeline')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing only (load .set, interpolate, save)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis only (load saved data, compute correlations)')
    args = parser.parse_args()

    # Default: run both
    run_preprocess = args.preprocess or (not args.preprocess and not args.analyze)
    run_analyze = args.analyze or (not args.preprocess and not args.analyze)

    if run_preprocess:
        print("=" * 80)
        print("STEP 1: PREPROCESSING")
        print("=" * 80)
        preprocess()

    if run_analyze:
        print("=" * 80)
        print("STEP 2: CORRELATION ANALYSIS")
        print("NO CORRELATION THRESHOLD - ALL CORRELATIONS SAVED")
        print("=" * 80)

        print(f"\nTesting {len(K_VALUES)} k values: {K_VALUES}")

        all_results = {}
        for k in K_VALUES:
            config_name = f"k{k}_no_threshold"
            print(f"\n{'='*80}")
            print(f"Configuration: {config_name}")
            print(f"{'='*80}")

            results = run_analysis_for_k(k_value=k)
            all_results[config_name] = results

            output_file = os.path.join(OUTPUT_DIR, f'cluster_analysis_{config_name}.pkl')
            stppy.io.save_pickle(results, output_file)
            print(f"Saved to: {output_file}")

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
