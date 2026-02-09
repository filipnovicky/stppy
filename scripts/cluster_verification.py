#!/usr/bin/env python3
"""
Cluster Verification Analysis

Validates that avalanches constituting a cluster have higher correlation
with their own cluster than with other clusters.

Panels:
    (a) Cluster spatial correlation matrix (DTW-based correlation)
    (b) Cross-cluster detection heatmap (avg max |correlation| when detecting
        constituent avalanches)
    (c) Self-detection vs best competitor bar chart

Input files:
    - cluster_averages_kept.pkl: {cluster_id: array} for spatial correlation
    - cluster_assignments.pkl: array mapping avalanche_idx -> cluster_id
    - peak_avalanche_times.pkl: list of lists for dataset mapping
    - cluster_activation_times.pkl: {cluster_id: [(ds_idx, times, pols, corrs)]}
    - cluster_durations.pkl: {cluster_id: duration_seconds}

Output:
    - cluster_verification.pdf
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import stppy
import stppy.io

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from psiconnect_utils import BASE_DIR

# Paths
OUTPUT_FILE = Path(__file__).parent / 'cluster_verification.pdf'


def dtw_correlation(arr1, arr2):
    """
    Compute DTW-based similarity using frame-to-frame correlations.

    Args:
        arr1: shape (n_channels, T1)
        arr2: shape (n_channels, T2)

    Returns:
        Average correlation along the optimal DTW path.
    """
    T1, T2 = arr1.shape[1], arr2.shape[1]

    corr_mat = np.zeros((T1, T2))
    for i in range(T1):
        for j in range(T2):
            corr_mat[i, j] = np.corrcoef(arr1[:, i], arr2[:, j])[0, 1]

    dist_mat = 1 - np.abs(corr_mat)
    cost = np.full((T1, T2), np.inf)
    cost[0, 0] = dist_mat[0, 0]

    for i in range(T1):
        for j in range(T2):
            if i == 0 and j == 0:
                continue
            candidates = []
            if i > 0:
                candidates.append(cost[i - 1, j])
            if j > 0:
                candidates.append(cost[i, j - 1])
            if i > 0 and j > 0:
                candidates.append(cost[i - 1, j - 1])
            cost[i, j] = dist_mat[i, j] + min(candidates)

    path = [(T1 - 1, T2 - 1)]
    i, j = T1 - 1, T2 - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            candidates = [(cost[i - 1, j - 1], i - 1, j - 1),
                          (cost[i - 1, j], i - 1, j),
                          (cost[i, j - 1], i, j - 1)]
            _, i, j = min(candidates, key=lambda x: x[0])
        path.append((i, j))

    path_corrs = [np.abs(corr_mat[i, j]) for i, j in path]
    return np.mean(path_corrs)


def compute_spatial_correlation_matrix(cluster_averages):
    """
    Compute similarity between cluster average spatiotemporal patterns using DTW.
    """
    cluster_ids = sorted(cluster_averages.keys())
    n_clusters = len(cluster_ids)

    corr_matrix = np.full((n_clusters, n_clusters), np.nan)

    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            arr1 = cluster_averages[c1]
            arr2 = cluster_averages[c2]
            corr_matrix[i, j] = dtw_correlation(arr1, arr2)

    return corr_matrix, cluster_ids


def build_avalanche_to_dataset_mapping(peak_avalanche_times):
    """
    Build mapping from global avalanche index to (dataset_idx, local_start, local_end).
    """
    mapping = []
    for ds_idx, ds_avalanches in enumerate(peak_avalanche_times):
        for start, end in ds_avalanches:
            mapping.append((ds_idx, int(start), int(end)))

    mapping = mapping + mapping  # Double for ± polarity

    return mapping, len(mapping) // 2


def build_activation_lookup(cluster_activation_times):
    """
    Build a lookup structure for quick access to correlations.

    Returns:
        lookup[ds_idx][cluster_id] = list of (time_seconds, correlation)
    """
    lookup = defaultdict(lambda: defaultdict(list))

    for cluster_id, entries in cluster_activation_times.items():
        for entry in entries:
            ds_idx, times, polarities, correlations = entry
            for t, corr in zip(times, correlations):
                lookup[ds_idx][cluster_id].append((float(t), abs(float(corr))))

    return lookup


def compute_cross_detection_matrix(cluster_assignments, avalanche_to_ds, activation_lookup,
                                    cluster_ids, cluster_durations, srate=250):
    """
    Compute cross-detection matrix from constituent avalanches.
    """
    n_clusters = len(cluster_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    corr_accum = [[[] for _ in range(n_clusters)] for _ in range(n_clusters)]

    for source_cid in cluster_ids:
        source_idx = cid_to_idx[source_cid]

        avalanche_indices = np.where(cluster_assignments == source_cid)[0]
        print(f"  Cluster {source_cid}: {len(avalanche_indices)} constituent avalanches")

        for aval_idx in avalanche_indices:
            if aval_idx >= len(avalanche_to_ds):
                continue

            ds_idx, start_sample, end_sample = avalanche_to_ds[aval_idx]

            aval_start_sec = start_sample / srate
            aval_end_sec = end_sample / srate

            tolerance = 0.1  # 100ms tolerance

            for detector_cid in cluster_ids:
                detector_idx = cid_to_idx[detector_cid]

                detector_activations = activation_lookup.get(ds_idx, {}).get(detector_cid, [])

                max_corr = 0.0
                for t, corr in detector_activations:
                    if (aval_start_sec - tolerance) <= t <= (aval_end_sec + tolerance):
                        max_corr = max(max_corr, corr)

                corr_accum[source_idx][detector_idx].append(max_corr)

    cross_matrix = np.zeros((n_clusters, n_clusters))
    cross_std = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if corr_accum[i][j]:
                cross_matrix[i, j] = np.mean(corr_accum[i][j])
                cross_std[i, j] = np.std(corr_accum[i][j])

    return cross_matrix, cross_std


def main():
    print("Loading data...")
    cluster_averages = stppy.io.load_pickle(BASE_DIR / 'cluster_averages_kept.pkl')
    cluster_assignments = stppy.io.load_pickle(BASE_DIR / 'cluster_assignments.pkl')
    peak_avalanche_times = stppy.io.load_pickle(BASE_DIR / 'peak_avalanche_times.pkl')
    cluster_activation_times = stppy.io.load_pickle(BASE_DIR / 'cluster_activation_times.pkl')
    cluster_durations = stppy.io.load_pickle(BASE_DIR / 'cluster_durations.pkl')

    cluster_ids = sorted(cluster_averages.keys())
    n_clusters = len(cluster_ids)
    print(f"Found {n_clusters} clusters: {cluster_ids}")

    # Build avalanche-to-dataset mapping
    print("Building avalanche-to-dataset mapping...")
    avalanche_to_ds, n_base = build_avalanche_to_dataset_mapping(peak_avalanche_times)
    print(f"  {len(avalanche_to_ds)} avalanche mappings ({n_base} base × 2 polarities)")

    # Panel (a): Spatial correlation matrix
    print("Computing spatial correlation matrix...")
    spatial_corr, _ = compute_spatial_correlation_matrix(cluster_averages)

    # Build activation lookup
    print("Building activation lookup...")
    activation_lookup = build_activation_lookup(cluster_activation_times)

    # Panel (b) & (c): Cross-detection from constituent avalanches
    print("Computing cross-detection matrix from constituent avalanches...")
    cross_matrix, cross_std = compute_cross_detection_matrix(
        cluster_assignments, avalanche_to_ds, activation_lookup,
        cluster_ids, cluster_durations
    )

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_a, ax_b, ax_c = axes

    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 11
    LEGEND_SIZE = 11
    CBAR_SIZE = 11

    # --- Panel (a): Spatial correlation matrix ---
    im_a = ax_a.imshow(spatial_corr, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax_a.set_xticks(range(n_clusters))
    ax_a.set_yticks(range(n_clusters))
    ax_a.set_xticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax_a.set_yticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax_a.set_xlabel('Cluster ID', fontsize=LABEL_SIZE)
    ax_a.set_ylabel('Cluster ID', fontsize=LABEL_SIZE)
    ax_a.set_title('$\\mathbf{(a)}$\nCluster Spatial Correlation', fontsize=TITLE_SIZE)

    for i in range(n_clusters):
        for j in range(n_clusters):
            if not np.isnan(spatial_corr[i, j]):
                color = 'white' if abs(spatial_corr[i, j]) > 0.6 else 'black'
                ax_a.text(j, i, f'{spatial_corr[i, j]:.2f}',
                         ha='center', va='center', fontsize=7, color=color)

    cbar_a = plt.colorbar(im_a, ax=ax_a, label='Pearson r', shrink=0.8)
    cbar_a.ax.tick_params(labelsize=TICK_SIZE)
    cbar_a.set_label('Pearson r', fontsize=CBAR_SIZE)

    # --- Panel (b): Cross-detection heatmap ---
    im_b = ax_b.imshow(cross_matrix, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax_b.set_xticks(range(n_clusters))
    ax_b.set_yticks(range(n_clusters))
    ax_b.set_xticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax_b.set_yticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax_b.set_xlabel('Detector Cluster ID', fontsize=LABEL_SIZE)
    ax_b.set_ylabel('Source Cluster ID', fontsize=LABEL_SIZE)
    ax_b.set_title('$\\mathbf{(b)}$\nCross-Cluster Average Max |Correlation|', fontsize=TITLE_SIZE)

    for i in range(n_clusters):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor='blue', linewidth=2)
        ax_b.add_patch(rect)

    cbar_b = plt.colorbar(im_b, ax=ax_b, label='Avg Max |Correlation|', shrink=0.8)
    cbar_b.ax.tick_params(labelsize=TICK_SIZE)
    cbar_b.set_label('Avg Max |Correlation|', fontsize=CBAR_SIZE)

    # --- Panel (c): Self-detection vs best competitor ---
    self_detect = np.diag(cross_matrix)
    self_detect_std = np.diag(cross_std)
    best_competitor = np.zeros(n_clusters)
    best_competitor_std = np.zeros(n_clusters)

    for i in range(n_clusters):
        competitors = [(cross_matrix[i, j], cross_std[i, j]) for j in range(n_clusters) if j != i]
        best_idx = max(range(len(competitors)), key=lambda k: competitors[k][0]) if competitors else 0
        best_competitor[i] = competitors[best_idx][0] if competitors else 0
        best_competitor_std[i] = competitors[best_idx][1] if competitors else 0

    x = np.arange(n_clusters)
    width = 0.35

    bars_self = ax_c.bar(x - width/2, self_detect, width, label='Self-detection',
                         color='#2ca02c', edgecolor='black', linewidth=0.5,
                         yerr=self_detect_std, capsize=3, error_kw={'lw': 1})
    bars_comp = ax_c.bar(x + width/2, best_competitor, width, label='Best competitor',
                         color='#d62728', edgecolor='black', linewidth=0.5,
                         yerr=best_competitor_std, capsize=3, error_kw={'lw': 1})

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax_c.set_xlabel('Cluster ID', fontsize=LABEL_SIZE)
    ax_c.set_ylabel('Avg Max |Correlation|', fontsize=LABEL_SIZE)
    ax_c.set_title('$\\mathbf{(c)}$\nSelf-Detection vs Best Competitor', fontsize=TITLE_SIZE)
    ax_c.legend(loc='upper right', fontsize=LEGEND_SIZE)
    ax_c.tick_params(axis='y', labelsize=TICK_SIZE)
    ax_c.set_ylim(0, 1.0)
    ax_c.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_FILE}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSelf-detection vs Best Competitor:")
    print(f"{'Cluster':<10} {'Self':<10} {'Competitor':<12} {'Margin':<10}")
    print("-" * 42)
    for i, cid in enumerate(cluster_ids):
        margin = self_detect[i] - best_competitor[i]
        print(f"{cid:<10} {self_detect[i]:<10.3f} {best_competitor[i]:<12.3f} {margin:+.3f}")

    wins = sum(1 for i in range(n_clusters) if self_detect[i] > best_competitor[i])
    print(f"\nClusters where self-detection wins: {wins}/{n_clusters}")


if __name__ == '__main__':
    main()
