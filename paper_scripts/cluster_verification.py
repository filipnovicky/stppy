#!/usr/bin/env python3
"""Cluster detection validation (paper Figure 2).

Quantifies how distinguishable the 12 consensus cluster templates are when
used as detectors on the data. For each (avalanche, detector) pair the best
time offset is chosen for the SELF template only; at that same offset the
correlation with every other template is computed. This is done both for the
sign-encoded avalanches and for the continuous EEG at the same time windows,
showing that the self-vs-competitor separation that is clear in sign-encoded
data shrinks in continuous data (motivating the high backfitting threshold).

Layout:
    Row 1 (sign-encoded): (a) cross-detection heatmap  (b) self vs competitor
    Row 2 (continuous):   (c) cross-detection heatmap  (d) self vs competitor

Output: cluster_verification.pdf (to REGEN_OUTPUT_DIR, else the script dir).
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import stppy
import stppy.io
from stppy.detection.correlation import _vectorized_frame_correlations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from psiconnect_utils import BASE_DIR


def _out_dir() -> Path:
    """Output dir for the regenerated figure; defaults to script dir so the
    committed original is never overwritten unless REGEN_OUTPUT_DIR is set."""
    d = os.environ.get('REGEN_OUTPUT_DIR')
    return Path(d) if d else Path(__file__).parent


# Paths (all relative to PSICONNECT_DATA_DIR)
AVALANCHE_FILE = BASE_DIR / 'avalanches.pkl'
DATASETS_FILE = BASE_DIR / 'datasets.pkl'
PEAK_TIMES_FILE = BASE_DIR / 'avalanche_times.pkl'
OUTPUT_FILE = _out_dir() / 'cluster_verification.pdf'


# ======================================================================
# Shared functions
# ======================================================================

def sliding_max_correlation_with_offset(arr_a, arr_b):
    """
    Same as sliding_max_correlation but also returns the best offset
    (start index in the longer array).
    """
    T1, T2 = arr_a.shape[1], arr_b.shape[1]

    if T1 == T2:
        return abs(_vectorized_frame_correlations(arr_a, arr_b)), 0

    if T1 < T2:
        pattern, data = arr_a, arr_b
    else:
        pattern, data = arr_b, arr_a

    pattern_len = pattern.shape[1]
    data_len = data.shape[1]
    n_windows = data_len - pattern_len + 1

    max_corr = 0.0
    best_start = 0
    for start in range(n_windows):
        window = data[:, start:start + pattern_len]
        corr = abs(_vectorized_frame_correlations(pattern, window))
        if corr > max_corr:
            max_corr = corr
            best_start = start

    return max_corr, best_start


def correlation_at_offset(aval, template, offset):
    """
    Compute |frame-by-frame correlation| at a fixed offset (from self-detection)
    rather than sliding to find the max.
    """
    T_aval = aval.shape[1]
    T_tmpl = template.shape[1]

    if T_aval == T_tmpl:
        return abs(_vectorized_frame_correlations(aval, template))

    if T_aval >= T_tmpl:
        # offset is position in aval; extract window of template's length
        start = min(offset, T_aval - T_tmpl)
        window = aval[:, start:start + T_tmpl]
        return abs(_vectorized_frame_correlations(template, window))
    else:
        # offset is position in template; extract window of aval's length
        start = min(offset, T_tmpl - T_aval)
        window = template[:, start:start + T_aval]
        return abs(_vectorized_frame_correlations(aval, window))


def sliding_window_max_correlation(arr1, arr2):
    """
    Compute similarity between two spatiotemporal patterns using sliding window max.
    Slides the shorter pattern along the longer one. At each offset, computes the
    average signed spatial correlation across matched frames. Returns the value
    at the offset with the highest absolute average correlation.
    """
    T1, T2 = arr1.shape[1], arr2.shape[1]

    if T1 <= T2:
        short, long_ = arr1, arr2
        T_short, T_long = T1, T2
    else:
        short, long_ = arr2, arr1
        T_short, T_long = T2, T1

    n_offsets = T_long - T_short + 1
    avg_corrs = np.zeros(n_offsets)

    for k in range(n_offsets):
        frame_corrs = []
        for t in range(T_short):
            r = np.corrcoef(short[:, t], long_[:, t + k])[0, 1]
            if np.isnan(r):
                r = 0.0
            frame_corrs.append(r)
        avg_corrs[k] = np.mean(frame_corrs)

    best_k = np.argmax(np.abs(avg_corrs))
    return avg_corrs[best_k]


def compute_cluster_correlation_matrix(cluster_averages, cluster_ids):
    """Compute sliding-window-max correlation between all cluster pairs."""
    n_clusters = len(cluster_ids)
    corr_matrix = np.full((n_clusters, n_clusters), np.nan)

    for i, c1 in enumerate(cluster_ids):
        for j, c2 in enumerate(cluster_ids):
            corr_matrix[i, j] = sliding_window_max_correlation(
                cluster_averages[c1], cluster_averages[c2]
            )

    return corr_matrix


def plot_cluster_correlation(ax, corr_matrix, cluster_ids, n_clusters,
                             panel_label, TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
                             CBAR_SIZE):
    """Plot cluster correlation heatmap with diagonal masked."""
    plot_matrix = corr_matrix.copy()
    np.fill_diagonal(plot_matrix, np.nan)
    im = ax.imshow(plot_matrix, cmap='RdBu_r', vmin=-0.2, vmax=1,
                   interpolation='nearest')
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax.set_yticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax.set_xlabel('Cluster ID', fontsize=LABEL_SIZE)
    ax.set_ylabel('Cluster ID', fontsize=LABEL_SIZE)
    ax.set_title(f'$\\mathbf{{({panel_label})}}$\nCluster Correlation',
                 fontsize=TITLE_SIZE)

    for i in range(n_clusters):
        for j in range(n_clusters):
            if not np.isnan(plot_matrix[i, j]):
                color = 'white' if abs(plot_matrix[i, j]) > 0.6 else 'black'
                ax.text(j, i, f'{plot_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_label('Pearson r', fontsize=CBAR_SIZE)


def build_avalanche_to_dataset_mapping(peak_avalanche_times):
    """
    Build mapping from global avalanche index to (dataset_idx, local_start, local_end).
    """
    mapping = []
    for ds_idx, ds_avalanches in enumerate(peak_avalanche_times):
        for start, end in ds_avalanches:
            mapping.append((ds_idx, int(start), int(end)))

    return mapping, len(mapping)


# ======================================================================
# Sign-encoded avalanche cross-detection (forced)
# ======================================================================

def compute_cross_detection_binary_forced(flat_avalanches, cluster_averages,
                                          cluster_assignments, cluster_ids):
    """Compute cross-detection matrix for binary avalanches with forced."""
    n_clusters = len(cluster_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    corr_accum = [[[] for _ in range(n_clusters)] for _ in range(n_clusters)]

    for source_cid in cluster_ids:
        source_idx = cid_to_idx[source_cid]
        avalanche_indices = np.where(cluster_assignments == source_cid)[0]
        print(f"  Cluster {source_cid}: {len(avalanche_indices)} avalanches")

        self_template = cluster_averages[source_cid]

        for aval_idx in avalanche_indices:
            aval = flat_avalanches[aval_idx]

            # Step 1: Find best offset using SELF template only
            self_corr, best_offset = sliding_max_correlation_with_offset(
                aval, self_template
            )
            corr_accum[source_idx][source_idx].append(self_corr)

            # Step 2: For all OTHER detectors, evaluate at that same offset
            for detector_cid in cluster_ids:
                if detector_cid == source_cid:
                    continue
                detector_idx = cid_to_idx[detector_cid]
                detector_template = cluster_averages[detector_cid]
                corr = correlation_at_offset(aval, detector_template, best_offset)
                corr_accum[source_idx][detector_idx].append(corr)

    cross_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if corr_accum[i][j]:
                cross_matrix[i, j] = np.mean(corr_accum[i][j])

    return cross_matrix, corr_accum


# ======================================================================
# Continuous avalanche cross-detection (forced)
# ======================================================================

def compute_cross_detection_continuous_forced(datasets, avalanche_to_ds, n_base,
                                              cluster_averages, cluster_assignments,
                                              cluster_ids):
    """Compute cross-detection matrix for continuous EEG segments with forced."""
    n_clusters = len(cluster_ids)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    corr_accum = [[[] for _ in range(n_clusters)] for _ in range(n_clusters)]

    for source_cid in cluster_ids:
        source_idx = cid_to_idx[source_cid]
        avalanche_indices = np.where(cluster_assignments == source_cid)[0]
        print(f"  Cluster {source_cid}: {len(avalanche_indices)} avalanches")

        self_template = cluster_averages[source_cid]

        for aval_idx in avalanche_indices:
            ds_idx, start_sample, end_sample = avalanche_to_ds[aval_idx % n_base]
            aval = datasets[ds_idx][:, start_sample:end_sample]

            if aval_idx >= n_base:
                aval = -aval

            # Step 1: Find best offset using SELF template only
            self_corr, best_offset = sliding_max_correlation_with_offset(
                aval, self_template
            )
            corr_accum[source_idx][source_idx].append(self_corr)

            # Step 2: For all OTHER detectors, evaluate at that same offset
            for detector_cid in cluster_ids:
                if detector_cid == source_cid:
                    continue
                detector_idx = cid_to_idx[detector_cid]
                detector_template = cluster_averages[detector_cid]
                corr = correlation_at_offset(aval, detector_template, best_offset)
                corr_accum[source_idx][detector_idx].append(corr)

    cross_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if corr_accum[i][j]:
                cross_matrix[i, j] = np.mean(corr_accum[i][j])

    return cross_matrix, corr_accum


# ======================================================================
# Plotting helpers
# ======================================================================

def plot_heatmap(ax, cross_matrix, cluster_ids, n_clusters, panel_label,
                 TITLE_SIZE, LABEL_SIZE, TICK_SIZE, CBAR_SIZE):
    """Plot cross-detection heatmap on the given axis."""
    im = ax.imshow(cross_matrix, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax.set_yticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax.set_xlabel('Detector Cluster ID', fontsize=LABEL_SIZE)
    ax.set_ylabel('Source Cluster ID', fontsize=LABEL_SIZE)
    ax.set_title(f'$\\mathbf{{({panel_label})}}$\nCross-Cluster Avg Max |Correlation|',
                 fontsize=TITLE_SIZE)

    for i in range(n_clusters):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                             fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, label='Avg |Correlation|', shrink=0.8)
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.set_label('Avg |Correlation|', fontsize=CBAR_SIZE)


def plot_violin(ax, cross_matrix, corr_accum, cluster_ids, n_clusters,
                panel_label, TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE,
                show_legend=True):
    """Plot self-detection vs best competitor split violin on the given axis."""
    self_detect = np.diag(cross_matrix)
    best_competitor_idx = np.zeros(n_clusters, dtype=int)

    for i in range(n_clusters):
        comp_means = [cross_matrix[i, j] if j != i else -1 for j in range(n_clusters)]
        best_competitor_idx[i] = int(np.argmax(comp_means))

    x = np.arange(n_clusters)

    for i in range(n_clusters):
        self_data = np.array(corr_accum[i][i]) if corr_accum[i][i] else np.array([0.0])
        comp_data = (np.array(corr_accum[i][best_competitor_idx[i]])
                     if corr_accum[i][best_competitor_idx[i]] else np.array([0.0]))

        # Left half: self-detection
        if len(self_data) > 1:
            vp_self = ax.violinplot([self_data], positions=[x[i]], widths=0.8,
                                    showextrema=False)
            for body in vp_self['bodies']:
                m = np.mean(body.get_paths()[0].vertices[:, 0])
                body.get_paths()[0].vertices[:, 0] = np.clip(
                    body.get_paths()[0].vertices[:, 0], -np.inf, m)
                body.set_facecolor('#2ca02c')
                body.set_edgecolor('black')
                body.set_linewidth(0.5)
                body.set_alpha(0.8)

        # Right half: best competitor
        if len(comp_data) > 1:
            vp_comp = ax.violinplot([comp_data], positions=[x[i]], widths=0.8,
                                    showextrema=False)
            for body in vp_comp['bodies']:
                m = np.mean(body.get_paths()[0].vertices[:, 0])
                body.get_paths()[0].vertices[:, 0] = np.clip(
                    body.get_paths()[0].vertices[:, 0], m, np.inf)
                body.set_facecolor('#d62728')
                body.set_edgecolor('black')
                body.set_linewidth(0.5)
                body.set_alpha(0.8)

        # Mean markers
        ax.scatter(x[i] - 0.15, np.mean(self_data), marker='D', color='#2ca02c',
                   edgecolor='black', s=30, zorder=5, linewidths=0.5)
        ax.scatter(x[i] + 0.15, np.mean(comp_data), marker='D', color='#d62728',
                   edgecolor='black', s=30, zorder=5, linewidths=0.5)

    if show_legend:
        ax.legend(handles=[Patch(facecolor='#2ca02c', edgecolor='black', label='Self-detection'),
                           Patch(facecolor='#d62728', edgecolor='black', label='Best competitor')],
                  loc='upper right', fontsize=LEGEND_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(cluster_ids, fontsize=TICK_SIZE)
    ax.set_xlabel('Cluster ID', fontsize=LABEL_SIZE)
    ax.set_ylabel('Max |Correlation|', fontsize=LABEL_SIZE)
    ax.set_title(f'$\\mathbf{{({panel_label})}}$\nSelf-Detection vs Best Competitor',
                 fontsize=TITLE_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_SIZE)
    ax.set_ylim(0, 1.0)


def print_summary(label, cross_matrix, cluster_ids, n_clusters):
    """Print summary statistics for a given method."""
    self_detect = np.diag(cross_matrix)
    best_competitor_idx = np.zeros(n_clusters, dtype=int)

    for i in range(n_clusters):
        comp_means = [cross_matrix[i, j] if j != i else -1 for j in range(n_clusters)]
        best_competitor_idx[i] = int(np.argmax(comp_means))

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {label}")
    print(f"{'=' * 60}")
    print(f"\nSelf-detection vs Best Competitor:")
    print(f"{'Cluster':<10} {'Self':<10} {'Competitor':<12} {'Margin':<10}")
    print("-" * 42)
    for i, cid in enumerate(cluster_ids):
        comp_val = cross_matrix[i, best_competitor_idx[i]]
        margin = self_detect[i] - comp_val
        print(f"{cid:<10} {self_detect[i]:<10.3f} {comp_val:<12.3f} {margin:+.3f}")

    wins = sum(1 for i in range(n_clusters)
               if self_detect[i] > cross_matrix[i, best_competitor_idx[i]])
    print(f"\nClusters where self-detection wins: {wins}/{n_clusters}")


# ======================================================================
# Main
# ======================================================================

def main():
    # --- Load data ---
    print("Loading data...")
    flat_avalanches = stppy.io.load_pickle(AVALANCHE_FILE)
    datasets = stppy.io.load_pickle(DATASETS_FILE)
    peak_avalanche_times = stppy.io.load_pickle(PEAK_TIMES_FILE)
    cluster_averages = stppy.io.load_pickle(BASE_DIR / 'cluster_averages.pkl')
    cluster_assignments = stppy.io.load_pickle(BASE_DIR / 'cluster_assignments.pkl')

    print(f"  {len(flat_avalanches)} binary avalanches")
    print(f"  {len(datasets)} datasets")

    avalanche_to_ds, n_base = build_avalanche_to_dataset_mapping(peak_avalanche_times)
    print(f"  {n_base} base avalanche mappings")

    cluster_ids = sorted(cluster_averages.keys())
    n_clusters = len(cluster_ids)
    print(f"  {n_clusters} clusters: {cluster_ids}")

    # --- Sign-encoded cross-detection (forced) ---
    print("\nComputing binary cross-detection...")
    cross_bin, accum_bin = compute_cross_detection_binary_forced(
        flat_avalanches, cluster_averages, cluster_assignments, cluster_ids
    )

    # --- Continuous cross-detection (forced) ---
    print("\nComputing continuous cross-detection...")
    cross_cont, accum_cont = compute_cross_detection_continuous_forced(
        datasets, avalanche_to_ds, n_base,
        cluster_averages, cluster_assignments, cluster_ids
    )

    # --- Cluster correlation matrix ---
    print("\nComputing cluster correlation matrix...")
    cluster_corr = compute_cluster_correlation_matrix(cluster_averages, cluster_ids)

    # --- Create figure: 3 rows, (e) centered below c and d ---
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 11
    LEGEND_SIZE = 11
    CBAR_SIZE = 11

    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.98, top=0.97, bottom=0.04)

    # Row 1: Sign-encoded
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_heatmap(ax_a, cross_bin, cluster_ids, n_clusters, 'a',
                 TITLE_SIZE, LABEL_SIZE, TICK_SIZE, CBAR_SIZE)
    plot_violin(ax_b, cross_bin, accum_bin, cluster_ids, n_clusters, 'b',
                TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE)

    # Row 2: Continuous
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    plot_heatmap(ax_c, cross_cont, cluster_ids, n_clusters, 'c',
                 TITLE_SIZE, LABEL_SIZE, TICK_SIZE, CBAR_SIZE)
    plot_violin(ax_d, cross_cont, accum_cont, cluster_ids, n_clusters, 'd',
                TITLE_SIZE, LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, show_legend=False)

    # Row 3: Cluster correlation centered manually
    # gs[2, :] spans the full grid width which shifts right due to colorbars;
    # use fig.add_axes with explicit position instead
    gs_pos = gs[2, :].get_position(fig)
    panel_width = 0.38
    panel_left = 0.5 - panel_width / 2
    ax_e = fig.add_axes([panel_left, gs_pos.y0, panel_width, gs_pos.height])
    plot_cluster_correlation(ax_e, cluster_corr, cluster_ids, n_clusters, 'e',
                             TITLE_SIZE, LABEL_SIZE, TICK_SIZE, CBAR_SIZE)

    # Row labels on the left side
    fig.text(0.01, 0.82, 'Sign-encoded', va='center', ha='center', fontsize=16,
             fontweight='bold', rotation=90)
    fig.text(0.01, 0.50, 'Continuous', va='center', ha='center', fontsize=16,
             fontweight='bold', rotation=90)

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_FILE}")

    # --- Summary stats ---
    print_summary("Sign-encoded", cross_bin, cluster_ids, n_clusters)
    print_summary("Continuous", cross_cont, cluster_ids, n_clusters)


if __name__ == '__main__':
    main()
