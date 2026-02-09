#!/usr/bin/env python3
"""
Temporal Overlap Analysis

Analyzes temporal overlaps between bursts from different clusters.
Overlapping bursts share at least one cluster duration in time.

Panels:
    (a) Temporal overlap counts between clusters (heatmap)
    (b) Overlaps by polarity combination and drug condition (histogram)

Input files:
    - bursts_by_cluster.pkl: {cluster_id: [burst_dicts, ...]}
    - cluster_averages_kept.pkl: {cluster_id: array} for cluster durations

Output:
    - overlap_figure.pdf
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import stppy
import stppy.io

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from psiconnect_utils import parse_condition, BASE_DIR

# Paths
OUTPUT_FILE = Path(__file__).parent / 'overlap_figure.pdf'


def run_analysis():
    print("Loading data...")
    bursts_by_cluster = stppy.io.load_pickle(BASE_DIR / 'bursts_by_cluster.pkl')
    cluster_averages = stppy.io.load_pickle(BASE_DIR / 'cluster_averages_kept.pkl')

    cluster_ids = sorted(bursts_by_cluster.keys())

    # Cluster durations from averages (for overlap detection)
    cluster_durations = {}
    for cid, avg in cluster_averages.items():
        cluster_durations[cid] = avg.shape[1] / 250.0

    # Tag every burst with drug/subject/task
    for cid in cluster_ids:
        for b in bursts_by_cluster[cid]:
            b['_drug'], b['_subj'], b['_task'] = parse_condition(b['dataset'])

    # ==================================================================
    # ANALYSIS: TEMPORAL OVERLAPS ACROSS CLUSTERS
    # ==================================================================
    print("=" * 80)
    print("TEMPORAL OVERLAPS ACROSS CLUSTERS")
    print("=" * 80)
    print("Overlap = bursts from different clusters sharing >= 1 cluster duration\n")

    # Group all bursts by dataset
    bursts_by_dataset = defaultdict(list)
    for cid in cluster_ids:
        for b in bursts_by_cluster[cid]:
            bursts_by_dataset[b['dataset']].append({
                'cluster_id': cid,
                'start_time': b['start_time'],
                'end_time': b['end_time'],
                'polarity': b.get('polarity', '?'),
                'drug': b['_drug']
            })

    # Count overlaps per cluster pair
    min_overlap = min(cluster_durations.values()) if cluster_durations else 0.040
    pair_counts = defaultdict(int)
    overlap_records = []

    for dataset, ds_bursts in bursts_by_dataset.items():
        ds_bursts.sort(key=lambda x: x['start_time'])

        for i in range(len(ds_bursts)):
            for j in range(i + 1, len(ds_bursts)):
                b1, b2 = ds_bursts[i], ds_bursts[j]

                if b1['cluster_id'] == b2['cluster_id']:
                    continue

                if b2['start_time'] >= b1['end_time']:
                    break

                overlap_start = max(b1['start_time'], b2['start_time'])
                overlap_end = min(b1['end_time'], b2['end_time'])
                overlap_sec = overlap_end - overlap_start

                if overlap_sec >= min_overlap:
                    pair = tuple(sorted([b1['cluster_id'], b2['cluster_id']]))
                    pair_counts[pair] += 1
                    overlap_records.append({
                        'pair': pair,
                        'polarity_1': b1['polarity'],
                        'polarity_2': b2['polarity'],
                        'drug': b1['drug']
                    })

    total_overlaps = sum(pair_counts.values())
    print(f"Total overlapping burst pairs: {total_overlaps}")
    print(f"Minimum overlap threshold: {min_overlap*1000:.1f}ms\n")

    # Overlap matrix
    all_cids = sorted(set(cluster_ids) | set(cluster_averages.keys()))
    n_clusters = len(all_cids)
    cid_idx = {c: i for i, c in enumerate(all_cids)}
    overlap_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

    for (c1, c2), count in pair_counts.items():
        if c1 in cid_idx and c2 in cid_idx:
            i, j = cid_idx[c1], cid_idx[c2]
            overlap_matrix[i, j] = count
            overlap_matrix[j, i] = count

    header = "     | " + " | ".join(f"C{c:02d}" for c in all_cids)
    print(header)
    print("-" * len(header))

    for i, c1 in enumerate(all_cids):
        row_vals = []
        for j, c2 in enumerate(all_cids):
            if i == j:
                row_vals.append("  - ")
            elif overlap_matrix[i, j] > 0:
                row_vals.append(f"{overlap_matrix[i, j]:4d}")
            else:
                row_vals.append("   .")
        print(f"C{c1:02d}  | " + " | ".join(row_vals))

    # Top pairs
    if pair_counts:
        print(f"\nTop overlapping cluster pairs:")
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
        for i, ((c1, c2), count) in enumerate(sorted_pairs[:10], 1):
            print(f"  {i:2d}. C{c1:02d} <-> C{c2:02d}: {count:4d} overlaps")
    print()

    # ==================================================================
    # FIGURE: OVERLAP HEATMAP AND POLARITY HISTOGRAM
    # ==================================================================
    print("=" * 80)
    print("GENERATING FIGURE: OVERLAP ANALYSIS")
    print("=" * 80)

    # Font sizes
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12

    if overlap_records:
        polarity_combo_counts = defaultdict(lambda: defaultdict(int))
        drug_labels_fig = {'ctrl': 'Control', 'psi': 'Psilocybin'}

        abbrev = {'stable': 'S', 'switching': 'SW', 'mixed': 'M',
                  'long_switching': 'LSW', 'single': '1'}

        for rec in overlap_records:
            p1, p2 = rec['polarity_1'], rec['polarity_2']
            combo = tuple(sorted([p1, p2]))
            a1, a2 = abbrev.get(combo[0], combo[0]), abbrev.get(combo[1], combo[1])
            combo_str = f'{a1}–{a2}'
            polarity_combo_counts[combo_str][rec['drug']] += 1

        combo_totals = {k: sum(v.values()) for k, v in polarity_combo_counts.items()}
        sorted_combos = sorted(combo_totals, key=combo_totals.get, reverse=True)

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(18, 7),
                                          gridspec_kw={'width_ratios': [1, 1]})

        # (a) Overlap heatmap
        mask = np.eye(n_clusters, dtype=bool)
        plot_data = overlap_matrix.astype(float)
        plot_data[mask] = np.nan

        im = ax_a.imshow(plot_data, cmap='viridis', interpolation='nearest')
        ax_a.set_xticks(range(n_clusters))
        ax_a.set_yticks(range(n_clusters))
        ax_a.set_xticklabels([f'C{c}' for c in all_cids], fontsize=TICK_SIZE)
        ax_a.set_yticklabels([f'C{c}' for c in all_cids], fontsize=TICK_SIZE)
        ax_a.set_xlabel('Cluster', fontsize=LABEL_SIZE)
        ax_a.set_ylabel('Cluster', fontsize=LABEL_SIZE)
        ax_a.set_title('$\\mathbf{(a)}$ Temporal Overlap Counts Between Clusters', fontsize=TITLE_SIZE)

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i != j and overlap_matrix[i, j] > 0:
                    ax_a.text(j, i, str(overlap_matrix[i, j]),
                              ha='center', va='center', fontsize=7,
                              color='white' if overlap_matrix[i, j] > overlap_matrix.max() * 0.6 else 'black')

        cbar_a = plt.colorbar(im, ax=ax_a, label='Overlap count', fraction=0.046)
        cbar_a.ax.tick_params(labelsize=TICK_SIZE)
        cbar_a.set_label('Overlap count', fontsize=LABEL_SIZE)

        # (b) Polarity histogram
        x = np.arange(len(sorted_combos))
        width = 0.35
        ctrl_vals = [polarity_combo_counts[c].get('ctrl', 0) for c in sorted_combos]
        psi_vals = [polarity_combo_counts[c].get('psi', 0) for c in sorted_combos]

        ax_b.bar(x - width / 2, ctrl_vals, width, label='Control', color="#9A26F8")
        ax_b.bar(x + width / 2, psi_vals, width, label='Psilocybin', color="#F3C940")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(sorted_combos, fontsize=TICK_SIZE)
        ax_b.set_xlabel('Polarity combination', fontsize=LABEL_SIZE)
        ax_b.set_ylabel('Number of overlaps', fontsize=LABEL_SIZE)
        ax_b.set_title('$\\mathbf{(b)}$ Overlaps by Polarity and Drug Condition', fontsize=TITLE_SIZE)
        ax_b.legend(fontsize=12)
        ax_b.tick_params(axis='y', labelsize=TICK_SIZE)

        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {OUTPUT_FILE}")
        plt.close('all')
    else:
        print("No overlaps found, figure not generated.")

    # ==================================================================
    # ADDITIONAL ANALYSIS: COMBINED CLUSTERS 7+8
    # ==================================================================
    print("\n" + "=" * 80)
    print("COMBINED CLUSTER 7+8 ANALYSIS")
    print("=" * 80)
    print("Treating clusters 7 and 8 as a single combined cluster\n")

    combined_bursts_by_dataset = defaultdict(list)
    for cid in cluster_ids:
        for b in bursts_by_cluster[cid]:
            combined_cid = '7+8' if cid in (7, 8) else cid
            combined_bursts_by_dataset[b['dataset']].append({
                'cluster_id': combined_cid,
                'start_time': b['start_time'],
                'end_time': b['end_time'],
                'polarity': b.get('polarity', '?'),
                'drug': b['_drug']
            })

    # Merge overlapping bursts from combined cluster 7+8
    for dataset, ds_bursts in combined_bursts_by_dataset.items():
        combined_78 = [b for b in ds_bursts if b['cluster_id'] == '7+8']
        others = [b for b in ds_bursts if b['cluster_id'] != '7+8']

        if len(combined_78) > 1:
            combined_78.sort(key=lambda x: x['start_time'])

            merged = []
            current = combined_78[0].copy()

            for burst in combined_78[1:]:
                if burst['start_time'] <= current['end_time'] + min_overlap:
                    current['end_time'] = max(current['end_time'], burst['end_time'])
                    if current['polarity'] != burst['polarity']:
                        current['polarity'] = 'combined'
                else:
                    merged.append(current)
                    current = burst.copy()

            merged.append(current)
            combined_bursts_by_dataset[dataset] = others + merged
        else:
            combined_bursts_by_dataset[dataset] = others + combined_78

    # Count overlaps for combined analysis
    combined_pair_counts = defaultdict(int)
    combined_overlap_records = []

    for dataset, ds_bursts in combined_bursts_by_dataset.items():
        ds_bursts.sort(key=lambda x: x['start_time'])

        for i in range(len(ds_bursts)):
            for j in range(i + 1, len(ds_bursts)):
                b1, b2 = ds_bursts[i], ds_bursts[j]

                if b1['cluster_id'] == b2['cluster_id']:
                    continue

                if b2['start_time'] >= b1['end_time']:
                    break

                overlap_start = max(b1['start_time'], b2['start_time'])
                overlap_end = min(b1['end_time'], b2['end_time'])
                overlap_sec = overlap_end - overlap_start

                if overlap_sec >= min_overlap:
                    pair = tuple(sorted([str(b1['cluster_id']), str(b2['cluster_id'])]))
                    combined_pair_counts[pair] += 1
                    combined_overlap_records.append({
                        'pair': pair,
                        'polarity_1': b1['polarity'],
                        'polarity_2': b2['polarity'],
                        'drug': b1['drug']
                    })

    combined_total = sum(combined_pair_counts.values())
    print(f"Total overlapping burst pairs (with 7+8 combined): {combined_total}")
    print(f"Reduction from original: {total_overlaps} -> {combined_total} "
          f"({total_overlaps - combined_total} fewer, {(1 - combined_total/total_overlaps)*100:.1f}% reduction)\n")

    if combined_pair_counts:
        print("Top overlapping cluster pairs (combined 7+8):")
        sorted_combined = sorted(combined_pair_counts.items(), key=lambda x: x[1], reverse=True)
        for i, ((c1, c2), count) in enumerate(sorted_combined[:10], 1):
            print(f"  {i:2d}. {c1} <-> {c2}: {count:4d} overlaps")

    # Polarity breakdown for combined analysis
    print("\nPolarity breakdown (combined 7+8):")
    combined_polarity_counts = defaultdict(lambda: defaultdict(int))
    for rec in combined_overlap_records:
        p1, p2 = rec['polarity_1'], rec['polarity_2']
        combo = tuple(sorted([p1, p2]))
        combo_str = f'{combo[0]}–{combo[1]}'
        combined_polarity_counts[combo_str][rec['drug']] += 1

    for combo in sorted(combined_polarity_counts.keys()):
        ctrl = combined_polarity_counts[combo].get('ctrl', 0)
        psi = combined_polarity_counts[combo].get('psi', 0)
        print(f"  {combo:25s}: ctrl={ctrl:4d}, psi={psi:4d}, total={ctrl+psi:4d}")


if __name__ == '__main__':
    run_analysis()
