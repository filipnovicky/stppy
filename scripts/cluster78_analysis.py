#!/usr/bin/env python3
"""
Cluster 7+8 Combined Analysis

Detects bursts where EITHER cluster 7 OR cluster 8 (or both) contribute to
continuous activity. Classifies bursts as:
  - Pure C7: Only cluster 7 active
  - Pure C8: Only cluster 8 active
  - Combined: Both clusters contribute to the burst

Polarity is determined by the +/- pattern across the entire sequence.

Output: Table showing polarity profiles similar to LaTeX format.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import stppy
import stppy.io

import os
import numpy as np
from collections import defaultdict

from psiconnect_utils import (
    parse_condition, determine_polarity,
    find_continuous_periods, merge_periods_into_bursts,
    DRUG_LABELS, TASK_LABELS, POL_ORDER, BASE_DIR
)


def classify_burst(burst):
    """Classify burst as pure_7, pure_8, or combined."""
    unique_clusters = set(burst['clusters'])
    if unique_clusters == {7}:
        return 'pure_7'
    elif unique_clusters == {8}:
        return 'pure_8'
    else:
        return 'combined'


def count_cluster_lengths(burst):
    """Count length (number of periods) from each cluster in burst."""
    c7_count = sum(1 for c in burst['clusters'] if c == 7)
    c8_count = sum(1 for c in burst['clusters'] if c == 8)
    return c7_count, c8_count


def run_analysis():
    print("Loading data...")

    cat = stppy.io.load_pickle(BASE_DIR / 'cluster_activation_times.pkl')
    cluster_durations = stppy.io.load_pickle(BASE_DIR / 'cluster_durations.pkl')
    metadata = stppy.io.load_pickle(BASE_DIR / 'metadata.pkl')
    original_bursts = stppy.io.load_pickle(BASE_DIR / 'bursts_by_cluster.pkl')

    dataset_names = [os.path.basename(m['file_path']) for m in metadata]

    dur_7 = cluster_durations[7]
    dur_8 = cluster_durations[8]
    sfreq = 250.0
    burst_tolerance = 2.0 / sfreq  # 2 samples = 8ms

    print(f"Cluster 7 duration: {dur_7*1000:.1f}ms")
    print(f"Cluster 8 duration: {dur_8*1000:.1f}ms")
    print(f"Burst continuation tolerance: {burst_tolerance*1000:.1f}ms")
    print()

    # Detect combined bursts for each dataset
    all_bursts = []

    for ds_idx, ds_name in enumerate(dataset_names):
        _, times_7, pols_7, _ = cat[7][ds_idx]
        _, times_8, pols_8, _ = cat[8][ds_idx]

        periods_7 = find_continuous_periods(times_7, pols_7, dur_7, sfreq)
        periods_8 = find_continuous_periods(times_8, pols_8, dur_8, sfreq)

        for p in periods_7:
            p['cluster'] = 7
        for p in periods_8:
            p['cluster'] = 8

        all_periods = periods_7 + periods_8

        bursts = merge_periods_into_bursts(all_periods, burst_tolerance)

        # Filter bursts with >= 3 periods (minimum burst length)
        bursts = [b for b in bursts if len(b['clusters']) >= 3]

        for b in bursts:
            b['dataset'] = ds_name
            drug, _, task = parse_condition(ds_name)
            b['_drug'] = drug
            b['_task'] = task
            b['burst_type'] = classify_burst(b)
            b['polarity'] = determine_polarity(b['pols'])
            b['length'] = len(b['clusters'])

        all_bursts.extend(bursts)

    # Separate by burst type
    pure_7 = [b for b in all_bursts if b['burst_type'] == 'pure_7']
    pure_8 = [b for b in all_bursts if b['burst_type'] == 'pure_8']
    combined = [b for b in all_bursts if b['burst_type'] == 'combined']

    print(f"Total bursts detected: {len(all_bursts)}")
    print(f"  Pure C7: {len(pure_7)}")
    print(f"  Pure C8: {len(pure_8)}")
    print(f"  Combined: {len(combined)}")
    print()

    orig_7 = len(original_bursts[7])
    orig_8 = len(original_bursts[8])
    print(f"Original separate detection: C7={orig_7}, C8={orig_8}, total={orig_7+orig_8}")
    print()

    # ==================================================================
    # POLARITY PROFILE TABLE (LaTeX-style)
    # ==================================================================
    print("=" * 90)
    print("POLARITY PROFILE TABLE")
    print("=" * 90)
    print()
    print("Table: Polarity profiles of cluster 7, 8, and combined 7+8 bursts")
    print()

    def count_by_drug_pol(bursts):
        counts = defaultdict(lambda: defaultdict(int))
        for b in bursts:
            counts[b['_drug']][b['polarity']] += 1
        return counts

    pure_7_counts = count_by_drug_pol(pure_7)
    pure_8_counts = count_by_drug_pol(pure_8)
    combined_counts = count_by_drug_pol(combined)
    total_counts = count_by_drug_pol(all_bursts)

    ctrl_n = sum(total_counts['ctrl'].values())
    psi_n = sum(total_counts['psi'].values())

    print(f"{'Category':<15} | {'Control (n=' + str(ctrl_n) + ')':^40} | {'Psilocybin (n=' + str(psi_n) + ')':^40}")
    print(f"{'':15} | {'Stable':>10} {'Switch':>8} {'Mixed':>8} {'n':>8} | {'Stable':>10} {'Switch':>8} {'Mixed':>8} {'n':>8}")
    print("-" * 110)

    def print_row(label, counts, highlight=False):
        prefix = ">> " if highlight else "   "
        row = f"{prefix}{label:<12} |"
        for drug in ['ctrl', 'psi']:
            n = sum(counts[drug].values())
            stable = counts[drug].get('stable', 0)
            switch = counts[drug].get('switching', 0)
            mixed = counts[drug].get('mixed', 0)
            if n > 0:
                stable_pct = f"{stable/n*100:.1f}%"
                switch_pct = f"{switch/n*100:.1f}%"
                mixed_pct = f"{mixed/n*100:.1f}%"
            else:
                stable_pct = "-"
                switch_pct = "-"
                mixed_pct = "-"
            row += f" {stable_pct:>10} {switch_pct:>8} {mixed_pct:>8} {n:>8} |"
        print(row)

    print_row("Pure C7", pure_7_counts)
    print_row("Pure C8", pure_8_counts)
    print_row("Combined", combined_counts, highlight=True)
    print("-" * 110)
    print_row("Total 7+8", total_counts)
    print("-" * 110)
    print()

    # Key finding summary
    print("KEY FINDING:")
    n_combined = len(combined)
    n_total = len(all_bursts)
    pct_combined = n_combined / n_total * 100

    ctrl_combined_stable = combined_counts['ctrl'].get('stable', 0)
    ctrl_combined_n = sum(combined_counts['ctrl'].values())
    psi_combined_stable = combined_counts['psi'].get('stable', 0)
    psi_combined_n = sum(combined_counts['psi'].values())

    print(f"  {pct_combined:.1f}% of all C7+C8 bursts are COMBINED (both clusters contribute)")
    if ctrl_combined_n > 0 and psi_combined_n > 0:
        print(f"  Combined bursts are predominantly stable:")
        print(f"    Control: {ctrl_combined_stable/ctrl_combined_n*100:.1f}% stable")
        print(f"    Psilocybin: {psi_combined_stable/psi_combined_n*100:.1f}% stable")
    print()

    # ==================================================================
    # DETAILED BREAKDOWN
    # ==================================================================
    print("=" * 90)
    print("DETAILED BREAKDOWN")
    print("=" * 90)
    print()

    drug_labels = DRUG_LABELS
    task_labels = TASK_LABELS
    pol_order = POL_ORDER
    tasks = sorted({b['_task'] for b in all_bursts})

    for label, burst_subset in [('Pure Cluster 7', pure_7),
                                ('Pure Cluster 8', pure_8),
                                ('Combined (7+8)', combined)]:
        counts = count_by_drug_pol(burst_subset)
        print(f"{label}:")
        for drug in ['ctrl', 'psi']:
            n = sum(counts[drug].values())
            print(f"  {drug_labels[drug]} ({n} bursts):")
            for pol in pol_order:
                c = counts[drug].get(pol, 0)
                if c > 0:
                    print(f"    {pol:12s}: {c:>5} ({c/n*100:.1f}%)")
        print()

    # ==================================================================
    # PER-TASK BREAKDOWN (with drug separation)
    # ==================================================================
    print("=" * 90)
    print("PER-TASK BREAKDOWN")
    print("=" * 90)
    print()

    for task in tasks:
        task_bursts = [b for b in all_bursts if b['_task'] == task]
        n_task = len(task_bursts)
        if n_task == 0:
            continue
        print(f"{task_labels.get(task, task)} ({n_task} bursts):")
        for pol in pol_order:
            c = sum(1 for b in task_bursts if b['polarity'] == pol)
            if c > 0:
                print(f"  {pol:12s}: {c:>5} ({c/n_task*100:.1f}%)")
        for drug in ['ctrl', 'psi']:
            drug_task = [b for b in task_bursts if b['_drug'] == drug]
            n_dt = len(drug_task)
            if n_dt == 0:
                continue
            print(f"  {drug_labels[drug]} ({n_dt} bursts):")
            for pol in pol_order:
                c = sum(1 for b in drug_task if b['polarity'] == pol)
                if c > 0:
                    print(f"    {pol:12s}: {c:>5} ({c/n_dt*100:.1f}%)")
        print()

    # ==================================================================
    # COMBINED BURST DETAILS
    # ==================================================================
    if combined:
        print("=" * 90)
        print("COMBINED BURST STATISTICS")
        print("=" * 90)
        print()

        c7_contributions = []
        c8_contributions = []
        for b in combined:
            c7, c8 = count_cluster_lengths(b)
            c7_contributions.append(c7)
            c8_contributions.append(c8)

        print(f"In combined bursts:")
        print(f"  C7 contributions per burst: mean={np.mean(c7_contributions):.1f}, "
              f"min={min(c7_contributions)}, max={max(c7_contributions)}")
        print(f"  C8 contributions per burst: mean={np.mean(c8_contributions):.1f}, "
              f"min={min(c8_contributions)}, max={max(c8_contributions)}")
        print()

        print("Example combined bursts:")
        for b in combined[:5]:
            cluster_seq = '\u2192'.join(str(c) for c in b['clusters'])
            print(f"  {b['dataset']}: {b['start']:.3f}s-{b['end']:.3f}s")
            print(f"    Clusters: {cluster_seq}")
            print(f"    Polarity: {b['polarity']}")
            print()


if __name__ == '__main__':
    import io, contextlib

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        run_analysis()

    text = output.getvalue()
    print(text, end='')

    results_file = Path(__file__).parent / 'cluster78_analysis_results.txt'
    results_file.write_text(text)
    print(f"Results saved to: {results_file}")
