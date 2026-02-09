#!/usr/bin/env python3
"""
Main Analysis Script

Analyses:
  1. Polarity distribution per cluster
  2. Polarity summary (by drug, task, cluster)

Input files (from parent directory):
    - bursts_by_cluster.pkl: {cluster_id: [burst_dicts, ...]}
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import stppy
import stppy.io

import numpy as np

from psiconnect_utils import (
    parse_condition, polarity_summary, sig_stars, benjamini_hochberg,
    DRUG_LABELS, TASK_LABELS, POLARITY_TYPES, BASE_DIR
)


def run_analysis():
    print("Loading data...")
    bursts_by_cluster = stppy.io.load_pickle(BASE_DIR / 'bursts_by_cluster.pkl')

    cluster_ids = sorted(bursts_by_cluster.keys())

    polarity_types = POLARITY_TYPES
    drugs = ['ctrl', 'psi']
    tasks = sorted({parse_condition(b['dataset'])[2]
                     for bursts in bursts_by_cluster.values()
                     for b in bursts})

    # Tag every burst with drug/subject/task
    for cid in cluster_ids:
        for b in bursts_by_cluster[cid]:
            b['_drug'], b['_subj'], b['_task'] = parse_condition(b['dataset'])

    # ==================================================================
    # ANALYSIS 1: POLARITY DISTRIBUTION PER CLUSTER
    # ==================================================================
    print("=" * 80)
    print("ANALYSIS 1: POLARITY DISTRIBUTION PER CLUSTER")
    print("=" * 80)

    drug_labels = {'ctrl': 'CONTROL', 'psi': 'PSILOCYBIN'}  # uppercase for this section
    task_labels = TASK_LABELS

    for cid in cluster_ids:
        bursts = bursts_by_cluster[cid]
        print(f"\nCluster {cid}:")
        for drug in drugs:
            print(f"  {drug_labels[drug]}:")
            for task in tasks:
                subset = [b for b in bursts if b['_drug'] == drug and b['_task'] == task]
                if len(subset) == 0:
                    continue
                lengths = [b['length'] for b in subset]
                mean_l = np.mean(lengths)
                std_l = np.std(lengths)
                print(f"    {task_labels.get(task, task):15s}: {mean_l:.2f}±{std_l:.2f} (n={len(subset)} bursts)")
        polarity_summary(f"  All", bursts, polarity_types, indent=1)
        for drug in drugs:
            drug_subset = [b for b in bursts if b['_drug'] == drug]
            polarity_summary(f"  {drug_labels[drug]}", drug_subset, polarity_types, indent=1)

    # ==================================================================
    # SUMMARY: POLARITY DISTRIBUTION
    # ==================================================================
    print("\n" + "=" * 80)
    print("POLARITY SUMMARY")
    print("=" * 80)

    all_bursts = [b for bursts in bursts_by_cluster.values() for b in bursts]

    # By drug
    for drug in drugs:
        subset = [b for b in all_bursts if b['_drug'] == drug]
        polarity_summary(drug_labels[drug], subset, polarity_types)
        print()

    # By task (with drug breakdown)
    for task in tasks:
        subset = [b for b in all_bursts if b['_task'] == task]
        polarity_summary(task_labels.get(task, task), subset, polarity_types)
        for drug in drugs:
            drug_subset = [b for b in subset if b['_drug'] == drug]
            polarity_summary(drug_labels[drug], drug_subset, polarity_types, indent=1)
        print()

    # By drug x task
    for drug in drugs:
        print(f"{drug_labels[drug]}:")
        for task in tasks:
            subset = [b for b in all_bursts if b['_drug'] == drug and b['_task'] == task]
            polarity_summary(task_labels.get(task, task), subset, polarity_types, indent=1)
        print()

    # Per cluster (with drug breakdown)
    for cid in cluster_ids:
        bursts = bursts_by_cluster[cid]
        polarity_summary(f"Cluster {cid}", bursts, polarity_types)
        for drug in drugs:
            drug_subset = [b for b in bursts if b['_drug'] == drug]
            polarity_summary(f"  {drug_labels[drug]}", drug_subset, polarity_types, indent=1)
    print()

    # ==================================================================
    # STATISTICAL TESTS: Chi-square + FDR
    # ==================================================================
    print("=" * 80)
    print("STATISTICAL TESTS (Chi-square: stable vs switching, ctrl vs psi)")
    print("=" * 80)
    print()

    try:
        from scipy.stats import chi2_contingency
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("WARNING: scipy not available, skipping chi-square tests")

    sw_types = ('stable', 'switching')
    binary_bursts = [b for b in all_bursts if b['polarity'] in sw_types]
    ctrl_bin = [b for b in binary_bursts if b['_drug'] == 'ctrl']
    psi_bin = [b for b in binary_bursts if b['_drug'] == 'psi']

    cs_all = sum(1 for b in ctrl_bin if b['polarity'] == 'stable')
    cw_all = len(ctrl_bin) - cs_all
    ps_all = sum(1 for b in psi_bin if b['polarity'] == 'stable')
    pw_all = len(psi_bin) - ps_all

    cs_pct = cs_all / len(ctrl_bin) * 100 if ctrl_bin else 0
    cw_pct = 100 - cs_pct
    ps_pct = ps_all / len(psi_bin) * 100 if psi_bin else 0
    pw_pct = 100 - ps_pct

    # Collect all test labels and p-values for FDR
    test_labels = []
    test_pvalues = []

    # --- Overall ---
    p_ov = None
    sig_overall = ''
    if has_scipy and ctrl_bin and psi_bin:
        chi2_ov, p_ov, dof_ov, _ = chi2_contingency([[cs_all, cw_all], [ps_all, pw_all]])
        sig_overall = sig_stars(p_ov)
        test_labels.append('Overall')
        test_pvalues.append(p_ov)

    print("OVERALL (stable vs switching, ctrl vs psi):")
    print(f"  Control:    stable={cs_all} ({cs_pct:.1f}%), switching={cw_all} ({cw_pct:.1f}%), n={len(ctrl_bin)}")
    print(f"  Psilocybin: stable={ps_all} ({ps_pct:.1f}%), switching={pw_all} ({pw_pct:.1f}%), n={len(psi_bin)}")
    if p_ov is not None:
        print(f"  Chi-square={chi2_ov:.2f}, df={dof_ov}, p={p_ov:.6f} {sig_overall}")
    print()

    # --- Per task ---
    task_order = ['meditation', 'music', 'restingstate', 'video']
    task_display_map = TASK_LABELS

    print("PER TASK:")
    task_stats = {}
    for task in task_order:
        ct = [b for b in binary_bursts if b['_drug'] == 'ctrl' and b['_task'] == task]
        pt = [b for b in binary_bursts if b['_drug'] == 'psi' and b['_task'] == task]
        cs = sum(1 for b in ct if b['polarity'] == 'stable')
        cw = len(ct) - cs
        ps = sum(1 for b in pt if b['polarity'] == 'stable')
        pw = len(pt) - ps
        cs_p = cs / len(ct) * 100 if ct else 0
        ps_p = ps / len(pt) * 100 if pt else 0

        tname = task_display_map.get(task, task)
        print(f"  {tname}:")
        print(f"    Control:    stable={cs} ({cs_p:.1f}%), switching={cw}, n={len(ct)}")
        print(f"    Psilocybin: stable={ps} ({ps_p:.1f}%), switching={pw}, n={len(pt)}")

        p_t = None
        if has_scipy and ct and pt and min(cs, cw) + min(ps, pw) > 0:
            try:
                chi2_t, p_t, dof_t, _ = chi2_contingency([[cs, cw], [ps, pw]])
                print(f"    Chi-square={chi2_t:.2f}, df={dof_t}, p={p_t:.6f} {sig_stars(p_t)}")
                test_labels.append(f'Task: {tname}')
                test_pvalues.append(p_t)
            except ValueError:
                print(f"    Chi-square: skipped (zero cell)")
        task_stats[task] = {'cs_pct': cs_p, 'ps_pct': ps_p, 'p': p_t,
                            'cs': cs, 'cw': cw, 'ps': ps, 'pw': pw,
                            'n_ctrl': len(ct), 'n_psi': len(pt)}
    print()

    # --- Per cluster ---
    print("PER CLUSTER:")
    cluster_stats = {}
    for cid in cluster_ids:
        bc = [b for b in bursts_by_cluster[cid] if b['polarity'] in sw_types]
        ct = [b for b in bc if b['_drug'] == 'ctrl']
        pt = [b for b in bc if b['_drug'] == 'psi']
        cs = sum(1 for b in ct if b['polarity'] == 'stable')
        cw = len(ct) - cs
        ps = sum(1 for b in pt if b['polarity'] == 'stable')
        pw = len(pt) - ps
        cs_p = cs / len(ct) * 100 if ct else 0
        ps_p = ps / len(pt) * 100 if pt else 0

        print(f"  Cluster {cid}:")
        print(f"    Control:    stable={cs} ({cs_p:.1f}%), switching={cw}, n={len(ct)}")
        print(f"    Psilocybin: stable={ps} ({ps_p:.1f}%), switching={pw}, n={len(pt)}")

        p_c = None
        if has_scipy and ct and pt and min(cs, cw) + min(ps, pw) > 0:
            try:
                chi2_c, p_c, dof_c, _ = chi2_contingency([[cs, cw], [ps, pw]])
                print(f"    Chi-square={chi2_c:.2f}, df={dof_c}, p={p_c:.6f} {sig_stars(p_c)}")
                test_labels.append(f'Cluster {cid}')
                test_pvalues.append(p_c)
            except ValueError:
                print(f"    Chi-square: skipped (zero cell)")
        elif has_scipy and (not ct or not pt):
            print(f"    Chi-square: skipped (missing drug group)")
        cluster_stats[cid] = {'cs': cs, 'cw': cw, 'ps': ps, 'pw': pw,
                              'cs_pct': cs_p, 'ps_pct': ps_p, 'p': p_c}
    print()

    # --- FDR correction ---
    if has_scipy and len(test_pvalues) > 0:
        raw_p = np.array(test_pvalues)
        fdr_p = benjamini_hochberg(raw_p)

        print("-" * 80)
        print("FDR-CORRECTED SUMMARY (Benjamini-Hochberg)")
        print("-" * 80)
        print(f"  {'Test':<25s} {'p (raw)':>12s} {'p (FDR)':>12s} {'raw':>6s} {'FDR':>6s}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*6} {'-'*6}")
        for i, label in enumerate(test_labels):
            print(f"  {label:<25s} {raw_p[i]:>12.6f} {fdr_p[i]:>12.6f} "
                  f"{sig_stars(raw_p[i]):>6s} {sig_stars(fdr_p[i]):>6s}")
        print()

        n_sig_raw = sum(1 for p in raw_p if p < 0.05)
        n_sig_fdr = sum(1 for p in fdr_p if p < 0.05)
        print(f"  Significant (p<0.05): {n_sig_raw}/{len(raw_p)} raw, {n_sig_fdr}/{len(fdr_p)} FDR-corrected")
        print()

    # ==================================================================
    # FIGURE: MAIN RESULTS
    # ==================================================================
    print("=" * 80)
    print("GENERATING FIGURE: MAIN RESULTS")
    print("=" * 80)

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    OUTPUT_FILE = Path(__file__).parent / 'main_results.pdf'

    col_s = '#1a1a1a'
    col_w = '#C8C3B0'

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(20, 7),
        gridspec_kw={'width_ratios': [0.8, 1.2, 1.8]})

    # ---- (a) Overall Polarity Shift ----
    ax_a.bar(0, cs_pct, width=0.55, color=col_s)
    ax_a.bar(0, cw_pct, bottom=cs_pct, width=0.55, color=col_w)
    ax_a.bar(1, ps_pct, width=0.55, color=col_s)
    ax_a.bar(1, pw_pct, bottom=ps_pct, width=0.55, color=col_w)

    def _bar_label(ax, x, y_center, pct, dark):
        if pct > 8:
            ax.text(x, y_center, f'{pct:.1f}%', ha='center', va='center',
                    color='white' if dark else 'black', fontsize=13, fontweight='bold')

    _bar_label(ax_a, 0, cs_pct / 2, cs_pct, True)
    _bar_label(ax_a, 0, cs_pct + cw_pct / 2, cw_pct, False)
    _bar_label(ax_a, 1, ps_pct / 2, ps_pct, True)
    _bar_label(ax_a, 1, ps_pct + pw_pct / 2, pw_pct, False)

    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(['Control', 'Psilocybin'], fontsize=12)
    ax_a.set_ylabel('Percentage of Bursts (%)', fontsize=12)
    ax_a.set_title('(a) Overall Polarity Shift', fontsize=14, fontweight='bold')
    ax_a.set_ylim(0, 112)
    ax_a.set_axisbelow(True)
    ax_a.yaxis.grid(True, linestyle='--', alpha=0.3)

    if sig_overall:
        ax_a.plot([0, 0, 1, 1], [102, 105, 105, 102], 'k-', lw=1.5, clip_on=False)
        ax_a.text(0.5, 106, sig_overall, ha='center', va='bottom',
                  fontsize=14, fontweight='bold')

    ax_a.legend([Patch(facecolor=col_s), Patch(facecolor=col_w)],
                ['Stable', 'Switching'], loc='upper right', fontsize=10, framealpha=0.8)

    # ---- (b) Consistency Across Tasks ----
    task_display = ['Meditation', 'Music', 'Rest', 'Video']

    x_b = np.arange(len(task_order))
    w_b = 0.35
    ctrl_pcts_b, psi_pcts_b, sigs_b = [], [], []

    for task in task_order:
        ts = task_stats[task]
        ctrl_pcts_b.append(ts['cs_pct'])
        psi_pcts_b.append(ts['ps_pct'])
        if ts['p'] is not None:
            sigs_b.append(sig_stars(ts['p']))
        else:
            sigs_b.append('')

    ax_b.bar(x_b - w_b / 2, ctrl_pcts_b, w_b, label='Control', color=col_s)
    ax_b.bar(x_b + w_b / 2, psi_pcts_b, w_b, label='Psilocybin', color=col_w)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(task_display, fontsize=11)
    ax_b.set_ylabel('Stable Bursts (%)', fontsize=12)
    ax_b.set_title('(b) Consistency Across Tasks', fontsize=14, fontweight='bold')
    ax_b.set_ylim(0, 115)
    ax_b.legend(fontsize=10)
    ax_b.set_axisbelow(True)
    ax_b.yaxis.grid(True, linestyle='--', alpha=0.3)

    for i, sig in enumerate(sigs_b):
        if sig:
            top = max(ctrl_pcts_b[i], psi_pcts_b[i]) + 2
            ax_b.plot([i - w_b / 2, i - w_b / 2, i + w_b / 2, i + w_b / 2],
                      [top, top + 2, top + 2, top], 'k-', lw=1)
            ax_b.text(i, top + 3, sig, ha='center', va='bottom',
                      fontsize=12, fontweight='bold')

    # ---- (c) Cluster-Specific Burst Counts ----
    c_stable, c_switch, c_labels, c_dominant = [], [], [], []

    for cid in cluster_ids:
        st = cluster_stats[cid]
        s = st['cs'] + st['ps']  # total stable across drugs
        w = st['cw'] + st['pw']  # total switching across drugs
        c_stable.append(s)
        c_switch.append(w)
        c_labels.append(str(cid))
        c_dominant.append('stable' if s >= w else 'switching')

    x_c = np.arange(len(cluster_ids))
    w_c = 0.35

    for i, dom in enumerate(c_dominant):
        bg = '#D6E4F0' if dom == 'stable' else '#F5D6D6'
        ax_c.axvspan(i - 0.48, i + 0.48, color=bg, alpha=0.5, zorder=0)

    ax_c.bar(x_c - w_c / 2, c_stable, w_c, color=col_s, zorder=2)
    ax_c.bar(x_c + w_c / 2, c_switch, w_c, color=col_w, zorder=2)
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(c_labels, fontsize=10)
    ax_c.set_xlabel('Cluster ID', fontsize=12)
    ax_c.set_ylabel('Number of Bursts', fontsize=12)
    ax_c.set_title('(c) Cluster-Specific Burst Counts', fontsize=14, fontweight='bold')
    ax_c.set_axisbelow(True)
    ax_c.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax_c.legend(
        [Patch(facecolor=col_s), Patch(facecolor=col_w),
         Patch(facecolor='#D6E4F0', alpha=0.5), Patch(facecolor='#F5D6D6', alpha=0.5)],
        ['Stable', 'Switching', 'Stable-dominant', 'Switching-dominant'],
        fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_FILE}")
    plt.close('all')


if __name__ == '__main__':
    import io, contextlib

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        run_analysis()

    text = output.getvalue()
    print(text, end='')

    results_file = Path(__file__).parent / 'main_analysis_results.txt'
    results_file.write_text(text)
    print(f"Results saved to: {results_file}")
