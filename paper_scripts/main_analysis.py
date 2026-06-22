#!/usr/bin/env python3
"""Psilocybin polarity analysis: stable vs oscillating recurring sequences.

Loads the back-fitted cluster-activation sequences, removes temporally
overlapping detections (keeping only non-overlapping "pure" bursts), and
quantifies how the balance of stable vs oscillating sequences differs
between control and psilocybin sessions, overall and per task/cluster/group.

Inference is by SUBJECT-LEVEL PERMUTATION (see psiconnect_permutation.py):
a within-subject control/psilocybin sign-flip for the drug effect and a
between-subject label shuffle for trained vs untrained, with Benjamini-
Hochberg FDR. A pooled 2x2 chi-square block is also printed for reference,
but it is anti-conservative -- sequences are nested within participants
(pseudoreplication) -- and is NOT the headline test. Because the two drug
conditions differ in total sequence yield, the figure reports within-
condition proportions, with significance asterisks taken from the
FDR-corrected permutation tests.

Input files (from PSICONNECT_DATA_DIR):
    - sequences_by_cluster.pkl: {cluster_id: [sequence_dicts, ...]}
    - cluster_averages.pkl:     {cluster_id: array} for cluster durations
Outputs (to REGEN_OUTPUT_DIR, else the script directory):
    - main_results.pdf, main_analysis_results.txt
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import stppy
import stppy.io


def _out_dir() -> Path:
    """Output directory for regenerated results/figure.

    Defaults to the script directory (original behaviour); set
    REGEN_OUTPUT_DIR to redirect regenerated outputs elsewhere so the
    committed originals in paper_scripts/ are never overwritten.
    """
    d = os.environ.get('REGEN_OUTPUT_DIR')
    return Path(d) if d else Path(__file__).parent

import numpy as np
from collections import defaultdict

from psiconnect_utils import (
    parse_condition, polarity_summary, sig_stars, benjamini_hochberg,
    DRUG_LABELS, TASK_LABELS, POLARITY_TYPES, BASE_DIR
)
from psiconnect_permutation import (
    within_subject_permutation_test, between_subject_permutation_test
)


def remove_overlapping_bursts(bursts_by_cluster, cluster_averages):
    """
    Remove bursts that temporally overlap with bursts from other clusters.

    Returns a new dict with only non-overlapping ("pure") bursts per cluster.
    """
    cluster_ids = sorted(bursts_by_cluster.keys())

    # Cluster durations from averages (samples / sfreq)
    cluster_durations = {}
    for cid, avg in cluster_averages.items():
        cluster_durations[cid] = avg.shape[1] / 250.0

    min_overlap = min(cluster_durations.values()) if cluster_durations else 0.040

    # Group all bursts by dataset, keeping track of cluster and original index
    bursts_by_dataset = defaultdict(list)
    for cid in cluster_ids:
        for idx, b in enumerate(bursts_by_cluster[cid]):
            bursts_by_dataset[b['dataset']].append({
                'cluster_id': cid,
                'burst_idx': idx,
                'start_time': b['start_time'],
                'end_time': b['end_time'],
            })

    # Find all overlapping burst indices per cluster
    overlapping = {cid: set() for cid in cluster_ids}

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
                    overlapping[b1['cluster_id']].add(b1['burst_idx'])
                    overlapping[b2['cluster_id']].add(b2['burst_idx'])

    # Build clean dict
    clean = {}
    total_orig = 0
    total_removed = 0

    print("OVERLAP FILTERING SUMMARY")
    print("=" * 60)
    print(f"Minimum overlap threshold: {min_overlap*1000:.1f}ms")
    print()

    for cid in cluster_ids:
        orig = bursts_by_cluster[cid]
        n_orig = len(orig)
        n_removed = len(overlapping[cid])
        clean[cid] = [b for idx, b in enumerate(orig) if idx not in overlapping[cid]]
        n_kept = len(clean[cid])

        total_orig += n_orig
        total_removed += n_removed

        print(f"  Cluster {cid:2d}: {n_orig:5d} -> {n_kept:5d} "
              f"(removed {n_removed:4d}, {n_removed/n_orig*100:.1f}%)" if n_orig > 0
              else f"  Cluster {cid:2d}: 0 -> 0")

    print(f"\n  TOTAL:      {total_orig:5d} -> {total_orig - total_removed:5d} "
          f"(removed {total_removed:4d}, {total_removed/total_orig*100:.1f}%)")
    print()

    return clean


POLARITY_RELABEL = {'standing': 'stable', 'rotating': 'oscillating'}


def run_analysis():
    print("Loading data...")
    bursts_by_cluster_raw = stppy.io.load_pickle(BASE_DIR / 'sequences_by_cluster.pkl')
    cluster_averages = stppy.io.load_pickle(BASE_DIR / 'cluster_averages.pkl')

    # Relabel stppy's polarity vocabulary to the paper convention:
    # standing wave -> 'stable', rotating wave -> 'oscillating'.
    for seqs in bursts_by_cluster_raw.values():
        for b in seqs:
            b['polarity'] = POLARITY_RELABEL.get(b.get('polarity'), b.get('polarity'))

    # Remove overlapping bursts
    bursts_by_cluster = remove_overlapping_bursts(bursts_by_cluster_raw, cluster_averages)

    cluster_ids = sorted(bursts_by_cluster.keys())

    polarity_types = ['stable', 'oscillating']
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
    print("STATISTICAL TESTS (Chi-square: stable vs oscillating, ctrl vs psi)")
    print("=" * 80)
    print()

    try:
        from scipy.stats import chi2_contingency
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("WARNING: scipy not available, skipping chi-square tests")

    sw_types = ('stable', 'oscillating')
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

    print("OVERALL (stable vs oscillating, ctrl vs psi):")
    print(f"  Control:    stable={cs_all} ({cs_pct:.1f}%), oscillating={cw_all} ({cw_pct:.1f}%), n={len(ctrl_bin)}")
    print(f"  Psilocybin: stable={ps_all} ({ps_pct:.1f}%), oscillating={pw_all} ({pw_pct:.1f}%), n={len(psi_bin)}")
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
        print(f"    Control:    stable={cs} ({cs_p:.1f}%), oscillating={cw}, n={len(ct)}")
        print(f"    Psilocybin: stable={ps} ({ps_p:.1f}%), oscillating={pw}, n={len(pt)}")

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

    # --- Total sequences per task per drug (all polarity types) for figure (b) ---
    task_totals = {}
    for task in task_order:
        ct_all = [b for b in all_bursts if b['_drug'] == 'ctrl' and b['_task'] == task]
        pt_all = [b for b in all_bursts if b['_drug'] == 'psi' and b['_task'] == task]
        task_totals[task] = {'n_ctrl_all': len(ct_all), 'n_psi_all': len(pt_all)}

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
        print(f"    Control:    stable={cs} ({cs_p:.1f}%), oscillating={cw}, n={len(ct)}")
        print(f"    Psilocybin: stable={ps} ({ps_p:.1f}%), oscillating={pw}, n={len(pt)}")

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

    # --- Trained vs Untrained ---
    # Trained subjects: sub2XX, Untrained: sub0XX
    def is_trained(b):
        subj = b.get('_subj', '')
        return len(subj) > 3 and subj[3] == '2'

    trained_bin = [b for b in binary_bursts if is_trained(b)]
    untrained_bin = [b for b in binary_bursts if not is_trained(b)]

    ts_all = sum(1 for b in trained_bin if b['polarity'] == 'stable')
    tw_all = len(trained_bin) - ts_all
    us_all = sum(1 for b in untrained_bin if b['polarity'] == 'stable')
    uw_all = len(untrained_bin) - us_all

    # Per-drug breakdown for trained/untrained
    trained_ctrl = [b for b in trained_bin if b['_drug'] == 'ctrl']
    trained_psi  = [b for b in trained_bin if b['_drug'] == 'psi']
    untrained_ctrl = [b for b in untrained_bin if b['_drug'] == 'ctrl']
    untrained_psi  = [b for b in untrained_bin if b['_drug'] == 'psi']

    tc_s = sum(1 for b in trained_ctrl if b['polarity'] == 'stable')
    tc_r = len(trained_ctrl) - tc_s
    tp_s = sum(1 for b in trained_psi if b['polarity'] == 'stable')
    tp_r = len(trained_psi) - tp_s
    uc_s = sum(1 for b in untrained_ctrl if b['polarity'] == 'stable')
    uc_r = len(untrained_ctrl) - uc_s
    up_s = sum(1 for b in untrained_psi if b['polarity'] == 'stable')
    up_r = len(untrained_psi) - up_s

    p_tu = None
    sig_trained = ''
    if has_scipy and trained_bin and untrained_bin:
        chi2_tu, p_tu, dof_tu, _ = chi2_contingency([[ts_all, tw_all], [us_all, uw_all]])
        sig_trained = sig_stars(p_tu)
        test_labels.append('Trained vs Untrained')
        test_pvalues.append(p_tu)

    print("TRAINED vs UNTRAINED (stable vs oscillating):")
    n_trained = len(trained_bin)
    n_untrained = len(untrained_bin)
    ts_pct = ts_all / n_trained * 100 if n_trained else 0
    us_pct = us_all / n_untrained * 100 if n_untrained else 0
    print(f"  Trained:   stable={ts_all} ({ts_pct:.1f}%), oscillating={tw_all}, n={n_trained}")
    print(f"  Untrained: stable={us_all} ({us_pct:.1f}%), oscillating={uw_all}, n={n_untrained}")
    if p_tu is not None:
        print(f"  Chi-square={chi2_tu:.2f}, df={dof_tu}, p={p_tu:.6f} {sig_trained}")
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
    # SUBJECT-LEVEL PERMUTATION TESTS (Route A)
    # ------------------------------------------------------------------
    # The chi-square tests above pool bursts across subjects and treat
    # each burst as independent. Bursts are nested within subjects, so
    # that is pseudoreplication and the p-values are anti-conservative.
    # These permutation tests respect the design:
    #   - drug = within-subject crossover  -> per-subject ctrl/psi sign-flip
    #   - trained/untrained = between-subject -> shuffle group across subjects
    # Drug tests use ONLY subjects present in both arms of the slice;
    # single-arm subjects are dropped and reported (not silently removed).
    # ==================================================================
    N_PERM = 10000
    PERM_SEED = 0

    print("-" * 80)
    print("SUBJECT-LEVEL PERMUTATION TESTS (Route A)")
    print(f"  {N_PERM} permutations; drug=within-subject sign-flip, "
          f"trained/untrained=between-subject label shuffle")
    print("-" * 80)

    def _drug_records(bursts):
        return [{'subj': b['_subj'], 'drug': b['_drug'],
                 'standing': b['polarity'] == 'stable'} for b in bursts]

    perm_labels = []
    perm_results = []

    # Overall (within-subject drug effect)
    res = within_subject_permutation_test(
        _drug_records(binary_bursts), n_perm=N_PERM, seed=PERM_SEED)
    perm_labels.append('Overall')
    perm_results.append(res)

    # Per task (within-subject drug effect)
    for task in task_order:
        tname = task_display_map.get(task, task)
        recs = _drug_records([b for b in binary_bursts if b['_task'] == task])
        res = within_subject_permutation_test(recs, n_perm=N_PERM, seed=PERM_SEED)
        perm_labels.append(f'Task: {tname}')
        perm_results.append(res)

    # Per cluster (within-subject drug effect)
    for cid in cluster_ids:
        bc = [b for b in bursts_by_cluster[cid] if b['polarity'] in sw_types]
        res = within_subject_permutation_test(
            _drug_records(bc), n_perm=N_PERM, seed=PERM_SEED)
        if res.n_units == 0:
            continue  # no paired subjects in this cluster -> skip
        perm_labels.append(f'Cluster {cid}')
        perm_results.append(res)

    # Trained vs Untrained (between-subject)
    tu_records = [{'subj': b['_subj'],
                   'group': 'trained' if is_trained(b) else 'untrained',
                   'standing': b['polarity'] == 'stable'} for b in binary_bursts]
    res_tu = between_subject_permutation_test(
        tu_records, n_perm=N_PERM, seed=PERM_SEED)
    perm_labels.append('Trained vs Untrained')
    perm_results.append(res_tu)

    # FDR across the permutation p-values
    perm_p = np.array([r.p_value for r in perm_results])
    perm_fdr = benjamini_hochberg(perm_p)

    print(f"  {'Test':<25s} {'chi2':>9s} {'p (perm)':>10s} {'p (FDR)':>10s} "
          f"{'units':>6s} {'drop':>5s} {'n':>6s} {'raw':>5s} {'FDR':>5s}")
    print(f"  {'-'*25} {'-'*9} {'-'*10} {'-'*10} {'-'*6} {'-'*5} {'-'*6} {'-'*5} {'-'*5}")
    for i, label in enumerate(perm_labels):
        r = perm_results[i]
        print(f"  {label:<25s} {r.observed:>9.2f} {r.p_value:>10.4f} "
              f"{perm_fdr[i]:>10.4f} {r.n_units:>6d} {r.n_dropped:>5d} "
              f"{r.n_bursts:>6d} {sig_stars(r.p_value):>5s} "
              f"{sig_stars(perm_fdr[i]):>5s}")
    print()
    n_sig_perm = sum(1 for p in perm_p if p < 0.05)
    n_sig_perm_fdr = sum(1 for p in perm_fdr if p < 0.05)
    print(f"  Significant (p<0.05): {n_sig_perm}/{len(perm_p)} permutation, "
          f"{n_sig_perm_fdr}/{len(perm_fdr)} permutation+FDR")
    print("  Note: 'units' = paired subjects (drug tests) or subjects "
          "(trained/untrained); 'drop' = single-arm subjects excluded.")
    print()

    # Permutation p-values used for the FIGURE significance asterisks. We use the
    # FDR-CORRECTED permutation p-values so the figure matches the multiplicity-
    # corrected inference reported in the text (raw p would over-state the
    # per-task panel). Overall (panel a) and per-task (panel b) FDR values are
    # taken from the main family above (perm_fdr). Panel c shows the within-
    # subject drug effect computed separately within trained and untrained
    # participants; those two tests are FDR-corrected as their own pair.
    perm_fig_p = {'overall': perm_fdr[0]}
    for i, task in enumerate(task_order):
        perm_fig_p[f'task:{task}'] = perm_fdr[1 + i]
    td_raw = within_subject_permutation_test(
        _drug_records(trained_bin), n_perm=N_PERM, seed=PERM_SEED).p_value
    ud_raw = within_subject_permutation_test(
        _drug_records(untrained_bin), n_perm=N_PERM, seed=PERM_SEED).p_value
    td_fdr, ud_fdr = benjamini_hochberg(np.array([td_raw, ud_raw]))
    perm_fig_p['trained_drug'] = td_fdr
    perm_fig_p['untrained_drug'] = ud_fdr
    print(f"  Panel-c within-group drug effect (FDR-corrected pair): "
          f"trained p_raw={td_raw:.4f} p_FDR={td_fdr:.4f}; "
          f"untrained p_raw={ud_raw:.4f} p_FDR={ud_fdr:.4f}")

    # ==================================================================
    # FIGURE: MAIN RESULTS (CLEAN)
    # ==================================================================
    print("=" * 80)
    print("GENERATING FIGURE: MAIN RESULTS (CLEAN)")
    print("=" * 80)

    import matplotlib.pyplot as plt
    plt.rcParams['ytick.labelsize'] = 15
    from matplotlib.patches import Patch
    from scipy.stats import chisquare

    OUTPUT_FILE = _out_dir() / 'main_results.pdf'

    # Colors
    col_ctrl = '#2166AC'       # blue for control (panels a-c)
    col_psi  = '#D6604D'       # red for psilocybin (panels a-c)
    col_standing = '#2166AC'   # blue for standing (panel d)
    col_rotating = '#D6604D'   # red-orange for rotating (panel d)

    fig, axes = plt.subplots(2, 2, figsize=(20, 12),
                             gridspec_kw={'width_ratios': [1, 2]})
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # Bars are normalized to PERCENTAGES WITHIN each drug group (control /
    # psilocybin) so the ~2:1 data imbalance (control has ~2x the sequences
    # of psilocybin) does not distort the visual comparison. Significance
    # asterisks come from the FDR-corrected SUBJECT-LEVEL PERMUTATION tests
    # (perm_fig_p), not the pseudoreplicated chi-square, so the figure matches
    # the multiplicity-corrected headline inference.
    def _pct_within(stand, rot):
        tot = stand + rot
        if tot == 0:
            return 0.0, 0.0
        return 100.0 * stand / tot, 100.0 * rot / tot

    def _proportion_p(a_stand, a_rot, b_stand, b_rot):
        """2x2 chi-square p: (ctrl/psi) x (standing/rotating). None if degenerate."""
        from scipy.stats import chi2_contingency
        if (a_stand + a_rot) == 0 or (b_stand + b_rot) == 0:
            return None
        if (a_stand + b_stand) == 0 or (a_rot + b_rot) == 0:
            return None
        try:
            _, p, _, _ = chi2_contingency([[a_stand, a_rot], [b_stand, b_rot]])
            return p
        except ValueError:
            return None

    # Helper: draw grouped split bars (percentages within group) with the
    # ctrl-vs-psi proportion test annotated on each polarity group.
    def _plot_split_panel(ax, label_a, count_a_stand, count_a_rot,
                          label_b, count_b_stand, count_b_rot,
                          title, sig_p=None):
        w = 0.3
        a_stand_pct, a_rot_pct = _pct_within(count_a_stand, count_a_rot)
        b_stand_pct, b_rot_pct = _pct_within(count_b_stand, count_b_rot)

        # Standing group
        ax.bar(0 - w/2, a_stand_pct, w, color=col_ctrl, zorder=2, label=f'{label_a}')
        ax.bar(0 + w/2, b_stand_pct, w, color=col_psi, zorder=2, label=f'{label_b}')
        # Rotating group
        ax.bar(1 - w/2, a_rot_pct, w, color=col_ctrl, zorder=2)
        ax.bar(1 + w/2, b_rot_pct, w, color=col_psi, zorder=2)

        for x, pct in [(0 - w/2, a_stand_pct), (0 + w/2, b_stand_pct),
                       (1 - w/2, a_rot_pct), (1 + w/2, b_rot_pct)]:
            ax.text(x, pct + 1.5, f'{pct:.1f}%', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Stable', 'Oscillating'], fontsize=18)
        ax.set_ylabel('Percentage within group (%)', fontsize=18)
        ax.set_title(title, fontsize=21, fontweight='bold')
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=15, loc='upper center', ncol=2, framealpha=0.9)

        if sig_p is not None:
            sig_g = sig_stars(sig_p)
            for gx, va, vb in [(0, a_stand_pct, b_stand_pct),
                               (1, a_rot_pct, b_rot_pct)]:
                top = max(va, vb)
                bracket_y = top + 6
                ax.plot([gx - w/2, gx - w/2, gx + w/2, gx + w/2],
                        [bracket_y, bracket_y + 1.5, bracket_y + 1.5, bracket_y],
                        'k-', lw=1.5, clip_on=False)
                ax.text(gx, bracket_y + 2.5, sig_g, ha='center', va='bottom',
                        fontsize=16, fontweight='bold')

        ax.set_ylim(0, 130)

    # ---- (a) Control vs Psilocybin (split) ----
    _plot_split_panel(ax_a, 'Control', cs_all, cw_all,
                      'Psilocybin', ps_all, pw_all,
                      '(a) Control vs Psilocybin',
                      sig_p=perm_fig_p['overall'])

    # ---- (b) Consistency Across Tasks (split: Standing | Rotating) ----
    task_display = ['Meditation', 'Music', 'Rest', 'Video']
    w_b = 0.18

    ctrl_stable_vals, ctrl_switch_vals = [], []
    psi_stable_vals, psi_switch_vals = [], []

    for task in task_order:
        ts = task_stats[task]
        ctrl_stable_vals.append(ts['cs'])
        ctrl_switch_vals.append(ts['cw'])
        psi_stable_vals.append(ts['ps'])
        psi_switch_vals.append(ts['pw'])

    # For each task: two groups (Standing, Rotating), each with Ctrl + Psi bars
    # x layout: task_i*3 + 0 = Standing group, task_i*3 + 1 = Rotating group
    spacing = 2.5

    for i, task in enumerate(task_order):
        cs_v, cw_v = ctrl_stable_vals[i], ctrl_switch_vals[i]
        ps_v, pw_v = psi_stable_vals[i], psi_switch_vals[i]
        base = i * spacing

        cs_p, cw_p = _pct_within(cs_v, cw_v)   # control standing/rotating %
        ps_p, pw_p = _pct_within(ps_v, pw_v)   # psilocybin standing/rotating %

        # Standing group: Ctrl blue, Psi red
        ax_b.bar(base - w_b/2, cs_p, w_b, color=col_ctrl, zorder=2,
                 label='Control' if i == 0 else None)
        ax_b.bar(base + w_b/2, ps_p, w_b, color=col_psi, zorder=2,
                 label='Psilocybin' if i == 0 else None)

        # Rotating group: Ctrl blue, Psi red
        ax_b.bar(base + 1 - w_b/2, cw_p, w_b, color=col_ctrl, zorder=2)
        ax_b.bar(base + 1 + w_b/2, pw_p, w_b, color=col_psi, zorder=2)

        # within-subject drug permutation p per task, annotated on both groups
        p_prop = perm_fig_p.get(f'task:{task}')
        if p_prop is not None:
            sig_g = sig_stars(p_prop)
            for gx, va, vb in [(base, cs_p, ps_p), (base + 1, cw_p, pw_p)]:
                top = max(va, vb)
                bracket_y = top + 5
                ax_b.plot([gx - w_b/2, gx - w_b/2, gx + w_b/2, gx + w_b/2],
                          [bracket_y, bracket_y + 1.5, bracket_y + 1.5, bracket_y],
                          'k-', lw=1, clip_on=False)
                ax_b.text(gx, bracket_y + 2, sig_g,
                          ha='center', va='bottom', fontsize=16, fontweight='bold')

    # X-axis: label each task with Standing/Rotating sublabels
    task_centers = [i * spacing + 0.5 for i in range(len(task_order))]
    ax_b.set_xticks([i * spacing for i in range(len(task_order))] +
                    [i * spacing + 1 for i in range(len(task_order))])
    sublabels = []
    for t in task_display:
        sublabels.append('S')
        sublabels.append('O')
    # Interleave: S, R for each task
    tick_positions = []
    for i in range(len(task_order)):
        tick_positions.append(i * spacing)
        tick_positions.append(i * spacing + 1)
    ax_b.set_xticks(tick_positions)
    ax_b.set_xticklabels(sublabels, fontsize=16)

    # Add task names as secondary labels below
    for i, tname in enumerate(task_display):
        ax_b.text(i * spacing + 0.5, -0.10, tname,
                  ha='center', va='top', fontsize=16, fontweight='bold',
                  transform=ax_b.get_xaxis_transform())

    ax_b.set_ylabel('Percentage within group (%)', fontsize=18)
    ax_b.set_title('(b) Consistency Across Tasks', fontsize=21, fontweight='bold')
    ax_b.set_ylim(0, 130)
    ax_b.legend(fontsize=15, loc='upper center', ncol=2, framealpha=0.9)
    ax_b.set_axisbelow(True)
    ax_b.yaxis.grid(True, linestyle='--', alpha=0.3)

    # ---- (c) Trained vs Untrained (split by Ctrl/Psi, like panel b) ----
    tu_labels = ['Trained', 'Untrained']
    tu_cs = [tc_s, uc_s]   # ctrl standing
    tu_cr = [tc_r, uc_r]   # ctrl rotating
    tu_ps = [tp_s, up_s]   # psi standing
    tu_pr = [tp_r, up_r]   # psi rotating

    w_c = 0.18
    spacing_c = 2.5

    for i, tl in enumerate(tu_labels):
        base = i * spacing_c
        cs_p, cr_p = _pct_within(tu_cs[i], tu_cr[i])   # control standing/rotating %
        ps_p, pr_p = _pct_within(tu_ps[i], tu_pr[i])   # psilocybin standing/rotating %

        # Standing group: Ctrl blue, Psi red
        ax_c.bar(base - w_c/2, cs_p, w_c, color=col_ctrl, zorder=2,
                 label='Control' if i == 0 else None)
        ax_c.bar(base + w_c/2, ps_p, w_c, color=col_psi, zorder=2,
                 label='Psilocybin' if i == 0 else None)

        # Rotating group: Ctrl blue, Psi red
        ax_c.bar(base + 1 - w_c/2, cr_p, w_c, color=col_ctrl, zorder=2)
        ax_c.bar(base + 1 + w_c/2, pr_p, w_c, color=col_psi, zorder=2)

        # within-subject drug permutation p, restricted to this training group
        p_prop = perm_fig_p['trained_drug'] if i == 0 else perm_fig_p['untrained_drug']
        if p_prop is not None:
            sig_g = sig_stars(p_prop)
            for gx, va, vb in [(base, cs_p, ps_p), (base + 1, cr_p, pr_p)]:
                top = max(va, vb)
                bracket_y = top + 5
                ax_c.plot([gx - w_c/2, gx - w_c/2, gx + w_c/2, gx + w_c/2],
                          [bracket_y, bracket_y + 1.5, bracket_y + 1.5, bracket_y],
                          'k-', lw=1, clip_on=False)
                ax_c.text(gx, bracket_y + 2, sig_g,
                          ha='center', va='bottom', fontsize=16, fontweight='bold')

    # X-axis: S/O sublabels + group names below
    tick_positions_c = []
    sublabels_c = []
    for i in range(len(tu_labels)):
        tick_positions_c.append(i * spacing_c)
        tick_positions_c.append(i * spacing_c + 1)
        sublabels_c.extend(['S', 'O'])
    ax_c.set_xticks(tick_positions_c)
    ax_c.set_xticklabels(sublabels_c, fontsize=16)

    for i, tl in enumerate(tu_labels):
        ax_c.text(i * spacing_c + 0.5, -0.10, tl,
                  ha='center', va='top', fontsize=16, fontweight='bold',
                  transform=ax_c.get_xaxis_transform())

    ax_c.set_ylabel('Percentage within group (%)', fontsize=18)
    ax_c.set_title('(c) Trained vs Untrained', fontsize=21, fontweight='bold')
    ax_c.set_ylim(0, 130)
    ax_c.legend(fontsize=15, loc='upper center', ncol=2, framealpha=0.9)
    ax_c.set_axisbelow(True)
    ax_c.yaxis.grid(True, linestyle='--', alpha=0.3)

    # ---- (d) Cluster-Specific Sequence Counts ----
    undefined_clusters = {0, 1, 6, 8}

    c_labels, c_dominant = [], []
    c_cs, c_ps, c_cw, c_pw = [], [], [], []

    for cid in cluster_ids:
        st = cluster_stats[cid]
        c_cs.append(st['cs'])   # ctrl standing
        c_ps.append(st['ps'])   # psi standing
        c_cw.append(st['cw'])   # ctrl rotating
        c_pw.append(st['pw'])   # psi rotating
        c_labels.append(str(cid))
        s = st['cs'] + st['ps']
        w = st['cw'] + st['pw']
        if cid in undefined_clusters:
            c_dominant.append('undefined')
        elif s >= w:
            c_dominant.append('standing')
        else:
            c_dominant.append('rotating')

    x_d = np.arange(len(cluster_ids))
    w_d = 0.35

    # Normalize each drug to % of ITS OWN total sequences (control vs psilocybin),
    # so the ~2:1 control/psilocybin imbalance does not dominate the per-cluster view.
    ctrl_total = (sum(c_cs) + sum(c_cw)) or 1
    psi_total = (sum(c_ps) + sum(c_pw)) or 1
    c_cs_p = np.array(c_cs) / ctrl_total * 100   # ctrl standing % of all ctrl
    c_cw_p = np.array(c_cw) / ctrl_total * 100   # ctrl rotating % of all ctrl
    c_ps_p = np.array(c_ps) / psi_total * 100    # psi standing % of all psi
    c_pw_p = np.array(c_pw) / psi_total * 100    # psi rotating % of all psi

    # Stable bars: ctrl (solid blue) + psi (hatched blue)
    ax_d.bar(x_d - w_d / 2, c_cs_p, w_d, color=col_standing, zorder=2,
             label='Stable (Ctrl)')
    ax_d.bar(x_d - w_d / 2, c_ps_p, w_d, bottom=c_cs_p, color=col_standing,
             alpha=0.6, hatch='//', edgecolor='white', zorder=2,
             label='Stable (Psi)')

    # Oscillating bars: ctrl (solid red) + psi (hatched red)
    ax_d.bar(x_d + w_d / 2, c_cw_p, w_d, color=col_rotating, zorder=2,
             label='Oscillating (Ctrl)')
    ax_d.bar(x_d + w_d / 2, c_pw_p, w_d, bottom=c_cw_p, color=col_rotating,
             alpha=0.6, hatch='//', edgecolor='white', zorder=2,
             label='Oscillating (Psi)')

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(c_labels, fontsize=16)
    ax_d.set_xlabel('Cluster ID', fontsize=18)
    ax_d.set_ylabel("Percentage of group's sequences (%)", fontsize=18)
    ax_d.set_title('(d) Cluster-Specific Sequence Distribution', fontsize=21, fontweight='bold')
    ax_d.set_axisbelow(True)
    ax_d.yaxis.grid(True, linestyle='--', alpha=0.3)

    legend_elements = [
        Patch(facecolor=col_standing, label='Stable (Ctrl)'),
        Patch(facecolor=col_standing, alpha=0.6, hatch='//', edgecolor='white',
              label='Stable (Psi)'),
        Patch(facecolor=col_rotating, label='Oscillating (Ctrl)'),
        Patch(facecolor=col_rotating, alpha=0.6, hatch='//', edgecolor='white',
              label='Oscillating (Psi)'),
    ]
    ax_d.set_ylim(0, max(1.0, (c_cs_p + c_ps_p).max(), (c_cw_p + c_pw_p).max()) * 1.28)
    ax_d.legend(handles=legend_elements, fontsize=15, loc='upper center', ncol=2,
                framealpha=0.9)

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

    results_file = _out_dir() / 'main_analysis_results.txt'
    results_file.write_text(text)
    print(f"Results saved to: {results_file}")
