#!/usr/bin/env python3
"""
Per-Participant Polarity Shift Analysis

Checks whether the group-level finding (psi increases stable%, decreases switching%)
holds at the individual participant level.

Expected directions:
  - Non-video tasks (meditation, music, rest): S↑ SW↓ under psilocybin
  - Video task (opposite): S↓ SW↑ under psilocybin

Input files (from parent directory):
    - bursts_by_cluster.pkl: {cluster_id: [burst_dicts, ...]}

Output:
    - participant_shift.pdf
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from matplotlib.patches import Patch

from psiconnect_utils import parse_condition, BASE_DIR

# Paths
OUTPUT_FILE = Path(__file__).parent / 'participant_shift.pdf'


def load_data():
    """Load all required pickle files."""
    with open(BASE_DIR / 'bursts_by_cluster.pkl', 'rb') as f:
        bursts_by_cluster = pickle.load(f)
    return bursts_by_cluster


def run_analysis():
    print("Loading data...")
    bursts_by_cluster = load_data()

    cluster_ids = sorted(bursts_by_cluster.keys())

    # Tag every burst with drug/subject/task
    for cid in cluster_ids:
        for b in bursts_by_cluster[cid]:
            b['_drug'], b['_subj'], b['_task'] = parse_condition(b['dataset'])

    all_bursts = [b for bursts in bursts_by_cluster.values() for b in bursts]

    # ==================================================================
    # ANALYSIS: PER-PARTICIPANT POLARITY SHIFT
    # ==================================================================
    print("=" * 80)
    print("PER-PARTICIPANT POLARITY SHIFT ANALYSIS")
    print("=" * 80)
    print("Expected: S↑ SW↓ (psi increases stable, decreases switching)")
    print("Video (opposite): S↓ SW↑\n")

    # Group ALL bursts by (subject, task, drug) — no filtering
    subj_bursts = defaultdict(list)
    for b in all_bursts:
        subj_bursts[(b['_subj'], b['_task'], b['_drug'])].append(b)

    subj_task_pairs = sorted({(b['_subj'], b['_task']) for b in all_bursts})

    cat_keys = ['yes_both', 'yes_s', 'yes_sw',
                'no_both', 'no_s', 'no_sw', 'no_change', 'no_data']
    categories = {k: [] for k in cat_keys}
    deltas = {}  # (subj, task) → (delta_s, delta_sw) percentages
    deltas_abs = {}  # (subj, task) → (delta_s_count, delta_sw_count) raw counts

    for subj, task in subj_task_pairs:
        ctrl_list = subj_bursts.get((subj, task, 'ctrl'), [])
        psi_list = subj_bursts.get((subj, task, 'psi'), [])

        if not ctrl_list and not psi_list:
            categories['no_data'].append((subj, task))
            continue

        # 0 bursts → 0% stable, 0% switching
        ctrl_s = sum(1 for b in ctrl_list if b['polarity'] == 'stable') / len(ctrl_list) * 100 if ctrl_list else 0
        ctrl_sw = sum(1 for b in ctrl_list if b['polarity'] == 'switching') / len(ctrl_list) * 100 if ctrl_list else 0
        psi_s = sum(1 for b in psi_list if b['polarity'] == 'stable') / len(psi_list) * 100 if psi_list else 0
        psi_sw = sum(1 for b in psi_list if b['polarity'] == 'switching') / len(psi_list) * 100 if psi_list else 0

        delta_s = psi_s - ctrl_s
        delta_sw = psi_sw - ctrl_sw
        deltas[(subj, task)] = (delta_s, delta_sw)

        # Absolute burst count deltas (for panel c)
        ctrl_s_n = sum(1 for b in ctrl_list if b['polarity'] == 'stable')
        ctrl_sw_n = sum(1 for b in ctrl_list if b['polarity'] == 'switching')
        psi_s_n = sum(1 for b in psi_list if b['polarity'] == 'stable')
        psi_sw_n = sum(1 for b in psi_list if b['polarity'] == 'switching')
        deltas_abs[(subj, task)] = (psi_s_n - ctrl_s_n, psi_sw_n - ctrl_sw_n)

        if task == 'video':
            s_ok = delta_s < 0       # expected: S↓
            sw_ok = delta_sw > 0     # expected: SW↑
            s_wrong = delta_s > 0
            sw_wrong = delta_sw < 0
        else:
            s_ok = delta_s > 0       # expected: S↑
            sw_ok = delta_sw < 0     # expected: SW↓
            s_wrong = delta_s < 0
            sw_wrong = delta_sw > 0

        # Check for no change first (both deltas are exactly 0)
        if delta_s == 0 and delta_sw == 0:
            categories['no_change'].append((subj, task))
        elif s_ok and sw_ok:
            categories['yes_both'].append((subj, task))
        elif s_ok:
            categories['yes_s'].append((subj, task))
        elif sw_ok:
            categories['yes_sw'].append((subj, task))
        elif s_wrong and sw_wrong:
            categories['no_both'].append((subj, task))
        elif s_wrong:
            categories['no_s'].append((subj, task))
        elif sw_wrong:
            categories['no_sw'].append((subj, task))
        else:
            categories['no_change'].append((subj, task))  # one changed, one stayed same

    # Print per-group summaries
    groups = [
        ('Med + Music + Rest', ['meditation', 'music', 'restingstate'],
         'S↑ SW↓', 'S↑', 'SW↓', 'S↓ SW↑', 'S↓', 'SW↑'),
        ('Video (opposite)', ['video'],
         'S↓ SW↑', 'S↓', 'SW↑', 'S↑ SW↓', 'S↑', 'SW↓'),
    ]

    for group_name, group_tasks, yb, ys, ysw, nb, ns, nsw in groups:
        pairs_in_group = [(s, t) for s, t in subj_task_pairs if t in group_tasks]
        n_total = len(pairs_in_group)
        n_nd = sum(1 for p in pairs_in_group if p in categories['no_data'])
        n_paired = n_total - n_nd
        n_yes = sum(1 for p in pairs_in_group
                    if p in categories['yes_both'] or p in categories['yes_s'] or p in categories['yes_sw'])

        print(f"{group_name} ({n_paired} paired, {n_nd} no data):")
        if n_paired > 0:
            print(f"  Expected shift: {n_yes}/{n_paired} ({n_yes/n_paired*100:.1f}%)\n")

        cat_info = [
            ('yes_both', f'YES {yb}'), ('yes_s', f'YES {ys} only'),
            ('yes_sw', f'YES {ysw} only'),
            ('no_both', f'NO  {nb}'), ('no_s', f'NO  {ns} only'),
            ('no_sw', f'NO  {nsw} only'),
            ('no_change', 'NO CHANGE (Δ=0)'),
        ]

        print(f"  {'Category':25s} | {'Count':>5s} | {'%':>6s}")
        print(f"  {'-'*43}")
        for key, label in cat_info:
            c = sum(1 for p in pairs_in_group if p in categories[key])
            pct = c / n_paired * 100 if n_paired > 0 else 0
            print(f"  {label:25s} | {c:5d} | {pct:5.1f}%")
        print(f"  {'No data':25s} | {n_nd:5d} |     —")
        print()

    print()

    # ==================================================================
    # FIGURE: PER-PARTICIPANT POLARITY SHIFT
    # ==================================================================
    print("=" * 80)
    print("GENERATING FIGURE: PER-PARTICIPANT POLARITY SHIFT")
    print("=" * 80)

    from scipy.stats import binomtest, mannwhitneyu

    # Font sizes
    TITLE_SIZE = 13
    LABEL_SIZE = 11
    TICK_SIZE = 10

    cat_sets = {k: set(v) for k, v in categories.items()}

    div_groups = [
        ('Meditation\n(S↑ SW↓)', ['meditation']),
        ('Music\n(S↑ SW↓)', ['music']),
        ('Resting state\n(S↑ SW↓)', ['restingstate']),
        ('Video\n(S↓ SW↑)', ['video']),
    ]

    # Count expected / opposite per group
    div_data = []
    for _, gtasks in div_groups:
        exp = np.array([
            sum(1 for _, t in cat_sets['yes_both'] if t in gtasks),
            sum(1 for _, t in cat_sets['yes_s'] if t in gtasks),
            sum(1 for _, t in cat_sets['yes_sw'] if t in gtasks),
        ])
        opp = np.array([
            sum(1 for _, t in cat_sets['no_both'] if t in gtasks),
            sum(1 for _, t in cat_sets['no_s'] if t in gtasks),
            sum(1 for _, t in cat_sets['no_sw'] if t in gtasks),
        ])
        div_data.append((exp, opp))

    fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=(12, 18),
                                             gridspec_kw={'height_ratios': [1, 1, 1],
                                                          'hspace': 0.35})

    # ---- (a) Diverging bar chart per task ----
    y = np.arange(len(div_groups))
    h = 0.45

    col_exp = ['#2ca02c', '#66c266', '#a3d9a3']
    col_opp = ['#d62728', '#e87c7d', '#f2b4b4']
    lbl_sub = ['S + SW', 'S only', 'SW only']

    # Expected direction (right)
    left_r = np.zeros(len(div_groups))
    for i in range(3):
        vals = np.array([d[0][i] for d in div_data])
        ax_a.barh(y, vals, left=left_r, height=h, color=col_exp[i],
                  label=f'Expected: {lbl_sub[i]}')
        for j, v in enumerate(vals):
            if v > 1:
                ax_a.text(left_r[j] + v / 2, j, str(v), ha='center', va='center',
                          fontsize=10, fontweight='bold',
                          color='white' if i == 0 else 'black')
        left_r += vals

    # Opposite direction (left)
    left_l = np.zeros(len(div_groups))
    for i in range(3):
        vals = np.array([d[1][i] for d in div_data])
        ax_a.barh(y, -vals, left=-left_l, height=h, color=col_opp[i],
                  label=f'Opposite: {lbl_sub[i]}')
        for j, v in enumerate(vals):
            if v > 1:
                ax_a.text(-(left_l[j] + v / 2), j, str(v), ha='center', va='center',
                          fontsize=10, fontweight='bold',
                          color='white' if i == 0 else 'black')
        left_l += vals

    ax_a.axvline(0, color='black', linewidth=1, zorder=3)
    ax_a.text(-2, -1.0, '← Opposite',
              ha='right', va='top', fontsize=10, fontstyle='italic', color='#888888')
    ax_a.text(2, -1.0, 'Expected →',
              ha='left', va='top', fontsize=10, fontstyle='italic', color='#888888')

    ax_a.set_yticks(y)
    ax_a.set_yticklabels([name for name, _ in div_groups], fontsize=TICK_SIZE)
    ax_a.set_xlabel('Number of subject × task pairs', fontsize=LABEL_SIZE)
    ax_a.set_title('$\\mathbf{(a)}$ Per-Task Breakdown', fontsize=TITLE_SIZE)

    x_max = max(left_r.max(), left_l.max()) + 10
    ax_a.set_xlim(-x_max, x_max)
    ticks = ax_a.get_xticks()
    ax_a.set_xticks(ticks)
    ax_a.set_xticklabels([str(int(abs(t))) for t in ticks])

    ax_a.legend(loc='upper right', fontsize=7, ncol=1)
    ax_a.set_axisbelow(True)
    ax_a.xaxis.grid(True, linestyle='--', alpha=0.3)

    # ---- (b) Aggregated histogram: non-video vs video ----
    # div_data order: [Meditation, Music, Resting state, Video]
    # non-video = first 3, video = last 1
    nonvideo_exp = sum(d[0] for d in div_data[:3])
    nonvideo_opp = sum(d[1] for d in div_data[:3])
    video_exp = div_data[3][0]
    video_opp = div_data[3][1]

    # 6 groups: 3 categories x 2 task groups
    x_labels_b = ['S+SW\n(non-video)', 'S+SW\n(video)',
                  'S only\n(non-video)', 'S only\n(video)',
                  'SW only\n(non-video)', 'SW only\n(video)']
    exp_vals_b = [nonvideo_exp[0], video_exp[0],
                  nonvideo_exp[1], video_exp[1],
                  nonvideo_exp[2], video_exp[2]]
    opp_vals_b = [nonvideo_opp[0], video_opp[0],
                  nonvideo_opp[1], video_opp[1],
                  nonvideo_opp[2], video_opp[2]]

    x_b = np.arange(6)
    w_b = 0.35
    ax_b.bar(x_b - w_b / 2, exp_vals_b, w_b, label='Expected', color='#2ca02c')
    ax_b.bar(x_b + w_b / 2, opp_vals_b, w_b, label='Opposite', color='#d62728')

    # Count labels
    for i in range(6):
        ax_b.text(i - w_b / 2, exp_vals_b[i] + 0.5, str(exp_vals_b[i]),
                  ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_b.text(i + w_b / 2, opp_vals_b[i] + 0.5, str(opp_vals_b[i]),
                  ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Significance tests (binomial)
    for i in range(6):
        n_total = exp_vals_b[i] + opp_vals_b[i]
        if n_total > 0:
            result = binomtest(int(exp_vals_b[i]), int(n_total), p=0.5)
            p = result.pvalue
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        else:
            sig = ''

        if sig:
            top = max(exp_vals_b[i], opp_vals_b[i]) + 3
            ax_b.plot([i - w_b / 2, i - w_b / 2, i + w_b / 2, i + w_b / 2],
                      [top, top + 1.5, top + 1.5, top], 'k-', lw=1)
            ax_b.text(i, top + 2, sig, ha='center', va='bottom',
                      fontsize=10, fontweight='bold')

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(x_labels_b, fontsize=9)
    ax_b.set_ylabel('Total subject × task pairs', fontsize=LABEL_SIZE)
    ax_b.set_title('$\\mathbf{(b)}$ Aggregated: Non-Video vs Video', fontsize=TITLE_SIZE)
    ax_b.legend(fontsize=10)
    ax_b.set_axisbelow(True)
    ax_b.yaxis.grid(True, linestyle='--', alpha=0.3)

    # ---- (c) Magnitude of shift (violin plot): non-video vs video ----
    # 8 groups: 4 categories x 2 task groups (non-video, video)
    nonvideo_tasks = {'meditation', 'music', 'restingstate'}

    # Helper to filter by task group
    def filter_by_task(cat_set, is_video):
        if is_video:
            return [k for k in cat_set if k[1] == 'video']
        else:
            return [k for k in cat_set if k[1] in nonvideo_tasks]

    # Build groups: (exp_vals, opp_vals) for each of 8 combinations
    # Order: S[both] non-video, S[both] video, SW[both] non-video, SW[both] video,
    #        S only non-video, S only video, SW only non-video, SW only video
    groups_c = []
    cat_pairs = [
        ('yes_both', 'no_both', 0),   # S from both
        ('yes_both', 'no_both', 1),   # SW from both
        ('yes_s', 'no_s', 0),         # S only
        ('yes_sw', 'no_sw', 1),       # SW only
    ]

    for exp_cat, opp_cat, idx in cat_pairs:
        for is_video in [False, True]:
            exp_keys = filter_by_task(cat_sets[exp_cat], is_video)
            opp_keys = filter_by_task(cat_sets[opp_cat], is_video)
            exp_vals = [abs(deltas_abs[k][idx]) for k in exp_keys if k in deltas_abs]
            opp_vals = [abs(deltas_abs[k][idx]) for k in opp_keys if k in deltas_abs]
            groups_c.append((exp_vals, opp_vals))

    mag_labels_c = ['S [both]\nnon-video', 'S [both]\nvideo',
                    'SW [both]\nnon-video', 'SW [both]\nvideo',
                    'S only\nnon-video', 'S only\nvideo',
                    'SW only\nnon-video', 'SW only\nvideo']

    # Mann-Whitney U test for comparing distributions
    mag_sigs_c = []
    for exp_v, opp_v in groups_c:
        if len(exp_v) >= 1 and len(opp_v) >= 1:
            _, p = mannwhitneyu(exp_v, opp_v, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        else:
            sig = ''
        mag_sigs_c.append(sig)

    # Prepare data for violin plot
    violin_data = []
    violin_positions = []
    w_c = 0.35

    for i, (exp_v, opp_v) in enumerate(groups_c):
        if exp_v:
            violin_data.append(exp_v)
            violin_positions.append(i - w_c / 2)
        if opp_v:
            violin_data.append(opp_v)
            violin_positions.append(i + w_c / 2)

    # Draw violin plot
    if violin_data:
        parts = ax_c.violinplot(violin_data, positions=violin_positions, widths=0.3,
                                showmeans=True, showmedians=False, showextrema=False)

        # Color the violins
        color_idx = 0
        for i, (exp_v, opp_v) in enumerate(groups_c):
            if exp_v:
                parts['bodies'][color_idx].set_facecolor('#2ca02c')
                parts['bodies'][color_idx].set_alpha(0.7)
                color_idx += 1
            if opp_v:
                parts['bodies'][color_idx].set_facecolor('#d62728')
                parts['bodies'][color_idx].set_alpha(0.7)
                color_idx += 1

        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)

    # Significance brackets
    for i in range(8):
        if mag_sigs_c[i]:
            exp_v, opp_v = groups_c[i]
            max_val = max(max(exp_v) if exp_v else 0, max(opp_v) if opp_v else 0)
            top = max_val + 3
            ax_c.plot([i - w_c / 2, i - w_c / 2, i + w_c / 2, i + w_c / 2],
                      [top, top + 2, top + 2, top], 'k-', lw=1)
            ax_c.text(i, top + 3, mag_sigs_c[i], ha='center', va='bottom',
                      fontsize=9, fontweight='bold')

    ax_c.set_xticks(range(8))
    ax_c.set_xticklabels(mag_labels_c, fontsize=8)
    ax_c.set_ylabel('|Δ| per participant (burst count)', fontsize=LABEL_SIZE)
    ax_c.set_title('$\\mathbf{(c)}$ Magnitude of Shift: Non-Video vs Video', fontsize=TITLE_SIZE)

    # Custom legend for violin
    ax_c.legend([Patch(facecolor='#2ca02c', alpha=0.7),
                 Patch(facecolor='#d62728', alpha=0.7)],
                ['Expected', 'Opposite'], fontsize=10)
    ax_c.set_axisbelow(True)
    ax_c.yaxis.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Per-Participant Polarity Shift (ctrl → psi)',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_FILE}")
    plt.close('all')


if __name__ == '__main__':
    run_analysis()
