"""Figure 3e, drawn by main_analysis.py: participant-level stable proportion, control vs psilocybin.

Each participant with sequences in both sessions contributes one paired
proportion P(stable) = stable / (stable + oscillating). Inference is the
unweighted within-participant change, tested by a sign-flip permutation, so
every participant weighs the same regardless of how many sequences they
contribute (same test as participant_level.py).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .stats import signflip_p

BINARY = ('stable', 'oscillating')
N_PERM = 100000  # Monte-Carlo SE on p ~0.0002 at p ~0.003 (matches paired_tests.csv)
DOT_AREA_PER_SEQ = 4.0
DOT_AREA_MIN = 12.0


def paired_stable_proportions(sequences) -> dict:
    """Per-participant P(stable) for participants present in both sessions."""
    counts = defaultdict(lambda: {'ctrl': [0, 0], 'psi': [0, 0]})
    for s in sequences:
        if s['polarity'] not in BINARY:
            continue
        c = counts[s['_subj']][s['_drug']]
        c[0] += s['polarity'] == 'stable'
        c[1] += 1
    subjects = sorted(k for k, v in counts.items() if v['ctrl'][1] and v['psi'][1])
    n_ctrl = np.array([counts[k]['ctrl'][1] for k in subjects])
    n_psi = np.array([counts[k]['psi'][1] for k in subjects])
    return {
        'subjects': subjects,
        'ctrl': np.array([counts[k]['ctrl'][0] for k in subjects]) / np.maximum(n_ctrl, 1),
        'psi': np.array([counts[k]['psi'][0] for k in subjects]) / np.maximum(n_psi, 1),
        'n_ctrl': n_ctrl,
        'n_psi': n_psi,
    }


def plot_participant_panel(ax, sequences, col_ctrl, col_psi, sig_stars, seed=0):
    """Slopegraph of per-participant stable share (%), median bars, sign-flip p."""
    p = paired_stable_proportions(sequences)
    if not p['subjects']:
        raise ValueError('no participant has sequences in both sessions')
    ctrl, psi = 100 * p['ctrl'], 100 * p['psi']
    pval = signflip_p(psi - ctrl, np.random.default_rng(seed), n_perm=N_PERM)

    x = np.array([0.0, 1.0])
    for a, b in zip(ctrl, psi):
        ax.plot(x, [a, b], color='#b5b4ad', lw=1.0, zorder=1)
    for xi, vals, n, col in ((0.0, ctrl, p['n_ctrl'], col_ctrl), (1.0, psi, p['n_psi'], col_psi)):
        sizes = np.maximum(DOT_AREA_MIN, DOT_AREA_PER_SEQ * n)
        ax.scatter(np.full(vals.size, xi), vals, s=sizes, color=col, alpha=0.85,
                   edgecolor='white', lw=0.6, zorder=2)
        med = np.median(vals)
        ax.plot([xi - 0.2, xi + 0.2], [med, med], color='black', lw=3, zorder=3)

    top = 108
    ax.plot([0, 0, 1, 1], [top - 2, top, top, top - 2], color='black', lw=1.2)
    ax.text(0.5, top + 1, sig_stars(pval), ha='center', va='bottom',
            fontsize=16, fontweight='bold')
    ax.text(0.5, 0.985, f"n = {len(p['subjects'])} paired, "
            f"mean {np.mean(psi - ctrl):+.1f} pp\nsign-flip p = {pval:.3f}",
            transform=ax.transAxes, ha='center', va='top', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Control', 'Psilocybin'], fontsize=16)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-4, 135)
    ax.set_ylabel('Stable sequences per participant (%)', fontsize=18)
    ax.set_title('(e) Per Participant', fontsize=21, fontweight='bold')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    return {'n': len(p['subjects']), 'p': pval}
