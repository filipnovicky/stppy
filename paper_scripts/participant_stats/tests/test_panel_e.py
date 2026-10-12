"""Tests for participant_stats.panel_e (Figure 3e)."""

import pytest

from participant_stats.panel_e import paired_stable_proportions


def _seq(subj, drug, pol):
    return {'_subj': subj, '_drug': drug, 'polarity': pol}


def test_paired_proportions_keep_only_both_sessions():
    seqs = ([_seq('a', 'ctrl', 'stable')] + [_seq('a', 'ctrl', 'oscillating')] * 3
            + [_seq('a', 'psi', 'stable')] * 2
            + [_seq('b', 'ctrl', 'stable')])  # single-arm -> dropped
    p = paired_stable_proportions(seqs)
    assert p['subjects'] == ['a']
    assert p['ctrl'] == pytest.approx([0.25])
    assert p['psi'] == pytest.approx([1.0])
    assert p['n_ctrl'].tolist() == [4]
    assert p['n_psi'].tolist() == [2]


def test_paired_proportions_ignore_non_binary_polarity():
    seqs = [_seq('a', 'ctrl', 'stable'), _seq('a', 'ctrl', 'mixed'),
            _seq('a', 'psi', 'oscillating')]
    p = paired_stable_proportions(seqs)
    assert p['ctrl'] == pytest.approx([1.0])
    assert p['n_ctrl'].tolist() == [1]
