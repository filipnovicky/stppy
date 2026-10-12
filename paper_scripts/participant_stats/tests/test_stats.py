"""Tests for participant_stats.stats."""

import numpy as np
import pytest

from participant_stats.stats import contingency_chi2, paired_summary, rank_biserial, signflip_p


def test_signflip_p_detects_consistent_shift_and_null():
    rng = np.random.default_rng(0)
    assert signflip_p(np.full(20, 0.3), rng, n_perm=2000) < 0.001
    assert signflip_p(np.array([1.0, -1.0] * 10), rng, n_perm=2000) == pytest.approx(1.0)


def test_signflip_p_rejects_empty():
    with pytest.raises(ValueError):
        signflip_p(np.array([]), np.random.default_rng(0))


def test_rank_biserial_extremes():
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert rank_biserial(np.array([-1.0, -2.0])) == pytest.approx(-1.0)
    assert rank_biserial(np.zeros(4)) == 0.0


def test_paired_summary_counts_direction():
    ctrl = np.array([0.1, 0.2, 0.3, 0.4])
    psi = np.array([0.3, 0.4, 0.2, 0.6])
    r = paired_summary("x", ctrl, psi, np.random.default_rng(0))
    assert r["n_participants"] == 4
    assert (r["n_increase"], r["n_decrease"]) == (3, 1)
    assert r["mean_diff"] == pytest.approx(0.125)
    assert r["diff_ci_lo"] <= r["mean_diff"] <= r["diff_ci_hi"]


def test_contingency_chi2_matches_scipy():
    from scipy.stats import chi2_contingency
    rows = np.array([0] * 30 + [1] * 20 + [0] * 10 + [1] * 40)
    cols = np.array(["a"] * 50 + ["b"] * 50)
    tab = [[30, 10], [20, 40]]
    assert contingency_chi2(rows, cols) == pytest.approx(chi2_contingency(tab, correction=False)[0])
    assert contingency_chi2(np.array([0, 0, 1, 1]), np.array(["a", "b", "a", "b"])) == 0.0
