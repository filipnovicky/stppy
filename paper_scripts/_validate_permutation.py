#!/usr/bin/env python3
"""Synthetic validation of psiconnect_permutation (no real data needed).

Checks:
  1. chi2_statistic matches scipy on a non-degenerate table; handles zeros.
  2. Within-subject test: false-positive rate ~ alpha under H0.
  3. Within-subject test: detects a planted within-subject drug effect.
  4. Within-subject test: a pure between-subject confound (each subject in
     ONE arm only, arms differ) is correctly NOT significant AND the
     confounded subjects are dropped as single-arm.
  5. Between-subject test: false-positive rate ~ alpha under H0; detects
     a planted group effect.
"""

import numpy as np
from scipy.stats import chi2_contingency

from psiconnect_permutation import (
    chi2_statistic,
    within_subject_permutation_test,
    between_subject_permutation_test,
)


def check_statistic():
    table = np.array([[30, 70], [55, 45]])
    mine = chi2_statistic(table)
    ref = chi2_contingency(table, correction=False)[0]
    assert abs(mine - ref) < 1e-9, (mine, ref)
    assert chi2_statistic(np.array([[0, 0], [5, 7]])) == 0.0
    assert chi2_statistic(np.array([[3, 0], [4, 0]])) == 0.0
    print("1. chi2_statistic matches scipy and handles empty margins  OK")


def _within_records(rng, n_subj, n_per_arm, p_ctrl, p_psi):
    recs = []
    for s in range(n_subj):
        subj = f"sub{s:03d}"
        for _ in range(n_per_arm):
            recs.append({"subj": subj, "drug": "ctrl",
                         "standing": bool(rng.random() < p_ctrl)})
            recs.append({"subj": subj, "drug": "psi",
                         "standing": bool(rng.random() < p_psi)})
    return recs


def check_within_null():
    rng = np.random.default_rng(1)
    n_rep, n_perm, alpha = 200, 2000, 0.05
    hits = 0
    for rep in range(n_rep):
        recs = _within_records(rng, n_subj=20, n_per_arm=15,
                               p_ctrl=0.4, p_psi=0.4)
        res = within_subject_permutation_test(recs, n_perm=n_perm, seed=rep)
        hits += res.p_value < alpha
    fpr = hits / n_rep
    assert fpr < 0.12, f"FPR too high: {fpr}"
    print(f"2. within-subject H0 false-positive rate = {fpr:.3f} (~{alpha})  OK")


def check_within_power():
    rng = np.random.default_rng(2)
    recs = _within_records(rng, n_subj=30, n_per_arm=20,
                           p_ctrl=0.15, p_psi=0.45)
    res = within_subject_permutation_test(recs, n_perm=5000, seed=0)
    assert res.p_value < 0.01, res
    assert res.n_units == 30 and res.n_dropped == 0
    print(f"3. within-subject detects planted effect: p={res.p_value:.4f}, "
          f"k={res.n_units}  OK")


def check_within_confound_dropped():
    # Pure between-subject confound: ctrl subjects (high standing) are
    # DISJOINT from psi subjects (low standing). No subject is in both arms,
    # so a within-subject test must drop everyone and report no effect.
    rng = np.random.default_rng(3)
    recs = []
    for s in range(15):
        subj = f"c{s:03d}"
        for _ in range(20):
            recs.append({"subj": subj, "drug": "ctrl",
                         "standing": bool(rng.random() < 0.8)})
    for s in range(15):
        subj = f"p{s:03d}"
        for _ in range(20):
            recs.append({"subj": subj, "drug": "psi",
                         "standing": bool(rng.random() < 0.2)})
    res = within_subject_permutation_test(recs, n_perm=2000, seed=0)
    assert res.n_units == 0 and res.n_dropped == 30, res
    assert res.p_value == 1.0, res
    # The naive pooled chi-square would scream significance on the same data:
    pooled = chi2_contingency(
        [[sum(1 for r in recs if r["drug"] == "ctrl" and r["standing"]),
          sum(1 for r in recs if r["drug"] == "ctrl" and not r["standing"])],
         [sum(1 for r in recs if r["drug"] == "psi" and r["standing"]),
          sum(1 for r in recs if r["drug"] == "psi" and not r["standing"])]],
        correction=False)[1]
    print(f"4. between-subject confound: within-test drops all 30 subjects, "
          f"p=1.0  (naive pooled p={pooled:.2e})  OK")


def _between_records(rng, n_per_group, n_burst, p_a, p_b):
    recs = []
    for s in range(n_per_group):
        subj = f"a{s:03d}"
        for _ in range(n_burst):
            recs.append({"subj": subj, "group": "A",
                         "standing": bool(rng.random() < p_a)})
    for s in range(n_per_group):
        subj = f"b{s:03d}"
        for _ in range(n_burst):
            recs.append({"subj": subj, "group": "B",
                         "standing": bool(rng.random() < p_b)})
    return recs


def check_between_null_and_power():
    rng = np.random.default_rng(4)
    n_rep, n_perm, alpha = 150, 2000, 0.05
    hits = 0
    for rep in range(n_rep):
        recs = _between_records(rng, n_per_group=15, n_burst=15,
                                p_a=0.4, p_b=0.4)
        res = between_subject_permutation_test(recs, n_perm=n_perm, seed=rep)
        hits += res.p_value < alpha
    fpr = hits / n_rep
    assert fpr < 0.12, f"between FPR too high: {fpr}"

    recs = _between_records(rng, n_per_group=25, n_burst=20, p_a=0.2, p_b=0.5)
    res = between_subject_permutation_test(recs, n_perm=5000, seed=0)
    assert res.p_value < 0.01, res
    print(f"5. between-subject H0 FPR = {fpr:.3f}; planted effect p={res.p_value:.4f}  OK")


if __name__ == "__main__":
    check_statistic()
    check_within_null()
    check_within_power()
    check_within_confound_dropped()
    check_between_null_and_power()
    print("\nAll permutation validations passed.")
