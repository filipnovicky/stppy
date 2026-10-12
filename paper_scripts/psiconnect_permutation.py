#!/usr/bin/env python3
"""Subject-level permutation tests for the psiconnect polarity analysis.

The pooled chi-square in ``main_analysis.py`` treats every burst as an
independent observation. Bursts are nested within subjects, so that
violates independence (pseudoreplication) and makes the p-values
anti-conservative. These permutation tests fix that by permuting labels
at the *subject* level, which respects the design:

- Drug (ctrl vs psi) is a **within-subject crossover** factor. The null
  is tested by independently sign-flipping each subject's ctrl/psi label
  (``within_subject_permutation_test``). Only subjects present in BOTH
  arms of the slice carry within-subject drug information; single-arm
  subjects are dropped and reported (never silently).

- Trained vs untrained is a **between-subject** factor. The null is
  tested by shuffling the group label across subjects, each subject
  keeping its own burst counts (``between_subject_permutation_test``).

The test statistic is the Pearson chi-square *value* of the pooled 2x2
table. Its validity here comes from the permutation null, not from the
asymptotic chi-square distribution, so it stays valid for sparse tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def chi2_statistic(table: np.ndarray) -> float:
    """Pearson chi-square statistic (no continuity correction) for a 2x2 table.

    Returns 0.0 when any row or column margin is zero (an empty margin
    carries no association information).

    Parameters
    ----------
    table : np.ndarray
        2x2 contingency table of non-negative counts.

    Returns
    -------
    float
        The chi-square statistic value.
    """
    table = np.asarray(table, dtype=float)
    n = table.sum()
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    if n == 0 or (row == 0).any() or (col == 0).any():
        return 0.0
    expected = row @ col / n
    return float(((table - expected) ** 2 / expected).sum())


@dataclass(frozen=True)
class PermutationResult:
    """Outcome of a subject-level permutation test.

    Attributes
    ----------
    observed : float
        Observed chi-square statistic on the real labelling.
    p_value : float
        Two-sided permutation p-value, (#{stat >= observed} + 1)/(n_perm + 1).
    n_perm : int
        Number of Monte-Carlo permutations drawn.
    n_units : int
        Number of permutation units used (paired subjects, or subjects).
    n_dropped : int
        Subjects excluded from the test (single-arm, for the within-subject case).
    n_bursts : int
        Bursts contributing to the test (after any drop).
    """

    observed: float
    p_value: float
    n_perm: int
    n_units: int
    n_dropped: int
    n_bursts: int


def _two_sided_p(observed: float, null: np.ndarray, n_perm: int) -> float:
    """Add-one two-sided p-value for a non-negative discrepancy statistic."""
    n_ge = int(np.count_nonzero(null >= observed))
    return (n_ge + 1) / (n_perm + 1)


def within_subject_permutation_test(
    records: Sequence[dict],
    n_perm: int = 10000,
    seed: int = 0,
) -> PermutationResult:
    """Within-subject (crossover) permutation test of the drug effect.

    Each record is ``{'subj': str, 'drug': 'ctrl'|'psi', 'standing': bool}``.
    Subjects must appear in both drug arms to contribute; single-arm
    subjects are dropped (and counted in ``n_dropped``).

    Under H0 (drug does not affect polarity) the two sessions of a subject
    are exchangeable, so each permutation independently swaps a subject's
    ctrl/psi labels with probability 0.5.

    Parameters
    ----------
    records : sequence of dict
        Per-burst records for the slice being tested.
    n_perm : int
        Number of permutations.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    PermutationResult
    """
    # Per-subject counts: [std_ctrl, sw_ctrl, std_psi, sw_psi]
    counts: dict[str, np.ndarray] = {}
    for r in records:
        c = counts.setdefault(r["subj"], np.zeros(4, dtype=float))
        drug_off = 0 if r["drug"] == "ctrl" else 2
        std_off = 0 if r["standing"] else 1
        c[drug_off + std_off] += 1

    paired, dropped, n_bursts = [], 0, 0
    for subj, c in counts.items():
        has_ctrl = (c[0] + c[1]) > 0
        has_psi = (c[2] + c[3]) > 0
        if has_ctrl and has_psi:
            paired.append(c)
            n_bursts += int(c.sum())
        else:
            dropped += 1

    if not paired:
        return PermutationResult(0.0, 1.0, n_perm, 0, dropped, 0)

    # subj_counts[i] = [std_ctrl, sw_ctrl, std_psi, sw_psi]
    subj_counts = np.vstack(paired)  # (k, 4)
    k = subj_counts.shape[0]

    def table_from_assignment(swap: np.ndarray) -> np.ndarray:
        # swap[i] True -> subject i's ctrl/psi labels are exchanged
        ctrl_std = np.where(swap, subj_counts[:, 2], subj_counts[:, 0]).sum()
        ctrl_sw = np.where(swap, subj_counts[:, 3], subj_counts[:, 1]).sum()
        psi_std = np.where(swap, subj_counts[:, 0], subj_counts[:, 2]).sum()
        psi_sw = np.where(swap, subj_counts[:, 1], subj_counts[:, 3]).sum()
        return np.array([[ctrl_std, ctrl_sw], [psi_std, psi_sw]])

    observed = chi2_statistic(table_from_assignment(np.zeros(k, dtype=bool)))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        swap = rng.random(k) < 0.5
        null[i] = chi2_statistic(table_from_assignment(swap))

    return PermutationResult(
        observed=observed,
        p_value=_two_sided_p(observed, null, n_perm),
        n_perm=n_perm,
        n_units=k,
        n_dropped=dropped,
        n_bursts=n_bursts,
    )


def between_subject_permutation_test(
    records: Sequence[dict],
    n_perm: int = 10000,
    seed: int = 0,
) -> PermutationResult:
    """Between-subject permutation test (e.g. trained vs untrained).

    Each record is ``{'subj': str, 'group': str, 'standing': bool}``.
    Each subject keeps its own (standing, switching) counts; the group
    label is shuffled across subjects, preserving the group sizes.

    Parameters
    ----------
    records : sequence of dict
        Per-burst records, each tagged with a between-subject ``group``.
    n_perm : int
        Number of permutations.
    seed : int
        RNG seed.

    Returns
    -------
    PermutationResult
    """
    # Per-subject: [n_standing, n_switching], plus its group.
    counts: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}
    for r in records:
        c = counts.setdefault(r["subj"], np.zeros(2, dtype=float))
        c[0 if r["standing"] else 1] += 1
        groups[r["subj"]] = r["group"]

    subjects = list(counts.keys())
    if len(set(groups[s] for s in subjects)) < 2:
        # Only one group present -> no contrast.
        return PermutationResult(0.0, 1.0, n_perm, len(subjects), 0,
                                 int(sum(c.sum() for c in counts.values())))

    subj_counts = np.vstack([counts[s] for s in subjects])  # (m, 2)
    group_names = sorted(set(groups.values()))
    g0 = np.array([groups[s] == group_names[0] for s in subjects])  # bool mask

    def table_from_mask(mask: np.ndarray) -> np.ndarray:
        a = subj_counts[mask].sum(axis=0)
        b = subj_counts[~mask].sum(axis=0)
        return np.array([a, b])

    observed = chi2_statistic(table_from_mask(g0))

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(g0)
        null[i] = chi2_statistic(table_from_mask(perm))

    return PermutationResult(
        observed=observed,
        p_value=_two_sided_p(observed, null, n_perm),
        n_perm=n_perm,
        n_units=len(subjects),
        n_dropped=0,
        n_bursts=int(subj_counts.sum()),
    )
