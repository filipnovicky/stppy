"""Tests for participant_counts and the Table 1 permutations (synthetic data)."""

import numpy as np
import pandas as pd

from participant_stats.participant_counts import count_table, slice_counts
from participant_stats.table1_composition import (composition_tests, perm_drug,
                                                  perm_task, perm_training)


def _seqs():
    return pd.DataFrame({
        "subj": ["a", "a", "a", "b", "c", "c"],
        "drug": ["ctrl", "psi", "psi", "ctrl", "ctrl", "psi"],
        "task": ["music", "music", "video", "music", "video", "video"],
        "cluster": [0, 0, 1, 1, 0, 1],
        "polarity": ["stable", "oscillating", "stable", "stable", "mixed", "stable"],
    })


def test_slice_counts_paired_participants():
    r = slice_counts(_seqs(), "all")
    assert (r["participants_ctrl"], r["participants_psi"]) == (3, 2)
    assert (r["participants_any"], r["participants_paired"]) == (3, 2)
    assert (r["sequences"], r["sequences_paired"]) == (6, 5)


def test_count_table_drops_non_binary_and_slices():
    t = count_table(_seqs()).set_index("slice")
    assert t.loc["overall", "sequences"] == 5          # the 'mixed' row is dropped
    assert t.loc["overall", "participants_paired"] == 1  # c loses its ctrl sequence
    assert t.loc["cluster:0", "participants_paired"] == 1
    assert t.loc["task:video", "participants_any"] == 2


def _events(n_subj=8, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subj):
        for drug in ("ctrl", "psi"):
            for k in range(per):
                rows.append({"subj": f"s{s}", "drug": drug, "task": ["music", "video"][k % 2],
                             "trained": s % 2, "cluster": int(rng.integers(0, 3))})
    return pd.DataFrame(rows)


def test_permutations_respect_the_design():
    ev, rng = _events(), np.random.default_rng(1)
    d = perm_drug(ev, rng)
    for s, g in ev.assign(new=d).groupby("subj"):   # whole participant swapped or kept
        assert (g["new"] == g["drug"]).all() or (g["new"] != g["drug"]).all()
    t = perm_task(ev, rng)
    for _, g in ev.assign(new=t).groupby(["subj", "drug"]):
        assert sorted(g["new"]) == sorted(g["task"])
    tr = perm_training(ev, rng)
    assert ev.assign(new=tr).groupby("subj")["new"].nunique().eq(1).all()
    assert tr.sum() == ev["trained"].sum()


def test_composition_tests_null_data_not_significant():
    res = composition_tests(_events(), n_perm=200)
    assert list(res["factor"]) == ["drug", "task", "training"]
    assert (res["p_fdr"] >= res["p_perm"] - 1e-12).all()
    assert (res["p_perm"] > 0.01).all()
