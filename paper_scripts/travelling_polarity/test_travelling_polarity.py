import numpy as np

from travelling_polarity import (
    cluster_fisher,
    cluster_table,
    count_sequences,
    predominant,
    sequence_table,
)

# cluster -> (ctrl stable, ctrl oscillating, psi stable, psi oscillating)
COUNTS = {
    0: (0, 134, 0, 40), 1: (1, 42, 0, 8), 2: (1, 663, 0, 176), 4: (0, 64, 0, 30),
    5: (0, 666, 0, 203), 6: (19, 54, 4, 12), 7: (75, 1, 51, 1), 8: (108, 248, 191, 80),
    9: (82, 4, 35, 0), 10: (0, 18, 0, 3), 11: (24, 0, 6, 0),
}


def _counts():
    seqs = {}
    for cid, (cs, co, ps, po) in COUNTS.items():
        seqs[cid] = ([{"dataset": "ctrl", "polarity": "stable"}] * cs
                     + [{"dataset": "ctrl", "polarity": "oscillating"}] * co
                     + [{"dataset": "psi", "polarity": "stable"}] * ps
                     + [{"dataset": "psi", "polarity": "oscillating"}] * po)
    seqs[3] = [{"dataset": "ctrl", "polarity": "oscillating"}]
    return count_sequences(seqs, lambda name: name)


def test_count_ignores_other_polarities():
    seqs = {0: [{"dataset": "ctrl", "polarity": "mixed"},
                {"dataset": "ctrl", "polarity": "stable"}]}
    c = count_sequences(seqs, lambda name: name)
    assert c[(0, "ctrl")] == {"oscillating": 0, "stable": 1}
    assert c[(0, "psi")] == {"oscillating": 0, "stable": 0}


def test_predominant_majority_and_tie():
    assert predominant({"oscillating": 3, "stable": 1}) == "oscillating"
    assert predominant({"oscillating": 1, "stable": 3}) == "stable"
    assert predominant({"oscillating": 2, "stable": 2}) is None


def test_cluster_table_matches_paper_baseline():
    t = cluster_table(_counts(), "ctrl")
    assert t[("travelling", "oscillating")] == [0, 2, 4, 5]
    assert t[("travelling", "stable")] == []
    assert t[("standing", "oscillating")] == [1, 6, 8, 10]
    assert t[("standing", "stable")] == [7, 9, 11]


def test_cluster_8_changes_cell_under_psilocybin():
    t = cluster_table(_counts(), "psi")
    assert 8 in t[("standing", "stable")]
    assert 8 not in t[("standing", "oscillating")]


def test_sequence_table_excludes_cluster_3():
    st = sequence_table(_counts(), "ctrl")
    assert np.array_equal(st, [[1527, 1], [367, 309]])


def test_fisher_on_cluster_counts():
    m, p_two, p_one = cluster_fisher(cluster_table(_counts(), "ctrl"))
    assert m.tolist() == [[4, 0], [4, 3]]
    assert 0 < p_one <= p_two <= 1
