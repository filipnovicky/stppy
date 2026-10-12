import numpy as np

from travelling_polarity import (
    TABLE2,
    check_table2,
    cluster_fisher,
    cluster_table,
    count_sequences,
    predominant,
    sequence_table,
)


def _counts_from_table2():
    seqs = {}
    for cid, (cs, co, ps, po) in TABLE2.items():
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


def test_table2_counts_roundtrip():
    assert check_table2(_counts_from_table2()) == []


def test_cluster_table_matches_paper_baseline():
    t = cluster_table(_counts_from_table2(), "ctrl")
    assert t[("travelling", "oscillating")] == [0, 2, 4, 5]
    assert t[("travelling", "stable")] == []
    assert t[("standing", "oscillating")] == [1, 6, 8, 10]
    assert t[("standing", "stable")] == [7, 9, 11]


def test_cluster_8_changes_cell_under_psilocybin():
    t = cluster_table(_counts_from_table2(), "psi")
    assert 8 in t[("standing", "stable")]
    assert 8 not in t[("standing", "oscillating")]


def test_sequence_table_excludes_cluster_3():
    st = sequence_table(_counts_from_table2(), "ctrl")
    assert np.array_equal(st, [[1464, 1], [332, 300]])


def test_fisher_on_cluster_counts():
    m, p_two, p_one = cluster_fisher(cluster_table(_counts_from_table2(), "ctrl"))
    assert m.tolist() == [[4, 0], [4, 3]]
    assert 0 < p_one <= p_two <= 1
