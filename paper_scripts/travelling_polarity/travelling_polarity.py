#!/usr/bin/env python3
"""Travelling/standing x stable/oscillating cross-tabulation (response to reviewers, R1.3).

Two properties of a cluster, measured at different timescales:

- travelling / standing: spatial evolution WITHIN one template (40-44 ms). This is the
  paper's classification (TRAVELLING below), which the stppy020 pole-arrow measure
  reproduces (stronger pole >= 0.75 of the longest arrow).
- stable / oscillating: polarity relation BETWEEN successive back-fitted occurrences of the
  template in the continuous EEG (a sequence).

Sequences are the paper's back-fitted sequences after overlap removal, exactly as in Table 2
(main_analysis.remove_overlapping_bursts). The script prints

1. per-cluster stable/oscillating counts per session, checked against the paper's Table 2;
2. the cluster-level 2x2 (each cluster placed by its predominant polarity in a session);
3. the sequence-level 2x2 (descriptive only);
4. a Fisher exact test on the cluster-level 2x2.

The test is done on clusters, not sequences, because travelling/standing is a property of a
cluster: every sequence of a cluster carries the same label, so sequences are not independent
observations of the association (thousands of sequences, only 11 labelled clusters).
Cluster 3 is excluded: it produced a single sequence.

Run: python travelling_polarity.py        (reads PAPER_DIR, writes OUT_DIR, a few seconds)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import paper_config  # noqa: E402  (all paths: paper_config.py)

PAPER_DIR = paper_config.RUN_DIR
OUT_DIR = paper_config.RESULTS_DIR / "travelling_polarity"

TRAVELLING = frozenset({0, 2, 3, 4, 5})  # paper classification; all other clusters standing
EXCLUDED = frozenset({3})                # a single sequence, no polarity profile
SESSIONS = ("ctrl", "psi")
SESSION_NAMES = {"ctrl": "baseline", "psi": "psilocybin"}
POLARITIES = ("oscillating", "stable")

# Paper Table 2: cluster -> (ctrl stable, ctrl oscillating, psi stable, psi oscillating)
TABLE2 = {
    0: (0, 128, 0, 38), 1: (1, 38, 0, 6), 2: (1, 630, 0, 171), 4: (0, 64, 0, 29),
    5: (0, 642, 0, 195), 6: (18, 50, 4, 12), 7: (74, 1, 49, 1), 8: (103, 225, 192, 78),
    9: (80, 4, 33, 0), 10: (0, 14, 0, 3), 11: (24, 0, 6, 0),
}

Counts = dict[tuple[int, str], dict[str, int]]


def count_sequences(sequences_by_cluster: dict, session_of) -> Counts:
    """(cluster, session) -> {'stable': n, 'oscillating': n}; other polarities are ignored."""
    counts: Counts = {}
    for cid, seqs in sequences_by_cluster.items():
        for s in SESSIONS:
            counts[(cid, s)] = {p: 0 for p in POLARITIES}
        for seq in seqs:
            if seq["polarity"] in POLARITIES:
                counts[(cid, session_of(seq["dataset"]))][seq["polarity"]] += 1
    return counts


def predominant(c: dict[str, int]) -> str | None:
    """Majority polarity of one cluster in one session; None for a tie or no sequences."""
    if c["oscillating"] == c["stable"]:
        return None
    return "oscillating" if c["oscillating"] > c["stable"] else "stable"


def spatial(cid: int) -> str:
    return "travelling" if cid in TRAVELLING else "standing"


def cluster_table(counts: Counts, session: str) -> dict[tuple[str, str], list[int]]:
    """(spatial, polarity) -> cluster ids, each cluster by its predominant polarity."""
    table = {(sp, p): [] for sp in ("travelling", "standing") for p in POLARITIES}
    for cid in sorted({c for c, _ in counts} - EXCLUDED):
        p = predominant(counts[(cid, session)])
        if p is not None:
            table[(spatial(cid), p)].append(cid)
    return table


def sequence_table(counts: Counts, session: str) -> np.ndarray:
    """2x2 sequence counts, rows travelling/standing, columns oscillating/stable."""
    out = np.zeros((2, 2), dtype=int)
    for (cid, s), c in counts.items():
        if s != session or cid in EXCLUDED:
            continue
        row = 0 if cid in TRAVELLING else 1
        out[row] += [c["oscillating"], c["stable"]]
    return out


def cluster_fisher(table: dict[tuple[str, str], list[int]]) -> tuple[np.ndarray, float, float]:
    """Fisher exact test on cluster counts: two-sided p and one-sided p (travelling -> oscillating)."""
    m = np.array([[len(table[(sp, p)]) for p in POLARITIES] for sp in ("travelling", "standing")])
    _, p_two = fisher_exact(m, alternative="two-sided")
    _, p_one = fisher_exact(m, alternative="greater")
    return m, float(p_two), float(p_one)


def check_table2(counts: Counts) -> list[str]:
    """Clusters whose counts differ from the paper's Table 2 (empty list = exact match)."""
    bad = []
    for cid, (cs, co, ps, po) in TABLE2.items():
        got = (counts[(cid, "ctrl")]["stable"], counts[(cid, "ctrl")]["oscillating"],
               counts[(cid, "psi")]["stable"], counts[(cid, "psi")]["oscillating"])
        if got != (cs, co, ps, po):
            bad.append(f"cluster {cid}: got {got}, Table 2 {(cs, co, ps, po)}")
    return bad


def report(counts: Counts) -> str:
    lines = ["Travelling/standing x stable/oscillating (R1.3)",
             f"travelling clusters: {sorted(TRAVELLING)}; excluded: {sorted(EXCLUDED)}", ""]
    bad = check_table2(counts)
    lines.append("Counts match paper Table 2: " + ("yes" if not bad else "NO"))
    lines += [f"  {b}" for b in bad]
    lines.append("")
    lines.append("1. Per cluster (stable / oscillating)")
    lines.append(f"{'cluster':>7} {'spatial':>10} {'baseline':>12} {'psilocybin':>12}")
    for cid in sorted({c for c, _ in counts}):
        cells = [f"{counts[(cid, s)]['stable']:>4} / {counts[(cid, s)]['oscillating']:<5}"
                 for s in SESSIONS]
        lines.append(f"{cid:>7} {spatial(cid):>10} {cells[0]:>12} {cells[1]:>12}")
    for s in SESSIONS:
        name = SESSION_NAMES[s]
        ct = cluster_table(counts, s)
        m, p_two, p_one = cluster_fisher(ct)
        st = sequence_table(counts, s)
        lines += ["", f"2. Cluster-level 2x2, {name} (cluster ids by predominant polarity)",
                  f"{'':>11} {'oscillating':>20} {'stable':>20}"]
        for sp in ("travelling", "standing"):
            cells = [", ".join(map(str, ct[(sp, p)])) or "-" for p in POLARITIES]
            lines.append(f"{sp:>11} {cells[0]:>20} {cells[1]:>20}")
        lines += [f"   cluster counts {m.tolist()}; Fisher exact p = {p_two:.3f} (two-sided), "
                  f"{p_one:.3f} (one-sided, travelling -> oscillating)",
                  "", f"3. Sequence-level 2x2, {name} (descriptive; not independent)",
                  f"{'':>11} {'oscillating':>12} {'stable':>8}"]
        for i, sp in enumerate(("travelling", "standing")):
            lines.append(f"{sp:>11} {st[i, 0]:>12} {st[i, 1]:>8}")
        osc_tr = st[0, 0] / st[:, 0].sum()
        st_sd = st[1, 1] / st[:, 1].sum()
        lines.append(f"   oscillating sequences from travelling clusters: {st[0, 0]}/"
                     f"{st[:, 0].sum()} ({100 * osc_tr:.1f}%); stable sequences from standing "
                     f"clusters: {st[1, 1]}/{st[:, 1].sum()} ({100 * st_sd:.1f}%)")
    return "\n".join(lines)


def load_counts() -> Counts:
    import stppy.io
    from main_analysis import POLARITY_RELABEL, remove_overlapping_bursts
    from psiconnect_utils import parse_condition

    raw = stppy.io.load_pickle(str(PAPER_DIR / "sequences_by_cluster.pkl"))
    averages = stppy.io.load_pickle(str(PAPER_DIR / "cluster_averages.pkl"))
    for seqs in raw.values():
        for s in seqs:
            s["polarity"] = POLARITY_RELABEL.get(s["polarity"], s["polarity"])
    clean = remove_overlapping_bursts(raw, averages)
    return count_sequences(clean, lambda name: parse_condition(name)[0])


def main() -> int:
    text = report(load_counts())
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "travelling_polarity_2x2.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
