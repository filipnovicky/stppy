#!/usr/bin/env python3
"""Table 1 composition test: does cluster membership depend on drug, task or training?

One test per factor on the 12 x levels contingency table of the 577 clustered
events (each counted once, see ``data.load_clustered_events``). Statistic =
Pearson chi-square; null by permutation that respects the design:

  drug     - per-participant swap of the ctrl/psi labels (within-subject)
  task     - shuffle task labels across a participant-session's events
  training - shuffle trained/untrained across participants (between-subject)

Benjamini-Hochberg across the 3 factors. Also writes Table 1 itself
(table1.md) and the per-cluster composition behind it (composition.csv).

    cd paper_scripts && python -m participant_stats.table1_composition

Outputs -> RESULTS_DIR/table1/ (paper_config.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .data import N_CLUSTERS, load_clustered_events
from .stats import N_PERM, contingency_chi2
from psiconnect_utils import benjamini_hochberg  # noqa: E402

SEED = 0


def perm_drug(ev: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    flip = {s: rng.random() < 0.5 for s in ev["subj"].unique()}
    swap = {"ctrl": "psi", "psi": "ctrl"}
    return np.array([swap[d] if flip[s] else d for s, d in zip(ev["subj"], ev["drug"])])


def perm_task(ev: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    out = ev["task"].to_numpy().copy()
    for idx in ev.groupby(["subj", "drug"]).indices.values():
        out[idx] = rng.permutation(out[idx])
    return out


def perm_training(ev: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    lab = ev.groupby("subj")["trained"].first()
    shuffled = dict(zip(lab.index, rng.permutation(lab.to_numpy())))
    return ev["subj"].map(shuffled).to_numpy()


FACTORS = (("drug", "drug", perm_drug), ("task", "task", perm_task),
           ("training", "trained", perm_training))


def composition_tests(ev: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cluster = ev["cluster"].to_numpy()
    rows = []
    for factor, col, permute in FACTORS:
        obs = contingency_chi2(cluster, ev[col].to_numpy())
        null = np.array([contingency_chi2(cluster, permute(ev, rng)) for _ in range(n_perm)])
        rows.append({"factor": factor, "chi2": obs,
                     "df": (ev["cluster"].nunique() - 1) * (ev[col].nunique() - 1),
                     "p_perm": (np.count_nonzero(null >= obs) + 1) / (n_perm + 1),
                     "n_events": len(ev), "n_participants": ev["subj"].nunique()})
    res = pd.DataFrame(rows)
    return res.assign(p_fdr=benjamini_hochberg(res["p_perm"].to_numpy()))


def composition(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster rows of Table 1 (count, participants, % by condition)."""
    task_pct = pd.crosstab(ev["cluster"], ev["task"], normalize="index") * 100
    return (ev.groupby("cluster")
            .agg(count=("event", "size"), subj=("subj", "nunique"),
                 ctrl_pct=("drug", lambda d: 100 * (d == "ctrl").mean()),
                 trained_pct=("trained", lambda t: 100 * t.mean()))
            .join(task_pct.add_suffix("_pct")))


def tightness() -> pd.Series:
    """Mean within-cluster distance (upper triangle) of clusters 0-11, as in
    stppy.clustering.simple_hierarchical_clustering."""
    labels = np.asarray(_load_pickle("cluster_assignments.pkl"))
    sim = np.asarray(_load_pickle("similarity_matrix.pkl"))
    out = {}
    for k in range(N_CLUSTERS):
        idx = np.flatnonzero(labels == k)
        sub = sim[np.ix_(idx, idx)]
        out[k] = float(sub[np.triu_indices(idx.size, 1)].mean())
    return pd.Series(out, name="tightness")


def _load_pickle(name: str):
    import pickle
    with open(paths.PAPER_DIR / name, "rb") as f:
        return pickle.load(f)


def table1(comp: pd.DataFrame, ev: pd.DataFrame) -> str:
    """Table 1 as in the paper (values in %), with the totals over the 577 events."""
    t = comp.join(tightness())
    total = composition(ev.assign(cluster=-1)).iloc[0]
    rows = ["| Cluster | Count | Tight. | Subj. | Control | Psilo. | Med. | Music | Rest | Video | Trained | Untr. |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def fmt(name, r, tight, subj):
        return (f"| {name} | {int(r['count'])} | {tight} | {subj} | {r['ctrl_pct']:.1f} | {100 - r['ctrl_pct']:.1f} | "
                f"{r['meditation_pct']:.1f} | {r['music_pct']:.1f} | {r['restingstate_pct']:.1f} | "
                f"{r['video_pct']:.1f} | {r['trained_pct']:.1f} | {100 - r['trained_pct']:.1f} |")
    for k, r in t.iterrows():
        rows.append(fmt(k, r, f"{r['tightness']:.2f}", int(r["subj"])))
    rows.append(fmt("Total", total, f"{t['tightness'].mean():.2f} (mean)", f"{ev['subj'].nunique()}"))
    return "\n".join(rows)


def main() -> None:
    ev = load_clustered_events()
    if ev["cluster"].nunique() != N_CLUSTERS:
        raise ValueError(f"expected {N_CLUSTERS} clusters, got {ev['cluster'].nunique()}")
    paths.OUT_TABLE1.mkdir(parents=True, exist_ok=True)
    comp, res = composition(ev), composition_tests(ev)
    comp.to_csv(paths.OUT_TABLE1 / "composition.csv")
    res.to_csv(paths.OUT_TABLE1 / "composition_tests.csv", index=False)
    md = table1(comp, ev)
    (paths.OUT_TABLE1 / "table1.md").write_text(md + "\n")
    print(md, "\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
