"""Loaders for the paper's sequences, clustered events and recordings.

Everything is read from the paper run (``paths.PAPER_DIR``); nothing is
recomputed. Sequences are the back-fitted ones after the cross-cluster
overlap removal (>= 11 samples) of ``main_analysis.remove_overlapping_bursts`` (3045).
"""

from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np
import pandas as pd

from . import paths
from main_analysis import POLARITY_RELABEL, remove_overlapping_bursts  # noqa: E402
from psiconnect_utils import parse_condition  # noqa: E402

N_CLUSTERS = 12          # paper clusters 0-11; 12-23 are their polarity mirrors
BINARY = ("stable", "oscillating")


def _load(name: str):
    with open(paths.PAPER_DIR / name, "rb") as f:
        return pickle.load(f)


def is_trained(subj: pd.Series) -> pd.Series:
    """MBCT-trained participants carry a '2' as the 4th character (sub2xx)."""
    return subj.str[3].eq("2").astype(int)


def tag_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Add drug / subj / task / trained columns parsed from the dataset name."""
    parsed = df["dataset"].map(parse_condition)
    out = df.assign(drug=[p[0] for p in parsed], subj=[p[1] for p in parsed],
                    task=[p[2] for p in parsed])
    return out.assign(trained=is_trained(out["subj"]))


def load_sequences() -> pd.DataFrame:
    """Paper sequences after overlap removal, one row per sequence."""
    raw = _load("sequences_by_cluster.pkl")
    avgs = _load("cluster_averages.pkl")
    relabelled = {cid: [{**b, "polarity": POLARITY_RELABEL.get(b["polarity"], b["polarity"])}
                        for b in seqs] for cid, seqs in raw.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        clean = remove_overlapping_bursts(relabelled, avgs)
    rows = [{"dataset": b["dataset"], "cluster": cid, "polarity": b["polarity"]}
            for cid, seqs in clean.items() for b in seqs]
    return tag_conditions(pd.DataFrame(rows))


def load_clustered_events() -> pd.DataFrame:
    """The avalanches in clusters 0-11 (Table 1), one row per original event.

    The 10216 clustered entries are the 5108 avalanches followed by their
    polarity-reversed copies; an event and its copy fall in mirror clusters
    k and k+12, so clusters 0-11 hold each event exactly once.
    """
    labels = np.asarray(_load("cluster_assignments.pkl"))
    times = _load("peak_avalanche_times.pkl")
    files = _load("files.pkl")
    dataset = np.concatenate([[f] * len(t) for f, t in zip(files, times)])
    n = dataset.size
    if labels.size != 2 * n:
        raise ValueError(f"expected 2 x {n} cluster labels, got {labels.size}")
    idx = np.flatnonzero((labels >= 0) & (labels < N_CLUSTERS))
    ev = pd.DataFrame({"cluster": labels[idx], "dataset": dataset[idx % n], "event": idx % n})
    if ev["event"].duplicated().any():
        raise ValueError("an avalanche appears twice in clusters 0-11")
    return tag_conditions(ev)


def recording_table(seqs: pd.DataFrame) -> pd.DataFrame:
    """One row per recording with its sequence counts (zero-sequence recordings kept)."""
    if not paths.RECORDINGS_CSV.exists():
        raise FileNotFoundError(f"{paths.RECORDINGS_CSV} missing; run "
                                "python -m participant_stats.build_recordings first")
    rec = tag_conditions(pd.read_csv(paths.RECORDINGS_CSV))
    rec = rec.assign(duration_min=rec["duration_s"] / 60.0)
    counts = (seqs.assign(n=1).pivot_table(index="dataset", columns="polarity",
                                           values="n", aggfunc="sum", fill_value=0)
              .reindex(columns=[*BINARY, "mixed"], fill_value=0))
    rec = rec.merge(counts, left_on="dataset", right_index=True, how="left")
    for c in [*BINARY, "mixed"]:
        rec[c] = rec[c].fillna(0).astype(int)
    return rec.assign(total=rec["stable"] + rec["oscillating"] + rec["mixed"],
                      psi=(rec["drug"] == "psi").astype(int))
