#!/usr/bin/env python
"""Candidate repertoire for matching to the paper: cluster at CLUSTER_THRESHOLD with a small
minimum size (MIN_SIZE, default 2, instead of the notebook's 30), DTW consensus per cluster,
polarity dedup. Writes candidates_t<t>_min<m>.npz for match_paper.py (stppy020, separate process).

    RUN=ds4148_EC CLUSTER_THRESHOLD=3.5 MIN_SIZE=2 python candidates.py
"""
from __future__ import annotations

import contextlib
import io as _io
import os

import numpy as np

import cluster as cl
import config as cfg
from config import stppy

MIN_SIZE = int(os.environ.get("MIN_SIZE", 2))


def main() -> int:
    avalanches, _ = cfg.load_pool()
    _, K_norm = cfg.ground_metric_norm()
    sim = np.load(cfg.SIMILARITY)["similarity_matrix"]
    with contextlib.redirect_stdout(_io.StringIO()):
        labels, info = stppy.clustering.simple_hierarchical_clustering(
            sim, threshold=cfg.CLUSTER_THRESHOLD, method=cfg.CLUSTER_METHOD,
            min_cluster_size=MIN_SIZE)
    avgs, members = {}, {}
    for cid in range(len(info)):
        avgs[cid], members[cid], _ = cl.consensus(avalanches, K_norm, sim, labels, cid)
    with contextlib.redirect_stdout(_io.StringIO()):
        keep = stppy.clustering.remove_duplicate_polarities(avgs, threshold=cfg.DEDUP_THRESHOLD)
    keep = sorted(keep, key=lambda c: -len(members[c]))
    out = cfg.OUT / f"candidates_t{cfg.CLUSTER_THRESHOLD:g}_min{MIN_SIZE}.npz"
    np.savez_compressed(out, keep=np.array(keep), sizes=np.array([len(members[c]) for c in keep]),
                        assigned=int((labels >= 0).sum()), n_arrays=len(avalanches),
                        **{f"consensus_{i:03d}": avgs[c] for i, c in enumerate(keep)})
    print(f"{cfg.RUN} t={cfg.CLUSTER_THRESHOLD:g} min={MIN_SIZE}: {len(info)} clusters, "
          f"{len(keep)} after dedup, {int((labels >= 0).sum())}/{len(avalanches)} assigned -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
