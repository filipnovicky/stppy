#!/usr/bin/env python
"""Stage 3: quickstart clustering + DTW consensi + consensus topomap figures.

simple_hierarchical_clustering(threshold=3.5, average, min_cluster_size=30), consensus per
cluster via build_consensus_alignment, then remove_duplicate_polarities(-0.999). The full path
matrix is not stored, so only each medoid's DTW row is recomputed (the only row
build_consensus_alignment reads); off-medoid distances are set to 1e9 so the medoid stays put.
"""
from __future__ import annotations

import contextlib
import csv
import io as _io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

import config as cfg                                                      # noqa: E402
from config import stppy                                                  # noqa: E402


def medoid_paths(avalanches, K_norm, member_idx, medoid):
    local = [medoid] + [int(g) for g in member_idx if int(g) != medoid]
    m = len(local)
    sim = np.full((m, m), 1e9)
    np.fill_diagonal(sim, 0.0)
    paths = np.full((m, m), None, dtype=object)
    paths[0, 0] = [[(t, t) for t in range(avalanches[local[0]].shape[1])]]
    for j in range(1, m):
        cost = stppy.alignment.create_temporal_similarity_matrix(
            avalanches[local[0]], avalanches[local[j]], K_norm)
        d, p, _ = stppy.alignment.dtw_on_similarity_matrix(cost)
        sim[0, j] = sim[j, 0] = d
        paths[0, j] = p
        paths[j, 0] = [[(q[1], q[0]) for q in path] for path in p] if p else p
    return local, sim, paths


def consensus(avalanches, K_norm, similarity, clusters, cid):
    member_idx = np.flatnonzero(clusters == cid)
    sub = similarity[np.ix_(member_idx, member_idx)]
    medoid = int(member_idx[int(np.argmin(sub.sum(1)))])
    local, sim, paths = medoid_paths(avalanches, K_norm, member_idx, medoid)
    with contextlib.redirect_stdout(_io.StringIO()):
        _al, average, _info, medoid_local, _c = stppy.clustering.build_consensus_alignment(
            cluster_id=0, clusters=np.zeros(len(local), dtype=np.int64),
            avalanche_arrays=[avalanches[g] for g in local],
            avalanche_paths=paths, similarity_clean=sim)
    if medoid_local != 0:
        raise RuntimeError(f"medoid moved for cluster {cid}")
    return average, member_idx, medoid


def main() -> int:
    avalanches, _times = cfg.load_pool()
    _, K_norm = cfg.ground_metric_norm()
    similarity = np.load(cfg.SIMILARITY)["similarity_matrix"]
    if len(similarity) != len(avalanches):
        raise ValueError(f"matrix {len(similarity)} != pool {len(avalanches)}")

    clusters, info = stppy.clustering.simple_hierarchical_clustering(
        similarity, threshold=cfg.CLUSTER_THRESHOLD, method=cfg.CLUSTER_METHOD,
        min_cluster_size=cfg.MIN_CLUSTER_SIZE)
    print(f"{len(info)} clusters, {(clusters == -1).sum()} unassigned of {len(avalanches)}",
          flush=True)

    averages, members, medoids = {}, {}, {}
    for cid in range(len(info)):
        averages[cid], members[cid], medoids[cid] = consensus(
            avalanches, K_norm, similarity, clusters, cid)
        print(f"  cluster {cid}: n={len(members[cid])} T={averages[cid].shape[1]}", flush=True)
    keep = stppy.clustering.remove_duplicate_polarities(averages, threshold=cfg.DEDUP_THRESHOLD)
    print(f"kept after polarity dedup: {len(keep)} of {len(info)}")

    meta = dict(threshold=cfg.CLUSTER_THRESHOLD, method=cfg.CLUSTER_METHOD,
                min_cluster_size=cfg.MIN_CLUSTER_SIZE, n_arrays=len(avalanches),
                n_clusters=len(info), n_kept=len(keep))
    np.savez_compressed(
        cfg.CLUSTERS_NPZ, meta=np.array(json.dumps(meta)), kept=np.array(sorted(keep)),
        labels=clusters, medoids=np.array([medoids[c] for c in range(len(info))]),
        **{f"consensus_{c:02d}": averages[c] for c in range(len(info))},
        **{f"members_{c:02d}": members[c] for c in range(len(info))})
    with open(cfg.CLUSTERS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster", "n_members", "medoid", "n_frames", "kept_after_dedup"])
        for c in range(len(info)):
            w.writerow([c, len(members[c]), medoids[c], averages[c].shape[1], int(c in keep)])

    cfg.FIGDIR.mkdir(parents=True, exist_ok=True)
    for old in cfg.FIGDIR.glob("cluster_*.png"):
        old.unlink()
    for i, cid in enumerate(sorted(keep, key=lambda c: -len(members[c]))):
        fig = stppy.visualization.plot_cluster_average(
            averages[cid], cfg.plot_positions(), cluster_id=cid, vlim=[-1, 1], show=False)
        fig.savefig(cfg.FIGDIR / f"cluster_{i:02d}_c{cid}_n{len(members[cid])}.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)
    print(f"{len(keep)} consensus figures -> {cfg.FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
