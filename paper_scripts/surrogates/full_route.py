#!/usr/bin/env python3
"""The paper's full route on a surrogate pool: UOT + subDTW -> clusters -> consensi.

    python full_route.py blocks <arm>      BLOCK/NBLOCKS from env (PBS array), rows by stride
    python full_route.py cluster <arm>     merge blocks, cluster, consensus, compare

The pair distance is stppy's `create_temporal_similarity_matrix` + `dtw_on_similarity_matrix`.
Clustering is the paper's stage 3 unchanged: threshold 3.5, average linkage, min 30, polarity
dedup -0.999, `build_consensus_alignment` on the medoid's DTW paths (recomputed for that row
only, as the full 10k x 10k path matrix is not needed for a consensus).
"""

from __future__ import annotations

import contextlib
import csv
import io as _io
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP", "MKL", "OPENBLAS", "NUMBA"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))                     # paper_scripts (paper_config)
sys.path.insert(0, str(HERE.parents[1]))                  # repository root (stppy)

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402

import compare                                                            # noqa: E402
import paper_config                                                       # noqa: E402
import stppy                                                              # noqa: E402
import stppy.clustering                                                   # noqa: E402
import stppy.visualization                                                # noqa: E402

PAPER_DIR = paper_config.RUN_DIR                  # the paper's templates (cluster_averages.pkl)
POOLS = paper_config.SURROGATE_DIR / "pools"      # pools.py
OUT = paper_config.SURROGATE_DIR / "full"
NCPUS = int(os.environ.get("NCPUS", 32))
BLOCK = int(os.environ.get("BLOCK", os.environ.get("PBS_ARRAY_INDEX", 1)))
NBLOCKS = int(os.environ.get("NBLOCKS", 1))
THRESHOLD = 3.5
MIN_CLUSTER_SIZE = 30
DEDUP = -0.999
N_SHOW = 12

_AV: list[np.ndarray] = []
_KN = None


def load(arm: str) -> list[np.ndarray]:
    with contextlib.redirect_stdout(_io.StringIO()):
        return stppy.io.load_avalanche_data(str(POOLS / f"{arm}_peak_avalanches.pkl"),
                                            min_time_length=11, include_flipped=True)


def ground() -> np.ndarray:
    with contextlib.redirect_stdout(_io.StringIO()):
        K, _ = stppy.utils.create_eeg_ground_metric(compare.POS_FILE)
        return stppy.utils.normalize_distance_matrix(K, method="mean")


def pair(A, B):
    cost = stppy.alignment.create_temporal_similarity_matrix(A, B, _KN)
    return stppy.alignment.dtw_on_similarity_matrix(cost)


def _row(i):
    n = len(_AV) // 2
    a, b = np.zeros(n), np.zeros(n)
    for j in range(i + 1, n):
        a[j] = pair(_AV[i], _AV[j])[0]
    for j in range(i, n):
        b[j] = pair(_AV[i], _AV[n + j])[0]
    return i, a, b


def blocks(arm: str) -> None:
    global _AV, _KN
    out = OUT / arm / f"block_{BLOCK:02d}of{NBLOCKS:02d}.npz"
    if out.exists():
        print(f"{out} exists")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    _AV, _KN = load(arm), ground()
    n = len(_AV) // 2
    rows = np.arange(BLOCK - 1, n, NBLOCKS)
    t0 = time.time()
    A, B = np.zeros((len(rows), n)), np.zeros((len(rows), n))
    with mp.get_context("fork").Pool(NCPUS) as pool:
        for i, a, b in pool.imap_unordered(_row, rows.tolist(), chunksize=1):
            k = int(np.searchsorted(rows, i))
            A[k], B[k] = a, b
    np.savez_compressed(out, rows=rows, A=A, B=B, n_orig=n)
    print(f"{arm} block {BLOCK}/{NBLOCKS}: {len(rows)} rows of {n} in "
          f"{time.time() - t0:.0f}s -> {out}", flush=True)


def merged(arm: str, n: int) -> np.ndarray:
    A, B = np.zeros((n, n)), np.zeros((n, n))
    seen = np.zeros(n, bool)
    for f in sorted((OUT / arm).glob(f"block_*of{NBLOCKS:02d}.npz")):
        with np.load(f) as z:
            if int(z["n_orig"]) != n:
                raise SystemExit(f"{f} is for n={int(z['n_orig'])}, pool has {n}")
            A[z["rows"]], B[z["rows"]] = z["A"], z["B"]
            seen[z["rows"]] = True
    if not seen.all():
        raise SystemExit(f"{arm}: {int((~seen).sum())} rows missing -- blocks not finished")
    A = A + A.T
    B = np.triu(B) + np.triu(B, 1).T
    return np.block([[A, B], [B, A]])


def consensus(sim, labels, cid):
    """build_consensus_alignment on the medoid's recomputed DTW paths."""
    members = np.flatnonzero(labels == cid)
    medoid = int(members[int(np.argmin(sim[np.ix_(members, members)].sum(1)))])
    local = [medoid] + [int(g) for g in members if int(g) != medoid]
    m = len(local)
    s = np.full((m, m), 1e9)
    np.fill_diagonal(s, 0.0)
    paths = np.full((m, m), None, dtype=object)
    paths[0, 0] = [[(t, t) for t in range(_AV[local[0]].shape[1])]]
    for j in range(1, m):
        d, p, _ = pair(_AV[local[0]], _AV[local[j]])
        s[0, j] = s[j, 0] = d
        paths[0, j] = p
        paths[j, 0] = [[(q[1], q[0]) for q in path] for path in p] if p else p
    with contextlib.redirect_stdout(_io.StringIO()):
        _al, avg, _info, med_local, _c = stppy.clustering.build_consensus_alignment(
            cluster_id=0, clusters=np.zeros(m, dtype=np.int64),
            avalanche_arrays=[_AV[g] for g in local], avalanche_paths=paths,
            similarity_clean=s)
    if med_local != 0:
        raise RuntimeError(f"medoid moved for cluster {cid}")
    return avg, members, medoid


def cluster(arm: str) -> None:
    global _AV, _KN
    _AV, _KN = load(arm), ground()
    n = len(_AV) // 2
    d = OUT / arm
    d.mkdir(parents=True, exist_ok=True)
    paper = stppy.io.load_pickle(str(PAPER_DIR / "cluster_averages.pkl"))
    ref = [paper[k] for k in sorted(paper)]
    ref_arrow = float(compare.arrow_magnitudes(ref).max())
    if n < MIN_CLUSTER_SIZE:
        avgs, members, keep, labels = {}, {}, [], np.full(2 * n, -1)
    else:
        sim = merged(arm, n)
        with contextlib.redirect_stdout(_io.StringIO()):
            labels, info = stppy.clustering.simple_hierarchical_clustering(
                sim, threshold=THRESHOLD, method="average", min_cluster_size=MIN_CLUSTER_SIZE)
        avgs, members = {}, {}
        for cid in range(len(info)):
            avgs[cid], members[cid], _ = consensus(sim, labels, cid)
        with contextlib.redirect_stdout(_io.StringIO()):
            keep = stppy.clustering.remove_duplicate_polarities(avgs, threshold=DEDUP) \
                if avgs else []
    keep = sorted(keep, key=lambda c: -len(members[c]))
    s = compare.summarise(ref, [avgs[c] for c in keep], ref_arrow)
    np.savez_compressed(d / "clusters.npz", labels=labels, keep=np.array(keep),
                        sizes=np.array([len(members[c]) for c in keep]), match=s["match"],
                        arrows=s["arrows"],
                        **{f"consensus_{i:02d}": avgs[c] for i, c in enumerate(keep)})
    if keep:
        show = keep[:N_SHOW]
        fig = stppy.visualization.plot_clusters_with_arrows(
            {c: avgs[c] for c in show}, compare.POS_FILE,
            cluster_sizes={c: len(members[c]) for c in show},
            title=f"{arm} -- paper pipeline (UOT + subDTW, threshold 3.5)")
        fig.savefig(d / "clusters_top12.png", dpi=90, bbox_inches="tight")
        plt.close(fig)
    summary = OUT / "summary.csv"
    new = not summary.exists()
    with open(summary, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["arm", "n_avalanches", "n_clusters", "n_kept", "assigned",
                        "match_mean", "null_mean", "null_p95", "n_match_ge_0.8",
                        "n_travelling", "match_per_template"])
        w.writerow([arm, n, len(avgs), len(keep), int((labels >= 0).sum()),
                    round(s["match_mean"], 3), round(s["null_mean"], 3),
                    round(s["null_p95"], 3), int((s["match"] >= 0.8).sum()),
                    s["n_travelling"], " ".join(f"{v:.2f}" for v in s["match"])])
    print(f"{arm}: {n} avalanches, {len(avgs)} clusters, {len(keep)} kept, match "
          f"{s['match_mean']:.3f} (null {s['null_mean']:.3f}), travelling "
          f"{s['n_travelling']}", flush=True)


if __name__ == "__main__":
    mode, arm = sys.argv[1], sys.argv[2]
    {"blocks": blocks, "cluster": cluster}[mode](arm)
