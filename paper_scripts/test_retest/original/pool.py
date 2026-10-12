#!/usr/bin/env python
"""Stage 1b: draw N_ORIG originals evenly across recordings (random within a recording).

Quota is N_ORIG / n_recordings; a recording with fewer events gives all of them and the
shortfall is redistributed over the rest. Writes the stppy pickles in the quickstart's
per-participant list-of-lists layout, so load_avalanche_data reads them unchanged.
"""
from __future__ import annotations

import pickle

import numpy as np

import config as cfg


def quotas(counts: np.ndarray, total: int) -> np.ndarray:
    q = np.zeros_like(counts)
    left = min(total, int(counts.sum()))
    open_ = counts > 0
    while left > 0 and open_.any():
        share = max(left // int(open_.sum()), 1)
        for i in np.flatnonzero(open_):
            take = min(share, counts[i] - q[i], left)
            q[i] += take
            left -= take
            if q[i] == counts[i]:
                open_[i] = False
            if left == 0:
                break
    return q


def main() -> int:
    files = sorted(cfg.PER_SUBJECT_DIR.glob("*.pkl"))
    expected = len(cfg.recordings())
    if len(files) != expected:
        raise SystemExit(f"{len(files)} of {expected} per-recording pickles present")
    recs = [pickle.load(open(f, "rb")) for f in files]
    counts = np.array([len(r["peaks"]) for r in recs])
    q = quotas(counts, cfg.N_ORIG)
    rng = np.random.default_rng(cfg.SEED)

    peaks, times = [], []
    for r, k in zip(recs, q):
        pick = np.sort(rng.choice(len(r["peaks"]), size=int(k), replace=False))
        peaks.append([r["peaks"][i] for i in pick])
        times.append([r["times"][i] for i in pick])

    pickle.dump(peaks, open(cfg.PEAKS_PKL, "wb"))
    pickle.dump(times, open(cfg.TIMES_PKL, "wb"))
    with open(cfg.SUBJECTS_TXT, "w") as fh:
        fh.write("recording\tn_raw\tn_filtered\tn_drawn\n")
        for r, c, k in zip(recs, counts, q):
            fh.write(f"{r['recording']}\t{r['n_raw']}\t{c}\t{k}\n")
    durs = [a.shape[1] for p in peaks for a in p]
    print(f"{counts.sum()} filtered events over {len(recs)} recordings "
          f"(min {counts.min()}, max {counts.max()}); drew {q.sum()}")
    print(f"duration samples: median {int(np.median(durs))}, p95 {int(np.percentile(durs, 95))}, "
          f"max {max(durs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
