#!/usr/bin/env python
"""Stage 2: one strided block of the Wasserstein + subDTW matrix (PBS array index = block).

Pair distance = create_temporal_similarity_matrix -> dtw_on_similarity_matrix, the two calls
compute_similarity_block makes. Only originals are aligned: A = d(x_i, x_j), B = d(x_i, -x_j),
and the full matrix over originals + flips is [[A, B], [B, A]]. Rows are strided so blocks
cost the same (row i has ~2(n - i) pairs).

    BLOCK=1 NBLOCKS=5 NCPUS=32 python matrix_block.py
"""
from __future__ import annotations

import os

for _v in ("OMP", "MKL", "OPENBLAS", "NUMBA"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "1")

import time                                                               # noqa: E402
from multiprocessing import Pool                                          # noqa: E402

import numpy as np                                                        # noqa: E402

import config as cfg                                                      # noqa: E402
from config import stppy                                                  # noqa: E402

BLOCK = int(os.environ.get("BLOCK", os.environ.get("PBS_ARRAY_INDEX", 1)))
NBLOCKS = int(os.environ.get("NBLOCKS", 1))
NCPUS = int(os.environ.get("NCPUS", 32))

_AV, _N, _KN = None, 0, None


def block_path(block: int, nblocks: int):
    return cfg.BLOCK_DIR / f"block_{block:02d}of{nblocks:02d}.npz"


def _init(originals, K_norm):
    global _AV, _N, _KN
    _AV, _N, _KN = originals, len(originals), K_norm


def pair_distance(a, b):
    cost = stppy.alignment.create_temporal_similarity_matrix(a, b, _KN)
    return stppy.alignment.dtw_on_similarity_matrix(cost)[0]


def _row(i):
    a = np.zeros(_N)
    b = np.zeros(_N)
    for j in range(i + 1, _N):
        a[j] = pair_distance(_AV[i], _AV[j])
    for j in range(i, _N):
        b[j] = pair_distance(_AV[i], -_AV[j])
    return i, a, b


def main() -> int:
    if not 1 <= BLOCK <= NBLOCKS:
        raise SystemExit(f"BLOCK must be in 1..{NBLOCKS}, got {BLOCK}")
    cfg.BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    out = block_path(BLOCK, NBLOCKS)
    avalanches, _ = cfg.load_pool()
    n_orig = len(avalanches) // 2
    originals = avalanches[:n_orig]
    _, K_norm = cfg.ground_metric_norm()

    rows = np.arange(BLOCK - 1, n_orig, NBLOCKS)
    print(f"block {BLOCK}/{NBLOCKS}  n_orig={n_orig}  {len(rows)} rows on {NCPUS} cpus",
          flush=True)
    t0 = time.time()
    A = np.zeros((len(rows), n_orig))
    B = np.zeros((len(rows), n_orig))
    with Pool(NCPUS, initializer=_init, initargs=(originals, K_norm)) as pool:
        for done, (i, a, b) in enumerate(
                pool.imap_unordered(_row, rows.tolist(), chunksize=1), 1):
            k = int(np.searchsorted(rows, i))
            A[k], B[k] = a, b
            if done % 50 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(rows)} rows  {el:6.0f}s  "
                      f"eta {el * (len(rows) - done) / done:6.0f}s", flush=True)
    np.savez_compressed(out, rows=rows, A=A, B=B, n_orig=n_orig)
    print(f"{len(rows)} rows in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    # Entry point now delegates to the chunked, checkpointed worker (same output, bit-identical
    # distances): single long W rows exceeded the 24 h walltime. main() above is kept for reference.
    import matrix_chunked
    raise SystemExit(matrix_chunked.main())
