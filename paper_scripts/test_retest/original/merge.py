#!/usr/bin/env python
"""Stage 2b: assemble strided blocks into similarity.npz ([[A, B], [B, A]]). No partial writes.

    NBLOCKS=5 python merge.py
"""
from __future__ import annotations

import os

import numpy as np

import config as cfg
from matrix_block import block_path

NBLOCKS = int(os.environ.get("NBLOCKS", 1))


def main() -> int:
    paths = [block_path(b, NBLOCKS) for b in range(1, NBLOCKS + 1)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {NBLOCKS} blocks missing, first: {missing[0]}")
    A = B = seen = None
    for path in paths:
        with np.load(path) as blk:
            n = int(blk["n_orig"])
            if A is None:
                A, B, seen = np.zeros((n, n)), np.zeros((n, n)), np.zeros(n, bool)
            elif n != len(A):
                raise SystemExit(f"{path} has n_orig={n}, expected {len(A)}")
            for k, i in enumerate(blk["rows"].astype(int)):
                if seen[i]:
                    raise SystemExit(f"row {i} in more than one block")
                a, b = blk["A"][k], blk["B"][k]
                A[i, i + 1:] = A[i + 1:, i] = a[i + 1:]
                B[i, i:] = b[i:]
                B[i:, i] = b[i:]
                seen[i] = True
    if not seen.all():
        raise SystemExit(f"{int((~seen).sum())} rows never filled -- refusing to write")
    sim = np.block([[A, B], [B, A]])
    np.savez(cfg.SIMILARITY, similarity_matrix=sim, n_orig=len(A))
    off = sim[np.triu_indices(len(sim), 1)]
    print(f"{sim.shape} -> {cfg.SIMILARITY}\n  distances: " +
          "  ".join(f"p{q}={np.percentile(off, q):.2f}" for q in (1, 5, 25, 50, 75, 95)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
