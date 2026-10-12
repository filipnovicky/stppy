#!/usr/bin/env python
"""Stage 2, chunked: the same strided block as matrix_block.py, but with no task over ~1 CPU-h.

matrix_block.py gives each row to one process. For W the longest events (up to ~5,900 frames)
make single rows cost 20-50 CPU-h, over the 24 h walltime, and the block is lost because it
only writes at the end. Here each row's column range is cut into chunks of bounded cost
(c * T_i * sum T_j <= CHUNK_FRAMES), so long rows spread across the pool, and finished chunks
are checkpointed every CKPT_SEC, so a killed job resumes instead of starting over. The pair
distance is matrix_block.pair_distance, and the output file is the one merge.py reads.

    RUN=anphy61_W BLOCK=13 NBLOCKS=16 NCPUS=32 python matrix_chunked.py
"""
from __future__ import annotations

import os

for _v in ("OMP", "MKL", "OPENBLAS", "NUMBA"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "1")

import time                                                               # noqa: E402
from multiprocessing import Pool                                          # noqa: E402

import numpy as np                                                        # noqa: E402

import config as cfg                                                      # noqa: E402
import matrix_block as mb                                                 # noqa: E402

BLOCK = int(os.environ.get("BLOCK", os.environ.get("PBS_ARRAY_INDEX", 1)))
NBLOCKS = int(os.environ.get("NBLOCKS", 1))
NCPUS = int(os.environ.get("NCPUS", 32))
# 5.5e7 frame pairs ~ 1 CPU-h at the measured 6.5e-5 s per frame pair.
CHUNK_FRAMES = float(os.environ.get("CHUNK_FRAMES", 5.5e7))
CKPT_SEC = float(os.environ.get("CKPT_SEC", 1800))


def chunks(rows: np.ndarray, T: np.ndarray) -> list[tuple[int, int, int]]:
    """(i, j0, j1) covering columns j >= i of every row, each chunk <= CHUNK_FRAMES of work."""
    out = []
    n = len(T)
    for i in rows.tolist():
        j0 = i
        acc = 0.0
        for j in range(i, n):
            acc += 2.0 * T[i] * T[j]
            if acc >= CHUNK_FRAMES:
                out.append((i, j0, j + 1))
                j0, acc = j + 1, 0.0
        if j0 < n:
            out.append((i, j0, n))
    return out


def _chunk(task):
    i, j0, j1 = task
    a = np.zeros(j1 - j0)
    b = np.zeros(j1 - j0)
    for k, j in enumerate(range(j0, j1)):
        if j > i:
            a[k] = mb.pair_distance(mb._AV[i], mb._AV[j])
        b[k] = mb.pair_distance(mb._AV[i], -mb._AV[j])
    return task, a, b


def partial_path():
    return cfg.BLOCK_DIR / f"block_{BLOCK:02d}of{NBLOCKS:02d}.partial.npz"


def main() -> int:
    if not 1 <= BLOCK <= NBLOCKS:
        raise SystemExit(f"BLOCK must be in 1..{NBLOCKS}, got {BLOCK}")
    cfg.BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    out = mb.block_path(BLOCK, NBLOCKS)
    if out.exists():
        print(f"block already on disk: {out}")
        return 0
    avalanches, _ = cfg.load_pool()
    n_orig = len(avalanches) // 2
    originals = avalanches[:n_orig]
    T = np.array([a.shape[1] for a in originals], float)
    _, K_norm = cfg.ground_metric_norm()

    rows = np.arange(BLOCK - 1, n_orig, NBLOCKS)
    tasks = chunks(rows, T)
    A = np.zeros((len(rows), n_orig))
    B = np.zeros((len(rows), n_orig))
    done = np.zeros(len(tasks), bool)
    part = partial_path()
    if part.exists():
        with np.load(part) as p:
            if int(p["n_orig"]) == n_orig and len(p["done"]) == len(tasks):
                A, B, done = p["A"].copy(), p["B"].copy(), p["done"].copy()
                print(f"resumed {int(done.sum())}/{len(tasks)} chunks from {part}", flush=True)
    todo = [t for t, d in zip(tasks, done) if not d]
    index = {t: k for k, t in enumerate(tasks)}
    print(f"block {BLOCK}/{NBLOCKS}  n_orig={n_orig}  {len(rows)} rows = {len(tasks)} chunks "
          f"({len(todo)} to do) on {NCPUS} cpus", flush=True)

    t0 = last = time.time()
    with Pool(NCPUS, initializer=mb._init, initargs=(originals, K_norm)) as pool:
        # Largest chunks first so the tail of the run is short ones.
        todo.sort(key=lambda t: -T[t[0]] * T[t[1]:t[2]].sum())
        for n_done, (task, a, b) in enumerate(pool.imap_unordered(_chunk, todo, chunksize=1), 1):
            i, j0, j1 = task
            k = int(np.searchsorted(rows, i))
            A[k, j0:j1] = a
            B[k, j0:j1] = b
            done[index[task]] = True
            if time.time() - last > CKPT_SEC:
                np.savez(part, A=A, B=B, done=done, n_orig=n_orig)
                last = time.time()
                print(f"  {int(done.sum())}/{len(tasks)} chunks  {last - t0:6.0f}s  checkpointed",
                      flush=True)
    np.savez_compressed(out, rows=rows, A=A, B=B, n_orig=n_orig)
    if part.exists():
        part.unlink()
    print(f"{len(rows)} rows in {time.time() - t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
