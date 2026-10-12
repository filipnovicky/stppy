#!/usr/bin/env python3
"""Stage A: one surrogate per recording, then the paper's avalanche detection, unchanged.

Input is the paper run's `datasets.pkl` (RUN_DIR: 429 recordings, 64 ch, 250 Hz). For each arm
every recording gets one surrogate (seed SEED_BASE + 1000 * REAL + recording index), and the
paper's detection runs on it:

    create_avalanche_filter(+-3 SD) -> detect_avalanches(min_duration 11)
    -> filter_avalanche_propagation(5/8 both-polarity neighbours) -> convert_to_peaks

Arms: real (must reproduce the paper's 5108 avalanches), iaaft, ft, modewise, shift.

Writes to SURROGATE_DIR/pools/ (paper_config.py):
    <arm>_peak_avalanches.pkl, <arm>_peak_avalanche_times.pkl   the clustering inputs
    <arm>_durations.npz   durations/sizes of EVERY supra-3SD run (no min duration, no
                          filter) -- the duration distributions of Figure 10

    cd paper_scripts && ARMS=real,iaaft python surrogates/pools.py
"""

from __future__ import annotations

import contextlib
import io as _io
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP", "MKL", "OPENBLAS", "NUMBA"):
    os.environ.setdefault(f"{_v}_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                      # surrogates.py
sys.path.insert(0, str(HERE.parent))               # paper_config.py
sys.path.insert(0, str(HERE.parents[1]))           # repository root (stppy)

import numpy as np                                                        # noqa: E402

import paper_config                                                       # noqa: E402
import stppy                                                              # noqa: E402
from surrogates import (circular_shift, modewise_iaaft,                   # noqa: E402
                        modewise_phase_randomise, multivariate_iaaft, phase_randomise)

POS_FILE = str(paper_config.POS_FILE)
PAPER_DIR = paper_config.RUN_DIR
OUT = paper_config.SURROGATE_DIR / "pools"
ARMS = os.environ.get("ARMS", "real,iaaft,ft,modewise,shift").split(",")
NCPUS = int(os.environ.get("NCPUS", 32))
LIMIT = int(os.environ.get("LIMIT", 0))
REAL = int(os.environ.get("REAL", 0))   # surrogate realisation
SEED_BASE = 20260925
MIN_SHIFT = 500                         # 2 s at 250 Hz (circular-shift arm)
THRESHOLDS = [-3, 3]
MIN_DURATION = 11
MIN_ACTIVE_NEIGHBORS = 5

_DATA: list[np.ndarray] = []
_K = None


def make_surrogate(kind: str, x: np.ndarray, i: int) -> np.ndarray:
    rng = np.random.default_rng(SEED_BASE + 1000 * REAL + i)
    if kind == "real":
        return x
    if kind == "iaaft":
        return multivariate_iaaft(x, rng)
    if kind == "ft":
        return phase_randomise(x, rng)
    if kind == "modewise":
        return modewise_phase_randomise(x, rng)
    if kind == "modewise_iaaft":
        return modewise_iaaft(x, rng)
    if kind == "shift":
        return circular_shift(x, rng, MIN_SHIFT)[0]
    raise ValueError(f"unknown surrogate arm {kind!r}")


def run_lengths(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(durations, sizes) of contiguous samples where any channel has |z| >= 3."""
    z = (sig - sig.mean(1, keepdims=True)) / sig.std(1, keepdims=True)
    active = (np.abs(z) >= 3).sum(0)
    d = np.diff(np.r_[0, (active > 0).astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    csum = np.r_[0, np.cumsum(active)]
    return (ends - starts).astype(np.int32), (csum[ends] - csum[starts]).astype(np.int32)


def detect(sig: np.ndarray):
    with contextlib.redirect_stdout(_io.StringIO()):
        masks = stppy.detection.create_avalanche_filter([sig], THRESHOLDS)
        av, t = stppy.detection.detect_avalanches([sig], masks, THRESHOLDS,
                                                  min_duration=MIN_DURATION, verbose=False)
        fav, ft = stppy.detection.filter_avalanche_propagation(
            av, t, _K, POS_FILE, min_active_neighbors=MIN_ACTIVE_NEIGHBORS)
        peaks = stppy.detection.convert_to_peaks(fav)
    return len(av[0]), peaks[0], ft[0]


def _one(args):
    kind, i = args
    y = make_surrogate(kind, _DATA[i], i)
    n_raw, peaks, times = detect(y)
    dur, size = run_lengths(y)
    return i, n_raw, peaks, times, dur, size


def run(kind: str) -> None:
    peaks_pkl = OUT / f"{kind}_peak_avalanches.pkl"
    if peaks_pkl.exists():
        print(f"{peaks_pkl} exists, skipping", flush=True)
        return
    n = len(_DATA)
    peaks, times, durs, sizes, raw = [None] * n, [None] * n, [None] * n, [None] * n, 0
    t0 = time.time()
    with mp.get_context("fork").Pool(NCPUS) as pool:
        for i, n_raw, pk, tm, dur, size in pool.imap_unordered(
                _one, [(kind, i) for i in range(n)], chunksize=1):
            peaks[i], times[i], durs[i], sizes[i] = pk, tm, dur, size
            raw += n_raw
    stppy.io.save_pickle(peaks, str(peaks_pkl))
    stppy.io.save_pickle(times, str(OUT / f"{kind}_peak_avalanche_times.pkl"))
    np.savez_compressed(OUT / f"{kind}_durations.npz",
                        durations=np.concatenate(durs), sizes=np.concatenate(sizes),
                        per_recording=np.array([len(d) for d in durs]))
    kept = sum(len(p) for p in peaks)
    print(f"{kind}: {raw} raw avalanches (>= {MIN_DURATION} samples) -> {kept} after the "
          f"5/8 filter, over {n} recordings  [{time.time() - t0:.0f}s]", flush=True)


def main() -> int:
    global _DATA, _K
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _DATA = stppy.io.load_pickle(str(PAPER_DIR / "datasets.pkl"))
    if LIMIT:
        _DATA = _DATA[:LIMIT]
    with contextlib.redirect_stdout(_io.StringIO()):
        _K, _ = stppy.utils.create_eeg_ground_metric(POS_FILE)
    print(f"{len(_DATA)} recordings loaded in {time.time() - t0:.0f}s; arms {ARMS}",
          flush=True)
    for kind in ARMS:
        run(kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
