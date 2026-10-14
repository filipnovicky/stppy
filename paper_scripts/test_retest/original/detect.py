#!/usr/bin/env python
"""Stage 1: quickstart detection for one recording of RUN (PBS array index = recording).

    RUN=ds4148_EC REC_INDEX=1 python detect.py
"""
from __future__ import annotations

import contextlib
import io as _io
import os
import pickle

import mne
import numpy as np

import config as cfg
from config import stppy


def main() -> int:
    files = cfg.recordings()
    idx = int(os.environ.get("REC_INDEX", os.environ.get("PBS_ARRAY_INDEX", 1)))
    if not 1 <= idx <= len(files):
        raise SystemExit(f"REC_INDEX must be in 1..{len(files)}, got {idx}")
    path = files[idx - 1]
    out = cfg.PER_SUBJECT_DIR / f"{path.stem}.pkl"
    cfg.PER_SUBJECT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(path, allow_pickle=True) as d:
        sig = d[cfg.SIGNAL_KEY].astype(np.float64)
        sfreq = float(d["sfreq"])
    if sfreq != cfg.SFREQ:
        # Same FFT resampler the v4 preprocessing uses (raw.resample); data are already 1-40 Hz.
        down = sfreq / cfg.SFREQ
        if down != int(down):
            raise ValueError(f"{path.name}: sfreq {sfreq} is not a multiple of {cfg.SFREQ}")
        sig = mne.filter.resample(sig, down=down, npad="auto", verbose=False)
    K, _ = cfg.ground_metric_norm()

    with contextlib.redirect_stdout(_io.StringIO()):
        masks = stppy.detection.create_avalanche_filter([sig], cfg.CROSSOVER_THRESHOLDS)
        all_av, all_t = stppy.detection.detect_avalanches(
            [sig], masks, cfg.CROSSOVER_THRESHOLDS, min_duration=cfg.MIN_DURATION,
            verbose=False,
        )
        filt_av, filt_t = stppy.detection.filter_avalanche_propagation(
            all_av, all_t, K, cfg.POSITIONS, min_active_neighbors=cfg.MIN_ACTIVE_NEIGHBORS,
        )
        peaks = stppy.detection.convert_to_peaks(filt_av)[0]

    with open(out, "wb") as fh:
        pickle.dump(dict(recording=path.stem, peaks=peaks, times=filt_t[0],
                         n_raw=len(all_av[0])), fh)
    print(f"{path.stem}: {len(all_av[0])} raw -> {len(peaks)} after neighbour filter -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
