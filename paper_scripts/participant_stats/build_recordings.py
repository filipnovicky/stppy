#!/usr/bin/env python3
"""Build recordings.csv: retained duration and avalanche count per recording.

Duration is the length of each preprocessed .set file (segments rejected in
preprocessing are already cut out; ``n_boundaries`` counts the cuts).
``n_avalanches`` is the number of base avalanches detected in the recording.

    cd paper_scripts && python -m participant_stats.build_recordings
"""

from __future__ import annotations

import pickle

import mne
import pandas as pd

from . import paths


def _find_set(name: str):
    hits = list(paths.RAW_DIR.rglob(name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: expected one file under {paths.RAW_DIR}, found {len(hits)}")
    return hits[0]


def recording_row(name: str, n_avalanches: int) -> dict:
    raw = mne.io.read_raw_eeglab(_find_set(name), preload=False, verbose="error")
    n_boundaries = sum(d.lower() == "boundary" for d in raw.annotations.description)
    return {"dataset": name, "duration_s": raw.n_times / raw.info["sfreq"],
            "sfreq": raw.info["sfreq"], "n_boundaries": n_boundaries,
            "n_avalanches": n_avalanches}


def main() -> None:
    with open(paths.PAPER_DIR / "files.pkl", "rb") as f:
        files = pickle.load(f)
    with open(paths.PAPER_DIR / "peak_avalanche_times.pkl", "rb") as f:
        times = pickle.load(f)
    rows = [recording_row(name, len(t)) for name, t in zip(files, times)]
    paths.OUT_PARTICIPANT_LEVEL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.RECORDINGS_CSV, index=False)
    print(f"{len(rows)} recordings -> {paths.RECORDINGS_CSV}")


if __name__ == "__main__":
    main()
