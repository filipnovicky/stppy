#!/usr/bin/env python3
"""Whole-recording power spectrum of every paper recording (response to reviewers, R1.1 / R2.5).

Reviewer 1 asks whether the psilocybin shift from oscillating to stable sequences restates the
well-known psilocybin reduction of alpha power. This script gives every recording's ordinary
alpha power, independent of any sequence: Welch spectrum of the whole preprocessed recording
(datasets.pkl: 64 channels, 250 Hz), 1.024 s Hann segments (as the sequence windows), 50%
overlap, zero-padded to NFFT, averaged over channels.

Output: OUT_DIR/recording_spectra.npz (dataset, drug, subject, task, duration_s, freqs, psd).
Features and the figure are in plot_alpha_vs_polarity.py.
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
import stppy.io  # noqa: E402
from psiconnect_utils import parse_condition  # noqa: E402
from spectral_tools import SFREQ  # noqa: E402

import paper_config  # noqa: E402  (all paths: paper_config.py)

PAPER_DIR = paper_config.RUN_DIR
OUT_DIR = paper_config.results("spectral")
SEGMENT = 256  # 1.024 s, the sequence-window length
NFFT = 1024


def recording_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Channel-averaged Welch power density of one (n_channels, n_samples) recording."""
    freqs, psd = welch(x, fs=SFREQ, window="hann", nperseg=SEGMENT, noverlap=SEGMENT // 2,
                       nfft=NFFT, detrend="linear", scaling="density", axis=-1)
    return freqs, psd.mean(axis=0)


def main() -> int:
    t0 = time.time()
    data = stppy.io.load_pickle(str(PAPER_DIR / "datasets.pkl"))
    names = stppy.io.load_pickle(str(PAPER_DIR / "files.pkl"))
    if len(data) != len(names):
        raise ValueError(f"{len(data)} recordings but {len(names)} file names")
    print(f"{len(data)} recordings loaded [{time.time() - t0:.0f}s]", flush=True)
    psds, freqs = [], None
    for x in data:
        freqs, p = recording_psd(np.asarray(x, dtype=float))
        psds.append(p)
    meta = [parse_condition(n) for n in names]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "recording_spectra.npz",
        dataset=np.array(names), drug=np.array([m[0] for m in meta]),
        subject=np.array([m[1] for m in meta]), task=np.array([m[2] for m in meta]),
        duration_s=np.array([x.shape[1] / SFREQ for x in data]),
        freqs=freqs, psd=np.stack(psds),
    )
    print(f"saved recording_spectra.npz [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
