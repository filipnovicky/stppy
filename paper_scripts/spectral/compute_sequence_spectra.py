#!/usr/bin/env python3
"""Power spectra and time-frequency power of the EEG around the detected sequences.

Sequences: the paper's back-fitted sequences (sequences_by_cluster.pkl) after overlap removal
(main_analysis.remove_overlapping_bursts), stable and oscillating only. EEG: the paper's
preprocessed recordings (datasets.pkl: 64 channels, 250 Hz).

For every sequence
  - PSD: Hann periodogram of a 1.024 s window centred on the sequence midpoint, averaged over
    channels; the same for N_NULL random windows from the same recording (their mean is stored).
  - TFR: Morlet power over +-1 s around the midpoint, averaged over channels; the same for the
    first N_NULL_TFR random windows (their mean is stored).
Random window centres lie at least GUARD_S from every detected sequence of any cluster.

Output: OUT_DIR/sequence_spectra.npz, read by plot_sequence_spectra.py.
"""

import sys
import time
from pathlib import Path

import numpy as np
from mne.time_frequency import tfr_array_morlet

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent.parent)]
import stppy.io  # noqa: E402
from main_analysis import POLARITY_RELABEL, remove_overlapping_bursts  # noqa: E402
from psiconnect_utils import parse_condition  # noqa: E402
from spectral_tools import SFREQ, extract_window, sample_null_centers, window_psd  # noqa: E402

import paper_config  # noqa: E402  (all paths: paper_config.py)

PAPER_DIR = paper_config.RUN_DIR
OUT_DIR = paper_config.results("spectral")

POLARITIES = ("stable", "oscillating")
PSD_WINDOW = 256  # 1.024 s
NFFT = 1024  # zero-padded to 0.244 Hz bins
FMAX = 45.0
N_NULL = 20  # random windows per sequence (PSD)
N_NULL_TFR = 5  # of which used for the TFR
GUARD_S = 0.5
TFR_HALF_S = 1.5  # Morlet epoch +-1.5 s, +-1.0 s kept (edge effects)
TFR_KEEP_S = 1.0
TFR_FREQS = np.arange(3.0, 41.0, 1.0)
TFR_CYCLES = np.maximum(TFR_FREQS / 2.0, 2.0)
TFR_DECIM = 5  # 20 ms steps
SEED = 20260929


def _load_sequences() -> tuple[list[tuple[int, dict]], dict[str, list[tuple[float, float]]]]:
    """(cluster id, sequence) pairs to analyse; every detected sequence interval per recording."""
    raw = stppy.io.load_pickle(str(PAPER_DIR / "sequences_by_cluster.pkl"))
    averages = stppy.io.load_pickle(str(PAPER_DIR / "cluster_averages.pkl"))
    busy: dict[str, list[tuple[float, float]]] = {}
    for seqs in raw.values():
        for s in seqs:
            s["polarity"] = POLARITY_RELABEL.get(s["polarity"], s["polarity"])
            busy.setdefault(s["dataset"], []).append((s["start_time"], s["end_time"]))
    clean = remove_overlapping_bursts(raw, averages)
    rows = [
        (cid, s) for cid, seqs in sorted(clean.items()) for s in seqs if s["polarity"] in POLARITIES
    ]
    return rows, busy


def _mean_psd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs, p = window_psd(x, NFFT)
    keep = freqs <= FMAX
    return freqs[keep], p[:, keep].mean(axis=0)


def _mean_tfr(epochs: np.ndarray) -> np.ndarray:
    """(n_epochs, n_channels, n_times) -> channel- and epoch-mean Morlet power (n_freqs, n_kept)."""
    power = tfr_array_morlet(
        epochs,
        sfreq=SFREQ,
        freqs=TFR_FREQS,
        n_cycles=TFR_CYCLES,
        output="power",
        decim=TFR_DECIM,
        n_jobs=1,
        verbose=False,
    )
    t = np.arange(power.shape[-1]) * TFR_DECIM / SFREQ - TFR_HALF_S
    return power.mean(axis=(0, 1))[:, np.abs(t) <= TFR_KEEP_S + 1e-9]


def _sequence_spectra(
    x: np.ndarray, center: float, busy: list[tuple[float, float]], rng: np.random.Generator
) -> tuple[np.ndarray, ...] | None:
    """(freqs, psd_seq, psd_null, tfr_seq, tfr_null) for one sequence; None near an edge or when
    the recording has no room for N_NULL random windows."""
    tfr_len = int(round(2 * TFR_HALF_S * SFREQ))
    window = extract_window(x, center, PSD_WINDOW)
    epoch = extract_window(x, center, tfr_len)
    nulls = sample_null_centers(x.shape[1], busy, tfr_len, N_NULL, rng, GUARD_S)
    if window is None or epoch is None or len(nulls) < N_NULL:
        return None
    freqs, psd_seq = _mean_psd(window)
    psd_null = np.mean([_mean_psd(extract_window(x, c, PSD_WINDOW))[1] for c in nulls], axis=0)
    null_epochs = np.stack([extract_window(x, c, tfr_len) for c in nulls[:N_NULL_TFR]])
    return freqs, psd_seq, psd_null, _mean_tfr(epoch[None]), _mean_tfr(null_epochs)


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, busy = _load_sequences()
    data = stppy.io.load_pickle(str(PAPER_DIR / "datasets.pkl"))
    index = {name: i for i, name in enumerate(stppy.io.load_pickle(str(PAPER_DIR / "files.pkl")))}
    print(
        f"{len(rows)} sequences, {len(data)} recordings loaded [{time.time() - t0:.0f}s]",
        flush=True,
    )

    rng = np.random.default_rng(SEED)
    results, kept = [], []
    for cid, s in rows:
        center = 0.5 * (s["start_time"] + s["end_time"])
        out = _sequence_spectra(data[index[s["dataset"]]], center, busy[s["dataset"]], rng)
        if out is not None:
            results.append(out)
            kept.append(
                (cid, s["polarity"], s["dataset"], parse_condition(s["dataset"])[1], center)
            )
    freqs, psd_seq, psd_null, tfr_seq, tfr_null = (np.stack(c) for c in zip(*results))
    cluster, polarity, dataset, subject, center_s = (np.array(c) for c in zip(*kept))

    np.savez_compressed(
        OUT_DIR / "sequence_spectra.npz",
        freqs=freqs[0],
        psd_seq=psd_seq,
        psd_null=psd_null,
        tfr_freqs=TFR_FREQS,
        tfr_times=np.arange(-TFR_KEEP_S, TFR_KEEP_S + 1e-9, TFR_DECIM / SFREQ),
        tfr_seq=tfr_seq,
        tfr_null=tfr_null,
        cluster=cluster,
        polarity=polarity,
        dataset=dataset,
        subject=subject,
        center_s=center_s,
    )
    print(
        f"saved {len(kept)} sequences ({len(rows) - len(kept)} skipped at edges) "
        f"[{time.time() - t0:.0f}s]",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
