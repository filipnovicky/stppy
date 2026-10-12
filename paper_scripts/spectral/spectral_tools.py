"""Windowing, spectra, alpha-peak features and permutation test of the sequence spectral analysis.

Pure functions on numpy arrays, no file I/O (tested in test_spectral_tools.py).
"""

import numpy as np
from scipy.signal import periodogram

SFREQ = 250.0
ALPHA_BAND = (8.0, 13.0)
PEAK_BAND = (5.0, 15.0)  # search range of the spectral peak
FIT_RANGE = (2.0, 40.0)  # aperiodic (1/f) fit range ...
FIT_EXCLUDE = (5.0, 15.0)  # ... without the peak range
MIN_PEAK_DB = 1.0  # a peak must stand >= 1 dB above the aperiodic fit


def extract_window(
    data: np.ndarray, center_s: float, n_samples: int, sfreq: float = SFREQ
) -> np.ndarray | None:
    """(n_channels, n_samples) slice of ``data`` centred on ``center_s`` seconds; None if the
    window does not fit inside the recording."""
    start = int(round(center_s * sfreq)) - n_samples // 2
    if start < 0 or start + n_samples > data.shape[1]:
        return None
    return data[:, start : start + n_samples]


def window_psd(x: np.ndarray, nfft: int, sfreq: float = SFREQ) -> tuple[np.ndarray, np.ndarray]:
    """Hann periodogram (power density) of every channel of ``x``, linear detrend per channel."""
    return periodogram(
        x, fs=sfreq, window="hann", nfft=nfft, detrend="linear", scaling="density", axis=-1
    )


def sample_null_centers(
    n_samples: int,
    busy_s: list[tuple[float, float]],
    window_samples: int,
    n_draws: int,
    rng: np.random.Generator,
    guard_s: float,
    sfreq: float = SFREQ,
    max_tries: int = 10_000,
) -> np.ndarray:
    """Up to ``n_draws`` random window centres (s) in one recording. Each window fits inside the
    recording and each centre lies at least ``guard_s`` from every busy interval (start, end)."""
    half = window_samples / 2 / sfreq
    lo, hi = half, n_samples / sfreq - half
    if hi <= lo:
        return np.empty(0)
    busy = np.asarray(busy_s, dtype=float).reshape(-1, 2)
    centers = []
    for _ in range(max_tries):
        if len(centers) == n_draws:
            break
        c = rng.uniform(lo, hi)
        if not np.any((c > busy[:, 0] - guard_s) & (c < busy[:, 1] + guard_s)):
            centers.append(c)
    return np.asarray(centers)


def aperiodic_fit(
    freqs: np.ndarray,
    log_psd: np.ndarray,
    fit_range: tuple[float, float] = FIT_RANGE,
    exclude: tuple[float, float] = FIT_EXCLUDE,
) -> np.ndarray:
    """Straight line in log10 power vs log10 frequency, fitted over ``fit_range`` minus ``exclude``;
    evaluated inside ``fit_range`` (NaN outside)."""
    in_range = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    use = in_range & ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
    coef = np.polyfit(np.log10(freqs[use]), log_psd[use], 1)
    fit = np.full_like(log_psd, np.nan)
    fit[in_range] = np.polyval(coef, np.log10(freqs[in_range]))
    return fit


def peak_frequency(
    freqs: np.ndarray, curve: np.ndarray, band: tuple[float, float] = PEAK_BAND
) -> float:
    """Frequency of the maximum of ``curve`` inside ``band``; NaN if it sits on a band edge."""
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    i = idx[np.argmax(curve[idx])]
    return float("nan") if i in (idx[0], idx[-1]) else float(freqs[i])


def alpha_features(freqs: np.ndarray, log_psd: np.ndarray) -> tuple[float, float]:
    """(alpha height in dB, peak frequency in Hz) of one log10 power spectrum.

    Height = maximum of the aperiodic-corrected spectrum in ALPHA_BAND, in dB. Peak frequency =
    interior maximum of the corrected spectrum in PEAK_BAND, NaN unless >= MIN_PEAK_DB high.
    """
    corrected = 10.0 * (log_psd - aperiodic_fit(freqs, log_psd))
    alpha = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    peak = peak_frequency(freqs, corrected)
    if np.isfinite(peak) and corrected[np.argmin(np.abs(freqs - peak))] < MIN_PEAK_DB:
        peak = float("nan")
    return float(np.max(corrected[alpha])), peak


def unit_means(
    values: np.ndarray, units: np.ndarray, mask: np.ndarray, min_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the rows of ``values`` per unit among ``mask``; units with < ``min_rows`` rows are
    dropped. Returns (means, unit ids)."""
    ids = [u for u in np.unique(units[mask]) if np.sum(mask & (units == u)) >= min_rows]
    means = [values[mask & (units == u)].mean(axis=0) for u in ids]
    return np.asarray(means), np.asarray(ids)


def sign_flip_test(diff: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided one-sample sign-flip permutation test of mean(diff) = 0 (one value per unit),
    on the one-sample t statistic."""
    diff = np.asarray(diff, dtype=float)

    def t_stat(d: np.ndarray) -> np.ndarray:
        return d.mean(axis=-1) / (d.std(axis=-1, ddof=1) / np.sqrt(d.shape[-1]))

    flips = rng.choice([-1.0, 1.0], size=(n_perm, diff.size))
    null = np.abs(t_stat(flips * diff))
    return float((1 + np.sum(null >= abs(t_stat(diff)))) / (n_perm + 1))
