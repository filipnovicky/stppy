"""Null-model surrogates for continuous multichannel EEG (Section 2.1, Figures 9-10).

phase_randomise     multivariate FT surrogate (Prichard & Theiler 1994): ONE random phase per
                    frequency, added to every channel, so all auto- AND cross-spectra are
                    kept exactly -- the full linear spatiotemporal covariance, spatial phase
                    gradients included.
multivariate_iaaft  its amplitude-adjusted version (Schreiber & Schmitz 2000): alternates
                    between imposing the original spectra, with the cross-channel phase
                    differences held fixed (a common phase rotation alpha(f) chosen to stay
                    closest to the current iterate), and rank-remapping every channel onto its
                    own original values. Returned after the rank step, so each channel's
                    amplitude distribution is EXACT and the spectra are matched to within the
                    iteration tolerance. This is the version avalanche detection needs: the
                    +-3 SD threshold is a statement about each channel's amplitude
                    distribution, which a Gaussian surrogate would change.
modewise_phase_randomise
                    the NO-PROPAGATION null: rotate onto the data's spatial PCA modes, give
                    each mode its OWN random phases, rotate back. Every mode keeps its exact
                    periodogram, so the zero-lag spatial covariance (every topography) is kept
                    in expectation, but the lagged cross-mode phase relations -- the only
                    thing that lets a pattern move rather than grow and decay in place -- are
                    destroyed. Paired with phase_randomise it isolates linear propagation.
modewise_iaaft      mode-wise phase randomisation made amplitude-adjusted: IAAFT run in the
                    spatial-mode basis (each mode keeps its periodogram, with its OWN phases)
                    and rank-remapped per channel. Keeps zero-lag covariance and every
                    channel's amplitude distribution, removes directed lagged coupling. Every
                    step is time-reversal equivariant and the start is time-reversible, so the
                    output has no preferred direction: patterns can grow and decay in place
                    but not travel. The amplitude-matched no-propagation null next to IAAFT.
circular_shift      independent circular shift per channel (a "temporal-shift"
                    control): each channel is kept exactly, cross-channel timing is destroyed.

All three treat the recording as periodic. The wrap-around joins the recording's last sample
to its first; for a 5-minute recording that is one discontinuity among ~75k samples.
"""

from __future__ import annotations

import numpy as np

IAAFT_MAX_ITER = 200
IAAFT_TOL = 1e-8          # stop when the relative spectral misfit changes by less than this


def _random_phases(n_freq: int, n: int, rng) -> np.ndarray:
    """Common phase per frequency; DC (and Nyquist for even n) stay real."""
    theta = rng.uniform(0.0, 2.0 * np.pi, n_freq)
    theta[0] = 0.0
    if n % 2 == 0:
        theta[-1] = 0.0
    return theta


def phase_randomise(x: np.ndarray, rng) -> np.ndarray:
    """Multivariate FT surrogate of x (channels x samples)."""
    n = x.shape[1]
    X = np.fft.rfft(x, axis=1)
    theta = _random_phases(X.shape[1], n, rng)
    return np.fft.irfft(X * np.exp(1j * theta)[None, :], n=n, axis=1)


def _rank_remap(y: np.ndarray, sorted_x: np.ndarray) -> np.ndarray:
    ranks = np.argsort(np.argsort(y, axis=1, kind="stable"), axis=1, kind="stable")
    return np.take_along_axis(sorted_x, ranks, axis=1)


def multivariate_iaaft(x: np.ndarray, rng, max_iter: int = IAAFT_MAX_ITER,
                       tol: float = IAAFT_TOL) -> np.ndarray:
    """Multivariate IAAFT surrogate of x (channels x samples)."""
    n = x.shape[1]
    X = np.fft.rfft(x, axis=1)                      # target: magnitudes + relative phases
    sorted_x = np.sort(x, axis=1)
    y = _rank_remap(phase_randomise(x, rng), sorted_x)
    norm = float(np.sum(np.abs(X) ** 2))
    prev = np.inf
    for _ in range(max_iter):
        Y = np.fft.rfft(y, axis=1)
        # common rotation alpha(f) that brings X e^{i alpha} closest to Y, all channels jointly
        alpha = np.angle(np.sum(Y * np.conj(X), axis=0))
        y = _rank_remap(np.fft.irfft(X * np.exp(1j * alpha)[None, :], n=n, axis=1),
                        sorted_x)
        misfit = float(np.sum((np.abs(np.fft.rfft(y, axis=1)) - np.abs(X)) ** 2)) / norm
        if abs(prev - misfit) < tol * max(misfit, 1e-30):
            break
        prev = misfit
    return y


def spatial_modes(x: np.ndarray) -> np.ndarray:
    """Orthonormal spatial PCA modes (channels x channels), strongest first."""
    xc = x - x.mean(1, keepdims=True)
    _w, V = np.linalg.eigh(xc @ xc.T / xc.shape[1])
    return V[:, ::-1]


def modewise_phase_randomise(x: np.ndarray, rng) -> np.ndarray:
    """Independent FT surrogate of every spatial mode, mapped back to channels."""
    n = x.shape[1]
    mean = x.mean(1, keepdims=True)
    U = spatial_modes(x)
    M = np.fft.rfft(U.T @ (x - mean), axis=1)
    theta = np.stack([_random_phases(M.shape[1], n, rng) for _ in range(M.shape[0])])
    return U @ np.fft.irfft(M * np.exp(1j * theta), n=n, axis=1) + mean


def modewise_iaaft(x: np.ndarray, rng, max_iter: int = IAAFT_MAX_ITER,
                   tol: float = IAAFT_TOL) -> np.ndarray:
    """Mode-wise IAAFT surrogate of x (channels x samples)."""
    n = x.shape[1]
    mean = x.mean(1, keepdims=True)
    U = spatial_modes(x)
    target = np.abs(np.fft.rfft(U.T @ (x - mean), axis=1))      # per-mode amplitudes
    sorted_x = np.sort(x, axis=1)
    y = _rank_remap(modewise_phase_randomise(x, rng), sorted_x)
    norm = float(np.sum(target ** 2))
    prev = np.inf
    for _ in range(max_iter):
        Z = np.fft.rfft(U.T @ (y - mean), axis=1)
        # impose every mode's amplitude, keep its CURRENT phase: modes stay independent
        Z = target * np.exp(1j * np.angle(Z))
        y = _rank_remap(U @ np.fft.irfft(Z, n=n, axis=1) + mean, sorted_x)
        misfit = float(np.sum((np.abs(np.fft.rfft(U.T @ (y - mean), axis=1)) - target) ** 2)) / norm
        if abs(prev - misfit) < tol * max(misfit, 1e-30):
            break
        prev = misfit
    return y


def circular_shift(x: np.ndarray, rng, min_shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent circular shift per channel, each by at least `min_shift` samples."""
    n = x.shape[1]
    if not 0 < min_shift < n // 2:
        raise ValueError(f"min_shift must be in (0, {n // 2}), got {min_shift}")
    shifts = rng.integers(min_shift, n - min_shift + 1, size=x.shape[0])
    return np.stack([np.roll(x[c], s) for c, s in enumerate(shifts)]), shifts
