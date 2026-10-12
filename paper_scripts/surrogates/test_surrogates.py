"""Surrogates for the reviewer null models (response S1).

Multivariate IAAFT must keep, per channel, the exact amplitude distribution and (to within
the iteration tolerance) the power spectrum, and across channels the cross-spectral phase
relations -- i.e. the full linear spatiotemporal covariance -- while destroying everything
else. Circular shifts keep each channel exactly and destroy cross-channel timing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from surrogates import circular_shift, multivariate_iaaft, phase_randomise   # noqa: E402

pytestmark = pytest.mark.unit


def coupled_signal(n_ch: int = 6, n: int = 8192, seed: int = 0) -> np.ndarray:
    """Heavy-tailed, lag-coupled channels: a lagged copy chain plus Laplace noise."""
    rng = np.random.default_rng(seed)
    base = np.convolve(rng.laplace(size=n + 64), np.hanning(16), "same")[:n + 64]
    x = np.stack([base[32 - 3 * c:32 - 3 * c + n] + 0.3 * rng.laplace(size=n)
                  for c in range(n_ch)])
    return x


def power(x: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.rfft(x - x.mean(1, keepdims=True), axis=1)) ** 2


def smooth(p: np.ndarray, w: int = 64) -> np.ndarray:
    k = np.ones(w) / w
    return np.stack([np.convolve(r, k, "valid") for r in p])


def xcorr(x: np.ndarray, i: int, j: int, max_lag: int = 12) -> np.ndarray:
    a = (x[i] - x[i].mean()) / x[i].std()
    b = (x[j] - x[j].mean()) / x[j].std()
    return np.array([np.mean(a[max(0, -L):len(a) - max(0, L)] *
                             b[max(0, L):len(b) - max(0, -L)])
                     for L in range(-max_lag, max_lag + 1)])


def test_phase_randomisation_keeps_cross_spectrum_exactly():
    x = coupled_signal()
    y = phase_randomise(x, np.random.default_rng(1))
    X, Y = np.fft.rfft(x, axis=1), np.fft.rfft(y, axis=1)
    assert np.allclose(np.abs(X), np.abs(Y), rtol=1e-8, atol=1e-8)
    cs_x = X[0] * np.conj(X[3])
    cs_y = Y[0] * np.conj(Y[3])
    assert np.allclose(cs_x, cs_y, rtol=1e-6, atol=1e-6)
    assert not np.allclose(x, y)


def test_iaaft_keeps_amplitude_distribution_exactly():
    x = coupled_signal()
    y = multivariate_iaaft(x, np.random.default_rng(2))
    assert np.array_equal(np.sort(x, axis=1), np.sort(y, axis=1))


def test_iaaft_keeps_power_spectrum():
    x = coupled_signal()
    y = multivariate_iaaft(x, np.random.default_rng(3))
    px, py = smooth(power(x)), smooth(power(y))
    assert np.median(np.abs(py / px - 1)) < 0.05


def test_iaaft_keeps_cross_correlation_and_lag():
    x = coupled_signal()
    y = multivariate_iaaft(x, np.random.default_rng(4))
    cx, cy = xcorr(x, 0, 3), xcorr(y, 0, 3)
    assert np.argmax(cx) == np.argmax(cy)                 # the 9-sample lag survives
    assert np.abs(cx - cy).max() < 0.05


def test_iaaft_destroys_the_time_course():
    x = coupled_signal()
    y = multivariate_iaaft(x, np.random.default_rng(5))
    r = np.corrcoef(x[0], y[0])[0, 1]
    assert abs(r) < 0.1


def test_iaaft_is_reproducible_from_the_seed():
    x = coupled_signal()
    a = multivariate_iaaft(x, np.random.default_rng(6))
    b = multivariate_iaaft(x, np.random.default_rng(6))
    assert np.array_equal(a, b)


def test_circular_shift_keeps_each_channel_and_breaks_lags():
    x = coupled_signal()
    y, shifts = circular_shift(x, np.random.default_rng(7), min_shift=500)
    for c in range(len(x)):
        assert np.array_equal(np.roll(x[c], shifts[c]), y[c])
    assert np.all((shifts >= 500) & (shifts <= x.shape[1] - 500))
    assert np.abs(xcorr(y, 0, 3)).max() < 0.1


def test_odd_length_is_supported():
    x = coupled_signal(n=4097)
    y = multivariate_iaaft(x, np.random.default_rng(8))
    assert y.shape == x.shape
    assert np.array_equal(np.sort(x, axis=1), np.sort(y, axis=1))


def test_modewise_keeps_zero_lag_covariance():
    from surrogates import modewise_phase_randomise
    x = coupled_signal(n=16384)
    y = modewise_phase_randomise(x, np.random.default_rng(9))
    Cx, Cy = np.cov(x), np.cov(y)
    assert np.abs(Cx - Cy).max() / np.abs(Cx).max() < 0.1


def test_modewise_keeps_mode_spectra_exactly():
    from surrogates import modewise_phase_randomise, spatial_modes
    x = coupled_signal()
    U = spatial_modes(x)
    y = modewise_phase_randomise(x, np.random.default_rng(10))
    px = np.abs(np.fft.rfft(U.T @ (x - x.mean(1, keepdims=True)), axis=1))
    py = np.abs(np.fft.rfft(U.T @ (y - y.mean(1, keepdims=True)), axis=1))
    assert np.allclose(px, py, rtol=1e-6, atol=1e-6 * px.max())


def test_modewise_destroys_the_lagged_coupling():
    """The source has channel 3 lagging channel 0 by 9 samples; mode-wise randomisation must
    leave a lag-symmetric cross-correlation (no travelling), unlike the multivariate FT."""
    from surrogates import modewise_phase_randomise
    x = coupled_signal(n=16384)
    ft = phase_randomise(x, np.random.default_rng(11))
    mw = modewise_phase_randomise(x, np.random.default_rng(11))
    asym = lambda c: np.abs(c - c[::-1]).max()
    assert asym(xcorr(ft, 0, 3)) > 0.3
    assert asym(xcorr(mw, 0, 3)) < 0.1


def lag_asymmetry(x: np.ndarray, max_lag: int = 12) -> float:
    """Directed part of the lagged covariance: max over lags of ||G(tau) - G(tau)^T|| / ||G(0)||."""
    xc = x - x.mean(1, keepdims=True)
    n = xc.shape[1]
    g0 = xc @ xc.T / n
    return max(np.linalg.norm(xc[:, :-L] @ xc[:, L:].T / n
                              - (xc[:, :-L] @ xc[:, L:].T / n).T) / np.linalg.norm(g0)
               for L in range(1, max_lag + 1))


def test_modewise_iaaft_keeps_amplitude_distribution_exactly():
    from surrogates import modewise_iaaft
    x = coupled_signal()
    y = modewise_iaaft(x, np.random.default_rng(12))
    assert np.array_equal(np.sort(x, axis=1), np.sort(y, axis=1))


def test_modewise_iaaft_keeps_zero_lag_covariance():
    from surrogates import modewise_iaaft
    x = coupled_signal(n=16384)
    y = modewise_iaaft(x, np.random.default_rng(13))
    Cx, Cy = np.corrcoef(x), np.corrcoef(y)
    assert np.abs(Cx - Cy).max() < 0.1


def test_modewise_iaaft_removes_directed_lags():
    """The source is a lagged copy chain (strongly directed); IAAFT must keep that direction,
    mode-wise IAAFT must remove it."""
    from surrogates import modewise_iaaft
    x = coupled_signal(n=16384)
    a_x = lag_asymmetry(x)
    a_iaaft = lag_asymmetry(multivariate_iaaft(x, np.random.default_rng(14)))
    a_mw = lag_asymmetry(modewise_iaaft(x, np.random.default_rng(14)))
    assert a_iaaft > 0.5 * a_x
    assert a_mw < 0.15 * a_x
