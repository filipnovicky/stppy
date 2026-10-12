import numpy as np

from spectral_tools import (
    SFREQ,
    aperiodic_fit,
    alpha_features,
    extract_window,
    peak_frequency,
    sample_null_centers,
    sign_flip_test,
    unit_means,
    window_psd,
)


def _spectrum(freqs: np.ndarray, peak_db: float, peak_hz: float = 10.0) -> np.ndarray:
    """log10 power: 1/f^1.5 background plus a Gaussian bump of ``peak_db`` at ``peak_hz``."""
    return -1.5 * np.log10(freqs) + peak_db / 10 * np.exp(-((freqs - peak_hz) ** 2) / 2)


def test_extract_window_centred_and_off_edge():
    x = np.arange(1000)[None, :]
    assert extract_window(x, 2.0, 250)[0, 0] == 375
    assert extract_window(x, 0.1, 250) is None
    assert extract_window(x, 3.9, 250) is None


def test_psd_peak_at_sinusoid():
    t = np.arange(256) / SFREQ
    f, p = window_psd(np.sin(2 * np.pi * 10.5 * t)[None, :], nfft=1024)
    assert abs(f[np.argmax(p[0])] - 10.5) < 0.3


def test_null_centers_avoid_busy_intervals_and_edges():
    busy = [(4.0, 4.1), (6.0, 6.2)]
    c = sample_null_centers(2500, busy, 250, 200, np.random.default_rng(0), guard_s=0.5)
    assert len(c) == 200
    assert np.all((c >= 0.5) & (c <= 9.5))
    for start, end in busy:
        assert not np.any((c > start - 0.5) & (c < end + 0.5))


def test_aperiodic_fit_ignores_the_excluded_peak():
    f = np.linspace(1, 40, 200)
    fit = aperiodic_fit(f, _spectrum(f, peak_db=5))
    ok = ~np.isnan(fit)
    assert np.allclose(fit[ok], -1.5 * np.log10(f[ok]), atol=0.01)


def test_peak_on_band_edge_is_nan():
    f = np.linspace(1, 40, 200)
    assert np.isnan(peak_frequency(f, -f))


def test_alpha_features_height_and_peak():
    f = np.linspace(1, 40, 400)
    height, peak = alpha_features(f, _spectrum(f, peak_db=12, peak_hz=9.5))
    assert abs(height - 12) < 0.5
    assert abs(peak - 9.5) < 0.2


def test_alpha_features_rejects_peak_below_threshold():
    f = np.linspace(1, 40, 400)
    height, peak = alpha_features(f, _spectrum(f, peak_db=0.5))
    assert height < 1.0
    assert np.isnan(peak)


def test_unit_means_drops_small_units():
    values = np.arange(6.0)[:, None]
    units = np.array(["a", "a", "a", "b", "b", "c"])
    means, ids = unit_means(values, units, np.ones(6, bool), min_rows=2)
    assert list(ids) == ["a", "b"]
    assert np.allclose(means[:, 0], [1.0, 3.5])


def test_sign_flip_test_detects_effect_and_keeps_null():
    rng = np.random.default_rng(1)
    assert sign_flip_test(rng.normal(size=40), 2000, rng) > 0.01
    assert sign_flip_test(rng.normal(1.0, 1.0, size=40), 2000, rng) < 0.001
