"""Hilbert phase gradient on an irregular electrode layout.

Ported from stppy 1.0.0 (``stppy.jacobian.hilbert.compute_phase_gradient``) and the
operator builder of complete_analysis_stppy100 (``compute_consensus_jacobian``), so
0.2.0 needs neither package at runtime.

Sign convention: Hilbert phase ``phi = wt - k.r`` gives ``grad(phi) = -k``. The raw
output of :func:`compute_phase_gradient` is therefore opposite to the propagation
direction; :func:`propagation_field` returns the negated (physical) field.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert

# Kernel widths are this multiple of the mean nearest-neighbour distance, so the same
# operators work for 61-, 64-, 83- or 272-channel layouts.
KERNEL_NN_MULTIPLE = 2.0
# Temporal smoothing (samples) before the Hilbert transform; tames FFT edge ringing on
# short consensi (T ~ 11 frames is typical).
TEMPORAL_SMOOTH_SIGMA = 1.0


def build_gradient_operators(
    xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-least-squares spatial gradient operators and a smoothing matrix.

    Parameters
    ----------
    xy : (n_elec, 2) array
        2D electrode coordinates (the frame the gradient is expressed in).

    Returns
    -------
    Gx, Gy : (n_elec, n_elec) arrays
        ``Gx @ f`` / ``Gy @ f`` estimate df/dx, df/dy at every electrode.
    S : (n_elec, n_elec) array
        Row-normalised Gaussian spatial smoothing matrix.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] < 3:
        raise ValueError(f"xy must be (n_elec >= 3, 2), got {xy.shape}")
    n = xy.shape[0]
    dx = xy[None, :, 0] - xy[:, None, 0]          # dx[i, j] = x_j - x_i
    dy = xy[None, :, 1] - xy[:, None, 1]
    dists = np.hypot(dx, dy)
    nn_dist = float(np.mean(np.partition(dists + np.eye(n) * 1e10, 1, axis=1)[:, 0]))
    sigma = KERNEL_NN_MULTIPLE * nn_dist

    S = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    S /= S.sum(axis=1, keepdims=True)

    Gx = np.zeros((n, n))
    Gy = np.zeros((n, n))
    for i in range(n):
        d2 = dists[i] ** 2
        d2[i] = np.inf
        w = np.exp(-d2 / (2 * sigma ** 2))
        D = np.stack([dx[i], dy[i]], axis=1)
        WD = w[:, None] * D
        coeffs = np.linalg.inv(D.T @ WD) @ WD.T
        Gx[i], Gy[i] = coeffs[0], coeffs[1]
        # The fit is on differences f_j - f_i: put the -f_i term on the diagonal.
        # (stppy 1.0.0 omitted it; harmless for the phase gradient, where it adds
        # (sum c)|z_i|^2, a real number that Im() discards, but wrong for any other f.)
        Gx[i, i] -= coeffs[0].sum()
        Gy[i, i] -= coeffs[1].sum()
    return Gx, Gy, S


def compute_phase_gradient(
    data: np.ndarray, Gx: np.ndarray, Gy: np.ndarray, S: np.ndarray, eps: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial gradient of the instantaneous (Hilbert) phase, no unwrapping.

    Uses ``grad(phi) = Im(grad(z) * conj(z)) / |z|^2`` on the spatially smoothed
    analytic signal ``z``.

    Returns
    -------
    phi_grad_y, phi_grad_x : (n_elec, T) arrays
        Raw ``grad(phi)`` in (dy, dx) order (stppy NPZ convention).
    """
    z = hilbert(S @ np.asarray(data, dtype=float), axis=1)
    z_conj = np.conj(z)
    z_mag_sq = np.abs(z) ** 2 + eps
    phi_grad_x = np.imag((Gx @ z) * z_conj) / z_mag_sq
    phi_grad_y = np.imag((Gy @ z) * z_conj) / z_mag_sq
    return phi_grad_y, phi_grad_x


def propagation_field(
    data: np.ndarray, xy: np.ndarray, temporal_sigma: float = TEMPORAL_SMOOTH_SIGMA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Physical propagation field ``-grad(phi)`` of an (n_elec, T) pattern.

    Parameters
    ----------
    data : (n_elec, T) array
        Spatiotemporal pattern (e.g. a cluster consensus average).
    xy : (n_elec, 2) array
        2D electrode coordinates; the returned vectors live in this frame.
    temporal_sigma : float
        Gaussian smoothing along time (samples) before the Hilbert transform.
        0 disables it.

    Returns
    -------
    u, v : (n_elec, T) arrays
        x and y components of the propagation direction at each electrode/frame.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(xy):
        raise ValueError(f"data must be (n_elec={len(xy)}, T), got {data.shape}")
    if temporal_sigma > 0:
        data = gaussian_filter1d(data, sigma=temporal_sigma, axis=1, mode="nearest")
    Gx, Gy, S = build_gradient_operators(xy)
    dy, dx = compute_phase_gradient(data, Gx, Gy, S)
    return -dx, -dy
