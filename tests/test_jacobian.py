"""Tests for stppy.jacobian: phase gradient and per-pole movement arrows."""
import numpy as np
import pytest

from stppy.jacobian import (
    build_gradient_operators,
    classify_propagation,
    compute_phase_gradient,
    pole_arrows,
    propagation_field,
)


@pytest.fixture
def grid_xy():
    rng = np.random.default_rng(0)
    g = np.stack(np.meshgrid(np.linspace(-1, 1, 8), np.linspace(-1, 1, 8)), -1).reshape(-1, 2)
    return g + rng.normal(0, 0.02, g.shape)     # irregular like a real montage


def _travelling_wave(xy, direction, k=2.0, f=10.0, sfreq=250.0, n=200):
    t = np.arange(n) / sfreq
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    return np.cos(2 * np.pi * f * t[None, :] - k * (xy @ d)[:, None])


def test_operators_exact_on_linear_field(grid_xy):
    Gx, Gy, S = build_gradient_operators(grid_xy)
    f = 3.0 * grid_xy[:, 0] - 2.0 * grid_xy[:, 1]
    np.testing.assert_allclose(Gx @ f, 3.0, atol=1e-8)
    np.testing.assert_allclose(Gy @ f, -2.0, atol=1e-8)
    np.testing.assert_allclose(S.sum(1), 1.0)


def test_operators_reject_bad_shape():
    with pytest.raises(ValueError):
        build_gradient_operators(np.zeros((5, 3)))


def test_raw_gradient_points_against_propagation(grid_xy):
    data = _travelling_wave(grid_xy, (1, 0))
    Gx, Gy, S = build_gradient_operators(grid_xy)
    dy, dx = compute_phase_gradient(data, Gx, Gy, S)
    mid = slice(50, 150)                        # away from Hilbert edge effects
    assert np.median(dx[:, mid]) < 0            # grad(phi) = -k


@pytest.mark.parametrize('direction', [(1, 0), (0, 1), (-1, 1)])
def test_propagation_field_recovers_direction(grid_xy, direction):
    u, v = propagation_field(_travelling_wave(grid_xy, direction), grid_xy)
    mid = slice(50, 150)
    est = np.array([np.median(u[:, mid]), np.median(v[:, mid])])
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    assert est @ d / np.linalg.norm(est) > 0.95


def test_propagation_field_rejects_mismatch(grid_xy):
    with pytest.raises(ValueError):
        propagation_field(np.zeros((3, 10)), grid_xy)


def test_pole_arrows_moving_wave_has_net_vector(grid_xy):
    arrows = pole_arrows(_travelling_wave(grid_xy, (0, 1), n=60), grid_xy)
    for a in arrows.values():
        assert a.vector[1] > 0 and a.magnitude > 0
        assert a.centroids.shape == (60, 2)


def test_pole_arrows_standing_dipole_is_zero(grid_xy):
    # Fixed dipole whose amplitude only waxes and wanes: no propagation.
    space = grid_xy[:, 1]
    data = space[:, None] * np.sin(np.linspace(0.1, np.pi - 0.1, 30))[None, :]
    arrows = pole_arrows(data, grid_xy)
    for a in arrows.values():
        assert a.magnitude < 1e-6


def test_pole_arrows_absent_pole(grid_xy):
    data = np.abs(_travelling_wave(grid_xy, (1, 0), n=40))   # positive only
    arrows = pole_arrows(data, grid_xy)
    assert np.all(np.isnan(arrows['neg'].centroids))
    assert arrows['neg'].magnitude == 0.0


def test_classify_propagation(grid_xy):
    moving = pole_arrows(_travelling_wave(grid_xy, (0, 1), n=60), grid_xy)
    space = grid_xy[:, 1]
    standing = pole_arrows(space[:, None] * np.sin(np.linspace(0.1, 3.0, 30))[None, :], grid_xy)
    ref = max(a.magnitude for a in moving.values())
    assert classify_propagation(moving, ref) == 'travelling'
    assert classify_propagation(standing, ref) == 'standing'
    assert classify_propagation(standing, 0.0) == 'standing'
