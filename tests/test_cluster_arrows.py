"""Smoke/integration test for stppy.visualization.plot_clusters_with_arrows."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import stppy  # noqa: E402

POS_FILE = str(Path(__file__).resolve().parents[1] / 'electrode_positions.json')


def test_figure_layout():
    rng = np.random.default_rng(1)
    averages = {0: rng.uniform(-1, 1, (64, 11)), 1: rng.uniform(-1, 1, (64, 12))}
    fig = stppy.visualization.plot_clusters_with_arrows(
        averages, POS_FILE, cluster_sizes={0: 30, 1: 40})
    axes = np.array(fig.axes).reshape(2, -1)   # colorbars are not added
    assert axes.shape == (2, 12)
    assert not axes[0, 11].axison                # shorter cluster padded with blanks
    assert fig._suptitle.get_text() == 'Discovered Clusters'
    plt.close(fig)


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        stppy.visualization.plot_clusters_with_arrows({}, POS_FILE)


def _labels(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts
            if t.get_text() in ('travelling', 'standing')]


def test_reference_sets_the_travelling_scale():
    """Without `reference` the cut is relative to this figure's own longest arrow; with it,
    two figures can be classified on one scale."""
    rng = np.random.default_rng(2)
    averages = {0: rng.uniform(-1, 1, (64, 11)), 1: rng.uniform(-1, 1, (64, 11))}
    own = stppy.visualization.plot_clusters_with_arrows(averages, POS_FILE)
    assert 'travelling' in _labels(own)          # the longest row always travels on its own
    plt.close(own)
    huge = stppy.visualization.plot_clusters_with_arrows(averages, POS_FILE, reference=1e6)
    assert _labels(huge) == ['standing', 'standing']
    plt.close(huge)
    tiny = stppy.visualization.plot_clusters_with_arrows(averages, POS_FILE, reference=1e-9)
    assert _labels(tiny) == ['travelling', 'travelling']
    plt.close(tiny)
