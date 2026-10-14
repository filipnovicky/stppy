"""All clusters in one figure: consensus topomaps + per-polarity movement arrows."""

from typing import Dict, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords

from ..jacobian import classify_propagation, pole_arrows
from ._helpers import _load_montage

POS_COLOR = '#b40000'
NEG_COLOR = '#0000b4'
CLASS_COLOR = {'travelling': '#1a7f37', 'standing': '0.35'}


def topomap_coords(electrode_positions_file: str, n_electrodes: int, sfreq: float = 250.0):
    """MNE Info and the 2D topomap coordinates that plot_topomap(sphere='auto') uses."""
    channels, montage = _load_montage(electrode_positions_file)
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    xy = _find_topomap_coords(info, picks='eeg', sphere='auto')[:, :2]
    return info, xy


def _draw_pole(ax, arrow, t, color, scale, standing_cut, arrow_width) -> None:
    anchor = arrow.centroids[t]
    if np.isnan(anchor[0]):
        return
    if arrow.magnitude < standing_cut:
        # standing pole: no net movement, dot only
        ax.scatter(*anchor, s=18, c=color, edgecolors='white', linewidths=0.7, zorder=5)
        return
    # Centred on the pole centroid; minshaft=1.5 shrinks short arrows as a whole
    # (head included) instead of putting a full-size head on a stub.
    ax.quiver(anchor[0], anchor[1], arrow.vector[0], arrow.vector[1], color=color,
              scale=scale, scale_units='width', width=arrow_width, headwidth=3.2,
              headlength=3.6, headaxislength=3.2, minshaft=1.5, edgecolor='white',
              linewidth=0.4, pivot='middle', clip_on=False, zorder=6)


def plot_clusters_with_arrows(
    cluster_averages: Dict[int, np.ndarray],
    electrode_positions_file: str,
    cluster_sizes: Optional[Dict[int, int]] = None,
    sfreq: float = 250.0,
    title: Optional[str] = 'Discovered Clusters',
    vlim: tuple = (-1.0, 1.0),
    arrow_frac: float = 0.25,
    standing_frac: float = 0.25,
    arrow_width: float = 0.055,
    cell: float = 1.1,
    title_fontsize: float = 24,
    label_fontsize: float = 15,
    time_fontsize: float = 14,
    classify: bool = True,
    classify_frac: float = 0.70,
    reference: Optional[float] = None,
) -> plt.Figure:
    """One row per cluster, one column per frame, with a movement arrow per pole.

    Topomaps use the same settings as :func:`plot_cluster_average`. Each pole gets
    the time-averaged propagation arrow from :func:`stppy.jacobian.pole_arrows`,
    drawn identically on every frame at that frame's pole centroid (red = positive,
    blue = negative). All clusters share one length scale.

    Parameters
    ----------
    cluster_averages : dict
        cluster_id -> consensus average (n_electrodes, n_timepoints).
    electrode_positions_file : str
        Electrode positions JSON; row order must match the averages.
    cluster_sizes : dict, optional
        cluster_id -> number of members, shown under the row label.
    arrow_frac : float
        The longest arrow in the figure spans this fraction of a panel width.
    standing_frac : float
        Poles whose arrow is shorter than this fraction of the longest arrow are
        drawn as a dot only (no net movement).
    classify : bool
        Label each row 'travelling' / 'standing' with
        :func:`stppy.jacobian.classify_propagation`.
    classify_frac : float
        Travelling iff the stronger pole's net arrow reaches this fraction of the
        longest arrow in the figure. Separate from ``standing_frac`` (which only
        decides dot vs arrow), so weak but visible movement can still be standing.
    reference : float, optional
        Arrow length that replaces this figure's own longest arrow for the arrow scale,
        the dot cut and the travelling criterion. Pass another repertoire's longest arrow
        (e.g. the real data's, when drawing surrogate templates) so that two figures share
        one scale; by default each figure is scaled and classified on its own.

    Returns
    -------
    fig : matplotlib Figure
    """
    if not cluster_averages:
        raise ValueError('cluster_averages is empty')
    cids = sorted(cluster_averages)
    n_electrodes = np.asarray(cluster_averages[cids[0]]).shape[0]
    info, xy = topomap_coords(electrode_positions_file, n_electrodes, sfreq)

    arrows = {c: pole_arrows(cluster_averages[c], xy) for c in cids}
    longest = (reference if reference is not None
               else max(a.magnitude for c in cids for a in arrows[c].values()))
    longest = longest if longest > 0 else 1.0
    scale = longest / arrow_frac   # scale_units='width': magnitude m spans m/scale
    standing_cut = standing_frac * longest

    max_t = max(np.asarray(cluster_averages[c]).shape[1] for c in cids)
    fig, axes = plt.subplots(len(cids), max_t, figsize=(max_t * cell, len(cids) * cell),
                             squeeze=False)
    for ri, c in enumerate(cids):
        avg = np.asarray(cluster_averages[c], dtype=float)
        for t in range(max_t):
            ax = axes[ri, t]
            if t >= avg.shape[1]:
                ax.set_axis_off()
                continue
            mne.viz.plot_topomap(
                avg[:, t], info, axes=ax, show=False, vlim=vlim, cmap='RdBu_r',
                image_interp='linear', res=256, extrapolate='head', outlines='head',
                contours=0, sensors=False, sphere='auto')
            _draw_pole(ax, arrows[c]['pos'], t, POS_COLOR, scale, standing_cut, arrow_width)
            _draw_pole(ax, arrows[c]['neg'], t, NEG_COLOR, scale, standing_cut, arrow_width)
            if ri == 0:
                ax.set_title(f'{t / sfreq * 1000:.0f} ms', fontsize=time_fontsize)
        label = f'Cluster {c}'
        if cluster_sizes is not None and c in cluster_sizes:
            label += f'\nn={cluster_sizes[c]}'
        axes[ri, 0].text(-0.08, 0.5 if not classify else 0.62, label,
                         transform=axes[ri, 0].transAxes, ha='right', va='center',
                         fontsize=label_fontsize)
        if classify:
            kind = classify_propagation(arrows[c], longest, classify_frac)
            axes[ri, 0].text(-0.08, 0.12, kind, transform=axes[ri, 0].transAxes,
                             ha='right', va='center', fontsize=label_fontsize,
                             fontweight='bold', color=CLASS_COLOR[kind])
    if title:
        fig.suptitle(title, fontsize=title_fontsize, fontweight='bold', y=1.025)
    plt.subplots_adjust(left=0.08, right=0.995, top=0.97, bottom=0.005,
                        hspace=0.08, wspace=0.04)
    return fig
