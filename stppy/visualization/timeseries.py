"""
Time series visualization utilities for avalanche analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Tuple, Optional, Dict
import matplotlib.cm as cm


def plot_avalanche_timeline(
    data: np.ndarray,
    avalanche_times: List[Tuple[int, int]],
    cluster_assignments: Optional[np.ndarray] = None,
    channel_idx: int = 0,
    sfreq: float = 250.0,
    title: str = "Avalanche Timeline",
    figsize: tuple = (20, 6),
    show_legend: bool = True
) -> plt.Figure:
    """
    Plot raw EEG data with avalanche periods highlighted.

    Parameters
    ----------
    data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    avalanche_times : list of tuples
        List of (start_idx, end_idx) for each avalanche
    cluster_assignments : np.ndarray, optional
        Cluster ID for each avalanche (for color coding)
    channel_idx : int
        Channel index to plot (default: 0)
    sfreq : float
        Sampling frequency in Hz (default: 250.0)
    title : str
        Plot title
    figsize : tuple
        Figure size (default: (20, 6))
    show_legend : bool
        Whether to show cluster legend (default: True)

    Returns
    -------
    fig : matplotlib Figure
        The created figure

    Examples
    --------
    >>> # Plot timeline for channel 0 with cluster colors
    >>> fig = plot_avalanche_timeline(
    ...     raw_data,
    ...     avalanche_times,
    ...     cluster_assignments,
    ...     channel_idx=0
    ... )
    >>>
    >>> # Plot without cluster coloring
    >>> fig = plot_avalanche_timeline(
    ...     raw_data,
    ...     avalanche_times,
    ...     channel_idx=5
    ... )
    """

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.shape[0] > data.shape[1]:
        # Assume shape is (n_samples, n_channels), transpose
        data = data.T

    n_channels, n_samples = data.shape

    if channel_idx >= n_channels:
        raise ValueError(f"channel_idx {channel_idx} out of bounds (n_channels={n_channels})")

    # Get channel data
    channel_data = data[channel_idx, :]

    # Create time axis in seconds
    time = np.arange(n_samples) / sfreq

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot raw data
    ax.plot(time, channel_data, 'k-', alpha=0.6, linewidth=0.5, label='Raw data')

    # Determine colors for avalanches
    if cluster_assignments is not None:
        unique_clusters = np.unique(cluster_assignments)
        # Filter out -1 (noise)
        unique_clusters = unique_clusters[unique_clusters != -1]

        # Create colormap
        if len(unique_clusters) > 0:
            cmap = cm.get_cmap('tab10' if len(unique_clusters) <= 10 else 'tab20')
            cluster_colors = {
                cluster_id: cmap(i / max(1, len(unique_clusters) - 1))
                for i, cluster_id in enumerate(unique_clusters)
            }
            # Noise color
            cluster_colors[-1] = (0.5, 0.5, 0.5, 0.3)
        else:
            cluster_colors = {}
    else:
        # No cluster info - use single color
        cluster_colors = None

    # Plot avalanche periods
    avalanche_handles = {}

    for idx, (start, end) in enumerate(avalanche_times):
        if idx >= len(avalanche_times):
            break

        # Get cluster color
        if cluster_assignments is not None:
            cluster_id = cluster_assignments[idx]
            color = cluster_colors.get(cluster_id, (0.5, 0.5, 0.5, 0.3))
            label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
        else:
            color = (1.0, 0.7, 0.7, 0.4)
            label = 'Avalanche'

        # Convert to time
        start_time = start / sfreq
        end_time = end / sfreq

        # Add shaded region
        if label not in avalanche_handles:
            rect = ax.axvspan(start_time, end_time, alpha=0.3, color=color, label=label)
            avalanche_handles[label] = rect
        else:
            ax.axvspan(start_time, end_time, alpha=0.3, color=color)

    # Labels and formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'{title} - Channel {channel_idx}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if show_legend and avalanche_handles:
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    return fig


def plot_multi_channel_timeline(
    data: np.ndarray,
    avalanche_times: List[Tuple[int, int]],
    cluster_assignments: Optional[np.ndarray] = None,
    channel_indices: Optional[List[int]] = None,
    sfreq: float = 250.0,
    title: str = "Multi-Channel Avalanche Timeline",
    figsize: tuple = (20, 12)
) -> plt.Figure:
    """
    Plot multiple channels with avalanche periods highlighted.

    Parameters
    ----------
    data : np.ndarray
        EEG data, shape (n_channels, n_samples)
    avalanche_times : list of tuples
        List of (start_idx, end_idx) for each avalanche
    cluster_assignments : np.ndarray, optional
        Cluster ID for each avalanche
    channel_indices : list of int, optional
        Which channels to plot (default: first 4)
    sfreq : float
        Sampling frequency in Hz (default: 250.0)
    title : str
        Plot title
    figsize : tuple
        Figure size (default: (20, 12))

    Returns
    -------
    fig : matplotlib Figure
        The created figure
    """

    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.shape[0] > data.shape[1]:
        data = data.T

    n_channels, n_samples = data.shape

    if channel_indices is None:
        channel_indices = list(range(min(4, n_channels)))

    n_plots = len(channel_indices)

    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    if n_plots == 1:
        axes = [axes]

    time = np.arange(n_samples) / sfreq

    # Determine colors
    if cluster_assignments is not None:
        unique_clusters = np.unique(cluster_assignments)
        unique_clusters = unique_clusters[unique_clusters != -1]

        if len(unique_clusters) > 0:
            cmap = cm.get_cmap('tab10' if len(unique_clusters) <= 10 else 'tab20')
            cluster_colors = {
                cluster_id: cmap(i / max(1, len(unique_clusters) - 1))
                for i, cluster_id in enumerate(unique_clusters)
            }
            cluster_colors[-1] = (0.5, 0.5, 0.5, 0.3)
        else:
            cluster_colors = {}
    else:
        cluster_colors = None

    for plot_idx, ch_idx in enumerate(channel_indices):
        ax = axes[plot_idx]

        # Plot channel data
        channel_data = data[ch_idx, :]
        ax.plot(time, channel_data, 'k-', alpha=0.6, linewidth=0.5)

        # Plot avalanche periods
        for idx, (start, end) in enumerate(avalanche_times):
            if cluster_assignments is not None:
                cluster_id = cluster_assignments[idx]
                color = cluster_colors.get(cluster_id, (0.5, 0.5, 0.5, 0.3))
            else:
                color = (1.0, 0.7, 0.7, 0.4)

            start_time = start / sfreq
            end_time = end / sfreq
            ax.axvspan(start_time, end_time, alpha=0.3, color=color)

        ax.set_ylabel(f'Ch {ch_idx}', fontsize=10)
        ax.grid(True, alpha=0.3)

        if plot_idx == 0:
            ax.set_title(title, fontsize=14, fontweight='bold')

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)

    plt.tight_layout()

    return fig


def plot_cluster_correlations(
    cluster_averages: Dict[int, np.ndarray],
    method: str = 'pearson',
    figsize: tuple = (10, 8),
    cmap: str = 'RdBu_r',
    title: str = "Cluster Pattern Correlations"
) -> plt.Figure:
    """
    Plot correlation matrix between cluster average patterns.

    Parameters
    ----------
    cluster_averages : dict
        Dictionary mapping cluster_id -> average pattern array
    method : str
        Correlation method ('pearson' or 'spearman')
    figsize : tuple
        Figure size (default: (10, 8))
    cmap : str
        Colormap name (default: 'RdBu_r')
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib Figure
        The created figure

    Examples
    --------
    >>> # Compute correlation between cluster patterns
    >>> cluster_averages = {0: avg0, 1: avg1, 2: avg2}
    >>> fig = plot_cluster_correlations(cluster_averages)
    """
    from scipy.stats import pearsonr, spearmanr

    cluster_ids = sorted(cluster_averages.keys())
    n_clusters = len(cluster_ids)

    # Initialize correlation matrix
    corr_matrix = np.zeros((n_clusters, n_clusters))

    # Compute correlations
    for i, id_i in enumerate(cluster_ids):
        for j, id_j in enumerate(cluster_ids):
            avg_i = cluster_averages[id_i].flatten()
            avg_j = cluster_averages[id_j].flatten()

            if method == 'pearson':
                corr, _ = pearsonr(avg_i, avg_j)
            elif method == 'spearman':
                corr, _ = spearmanr(avg_i, avg_j)
            else:
                raise ValueError(f"Unknown method: {method}")

            corr_matrix[i, j] = corr

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels([f'C{id}' for id in cluster_ids])
    ax.set_yticklabels([f'C{id}' for id in cluster_ids])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)

    # Add correlation values
    for i in range(n_clusters):
        for j in range(n_clusters):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'{title} ({method.capitalize()})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)

    plt.tight_layout()

    return fig


def plot_avalanche_raster(
    avalanche_times: List[Tuple[int, int]],
    cluster_assignments: Optional[np.ndarray] = None,
    sfreq: float = 250.0,
    figsize: tuple = (20, 8),
    title: str = "Avalanche Raster Plot"
) -> plt.Figure:
    """
    Create a raster plot showing when avalanches occur.

    Parameters
    ----------
    avalanche_times : list of tuples
        List of (start_idx, end_idx) for each avalanche
    cluster_assignments : np.ndarray, optional
        Cluster ID for each avalanche (for color coding)
    sfreq : float
        Sampling frequency in Hz (default: 250.0)
    figsize : tuple
        Figure size (default: (20, 8))
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib Figure
        The created figure
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if cluster_assignments is not None:
        unique_clusters = np.unique(cluster_assignments)
        unique_clusters = unique_clusters[unique_clusters != -1]

        if len(unique_clusters) > 0:
            cmap = cm.get_cmap('tab10' if len(unique_clusters) <= 10 else 'tab20')
            cluster_colors = {
                cluster_id: cmap(i / max(1, len(unique_clusters) - 1))
                for i, cluster_id in enumerate(unique_clusters)
            }
            cluster_colors[-1] = (0.5, 0.5, 0.5)
        else:
            cluster_colors = {}
    else:
        cluster_colors = None

    # Plot each avalanche as a horizontal bar
    for idx, (start, end) in enumerate(avalanche_times):
        start_time = start / sfreq
        duration = (end - start) / sfreq

        if cluster_assignments is not None:
            cluster_id = cluster_assignments[idx]
            color = cluster_colors.get(cluster_id, (0.5, 0.5, 0.5))
        else:
            color = (0.3, 0.6, 0.9)

        ax.barh(idx, duration, left=start_time, height=0.8, color=color, alpha=0.7)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Avalanche Index', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    # Add cluster legend if applicable
    if cluster_assignments is not None and cluster_colors:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, alpha=0.7, label=f'Cluster {cid}')
            for cid, color in sorted(cluster_colors.items()) if cid != -1
        ]
        if -1 in cluster_colors:
            legend_elements.append(
                Patch(facecolor=cluster_colors[-1], alpha=0.7, label='Noise')
            )
        ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    return fig
