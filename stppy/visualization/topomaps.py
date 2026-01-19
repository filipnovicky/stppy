"""
Topomap visualization utilities using MNE.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.channels import make_dig_montage
import json
from typing import Optional, List


def plot_avalanche_topomaps(
    data: List[np.ndarray],
    electrode_positions_file: str,
    indices: Optional[List[int]] = None,
    clusters: Optional[np.ndarray] = None,
    cluster_id: Optional[int] = None,
    n_samples: int = 3,
    random_seed: Optional[int] = None,
    ncols: int = 10,
    size: float = 2.5,
    vlim: Optional[tuple] = None,
    cmap: str = 'RdBu_r',
    show: bool = True
) -> plt.Figure:
    """
    Plot avalanche topomaps using MNE's visualization.

    Parameters
    ----------
    data : list of arrays
        List of avalanche data arrays
    electrode_positions_file : str
        Path to electrode positions JSON file
    indices : list of int, optional
        Specific avalanche indices to plot (manual selection).
        If provided, clusters/cluster_id are ignored.
    clusters : np.ndarray, optional
        Cluster assignments for each avalanche (from clustering).
        If provided, cluster_id must also be specified.
    cluster_id : int, optional
        Which cluster to sample from (0, 1, 2, ... or -1 for noise).
        Required if clusters is provided. Use clusters=None to sample from all data.
    n_samples : int
        Number of avalanches to plot when using cluster sampling (default: 3)
    random_seed : int, optional
        Random seed for reproducible sampling
    ncols : int
        Number of columns for topomap layout
    size : float
        Size of each topomap
    vlim : tuple, optional
        Color scale limits (vmin, vmax)
    cmap : str
        Colormap name
    show : bool
        Whether to show the plot

    Returns
    -------
    fig : matplotlib Figure or list of Figures
        The created figure(s)

    Examples
    --------
    >>> # Sample from ALL avalanches (no clustering filter)
    >>> fig = plot_avalanche_topomaps(
    ...     avalanches, 'positions.json', n_samples=10
    ... )
    >>>
    >>> # Sample 5 avalanches from cluster 0
    >>> fig = plot_avalanche_topomaps(
    ...     avalanches, 'positions.json',
    ...     clusters=clusters, cluster_id=0, n_samples=5
    ... )
    >>>
    >>> # Sample from noise/discarded avalanches
    >>> fig = plot_avalanche_topomaps(
    ...     avalanches, 'positions.json',
    ...     clusters=clusters, cluster_id=-1, n_samples=10
    ... )
    >>>
    >>> # Manual indices (backwards compatible)
    >>> fig = plot_avalanche_topomaps(avalanches, 'positions.json', indices=[0, 5, 10])
    """

    # Load electrode positions
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())

    # Create custom montage
    ch_pos = {}
    for ch in channels:
        pos = electrode_data[ch]
        ch_pos[ch] = [pos['x'], pos['y'], pos['z']]

    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # Determine which indices to plot
    if indices is not None:
        # Manual indices provided - use them directly
        valid_indices = [idx for idx in indices if 0 <= idx < len(data)]
    elif clusters is not None:
        # Cluster-based sampling - only look at avalanches within specified cluster
        if cluster_id is None:
            raise ValueError(
                "When clusters is provided, cluster_id must be specified. "
                "Use clusters=None to sample from all avalanches."
            )

        cluster_mask = clusters == cluster_id
        available_indices = np.where(cluster_mask)[0]
        cluster_label = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise/unassigned"

        if len(available_indices) == 0:
            print(f"No avalanches found for {cluster_label}")
            return None

        n_to_sample = min(n_samples, len(available_indices))

        if random_seed is not None:
            np.random.seed(random_seed)

        valid_indices = np.random.choice(available_indices, size=n_to_sample, replace=False).tolist()
        print(f"Plotting {n_to_sample} avalanches from {cluster_label} "
              f"(out of {len(available_indices)} available)")
    else:
        # No clusters - sample randomly from ALL avalanches
        n_to_sample = min(n_samples, len(data))

        if random_seed is not None:
            np.random.seed(random_seed)

        valid_indices = np.random.choice(len(data), size=n_to_sample, replace=False).tolist()
        print(f"Plotting {n_to_sample} avalanches from all data "
              f"(out of {len(data)} available)")

    if not valid_indices:
        print("No valid indices to plot")
        return None

    figures = []

    for avalanche_idx in valid_indices:
        avalanche = data[avalanche_idx]
        array_data = np.array(avalanche)
        n_electrodes, n_timepoints = array_data.shape

        # Create MNE info
        info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')

        # Create evoked object
        evoked = mne.EvokedArray(array_data, info)
        evoked.set_montage(montage)

        # Adaptive vlim if not provided
        if vlim is None:
            data_max = np.max(np.abs(array_data))
            vlim_use = (-data_max, data_max) if data_max > 0 else (-1, 1)
        else:
            vlim_use = vlim

        # Create time points
        times = np.arange(n_timepoints) / 250.0

        # Plot topomaps
        fig = evoked.plot_topomap(
            times=times,
            show=show,
            ncols=ncols,
            nrows='auto',
            size=size,
            vlim=vlim_use,
            units='Activity',
            image_interp='linear',
            scalings=1,
            cmap=cmap,
            res=256,
            extrapolate='head',
            outlines='head',
            contours=0,
            sensors=True,
            sphere='auto'
        )

        fig.suptitle(f'Avalanche {avalanche_idx}', fontsize=16)
        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures


def plot_cluster_average(
    average: np.ndarray,
    electrode_positions_file: str,
    cluster_id: Optional[int] = None,
    ncols: int = 6,
    size: float = 1.5,
    vlim: Optional[tuple] = None,
    cmap: str = 'RdBu_r',
    show: bool = True
) -> plt.Figure:
    """
    Plot cluster consensus average as topomaps.

    Parameters
    ----------
    average : np.ndarray
        Consensus average array, shape (n_electrodes, n_timepoints)
    electrode_positions_file : str
        Path to electrode positions JSON file
    cluster_id : int, optional
        Cluster ID for title
    ncols : int
        Number of columns
    size : float
        Size of each topomap
    vlim : tuple, optional
        Color scale limits
    cmap : str
        Colormap name
    show : bool
        Whether to show the plot

    Returns
    -------
    fig : matplotlib Figure
    """

    # Load electrode positions
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())

    # Create custom montage
    ch_pos = {}
    for ch in channels:
        pos = electrode_data[ch]
        ch_pos[ch] = [pos['x'], pos['y'], pos['z']]

    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    n_electrodes, n_timepoints = average.shape

    # Create MNE info
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')

    # Create evoked object
    evoked = mne.EvokedArray(average, info)
    evoked.set_montage(montage)

    # Adaptive vlim
    if vlim is None:
        data_max = np.max(np.abs(average))
        vlim = (-data_max, data_max) if data_max > 0 else (-1, 1)

    # Create time points
    times = np.arange(n_timepoints) / 250.0

    # Plot
    fig = evoked.plot_topomap(
        times=times,
        show=show,
        ncols=ncols,
        nrows='auto',
        size=size,
        vlim=vlim,
        units='',
        image_interp='linear',
        scalings=1,
        cmap=cmap,
        res=256,
        extrapolate='head',
        outlines='head',
        contours=0,
        sensors=False,
        sphere='auto'
    )

    if cluster_id is not None:
        fig.suptitle(f'Cluster {cluster_id} Consensus Average', fontsize=14)

    return fig
