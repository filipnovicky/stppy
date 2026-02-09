"""
Topomap visualization utilities using MNE.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.channels import make_dig_montage
import json
from typing import Optional, List, Dict, Union


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
            sphere='auto',
            time_unit='ms'
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
        sphere='auto',
        time_unit='ms'
    )

    if cluster_id is not None:
        fig.suptitle(f'Cluster {cluster_id} Consensus Average', fontsize=14)

    return fig


def plot_burst_topomaps(
    bursts_by_cluster: Dict,
    datasets: List[np.ndarray],
    dataset_names: List[str],
    electrode_positions_file: str,
    cluster_id: Optional[int] = None,
    focus: Optional[Union[str, List[str]]] = None,
    polarity: Optional[str] = None,
    sort_by_length: bool = False,
    n_bursts: int = 3,
    random_seed: Optional[int] = None,
    sfreq: float = 250.0,
    ncols: int = 10,
    size: float = 1.5,
    vlim: Optional[tuple] = None,
    cmap: str = 'RdBu_r',
    show: bool = True
) -> Union[plt.Figure, List[plt.Figure], None]:
    """
    Plot burst topomaps from pre-loaded EEG datasets.

    Extracts burst time windows from EEG data and plots each
    timepoint as a scalp topomap. Values are displayed in microvolts.

    Parameters
    ----------
    bursts_by_cluster : dict
        {cluster_id: [burst_dicts, ...]} from burst detection.
        Each burst dict must have 'dataset', 'start_time', 'end_time'.
    datasets : list of np.ndarray
        Pre-loaded EEG arrays, each shape (n_electrodes, n_samples), in Volts.
    dataset_names : list of str
        Basenames corresponding to each array in datasets
        (e.g. 'ctrl_sub001_meditation.set').
    electrode_positions_file : str
        Path to electrode positions JSON file.
    cluster_id : int, optional
        Filter bursts to this cluster. None plots from all clusters.
    focus : str or list of str, optional
        Filter bursts to specific dataset(s). A single string or a list
        of dataset filenames. None allows all datasets.
    polarity : str, optional
        Filter bursts by polarity ('stable', 'switching', 'mixed', etc.).
        None allows all polarities.
    sort_by_length : bool
        If True, select the longest bursts instead of random sampling
        (default: False).
    n_bursts : int
        Number of bursts to plot (default: 3).
    random_seed : int, optional
        Random seed for reproducible burst selection.
    sfreq : float
        Sampling frequency in Hz (default: 250.0).
    ncols : int
        Number of topomap columns per row (default: 10).
    size : float
        Size of each topomap (default: 1.5).
    vlim : tuple, optional
        Color scale limits in microvolts. None uses adaptive per-burst scaling.
    cmap : str
        Colormap name (default: 'RdBu_r').
    show : bool
        Whether to display the plot (default: True).

    Returns
    -------
    fig : Figure, list of Figures, or None
        The created figure(s). None if no matching bursts found.

    Examples
    --------
    >>> # Longest stable bursts from cluster 5
    >>> fig = plot_burst_topomaps(
    ...     bursts, datasets, dataset_names,
    ...     'electrode_positions.json',
    ...     cluster_id=5, polarity='stable', sort_by_length=True
    ... )
    >>> # Random switching bursts from a specific dataset
    >>> fig = plot_burst_topomaps(
    ...     bursts, datasets, dataset_names,
    ...     'electrode_positions.json',
    ...     focus='ctrl_sub001_meditation.set', polarity='switching'
    ... )
    """
    # Handle length mismatch
    if len(dataset_names) != len(datasets):
        print(f"dataset_names ({len(dataset_names)}) != datasets ({len(datasets)}). "
              f"Pass the full dataset_names list and use focus='filename.set' to filter.")
        return None

    # Build name -> index lookup for datasets
    ds_name_to_idx = {name: i for i, name in enumerate(dataset_names)}

    # Normalize focus to a set for fast lookup
    if focus is not None:
        if isinstance(focus, str):
            focus_set = {focus}
        else:
            focus_set = set(focus)
        missing = focus_set - set(ds_name_to_idx)
        if missing:
            print(f"Not found in dataset_names: {missing}")
            return None
        print(f"Filtering to {len(focus_set)} dataset(s): {', '.join(sorted(focus_set))}")
    else:
        focus_set = None

    # Collect matching bursts
    candidates = []
    cluster_ids = [cluster_id] if cluster_id is not None else sorted(bursts_by_cluster.keys())

    for cid in cluster_ids:
        if cid not in bursts_by_cluster:
            continue
        for burst in bursts_by_cluster[cid]:
            if focus_set is not None and burst['dataset'] not in focus_set:
                continue
            if polarity is not None and burst.get('polarity') != polarity:
                continue
            candidates.append((cid, burst))

    if not candidates:
        filters = f"cluster_id={cluster_id}, focus={focus}, polarity={polarity}"
        print(f"No bursts found matching {filters}")
        return None

    # Select bursts
    n_to_plot = min(n_bursts, len(candidates))

    if sort_by_length:
        candidates.sort(key=lambda x: x[1].get('length', 0), reverse=True)
        selected = candidates[:n_to_plot]
        print(f"Plotting {n_to_plot} longest burst(s) from {len(candidates)} candidates")
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
        selected_idx = np.random.choice(len(candidates), size=n_to_plot, replace=False)
        selected = [candidates[i] for i in selected_idx]
        print(f"Plotting {n_to_plot} burst(s) from {len(candidates)} candidates")

    # Load electrode positions and create montage
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())
    ch_pos = {ch: [electrode_data[ch]['x'], electrode_data[ch]['y'], electrode_data[ch]['z']]
              for ch in channels}
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    figures = []

    for cid, burst in selected:
        ds_name = burst['dataset']

        if ds_name not in ds_name_to_idx:
            print(f"  Skipping burst: {ds_name} not found in dataset_names")
            continue

        data = datasets[ds_name_to_idx[ds_name]] * 1e6  # V to µV

        # Extract burst window
        start_idx = int(burst['start_time'] * sfreq)
        end_idx = int(burst['end_time'] * sfreq)
        start_idx = max(0, start_idx)
        end_idx = min(data.shape[1], end_idx)
        burst_data = data[:, start_idx:end_idx]
        n_electrodes, n_timepoints = burst_data.shape

        if n_timepoints == 0:
            print(f"  Skipping empty burst: {ds_name} {burst['start_time']:.3f}-{burst['end_time']:.3f}s")
            continue

        # Create MNE evoked object
        info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=sfreq, ch_types='eeg')
        evoked = mne.EvokedArray(burst_data, info)
        evoked.set_montage(montage)

        # Determine vlim
        if vlim is None:
            data_max = np.max(np.abs(burst_data))
            vlim_use = (-data_max, data_max) if data_max > 0 else (-1, 1)
        else:
            vlim_use = vlim

        # Time points relative to burst start
        times = np.arange(n_timepoints) / sfreq

        fig = evoked.plot_topomap(
            times=times,
            show=False,
            ncols=ncols,
            nrows='auto',
            size=size,
            vlim=vlim_use,
            units='\u00b5V',
            image_interp='linear',
            scalings=1,
            cmap=cmap,
            res=256,
            extrapolate='head',
            outlines='head',
            contours=0,
            sensors=False,
            sphere='auto',
            time_unit='ms'
        )

        polarity = burst.get('polarity', '?')
        length = burst.get('length', '?')
        title = (f'Cluster {cid} | {ds_name}\n'
                 f'{burst["start_time"]:.3f}\u2013{burst["end_time"]:.3f}s | '
                 f'polarity: {polarity} | length: {length}')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        if show:
            plt.show()
        figures.append(fig)

    if not figures:
        print("No bursts could be plotted")
        return None

    return figures[0] if len(figures) == 1 else figures
