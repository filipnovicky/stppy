"""
Topomap visualization utilities using MNE.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
import json
from typing import Optional, List, Dict, Union

from ._helpers import _load_montage, _compute_vlim, _select_sequences


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
    channels, montage = _load_montage(electrode_positions_file)

    # Determine which indices to plot
    valid_indices = _resolve_avalanche_indices(
        data, indices, clusters, cluster_id, n_samples, random_seed
    )

    if not valid_indices:
        print("No valid indices to plot")
        return None

    figures = []

    for avalanche_idx in valid_indices:
        avalanche = data[avalanche_idx]
        array_data = np.array(avalanche)
        n_electrodes, n_timepoints = array_data.shape

        info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')
        evoked = mne.EvokedArray(array_data, info)
        evoked.set_montage(montage)

        vlim_use = _compute_vlim(array_data, vlim)
        times = np.arange(n_timepoints) / 250.0

        fig = evoked.plot_topomap(
            times=times, show=show, ncols=ncols, nrows='auto', size=size,
            vlim=vlim_use, units='Activity', image_interp='linear', scalings=1,
            cmap=cmap, res=256, extrapolate='head', outlines='head',
            contours=0, sensors=True, sphere='auto', time_unit='ms'
        )

        fig.suptitle(f'Avalanche {avalanche_idx}', fontsize=16)
        figures.append(fig)

    return figures[0] if len(figures) == 1 else figures


def _resolve_avalanche_indices(data, indices, clusters, cluster_id, n_samples, random_seed):
    """Determine which avalanche indices to plot based on selection mode."""
    if indices is not None:
        return [idx for idx in indices if 0 <= idx < len(data)]

    if clusters is not None:
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
            return []

        n_to_sample = min(n_samples, len(available_indices))
        if random_seed is not None:
            np.random.seed(random_seed)
        result = np.random.choice(available_indices, size=n_to_sample, replace=False).tolist()
        print(f"Plotting {n_to_sample} avalanches from {cluster_label} "
              f"(out of {len(available_indices)} available)")
        return result

    # No clusters — sample from all
    n_to_sample = min(n_samples, len(data))
    if random_seed is not None:
        np.random.seed(random_seed)
    result = np.random.choice(len(data), size=n_to_sample, replace=False).tolist()
    print(f"Plotting {n_to_sample} avalanches from all data "
          f"(out of {len(data)} available)")
    return result


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
    channels, montage = _load_montage(electrode_positions_file)
    n_electrodes, n_timepoints = average.shape

    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')
    evoked = mne.EvokedArray(average, info)
    evoked.set_montage(montage)

    vlim = _compute_vlim(average, vlim)
    times = np.arange(n_timepoints) / 250.0

    fig = evoked.plot_topomap(
        times=times, show=show, ncols=ncols, nrows='auto', size=size,
        vlim=vlim, units='', image_interp='linear', scalings=1,
        cmap=cmap, res=256, extrapolate='head', outlines='head',
        contours=0, sensors=False, sphere='auto', time_unit='ms'
    )

    if cluster_id is not None:
        fig.suptitle(f'Cluster {cluster_id} Consensus Average', fontsize=14)

    return fig


def plot_sequence_topomaps(
    sequences_by_cluster: Dict,
    datasets: List[np.ndarray],
    dataset_names: List[str],
    electrode_positions_file: str,
    cluster_id: Optional[int] = None,
    focus: Optional[Union[str, List[str]]] = None,
    polarity: Optional[str] = None,
    sort_by_length: bool = False,
    n_sequences: int = 3,
    random_seed: Optional[int] = None,
    sfreq: float = 250.0,
    ncols: int = 10,
    size: float = 1.5,
    vlim: Optional[tuple] = None,
    cmap: str = 'RdBu_r',
    show: bool = True
) -> Union[plt.Figure, List[plt.Figure], None]:
    """
    Plot sequence topomaps from pre-loaded EEG datasets.

    Extracts sequence time windows from EEG data and plots each
    timepoint as a scalp topomap. Values are displayed in microvolts.

    Parameters
    ----------
    sequences_by_cluster : dict
        {cluster_id: [sequence_dicts, ...]} from sequence detection.
        Each sequence dict must have 'dataset', 'start_time', 'end_time'.
    datasets : list of np.ndarray
        Pre-loaded EEG arrays, each shape (n_electrodes, n_samples), in Volts.
    dataset_names : list of str
        Basenames corresponding to each array in datasets
        (e.g. 'ctrl_sub001_meditation.set').
    electrode_positions_file : str
        Path to electrode positions JSON file.
    cluster_id : int, optional
        Filter sequences to this cluster. None plots from all clusters.
    focus : str or list of str, optional
        Filter sequences to specific dataset(s). A single string or a list
        of dataset filenames. None allows all datasets.
    polarity : str, optional
        Filter sequences by polarity ('standing', 'rotating', 'mixed', etc.).
        None allows all polarities.
    sort_by_length : bool
        If True, select the longest sequences instead of random sampling
        (default: False).
    n_sequences : int
        Number of sequences to plot (default: 3).
    random_seed : int, optional
        Random seed for reproducible sequence selection.
    sfreq : float
        Sampling frequency in Hz (default: 250.0).
    ncols : int
        Number of topomap columns per row (default: 10).
    size : float
        Size of each topomap (default: 1.5).
    vlim : tuple, optional
        Color scale limits in microvolts. None uses adaptive per-sequence scaling.
    cmap : str
        Colormap name (default: 'RdBu_r').
    show : bool
        Whether to display the plot (default: True).

    Returns
    -------
    fig : Figure, list of Figures, or None
        The created figure(s). None if no matching sequences found.

    Examples
    --------
    >>> # Longest stable sequences from cluster 5
    >>> fig = plot_sequence_topomaps(
    ...     sequences, datasets, dataset_names,
    ...     'electrode_positions.json',
    ...     cluster_id=5, polarity='standing', sort_by_length=True
    ... )
    >>> # Random rotating sequences from a specific dataset
    >>> fig = plot_sequence_topomaps(
    ...     sequences, datasets, dataset_names,
    ...     'electrode_positions.json',
    ...     focus='ctrl_sub001_meditation.set', polarity='rotating'
    ... )
    """
    if len(dataset_names) != len(datasets):
        print(f"dataset_names ({len(dataset_names)}) != datasets ({len(datasets)}). "
              f"Pass the full dataset_names list and use focus='filename.set' to filter.")
        return None

    selected, ds_name_to_idx = _select_sequences(
        sequences_by_cluster, dataset_names, cluster_id=cluster_id,
        focus=focus, polarity=polarity, sort_by_length=sort_by_length,
        n_sequences=n_sequences, random_seed=random_seed
    )
    if selected is None:
        return None

    channels, montage = _load_montage(electrode_positions_file)
    figures = []

    for cid, seq in selected:
        ds_name = seq['dataset']
        if ds_name not in ds_name_to_idx:
            print(f"  Skipping sequence: {ds_name} not found in dataset_names")
            continue

        data = datasets[ds_name_to_idx[ds_name]] * 1e6  # V to \u00b5V

        start_idx = max(0, int(seq['start_time'] * sfreq))
        end_idx = min(data.shape[1], int(seq['end_time'] * sfreq))
        seq_data = data[:, start_idx:end_idx]
        n_electrodes, n_timepoints = seq_data.shape

        if n_timepoints == 0:
            print(f"  Skipping empty sequence: {ds_name} {seq['start_time']:.3f}-{seq['end_time']:.3f}s")
            continue

        info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=sfreq, ch_types='eeg')
        evoked = mne.EvokedArray(seq_data, info)
        evoked.set_montage(montage)

        vlim_use = _compute_vlim(seq_data, vlim)
        times = np.arange(n_timepoints) / sfreq

        fig = evoked.plot_topomap(
            times=times, show=False, ncols=ncols, nrows='auto', size=size,
            vlim=vlim_use, units='\u00b5V', image_interp='linear', scalings=1,
            cmap=cmap, res=256, extrapolate='head', outlines='head',
            contours=0, sensors=False, sphere='auto', time_unit='ms'
        )

        seq_polarity = seq.get('polarity', '?')
        length = seq.get('length', '?')
        title = (f'Cluster {cid} | {ds_name}\n'
                 f'{seq["start_time"]:.3f}\u2013{seq["end_time"]:.3f}s | '
                 f'polarity: {seq_polarity} | length: {length}')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        if show:
            plt.show()
        figures.append(fig)

    if not figures:
        print("No sequences could be plotted")
        return None

    return figures[0] if len(figures) == 1 else figures
