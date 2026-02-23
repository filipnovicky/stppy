"""Shared helpers for visualization modules."""

import numpy as np
import json
from mne.channels import make_dig_montage


def _load_montage(electrode_positions_file):
    """Load electrode positions and create MNE montage.

    Parameters
    ----------
    electrode_positions_file : str
        Path to JSON file with electrode positions.

    Returns
    -------
    channels : list of str
        Electrode channel names in order.
    montage : mne.channels.DigMontage
        MNE-compatible montage object.
    """
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)
    channels = list(electrode_data.keys())
    ch_pos = {ch: [electrode_data[ch]['x'], electrode_data[ch]['y'], electrode_data[ch]['z']]
              for ch in channels}
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    return channels, montage


def _compute_vlim(data, vlim=None):
    """Compute symmetric color scale limits.

    Parameters
    ----------
    data : np.ndarray
        Data array to compute limits from.
    vlim : tuple, optional
        Pre-set limits. If provided, returned as-is.

    Returns
    -------
    vlim : tuple
        (vmin, vmax) symmetric around zero.
    """
    if vlim is not None:
        return vlim
    data_max = np.max(np.abs(data))
    return (-data_max, data_max) if data_max > 0 else (-1, 1)


def _select_sequences(sequences_by_cluster, dataset_names, cluster_id=None,
                      focus=None, polarity=None, sort_by_length=False,
                      n_sequences=3, random_seed=None):
    """Filter and select sequence candidates.

    Parameters
    ----------
    sequences_by_cluster : dict
        {cluster_id: [sequence_dicts, ...]}.
    dataset_names : list of str
        Basenames corresponding to datasets.
    cluster_id : int, optional
        Filter to this cluster.
    focus : str or list of str, optional
        Filter to specific dataset(s).
    polarity : str, optional
        Filter by polarity.
    sort_by_length : bool
        Select longest sequences instead of random.
    n_sequences : int
        Number of sequences to select.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    selected : list of (cid, sequence_dict) tuples, or None if no matches.
    ds_name_to_idx : dict mapping dataset name to index.
    """
    ds_name_to_idx = {name: i for i, name in enumerate(dataset_names)}

    # Normalize focus
    if focus is not None:
        if isinstance(focus, str):
            focus_set = {focus}
        else:
            focus_set = set(focus)
        missing = focus_set - set(ds_name_to_idx)
        if missing:
            print(f"Not found in dataset_names: {missing}")
            return None, ds_name_to_idx
        print(f"Filtering to {len(focus_set)} dataset(s): {', '.join(sorted(focus_set))}")
    else:
        focus_set = None

    # Collect matching sequences
    candidates = []
    cluster_ids = [cluster_id] if cluster_id is not None else sorted(sequences_by_cluster.keys())

    for cid in cluster_ids:
        if cid not in sequences_by_cluster:
            continue
        for seq in sequences_by_cluster[cid]:
            if focus_set is not None and seq['dataset'] not in focus_set:
                continue
            if polarity is not None and seq.get('polarity') != polarity:
                continue
            candidates.append((cid, seq))

    if not candidates:
        filters = f"cluster_id={cluster_id}, focus={focus}, polarity={polarity}"
        print(f"No sequences found matching {filters}")
        return None, ds_name_to_idx

    # Select sequences
    n_to_plot = min(n_sequences, len(candidates))

    if sort_by_length:
        candidates.sort(key=lambda x: x[1].get('length', 0), reverse=True)
        selected = candidates[:n_to_plot]
        print(f"Selected {n_to_plot} longest sequence(s) from {len(candidates)} candidates")
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
        selected_idx = np.random.choice(len(candidates), size=n_to_plot, replace=False)
        selected = [candidates[i] for i in selected_idx]
        print(f"Selected {n_to_plot} sequence(s) from {len(candidates)} candidates")

    return selected, ds_name_to_idx
