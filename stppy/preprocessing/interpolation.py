"""
Electrode interpolation utilities.
"""

import numpy as np
import mne
import json
from typing import List, Tuple, Optional, Union


def _load_electrode_positions(positions_file: str):
    """Load electrode positions from JSON.

    Returns
    -------
    ch_pos : dict
        {channel_name: np.array([x, y, z])}
    expected_order : list of str
        Channel names in JSON order.
    """
    with open(positions_file, 'r') as f:
        electrode_data = json.load(f)
    ch_pos = {ch: np.array([v['x'], v['y'], v['z']]) for ch, v in electrode_data.items()}
    return ch_pos, list(electrode_data.keys())


def _interpolate_channel(ch, ch_pos, sources, original_data, ch_to_idx):
    """Compute inverse-distance-weighted interpolation for one channel.

    Parameters
    ----------
    ch : str
        Target channel name.
    ch_pos : dict
        {name: np.array([x, y, z])} for all known channels.
    sources : dict
        {name: np.array([x, y, z])} for available source channels.
    original_data : np.ndarray
        Shape (n_channels, n_times) for Raw or (n_epochs, n_channels, n_times) for Epochs.
    ch_to_idx : dict
        {channel_name: index} into original_data.

    Returns
    -------
    signal : np.ndarray or None
        Interpolated signal, or None if no valid neighbors.
    neighbor_names : list of str
        Names of the neighbors used.
    """
    target_pos = ch_pos[ch]
    distances = [(name, np.linalg.norm(target_pos - pos)) for name, pos in sources.items()]
    neighbors = sorted(distances, key=lambda x: x[1])[:6]

    if not neighbors:
        return None, []

    weights = np.array([1.0 / (d**2) if d > 1e-10 else 1.0 for _, d in neighbors])
    weights /= weights.sum()

    neighbor_idx = [ch_to_idx[name] for name, _ in neighbors if name in ch_to_idx]
    if not neighbor_idx:
        return None, []

    w = weights[:len(neighbor_idx)]
    if original_data.ndim == 2:
        signal = sum(wi * original_data[i] for wi, i in zip(w, neighbor_idx))
    else:
        signal = sum(wi * original_data[:, i, :] for wi, i in zip(w, neighbor_idx))

    neighbor_names = [name for name, _ in neighbors[:len(neighbor_idx)]]
    return signal, neighbor_names


def interpolate_missing_electrodes(
    raw_or_epochs: Union[mne.io.Raw, mne.Epochs],
    positions_file: str = 'electrode_positions.json',
    force_interpolate: Optional[List[str]] = None,
    apply_average_reference: bool = False,
    verbose: bool = True
) -> Tuple[Union[mne.io.Raw, mne.Epochs], List[str]]:
    """Interpolate missing electrodes using custom positions with proper channel ordering.

    Parameters
    ----------
    raw_or_epochs : mne.Raw or mne.Epochs
        Input EEG data.
    positions_file : str
        Path to JSON file with electrode positions.
    force_interpolate : list or None
        Channels to force re-interpolation even if present.
    apply_average_reference : bool
        If True, apply average reference after interpolation.
    verbose : bool
        Print progress.

    Returns
    -------
    data_interpolated : mne.Raw or mne.Epochs
        Data with interpolated channels in correct order.
    interpolated_channels : list of str
        Names of interpolated channels.
    """
    if force_interpolate is None:
        force_interpolate = []

    ch_pos, expected_order = _load_electrode_positions(positions_file)
    data_copy = raw_or_epochs.copy()

    # Determine which channels need interpolation
    current = set(data_copy.ch_names)
    missing = list(set(ch_pos.keys()) - current)
    forced = [ch for ch in force_interpolate if ch in current]
    to_interpolate = list(set(missing + forced))

    # If nothing to interpolate, just reorder
    if not to_interpolate:
        try:
            in_order = [ch for ch in expected_order if ch in data_copy.ch_names]
            data_copy = data_copy.reorder_channels(in_order)
            if verbose:
                print("Reordered existing channels to match electrode positions")
        except ValueError as e:
            raise ValueError(
                f"Channel reorder failed — electrode names in the data don't match "
                f"the positions file '{positions_file}'. Check channel names."
            ) from e
        return data_copy, []

    if verbose:
        print(f"Interpolating {len(to_interpolate)} channels: {to_interpolate}")

    # Source channels: present in data and in positions, not being replaced
    sources = {ch: ch_pos[ch] for ch in data_copy.ch_names
               if ch not in to_interpolate and ch in ch_pos}

    original_data = data_copy.get_data()
    original_ch_names = data_copy.ch_names.copy()
    ch_to_idx = {ch: idx for idx, ch in enumerate(original_ch_names)}

    # Drop channels being re-interpolated
    existing_to_remove = [ch for ch in to_interpolate if ch in data_copy.ch_names]
    if existing_to_remove:
        keep = [ch for ch in data_copy.ch_names if ch not in existing_to_remove]
        data_copy = data_copy.pick_channels(keep)

    # Interpolate each missing channel
    interpolated_data = []
    successful = []

    for ch in to_interpolate:
        if ch not in ch_pos:
            continue
        signal, neighbor_names = _interpolate_channel(ch, ch_pos, sources, original_data, ch_to_idx)
        if signal is None:
            if verbose:
                print(f"  {ch}: No valid neighbors found, skipping")
            continue
        interpolated_data.append(signal)
        successful.append(ch)
        if verbose:
            print(f"  {ch}: from {neighbor_names}")

    # Add interpolated channels back
    if successful:
        if original_data.ndim == 2:
            interp_array = np.array(interpolated_data)
            interp_obj = mne.io.RawArray(
                interp_array,
                mne.create_info(successful, data_copy.info['sfreq'], ['eeg'] * len(successful)),
                verbose=False
            )
        else:
            interp_array = np.transpose(np.array(interpolated_data), (1, 0, 2))
            interp_obj = mne.EpochsArray(
                interp_array,
                mne.create_info(successful, data_copy.info['sfreq'], ['eeg'] * len(successful)),
                verbose=False
            )
        data_copy = data_copy.add_channels([interp_obj], force_update_info=True)

    # Reorder to match electrode positions JSON
    try:
        final_order = [ch for ch in expected_order if ch in data_copy.ch_names]
        data_copy = data_copy.reorder_channels(final_order)
    except ValueError as e:
        raise ValueError(
            f"Channel reorder after interpolation failed — electrode names in the data "
            f"don't match the positions file '{positions_file}'. Check channel names."
        ) from e

    if verbose:
        print(f"Interpolated {len(successful)} channels")
        print(f"Final data has {len(data_copy.ch_names)} channels in correct order")

    if apply_average_reference:
        data_copy.set_eeg_reference('average', projection=False, verbose=False)
        if verbose:
            print("Applied average reference")

    return data_copy, successful
