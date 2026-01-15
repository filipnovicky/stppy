"""
Electrode interpolation utilities.
"""

import numpy as np
import mne
import json
from typing import List, Tuple, Optional, Union


def interpolate_missing_electrodes(
    raw_or_epochs: Union[mne.io.Raw, mne.Epochs],
    positions_file: str = 'electrode_positions.json',
    force_interpolate: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[Union[mne.io.Raw, mne.Epochs], List[str]]:
    """
    Interpolate missing electrodes using custom positions with proper channel ordering.

    Parameters
    ----------
    raw_or_epochs : mne.Raw or mne.Epochs
        Input EEG data
    positions_file : str
        Path to JSON file with electrode positions
    force_interpolate : list or None
        Channels to force re-interpolation
    verbose : bool
        Print progress

    Returns
    -------
    data_interpolated : mne.Raw or mne.Epochs
        Data with interpolated channels in correct order
    interpolated_channels : list
        List of interpolated channels
    """

    if force_interpolate is None:
        force_interpolate = []

    data_copy = raw_or_epochs.copy()

    # Load positions
    with open(positions_file, 'r') as f:
        electrode_data = json.load(f)

    ch_pos = {}
    for ch, pos_dict in electrode_data.items():
        ch_pos[ch] = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']])

    # Get the expected channel order from JSON
    expected_channel_order = list(electrode_data.keys())

    # Rename FP1 -> Fp1 if needed
    if 'FP1' in data_copy.ch_names and 'FP1' not in ch_pos and 'Fp1' in ch_pos:
        data_copy.rename_channels({'FP1': 'Fp1'})
        if verbose:
            print("Renamed FP1 -> Fp1")

    # Find channels to interpolate
    current = set(data_copy.ch_names)
    available = set(ch_pos.keys())
    missing = list(available - current)
    forced = [ch for ch in force_interpolate if ch in current]
    to_interpolate = list(set(missing + forced))

    if not to_interpolate:
        try:
            existing_in_expected = [ch for ch in expected_channel_order if ch in data_copy.ch_names]
            data_copy = data_copy.reorder_channels(existing_in_expected)
            if verbose:
                print("Reordered existing channels to match electrode positions")
        except:
            if verbose:
                print("Could not reorder channels - some channels may not match electrode positions")
        return data_copy, []

    if verbose:
        print(f"Interpolating {len(to_interpolate)} channels: {to_interpolate}")

    # Source channels for interpolation
    sources = {ch: ch_pos[ch] for ch in data_copy.ch_names
               if ch not in to_interpolate and ch in ch_pos}

    # Store original data before removing channels
    original_data = data_copy.get_data()
    original_ch_names = data_copy.ch_names.copy()
    ch_to_idx = {ch: idx for idx, ch in enumerate(original_ch_names)}

    # Remove existing channels that will be re-interpolated
    existing_to_remove = [ch for ch in to_interpolate if ch in data_copy.ch_names]
    if existing_to_remove:
        data_copy = data_copy.pick_channels([ch for ch in data_copy.ch_names if ch not in existing_to_remove])

    # Interpolate each channel
    interpolated_data = []
    successful = []

    for ch in to_interpolate:
        if ch not in ch_pos:
            continue

        # Find 6 nearest neighbors
        target_pos = ch_pos[ch]
        distances = [(name, np.linalg.norm(target_pos - pos)) for name, pos in sources.items()]
        neighbors = sorted(distances, key=lambda x: x[1])[:6]

        if not neighbors:
            if verbose:
                print(f"  {ch}: No neighbors found, skipping")
            continue

        # Weights by inverse squared distance
        weights = np.array([1/(d**2) if d > 1e-10 else 1.0 for _, d in neighbors])
        weights /= weights.sum()

        # Weighted average
        neighbor_idx = [ch_to_idx[name] for name, _ in neighbors if name in ch_to_idx]
        if not neighbor_idx:
            if verbose:
                print(f"  {ch}: No valid neighbor indices found, skipping")
            continue

        if original_data.ndim == 2:  # Raw
            signal = sum(w * original_data[idx] for w, idx in zip(weights[:len(neighbor_idx)], neighbor_idx))
        else:  # Epochs
            signal = sum(w * original_data[:, idx, :] for w, idx in zip(weights[:len(neighbor_idx)], neighbor_idx))

        interpolated_data.append(signal)
        successful.append(ch)

        if verbose:
            neighbor_names = [name for name, _ in neighbors[:len(neighbor_idx)]]
            print(f"  {ch}: from {neighbor_names}")

    # Add interpolated channels back
    if successful:
        if original_data.ndim == 2:
            interp_array = np.array(interpolated_data)
        else:
            interp_array = np.transpose(np.array(interpolated_data), (1, 0, 2))

        interp_info = mne.create_info(successful, data_copy.info['sfreq'], ['eeg'] * len(successful))

        if original_data.ndim == 2:
            interp_obj = mne.io.RawArray(interp_array, interp_info, verbose=False)
        else:
            interp_obj = mne.EpochsArray(interp_array, interp_info, verbose=False)

        data_copy = data_copy.add_channels([interp_obj], force_update_info=True)

    # Reorder channels to match electrode positions JSON
    try:
        final_channel_order = [ch for ch in expected_channel_order if ch in data_copy.ch_names]

        if verbose:
            print(f"✓ Reordered {len(data_copy.ch_names)} channels to match electrode positions")

        data_copy = data_copy.reorder_channels(final_channel_order)

    except Exception as e:
        if verbose:
            print(f"Warning: Could not reorder channels: {e}")
            print("Channel order may not match electrode positions!")

    if verbose:
        print(f"✓ Interpolated {len(successful)} channels")
        print(f"✓ Final data has {len(data_copy.ch_names)} channels in correct order")

    return data_copy, successful
