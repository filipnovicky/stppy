"""
Distance metrics and ground metric creation utilities.
"""

import numpy as np
import json
from typing import Tuple


def create_eeg_ground_metric(positions_file: str = 'electrode_positions.json') -> Tuple[np.ndarray, list]:
    """
    Create ground metric based on actual EEG electrode positions from JSON file.

    Parameters
    ----------
    positions_file : str
        Path to JSON file containing electrode positions

    Returns
    -------
    K : numpy.ndarray
        Ground metric matrix based on electrode distances
    channel_names : list
        List of channel names used
    """

    print(f"Creating EEG ground metric from {positions_file}...")

    with open(positions_file, 'r') as f:
        electrode_data = json.load(f)

    channel_names = list(electrode_data.keys())
    n_channels = len(channel_names)

    print(f"Loaded {n_channels} channels")

    # Convert positions to numpy array
    positions_3d = []
    for ch_name in channel_names:
        pos_dict = electrode_data[ch_name]
        pos = np.array([pos_dict['x'], pos_dict['y'], pos_dict['z']])
        positions_3d.append(pos)

    positions_3d = np.array(positions_3d)
    print(f"Using {len(positions_3d)} electrode positions")

    # Compute pairwise Euclidean distances
    K = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(n_channels):
            if i == j:
                K[i, j] = 0.0
            else:
                dist = np.linalg.norm(positions_3d[i] - positions_3d[j])
                K[i, j] = dist

    print(f"Ground metric statistics:")
    print(f"  Shape: {K.shape}")
    print(f"  Min distance: {K.min():.4f}")
    print(f"  Max distance: {K.max():.4f}")
    print(f"  Median distance: {np.median(K[K > 0]):.4f}")

    return K, channel_names


def normalize_distance_matrix(K: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Normalize distance matrix.

    Parameters
    ----------
    K : np.ndarray
        Distance matrix
    method : str
        Normalization method ('mean', 'max', or 'none')

    Returns
    -------
    K_normalized : np.ndarray
        Normalized distance matrix
    """

    if method == 'none':
        return K

    if method == 'mean':
        mean_distance = K[K > 0].mean()
        if mean_distance > 0:
            K_normalized = K / mean_distance
            print(f"Normalized by mean distance: {mean_distance:.3f}")
            return K_normalized

    elif method == 'max':
        max_distance = K.max()
        if max_distance > 0:
            K_normalized = K / max_distance
            print(f"Normalized by max distance: {max_distance:.3f}")
            return K_normalized

    print("Warning: normalization failed, returning original K")
    return K
