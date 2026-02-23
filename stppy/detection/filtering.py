"""
Spatial clustering filters for avalanche detection.
"""

import numpy as np
import json
from scipy.spatial import ConvexHull
from typing import List, Tuple, Dict, Set


def filter_avalanche_propagation(
    all_avalanches: List[List[np.ndarray]],
    all_avalanche_times: List[List[Tuple[int, int]]],
    K: np.ndarray,
    electrode_positions_file: str,
    min_active_neighbors: int = 5,
    n_nearest: int = 8
) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int, int]]]]:
    """
    Filter avalanches to keep only those showing dense spatial clustering in BOTH
    negative and positive activations, excluding boundary electrodes.

    Parameters
    ----------
    all_avalanches : list of lists
        List containing avalanche lists for each participant
    all_avalanche_times : list of lists
        List containing avalanche timing lists for each participant
    K : np.ndarray
        Distance matrix between electrodes (n_electrodes, n_electrodes)
    electrode_positions_file : str
        Path to JSON file with electrode positions
    min_active_neighbors : int
        Minimum number of active neighbors required (default: 5)
    n_nearest : int
        Number of nearest neighbors to consider (default: 8)

    Returns
    -------
    filtered_avalanches : list of lists
        Avalanches with spatial clustering in both polarities
    filtered_avalanche_times : list of lists
        Timing info for filtered avalanches
    """

    if len(all_avalanches) != len(all_avalanche_times):
        raise ValueError(f"Number of avalanche lists ({len(all_avalanches)}) must match "
                        f"number of timing lists ({len(all_avalanche_times)})")

    # Load electrode positions
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    excluded_electrodes = get_boundary_electrodes_with_occipital_exception(electrode_data)

    all_channels = list(electrode_data.keys())
    excluded_indices = {i for i, ch in enumerate(all_channels) if ch in excluded_electrodes}

    print(f"Total electrodes: {len(all_channels)}")
    print(f"Excluded boundary electrodes: {len(excluded_electrodes)}")
    print(f"Excluded electrodes: {sorted(excluded_electrodes)}")

    nearest_neighbors = get_nearest_neighbors(K, n_nearest)

    filtered_avalanches = []
    filtered_avalanche_times = []

    for participant_idx, (participant_avalanches, participant_times) in enumerate(
            zip(all_avalanches, all_avalanche_times)):

        if len(participant_avalanches) != len(participant_times):
            raise ValueError(f"Participant {participant_idx + 1}: Number of avalanches "
                           f"({len(participant_avalanches)}) must match number of timing entries "
                           f"({len(participant_times)})")

        kept_avalanches = []
        kept_times = []

        print(f"\nProcessing participant {participant_idx + 1}: {len(participant_avalanches)} avalanches")

        for avalanche_idx, (avalanche, avalanche_time) in enumerate(
                zip(participant_avalanches, participant_times)):
            has_neg, has_pos = has_spatial_clustering_both_polarities(
                avalanche, excluded_indices, nearest_neighbors, min_active_neighbors
            )

            if has_neg and has_pos:
                kept_avalanches.append(avalanche)
                kept_times.append(avalanche_time)

        print(f"  Kept {len(kept_avalanches)}/{len(participant_avalanches)} avalanches "
              f"(both polarities with clustering)")

        filtered_avalanches.append(kept_avalanches)
        filtered_avalanche_times.append(kept_times)

    return filtered_avalanches, filtered_avalanche_times


def get_boundary_electrodes_with_occipital_exception(electrode_data: dict) -> Set[str]:
    """
    Find boundary electrodes using convex hull, but exclude occipital electrodes
    from boundary removal.

    Parameters
    ----------
    electrode_data : dict
        Dictionary with electrode names as keys and position dicts as values

    Returns
    -------
    excluded_electrodes : set
        Set of electrode names on the boundary (excluding occipitals)
    """

    electrode_names = list(electrode_data.keys())
    positions_2d = []

    for ch in electrode_names:
        pos = electrode_data[ch]
        positions_2d.append([pos['x'], pos['y']])

    positions_2d = np.array(positions_2d)

    try:
        hull = ConvexHull(positions_2d)
        boundary_indices = set(hull.vertices)
    except Exception as e:
        raise ValueError(
            "Could not compute convex hull from electrode positions — "
            "check that electrode_data contains valid 2D coordinates."
        ) from e

    boundary_electrodes = {electrode_names[i] for i in boundary_indices}

    # Special handling: remove occipital electrodes from boundary exclusion
    occipital_electrodes = {'O1', 'O2', 'Oz', 'PO7', 'PO8'}
    occipitals_in_data = occipital_electrodes.intersection(set(electrode_names))

    excluded_electrodes = boundary_electrodes - occipitals_in_data

    return excluded_electrodes


def has_spatial_clustering_both_polarities(
    avalanche: np.ndarray,
    excluded_indices: Set[int],
    nearest_neighbors: Dict[int, List[int]],
    min_active_neighbors: int = 5
) -> Tuple[bool, bool]:
    """
    Check if avalanche contains dense spatial clustering in BOTH negative and
    positive values.

    Parameters
    ----------
    avalanche : np.ndarray
        Avalanche matrix with shape (n_electrodes, duration)
    excluded_indices : set
        Set of electrode indices to exclude from clustering analysis
    nearest_neighbors : dict
        Pre-computed nearest neighbors mapping
    min_active_neighbors : int
        Minimum number of active neighbors required

    Returns
    -------
    has_negative_clustering : bool
    has_positive_clustering : bool
    """

    n_electrodes, duration = avalanche.shape

    seed_electrode_indices = [i for i in range(n_electrodes) if i not in excluded_indices]

    has_negative_clustering = False
    has_positive_clustering = False

    for t in range(duration):
        timestep_data = avalanche[:, t]

        negative_electrodes = np.where(timestep_data < 0)[0]
        positive_electrodes = np.where(timestep_data > 0)[0]

        # Check negative clustering
        if not has_negative_clustering and len(negative_electrodes) > 0:
            negative_set = set(negative_electrodes)
            seed_negative_electrodes = [idx for idx in negative_electrodes if idx in seed_electrode_indices]

            for electrode_idx in seed_negative_electrodes:
                neighbors = nearest_neighbors[electrode_idx]
                valid_neighbors = [n for n in neighbors if n not in excluded_indices]
                active_neighbor_count = sum(1 for neighbor in valid_neighbors if neighbor in negative_set)

                if active_neighbor_count >= min_active_neighbors:
                    has_negative_clustering = True
                    break

        # Check positive clustering
        if not has_positive_clustering and len(positive_electrodes) > 0:
            positive_set = set(positive_electrodes)
            seed_positive_electrodes = [idx for idx in positive_electrodes if idx in seed_electrode_indices]

            for electrode_idx in seed_positive_electrodes:
                neighbors = nearest_neighbors[electrode_idx]
                valid_neighbors = [n for n in neighbors if n not in excluded_indices]
                active_neighbor_count = sum(1 for neighbor in valid_neighbors if neighbor in positive_set)

                if active_neighbor_count >= min_active_neighbors:
                    has_positive_clustering = True
                    break

        if has_negative_clustering and has_positive_clustering:
            return True, True

    return has_negative_clustering, has_positive_clustering


def get_nearest_neighbors(K: np.ndarray, n_nearest: int) -> Dict[int, List[int]]:
    """
    Find the n nearest neighbors for each electrode based on distance matrix.

    Parameters
    ----------
    K : np.ndarray
        Distance matrix between electrodes (n_electrodes, n_electrodes)
    n_nearest : int
        Number of nearest neighbors to find

    Returns
    -------
    nearest_neighbors : dict
        Dictionary mapping electrode_idx -> list of nearest neighbor indices
    """

    n_electrodes = K.shape[0]
    nearest_neighbors = {}

    for electrode_idx in range(n_electrodes):
        distances = K[electrode_idx, :].copy()
        distances[electrode_idx] = np.inf

        sorted_indices = np.argsort(distances)
        neighbors = sorted_indices[:n_nearest].tolist()

        nearest_neighbors[electrode_idx] = neighbors

    return nearest_neighbors
