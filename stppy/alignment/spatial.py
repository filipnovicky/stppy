"""
Spatial alignment using optimal transport for EEG avalanches.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple


def spatial_alignment_distance(x_pattern: np.ndarray, y_pattern: np.ndarray,
                               K: np.ndarray) -> float:
    """
    Calculate optimal transport distance with penalties for polarity mismatches
    and cardinality imbalances.

    Parameters
    ----------
    x_pattern : np.ndarray
        First spatial pattern (electrode activities at one timepoint)
    y_pattern : np.ndarray
        Second spatial pattern (electrode activities at one timepoint)
    K : np.ndarray
        Physical distance matrix between electrodes

    Returns
    -------
    normalized_cost : float
        Normalized transport cost per electrode
    """

    pos_x_idx = np.where(x_pattern == 1)[0]
    neg_x_idx = np.where(x_pattern == -1)[0]
    pos_y_idx = np.where(y_pattern == 1)[0]
    neg_y_idx = np.where(y_pattern == -1)[0]

    mean_distance = np.mean(K[K > 0])

    total_cost = 0

    # Handle positive activities
    if len(pos_x_idx) > 0 and len(pos_y_idx) > 0:
        total_cost += solve_optimal_assignment(pos_x_idx, pos_y_idx, K)
    elif len(pos_x_idx) > 0 or len(pos_y_idx) > 0:
        unmatched = len(pos_x_idx) + len(pos_y_idx)
        total_cost += unmatched * mean_distance

    # Handle negative activities
    if len(neg_x_idx) > 0 and len(neg_y_idx) > 0:
        total_cost += solve_optimal_assignment(neg_x_idx, neg_y_idx, K)
    elif len(neg_x_idx) > 0 or len(neg_y_idx) > 0:
        unmatched = len(neg_x_idx) + len(neg_y_idx)
        total_cost += unmatched * mean_distance

    # Normalize by total electrodes involved
    total_electrodes_involved = np.sum(x_pattern != 0) + np.sum(y_pattern != 0)

    if total_electrodes_involved > 0:
        normalized_cost = total_cost / total_electrodes_involved
    else:
        normalized_cost = 0

    return normalized_cost


def solve_optimal_assignment(source_idx: np.ndarray, target_idx: np.ndarray,
                             K: np.ndarray) -> float:
    """
    Solve optimal assignment using Hungarian algorithm with minimum-based dummy nodes.

    Parameters
    ----------
    source_idx : array
        Indices of source electrodes
    target_idx : array
        Indices of target electrodes
    K : np.ndarray
        Distance matrix

    Returns
    -------
    total_cost : float
        Optimal assignment cost including penalties for unmatched nodes
    """

    n_source = len(source_idx)
    n_target = len(target_idx)

    if n_source == 0 or n_target == 0:
        return 0

    cost_matrix = K[np.ix_(source_idx, target_idx)]

    if n_source == n_target:
        # Balanced case - standard Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()

    elif n_source < n_target:
        # Create augmented matrix (n_target × n_target)
        augmented_matrix = np.zeros((n_target, n_target))

        # Fill real sources
        augmented_matrix[:n_source, :] = cost_matrix

        # Fill dummy sources
        for j in range(n_target):
            min_dist_to_target_j = cost_matrix[:, j].min()
            augmented_matrix[n_source:, j] = min_dist_to_target_j

        row_ind, col_ind = linear_sum_assignment(augmented_matrix)
        return augmented_matrix[row_ind, col_ind].sum()

    else:  # n_source > n_target
        # Create augmented matrix (n_source × n_source)
        augmented_matrix = np.zeros((n_source, n_source))

        # Fill real targets
        augmented_matrix[:, :n_target] = cost_matrix

        # Fill dummy targets
        for i in range(n_source):
            min_dist_from_source_i = cost_matrix[i, :].min()
            augmented_matrix[i, n_target:] = min_dist_from_source_i

        row_ind, col_ind = linear_sum_assignment(augmented_matrix)
        return augmented_matrix[row_ind, col_ind].sum()


def create_temporal_similarity_matrix(avalanche_x: np.ndarray,
                                     avalanche_y: np.ndarray,
                                     K: np.ndarray,
                                     show_progress: bool = False) -> np.ndarray:
    """
    Create similarity matrix between time steps of two avalanches.

    Parameters
    ----------
    avalanche_x : np.ndarray
        First avalanche, shape (n_electrodes, n_timepoints_x)
    avalanche_y : np.ndarray
        Second avalanche, shape (n_electrodes, n_timepoints_y)
    K : np.ndarray
        Physical distance matrix between electrodes
    show_progress : bool
        Whether to show progress for large matrices

    Returns
    -------
    similarity_matrix : np.ndarray
        Similarity matrix, shape (n_timepoints_x, n_timepoints_y)
    """

    M = avalanche_x.shape[1]
    N = avalanche_y.shape[1]

    similarity_matrix = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            x_pattern = avalanche_x[:, i]
            y_pattern = avalanche_y[:, j]

            distance = spatial_alignment_distance(x_pattern, y_pattern, K)
            similarity_matrix[i, j] = distance

    return similarity_matrix
