"""
Dynamic Time Warping (DTW) for temporal alignment of avalanches.
"""

import numpy as np
from typing import Tuple, List


def dtw_on_similarity_matrix(similarity_matrix: np.ndarray) -> Tuple[float, List, bool]:
    """
    Apply subsequence DTW on similarity matrix to find optimal temporal alignment.
    Allows starting anywhere on the left edge and ending anywhere on the right edge.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix with shape (M, N)

    Returns
    -------
    min_cost : float
        Minimum DTW distance
    paths : list
        List of optimal paths (each path is a list of (i,j) tuples)
    was_transposed : bool
        Whether the matrix was transposed
    """

    M, N = similarity_matrix.shape

    # Check if we need to transpose (ensure X is shorter than Y)
    was_transposed = False
    if N > M:
        similarity_matrix = similarity_matrix.T
        M, N = similarity_matrix.shape
        was_transposed = True

    # Calculate adaptive penalty based on data statistics
    finite_costs = similarity_matrix[np.isfinite(similarity_matrix)]
    if len(finite_costs) > 0 and np.any(finite_costs > 0):
        min_nonzero_cost = np.min(finite_costs[finite_costs > 0])
        penalty = 0.001 * min_nonzero_cost
    else:
        penalty = 0.0

    # Initialize DTW matrix
    dtw_matrix = np.full((M + 1, N + 1), np.inf)

    # Calculate minimum required span
    min_required_span = N

    # Only allow starting from rows that can achieve minimum span
    max_start_row = int(M - min_required_span)

    for start_i in range(max_start_row + 1):
        dtw_matrix[start_i, 0] = 0

    # Fill DTW matrix with penalties for non-diagonal moves
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = similarity_matrix[i-1, j-1]

            candidates = [
                dtw_matrix[i-1, j-1],           # diagonal - no penalty
                dtw_matrix[i-1, j] + penalty,   # up (insertion) - small penalty
                dtw_matrix[i, j-1] + 2*penalty  # left (deletion) - 2x penalty
            ]

            finite_candidates = [c for c in candidates if not np.isinf(c)]

            if len(finite_candidates) > 0:
                dtw_matrix[i, j] = cost + min(finite_candidates)
            else:
                dtw_matrix[i, j] = np.inf

    # Valid ending positions sorted by cost
    valid_endings = [(dtw_matrix[end_i, N], end_i)
                    for end_i in range(N, M + 1)
                    if not np.isinf(dtw_matrix[end_i, N])]
    valid_endings.sort()

    # Try each ending position until we find a valid path
    for cost, end_i in valid_endings:
        path = backtrack_greedy(dtw_matrix, similarity_matrix, end_i, N, penalty)

        if len(path) > 0:
            start_row = path[0][0]
            end_row = path[-1][0]
            span = end_row - start_row + 1

            if span >= min_required_span:
                min_cost = cost

                # Transpose path back if needed
                if was_transposed:
                    path = [(j, i) for (i, j) in path]

                return min_cost, [path], was_transposed

    # No valid path found
    return np.inf, [], was_transposed


def backtrack_greedy(dtw_matrix: np.ndarray, similarity_matrix: np.ndarray,
                    end_i: int, end_j: int, penalty: float) -> List[Tuple[int, int]]:
    """
    Simple greedy backtrack through DTW matrix.

    Parameters
    ----------
    dtw_matrix : np.ndarray
        Filled DTW matrix
    similarity_matrix : np.ndarray
        Original similarity matrix
    end_i : int
        Ending row index
    end_j : int
        Ending column index
    penalty : float
        Penalty value used in DTW

    Returns
    -------
    path : list
        List of (i, j) coordinate tuples forming the path
    """

    if np.isinf(dtw_matrix[end_i, end_j]):
        return []

    path = []
    i, j = end_i, end_j

    while j > 0:
        path.append((i-1, j-1))
        cell_cost = similarity_matrix[i-1, j-1]

        # Try diagonal (preferred)
        if i > 0 and j > 0 and not np.isinf(dtw_matrix[i-1, j-1]):
            if np.isclose(dtw_matrix[i, j], cell_cost + dtw_matrix[i-1, j-1], atol=1e-9):
                i, j = i-1, j-1
                continue

        # Try up (second choice)
        if i > 0 and not np.isinf(dtw_matrix[i-1, j]):
            if np.isclose(dtw_matrix[i, j], cell_cost + dtw_matrix[i-1, j] + penalty, atol=1e-9):
                i, j = i-1, j
                continue

        # Try left (last resort)
        if j > 0 and not np.isinf(dtw_matrix[i, j-1]):
            if np.isclose(dtw_matrix[i, j], cell_cost + dtw_matrix[i, j-1] + 2 * penalty, atol=1e-9):
                i, j = i, j-1
                continue

        # No valid move found
        break

    return list(reversed(path))
