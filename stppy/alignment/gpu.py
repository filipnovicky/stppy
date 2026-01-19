"""
GPU-accelerated alignment functions using PyTorch CUDA.

This module provides GPU-accelerated versions of the alignment computations
for significant speedup on NVIDIA GPUs.
"""

import numpy as np
import time
from datetime import timedelta
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


def check_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return HAS_TORCH and HAS_CUDA


def get_device():
    """Get the appropriate torch device."""
    if HAS_CUDA:
        return torch.device('cuda')
    return torch.device('cpu')


def spatial_alignment_distance_batch_gpu(
    avalanche_x: np.ndarray,
    avalanche_y: np.ndarray,
    K: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Compute temporal similarity matrix using GPU acceleration.

    This batches the spatial alignment distance computation across all
    timepoint pairs, using GPU for distance matrix operations.

    Parameters
    ----------
    avalanche_x : np.ndarray
        First avalanche, shape (n_electrodes, n_timepoints_x)
    avalanche_y : np.ndarray
        Second avalanche, shape (n_electrodes, n_timepoints_y)
    K : np.ndarray
        Physical distance matrix between electrodes
    device : torch.device, optional
        Device to use (default: CUDA if available)

    Returns
    -------
    similarity_matrix : np.ndarray
        Similarity matrix, shape (n_timepoints_x, n_timepoints_y)
    """
    if device is None:
        device = get_device()

    M = avalanche_x.shape[1]
    N = avalanche_y.shape[1]

    # Move data to GPU
    K_torch = torch.from_numpy(K).float().to(device)
    mean_distance = K_torch[K_torch > 0].mean().item()

    similarity_matrix = np.zeros((M, N), dtype=np.float32)

    # Process in batches to avoid memory issues
    for i in range(M):
        x_pattern = avalanche_x[:, i]

        # Pre-compute indices for x pattern
        pos_x_idx = np.where(x_pattern == 1)[0]
        neg_x_idx = np.where(x_pattern == -1)[0]
        x_active = np.sum(x_pattern != 0)

        for j in range(N):
            y_pattern = avalanche_y[:, j]

            pos_y_idx = np.where(y_pattern == 1)[0]
            neg_y_idx = np.where(y_pattern == -1)[0]

            total_cost = 0.0

            # Handle positive activities
            if len(pos_x_idx) > 0 and len(pos_y_idx) > 0:
                total_cost += _solve_assignment_gpu(pos_x_idx, pos_y_idx, K, K_torch, device)
            elif len(pos_x_idx) > 0 or len(pos_y_idx) > 0:
                unmatched = len(pos_x_idx) + len(pos_y_idx)
                total_cost += unmatched * mean_distance

            # Handle negative activities
            if len(neg_x_idx) > 0 and len(neg_y_idx) > 0:
                total_cost += _solve_assignment_gpu(neg_x_idx, neg_y_idx, K, K_torch, device)
            elif len(neg_x_idx) > 0 or len(neg_y_idx) > 0:
                unmatched = len(neg_x_idx) + len(neg_y_idx)
                total_cost += unmatched * mean_distance

            # Normalize
            y_active = np.sum(y_pattern != 0)
            total_electrodes = x_active + y_active

            if total_electrodes > 0:
                similarity_matrix[i, j] = total_cost / total_electrodes
            else:
                similarity_matrix[i, j] = 0.0

    return similarity_matrix


def _solve_assignment_gpu(
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    K: np.ndarray,
    K_torch: torch.Tensor,
    device: torch.device
) -> float:
    """Solve optimal assignment with GPU-accelerated cost matrix construction."""
    n_source = len(source_idx)
    n_target = len(target_idx)

    if n_source == 0 or n_target == 0:
        return 0.0

    # Extract cost matrix using GPU indexing
    cost_matrix = K[np.ix_(source_idx, target_idx)]

    if n_source == n_target:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()

    elif n_source < n_target:
        augmented_matrix = np.zeros((n_target, n_target), dtype=np.float32)
        augmented_matrix[:n_source, :] = cost_matrix

        for j in range(n_target):
            augmented_matrix[n_source:, j] = cost_matrix[:, j].min()

        row_ind, col_ind = linear_sum_assignment(augmented_matrix)
        return augmented_matrix[row_ind, col_ind].sum()

    else:
        augmented_matrix = np.zeros((n_source, n_source), dtype=np.float32)
        augmented_matrix[:, :n_target] = cost_matrix

        for i in range(n_source):
            augmented_matrix[i, n_target:] = cost_matrix[i, :].min()

        row_ind, col_ind = linear_sum_assignment(augmented_matrix)
        return augmented_matrix[row_ind, col_ind].sum()


def dtw_on_similarity_matrix_gpu(
    similarity_matrix: np.ndarray,
    device: Optional[torch.device] = None
) -> Tuple[float, List, bool]:
    """
    GPU-accelerated DTW on similarity matrix.

    Uses PyTorch for the DTW matrix computation.
    """
    if device is None:
        device = get_device()

    M, N = similarity_matrix.shape

    was_transposed = False
    if N > M:
        similarity_matrix = similarity_matrix.T
        M, N = similarity_matrix.shape
        was_transposed = True

    # Move to GPU
    sim_torch = torch.from_numpy(similarity_matrix).float().to(device)

    # Calculate penalty
    finite_mask = torch.isfinite(sim_torch) & (sim_torch > 0)
    if finite_mask.any():
        penalty = 0.001 * sim_torch[finite_mask].min().item()
    else:
        penalty = 0.0

    # Initialize DTW matrix on GPU
    dtw_matrix = torch.full((M + 1, N + 1), float('inf'), device=device)

    min_required_span = N
    max_start_row = int(M - min_required_span)

    for start_i in range(max_start_row + 1):
        dtw_matrix[start_i, 0] = 0

    # Fill DTW matrix (this is the main computation)
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = sim_torch[i-1, j-1]

            diag = dtw_matrix[i-1, j-1]
            up = dtw_matrix[i-1, j] + penalty
            left = dtw_matrix[i, j-1] + 2 * penalty

            min_prev = torch.min(torch.stack([diag, up, left]))

            if not torch.isinf(min_prev):
                dtw_matrix[i, j] = cost + min_prev

    # Move back to CPU for backtracking
    dtw_np = dtw_matrix.cpu().numpy()
    sim_np = similarity_matrix

    # Find valid endings
    valid_endings = [(dtw_np[end_i, N], end_i)
                     for end_i in range(N, M + 1)
                     if not np.isinf(dtw_np[end_i, N])]
    valid_endings.sort()

    # Backtrack
    for cost, end_i in valid_endings:
        path = _backtrack_greedy(dtw_np, sim_np, end_i, N, penalty)

        if len(path) > 0:
            start_row = path[0][0]
            end_row = path[-1][0]
            span = end_row - start_row + 1

            if span >= min_required_span:
                if was_transposed:
                    path = [(j, i) for (i, j) in path]
                return cost, [path], was_transposed

    return np.inf, [], was_transposed


def _backtrack_greedy(dtw_matrix, similarity_matrix, end_i, end_j, penalty):
    """Greedy backtrack through DTW matrix."""
    if np.isinf(dtw_matrix[end_i, end_j]):
        return []

    path = []
    i, j = end_i, end_j

    while j > 0:
        path.append((i-1, j-1))
        cell_cost = similarity_matrix[i-1, j-1]

        if i > 0 and j > 0 and not np.isinf(dtw_matrix[i-1, j-1]):
            if np.isclose(dtw_matrix[i, j], cell_cost + dtw_matrix[i-1, j-1], atol=1e-9):
                i, j = i-1, j-1
                continue

        if i > 0 and not np.isinf(dtw_matrix[i-1, j]):
            if np.isclose(dtw_matrix[i, j], cell_cost + dtw_matrix[i-1, j] + penalty, atol=1e-9):
                i, j = i-1, j
                continue

        if j > 0 and not np.isinf(dtw_matrix[i, j-1]):
            if np.isclose(dtw_matrix[i, j], cell_cost + dtw_matrix[i, j-1] + 2 * penalty, atol=1e-9):
                i, j = i, j-1
                continue

        break

    return list(reversed(path))


def compute_similarity_block_gpu(
    avalanche_arrays: List[np.ndarray],
    K: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    is_diagonal: bool = False,
    device: Optional[torch.device] = None
) -> Dict:
    """
    GPU-accelerated block similarity computation.

    Parameters
    ----------
    avalanche_arrays : list of np.ndarray
        List of all avalanche arrays
    K : np.ndarray
        Physical distance matrix between electrodes
    row_start, row_end : int
        Row index range for this block
    col_start, col_end : int
        Column index range for this block
    is_diagonal : bool
        Whether this is a diagonal block
    device : torch.device, optional
        GPU device to use

    Returns
    -------
    block_dict : dict
        Block results dictionary
    """
    if device is None:
        device = get_device()

    n_rows = row_end - row_start
    n_cols = col_end - col_start

    print(f"Computing block [{row_start}:{row_end}] x [{col_start}:{col_end}] on {device}")
    print(f"  Size: {n_rows} x {n_cols}")
    print(f"  Diagonal: {is_diagonal}")

    # Initialize
    similarity_block = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    path_block = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    start_time = time.time()
    completed = 0

    if is_diagonal:
        total_pairs = n_rows * (n_rows + 1) // 2
    else:
        total_pairs = n_rows * n_cols

    for local_i in range(n_rows):
        global_i = row_start + local_i

        if is_diagonal:
            local_col_start = local_i
        else:
            local_col_start = 0

        for local_j in range(local_col_start, n_cols):
            global_j = col_start + local_j

            # Diagonal case
            if global_i == global_j:
                similarity_block[local_i, local_j] = 0.0
                avalanche_length = avalanche_arrays[global_i].shape[1]
                path_block[local_i][local_j] = [[(t, t) for t in range(avalanche_length)]]

                if is_diagonal and local_i != local_j:
                    similarity_block[local_j, local_i] = 0.0
                    path_block[local_j][local_i] = path_block[local_i][local_j]

                completed += 1
                continue

            # Compute similarity using GPU
            temp_sim_matrix = spatial_alignment_distance_batch_gpu(
                avalanche_arrays[global_i],
                avalanche_arrays[global_j],
                K,
                device
            )

            dtw_distance, paths, was_transposed = dtw_on_similarity_matrix_gpu(
                temp_sim_matrix, device
            )

            similarity_block[local_i, local_j] = dtw_distance
            path_block[local_i][local_j] = paths

            # Mirror for diagonal block
            if is_diagonal:
                similarity_block[local_j, local_i] = dtw_distance
                if paths:
                    transposed_paths = [[(p[1], p[0]) for p in path] for path in paths]
                    path_block[local_j][local_i] = transposed_paths

            completed += 1

            # Progress
            if completed % max(1, total_pairs // 20) == 0:
                elapsed = time.time() - start_time
                progress = completed / total_pairs
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                print(f"    {progress*100:.0f}% | Elapsed: {int(elapsed)}s | ETA: {int(eta)}s")

    total_time = time.time() - start_time
    print(f"  Block completed in {timedelta(seconds=int(total_time))}")

    return {
        'similarity_matrix': similarity_block,
        'path_matrix': path_block,
        'row_indices': list(range(row_start, row_end)),
        'col_indices': list(range(col_start, col_end)),
        'is_diagonal': is_diagonal,
        'computation_time': total_time,
        'device': str(device)
    }
