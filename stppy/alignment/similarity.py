"""
Similarity matrix computation for avalanche alignment.

Supports both serial (small datasets) and parallel (large datasets) computation.
"""

import numpy as np
import time
from datetime import timedelta
from typing import List, Tuple, Dict, Optional
import os
import glob
import pickle

from .spatial import create_temporal_similarity_matrix
from .temporal import dtw_on_similarity_matrix


def compute_full_similarity_matrix(
    avalanche_arrays: List[np.ndarray],
    K: np.ndarray,
    show_progress: bool = True,
    save_interval: Optional[int] = None,
    checkpoint_file: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full pairwise similarity matrix for all avalanches (serial computation).

    Use this for small-to-medium datasets (< 1000 avalanches).
    For larger datasets, use compute_similarity_blocks for parallel computation.

    Parameters
    ----------
    avalanche_arrays : list of np.ndarray
        List of avalanche data arrays
    K : np.ndarray
        Physical distance matrix between electrodes (normalized)
    show_progress : bool
        Whether to print progress updates (default: True)
    save_interval : int, optional
        Save checkpoint every N avalanche comparisons (default: None, no checkpoints)
    checkpoint_file : str, optional
        File path for checkpoints (default: None)

    Returns
    -------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix, shape (n_avalanches, n_avalanches)
    path_matrix : np.ndarray
        DTW paths for each pair, shape (n_avalanches, n_avalanches), dtype=object

    Examples
    --------
    >>> # Simple usage
    >>> similarity, paths = compute_full_similarity_matrix(avalanches, K)
    >>>
    >>> # With checkpointing
    >>> similarity, paths = compute_full_similarity_matrix(
    ...     avalanches, K,
    ...     save_interval=100,
    ...     checkpoint_file='similarity_checkpoint.pkl'
    ... )
    """

    n_avalanches = len(avalanche_arrays)

    # Initialize matrices
    similarity_matrix = np.full((n_avalanches, n_avalanches), np.nan, dtype=np.float32)
    path_matrix = np.full((n_avalanches, n_avalanches), None, dtype=object)

    if show_progress:
        print(f"Computing similarity for {n_avalanches} avalanches")
        print(f"Total pairs: {n_avalanches * (n_avalanches + 1) // 2:,} (upper triangle)")

    start_time = time.time()
    completed_pairs = 0
    total_pairs = n_avalanches * (n_avalanches + 1) // 2

    for i in range(n_avalanches):
        row_start = time.time()

        for j in range(i, n_avalanches):  # Upper triangle only

            # Diagonal case
            if i == j:
                similarity_matrix[i, j] = 0.0
                avalanche_length = avalanche_arrays[i].shape[1]
                path_matrix[i, j] = [[(t, t) for t in range(avalanche_length)]]
                completed_pairs += 1
                continue

            # Compute temporal similarity matrix
            temp_sim_matrix = create_temporal_similarity_matrix(
                avalanche_arrays[i],
                avalanche_arrays[j],
                K,
                show_progress=False
            )

            # Apply DTW
            dtw_distance, paths, was_transposed = dtw_on_similarity_matrix(temp_sim_matrix)

            # Store results
            similarity_matrix[i, j] = dtw_distance
            similarity_matrix[j, i] = dtw_distance  # Symmetric

            path_matrix[i, j] = paths
            # Transpose paths for symmetric entry
            if paths:
                transposed_paths = [[(p[1], p[0]) for p in path] for path in paths]
                path_matrix[j, i] = transposed_paths

            completed_pairs += 1

            # Checkpoint saving
            if save_interval and checkpoint_file and completed_pairs % save_interval == 0:
                checkpoint = {
                    'similarity_matrix': similarity_matrix,
                    'path_matrix': path_matrix,
                    'completed_pairs': completed_pairs,
                    'last_i': i,
                    'last_j': j
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
                if show_progress:
                    print(f"    Checkpoint saved: {completed_pairs}/{total_pairs} pairs")

        # Progress update per row
        if show_progress and (i + 1) % max(1, n_avalanches // 10) == 0:
            elapsed = time.time() - start_time
            progress = completed_pairs / total_pairs
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            print(f"  {progress*100:.1f}% complete | "
                  f"Elapsed: {timedelta(seconds=int(elapsed))} | "
                  f"ETA: {timedelta(seconds=int(eta))}")

    total_time = time.time() - start_time

    if show_progress:
        print(f"\n✓ Completed in {timedelta(seconds=int(total_time))}")
        print(f"  Average time per pair: {total_time/total_pairs:.3f}s")
        print(f"  Similarity range: [{np.nanmin(similarity_matrix):.3f}, "
              f"{np.nanmax(similarity_matrix):.3f}]")

    return similarity_matrix, path_matrix


def compute_similarity_block(
    avalanche_arrays: List[np.ndarray],
    K: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    is_diagonal: bool = False
) -> Dict:
    """
    Compute a single block of the similarity matrix.

    This function is designed for parallel computation where each worker
    computes a different block.

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
        Whether this is a diagonal block (compute upper triangle only)

    Returns
    -------
    block_dict : dict
        Dictionary containing:
        - 'similarity_matrix': block similarity matrix
        - 'path_matrix': block path matrix
        - 'row_indices': list of row indices
        - 'col_indices': list of column indices
        - 'is_diagonal': whether this was a diagonal block

    Examples
    --------
    >>> # Compute block 0-100 vs 0-100 (diagonal)
    >>> block = compute_similarity_block(
    ...     avalanches, K,
    ...     row_start=0, row_end=100,
    ...     col_start=0, col_end=100,
    ...     is_diagonal=True
    ... )
    """

    n_rows = row_end - row_start
    n_cols = col_end - col_start

    print(f"Computing block [{row_start}:{row_end}] x [{col_start}:{col_end}]")
    print(f"  Size: {n_rows} x {n_cols} = {n_rows * n_cols:,} pairs")
    print(f"  Diagonal: {is_diagonal}")

    # Initialize block matrices
    similarity_block = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    path_block = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    start_time = time.time()
    completed = 0

    if is_diagonal:
        # Diagonal block - compute upper triangle only
        total_pairs = n_rows * (n_rows + 1) // 2
    else:
        total_pairs = n_rows * n_cols

    for local_i in range(n_rows):
        global_i = row_start + local_i

        # Determine column range
        if is_diagonal:
            local_col_start = local_i  # Start from diagonal
        else:
            local_col_start = 0

        for local_j in range(local_col_start, n_cols):
            global_j = col_start + local_j

            # Diagonal case
            if global_i == global_j:
                similarity_block[local_i, local_j] = 0.0
                avalanche_length = avalanche_arrays[global_i].shape[1]
                path_block[local_i][local_j] = [[(t, t) for t in range(avalanche_length)]]

                # Mirror for diagonal block
                if is_diagonal and local_i != local_j:
                    similarity_block[local_j, local_i] = 0.0
                    path_block[local_j][local_i] = path_block[local_i][local_j]

                completed += 1
                continue

            # Compute similarity
            temp_sim_matrix = create_temporal_similarity_matrix(
                avalanche_arrays[global_i],
                avalanche_arrays[global_j],
                K,
                show_progress=False
            )

            dtw_distance, paths, was_transposed = dtw_on_similarity_matrix(temp_sim_matrix)

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
            if completed % max(1, total_pairs // 10) == 0:
                elapsed = time.time() - start_time
                progress = completed / total_pairs
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                print(f"    {progress*100:.0f}% | ETA: {int(eta)}s")

    total_time = time.time() - start_time
    print(f"  ✓ Block completed in {timedelta(seconds=int(total_time))}")

    return {
        'similarity_matrix': similarity_block,
        'path_matrix': path_block,
        'row_indices': list(range(row_start, row_end)),
        'col_indices': list(range(col_start, col_end)),
        'is_diagonal': is_diagonal,
        'computation_time': total_time
    }


def generate_block_plan(
    n_avalanches: int,
    block_size: int = 500
) -> List[Tuple[int, int, int, int, bool]]:
    """
    Generate a plan for computing similarity matrix in blocks.

    Returns list of (row_start, row_end, col_start, col_end, is_diagonal) tuples
    that cover the upper triangle of the full matrix.

    Parameters
    ----------
    n_avalanches : int
        Total number of avalanches
    block_size : int
        Size of each block (default: 500)

    Returns
    -------
    block_plan : list of tuples
        List of (row_start, row_end, col_start, col_end, is_diagonal)

    Examples
    --------
    >>> plan = generate_block_plan(1000, block_size=500)
    >>> print(f"Need to compute {len(plan)} blocks")
    >>> for i, (rs, re, cs, ce, diag) in enumerate(plan):
    ...     print(f"Block {i}: [{rs}:{re}] x [{cs}:{ce}] diagonal={diag}")
    """

    n_blocks = (n_avalanches + block_size - 1) // block_size
    block_plan = []

    for i in range(n_blocks):
        row_start = i * block_size
        row_end = min((i + 1) * block_size, n_avalanches)

        # Only compute upper triangle (j <= i)
        for j in range(i + 1):
            col_start = j * block_size
            col_end = min((j + 1) * block_size, n_avalanches)

            is_diagonal = (i == j)

            block_plan.append((row_start, row_end, col_start, col_end, is_diagonal))

    print(f"Block plan for {n_avalanches} avalanches with block_size={block_size}:")
    print(f"  {n_blocks} blocks per dimension")
    print(f"  {len(block_plan)} total blocks to compute")
    print(f"  Estimated {n_avalanches**2 / 2:,.0f} total pairs")

    return block_plan


def merge_blocks(
    block_files: List[str],
    output_file: Optional[str] = None
) -> Dict:
    """
    Merge multiple block files into a single structure.

    Parameters
    ----------
    block_files : list of str
        List of block file paths to merge
    output_file : str, optional
        If provided, save merged result to this file

    Returns
    -------
    merged : dict
        Merged block dictionary containing all blocks

    Examples
    --------
    >>> # Merge all blocks from a directory
    >>> import glob
    >>> block_files = sorted(glob.glob('blocks/block_*.pkl'))
    >>> merged = merge_blocks(block_files, output_file='merged.pkl')
    """

    if not block_files:
        raise ValueError("No block files provided")

    print(f"=== MERGING {len(block_files)} BLOCKS ===\n")

    for filename in block_files:
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  {os.path.basename(filename)} ({file_size_mb:.1f} MB)")

    # Load all blocks
    print(f"\nLoading blocks...")
    start_time = time.time()
    all_blocks = []

    for i, filename in enumerate(block_files):
        print(f"  Loading {i+1}/{len(block_files)}: {os.path.basename(filename)}")
        with open(filename, 'rb') as f:
            block_dict = pickle.load(f)
            all_blocks.append(block_dict)

    load_time = time.time() - start_time
    print(f"Loaded all blocks in {load_time:.1f}s")

    # Merge
    print(f"\nMerging...")
    merged = {
        'blocks': all_blocks,
        'n_blocks': len(all_blocks)
    }

    print(f"✓ Merged {len(all_blocks)} blocks")

    # Save if requested
    if output_file:
        print(f"\nSaving to {output_file}...")
        with open(output_file, 'wb') as f:
            pickle.dump(merged, f)
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✓ Saved ({file_size_mb:.1f} MB)")

    return merged


def reconstruct_full_matrices(
    merged_blocks: Dict,
    n_avalanches: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct full similarity and path matrices from merged blocks.

    Parameters
    ----------
    merged_blocks : dict
        Merged block dictionary from merge_blocks()
    n_avalanches : int
        Total number of avalanches

    Returns
    -------
    similarity_matrix : np.ndarray
        Full similarity matrix, shape (n_avalanches, n_avalanches)
    path_matrix : np.ndarray
        Full path matrix, shape (n_avalanches, n_avalanches), dtype=object

    Examples
    --------
    >>> merged = merge_blocks(block_files)
    >>> similarity, paths = reconstruct_full_matrices(merged, n_avalanches=1000)
    """

    print(f"Reconstructing full matrices for {n_avalanches} avalanches...")

    # Initialize
    similarity = np.full((n_avalanches, n_avalanches), np.nan, dtype=np.float32)
    paths = np.full((n_avalanches, n_avalanches), None, dtype=object)

    # Fill from blocks
    for block in merged_blocks['blocks']:
        row_indices = block['row_indices']
        col_indices = block['col_indices']

        # Convert to numpy arrays for fancy indexing
        row_idx = np.array(row_indices)
        col_idx = np.array(col_indices)

        # Fill similarity
        similarity[np.ix_(row_idx, col_idx)] = block['similarity_matrix']

        # Fill paths
        path_block = block['path_matrix']
        for local_i, global_i in enumerate(row_indices):
            for local_j, global_j in enumerate(col_indices):
                paths[global_i, global_j] = path_block[local_i][local_j]

    # Fill lower triangle by symmetry
    print("  Filling lower triangle by symmetry...")
    for i in range(n_avalanches):
        for j in range(i):
            if np.isnan(similarity[i, j]) and not np.isnan(similarity[j, i]):
                similarity[i, j] = similarity[j, i]
                if paths[j, i] is not None:
                    # Transpose path
                    paths[i, j] = [[(p[1], p[0]) for p in path] for path in paths[j, i]]

    print(f"✓ Reconstruction complete")
    print(f"  Similarity range: [{np.nanmin(similarity):.3f}, {np.nanmax(similarity):.3f}]")

    return similarity, paths


def get_similarity_from_blocks(
    i: int,
    j: int,
    merged_blocks: Dict
) -> float:
    """
    Get similarity value for a pair directly from merged blocks.

    More memory-efficient than reconstructing full matrix.

    Parameters
    ----------
    i, j : int
        Avalanche indices
    merged_blocks : dict
        Merged block dictionary

    Returns
    -------
    similarity : float
        Similarity value for pair (i, j)
    """

    for block in merged_blocks['blocks']:
        if i in block['row_indices'] and j in block['col_indices']:
            local_i = block['row_indices'].index(i)
            local_j = block['col_indices'].index(j)
            return block['similarity_matrix'][local_i, local_j]

    return np.nan


def get_path_from_blocks(
    i: int,
    j: int,
    merged_blocks: Dict
) -> list:
    """
    Get DTW path for a pair directly from merged blocks.

    Parameters
    ----------
    i, j : int
        Avalanche indices
    merged_blocks : dict
        Merged block dictionary

    Returns
    -------
    paths : list
        DTW paths for pair (i, j)
    """

    for block in merged_blocks['blocks']:
        if i in block['row_indices'] and j in block['col_indices']:
            local_i = block['row_indices'].index(i)
            local_j = block['col_indices'].index(j)
            return block['path_matrix'][local_i][local_j]

    return None
