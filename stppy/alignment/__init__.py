"""Spatial and temporal alignment for avalanches."""

from .spatial import (
    spatial_alignment_distance,
    solve_optimal_assignment,
    create_temporal_similarity_matrix
)
from .temporal import (
    dtw_on_similarity_matrix,
    backtrack_greedy
)
from .similarity import (
    compute_full_similarity_matrix,
    compute_similarity_block,
    generate_block_plan,
    merge_blocks,
    reconstruct_full_matrices,
    get_similarity_from_blocks,
    get_path_from_blocks
)

__all__ = [
    'spatial_alignment_distance',
    'solve_optimal_assignment',
    'create_temporal_similarity_matrix',
    'dtw_on_similarity_matrix',
    'backtrack_greedy',
    'compute_full_similarity_matrix',
    'compute_similarity_block',
    'generate_block_plan',
    'merge_blocks',
    'reconstruct_full_matrices',
    'get_similarity_from_blocks',
    'get_path_from_blocks'
]
