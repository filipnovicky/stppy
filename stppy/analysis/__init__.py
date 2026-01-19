"""Simple metadata tracking for avalanche datasets."""

from .metadata import (
    create_avalanche_index,
    analyze_cluster_composition,
    summarize_burst_statistics
)

__all__ = [
    'create_avalanche_index',
    'analyze_cluster_composition',
    'summarize_burst_statistics'
]
