"""Analysis utilities for avalanche metadata and statistics."""

from .metadata import (
    # Original functions (specific metadata)
    link_avalanches_to_datasets,
    analyze_clusters_by_avalanche_metadata,
    # Generic functions (universal)
    create_avalanche_index,
    query_avalanches,
    group_by_metadata_field
)

__all__ = [
    'link_avalanches_to_datasets',
    'analyze_clusters_by_avalanche_metadata',
    'create_avalanche_index',
    'query_avalanches',
    'group_by_metadata_field'
]
