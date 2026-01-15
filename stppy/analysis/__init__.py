"""Universal metadata analysis utilities for avalanche datasets."""

from .metadata import (
    create_avalanche_index,
    query_avalanches,
    group_by_metadata_field
)

__all__ = [
    'create_avalanche_index',
    'query_avalanches',
    'group_by_metadata_field'
]
