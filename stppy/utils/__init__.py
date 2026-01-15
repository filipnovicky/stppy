"""Utility functions for distances and metrics."""

from .distances import (
    create_eeg_ground_metric,
    normalize_distance_matrix
)

__all__ = [
    'create_eeg_ground_metric',
    'normalize_distance_matrix'
]
