"""Avalanche detection and filtering."""

from .avalanches import (
    create_avalanche_filter,
    detect_avalanches,
    convert_to_peaks
)
from .filtering import (
    filter_avalanche_propagation,
    get_boundary_electrodes_with_occipital_exception,
    has_spatial_clustering_both_polarities,
    get_nearest_neighbors
)

__all__ = [
    'create_avalanche_filter',
    'detect_avalanches',
    'convert_to_peaks',
    'filter_avalanche_propagation',
    'get_boundary_electrodes_with_occipital_exception',
    'has_spatial_clustering_both_polarities',
    'get_nearest_neighbors'
]
