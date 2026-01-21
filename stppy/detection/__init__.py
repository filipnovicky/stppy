"""Avalanche detection and filtering.

Provides two complementary detection approaches:
1. Threshold-based detection (avalanches.py) - De novo event detection
2. Correlation-based detection (correlation.py) - Pattern matching against templates
"""

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
from .correlation import (
    build_spatial_smoothing_matrix,
    spatiotemporal_correlation,
    detect_bursts_by_correlation,
    detect_bursts_in_clusters,
    compute_temporal_statistics
)

__all__ = [
    # Threshold-based detection
    'create_avalanche_filter',
    'detect_avalanches',
    'convert_to_peaks',
    'filter_avalanche_propagation',
    'get_boundary_electrodes_with_occipital_exception',
    'has_spatial_clustering_both_polarities',
    'get_nearest_neighbors',
    # Correlation-based detection
    'build_spatial_smoothing_matrix',
    'spatiotemporal_correlation',
    'detect_bursts_by_correlation',
    'detect_bursts_in_clusters',
    'compute_temporal_statistics'
]
