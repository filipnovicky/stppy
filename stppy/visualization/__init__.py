"""Visualization utilities for avalanche analysis."""

from .topomaps import (
    plot_avalanche_topomaps,
    plot_cluster_average
)

from .timeseries import (
    plot_avalanche_timeline,
    plot_multi_channel_timeline,
    plot_cluster_correlations,
    plot_avalanche_raster
)

# Video functions are optional (require cv2)
try:
    from .videos import (
        create_cluster_video,
        create_avalanche_video,
        combine_videos
    )
    _VIDEOS_AVAILABLE = True
except ImportError:
    _VIDEOS_AVAILABLE = False

__all__ = [
    'plot_avalanche_topomaps',
    'plot_cluster_average',
    'plot_avalanche_timeline',
    'plot_multi_channel_timeline',
    'plot_cluster_correlations',
    'plot_avalanche_raster',
]

if _VIDEOS_AVAILABLE:
    __all__.extend([
        'create_cluster_video',
        'create_avalanche_video',
        'combine_videos'
    ])
