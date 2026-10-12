"""
STPPY - Spatiotemporal Pattern Processing for pYthon

A package for analyzing EEG avalanches with spatial and temporal alignment.
"""

__version__ = '0.2.0'

from . import io
from . import preprocessing
from . import detection
from . import alignment
from . import clustering
from . import jacobian
from . import visualization
from . import utils

__all__ = [
    'io',
    'preprocessing',
    'detection',
    'alignment',
    'clustering',
    'jacobian',
    'visualization',
    'utils'
]
