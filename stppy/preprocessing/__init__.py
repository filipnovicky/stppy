"""Preprocessing utilities for EEG data."""

from .interpolation import interpolate_missing_electrodes
from .resampling import resample_data

__all__ = [
    'interpolate_missing_electrodes',
    'resample_data'
]
