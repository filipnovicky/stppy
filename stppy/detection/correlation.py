"""
Correlation-based burst detection using spatiotemporal pattern matching.

This module provides an alternative detection approach that matches data against
template patterns using spatial correlation. Unlike threshold-based detection,
this identifies recurring patterns based on similarity to cluster templates.
"""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional


def build_spatial_smoothing_matrix(
    electrode_positions: np.ndarray,
    k: float = 3.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build spatial smoothing matrix for frame-by-frame correlation.

    Creates a row-normalized matrix that weights nearby electrodes based on
    their physical distances. Used for spatiotemporal correlation computation.

    Parameters
    ----------
    electrode_positions : np.ndarray, shape (n_electrodes, 3)
        Physical x, y, z coordinates of electrodes
    k : float, default=3.0
        Decay rate for exponential distance weighting.
        Higher values = faster decay (less spatial smoothing)
        Lower values = slower decay (more spatial smoothing)
        k=0 uses raw distances without decay

    Returns
    -------
    W_smooth : np.ndarray, shape (n_electrodes, n_electrodes)
        Row-normalized smoothing matrix (each row sums to 1)
    distances : np.ndarray, shape (n_electrodes, n_electrodes)
        Pairwise Euclidean distances between electrodes
    mean_distance : float
        Mean inter-electrode distance (excluding diagonal)

    Examples
    --------
    >>> positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> W_smooth, distances, mean_dist = build_spatial_smoothing_matrix(positions, k=3.0)
    >>> W_smooth.shape
    (3, 3)
    >>> np.allclose(W_smooth.sum(axis=1), 1.0)  # Rows sum to 1
    True
    """
    n_electrodes = len(electrode_positions)

    # Compute pairwise Euclidean distances
    distances = cdist(electrode_positions, electrode_positions, metric='euclidean')

    # Mean distance (excluding diagonal)
    mask = ~np.eye(n_electrodes, dtype=bool)
    mean_distance = np.mean(distances[mask])

    # Normalize distances by mean
    W = distances / mean_distance

    # Compute spatial weights with exponential decay
    if k == 0:
        spatial_weights = W
    else:
        spatial_weights = np.exp(-k * W)

    # Row-normalize so each row sums to 1
    W_smooth = spatial_weights / spatial_weights.sum(axis=1, keepdims=True)

    return W_smooth, distances, mean_distance


def spatiotemporal_correlation(
    pattern: np.ndarray,
    data_window: np.ndarray,
    W_smooth: np.ndarray
) -> float:
    """
    Compute spatiotemporal correlation using frame-by-frame spatial correlation.

    For each timepoint:
    1. Spatially smooth both pattern and data frames
    2. Compute correlation between smoothed frames
    3. Average correlation across all timepoints

    This captures:
    - Spatial structure at each moment (via smoothing)
    - Temporal evolution (via frame-by-frame comparison)
    - Propagation patterns (spatial structure moving through time)

    Parameters
    ----------
    pattern : np.ndarray, shape (n_electrodes, n_timepoints)
        Template pattern showing spatiotemporal propagation
    data_window : np.ndarray, shape (n_electrodes, n_timepoints)
        Data window to compare against pattern
    W_smooth : np.ndarray, shape (n_electrodes, n_electrodes)
        Spatial smoothing matrix from build_spatial_smoothing_matrix()

    Returns
    -------
    correlation : float
        Spatiotemporal correlation in range [-1, 1]
        High positive values indicate strong pattern match
        Negative values indicate anti-correlation

    Examples
    --------
    >>> pattern = np.random.randn(10, 20)
    >>> data = np.random.randn(10, 20)
    >>> W_smooth = np.eye(10) / 10  # Simple uniform weighting
    >>> corr = spatiotemporal_correlation(pattern, data, W_smooth)
    >>> -1 <= corr <= 1
    True
    """
    n_electrodes, n_timepoints = pattern.shape

    frame_correlations = []

    # For each timepoint, compute spatial correlation
    for t in range(n_timepoints):
        # Get spatial frame at time t
        pattern_frame = pattern[:, t]  # (n_electrodes,)
        data_frame = data_window[:, t]  # (n_electrodes,)

        # Spatially smooth both frames
        # This allows nearby electrodes to contribute to each location
        pattern_smooth = W_smooth @ pattern_frame
        data_smooth = W_smooth @ data_frame

        # Compute correlation between smoothed spatial patterns
        if np.std(pattern_smooth) > 1e-6 and np.std(data_smooth) > 1e-6:
            corr = np.corrcoef(pattern_smooth, data_smooth)[0, 1]
            frame_correlations.append(corr)
        else:
            # If either frame is constant, correlation is undefined
            frame_correlations.append(0.0)

    # Average correlation across time
    # High value means spatial structure matches at each timepoint
    if len(frame_correlations) > 0:
        avg_correlation = np.mean(frame_correlations)
    else:
        avg_correlation = 0.0

    # Clip to valid range (handle numerical errors)
    avg_correlation = np.clip(avg_correlation, -1.0, 1.0)

    return avg_correlation


def detect_bursts_by_correlation(
    data: np.ndarray,
    pattern: np.ndarray,
    W_smooth: np.ndarray,
    threshold: float = 0.7,
    sfreq: float = 250.0,
    return_correlations: bool = False
) -> Dict:
    """
    Detect bursts by sliding pattern correlation across continuous data.

    Performs sliding window correlation to identify moments where the data
    strongly matches (or anti-matches) a template pattern. Useful for detecting
    recurring cluster patterns in continuous recordings.

    Parameters
    ----------
    data : np.ndarray, shape (n_electrodes, n_samples)
        Continuous EEG/MEG data
    pattern : np.ndarray, shape (n_electrodes, n_timepoints)
        Template pattern to search for (e.g., cluster average)
    W_smooth : np.ndarray, shape (n_electrodes, n_electrodes)
        Spatial smoothing matrix
    threshold : float, default=0.7
        Correlation threshold for burst detection.
        Positive bursts: correlation > threshold
        Negative bursts: correlation < -threshold
    sfreq : float, default=250.0
        Sampling frequency in Hz (for rate calculations)
    return_correlations : bool, default=False
        If True, return full correlation timeseries

    Returns
    -------
    burst_info : dict
        Dictionary containing:
        - 'positive_bursts': List of (start, end, peak_corr) tuples
        - 'negative_bursts': List of (start, end, peak_corr) tuples
        - 'n_positive': Number of positive bursts
        - 'n_negative': Number of negative bursts
        - 'rate_positive': Positive bursts per second
        - 'rate_negative': Negative bursts per second
        - 'correlations': Full correlation timeseries (if return_correlations=True)
        - 'positions': Sample indices for correlations (if return_correlations=True)

    Examples
    --------
    >>> data = np.random.randn(10, 1000)
    >>> pattern = np.random.randn(10, 20)
    >>> W_smooth = np.eye(10)
    >>> bursts = detect_bursts_by_correlation(data, pattern, W_smooth, threshold=0.7)
    >>> bursts['n_positive'] >= 0
    True
    """
    n_electrodes, n_samples = data.shape
    n_electrodes_p, pattern_length = pattern.shape

    if n_electrodes != n_electrodes_p:
        raise ValueError(f"Electrode count mismatch: data has {n_electrodes}, pattern has {n_electrodes_p}")

    # Sliding window correlation
    correlations = []
    positions = []

    for start in range(0, n_samples - pattern_length + 1):
        end = start + pattern_length
        window = data[:, start:end]

        corr = spatiotemporal_correlation(pattern, window, W_smooth)
        correlations.append(corr)
        positions.append(start)

    correlations = np.array(correlations)
    positions = np.array(positions)

    # Detect positive bursts (above threshold)
    positive_mask = correlations > threshold
    positive_bursts = _extract_burst_intervals(positions, correlations, positive_mask, pattern_length)

    # Detect negative bursts (below -threshold)
    negative_mask = correlations < -threshold
    negative_bursts = _extract_burst_intervals(positions, correlations, negative_mask, pattern_length)

    # Calculate rates
    duration = n_samples / sfreq
    rate_positive = len(positive_bursts) / duration if duration > 0 else 0
    rate_negative = len(negative_bursts) / duration if duration > 0 else 0

    result = {
        'positive_bursts': positive_bursts,
        'negative_bursts': negative_bursts,
        'n_positive': len(positive_bursts),
        'n_negative': len(negative_bursts),
        'rate_positive': rate_positive,
        'rate_negative': rate_negative,
        'threshold': threshold,
        'pattern_length': pattern_length,
        'duration': duration
    }

    if return_correlations:
        result['correlations'] = correlations
        result['positions'] = positions

    return result


def _extract_burst_intervals(
    positions: np.ndarray,
    correlations: np.ndarray,
    mask: np.ndarray,
    pattern_length: int
) -> List[Tuple[int, int, float]]:
    """
    Extract continuous burst intervals from boolean mask.

    Parameters
    ----------
    positions : np.ndarray
        Sample positions
    correlations : np.ndarray
        Correlation values
    mask : np.ndarray
        Boolean mask indicating burst samples
    pattern_length : int
        Length of pattern (for calculating burst end)

    Returns
    -------
    bursts : list of (start, end, peak_corr) tuples
    """
    if not np.any(mask):
        return []

    bursts = []
    positions_above = positions[mask]
    correlations_above = correlations[mask]

    # Find continuous intervals
    if len(positions_above) == 0:
        return []

    # Group consecutive positions
    intervals = []
    current_start = positions_above[0]
    current_end = positions_above[0] + pattern_length
    current_peak_idx = 0

    for i in range(1, len(positions_above)):
        if positions_above[i] <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, positions_above[i] + pattern_length)
            if abs(correlations_above[i]) > abs(correlations_above[current_peak_idx]):
                current_peak_idx = i
        else:  # Gap - start new interval
            peak_corr = correlations_above[current_peak_idx]
            bursts.append((current_start, current_end, peak_corr))

            current_start = positions_above[i]
            current_end = positions_above[i] + pattern_length
            current_peak_idx = i

    # Add final interval
    peak_corr = correlations_above[current_peak_idx]
    bursts.append((current_start, current_end, peak_corr))

    return bursts


def compute_temporal_statistics(
    correlations: np.ndarray,
    positions: np.ndarray,
    threshold: float = 1.0,
    sfreq: float = 250.0
) -> Dict:
    """
    Analyze temporal properties of correlations including bursts and intervals.

    Computes comprehensive statistics about correlation bursts:
    - Burst counts and rates (positive and negative)
    - Inter-burst intervals and their statistics
    - Positions and magnitudes of detected bursts

    Parameters
    ----------
    correlations : np.ndarray
        Correlation timeseries values
    positions : np.ndarray
        Sample positions corresponding to correlations
    threshold : float, default=1.0
        Correlation threshold for burst detection
    sfreq : float, default=250.0
        Sampling frequency in Hz

    Returns
    -------
    statistics : dict
        Dictionary containing:
        - 'n_above_threshold': Number of positive bursts
        - 'n_below_threshold': Number of negative bursts
        - 'rate_above': Positive burst rate (per second)
        - 'rate_below': Negative burst rate (per second)
        - 'mean_lag_above': Mean inter-burst interval (positive, seconds)
        - 'std_lag_above': Std of inter-burst intervals (positive)
        - 'median_lag_above': Median inter-burst interval (positive)
        - 'intervals_above': Array of inter-burst intervals (positive, seconds)
        - 'positions_above': Positions of positive bursts
        - 'correlations_above': Correlation values of positive bursts
        - 'mean_lag_below': Mean inter-burst interval (negative, seconds)
        - 'std_lag_below': Std of inter-burst intervals (negative)
        - 'median_lag_below': Median inter-burst interval (negative)
        - 'intervals_below': Array of inter-burst intervals (negative, seconds)
        - 'positions_below': Positions of negative bursts
        - 'correlations_below': Correlation values of negative bursts
        - 'threshold': Threshold used
        - 'total_duration': Total duration analyzed (seconds)
    """
    above_threshold = correlations > threshold
    n_above = np.sum(above_threshold)

    below_threshold = correlations < -threshold
    n_below = np.sum(below_threshold)

    positions_above = positions[above_threshold]
    correlations_above = correlations[above_threshold]

    positions_below = positions[below_threshold]
    correlations_below = correlations[below_threshold]

    # Compute intervals for positive bursts
    if len(positions_above) > 1:
        intervals_above = np.diff(positions_above)
        intervals_above_sec = intervals_above / sfreq
        mean_lag_above = np.mean(intervals_above_sec)
        std_lag_above = np.std(intervals_above_sec)
        median_lag_above = np.median(intervals_above_sec)
    else:
        intervals_above_sec = np.array([])
        mean_lag_above = np.nan
        std_lag_above = np.nan
        median_lag_above = np.nan

    # Compute intervals for negative bursts
    if len(positions_below) > 1:
        intervals_below = np.diff(positions_below)
        intervals_below_sec = intervals_below / sfreq
        mean_lag_below = np.mean(intervals_below_sec)
        std_lag_below = np.std(intervals_below_sec)
        median_lag_below = np.median(intervals_below_sec)
    else:
        intervals_below_sec = np.array([])
        mean_lag_below = np.nan
        std_lag_below = np.nan
        median_lag_below = np.nan

    total_duration = (positions[-1] - positions[0]) / sfreq if len(positions) > 0 else 0

    rate_above = n_above / total_duration if total_duration > 0 else 0
    rate_below = n_below / total_duration if total_duration > 0 else 0

    return {
        'n_above_threshold': n_above,
        'rate_above': rate_above,
        'mean_lag_above': mean_lag_above,
        'std_lag_above': std_lag_above,
        'median_lag_above': median_lag_above,
        'intervals_above': intervals_above_sec,
        'positions_above': positions_above,
        'correlations_above': correlations_above,

        'n_below_threshold': n_below,
        'rate_below': rate_below,
        'mean_lag_below': mean_lag_below,
        'std_lag_below': std_lag_below,
        'median_lag_below': median_lag_below,
        'intervals_below': intervals_below_sec,
        'positions_below': positions_below,
        'correlations_below': correlations_below,

        'threshold': threshold,
        'total_duration': total_duration
    }
