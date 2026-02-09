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

    return avg_correlation


def _vectorized_frame_correlations(
    pattern_smooth: np.ndarray,
    window_smooth: np.ndarray
) -> float:
    """
    Compute average frame-by-frame correlation using vectorized operations.

    This is an optimized version that computes correlations for all timepoints
    at once using broadcasting, avoiding Python loops.

    Parameters
    ----------
    pattern_smooth : np.ndarray, shape (n_electrodes, n_timepoints)
        Pre-smoothed pattern (W_smooth @ pattern already applied)
    window_smooth : np.ndarray, shape (n_electrodes, n_timepoints)
        Pre-smoothed data window (W_smooth @ window already applied)

    Returns
    -------
    avg_correlation : float
        Average correlation across all timepoints, in range [-1, 1]
    """
    # Compute mean and std for each frame (column)
    pattern_mean = pattern_smooth.mean(axis=0, keepdims=True)
    window_mean = window_smooth.mean(axis=0, keepdims=True)

    pattern_std = pattern_smooth.std(axis=0)
    window_std = window_smooth.std(axis=0)

    # Find valid frames (non-constant)
    valid_mask = (pattern_std > 1e-6) & (window_std > 1e-6)

    if not np.any(valid_mask):
        return 0.0

    # Center the data
    pattern_centered = pattern_smooth - pattern_mean
    window_centered = window_smooth - window_mean

    # Compute correlation for each frame: sum(x*y) / (n * std_x * std_y)
    n_electrodes = pattern_smooth.shape[0]
    covariance = np.sum(pattern_centered * window_centered, axis=0) / n_electrodes

    # Correlation = covariance / (std_x * std_y)
    correlations = np.zeros(pattern_smooth.shape[1])
    correlations[valid_mask] = covariance[valid_mask] / (
        pattern_std[valid_mask] * window_std[valid_mask]
    )

    # Average across timepoints
    avg_correlation = np.mean(correlations)

    return avg_correlation

def detect_bursts_by_correlation(
    data: np.ndarray,
    pattern: np.ndarray,
    W_smooth: np.ndarray,
    sfreq: float = 250.0,
    step: Optional[int] = None
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
    threshold : float, default=0.825
        Correlation threshold for detection.
        Positive bursts: correlation > threshold
        Negative bursts: correlation < -threshold
        Ignored when return_all=True.
    sfreq : float, default=250.0
        Sampling frequency in Hz (for rate calculations)
    use_optimized : bool, default=True
        If True, use optimized implementation with pre-smoothing and
        vectorized correlations (much faster for large datasets).
        If False, use original implementation.
    return_all : bool, default=False
        If True, return correlations for ALL sliding window positions
        without threshold filtering. This allows downstream code to apply
        any threshold as a free parameter.
        If False, return only threshold-filtered detections (original behavior).
    step : int or None, default=None
        Step size for sliding window in samples.
        None or 1: evaluate at every sample (overlapping windows).
        Set to pattern length for non-overlapping windows, e.g.
        ``step=pattern.shape[1]``.

    Returns
    -------
    burst_info : dict
        When return_all=False (default):
            - 'n_positive': Number of positive detections
            - 'n_negative': Number of negative detections
            - 'rate_positive': Positive detection rate per second
            - 'times': Array of detection times in seconds
            - 'polarities': Array of +1 (positive) or -1 (negative)
            - 'correlations': Array of correlation values at each detection

        When return_all=True:
            - 'times': Array of ALL window start times in seconds
            - 'polarities': Array of +1/-1 (sign of correlation) for every window
            - 'correlations': Array of raw correlation values for every window
            - 'n_windows': Total number of sliding windows

    Examples
    --------
    >>> data = np.random.randn(10, 1000)
    >>> pattern = np.random.randn(10, 20)
    >>> W_smooth = np.eye(10)
    >>> result = detect_bursts_by_correlation(data, pattern, W_smooth, threshold=0.825)
    >>> result['n_positive'] >= 0
    True

    >>> # Return all correlations for downstream threshold tuning
    >>> result = detect_bursts_by_correlation(data, pattern, W_smooth, return_all=True)
    >>> len(result['correlations']) == result['n_windows']
    True
    """
    n_electrodes, n_samples = data.shape
    n_electrodes_p, pattern_length = pattern.shape

    if n_electrodes != n_electrodes_p:
        raise ValueError(f"Electrode count mismatch: data has {n_electrodes}, pattern has {n_electrodes_p}")

    if step is None:
        step = 1

    # OPTIMIZATION 1: Pre-smooth data and pattern once
    # Instead of applying W_smooth @ frame for every window and every timepoint,
    # we smooth the entire arrays upfront
    data_smooth = W_smooth @ data  # Shape: (n_electrodes, n_samples)
    pattern_smooth = W_smooth @ pattern  # Shape: (n_electrodes, pattern_length)

    # Sliding window correlation with pre-smoothed data
    positions = np.arange(0, n_samples - pattern_length + 1, step)
    n_windows = len(positions)
    correlations = np.empty(n_windows)

    for idx, start in enumerate(positions):
        end = start + pattern_length
        window_smooth = data_smooth[:, start:end]

        # OPTIMIZATION 2: Vectorized frame correlations
        corr = _vectorized_frame_correlations(pattern_smooth, window_smooth)
        correlations[idx] = corr

    all_times = positions / sfreq
    all_polarities = np.where(correlations >= 0, 1, -1).astype(np.int8)
    return {
        'times': all_times,
        'polarities': all_polarities,
        'correlations': correlations,
        'n_windows': len(positions)
    }


def detect_bursts_in_clusters(
    cluster_activation_times: Dict,
    cluster_durations: Dict,
    dataset_names: List[str],
    tolerance_samples: int = 1,
    sfreq: float = 250.0,
    min_burst_length: int = 2,
    correlation_threshold: float = 0.825
) -> Dict:
    """
    Detect bursts from cluster activation times.

    A burst is defined as 2+ cluster activations where each subsequent activation
    starts within `tolerance_samples` frames after the previous activation ends.
    This groups temporally adjacent pattern occurrences into burst events.

    Parameters
    ----------
    cluster_activation_times : dict
        {cluster_id: [(dataset_idx, times, polarities, correlations), ...]}
        Output from detect_bursts_by_correlation() for each cluster and dataset
    cluster_durations : dict
        {cluster_id: duration_in_seconds}
        Duration of each cluster pattern in seconds
    dataset_names : list of str
        List of dataset names/filenames corresponding to dataset indices
    tolerance_samples : int, default=1
        Maximum number of samples (frames) after the end of one activation
        where the next activation can start to be considered part of the same burst.
        At 250Hz, 1 sample = 4ms tolerance.
    sfreq : float, default=250.0
        Sampling frequency in Hz (to convert tolerance_samples to seconds)
    min_burst_length : int, default=2
        Minimum number of activations required for a burst
    correlation_threshold : float, default=0.825
        Minimum |correlation| to include activation in burst detection

    Returns
    -------
    bursts_by_cluster : dict
        {cluster_id: [burst_1, burst_2, ...]} where each burst contains:
        - 'dataset': dataset name
        - 'start_time': burst start time (seconds)
        - 'end_time': burst end time (seconds)
        - 'length': number of activations in burst

    Examples
    --------
    >>> cluster_activation_times = {
    ...     0: [(0, np.array([1.0, 1.08]), np.array([1, 1]), np.array([0.9, 0.85]))]
    ... }
    >>> cluster_durations = {0: 0.08}  # 20 samples at 250Hz
    >>> dataset_names = ['dataset_1.set']
    >>> bursts = detect_bursts_in_clusters(
    ...     cluster_activation_times, cluster_durations, dataset_names,
    ...     tolerance_samples=2, sfreq=250.0
    ... )
    >>> len(bursts[0]) > 0
    True
    """
    bursts_by_cluster = {}

    # Convert tolerance from samples to seconds
    tolerance_sec = tolerance_samples / sfreq

    for cluster_id, activations in cluster_activation_times.items():
        cluster_bursts = []
        cluster_duration = cluster_durations[cluster_id]

        for dataset_idx, times, polarities, correlations in activations:
            # Filter by correlation threshold
            high_corr_mask = np.abs(correlations) > correlation_threshold
            times_filtered = times[high_corr_mask]
            polarities_filtered = polarities[high_corr_mask]
            correlations_filtered = correlations[high_corr_mask]

            if len(times_filtered) < min_burst_length:
                continue

            # Combined inline suppression + burst detection.
            # No pre-suppression pass: overlapping duplicates are only
            # skipped when they fall between two connected activations
            # (i.e., suppression is a consequence of confirming a burst,
            # not a precondition). If a burst attempt fails (too short),
            # the next detection is tried as a new start — duplicates
            # are not discarded until a connection is confirmed.
            # All times in seconds (cluster_duration, tolerance_sec).
            i = 0
            while i < len(times_filtered):
                # Start new burst attempt from detection i
                current_burst = {
                    'times': [times_filtered[i]],
                    'polarities': [polarities_filtered[i]],
                    'correlations': [correlations_filtered[i]]
                }

                j = i + 1
                while j < len(times_filtered):
                    last_time = current_burst['times'][-1]
                    connect_deadline = last_time + cluster_duration + tolerance_sec

                    if times_filtered[j] < last_time + cluster_duration:
                        # Overlapping duplicate — skip
                        j += 1
                    elif times_filtered[j] <= connect_deadline:
                        # Connected — add to burst, advance last_time
                        current_burst['times'].append(times_filtered[j])
                        current_burst['polarities'].append(polarities_filtered[j])
                        current_burst['correlations'].append(correlations_filtered[j])
                        j += 1
                    else:
                        # Beyond deadline — burst is over
                        break

                if len(current_burst['times']) >= min_burst_length:
                    burst = _finalize_burst(
                        current_burst, cluster_duration,
                        dataset_names[dataset_idx]
                    )
                    cluster_bursts.append(burst)
                    # Skip past all detections consumed by this burst
                    i = j
                else:
                    # Burst too short — try next detection as start
                    i += 1

        bursts_by_cluster[cluster_id] = cluster_bursts

    return bursts_by_cluster


def _finalize_burst(burst_data: Dict, cluster_duration: float, dataset_name: str) -> Dict:
    """
    Convert burst data into final burst dictionary.

    Parameters
    ----------
    burst_data : dict
        Dictionary with 'times', 'polarities', 'correlations' lists
    cluster_duration : float
        Duration of cluster pattern in seconds
    dataset_name : str
        Name of the dataset this burst comes from

    Returns
    -------
    burst : dict
        Finalized burst with dataset, start_time, end_time, length
    """
    pols = burst_data['polarities']
    n = len(pols)

    if n <= 1:
        polarity = 'single'
    else:
        all_same = all(p == pols[0] for p in pols)
        all_alternating = all(pols[j] * pols[j + 1] < 0 for j in range(n - 1))

        if all_same:
            polarity = 'stable'
        elif all_alternating:
            polarity = 'switching'
        elif n >= 4 and n % 2 == 0:
            half = n // 2
            first_uniform = all(p == pols[0] for p in pols[:half])
            second_uniform = all(p == pols[half] for p in pols[half:])
            opposite = pols[0] != pols[half]
            if first_uniform and second_uniform and opposite:
                polarity = 'long_switching'
            else:
                polarity = 'mixed'
        else:
            polarity = 'mixed'

    return {
        'dataset': dataset_name,
        'start_time': burst_data['times'][0],
        'end_time': burst_data['times'][-1] + cluster_duration,
        'length': n,
        'polarity': polarity
    }




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