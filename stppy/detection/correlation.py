"""
Correlation-based sequence detection using spatiotemporal pattern matching.

This module provides an alternative detection approach that matches data against
template patterns using spatial correlation. Unlike threshold-based detection,
this identifies recurring patterns based on similarity to cluster templates.
"""

import numpy as np
from typing import Dict, List, Optional


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
        Pattern array
    window_smooth : np.ndarray, shape (n_electrodes, n_timepoints)
        Data window array

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

def detect_sequences_by_correlation(
    data: np.ndarray,
    pattern: np.ndarray,
    sfreq: float = 250.0,
    step: Optional[int] = None
) -> Dict:
    """
    Detect sequences by sliding pattern correlation across continuous data.

    Performs sliding window correlation to identify moments where the data
    strongly matches (or anti-matches) a template pattern. Useful for detecting
    recurring cluster patterns in continuous recordings.

    Returns correlations for ALL sliding window positions. Apply a threshold
    downstream via detect_sequences_in_clusters().

    Parameters
    ----------
    data : np.ndarray, shape (n_electrodes, n_samples)
        Continuous EEG/MEG data
    pattern : np.ndarray, shape (n_electrodes, n_timepoints)
        Template pattern to search for (e.g., cluster average)
    sfreq : float, default=250.0
        Sampling frequency in Hz
    step : int or None, default=None
        Step size for sliding window in samples.
        None or 1: evaluate at every sample (overlapping windows).
        Set to pattern length for non-overlapping windows, e.g.
        ``step=pattern.shape[1]``.

    Returns
    -------
    sequence_info : dict
        - 'times': Array of ALL window start times in seconds
        - 'polarities': Array of +1/-1 (sign of correlation) for every window
        - 'correlations': Array of raw correlation values for every window
        - 'n_windows': Total number of sliding windows

    Examples
    --------
    >>> data = np.random.randn(10, 1000)
    >>> pattern = np.random.randn(10, 20)
    >>> result = detect_sequences_by_correlation(data, pattern)
    >>> len(result['correlations']) == result['n_windows']
    True
    """
    n_electrodes, n_samples = data.shape
    n_electrodes_p, pattern_length = pattern.shape

    if n_electrodes != n_electrodes_p:
        raise ValueError(f"Electrode count mismatch: data has {n_electrodes}, pattern has {n_electrodes_p}")

    if step is None:
        step = 1

    positions = np.arange(0, n_samples - pattern_length + 1, step)
    n_windows = len(positions)
    correlations = np.empty(n_windows)

    for idx, start in enumerate(positions):
        end = start + pattern_length
        corr = _vectorized_frame_correlations(pattern, data[:, start:end])
        correlations[idx] = corr

    all_times = positions / sfreq
    all_polarities = np.where(correlations >= 0, 1, -1).astype(np.int8)
    return {
        'times': all_times,
        'polarities': all_polarities,
        'correlations': correlations,
        'n_windows': len(positions)
    }


def detect_sequences_in_clusters(
    cluster_activation_times: Dict,
    cluster_durations: Dict,
    dataset_names: List[str],
    tolerance_samples: int = 1,
    sfreq: float = 250.0,
    min_sequence_length: int = 2,
    correlation_threshold: float = 0.825
) -> Dict:
    """
    Detect sequences from cluster activation times.

    A sequence is defined as 2+ cluster activations where each subsequent activation
    starts within `tolerance_samples` frames after the previous activation ends.
    This groups temporally adjacent pattern occurrences into sequences.

    Parameters
    ----------
    cluster_activation_times : dict
        {cluster_id: [(dataset_idx, times, polarities, correlations), ...]}
        Output from detect_sequences_by_correlation() for each cluster and dataset.
        {cluster_id: [(dataset_idx, times, polarities, correlations), ...]}
    cluster_durations : dict
        {cluster_id: duration_in_seconds}
        Duration of each cluster pattern in seconds
    dataset_names : list of str
        List of dataset names/filenames corresponding to dataset indices
    tolerance_samples : int, default=1
        Maximum number of samples (frames) after the end of one activation
        where the next activation can start to be considered part of the same sequence.
        At 250Hz, 1 sample = 4ms tolerance.
    sfreq : float, default=250.0
        Sampling frequency in Hz (to convert tolerance_samples to seconds)
    min_sequence_length : int, default=2
        Minimum number of activations required for a sequence
    correlation_threshold : float, default=0.825
        Minimum |correlation| to include activation in sequence detection

    Returns
    -------
    sequences_by_cluster : dict
        {cluster_id: [sequence_1, sequence_2, ...]} where each sequence contains:
        - 'dataset': dataset name
        - 'start_time': sequence start time (seconds)
        - 'end_time': sequence end time (seconds)
        - 'length': number of activations in sequence

    Examples
    --------
    >>> cluster_activation_times = {
    ...     0: [(0, np.array([1.0, 1.08]), np.array([1, 1]), np.array([0.9, 0.85]))]
    ... }
    >>> cluster_durations = {0: 0.08}  # 20 samples at 250Hz
    >>> dataset_names = ['dataset_1.set']
    >>> sequences = detect_sequences_in_clusters(
    ...     cluster_activation_times, cluster_durations, dataset_names,
    ...     tolerance_samples=2, sfreq=250.0
    ... )
    >>> len(sequences[0]) > 0
    True
    """
    sequences_by_cluster = {}

    # Convert tolerance from samples to seconds
    tolerance_sec = tolerance_samples / sfreq

    for cluster_id, activations in cluster_activation_times.items():
        cluster_sequences = []
        cluster_duration = cluster_durations[cluster_id]

        for dataset_idx, times, polarities, correlations in activations:
            print(f"Processing cluster {cluster_id} out of {len(cluster_activation_times)}")
            # Filter by correlation threshold
            high_corr_mask = np.abs(correlations) > correlation_threshold
            times_filtered = times[high_corr_mask]
            polarities_filtered = polarities[high_corr_mask]
            correlations_filtered = correlations[high_corr_mask]

            if len(times_filtered) < min_sequence_length:
                continue

            # Combined inline suppression + sequence detection.
            # No pre-suppression pass: overlapping duplicates are only
            # skipped when they fall between two connected activations
            # (i.e., suppression is a consequence of confirming a sequence,
            # not a precondition). If a sequence attempt fails (too short),
            # the next detection is tried as a new start — duplicates
            # are not discarded until a connection is confirmed.
            # All times in seconds (cluster_duration, tolerance_sec).
            i = 0
            while i < len(times_filtered):
                # Start new sequence attempt from detection i
                current_sequence = {
                    'times': [times_filtered[i]],
                    'polarities': [polarities_filtered[i]],
                    'correlations': [correlations_filtered[i]]
                }

                j = i + 1
                while j < len(times_filtered):
                    last_time = current_sequence['times'][-1]
                    connect_deadline = last_time + cluster_duration + tolerance_sec

                    if times_filtered[j] < last_time + cluster_duration:
                        # Overlapping duplicate — skip
                        j += 1
                    elif times_filtered[j] <= connect_deadline:
                        # Connected — add to sequence, advance last_time
                        current_sequence['times'].append(times_filtered[j])
                        current_sequence['polarities'].append(polarities_filtered[j])
                        current_sequence['correlations'].append(correlations_filtered[j])
                        j += 1
                    else:
                        # Beyond deadline — sequence is over
                        break

                if len(current_sequence['times']) >= min_sequence_length:
                    seq = _finalize_sequence(
                        current_sequence, cluster_duration,
                        dataset_names[dataset_idx]
                    )
                    cluster_sequences.append(seq)
                    # Skip past all detections consumed by this sequence
                    i = j
                else:
                    # Sequence too short — try next detection as start
                    i += 1

        sequences_by_cluster[cluster_id] = cluster_sequences

    return sequences_by_cluster


def _finalize_sequence(sequence_data: Dict, cluster_duration: float, dataset_name: str) -> Dict:
    """
    Convert sequence data into final sequence dictionary.

    Parameters
    ----------
    sequence_data : dict
        Dictionary with 'times', 'polarities', 'correlations' lists
    cluster_duration : float
        Duration of cluster pattern in seconds
    dataset_name : str
        Name of the dataset this sequence comes from

    Returns
    -------
    sequence : dict
        Finalized sequence with dataset, start_time, end_time, length
    """
    pols = sequence_data['polarities']
    n = len(pols)

    if n <= 1:
        polarity = 'single'
    else:
        all_same = all(p == pols[0] for p in pols)
        all_alternating = all(pols[j] * pols[j + 1] < 0 for j in range(n - 1))

        if all_same:
            polarity = 'standing'
        elif all_alternating:
            polarity = 'rotating'
        elif n >= 4 and n % 2 == 0:
            half = n // 2
            first_uniform = all(p == pols[0] for p in pols[:half])
            second_uniform = all(p == pols[half] for p in pols[half:])
            opposite = pols[0] != pols[half]
            if first_uniform and second_uniform and opposite:
                polarity = 'long_rotating'
            else:
                polarity = 'mixed'
        else:
            polarity = 'mixed'

    return {
        'dataset': dataset_name,
        'start_time': sequence_data['times'][0],
        'end_time': sequence_data['times'][-1] + cluster_duration,
        'length': n,
        'polarity': polarity
    }