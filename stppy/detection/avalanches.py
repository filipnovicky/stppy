"""
Avalanche detection and filtering functions.
"""

import numpy as np
from typing import List, Tuple


def create_avalanche_filter(datasets: List[np.ndarray],
                           crossover_thresholds: List[float]) -> List[np.ndarray]:
    """
    Create filter masks based on crossover thresholds for multiple datasets.

    Parameters
    ----------
    datasets : list of arrays
        List of EEG datasets, each with shape (n_channels, n_samples)
    crossover_thresholds : list
        Crossover points from threshold analysis

    Returns
    -------
    filter_masks : list of arrays
        List of boolean masks for each dataset, each with shape (n_samples,)
        indicating which time points have activity exceeding thresholds
    """

    filter_masks = []

    min_threshold = min(crossover_thresholds)
    max_threshold = max(crossover_thresholds)

    for i, dataset in enumerate(datasets):
        dataset = np.asarray(dataset)

        if dataset.ndim == 1:
            dataset = dataset.reshape(1, -1)
        elif dataset.ndim == 2:
            if dataset.shape[0] > dataset.shape[1]:
                dataset = dataset.T

        # Normalize data for threshold detection
        means = np.mean(dataset, axis=1, keepdims=True)
        stds = np.std(dataset, axis=1, keepdims=True)
        stds[stds == 0] = 1
        normalized_data = (dataset - means) / stds

        # Create filter mask: any electrode exceeding thresholds at each time point
        filter_mask = np.any((normalized_data <= min_threshold) | (normalized_data >= max_threshold), axis=0)

        filter_masks.append(filter_mask)

    return filter_masks


def detect_avalanches(
    datasets: List[np.ndarray],
    filter_masks: List[np.ndarray],
    crossover_thresholds: List[float],
    min_duration: int = 2,
    verbose: bool = True
) -> Tuple[List[List[np.ndarray]], List[List[Tuple[int, int]]]]:
    """
    Detect avalanches by tracking time indices and including all electrodes.

    Parameters
    ----------
    datasets : list of arrays
        List of original EEG datasets (real values), each with shape (n_channels, n_samples)
    filter_masks : list of arrays
        List of boolean masks indicating avalanche time points for each dataset
    crossover_thresholds : list
        Crossover points for avalanche detection
    min_duration : int
        Minimum duration for valid avalanches
    verbose : bool
        Print detection details

    Returns
    -------
    all_avalanches : list of lists
        List containing avalanche lists for each dataset. Each inner list contains
        avalanche matrices with original data values, shape (n_electrodes, duration)
    all_avalanche_times : list of lists
        List containing avalanche timing lists for each dataset. Each inner list contains
        tuples of (start_idx, end_idx) indicating when each avalanche occurs
    """

    if len(datasets) != len(filter_masks):
        raise ValueError(f"Number of datasets ({len(datasets)}) must match "
                        f"number of filter masks ({len(filter_masks)})")

    all_avalanches = []
    all_avalanche_times = []

    if verbose:
        print(f"Detecting avalanches for {len(datasets)} datasets...")

    min_threshold = min(crossover_thresholds)
    max_threshold = max(crossover_thresholds)

    for dataset_idx, (dataset, filter_mask) in enumerate(zip(datasets, filter_masks)):
        dataset = np.asarray(dataset)

        if dataset.ndim == 1:
            dataset = dataset.reshape(1, -1)
        elif dataset.ndim == 2:
            if dataset.shape[0] > dataset.shape[1]:
                dataset = dataset.T

        n_electrodes, n_samples = dataset.shape

        if np.sum(filter_mask) == 0:
            if verbose:
                print(f"Dataset {dataset_idx + 1}: Warning: No data samples available for avalanche detection")

            all_avalanches.append([])
            all_avalanche_times.append([])
            continue

        if verbose:
            print(f"\nDataset {dataset_idx + 1}:")
            print(f"  Data shape: {dataset.shape} (n_electrodes, n_samples)")
            print(f"  Active time points: {np.sum(filter_mask)} ({np.sum(filter_mask)/n_samples*100:.2f}%)")

        # Find runs of consecutive True values in the filter_mask
        padded_mask = np.concatenate(([False], filter_mask, [False]))
        diff = np.diff(padded_mask.astype(int))

        run_starts = np.where(diff == 1)[0]
        run_ends = np.where(diff == -1)[0]

        if verbose:
            print(f"  Found {len(run_starts)} continuous runs")

        avalanches = []
        avalanche_times = []

        for start_idx, end_idx in zip(run_starts, run_ends):
            duration = end_idx - start_idx

            if duration >= min_duration:
                time_slice = dataset[:, start_idx:end_idx]
                means = np.mean(dataset, axis=1, keepdims=True)
                stds = np.std(dataset, axis=1, keepdims=True)
                stds[stds == 0] = 1
                normalized_slice = (time_slice - means) / stds

                # Mask for electrodes exceeding thresholds
                electrode_mask = (normalized_slice <= min_threshold) | (normalized_slice >= max_threshold)

                # Zero out non-thresholded data points
                avalanche_matrix = time_slice.copy()
                avalanche_matrix[~electrode_mask] = 0

                avalanches.append(avalanche_matrix)
                avalanche_times.append((start_idx, end_idx))

        if verbose:
            print(f"  Avalanche Detection Results:")
            print(f"    Total avalanches detected: {len(run_starts)}")
            print(f"    Avalanches kept (≥{min_duration} steps): {len(avalanches)}")
            if len(avalanches) > 0:
                durations = [end - start for start, end in avalanche_times]
                print(f"    Duration range: {min(durations)} - {max(durations)} time steps")

        all_avalanches.append(avalanches)
        all_avalanche_times.append(avalanche_times)

    return all_avalanches, all_avalanche_times


def convert_to_peaks(filtered_data: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """
    Convert avalanches to peak representation for all participants.

    Parameters
    ----------
    filtered_data : list of lists
        List where each element is a participant, and each participant is a list
        of avalanches. Each avalanche is a 2D array with shape (electrodes, duration).

    Returns
    -------
    peak_avalanches : list of lists
        Same structure as input, but with peak representation where:
        - Positive values become 1
        - Negative values become -1
        - Zero values remain 0
    """
    peak_avalanches = []

    for participant_idx, participant_avalanches in enumerate(filtered_data):
        participant_peaks = []

        for avalanche_idx, avalanche in enumerate(participant_avalanches):
            peak_avalanche = np.sign(avalanche).astype(avalanche.dtype)
            participant_peaks.append(peak_avalanche)

        peak_avalanches.append(participant_peaks)
        print(f"Processed participant {participant_idx}: {len(participant_avalanches)} avalanches")

    return peak_avalanches
