"""
Simple metadata tracking for avalanche datasets.

Tracks only essential information: dataset name, timing, and activity.
Users can parse additional metadata from dataset names as needed.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict


def create_avalanche_index(
    all_avalanches: List[List[np.ndarray]],
    all_avalanche_times: List[List[Tuple[int, int]]],
    dataset_names: List[str]
) -> List[Dict]:
    """
    Create an index linking each avalanche to its dataset, timing, and name.

    Parameters
    ----------
    all_avalanches : list of lists
        List containing avalanche lists for each dataset.
        Structure: [[dataset0_avalanche0, dataset0_avalanche1, ...],
                    [dataset1_avalanche0, ...], ...]
    all_avalanche_times : list of lists
        List containing avalanche timing lists for each dataset.
        Each timing is a tuple (start_idx, end_idx)
    dataset_names : list of str
        Name for each dataset (typically filename).
        Example: ['subject01_rest.set', 'subject02_task.set', ...]

    Returns
    -------
    avalanche_index : list of dicts
        List where each entry describes one avalanche:
        {
            'dataset_idx': int,       # Which dataset (0-indexed)
            'dataset_name': str,      # Dataset name (e.g., filename)
            'avalanche_idx': int,     # Which avalanche in that dataset
            'start_time': int,        # Start sample index
            'end_time': int,          # End sample index
            'duration': int           # Duration in samples
        }

    Examples
    --------
    >>> # Create index with dataset names
    >>> dataset_names = ['S01_rest.set', 'S02_task.set']
    >>> index = create_avalanche_index(all_avalanches, all_avalanche_times, dataset_names)
    >>>
    >>> # Users can parse metadata from names
    >>> for entry in index:
    ...     subject = entry['dataset_name'].split('_')[0]  # Extract 'S01'
    ...     condition = entry['dataset_name'].split('_')[1].split('.')[0]  # Extract 'rest'
    """
    if len(all_avalanches) != len(all_avalanche_times):
        raise ValueError(
            f"Number of avalanche lists ({len(all_avalanches)}) must match "
            f"number of timing lists ({len(all_avalanche_times)})"
        )

    if len(dataset_names) != len(all_avalanches):
        raise ValueError(
            f"Number of dataset names ({len(dataset_names)}) must match "
            f"number of avalanche lists ({len(all_avalanches)})"
        )

    avalanche_index = []

    for dataset_idx, (dataset_avalanches, dataset_times, dataset_name) in enumerate(
            zip(all_avalanches, all_avalanche_times, dataset_names)):

        if len(dataset_avalanches) != len(dataset_times):
            raise ValueError(
                f"Dataset {dataset_idx} ({dataset_name}): Number of avalanches "
                f"({len(dataset_avalanches)}) must match number of timing entries "
                f"({len(dataset_times)})"
            )

        for avalanche_idx, (avalanche, avalanche_time) in enumerate(
                zip(dataset_avalanches, dataset_times)):

            start_time, end_time = avalanche_time
            duration = end_time - start_time

            entry = {
                'dataset_idx': dataset_idx,
                'dataset_name': dataset_name,
                'avalanche_idx': avalanche_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }

            avalanche_index.append(entry)

    print(f"Created index for {len(avalanche_index)} avalanches "
          f"across {len(all_avalanches)} datasets")

    return avalanche_index


def analyze_cluster_composition(
    clusters: np.ndarray,
    avalanche_index: List[Dict]
) -> Dict[int, Dict]:
    """
    Analyze which datasets contribute avalanches to each cluster.

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments for each avalanche (from hierarchical clustering)
    avalanche_index : list of dicts
        Avalanche index from create_avalanche_index()

    Returns
    -------
    composition : dict
        Dictionary mapping cluster IDs to their composition:
        {
            cluster_id: {
                'size': 150,
                'datasets': {'S01_rest.set': 90, 'S02_task.set': 60},
                'datasets_pct': {'S01_rest.set': 60.0, 'S02_task.set': 40.0}
            }
        }

    Examples
    --------
    >>> composition = analyze_cluster_composition(clusters, avalanche_index)
    >>> for cluster_id, stats in composition.items():
    ...     print(f"Cluster {cluster_id}: {stats['size']} avalanches")
    ...     print(f"  Datasets: {stats['datasets']}")
    """
    unique_clusters = np.unique(clusters)
    # Filter out noise cluster (-1) if present
    valid_clusters = unique_clusters[unique_clusters != -1]

    composition = {}

    for cluster_id in valid_clusters:
        # Get indices of avalanches in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)

        # Count avalanches by dataset
        dataset_counts = defaultdict(int)

        for idx in cluster_indices:
            if idx < len(avalanche_index):
                dataset_name = avalanche_index[idx].get('dataset_name')
                if dataset_name:
                    dataset_counts[dataset_name] += 1

        # Calculate percentages
        total = sum(dataset_counts.values())
        dataset_pct = {k: (v/total)*100 for k, v in dataset_counts.items()} if total > 0 else {}

        composition[int(cluster_id)] = {
            'size': cluster_size,
            'datasets': dict(dataset_counts),
            'datasets_pct': dataset_pct
        }

    # Print summary
    print(f"\n=== Cluster Composition Analysis ===")
    print(f"Analyzed {len(valid_clusters)} clusters")
    for cluster_id in sorted(composition.keys()):
        stats = composition[cluster_id]
        print(f"\nCluster {cluster_id}: {stats['size']} avalanches")
        if stats['datasets']:
            counts_str = ', '.join([f"{k}: {v}" for k, v in sorted(stats['datasets'].items())])
            print(f"  Datasets: {counts_str}")

    return composition


def summarize_burst_statistics(
    burst_stats: List[Dict]
) -> Dict[str, Dict]:
    """
    Summarize burst detection statistics by dataset.

    Parameters
    ----------
    burst_stats : list of dicts
        List of burst statistics for each dataset:
        [
            {
                'dataset_name': 'S01_rest.set',
                'n_positive': 10,
                'n_negative': 5,
                'rate': 0.5
            },
            ...
        ]

    Returns
    -------
    summary : dict
        Statistics grouped by dataset name:
        {
            'S01_rest.set': {
                'n_bursts': 10,
                'rate': 0.5
            },
            'S02_task.set': {...}
        }

    Examples
    --------
    >>> summary = summarize_burst_statistics(burst_stats)
    >>> for dataset, stats in summary.items():
    ...     print(f"{dataset}: {stats['n_bursts']} bursts (rate: {stats['rate']:.2f})")
    """
    summary = {}

    for entry in burst_stats:
        dataset_name = entry.get('dataset_name')
        if dataset_name:
            summary[dataset_name] = {
                'n_bursts': entry.get('n_positive', 0),
                'rate': entry.get('rate', 0.0)
            }

    # Print summary
    print(f"\n=== Burst Statistics Summary ===")
    for dataset_name in sorted(summary.keys()):
        stats = summary[dataset_name]
        print(f"{dataset_name}: {stats['n_bursts']} bursts (rate: {stats['rate']:.2f} bursts/sec)")

    return summary


def parse_dataset_metadata(
    file_paths: List[str],
    subject_delimiter: str = '_',
    subject_position: int = 0
) -> List[Dict]:
    """
    Parse dataset metadata from file paths.

    Extracts drug condition, task condition, and subject ID from
    the directory structure and filename. Expects the path layout::

        .../drug_condition/task_condition/subject_task.set

    Parameters
    ----------
    file_paths : list of str
        Full file paths to EEG datasets.
        Example: ['/data/Control/Meditation/S01_meditation.set', ...]
    subject_delimiter : str, default='_'
        Character used to split the filename for extracting subject ID
    subject_position : int, default=0
        Position of subject ID token after splitting filename by delimiter.
        0 means the first token (e.g., 'S01' from 'S01_meditation.set')

    Returns
    -------
    metadata : list of dicts
        One dict per file path, each containing:
        - 'filepath': Original full path
        - 'filename': Base filename (e.g., 'S01_meditation.set')
        - 'drug': Drug condition from grandparent directory (e.g., 'Control')
        - 'task': Task condition from parent directory (e.g., 'Meditation')
        - 'subject': Subject ID from filename (e.g., 'S01')

    Examples
    --------
    >>> paths = [
    ...     '/data/Control/Meditation/S01_meditation.set',
    ...     '/data/Psilocybin/Music/S02_music.set'
    ... ]
    >>> metadata = parse_dataset_metadata(paths)
    >>> metadata[0]['drug']
    'Control'
    >>> metadata[0]['task']
    'Meditation'
    >>> metadata[0]['subject']
    'S01'
    """
    metadata = []

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        parent_dir = os.path.dirname(filepath)
        task = os.path.basename(parent_dir)
        drug = os.path.basename(os.path.dirname(parent_dir))

        # Extract subject ID from filename
        name_without_ext = os.path.splitext(filename)[0]
        tokens = name_without_ext.split(subject_delimiter)
        subject = tokens[subject_position] if subject_position < len(tokens) else name_without_ext

        metadata.append({
            'filepath': filepath,
            'filename': filename,
            'drug': drug,
            'task': task,
            'subject': subject
        })

    # Print summary
    drugs = set(m['drug'] for m in metadata)
    tasks = set(m['task'] for m in metadata)
    subjects = set(m['subject'] for m in metadata)
    print(f"Parsed metadata for {len(metadata)} datasets")
    print(f"  Drug conditions: {sorted(drugs)}")
    print(f"  Task conditions: {sorted(tasks)}")
    print(f"  Subjects: {len(subjects)} unique ({sorted(subjects)[:5]}{'...' if len(subjects) > 5 else ''})")

    return metadata
