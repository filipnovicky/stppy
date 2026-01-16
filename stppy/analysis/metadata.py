"""
Universal metadata analysis utilities for avalanche datasets.

Provides generic functions that work with any user-defined metadata fields.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict


def create_avalanche_index(
    all_avalanches: List[List[np.ndarray]],
    all_avalanche_times: List[List[Tuple[int, int]]],
    dataset_metadata: Optional[List[Dict[str, Any]]] = None
) -> List[Dict]:
    """
    Create a universal index linking each avalanche to its dataset and timing.
    Does NOT assume any specific metadata fields - user provides their own.

    Parameters
    ----------
    all_avalanches : list of lists
        List containing avalanche lists for each dataset.
        Structure: [[dataset0_avalanche0, dataset0_avalanche1, ...],
                    [dataset1_avalanche0, ...], ...]
    all_avalanche_times : list of lists
        List containing avalanche timing lists for each dataset.
        Each timing is a tuple (start_idx, end_idx)
    dataset_metadata : list of dicts, optional
        User-defined metadata for each dataset. Can contain ANY fields.
        Example: [{'subject': 'S01', 'condition': 'rest'},
                  {'subject': 'S02', 'session': 2}, ...]

    Returns
    -------
    avalanche_index : list of dicts
        List where each entry describes one avalanche:
        {
            'dataset_idx': int,      # Which dataset (0-indexed)
            'avalanche_idx': int,    # Which avalanche in that dataset
            'start_time': int,       # Start sample index
            'end_time': int,         # End sample index
            'duration': int,         # Duration in samples
            'metadata': dict or None # User's metadata for this dataset
        }

    Examples
    --------
    >>> # Minimal usage - just link avalanches to datasets
    >>> index = create_avalanche_index(all_avalanches, all_avalanche_times)
    >>>
    >>> # With custom metadata
    >>> metadata = [
    ...     {'subject': 'S01', 'condition': 'rest', 'age': 25},
    ...     {'subject': 'S02', 'condition': 'task', 'age': 30}
    ... ]
    >>> index = create_avalanche_index(all_avalanches, all_avalanche_times, metadata)
    """

    if len(all_avalanches) != len(all_avalanche_times):
        raise ValueError(
            f"Number of avalanche lists ({len(all_avalanches)}) must match "
            f"number of timing lists ({len(all_avalanche_times)})"
        )

    if dataset_metadata is not None and len(dataset_metadata) != len(all_avalanches):
        raise ValueError(
            f"Number of metadata entries ({len(dataset_metadata)}) must match "
            f"number of avalanche lists ({len(all_avalanches)})"
        )

    avalanche_index = []

    for dataset_idx, (dataset_avalanches, dataset_times) in enumerate(
            zip(all_avalanches, all_avalanche_times)):

        if len(dataset_avalanches) != len(dataset_times):
            raise ValueError(
                f"Dataset {dataset_idx}: Number of avalanches "
                f"({len(dataset_avalanches)}) must match number of timing entries "
                f"({len(dataset_times)})"
            )

        # Get metadata for this dataset (if provided)
        metadata = dataset_metadata[dataset_idx] if dataset_metadata else None

        for avalanche_idx, (avalanche, avalanche_time) in enumerate(
                zip(dataset_avalanches, dataset_times)):

            start_time, end_time = avalanche_time
            duration = end_time - start_time

            entry = {
                'dataset_idx': dataset_idx,
                'avalanche_idx': avalanche_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'metadata': metadata
            }

            avalanche_index.append(entry)

    print(f"Created index for {len(avalanche_index)} avalanches "
          f"across {len(all_avalanches)} datasets")

    return avalanche_index


def query_avalanches(
    avalanche_index: List[Dict],
    field: Optional[str] = None,
    value: Optional[Any] = None,
    filter_func: Optional[callable] = None
) -> List[int]:
    """
    Query avalanche index to find avalanches matching criteria.

    Parameters
    ----------
    avalanche_index : list of dicts
        Avalanche index from create_avalanche_index()
    field : str, optional
        Metadata field to query (e.g., 'subject', 'condition')
    value : any, optional
        Value to match for the field
    filter_func : callable, optional
        Custom filter function: func(entry) -> bool
        Takes an avalanche entry dict and returns True/False

    Returns
    -------
    matching_indices : list of int
        Indices of avalanches that match the criteria

    Examples
    --------
    >>> # Find all avalanches from subject 'S01'
    >>> indices = query_avalanches(index, field='subject', value='S01')
    >>>
    >>> # Find avalanches with duration > 20 samples
    >>> indices = query_avalanches(
    ...     index,
    ...     filter_func=lambda x: x['duration'] > 20
    ... )
    >>>
    >>> # Complex query: subject S01 AND condition 'rest'
    >>> indices = query_avalanches(
    ...     index,
    ...     filter_func=lambda x: x['metadata']['subject'] == 'S01' and
    ...                           x['metadata']['condition'] == 'rest'
    ... )
    """

    matching_indices = []

    for idx, entry in enumerate(avalanche_index):
        match = False

        if filter_func is not None:
            # Use custom filter function
            match = filter_func(entry)
        elif field is not None and value is not None:
            # Simple field=value matching
            metadata = entry.get('metadata')
            if metadata and metadata.get(field) == value:
                match = True
        else:
            # No criteria - return all
            match = True

        if match:
            matching_indices.append(idx)

    print(f"Found {len(matching_indices)} matching avalanches")
    return matching_indices


def group_by_metadata_field(
    avalanche_index: List[Dict],
    field: str
) -> Dict[Any, List[int]]:
    """
    Group avalanches by a metadata field value.

    Parameters
    ----------
    avalanche_index : list of dicts
        Avalanche index from create_avalanche_index()
    field : str
        Metadata field to group by (e.g., 'subject', 'condition')

    Returns
    -------
    groups : dict
        Dictionary mapping field values to lists of avalanche indices

    Examples
    --------
    >>> # Group avalanches by subject
    >>> groups = group_by_metadata_field(index, 'subject')
    >>> # Result: {'S01': [0, 1, 5, ...], 'S02': [2, 3, 4, ...]}
    >>>
    >>> # Group by condition
    >>> groups = group_by_metadata_field(index, 'condition')
    >>> # Result: {'rest': [...], 'task': [...]}
    """

    groups = {}

    for idx, entry in enumerate(avalanche_index):
        metadata = entry.get('metadata')

        if metadata and field in metadata:
            value = metadata[field]

            if value not in groups:
                groups[value] = []

            groups[value].append(idx)

    print(f"Grouped into {len(groups)} categories by field '{field}'")
    for key, indices in groups.items():
        print(f"  {key}: {len(indices)} avalanches")

    return groups


def analyze_cluster_composition(
    clusters: np.ndarray,
    avalanche_index: List[Dict],
    metadata_fields: List[str]
) -> Dict[int, Dict]:
    """
    Analyze which metadata categories contribute avalanches to each cluster.

    For each cluster, count how many avalanches come from each metadata category
    (e.g., subject, condition, task).

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments for each avalanche (from hierarchical clustering)
    avalanche_index : list of dicts
        Avalanche index from create_avalanche_index()
    metadata_fields : list of str
        Metadata fields to analyze (e.g., ['subject', 'condition', 'task'])

    Returns
    -------
    composition : dict
        Dictionary mapping cluster IDs to their composition:
        {
            cluster_id: {
                'size': 150,
                'subject': {'S01': 90, 'S02': 60},
                'subject_pct': {'S01': 60.0, 'S02': 40.0},
                'condition': {'rest': 100, 'task': 50},
                'condition_pct': {'rest': 66.7, 'task': 33.3},
                ...
            }
        }

    Examples
    --------
    >>> composition = analyze_cluster_composition(
    ...     clusters,
    ...     avalanche_index,
    ...     metadata_fields=['subject', 'condition', 'task']
    ... )
    >>>
    >>> # Print composition for each cluster
    >>> for cluster_id, stats in composition.items():
    ...     print(f"Cluster {cluster_id}: {stats['size']} avalanches")
    ...     print(f"  Subjects: {stats['subject']}")
    ...     print(f"  Conditions: {stats['condition']}")
    """
    unique_clusters = np.unique(clusters)
    # Filter out noise cluster (-1) if present
    valid_clusters = unique_clusters[unique_clusters != -1]

    composition = {}

    for cluster_id in valid_clusters:
        # Get indices of avalanches in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)

        cluster_stats = {'size': cluster_size}

        # For each metadata field, count occurrences
        for field in metadata_fields:
            counts = defaultdict(int)

            for idx in cluster_indices:
                if idx < len(avalanche_index):
                    metadata = avalanche_index[idx].get('metadata')
                    if metadata and field in metadata:
                        category = metadata[field]
                        counts[category] += 1

            # Store counts
            cluster_stats[field] = dict(counts)

            # Calculate percentages
            if counts:
                total = sum(counts.values())
                percentages = {k: (v/total)*100 for k, v in counts.items()}
                cluster_stats[f'{field}_pct'] = percentages

        composition[int(cluster_id)] = cluster_stats

    # Print summary
    print(f"\n=== Cluster Composition Analysis ===")
    print(f"Analyzed {len(valid_clusters)} clusters")
    for cluster_id in sorted(composition.keys()):
        stats = composition[cluster_id]
        print(f"\nCluster {cluster_id}: {stats['size']} avalanches")
        for field in metadata_fields:
            if field in stats and stats[field]:
                counts_str = ', '.join([f"{k}: {v}" for k, v in sorted(stats[field].items())])
                print(f"  {field.capitalize()}: {counts_str}")

    return composition


def summarize_burst_statistics(
    burst_stats: List[Dict],
    group_by: str
) -> Dict[str, Dict]:
    """
    Summarize burst detection statistics by metadata category.

    Parameters
    ----------
    burst_stats : list of dicts
        List of burst statistics for each dataset:
        [
            {
                'dataset_idx': 0,
                'metadata': {'subject': 'S01', 'condition': 'rest', ...},
                'n_positive': 10,
                'n_negative': 5,
                'rate': 0.5
            },
            ...
        ]
    group_by : str
        Metadata field to group by (e.g., 'subject', 'condition')

    Returns
    -------
    summary : dict
        Statistics grouped by the specified field:
        {
            'S01': {
                'n_datasets': 3,
                'avg_bursts': 12.5,
                'total_bursts': 37,
                'avg_rate': 0.45
            },
            'S02': {...}
        }

    Examples
    --------
    >>> summary = summarize_burst_statistics(burst_stats, group_by='subject')
    >>> for category, stats in summary.items():
    ...     print(f"{category}: {stats['avg_bursts']:.1f} bursts/dataset")
    """
    grouped = defaultdict(lambda: {
        'n_datasets': 0,
        'total_bursts': 0,
        'total_rate': 0.0,
        'bursts_per_dataset': []
    })

    for entry in burst_stats:
        metadata = entry.get('metadata', {})
        category = metadata.get(group_by)

        if category is not None:
            grouped[category]['n_datasets'] += 1
            grouped[category]['total_bursts'] += entry.get('n_positive', 0)
            grouped[category]['total_rate'] += entry.get('rate', 0.0)
            grouped[category]['bursts_per_dataset'].append(entry.get('n_positive', 0))

    # Calculate averages
    summary = {}
    for category, data in grouped.items():
        n = data['n_datasets']
        summary[category] = {
            'n_datasets': n,
            'total_bursts': data['total_bursts'],
            'avg_bursts': data['total_bursts'] / n if n > 0 else 0,
            'avg_rate': data['total_rate'] / n if n > 0 else 0
        }

    # Print summary
    print(f"\n=== Burst Statistics Summary (grouped by {group_by}) ===")
    for category in sorted(summary.keys()):
        stats = summary[category]
        print(f"{category}: {stats['avg_bursts']:.1f} bursts/dataset "
              f"(rate: {stats['avg_rate']:.2f} bursts/sec, n={stats['n_datasets']} datasets)")

    return summary
