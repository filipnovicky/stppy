"""
Metadata analysis utilities for avalanche datasets.
"""

import numpy as np
from typing import List, Dict, Tuple


def link_avalanches_to_datasets(
    all_avalanches: List[List[np.ndarray]],
    all_avalanche_times: List[List[Tuple[int, int]]],
    file_metadata: List[Dict]
) -> List[Dict]:
    """
    Link each avalanche to its dataset labels and temporal information.

    Parameters
    ----------
    all_avalanches : list of lists
        List containing avalanche lists for each dataset
    all_avalanche_times : list of lists
        List containing avalanche timing lists for each dataset
    file_metadata : list of dicts
        List of metadata dictionaries for each dataset

    Returns
    -------
    avalanche_dataset_links : list of dicts
        List of dictionaries, each containing avalanche metadata
    """

    if len(all_avalanches) != len(all_avalanche_times):
        raise ValueError(f"Number of avalanche lists ({len(all_avalanches)}) must match "
                        f"number of timing lists ({len(all_avalanche_times)})")

    if len(all_avalanches) != len(file_metadata):
        raise ValueError(f"Number of avalanche lists ({len(all_avalanches)}) must match "
                        f"number of metadata entries ({len(file_metadata)})")

    avalanche_dataset_links = []

    for dataset_idx, (dataset_avalanches, dataset_times, metadata) in enumerate(
            zip(all_avalanches, all_avalanche_times, file_metadata)):

        if len(dataset_avalanches) != len(dataset_times):
            raise ValueError(f"Dataset {dataset_idx}: Number of avalanches "
                           f"({len(dataset_avalanches)}) must match number of timing entries "
                           f"({len(dataset_times)})")

        for avalanche_idx, (avalanche, avalanche_time) in enumerate(
                zip(dataset_avalanches, dataset_times)):

            start_time, end_time = avalanche_time
            duration = end_time - start_time

            link = {
                'avalanche_id': f"dataset_{dataset_idx}_avalanche_{avalanche_idx}",
                'drug_condition': metadata.get('drug_condition'),
                'task': metadata.get('task'),
                'is_trained': metadata.get('is_trained'),
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }
            avalanche_dataset_links.append(link)

    return avalanche_dataset_links


def analyze_clusters_by_avalanche_metadata(
    cluster_assignments: np.ndarray,
    avalanche_links: List[Dict],
    file_metadata: List[Dict]
) -> Dict:
    """
    Analyze which avalanches make up each cluster with metadata breakdown.

    Parameters
    ----------
    cluster_assignments : np.ndarray
        Cluster assignment for each avalanche
    avalanche_links : list of dict
        Avalanche-dataset linkage information
    file_metadata : list of dict
        Full metadata for each dataset

    Returns
    -------
    cluster_composition : dict
        Dictionary with cluster composition details
    """

    print("\n=== CLUSTER COMPOSITION ANALYSIS ===")

    unique_clusters = np.unique(cluster_assignments)
    valid_clusters = unique_clusters[unique_clusters != -1]

    print(f"Total clusters: {len(valid_clusters)}")
    print(f"Noise points: {np.sum(cluster_assignments == -1)}")

    cluster_composition = {}

    # Compute global counts per category
    all_drug_counts = {}
    all_task_counts = {}
    all_trained_counts = {True: 0, False: 0}

    for link in avalanche_links:
        dataset_idx = int(link['avalanche_id'].split('_')[1])
        metadata = file_metadata[dataset_idx]
        drug = metadata.get('drug_condition')
        task = metadata.get('task')
        trained = metadata.get('is_trained')

        if drug:
            all_drug_counts[drug] = all_drug_counts.get(drug, 0) + 1
        if task:
            all_task_counts[task] = all_task_counts.get(task, 0) + 1
        if trained is not None:
            all_trained_counts[trained] = all_trained_counts.get(trained, 0) + 1

    # Cluster-level analysis
    for cluster_id in valid_clusters:
        cluster_mask = cluster_assignments == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        print(f"\n--- CLUSTER {cluster_id} ({len(cluster_indices)} avalanches) ---")

        cluster_avalanches = []
        for idx in cluster_indices:
            dataset_idx = int(avalanche_links[idx]['avalanche_id'].split('_')[1])
            metadata = file_metadata[dataset_idx]
            full_info = {**avalanche_links[idx], **metadata}
            cluster_avalanches.append(full_info)

        drug_counts = {}
        task_counts = {}
        trained_counts = {}

        for av in cluster_avalanches:
            drug = av.get('drug_condition')
            task = av.get('task')
            trained = av.get('is_trained')

            if drug:
                drug_counts[drug] = drug_counts.get(drug, 0) + 1
            if task:
                task_counts[task] = task_counts.get(task, 0) + 1
            if trained is not None:
                trained_counts[trained] = trained_counts.get(trained, 0) + 1

        cluster_size = len(cluster_avalanches)

        # Drug breakdown
        print("Drug conditions:")
        for drug, count in drug_counts.items():
            pct_within = (count / cluster_size) * 100
            pct_global = (count / all_drug_counts[drug]) * 100 if all_drug_counts.get(drug, 0) > 0 else 0
            print(f"  {drug}: {count} avalanches ({pct_within:.1f}% of cluster, "
                  f"{pct_global:.2f}% of all {drug} avalanches)")

        # Task breakdown
        print("Tasks:")
        for task, count in sorted(task_counts.items()):
            pct_within = (count / cluster_size) * 100
            pct_global = (count / all_task_counts[task]) * 100 if all_task_counts.get(task, 0) > 0 else 0
            print(f"  {task}: {count} avalanches ({pct_within:.1f}% of cluster, "
                  f"{pct_global:.2f}% of all {task} avalanches)")

        cluster_composition[cluster_id] = {
            'size': cluster_size,
            'avalanche_indices': cluster_indices.tolist(),
            'drug_counts': drug_counts,
            'task_counts': task_counts,
            'trained_counts': trained_counts,
            'avalanches': cluster_avalanches
        }

    return cluster_composition
