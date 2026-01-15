"""
Hierarchical clustering for avalanche analysis.
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Tuple, List, Dict


def simple_hierarchical_clustering(
    distance_matrix: np.ndarray,
    threshold: float = 50.0,
    method: str = 'average',
    min_cluster_size: int = 1,
    use_lowest_tightness: bool = True
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Simple hierarchical clustering with cluster size filtering.

    Parameters
    ----------
    distance_matrix : array, shape (n_samples, n_samples)
        Pairwise distances between samples
    threshold : float
        Distance threshold for forming clusters (default: 50.0)
    method : str
        Linkage method ('average', 'single', 'complete', 'ward')
    min_cluster_size : int
        Minimum size for a cluster (smaller clusters assigned to -1)
    use_lowest_tightness : bool
        Whether to sort clusters by tightness

    Returns
    -------
    cluster_assignments : array
        Cluster ID for each sample (-1 for noise/small clusters, 0+ for clusters)
    cluster_info : list of dict
        Information about each cluster
    """

    # Convert distance matrix to condensed form
    condensed_distances = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method=method)

    # Cut the dendrogram at the specified threshold
    temp_clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Convert to 0-based indexing
    temp_clusters = temp_clusters - 1

    # Filter out small clusters
    cluster_assignments = -np.ones(len(temp_clusters), dtype=int)
    cluster_info = []
    cluster_counter = 0

    unique_clusters = np.unique(temp_clusters)

    print(f"=== HIERARCHICAL CLUSTERING ===")
    print(f"Method: {method}")
    print(f"Threshold: {threshold}")
    print(f"Min cluster size: {min_cluster_size}")

    for cluster_id in unique_clusters:
        cluster_mask = temp_clusters == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_size = len(cluster_indices)

        if cluster_size >= min_cluster_size:
            cluster_assignments[cluster_mask] = cluster_counter

            # Compute tightness (average intra-cluster distance)
            if cluster_size > 1:
                sub_matrix = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                triu_indices = np.triu_indices(cluster_size, k=1)
                tightness = np.mean(sub_matrix[triu_indices])
            else:
                tightness = 0.0

            cluster_info.append({
                'cluster_id': cluster_counter,
                'size': cluster_size,
                'tightness': tightness,
                'locked_at_threshold': threshold
            })

            print(f"  Cluster {cluster_counter}: size={cluster_size}, tightness={tightness:.2f}")
            cluster_counter += 1
        else:
            print(f"  Discarded small cluster: size={cluster_size} (< {min_cluster_size})")

    # Sort clusters by tightness
    if use_lowest_tightness:
        cluster_info.sort(key=lambda x: x['tightness'])

    n_noise = np.sum(cluster_assignments == -1)
    print(f"\nSummary:")
    print(f"  Total clusters: {len(cluster_info)}")
    print(f"  Unassigned/noise: {n_noise}")
    if cluster_info:
        print(f"  Tightest cluster: {cluster_info[0]['tightness']:.3f}")
        print(f"  Loosest cluster: {cluster_info[-1]['tightness']:.3f}")

    return cluster_assignments, cluster_info
