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


def remove_duplicate_polarities(
    cluster_averages: Dict[int, np.ndarray],
    threshold: float = -0.999
) -> List[int]:
    """
    Identify and remove clusters that are polarity-flipped duplicates.

    When include_flipped=True is used during avalanche loading, clusters may form
    that are identical except for polarity (correlation ≈ -1.0). This function
    identifies such pairs and returns only one representative from each pair.

    Parameters
    ----------
    cluster_averages : dict
        Dictionary mapping cluster IDs to their consensus averages
        (from build_consensus_alignment)
    threshold : float
        Correlation threshold for identifying duplicates (default: -0.999)
        Pairs with correlation at or below this are considered duplicates

    Returns
    -------
    clusters_to_keep : list of int
        List of cluster IDs to keep (duplicates removed)

    Examples
    --------
    >>> # After building consensus alignments for all clusters
    >>> cluster_averages = {
    ...     0: average_0,
    ...     1: average_1,
    ...     2: average_2
    ... }
    >>> clusters_to_keep = remove_duplicate_polarities(cluster_averages)
    >>> print(f"Keeping {len(clusters_to_keep)} unique clusters")
    """
    if not cluster_averages:
        return []

    cluster_ids = sorted(cluster_averages.keys())
    n_clusters = len(cluster_ids)

    if n_clusters == 1:
        return cluster_ids

    # Flatten each cluster average to 1D for correlation
    flattened = {}
    for cid in cluster_ids:
        avg = cluster_averages[cid]
        flattened[cid] = avg.flatten()

    # Find duplicate pairs (anti-correlated)
    to_remove = set()

    for i, cid_a in enumerate(cluster_ids):
        if cid_a in to_remove:
            continue

        for cid_b in cluster_ids[i+1:]:
            if cid_b in to_remove:
                continue

            # Compute correlation
            flat_a = flattened[cid_a]
            flat_b = flattened[cid_b]

            # Skip pairs with different lengths (can't be polarity duplicates)
            if len(flat_a) != len(flat_b):
                continue

            # Pearson correlation
            corr = np.corrcoef(flat_a, flat_b)[0, 1]

            # If highly anti-correlated, they're duplicates
            if corr <= threshold:
                # Keep the one with lower ID, remove the other
                to_remove.add(cid_b)
                print(f"  Removing cluster {cid_b} (duplicate polarity of cluster {cid_a}, corr={corr:.4f})")

    clusters_to_keep = [cid for cid in cluster_ids if cid not in to_remove]

    print(f"\n=== Duplicate Polarity Removal ===")
    print(f"Original clusters: {n_clusters}")
    print(f"Removed: {len(to_remove)}")
    print(f"Keeping: {len(clusters_to_keep)}")

    return clusters_to_keep
