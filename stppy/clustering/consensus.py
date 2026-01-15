"""
Consensus alignment building for avalanche clusters.
"""

import numpy as np
from typing import Tuple, List, Dict


def build_consensus_alignment(
    cluster_id: int,
    clusters: np.ndarray,
    avalanche_arrays: List[np.ndarray],
    avalanche_paths: np.ndarray,
    similarity_clean: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict, int, np.ndarray]:
    """
    Build consensus alignment using the medoid as reference timeline.

    Parameters
    ----------
    cluster_id : int
        The cluster ID to build consensus for
    clusters : np.ndarray
        Cluster assignments for each avalanche
    avalanche_arrays : list
        List of avalanche data arrays
    avalanche_paths : np.ndarray
        DTW path matrix
    similarity_clean : np.ndarray
        Similarity/distance matrix
    verbose : bool
        Whether to print detailed progress

    Returns
    -------
    aligned_avalanches : np.ndarray
        Array of aligned avalanches
    average_avalanche : np.ndarray
        Consensus average avalanche
    alignment_info : dict
        Information about how each avalanche was aligned
    medoid_idx : int
        Index of the medoid avalanche
    consensus_count : np.ndarray
        Number of avalanches contributing at each time point
    """

    cluster_indices = np.where(clusters == cluster_id)[0]
    n_in_cluster = len(cluster_indices)

    print(f"\n{'='*60}")
    print(f"Cluster {cluster_id}: {n_in_cluster} avalanches")
    print(f"{'='*60}")

    # Find medoid (most central avalanche)
    cluster_distances = similarity_clean[np.ix_(cluster_indices, cluster_indices)]
    sum_distances = cluster_distances.sum(axis=1)
    medoid_idx_in_cluster = np.argmin(sum_distances)
    medoid_idx = cluster_indices[medoid_idx_in_cluster]

    medoid_array = avalanche_arrays[medoid_idx]
    medoid_length = medoid_array.shape[1]
    n_channels = medoid_array.shape[0]

    print(f"\nMedoid: avalanche {medoid_idx}")
    print(f"  Shape: {medoid_array.shape}")
    print(f"  Length: {medoid_length} time steps")

    # Initialize consensus timeline
    consensus_sum = np.zeros((n_channels, medoid_length))
    consensus_count = np.zeros(medoid_length)

    aligned_avalanches = []
    alignment_info = {}

    if verbose:
        print(f"\n{'='*60}")
        print("Warping avalanches to medoid timeline...")
        print(f"{'='*60}")

    # Process each avalanche
    for idx in cluster_indices:
        # Medoid maps to itself
        if idx == medoid_idx:
            aligned = medoid_array.copy()
            consensus_sum += aligned
            consensus_count += 1
            aligned_avalanches.append(aligned)

            alignment_info[idx] = {
                'is_medoid': True,
                'frames_used': medoid_length,
                'frames_lost_start': 0,
                'frames_lost_end': 0
            }

            if verbose:
                print(f"  Avalanche {idx} (medoid): identity mapping")
            continue

        # Get DTW path
        i_min, i_max = min(idx, medoid_idx), max(idx, medoid_idx)

        try:
            path_list = avalanche_paths[i_min][i_max]

            if isinstance(path_list, list) and len(path_list) > 0:
                path = path_list[0]
            else:
                if verbose:
                    print(f"  ⚠️  Avalanche {idx}: Empty path, skipping")
                continue

            if len(path) == 0:
                if verbose:
                    print(f"  ⚠️  Avalanche {idx}: No path to medoid, skipping")
                continue

        except (IndexError, TypeError) as e:
            if verbose:
                print(f"  ⚠️  Avalanche {idx}: Error accessing path - {e}")
            continue

        # Extract frame correspondences
        if idx < medoid_idx:
            idx_times = np.array([p[0] for p in path])
            medoid_times = np.array([p[1] for p in path])
        else:
            medoid_times = np.array([p[0] for p in path])
            idx_times = np.array([p[1] for p in path])

        # Warp idx to medoid's timeline
        avalanche_data = avalanche_arrays[idx]
        avalanche_length = avalanche_data.shape[1]

        aligned = np.zeros((n_channels, medoid_length))

        for medoid_t in range(medoid_length):
            matching_idx_frames = idx_times[medoid_times == medoid_t]

            if len(matching_idx_frames) > 0:
                aligned[:, medoid_t] = avalanche_data[:, matching_idx_frames].mean(axis=1)
                consensus_count[medoid_t] += 1

        consensus_sum += aligned
        aligned_avalanches.append(aligned)

        # Track what was lost
        frames_in_path = len(np.unique(idx_times))
        frames_lost_start = idx_times.min()
        frames_lost_end = avalanche_length - idx_times.max() - 1

        alignment_info[idx] = {
            'is_medoid': False,
            'original_length': avalanche_length,
            'frames_used': frames_in_path,
            'frames_lost_start': frames_lost_start,
            'frames_lost_end': frames_lost_end,
            'aligned_region': (idx_times.min(), idx_times.max()),
            'medoid_region': (medoid_times.min(), medoid_times.max())
        }

        if verbose:
            print(f"  Avalanche {idx} (len={avalanche_length}):")
            print(f"    Aligned frames {idx_times.min()}-{idx_times.max()} → "
                  f"medoid frames {medoid_times.min()}-{medoid_times.max()}")

    # Compute consensus average
    consensus_count[consensus_count == 0] = 1
    average_avalanche = consensus_sum / consensus_count

    if verbose:
        print(f"\n✓ Consensus built from {len(aligned_avalanches)} avalanches")

    return (np.array(aligned_avalanches),
            average_avalanche,
            alignment_info,
            medoid_idx,
            consensus_count)
