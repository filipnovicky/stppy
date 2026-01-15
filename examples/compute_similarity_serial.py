#!/usr/bin/env python3
"""
Example: Compute full similarity matrix (serial).

Use this for small-to-medium datasets (< 1000 avalanches).
For larger datasets, use compute_similarity_parallel.py instead.
"""

import stppy
import numpy as np

# Configuration
AVALANCHE_FILE = 'avalanches.pkl'
ELECTRODE_POSITIONS = 'electrode_positions.json'
OUTPUT_SIMILARITY = 'similarity_matrix.pkl'
OUTPUT_PATHS = 'path_matrix.pkl'

print("=== COMPUTING SIMILARITY MATRIX (SERIAL) ===\n")

# 1. Load avalanches
print("Loading avalanches...")
avalanches = stppy.io.load_avalanche_data(
    AVALANCHE_FILE,
    min_time_length=11
)
print(f"Loaded {len(avalanches)} avalanches\n")

# 2. Create distance matrix
print("Creating electrode distance matrix...")
K, channel_names = stppy.utils.create_eeg_ground_metric(ELECTRODE_POSITIONS)
K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')
print(f"Distance matrix: {K_norm.shape}\n")

# 3. Compute full similarity matrix
print("Computing pairwise similarities...")
print("(This may take a while for large datasets)\n")

similarity_matrix, path_matrix = stppy.alignment.compute_full_similarity_matrix(
    avalanches,
    K_norm,
    show_progress=True,
    save_interval=100,  # Save checkpoint every 100 pairs
    checkpoint_file='similarity_checkpoint.pkl'
)

# 4. Save results
print("\nSaving results...")
stppy.io.save_pickle(similarity_matrix, OUTPUT_SIMILARITY)
stppy.io.save_pickle(path_matrix, OUTPUT_PATHS)

print(f"\n✓ Done!")
print(f"  Similarity matrix: {OUTPUT_SIMILARITY}")
print(f"  Path matrix: {OUTPUT_PATHS}")
print(f"\nYou can now cluster with:")
print(f"  similarity = stppy.io.load_pickle('{OUTPUT_SIMILARITY}')")
print(f"  paths = stppy.io.load_pickle('{OUTPUT_PATHS}')")
print(f"  clusters, info = stppy.clustering.simple_hierarchical_clustering(similarity)")
