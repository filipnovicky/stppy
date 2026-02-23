#!/usr/bin/env python3
"""Block 001: off-diagonal block (1, 0) - CPU parallel"""

import sys
sys.path.append('/home/novicky/stppy')
import stppy
import os
import time

# Ensure blocks directory exists
os.makedirs('/home/novicky/stppy/paper_scripts/blocks', exist_ok=True)

# Load avalanche data (include flipped versions)
print("Loading avalanche data...")
avalanches = stppy.io.load_avalanche_data(
    '/home/novicky/stppy/peak_avalanches.pkl',
    min_time_length=11,
    include_flipped=True
)
print(f"Loaded {len(avalanches)} avalanches (including flipped)")

# Create normalized distance matrix
print("Creating normalized distance matrix...")
K, channel_names = stppy.utils.create_eeg_ground_metric('/home/novicky/stppy/electrode_positions.json')
K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

# Block 1: off-diagonal block (1, 0)
print("Computing block 001...")
start_time = time.time()
block_1 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=2554, row_end=5108,
    col_start=0, col_end=2554,
    is_diagonal=False
)
elapsed = time.time() - start_time
stppy.io.save_pickle(block_1, '/home/novicky/stppy/paper_scripts/blocks/block_001.pkl')
print(f"Block 001 complete in {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
