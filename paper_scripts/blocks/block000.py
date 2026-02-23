#!/usr/bin/env python3
"""Block 000: diagonal block (0, 0) - CPU parallel"""

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

# Block 0: diagonal block (0, 0)
print("Computing block 000...")
start_time = time.time()
block_0 = stppy.alignment.compute_similarity_block(
    avalanches, K_norm,
    row_start=0, row_end=2554,
    col_start=0, col_end=2554,
    is_diagonal=True
)
elapsed = time.time() - start_time
stppy.io.save_pickle(block_0, '/home/novicky/stppy/paper_scripts/blocks/block_000.pkl')
print(f"Block 000 complete in {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
