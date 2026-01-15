#!/usr/bin/env python3
"""
Example: Compute similarity matrix in parallel blocks.

Use this for large datasets (> 1000 avalanches).

USAGE:
------
1. Generate block plan:
   python compute_similarity_parallel.py plan

2. Compute each block (run in parallel, different BLOCK_ID per job):
   python compute_similarity_parallel.py compute --block-id 0
   python compute_similarity_parallel.py compute --block-id 1
   ...

3. Merge all blocks:
   python compute_similarity_parallel.py merge

4. Reconstruct full matrices:
   python compute_similarity_parallel.py reconstruct
"""

import stppy
import numpy as np
import argparse
import os
import glob

# Configuration
AVALANCHE_FILE = 'avalanches.pkl'
ELECTRODE_POSITIONS = 'electrode_positions.json'
BLOCK_SIZE = 500  # Adjust based on memory
OUTPUT_DIR = 'similarity_blocks'
MERGED_FILE = 'similarity_merged.pkl'
OUTPUT_SIMILARITY = 'similarity_matrix.pkl'
OUTPUT_PATHS = 'path_matrix.pkl'


def generate_plan():
    """Step 1: Generate block computation plan."""
    print("=== GENERATING BLOCK PLAN ===\n")

    # Load avalanches to get count
    avalanches = stppy.io.load_avalanche_data(AVALANCHE_FILE, min_time_length=11)
    n_avalanches = len(avalanches)

    print(f"Total avalanches: {n_avalanches}")
    print(f"Block size: {BLOCK_SIZE}\n")

    # Generate plan
    plan = stppy.alignment.generate_block_plan(n_avalanches, BLOCK_SIZE)

    print(f"\n✓ Need to compute {len(plan)} blocks")
    print(f"\nRun these commands in parallel (one per compute node):")
    for i in range(min(5, len(plan))):
        print(f"  python {__file__} compute --block-id {i}")
    if len(plan) > 5:
        print(f"  ... (blocks 5 to {len(plan)-1})")

    # Save plan
    plan_file = 'block_plan.pkl'
    stppy.io.save_pickle(plan, plan_file)
    print(f"\nPlan saved to: {plan_file}")


def compute_block(block_id: int):
    """Step 2: Compute a single block."""
    print(f"=== COMPUTING BLOCK {block_id} ===\n")

    # Load plan
    plan = stppy.io.load_pickle('block_plan.pkl')

    if block_id >= len(plan):
        print(f"ERROR: block_id {block_id} out of range (max: {len(plan)-1})")
        return

    row_start, row_end, col_start, col_end, is_diagonal = plan[block_id]

    print(f"Block assignment:")
    print(f"  Rows: [{row_start}:{row_end}]")
    print(f"  Cols: [{col_start}:{col_end}]")
    print(f"  Diagonal: {is_diagonal}\n")

    # Load data
    print("Loading avalanches...")
    avalanches = stppy.io.load_avalanche_data(AVALANCHE_FILE, min_time_length=11)

    print("Loading distance matrix...")
    K, _ = stppy.utils.create_eeg_ground_metric(ELECTRODE_POSITIONS)
    K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

    # Compute block
    print("\nComputing block...")
    block_dict = stppy.alignment.compute_similarity_block(
        avalanches,
        K_norm,
        row_start, row_end,
        col_start, col_end,
        is_diagonal
    )

    # Save block
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    diag_tag = "_DIAG" if is_diagonal else ""
    output_file = os.path.join(
        OUTPUT_DIR,
        f'block_{block_id:03d}_R{row_start}-{row_end}_C{col_start}-{col_end}{diag_tag}.pkl'
    )

    stppy.io.save_pickle(block_dict, output_file)
    print(f"\n✓ Block saved to: {output_file}")


def merge_blocks_cmd():
    """Step 3: Merge all computed blocks."""
    print("=== MERGING BLOCKS ===\n")

    # Find all block files
    pattern = os.path.join(OUTPUT_DIR, 'block_*.pkl')
    block_files = sorted(glob.glob(pattern))

    if not block_files:
        print(f"ERROR: No block files found in {OUTPUT_DIR}/")
        return

    print(f"Found {len(block_files)} block files\n")

    # Merge
    merged = stppy.alignment.merge_blocks(block_files, output_file=MERGED_FILE)

    print(f"\n✓ Merged blocks saved to: {MERGED_FILE}")
    print(f"\nNext step:")
    print(f"  python {__file__} reconstruct")


def reconstruct_matrices():
    """Step 4: Reconstruct full matrices from merged blocks."""
    print("=== RECONSTRUCTING FULL MATRICES ===\n")

    # Load merged blocks
    print(f"Loading merged blocks from {MERGED_FILE}...")
    merged = stppy.io.load_pickle(MERGED_FILE)

    # Get n_avalanches
    avalanches = stppy.io.load_avalanche_data(AVALANCHE_FILE, min_time_length=11)
    n_avalanches = len(avalanches)

    print(f"Reconstructing matrices for {n_avalanches} avalanches...\n")

    # Reconstruct
    similarity_matrix, path_matrix = stppy.alignment.reconstruct_full_matrices(
        merged,
        n_avalanches
    )

    # Save
    print("\nSaving full matrices...")
    stppy.io.save_pickle(similarity_matrix, OUTPUT_SIMILARITY)
    stppy.io.save_pickle(path_matrix, OUTPUT_PATHS)

    print(f"\n✓ Done!")
    print(f"  Similarity matrix: {OUTPUT_SIMILARITY}")
    print(f"  Path matrix: {OUTPUT_PATHS}")
    print(f"\nYou can now cluster with:")
    print(f"  similarity = stppy.io.load_pickle('{OUTPUT_SIMILARITY}')")
    print(f"  clusters, info = stppy.clustering.simple_hierarchical_clustering(similarity)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel similarity matrix computation")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Plan command
    subparsers.add_parser('plan', help='Generate block computation plan')

    # Compute command
    compute_parser = subparsers.add_parser('compute', help='Compute a single block')
    compute_parser.add_argument('--block-id', type=int, required=True, help='Block ID to compute')

    # Merge command
    subparsers.add_parser('merge', help='Merge all computed blocks')

    # Reconstruct command
    subparsers.add_parser('reconstruct', help='Reconstruct full matrices')

    args = parser.parse_args()

    if args.command == 'plan':
        generate_plan()
    elif args.command == 'compute':
        compute_block(args.block_id)
    elif args.command == 'merge':
        merge_blocks_cmd()
    elif args.command == 'reconstruct':
        reconstruct_matrices()
    else:
        parser.print_help()
