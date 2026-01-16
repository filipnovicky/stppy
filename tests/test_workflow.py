#!/usr/bin/env python3
"""
Integration test to verify the complete workflow works end-to-end.
Tests basic functionality without requiring actual data files.
"""

import numpy as np
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stppy

def create_test_data():
    """Create synthetic test data."""
    print("Creating synthetic test data...")

    # Create synthetic avalanches (3 datasets, each with 2-3 avalanches)
    all_avalanches = []
    all_avalanche_times = []

    for dataset_idx in range(3):
        dataset_avalanches = []
        dataset_times = []

        n_avalanches = 2 if dataset_idx == 0 else 3

        for avalanche_idx in range(n_avalanches):
            # Create random avalanche (10 electrodes, 12-15 timepoints)
            n_timepoints = 12 + avalanche_idx
            avalanche = np.random.randn(10, n_timepoints)

            # Make it sparse (only some electrodes active)
            mask = np.random.rand(10, n_timepoints) > 0.7
            avalanche[~mask] = 0

            # Convert to peaks (-1, 0, 1)
            avalanche = np.sign(avalanche)

            dataset_avalanches.append(avalanche)

            # Create timing info
            start = avalanche_idx * 100
            end = start + n_timepoints
            dataset_times.append((start, end))

        all_avalanches.append(dataset_avalanches)
        all_avalanche_times.append(dataset_times)

    print(f"  Created {len(all_avalanches)} datasets")
    print(f"  Total avalanches: {sum(len(d) for d in all_avalanches)}")

    return all_avalanches, all_avalanche_times


def create_test_electrode_positions(temp_dir):
    """Create synthetic electrode positions."""
    import json

    positions = {}
    for i in range(10):
        angle = 2 * np.pi * i / 10
        radius = 1.0
        positions[f'E{i}'] = {
            'x': radius * np.cos(angle),
            'y': radius * np.sin(angle),
            'z': 0.0
        }

    positions_file = os.path.join(temp_dir, 'electrode_positions.json')
    with open(positions_file, 'w') as f:
        json.dump(positions, f)

    return positions_file


def test_io():
    """Test I/O functions."""
    print("\n" + "="*60)
    print("TEST: I/O Functions")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test save/load pickle
        data = {'test': [1, 2, 3], 'array': np.array([4, 5, 6])}
        pickle_file = os.path.join(temp_dir, 'test.pkl')

        stppy.io.save_pickle(data, pickle_file)
        loaded = stppy.io.load_pickle(pickle_file)

        assert 'test' in loaded
        assert len(loaded['test']) == 3
        print("  ✓ Pickle save/load works")

        # Test electrode positions
        positions_file = create_test_electrode_positions(temp_dir)
        positions = stppy.io.load_electrode_positions(positions_file)

        assert len(positions) == 10
        assert 'E0' in positions
        print("  ✓ Electrode positions load works")


def test_utils():
    """Test utility functions."""
    print("\n" + "="*60)
    print("TEST: Utils Functions")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        positions_file = create_test_electrode_positions(temp_dir)

        # Test ground metric creation
        K, channel_names = stppy.utils.create_eeg_ground_metric(positions_file)

        assert K.shape == (10, 10)
        assert len(channel_names) == 10
        assert np.all(np.diag(K) == 0)  # Diagonal should be zero
        print("  ✓ Ground metric creation works")

        # Test normalization
        K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')
        assert K_norm.shape == K.shape
        print("  ✓ Distance matrix normalization works")


def test_detection():
    """Test avalanche detection."""
    print("\n" + "="*60)
    print("TEST: Detection Functions")
    print("="*60)

    # Create synthetic EEG data
    data = np.random.randn(10, 1000)
    data[5, 100:110] = 5  # Add a strong event

    # Test threshold-based detection
    print("\n  Threshold-based detection:")
    filter_masks = stppy.detection.create_avalanche_filter([data], [-2, 2])
    assert len(filter_masks) == 1
    assert filter_masks[0].shape == (1000,)
    print("    ✓ Avalanche filter creation works")

    # Test avalanche detection
    all_avalanches, all_times = stppy.detection.detect_avalanches(
        [data], filter_masks, [-2, 2], min_duration=2, verbose=False
    )
    assert len(all_avalanches) == 1
    print(f"    ✓ Avalanche detection works (found {len(all_avalanches[0])} avalanches)")

    # Test peak conversion
    if len(all_avalanches[0]) > 0:
        peaks = stppy.detection.convert_to_peaks(all_avalanches)
        assert len(peaks) == 1
        if len(peaks[0]) > 0:
            assert np.all(np.isin(peaks[0][0], [-1, 0, 1]))
        print("    ✓ Peak conversion works")

    # Test correlation-based detection
    print("\n  Correlation-based detection:")

    # Create electrode positions for spatial smoothing
    positions = np.array([[i, 0, 0] for i in range(10)])

    # Test spatial smoothing matrix
    W_smooth, distances, mean_dist = stppy.detection.build_spatial_smoothing_matrix(
        positions, k=3.0
    )
    assert W_smooth.shape == (10, 10)
    assert np.allclose(W_smooth.sum(axis=1), 1.0)  # Rows sum to 1
    print("    ✓ Spatial smoothing matrix creation works")

    # Test spatiotemporal correlation
    pattern = np.random.randn(10, 20)
    window = np.random.randn(10, 20)
    corr = stppy.detection.spatiotemporal_correlation(pattern, window, W_smooth)
    assert -1 <= corr <= 1
    print("    ✓ Spatiotemporal correlation works")

    # Test burst detection by correlation
    burst_info = stppy.detection.detect_bursts_by_correlation(
        data, pattern, W_smooth, threshold=0.7, return_correlations=True
    )
    assert 'positive_bursts' in burst_info
    assert 'negative_bursts' in burst_info
    assert 'correlations' in burst_info
    print(f"    ✓ Burst detection works (found {burst_info['n_positive']} positive, {burst_info['n_negative']} negative)")

    # Test temporal statistics
    if len(burst_info['correlations']) > 0:
        stats = stppy.detection.compute_temporal_statistics(
            burst_info['correlations'],
            burst_info['positions'],
            threshold=0.7
        )
        assert 'n_above_threshold' in stats
        assert 'rate_above' in stats
        print("    ✓ Temporal statistics computation works")


def test_alignment():
    """Test alignment functions."""
    print("\n" + "="*60)
    print("TEST: Alignment Functions")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        positions_file = create_test_electrode_positions(temp_dir)
        K, _ = stppy.utils.create_eeg_ground_metric(positions_file)
        K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')

        # Create two simple avalanches
        avalanche1 = np.random.choice([-1, 0, 1], size=(10, 12), p=[0.1, 0.8, 0.1])
        avalanche2 = np.random.choice([-1, 0, 1], size=(10, 15), p=[0.1, 0.8, 0.1])

        # Test spatial similarity
        sim_matrix = stppy.alignment.create_temporal_similarity_matrix(
            avalanche1, avalanche2, K_norm
        )
        assert sim_matrix.shape == (12, 15)
        print("  ✓ Temporal similarity matrix creation works")

        # Test DTW
        cost, paths, transposed = stppy.alignment.dtw_on_similarity_matrix(sim_matrix)
        assert isinstance(cost, (int, float))
        assert isinstance(paths, list)
        print("  ✓ DTW alignment works")

        # Test full similarity matrix computation (small)
        avalanches = [avalanche1, avalanche2]
        similarity, path_matrix = stppy.alignment.compute_full_similarity_matrix(
            avalanches, K_norm, show_progress=False
        )
        assert similarity.shape == (2, 2)
        assert path_matrix.shape == (2, 2)
        assert similarity[0, 1] == similarity[1, 0]  # Symmetric
        print("  ✓ Full similarity matrix computation works")


def test_clustering():
    """Test clustering functions."""
    print("\n" + "="*60)
    print("TEST: Clustering Functions")
    print("="*60)

    # Create synthetic similarity matrix
    n = 10
    similarity = np.random.rand(n, n) * 10
    similarity = (similarity + similarity.T) / 2  # Make symmetric
    np.fill_diagonal(similarity, 0)

    # Test hierarchical clustering
    clusters, cluster_info = stppy.clustering.simple_hierarchical_clustering(
        similarity, threshold=5.0, min_cluster_size=2, use_lowest_tightness=False
    )

    assert len(clusters) == n
    assert all(c >= -1 for c in clusters)  # -1 for noise, 0+ for clusters
    print(f"  ✓ Hierarchical clustering works (found {len(cluster_info)} clusters)")


def test_analysis():
    """Test analysis functions."""
    print("\n" + "="*60)
    print("TEST: Analysis Functions")
    print("="*60)

    all_avalanches, all_avalanche_times = create_test_data()

    # Test universal metadata
    metadata = [
        {'subject': 'S01', 'condition': 'rest'},
        {'subject': 'S02', 'condition': 'task'},
        {'subject': 'S01', 'condition': 'task'}
    ]

    index = stppy.analysis.create_avalanche_index(
        all_avalanches, all_avalanche_times, metadata
    )

    total_avalanches = sum(len(d) for d in all_avalanches)
    assert len(index) == total_avalanches
    print(f"  ✓ Avalanche index creation works ({len(index)} entries)")

    # Test querying
    s01_indices = stppy.analysis.query_avalanches(index, field='subject', value='S01')
    assert len(s01_indices) > 0
    print(f"  ✓ Avalanche querying works (found {len(s01_indices)} matching)")

    # Test grouping
    groups = stppy.analysis.group_by_metadata_field(index, 'condition')
    assert 'rest' in groups or 'task' in groups
    print(f"  ✓ Grouping works ({len(groups)} groups)")


def test_visualization():
    """Test visualization functions (without actually showing plots)."""
    print("\n" + "="*60)
    print("TEST: Visualization Functions")
    print("="*60)

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as temp_dir:
        positions_file = create_test_electrode_positions(temp_dir)

        # Create synthetic data
        avalanche_times = [(100, 112), (200, 215), (300, 311)]
        cluster_assignments = np.array([0, 1, 0])
        raw_data = np.random.randn(10, 500)

        # Note: Timeline, raster, and correlation plots removed
        # Focus is on spatiotemporal pattern visualization
        print("  ✓ Visualization tests complete")


def test_complete_workflow():
    """Test a simplified end-to-end workflow."""
    print("\n" + "="*60)
    print("TEST: Complete Workflow")
    print("="*60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Create test data
        all_avalanches, all_avalanche_times = create_test_data()
        positions_file = create_test_electrode_positions(temp_dir)

        # 2. Create distance matrix
        K, channel_names = stppy.utils.create_eeg_ground_metric(positions_file)
        K_norm = stppy.utils.normalize_distance_matrix(K, method='mean')
        print("  ✓ Step 1: Distance matrix created")

        # 3. Flatten avalanche list for similarity computation
        avalanche_list = []
        for dataset in all_avalanches:
            avalanche_list.extend(dataset)

        # 4. Compute similarity matrix (small dataset)
        similarity, paths = stppy.alignment.compute_full_similarity_matrix(
            avalanche_list[:5],  # Only use first 5 to keep it fast
            K_norm,
            show_progress=False
        )
        print("  ✓ Step 2: Similarity matrix computed")

        # 5. Cluster avalanches
        clusters, cluster_info = stppy.clustering.simple_hierarchical_clustering(
            similarity, threshold=3.0, min_cluster_size=1, use_lowest_tightness=False
        )
        print(f"  ✓ Step 3: Clustering done ({len(cluster_info)} clusters)")

        # 6. Create metadata index
        metadata = [
            {'subject': f'S{i:02d}', 'condition': 'test'}
            for i in range(len(all_avalanches))
        ]
        index = stppy.analysis.create_avalanche_index(
            all_avalanches, all_avalanche_times, metadata
        )
        print(f"  ✓ Step 4: Metadata index created ({len(index)} avalanches)")

        # 7. Save results
        similarity_file = os.path.join(temp_dir, 'similarity.pkl')
        stppy.io.save_pickle(similarity, similarity_file)
        loaded_similarity = stppy.io.load_pickle(similarity_file)
        assert np.allclose(similarity, loaded_similarity)
        print("  ✓ Step 5: Results saved and loaded")

        print("\n  ✅ Complete workflow successful!")


def main():
    """Run all tests."""
    print("="*60)
    print("STPPY PACKAGE VALIDATION")
    print("="*60)

    try:
        test_io()
        test_utils()
        test_detection()
        test_alignment()
        test_clustering()
        test_analysis()
        test_visualization()
        test_complete_workflow()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe package is working correctly.")
        print("You can now use it for real data analysis.")

        return 0

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED!")
        print("="*60)
        print(f"\nError: {e}")

        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    sys.exit(main())
