"""
Video creation utilities for avalanche visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
from typing import Optional, List
import cv2
import mne
from mne.channels import make_dig_montage
import json


def create_cluster_video(
    average: np.ndarray,
    electrode_positions_file: str,
    output_file: str = 'cluster_video.mp4',
    cluster_id: Optional[int] = None,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (12, 8),
    vlim: Optional[tuple] = None
) -> None:
    """
    Create a video from the consensus average topomap sequence.

    Parameters
    ----------
    average : np.ndarray
        Consensus average array, shape (n_electrodes, n_timepoints)
    electrode_positions_file : str
        Path to JSON file with electrode positions
    output_file : str
        Output video filename (default: 'cluster_video.mp4')
    cluster_id : int, optional
        Cluster ID for title
    fps : int
        Frames per second (default: 10)
    dpi : int
        Resolution of saved frames (default: 100)
    figsize : tuple
        Figure size in inches (default: (12, 8))
    vlim : tuple, optional
        Color scale limits (vmin, vmax)

    Examples
    --------
    >>> create_cluster_video(
    ...     average,
    ...     'electrode_positions.json',
    ...     output_file='cluster_0.mp4',
    ...     cluster_id=0,
    ...     fps=5
    ... )
    """

    n_electrodes, n_timepoints = average.shape

    print(f"\nCreating video for cluster {cluster_id if cluster_id is not None else 'N/A'}")
    print(f"  Average shape: {average.shape}")
    print(f"  Creating video with {n_timepoints} frames at {fps} fps")

    # Load electrode positions
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())

    # Create custom montage
    ch_pos = {}
    for ch in channels:
        pos = electrode_data[ch]
        ch_pos[ch] = [pos['x'], pos['y'], pos['z']]

    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # Setup MNE info
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')

    # Calculate consistent vlim
    if vlim is None:
        data_max = np.max(np.abs(average))
        vlim = (-data_max, data_max) if data_max > 0 else (-1, 1)

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []

    try:
        # Generate frames for each timepoint
        for t_idx in range(n_timepoints):
            time_point = t_idx / 250.0

            # Create evoked object with single timepoint
            evoked = mne.EvokedArray(average[:, [t_idx]], info, tmin=0)
            evoked.set_montage(montage)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            # Plot topomap
            im, cn = mne.viz.plot_topomap(
                evoked.data[:, 0],
                evoked.info,
                axes=ax,
                show=False,
                vlim=vlim,
                cmap='RdBu_r',
                res=256,
                extrapolate='head',
                outlines='head',
                contours=0,
                sensors=True,
                sphere='auto'
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Activity', rotation=270, labelpad=20)

            # Add title with time information
            if cluster_id is not None:
                title = (f'Cluster {cluster_id} - Consensus Average\n'
                        f'Time: {time_point*1000:.1f} ms (Frame {t_idx+1}/{n_timepoints})')
            else:
                title = (f'Consensus Average\n'
                        f'Time: {time_point*1000:.1f} ms (Frame {t_idx+1}/{n_timepoints})')
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

            # Save frame
            frame_file = os.path.join(temp_dir, f'frame_{t_idx:04d}.png')
            plt.savefig(frame_file, bbox_inches='tight', dpi=dpi)
            frame_files.append(frame_file)
            plt.close(fig)

            if (t_idx + 1) % 10 == 0:
                print(f"  Generated frame {t_idx + 1}/{n_timepoints}")

        print(f"\n  All {n_timepoints} frames generated. Creating video...")

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Write all frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video.write(frame)

        video.release()

        print(f"\n✓ Video saved to: {output_file}")
        print(f"  Duration: {n_timepoints/fps:.2f} seconds")
        print(f"  Resolution: {width}x{height}")

    finally:
        # Cleanup temporary files
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                os.remove(frame_file)
        os.rmdir(temp_dir)


def create_avalanche_video(
    avalanche_data: np.ndarray,
    electrode_positions_file: str,
    output_file: str = 'avalanche_video.mp4',
    avalanche_idx: Optional[int] = None,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (12, 8),
    vlim: Optional[tuple] = None
) -> None:
    """
    Create a video from a single avalanche.

    Parameters
    ----------
    avalanche_data : np.ndarray
        Avalanche data, shape (n_electrodes, n_timepoints)
    electrode_positions_file : str
        Path to JSON file with electrode positions
    output_file : str
        Output video filename
    avalanche_idx : int, optional
        Avalanche index for title
    fps : int
        Frames per second (default: 10)
    dpi : int
        Resolution of saved frames (default: 100)
    figsize : tuple
        Figure size in inches (default: (12, 8))
    vlim : tuple, optional
        Color scale limits

    Examples
    --------
    >>> create_avalanche_video(
    ...     avalanche_arrays[0],
    ...     'electrode_positions.json',
    ...     output_file='avalanche_0.mp4'
    ... )
    """

    n_electrodes, n_timepoints = avalanche_data.shape

    print(f"\nCreating video for avalanche {avalanche_idx if avalanche_idx is not None else 'N/A'}")
    print(f"  Shape: {avalanche_data.shape}")
    print(f"  Creating video with {n_timepoints} frames at {fps} fps")

    # Load electrode positions
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())

    # Create custom montage
    ch_pos = {}
    for ch in channels:
        pos = electrode_data[ch]
        ch_pos[ch] = [pos['x'], pos['y'], pos['z']]

    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # Setup MNE info
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')

    # Calculate consistent vlim
    if vlim is None:
        data_max = np.max(np.abs(avalanche_data))
        vlim = (-data_max, data_max) if data_max > 0 else (-1, 1)

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []

    try:
        # Generate frames
        for t_idx in range(n_timepoints):
            time_point = t_idx / 250.0

            evoked = mne.EvokedArray(avalanche_data[:, [t_idx]], info, tmin=0)
            evoked.set_montage(montage)

            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            im, cn = mne.viz.plot_topomap(
                evoked.data[:, 0],
                evoked.info,
                axes=ax,
                show=False,
                vlim=vlim,
                cmap='RdBu_r',
                res=256,
                extrapolate='head',
                outlines='head',
                contours=0,
                sensors=True,
                sphere='auto'
            )

            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Activity', rotation=270, labelpad=20)

            if avalanche_idx is not None:
                title = (f'Avalanche {avalanche_idx}\n'
                        f'Time: {time_point*1000:.1f} ms (Frame {t_idx+1}/{n_timepoints})')
            else:
                title = f'Time: {time_point*1000:.1f} ms (Frame {t_idx+1}/{n_timepoints})'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

            frame_file = os.path.join(temp_dir, f'frame_{t_idx:04d}.png')
            plt.savefig(frame_file, bbox_inches='tight', dpi=dpi)
            frame_files.append(frame_file)
            plt.close(fig)

            if (t_idx + 1) % 10 == 0:
                print(f"  Generated frame {t_idx + 1}/{n_timepoints}")

        print(f"\n  All {n_timepoints} frames generated. Creating video...")

        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video.write(frame)

        video.release()

        print(f"\n✓ Video saved to: {output_file}")
        print(f"  Duration: {n_timepoints/fps:.2f} seconds")
        print(f"  Resolution: {width}x{height}")

    finally:
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                os.remove(frame_file)
        os.rmdir(temp_dir)


def create_cluster_bursts_video(
    bursts: List[dict],
    datasets: List[np.ndarray],
    dataset_names: List[str],
    electrode_positions_file: str,
    output_file: str = 'cluster_bursts.mp4',
    cluster_id: Optional[int] = None,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (12, 8),
    vlim: Optional[tuple] = None,
    separator_frames: int = 5
) -> None:
    """
    Create a video showing all bursts from a cluster sequentially.

    Extracts data segments for each burst from the continuous datasets
    and creates a single video playing all bursts in sequence.

    Parameters
    ----------
    bursts : list of dict
        List of burst dictionaries from detect_bursts_in_clusters().
        Each burst should have: 'dataset', 'start_time', 'end_time', 'length'
    datasets : list of np.ndarray
        List of continuous dataset arrays, shape (n_electrodes, n_samples)
    dataset_names : list of str
        List of dataset names corresponding to datasets (e.g., ['S01_rest.set', ...])
    electrode_positions_file : str
        Path to JSON file with electrode positions
    output_file : str
        Output video filename (default: 'cluster_bursts.mp4')
    cluster_id : int, optional
        Cluster ID for title
    fps : int
        Frames per second (default: 10)
    dpi : int
        Resolution of saved frames (default: 100)
    figsize : tuple
        Figure size in inches (default: (12, 8))
    vlim : tuple, optional
        Color scale limits (vmin, vmax). If None, computed from all burst data
    separator_frames : int
        Number of blank frames between bursts (default: 5)

    Examples
    --------
    >>> bursts = bursts_by_cluster[0]  # All bursts for cluster 0
    >>> dataset_names = [os.path.basename(f) for f in files]
    >>> create_cluster_bursts_video(
    ...     bursts,
    ...     datasets,
    ...     dataset_names,
    ...     './electrode_positions.json',
    ...     output_file='./cluster_0_bursts.mp4',
    ...     cluster_id=0
    ... )
    """

    if len(bursts) == 0:
        print("No bursts to visualize")
        return

    print(f"\nCreating burst video for cluster {cluster_id if cluster_id is not None else 'N/A'}")
    print(f"  Number of bursts: {len(bursts)}")

    # Load electrode positions
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())
    n_electrodes = datasets[0].shape[0]

    # Create custom montage
    ch_pos = {}
    for ch in channels:
        pos = electrode_data[ch]
        ch_pos[ch] = [pos['x'], pos['y'], pos['z']]

    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    # Setup MNE info
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')

    # Build mapping from dataset name to index
    dataset_name_to_idx = {name: idx for idx, name in enumerate(dataset_names)}

    # Extract all burst data segments
    burst_segments = []

    for burst_idx, burst in enumerate(bursts):
        dataset_name = burst['dataset']

        # Find dataset index
        if dataset_name not in dataset_name_to_idx:
            print(f"  Warning: Dataset '{dataset_name}' not found in dataset_names, skipping burst {burst_idx}")
            continue

        dataset_idx = dataset_name_to_idx[dataset_name]
        start_time = burst['start_time']
        end_time = burst['end_time']

        # Convert time to samples
        sfreq = 250.0
        start_sample = int(start_time * sfreq)
        end_sample = int(end_time * sfreq)

        # Extract segment
        dataset = datasets[dataset_idx]
        segment = dataset[:, start_sample:end_sample]
        burst_segments.append({
            'data': segment,
            'burst_idx': burst_idx,
            'dataset': dataset_name,
            'start_time': start_time,
            'end_time': end_time
        })

    if len(burst_segments) == 0:
        print("No valid burst segments extracted")
        return

    # Calculate consistent vlim across all bursts
    if vlim is None:
        all_data = np.concatenate([seg['data'] for seg in burst_segments], axis=1)
        data_max = np.max(np.abs(all_data))
        vlim = (-data_max, data_max) if data_max > 0 else (-1, 1)

    print(f"  Extracted {len(burst_segments)} valid burst segments")
    print(f"  Color scale: {vlim}")

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []

    try:
        frame_counter = 0

        # Generate frames for each burst
        for seg_idx, segment in enumerate(burst_segments):
            data = segment['data']
            n_timepoints = data.shape[1]

            print(f"  Processing burst {seg_idx + 1}/{len(burst_segments)} "
                  f"({n_timepoints} frames)")

            # Create frames for this burst
            for t_idx in range(n_timepoints):
                # Create evoked object
                evoked = mne.EvokedArray(data[:, [t_idx]], info, tmin=0)
                evoked.set_montage(montage)

                # Create figure
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

                # Plot topomap
                im, cn = mne.viz.plot_topomap(
                    evoked.data[:, 0],
                    evoked.info,
                    axes=ax,
                    show=False,
                    vlim=vlim,
                    cmap='RdBu_r',
                    res=256,
                    extrapolate='head',
                    outlines='head',
                    contours=0,
                    sensors=True,
                    sphere='auto'
                )

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Activity', rotation=270, labelpad=20)

                # Add title with burst information
                time_in_burst = t_idx / 250.0
                abs_time = segment['start_time'] + time_in_burst

                if cluster_id is not None:
                    title = (f'Cluster {cluster_id} - Burst {seg_idx + 1}/{len(burst_segments)}\n'
                            f'{segment["dataset"]} - Time: {abs_time:.3f}s '
                            f'(+{time_in_burst*1000:.0f}ms)')
                else:
                    title = (f'Burst {seg_idx + 1}/{len(burst_segments)}\n'
                            f'{segment["dataset"]} - Time: {abs_time:.3f}s '
                            f'(+{time_in_burst*1000:.0f}ms)')

                ax.set_title(title, fontsize=12, fontweight='bold', pad=20)

                # Save frame
                frame_file = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                plt.savefig(frame_file, bbox_inches='tight', dpi=dpi)
                frame_files.append(frame_file)
                plt.close(fig)

                frame_counter += 1

            # Add separator frames between bursts (if not last burst)
            if seg_idx < len(burst_segments) - 1:
                for _ in range(separator_frames):
                    # Create blank frame
                    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                    ax.text(0.5, 0.5, f'Next Burst: {seg_idx + 2}/{len(burst_segments)}',
                           ha='center', va='center', fontsize=20, transform=ax.transAxes)
                    ax.axis('off')

                    frame_file = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                    plt.savefig(frame_file, bbox_inches='tight', dpi=dpi)
                    frame_files.append(frame_file)
                    plt.close(fig)

                    frame_counter += 1

        print(f"\n  All {frame_counter} frames generated. Creating video...")

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        height, width, layers = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Write all frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            video.write(frame)

        video.release()

        print(f"\n✓ Video saved to: {output_file}")
        print(f"  Duration: {frame_counter/fps:.2f} seconds")
        print(f"  Resolution: {width}x{height}")

    finally:
        # Cleanup temporary files
        for frame_file in frame_files:
            if os.path.exists(frame_file):
                os.remove(frame_file)
        os.rmdir(temp_dir)


def combine_videos(
    video_files: List[str],
    output_file: str = 'combined.mp4',
    title_duration: int = 2,
    titles: Optional[List[str]] = None
) -> None:
    """
    Combine multiple videos with optional title frames.

    Parameters
    ----------
    video_files : list of str
        List of video file paths to combine
    output_file : str
        Output filename for combined video
    title_duration : int
        Duration of title frames in seconds (default: 2)
    titles : list of str, optional
        Custom titles for each video. If None, uses filenames

    Examples
    --------
    >>> combine_videos(
    ...     ['cluster_0.mp4', 'cluster_1.mp4'],
    ...     output_file='all_clusters.mp4',
    ...     titles=['Cluster 0', 'Cluster 1']
    ... )
    """

    if not video_files:
        print("No video files to combine")
        return

    print(f"Combining {len(video_files)} videos...")

    # Get video properties from first video
    first_video = cv2.VideoCapture(video_files[0])
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_video.get(cv2.CAP_PROP_FPS)
    first_video.release()

    print(f"Video properties: {width}x{height} @ {fps} fps")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    total_frames = 0

    for idx, video_file in enumerate(video_files):
        # Generate title
        if titles and idx < len(titles):
            title = titles[idx]
        else:
            title = Path(video_file).stem

        print(f"\nProcessing: {title}")

        # Add title frame
        print(f"  Adding title frame")
        title_frames = _create_title_frames(title, width, height, title_duration, int(fps))
        for frame in title_frames:
            out.write(frame)
            total_frames += 1

        # Add video content
        print(f"  Adding video content")
        cap = cv2.VideoCapture(video_file)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize if necessary
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)
            total_frames += 1
            frame_count += 1

        cap.release()
        print(f"  Added {frame_count} frames")

    out.release()

    total_duration = total_frames / fps
    print(f"\n✓ Combined video saved to: {output_file}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")


def _create_title_frames(text: str, width: int, height: int,
                         duration_seconds: int, fps: int) -> List[np.ndarray]:
    """Create title frames with text."""
    num_frames = int(duration_seconds * fps)
    frames = []

    for _ in range(num_frames):
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 8

        # Get text size to center it
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Calculate position to center text
        x = (width - text_width) // 2
        y = (height + text_height) // 2

        # Draw text in white
        cv2.putText(frame, text, (x, y), font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)

        frames.append(frame)

    return frames
