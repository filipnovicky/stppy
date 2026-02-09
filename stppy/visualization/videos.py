"""
Video creation utilities for avalanche visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Union
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
    bursts_by_cluster: Dict,
    datasets: List[np.ndarray],
    dataset_names: List[str],
    electrode_positions_file: str,
    output_file: str = 'cluster_bursts.mp4',
    cluster_id: Optional[int] = None,
    focus: Optional[Union[str, List[str]]] = None,
    polarity: Optional[str] = None,
    sort_by_length: bool = False,
    n_bursts: int = 10,
    random_seed: Optional[int] = None,
    sfreq: float = 250.0,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (12, 8),
    vlim: Optional[tuple] = None,
    cmap: str = 'RdBu_r',
    title_seconds: float = 1.0
) -> None:
    """
    Create a video showing bursts sequentially, with title cards between them.

    Mirrors plot_burst_topomaps: accepts bursts_by_cluster dict, pre-loaded
    datasets in Volts, and supports focus/n_bursts/random_seed filtering.
    Values are displayed in microvolts.

    Parameters
    ----------
    bursts_by_cluster : dict
        {cluster_id: [burst_dicts, ...]} from burst detection.
        Each burst dict must have 'dataset', 'start_time', 'end_time'.
    datasets : list of np.ndarray
        Pre-loaded EEG arrays, each shape (n_electrodes, n_samples), in Volts.
    dataset_names : list of str
        Basenames corresponding to each array in datasets.
    electrode_positions_file : str
        Path to electrode positions JSON file.
    output_file : str
        Output video filename (default: 'cluster_bursts.mp4').
    cluster_id : int, optional
        Filter bursts to this cluster. None plots from all clusters.
    focus : str or list of str, optional
        Filter bursts to specific dataset(s).
    polarity : str, optional
        Filter bursts by polarity ('stable', 'switching', 'mixed', etc.).
        None allows all polarities.
    sort_by_length : bool
        If True, select the longest bursts instead of random sampling
        (default: False).
    n_bursts : int
        Number of bursts to include (default: 10).
    random_seed : int, optional
        Random seed for reproducible burst selection.
    sfreq : float
        Sampling frequency in Hz (default: 250.0).
    fps : int
        Frames per second (default: 10).
    dpi : int
        Resolution of saved frames (default: 100).
    figsize : tuple
        Figure size in inches (default: (12, 8)).
    vlim : tuple, optional
        Color scale limits in microvolts. None uses adaptive scaling.
    cmap : str
        Colormap name (default: 'RdBu_r').
    title_seconds : float
        Duration of title card between bursts in seconds (default: 1.0).

    Examples
    --------
    >>> # Longest stable bursts from cluster 5
    >>> create_cluster_bursts_video(
    ...     bursts_by_cluster, datasets, dataset_names,
    ...     'electrode_positions.json',
    ...     output_file='cluster5_longest.mp4',
    ...     cluster_id=5, polarity='stable', sort_by_length=True
    ... )
    """
    # Handle length mismatch
    if len(dataset_names) != len(datasets):
        print(f"dataset_names ({len(dataset_names)}) != datasets ({len(datasets)}). "
              f"Pass the full dataset_names list and use focus='filename.set' to filter.")
        return

    # Build name -> index lookup
    ds_name_to_idx = {name: i for i, name in enumerate(dataset_names)}

    # Normalize focus
    if focus is not None:
        if isinstance(focus, str):
            focus_set = {focus}
        else:
            focus_set = set(focus)
        missing = focus_set - set(ds_name_to_idx)
        if missing:
            print(f"Not found in dataset_names: {missing}")
            return
        print(f"Filtering to {len(focus_set)} dataset(s): {', '.join(sorted(focus_set))}")
    else:
        focus_set = None

    # Collect matching bursts
    candidates = []
    cluster_ids = [cluster_id] if cluster_id is not None else sorted(bursts_by_cluster.keys())

    for cid in cluster_ids:
        if cid not in bursts_by_cluster:
            continue
        for burst in bursts_by_cluster[cid]:
            if focus_set is not None and burst['dataset'] not in focus_set:
                continue
            if polarity is not None and burst.get('polarity') != polarity:
                continue
            candidates.append((cid, burst))

    if not candidates:
        filters = f"cluster_id={cluster_id}, focus={focus}, polarity={polarity}"
        print(f"No bursts found matching {filters}")
        return

    # Select bursts
    n_to_plot = min(n_bursts, len(candidates))

    if sort_by_length:
        candidates.sort(key=lambda x: x[1].get('length', 0), reverse=True)
        selected = candidates[:n_to_plot]
        print(f"Creating video with {n_to_plot} longest burst(s) from {len(candidates)} candidates")
    else:
        if random_seed is not None:
            np.random.seed(random_seed)
        selected_idx = np.random.choice(len(candidates), size=n_to_plot, replace=False)
        selected = [candidates[i] for i in selected_idx]
        print(f"Creating video with {n_to_plot} burst(s) from {len(candidates)} candidates")

    # Load electrode positions and create montage
    with open(electrode_positions_file, 'r') as f:
        electrode_data = json.load(f)

    channels = list(electrode_data.keys())
    ch_pos = {ch: [electrode_data[ch]['x'], electrode_data[ch]['y'], electrode_data[ch]['z']]
              for ch in channels}
    montage = make_dig_montage(ch_pos=ch_pos, coord_frame='head')

    n_electrodes = datasets[0].shape[0]
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=sfreq, ch_types='eeg')

    # Extract burst segments
    burst_segments = []
    for cid, burst in selected:
        ds_name = burst['dataset']
        if ds_name not in ds_name_to_idx:
            print(f"  Skipping burst: {ds_name} not found in dataset_names")
            continue

        data = datasets[ds_name_to_idx[ds_name]] * 1e6  # V to µV

        start_idx = int(burst['start_time'] * sfreq)
        end_idx = int(burst['end_time'] * sfreq)
        start_idx = max(0, start_idx)
        end_idx = min(data.shape[1], end_idx)
        burst_data = data[:, start_idx:end_idx]

        if burst_data.shape[1] == 0:
            print(f"  Skipping empty burst: {ds_name} {burst['start_time']:.3f}-{burst['end_time']:.3f}s")
            continue

        polarity = burst.get('polarity', '?')
        length = burst.get('length', '?')
        title = (f'Cluster {cid} | {ds_name}\n'
                 f'{burst["start_time"]:.3f}\u2013{burst["end_time"]:.3f}s | '
                 f'polarity: {polarity} | length: {length}')

        burst_segments.append({
            'data': burst_data,
            'title': title,
            'cid': cid,
            'dataset': ds_name,
            'start_time': burst['start_time'],
        })

    if not burst_segments:
        print("No valid burst segments extracted")
        return

    # Compute consistent vlim across all bursts
    if vlim is None:
        all_data = np.concatenate([seg['data'] for seg in burst_segments], axis=1)
        data_max = np.max(np.abs(all_data))
        vlim_use = (-data_max, data_max) if data_max > 0 else (-1, 1)
    else:
        vlim_use = vlim

    print(f"  Extracted {len(burst_segments)} burst segments")
    print(f"  Color scale: {vlim_use} \u00b5V")

    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []

    try:
        frame_counter = 0
        title_frame_count = max(1, int(title_seconds * fps))

        for seg_idx, segment in enumerate(burst_segments):
            burst_data = segment['data']
            n_timepoints = burst_data.shape[1]
            burst_title = segment['title']

            # --- Title card frames ---
            for _ in range(title_frame_count):
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax.text(0.5, 0.55, burst_title,
                        ha='center', va='center', fontsize=18, fontweight='bold',
                        transform=ax.transAxes, linespacing=1.6)
                ax.text(0.5, 0.3, f'Burst {seg_idx + 1}/{len(burst_segments)}',
                        ha='center', va='center', fontsize=14, color='gray',
                        transform=ax.transAxes)
                ax.axis('off')
                ax.set_facecolor('white')
                fig.set_facecolor('white')

                frame_file = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                plt.savefig(frame_file, bbox_inches='tight', dpi=dpi, facecolor='white')
                frame_files.append(frame_file)
                plt.close(fig)
                frame_counter += 1

            print(f"  Processing burst {seg_idx + 1}/{len(burst_segments)} "
                  f"({n_timepoints} frames)")

            # --- Topomap frames ---
            for t_idx in range(n_timepoints):
                evoked = mne.EvokedArray(burst_data[:, [t_idx]], info, tmin=0)
                evoked.set_montage(montage)

                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

                im, cn = mne.viz.plot_topomap(
                    evoked.data[:, 0],
                    evoked.info,
                    axes=ax,
                    show=False,
                    vlim=vlim_use,
                    cmap=cmap,
                    res=256,
                    extrapolate='head',
                    outlines='head',
                    contours=0,
                    sensors=True,
                    sphere='auto'
                )

                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('\u00b5V', rotation=270, labelpad=20)

                time_in_burst = t_idx / sfreq
                abs_time = segment['start_time'] + time_in_burst
                frame_title = (f'{burst_title}\n'
                               f'Time: {abs_time:.3f}s (+{time_in_burst*1000:.0f}ms)')
                ax.set_title(frame_title, fontsize=11, fontweight='bold', pad=20)

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

        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            video.write(frame)

        video.release()

        print(f"\n  Video saved to: {output_file}")
        print(f"  Duration: {frame_counter/fps:.2f} seconds")
        print(f"  Resolution: {width}x{height}")

    finally:
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
