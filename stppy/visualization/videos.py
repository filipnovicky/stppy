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
import json

from ._helpers import _load_montage, _compute_vlim, _select_sequences


def _assemble_video(frame_files, output_file, fps):
    """Read frame PNGs and write to mp4.

    Returns
    -------
    width, height : int
        Video dimensions.
    """
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        video.write(frame)
    video.release()
    return width, height


def _render_topomap_frame(data_slice, info, montage, vlim, cmap, title,
                          figsize, dpi, frame_path):
    """Render a single topomap frame to a PNG file."""
    evoked = mne.EvokedArray(data_slice, info, tmin=0)
    evoked.set_montage(montage)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im, cn = mne.viz.plot_topomap(
        evoked.data[:, 0], evoked.info, axes=ax, show=False,
        vlim=vlim, cmap=cmap, res=256, extrapolate='head',
        outlines='head', contours=0, sensors=True, sphere='auto'
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activity', rotation=270, labelpad=20)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    plt.savefig(frame_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def _render_title_card_frame(seq_title, seg_idx, n_segments, figsize, dpi, frame_file):
    """Render a title card frame and save to disk."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.text(0.5, 0.55, seq_title,
            ha='center', va='center', fontsize=18, fontweight='bold',
            transform=ax.transAxes, linespacing=1.6)
    ax.text(0.5, 0.3, f'Sequence {seg_idx + 1}/{n_segments}',
            ha='center', va='center', fontsize=14, color='gray',
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    plt.savefig(frame_file, bbox_inches='tight', dpi=dpi, facecolor='white')
    plt.close(fig)


def _cleanup_frames(frame_files, temp_dir):
    """Remove temporary frame files and directory."""
    for frame_file in frame_files:
        if os.path.exists(frame_file):
            os.remove(frame_file)
    os.rmdir(temp_dir)


def create_topomap_video(
    data: np.ndarray,
    electrode_positions_file: str,
    output_file: str = 'video.mp4',
    title_prefix: str = '',
    item_id: Optional[int] = None,
    fps: int = 10,
    dpi: int = 100,
    figsize: tuple = (12, 8),
    vlim: Optional[tuple] = None
) -> None:
    """
    Create a video from a topomap sequence (cluster average or single avalanche).

    Parameters
    ----------
    data : np.ndarray
        Data array, shape (n_electrodes, n_timepoints).
    electrode_positions_file : str
        Path to JSON file with electrode positions.
    output_file : str
        Output video filename.
    title_prefix : str
        Text before the time info in the title.
    item_id : int, optional
        Item identifier for logging.
    fps : int
        Frames per second (default: 10).
    dpi : int
        Resolution of saved frames (default: 100).
    figsize : tuple
        Figure size in inches (default: (12, 8)).
    vlim : tuple, optional
        Color scale limits (vmin, vmax).
    """
    n_electrodes, n_timepoints = data.shape

    print(f"\nCreating video for {title_prefix or f'item {item_id}'}")
    print(f"  Shape: {data.shape}")
    print(f"  Creating video with {n_timepoints} frames at {fps} fps")

    channels, montage = _load_montage(electrode_positions_file)
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=250, ch_types='eeg')
    vlim = _compute_vlim(data, vlim)

    temp_dir = tempfile.mkdtemp()
    frame_files = []

    try:
        for t_idx in range(n_timepoints):
            time_point = t_idx / 250.0
            title = (f'{title_prefix}\n'
                     f'Time: {time_point*1000:.1f} ms (Frame {t_idx+1}/{n_timepoints})')

            frame_path = os.path.join(temp_dir, f'frame_{t_idx:04d}.png')
            _render_topomap_frame(
                data[:, [t_idx]], info, montage, vlim, 'RdBu_r',
                title, figsize, dpi, frame_path
            )
            frame_files.append(frame_path)

            if (t_idx + 1) % 10 == 0:
                print(f"  Generated frame {t_idx + 1}/{n_timepoints}")

        print(f"\n  All {n_timepoints} frames generated. Creating video...")
        width, height = _assemble_video(frame_files, output_file, fps)

        print(f"\n\u2713 Video saved to: {output_file}")
        print(f"  Duration: {n_timepoints/fps:.2f} seconds")
        print(f"  Resolution: {width}x{height}")

    finally:
        _cleanup_frames(frame_files, temp_dir)


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
    if cluster_id is not None:
        title_prefix = f'Cluster {cluster_id} - Consensus Average'
    else:
        title_prefix = 'Consensus Average'
    create_topomap_video(average, electrode_positions_file, output_file,
                         title_prefix=title_prefix, item_id=cluster_id,
                         fps=fps, dpi=dpi, figsize=figsize, vlim=vlim)


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
    if avalanche_idx is not None:
        title_prefix = f'Avalanche {avalanche_idx}'
    else:
        title_prefix = 'Avalanche'
    create_topomap_video(avalanche_data, electrode_positions_file, output_file,
                         title_prefix=title_prefix, item_id=avalanche_idx,
                         fps=fps, dpi=dpi, figsize=figsize, vlim=vlim)


def create_cluster_sequences_video(
    sequences_by_cluster: Dict,
    datasets: List[np.ndarray],
    dataset_names: List[str],
    electrode_positions_file: str,
    output_file: str = 'cluster_bursts.mp4',
    cluster_id: Optional[int] = None,
    focus: Optional[Union[str, List[str]]] = None,
    polarity: Optional[str] = None,
    sort_by_length: bool = False,
    n_sequences: int = 10,
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
    Create a video showing sequences sequentially, with title cards between them.

    Mirrors plot_sequence_topomaps: accepts sequences_by_cluster dict, pre-loaded
    datasets in Volts, and supports focus/n_sequences/random_seed filtering.
    Values are displayed in microvolts.

    Parameters
    ----------
    sequences_by_cluster : dict
        {cluster_id: [sequence_dicts, ...]} from sequence detection.
        Each sequence dict must have 'dataset', 'start_time', 'end_time'.
    datasets : list of np.ndarray
        Pre-loaded EEG arrays, each shape (n_electrodes, n_samples), in Volts.
    dataset_names : list of str
        Basenames corresponding to each array in datasets.
    electrode_positions_file : str
        Path to electrode positions JSON file.
    output_file : str
        Output video filename (default: 'cluster_sequences.mp4').
    cluster_id : int, optional
        Filter sequences to this cluster. None plots from all clusters.
    focus : str or list of str, optional
        Filter sequences to specific dataset(s).
    polarity : str, optional
        Filter sequences by polarity ('standing', 'rotating', 'mixed', etc.).
        None allows all polarities.
    sort_by_length : bool
        If True, select the longest sequences instead of random sampling
        (default: False).
    n_sequences : int
        Number of sequences to include (default: 10).
    random_seed : int, optional
        Random seed for reproducible sequence selection.
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
        Duration of title card between sequences in seconds (default: 1.0).

    Examples
    --------
    >>> # Longest standing sequences from cluster 5
    >>> create_cluster_sequences_video(
    ...     sequences_by_cluster, datasets, dataset_names,
    ...     'electrode_positions.json',
    ...     output_file='cluster5_longest.mp4',
    ...     cluster_id=5, polarity='standing', sort_by_length=True
    ... )
    """
    if len(dataset_names) != len(datasets):
        print(f"dataset_names ({len(dataset_names)}) != datasets ({len(datasets)}). "
              f"Pass the full dataset_names list and use focus='filename.set' to filter.")
        return

    selected, ds_name_to_idx = _select_sequences(
        sequences_by_cluster, dataset_names, cluster_id=cluster_id,
        focus=focus, polarity=polarity, sort_by_length=sort_by_length,
        n_sequences=n_sequences, random_seed=random_seed
    )
    if selected is None:
        return

    channels, montage = _load_montage(electrode_positions_file)
    n_electrodes = datasets[0].shape[0]
    info = mne.create_info(ch_names=channels[:n_electrodes], sfreq=sfreq, ch_types='eeg')

    # Extract sequence segments
    sequence_segments = []
    for cid, seq in selected:
        ds_name = seq['dataset']
        if ds_name not in ds_name_to_idx:
            print(f"  Skipping sequence: {ds_name} not found in dataset_names")
            continue

        data = datasets[ds_name_to_idx[ds_name]] * 1e6  # V to \u00b5V

        start_idx = max(0, int(seq['start_time'] * sfreq))
        end_idx = min(data.shape[1], int(seq['end_time'] * sfreq))
        seq_data = data[:, start_idx:end_idx]

        if seq_data.shape[1] == 0:
            print(f"  Skipping empty sequence: {ds_name} {seq['start_time']:.3f}-{seq['end_time']:.3f}s")
            continue

        seq_polarity = seq.get('polarity', '?')
        length = seq.get('length', '?')
        title = (f'Cluster {cid} | {ds_name}\n'
                 f'{seq["start_time"]:.3f}\u2013{seq["end_time"]:.3f}s | '
                 f'polarity: {seq_polarity} | length: {length}')

        sequence_segments.append({
            'data': seq_data,
            'title': title,
            'cid': cid,
            'dataset': ds_name,
            'start_time': seq['start_time'],
        })

    if not sequence_segments:
        print("No valid sequence segments extracted")
        return

    # Compute consistent vlim across all sequences
    if vlim is None:
        all_data = np.concatenate([seg['data'] for seg in sequence_segments], axis=1)
        vlim_use = _compute_vlim(all_data)
    else:
        vlim_use = vlim

    print(f"  Extracted {len(sequence_segments)} sequence segments")
    print(f"  Color scale: {vlim_use} \u00b5V")

    temp_dir = tempfile.mkdtemp()
    frame_files = []

    try:
        frame_counter = 0
        title_frame_count = max(1, int(title_seconds * fps))

        for seg_idx, segment in enumerate(sequence_segments):
            seq_data = segment['data']
            n_timepoints = seq_data.shape[1]
            seq_title = segment['title']

            # Title card frames
            for _ in range(title_frame_count):
                frame_file = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                _render_title_card_frame(
                    seq_title, seg_idx, len(sequence_segments), figsize, dpi, frame_file
                )
                frame_files.append(frame_file)
                frame_counter += 1

            print(f"  Processing sequence {seg_idx + 1}/{len(sequence_segments)} "
                  f"({n_timepoints} frames)")

            # Topomap frames
            for t_idx in range(n_timepoints):
                time_in_seq = t_idx / sfreq
                abs_time = segment['start_time'] + time_in_seq
                frame_title = (f'{seq_title}\n'
                               f'Time: {abs_time:.3f}s (+{time_in_seq*1000:.0f}ms)')

                frame_file = os.path.join(temp_dir, f'frame_{frame_counter:06d}.png')
                evoked = mne.EvokedArray(seq_data[:, [t_idx]], info, tmin=0)
                evoked.set_montage(montage)

                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                im, cn = mne.viz.plot_topomap(
                    evoked.data[:, 0], evoked.info, axes=ax, show=False,
                    vlim=vlim_use, cmap=cmap, res=256, extrapolate='head',
                    outlines='head', contours=0, sensors=True, sphere='auto'
                )
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('\u00b5V', rotation=270, labelpad=20)
                ax.set_title(frame_title, fontsize=11, fontweight='bold', pad=20)
                plt.savefig(frame_file, bbox_inches='tight', dpi=dpi)
                frame_files.append(frame_file)
                plt.close(fig)
                frame_counter += 1

        print(f"\n  All {frame_counter} frames generated. Creating video...")
        width, height = _assemble_video(frame_files, output_file, fps)

        print(f"\n  Video saved to: {output_file}")
        print(f"  Duration: {frame_counter/fps:.2f} seconds")
        print(f"  Resolution: {width}x{height}")

    finally:
        _cleanup_frames(frame_files, temp_dir)


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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    total_frames = 0

    for idx, video_file in enumerate(video_files):
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
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
            total_frames += 1
            frame_count += 1
        cap.release()
        print(f"  Added {frame_count} frames")

    out.release()

    total_duration = total_frames / fps
    print(f"\n\u2713 Combined video saved to: {output_file}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")


def _create_title_frames(text: str, width: int, height: int,
                         duration_seconds: int, fps: int) -> List[np.ndarray]:
    """Create title frames with text."""
    num_frames = int(duration_seconds * fps)
    frames = []

    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 8

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        x = (width - text_width) // 2
        y = (height + text_height) // 2

        cv2.putText(frame, text, (x, y), font, font_scale,
                   (255, 255, 255), thickness, cv2.LINE_AA)
        frames.append(frame)

    return frames
