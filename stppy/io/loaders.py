"""
Data loading utilities for STPPY package.
"""

import pickle
import glob
import os
import fnmatch
import json
from typing import List, Tuple, Optional, Union
import numpy as np
import mne


def load_eeg_files(base_dirs: List[str], conditions: List[str],
                   file_extension: str = '*.set') -> List[str]:
    """
    Collect all EEG files from specified directories and conditions.

    Parameters
    ----------
    base_dirs : list of str
        List of base directory paths to search
    conditions : list of str
        List of condition subdirectories to search within each base directory
    file_extension : str, optional
        File pattern to match (default: '*.set')

    Returns
    -------
    file_paths : list of str
        Sorted list of all found file paths

    Examples
    --------
    >>> base_dirs = ['/path/to/Control', '/path/to/Psilocybin']
    >>> conditions = ['Meditation', 'Music', 'Resting state', 'Video']
    >>> files = load_eeg_files(base_dirs, conditions)
    """
    file_paths = []

    for base_dir in base_dirs:
        for condition in conditions:
            pattern = os.path.join(base_dir, condition, file_extension)
            files = glob.glob(pattern)
            file_paths.extend(files)

    file_paths.sort()

    print(f"Found {len(file_paths)} EEG files")
    return file_paths


def find_eeg_files(paths: Union[str, List[str]], pattern: str = '*.set',
                   recursive: bool = True) -> List[str]:
    """
    Find all EEG files in given directories (simple universal approach).

    Parameters
    ----------
    paths : str or list of str
        Directory path(s) to search for EEG files
    pattern : str, optional
        File pattern to match (default: '*.set')
    recursive : bool, optional
        Search subdirectories recursively (default: True)

    Returns
    -------
    file_paths : list of str
        All found file paths, sorted

    Examples
    --------
    >>> # Single directory, recursive (finds all .set in subdirectories)
    >>> files = find_eeg_files('/data/eeg_study')
    >>>
    >>> # Multiple directories
    >>> files = find_eeg_files(['/data/control', '/data/drug'])
    >>>
    >>> # Different file type
    >>> files = find_eeg_files('/data', pattern='*.fif')
    >>>
    >>> # Non-recursive (only top-level)
    >>> files = find_eeg_files('/data', recursive=False)
    """
    if isinstance(paths, str):
        paths = [paths]

    file_paths = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
            continue

        if recursive:
            # Search all subdirectories
            for root, dirs, files in os.walk(path):
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        file_paths.append(os.path.join(root, file))
        else:
            # Just the directory itself
            file_paths.extend(glob.glob(os.path.join(path, pattern)))

    file_paths.sort()
    print(f"Found {len(file_paths)} files matching '{pattern}'")
    return file_paths


def load_eeg_raw(file_path: str, preload: bool = True,
                 pick_eeg: bool = True) -> mne.io.Raw:
    """
    Load a single EEG file.

    Parameters
    ----------
    file_path : str
        Path to the EEG file
    preload : bool, optional
        Whether to preload data into memory (default: True)
    pick_eeg : bool, optional
        Whether to keep only EEG channels (default: True)

    Returns
    -------
    raw : mne.io.Raw
        Loaded raw EEG data
    """
    raw = mne.io.read_raw_eeglab(file_path, preload=preload)

    if pick_eeg:
        raw.pick_types(eeg=True, exclude='bads')

    return raw


def load_avalanche_data(filepath: str, min_time_length: int = 1,
                       max_time_length: Optional[int] = None,
                       include_flipped: bool = False,
                       times_filepath: Optional[str] = None):
    """
    Load avalanche data from pickle file with optional temporal length filtering.

    Parameters
    ----------
    filepath : str
        Path to the pickle file containing avalanche data
    min_time_length : int, optional
        Minimum temporal length required (default: 1)
    max_time_length : int or None, optional
        Maximum temporal length allowed (default: None, no upper limit)
    include_flipped : bool, optional
        If True, include polarity-flipped versions of all avalanches (multiplied by -1).
        This doubles the dataset size and helps find spatiotemporal patterns regardless
        of activation polarity (default: False)
    times_filepath : str or None, optional
        Path to the pickle file containing avalanche timing data.
        If provided, timing info is loaded, filtered, and returned alongside arrays.
        Flipped avalanches reuse the same timing as their originals. (default: None)

    Returns
    -------
    avalanche_arrays : list of np.ndarray
        List of avalanche arrays, each with shape (n_electrodes, n_timepoints)
    avalanche_times : list of tuples (only if times_filepath is provided)
        List of (start_idx, end_idx) tuples indicating when each avalanche occurs

    Examples
    --------
    >>> # Load with flipped versions for clustering
    >>> avalanches = load_avalanche_data('data.pkl', min_time_length=11, include_flipped=True)
    >>> # Load with timing info
    >>> avalanches, times = load_avalanche_data('data.pkl', min_time_length=11,
    ...                                         include_flipped=True,
    ...                                         times_filepath='times.pkl')
    """
    with open(filepath, 'rb') as f:
        avalanches = pickle.load(f)

    print(f"Loaded {len(avalanches)} participants")

    has_times = times_filepath is not None
    if has_times:
        with open(times_filepath, 'rb') as f:
            all_avalanche_times = pickle.load(f)

    avalanche_arrays = []
    avalanche_times = []
    filtered_count_too_short = 0
    filtered_count_too_long = 0

    for participant_idx, participant_avalanches in enumerate(avalanches):
        participant_times = all_avalanche_times[participant_idx] if has_times else None

        for avalanche_idx, avalanche in enumerate(participant_avalanches):
            if hasattr(avalanche, 'shape') and avalanche.ndim == 2:
                temporal_length = avalanche.shape[1]

                if temporal_length < min_time_length:
                    filtered_count_too_short += 1
                elif max_time_length is not None and temporal_length > max_time_length:
                    filtered_count_too_long += 1
                else:
                    avalanche_arrays.append(avalanche)
                    if has_times:
                        avalanche_times.append(participant_times[avalanche_idx])

    print(f"Total avalanches extracted: {len(avalanche_arrays)}")
    print(f"Filtered out {filtered_count_too_short} avalanches with temporal length < {min_time_length}")
    if max_time_length is not None:
        print(f"Filtered out {filtered_count_too_long} avalanches with temporal length > {max_time_length}")

    # Include polarity-flipped versions if requested
    if include_flipped:
        original_count = len(avalanche_arrays)
        flipped_avalanches = [-arr for arr in avalanche_arrays]
        avalanche_arrays.extend(flipped_avalanches)
        if has_times:
            avalanche_times = avalanche_times + avalanche_times
        print(f"Included flipped versions: {len(avalanche_arrays)} total avalanches ({original_count} original + {original_count} flipped)")

    if has_times:
        return avalanche_arrays, avalanche_times
    return avalanche_arrays


def save_pickle(data, filepath: str) -> None:
    """
    Save data to pickle file.

    Parameters
    ----------
    data : any
        Data to save
    filepath : str
        Output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved data to {filepath}")


def load_pickle(filepath: str):
    """
    Load data from pickle file.

    Parameters
    ----------
    filepath : str
        Path to pickle file

    Returns
    -------
    data : any
        Loaded data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def load_electrode_positions(filepath: str) -> dict:
    """
    Load electrode positions from JSON file.

    Parameters
    ----------
    filepath : str
        Path to JSON file with electrode positions

    Returns
    -------
    electrode_data : dict
        Dictionary mapping electrode names to position dictionaries
        with 'x', 'y', 'z' coordinates
    """
    with open(filepath, 'r') as f:
        electrode_data = json.load(f)

    return electrode_data


def create_electrode_positions_file(
    source: Union[mne.io.Raw, mne.Epochs, mne.channels.DigMontage],
    output_file: str = 'electrode_positions.json',
    overwrite: bool = False
) -> None:
    """
    Create electrode positions JSON file from MNE data with montage.

    IMPORTANT: This function does NOT accept string montage names (like 'standard_1020').
    You MUST load your data, set the montage, and VERIFY it matches your actual electrode
    setup before calling this function. This forces you to check electrode positions are
    correct, which is critical for accurate spatial analysis.

    Parameters
    ----------
    source : mne.io.Raw, mne.Epochs, or mne.channels.DigMontage
        Source of electrode positions. Must have montage already set.
        - For Raw/Epochs: Use raw.set_montage() first
        - For DigMontage: Use mne.channels.make_standard_montage() or custom montage
    output_file : str, default='electrode_positions.json'
        Output file path for JSON file
    overwrite : bool, default=False
        If False, raises error if output file already exists.
        Set to True to overwrite existing file.

    Raises
    ------
    FileExistsError
        If output_file exists and overwrite=False
    ValueError
        If source has no montage set, or if requested channels not found

    Examples
    --------
    >>> # Load data and set montage
    >>> raw = mne.io.read_raw_eeglab('data.set', preload=True)
    >>> raw.set_montage('standard_1020')
    >>>
    >>> # Create positions file
    >>> stppy.io.create_electrode_positions_file(raw)
    >>>
    >>> # From custom montage
    >>> montage = mne.channels.make_standard_montage('biosemi64')
    >>> stppy.io.create_electrode_positions_file(montage)

    Notes
    -----
    The output JSON format is:
    {
        "Fp1": {"x": -0.029, "y": 0.084, "z": -0.007},
        "Fp2": {"x": 0.030, "y": 0.085, "z": -0.007},
        ...
    }

    This function intentionally does NOT accept string montage names to force
    users to verify electrode positions are correct before creating the file.
    """
    # Check if output file exists
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(
            f"Output file '{output_file}' already exists. "
            f"Set overwrite=True to replace it, or choose a different filename."
        )

    # Extract montage based on source type
    if isinstance(source, mne.channels.DigMontage):
        montage = source
    elif isinstance(source, (mne.io.BaseRaw, mne.BaseEpochs)):
        montage = source.get_montage()
        if montage is None:
            raise ValueError(
                "Raw/Epochs object has no montage set. "
                "Use raw.set_montage() first and verify positions are correct."
            )
    else:
        raise TypeError(
            f"source must be mne.io.BaseRaw, mne.BaseEpochs, or mne.channels.DigMontage, "
            f"got {type(source)}. "
            f"\n\nNOTE: String montage names like 'standard_1020' are NOT accepted. "
            f"You must load your data, set the montage, and verify it's correct first."
        )

    # Get positions
    positions = montage.get_positions()
    ch_pos = positions['ch_pos']

    if not ch_pos:
        raise ValueError("Montage has no channel positions.")

    # Convert to package JSON format
    electrode_data = {}
    for ch, pos in ch_pos.items():
        electrode_data[ch] = {
            'x': float(pos[0]),
            'y': float(pos[1]),
            'z': float(pos[2])
        }

    # Save to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(electrode_data, f, indent=2)

    print(f"\n✓ Created electrode positions file: {output_file}")
    print(f"  Channels: {len(electrode_data)}")
    print(f"  Channel names: {', '.join(sorted(list(electrode_data.keys())[:10]))}{'...' if len(electrode_data) > 10 else ''}")
    print(f"\nIMPORTANT: Verify these positions match your actual electrode setup!")
