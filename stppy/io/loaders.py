"""
Data loading utilities for STPPY package.
"""

import pickle
import glob
import os
import fnmatch
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
                       max_time_length: Optional[int] = None) -> List[np.ndarray]:
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

    Returns
    -------
    avalanche_arrays : list of np.ndarray
        List of avalanche arrays, each with shape (n_electrodes, n_timepoints)

    Examples
    --------
    >>> avalanches = load_avalanche_data('data.pkl', min_time_length=11)
    """
    with open(filepath, 'rb') as f:
        avalanches = pickle.load(f)

    print(f"Loaded {len(avalanches)} participants")

    avalanche_arrays = []
    filtered_count_too_short = 0
    filtered_count_too_long = 0

    for participant_idx, participant_avalanches in enumerate(avalanches):
        for avalanche_idx, avalanche in enumerate(participant_avalanches):
            if hasattr(avalanche, 'shape') and avalanche.ndim == 2:
                temporal_length = avalanche.shape[1]

                if temporal_length < min_time_length:
                    filtered_count_too_short += 1
                elif max_time_length is not None and temporal_length > max_time_length:
                    filtered_count_too_long += 1
                else:
                    avalanche_arrays.append(avalanche)

    print(f"Total avalanches extracted: {len(avalanche_arrays)}")
    print(f"Filtered out {filtered_count_too_short} avalanches with temporal length < {min_time_length}")
    if max_time_length is not None:
        print(f"Filtered out {filtered_count_too_long} avalanches with temporal length > {max_time_length}")

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
    import json

    with open(filepath, 'r') as f:
        electrode_data = json.load(f)

    return electrode_data
