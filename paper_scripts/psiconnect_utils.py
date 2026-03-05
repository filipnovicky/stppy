"""
Psiconnect project utilities.

Shared functions and constants for the psilocybin EEG analysis scripts.
These are project-specific (tied to the psiconnect dataset naming conventions)
and therefore live outside the general-purpose stppy package.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ======================================================================
# CONSTANTS
# ======================================================================

BASE_DIR = Path(os.environ.get('PSICONNECT_DATA_DIR', 'data'))

DRUG_LABELS = {'ctrl': 'Control', 'psi': 'Psilocybin'}

TASK_LABELS = {
    'meditation': 'Meditation',
    'music': 'Music',
    'restingstate': 'Resting state',
    'video': 'Video'
}

POL_ORDER = ['stable', 'switching', 'mixed', 'single']

POLARITY_TYPES = ['stable', 'switching', 'long_switching', 'mixed', 'single']


# ======================================================================
# CONDITION PARSING
# ======================================================================

def parse_condition(dataset_name: str):
    """Extract drug, subject and task from dataset filename.

    Expected format: ``ctrl_sub001_meditation.set`` or ``psi_sub201_music.set``

    Parameters
    ----------
    dataset_name : str
        Dataset filename (with or without path).

    Returns
    -------
    drug : str
        'ctrl', 'psi', or '?'
    subject : str
        Subject token (e.g. 'sub001') or '?'
    task : str
        Task token (e.g. 'meditation') or '?'
    """
    name = dataset_name.replace('.set', '').replace('.pkl', '')
    parts = name.split('_', 2)
    drug = 'ctrl' if name.startswith('ctrl') else 'psi' if name.startswith('psi') else '?'
    subject = parts[1] if len(parts) >= 2 else '?'
    task = parts[2] if len(parts) >= 3 else '?'
    return drug, subject, task


# ======================================================================
# POLARITY HELPERS
# ======================================================================

def determine_polarity(pols):
    """Determine sequence polarity from sequence of +/- values.

    Parameters
    ----------
    pols : list
        Sequence of polarity values (+1 / -1).

    Returns
    -------
    str
        'single', 'standing', 'rotating', or 'mixed'
    """
    if len(pols) < 2:
        return 'single'

    unique = set(pols)
    if len(unique) == 1:
        return 'stable'

    is_alternating = True
    for i in range(1, len(pols)):
        if pols[i] == pols[i-1]:
            is_alternating = False
            break

    if is_alternating:
        return 'switching'

    return 'mixed'


def polarity_summary(label, sequences, polarity_types, indent=0):
    """Print polarity counts for a group of sequences.

    Parameters
    ----------
    label : str
        Label to display.
    sequences : list of dict
        Sequence dictionaries with a 'polarity' key.
    polarity_types : list of str
        Polarity categories to report.
    indent : int
        Indentation level (each level = 2 spaces).
    """
    n = len(sequences)
    if n == 0:
        return
    pad = "  " * indent
    counts = defaultdict(int)
    for b in sequences:
        counts[b.get('polarity', '?')] += 1
    print(f"{pad}{label} ({n} sequences):")
    for pt in polarity_types:
        c = counts[pt]
        if c > 0:
            print(f"{pad}  {pt:16s}: {c:>5} ({c/n*100:.1f}%)")


# ======================================================================
# SEQUENCE DETECTION (for combined cluster analyses)
# ======================================================================

def find_continuous_periods(times, polarities, duration, sfreq=250.0):
    """Group consecutive activations into continuous detection periods.

    Consecutive means within 2 samples (8 ms at 250 Hz) of each other.

    Parameters
    ----------
    times : array-like
        Activation times in seconds.
    polarities : array-like
        Polarity values (+1 / -1) for each activation.
    duration : float
        Duration of one activation in seconds.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    list of dict
        Each dict has 'start', 'end', 'pol' keys.
    """
    if len(times) == 0:
        return []

    tolerance = 2.0 / sfreq  # 8ms

    periods = []
    start = times[0]
    end = times[0] + duration
    period_pols = [polarities[0]]

    for i in range(1, len(times)):
        if times[i] <= times[i-1] + tolerance:
            end = times[i] + duration
            period_pols.append(polarities[i])
        else:
            dominant_pol = 1 if sum(period_pols) > 0 else -1
            periods.append({
                'start': start,
                'end': end,
                'pol': dominant_pol
            })
            start = times[i]
            end = times[i] + duration
            period_pols = [polarities[i]]

    dominant_pol = 1 if sum(period_pols) > 0 else -1
    periods.append({'start': start, 'end': end, 'pol': dominant_pol})
    return periods


def merge_periods_into_sequences(periods, tolerance):
    """Merge overlapping/adjacent periods into sequences.

    Parameters
    ----------
    periods : list of dict
        Each dict must have 'start', 'end', 'cluster', 'pol' keys.
    tolerance : float
        Max gap (in seconds) between period end and next period start.

    Returns
    -------
    list of dict
        Each dict has 'start', 'end', 'clusters', 'pols' keys.
    """
    if not periods:
        return []

    periods = sorted(periods, key=lambda x: x['start'])

    sequences = []
    current = {
        'start': periods[0]['start'],
        'end': periods[0]['end'],
        'clusters': [periods[0]['cluster']],
        'pols': [periods[0]['pol']]
    }

    for p in periods[1:]:
        if p['start'] <= current['end'] + tolerance:
            current['end'] = max(current['end'], p['end'])
            current['clusters'].append(p['cluster'])
            current['pols'].append(p['pol'])
        else:
            sequences.append(current)
            current = {
                'start': p['start'],
                'end': p['end'],
                'clusters': [p['cluster']],
                'pols': [p['pol']]
            }

    sequences.append(current)
    return sequences


# ======================================================================
# STATISTICS
# ======================================================================

def sig_stars(p):
    """Return significance stars for a p-value.

    Parameters
    ----------
    p : float
        p-value.

    Returns
    -------
    str
        '***', '**', '*', or 'n.s.'
    """
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'n.s.'


def benjamini_hochberg(p_values):
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.

    Returns
    -------
    np.ndarray
        FDR-adjusted p-values.
    """
    n = len(p_values)
    if n == 0:
        return []
    ranked = np.argsort(p_values)
    adjusted = np.ones(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        raw = p_values[ranked[i]] * n / rank
        if i == n - 1:
            adjusted[ranked[i]] = min(raw, 1.0)
        else:
            adjusted[ranked[i]] = min(raw, adjusted[ranked[i + 1]], 1.0)
    return adjusted


# ======================================================================
# METADATA (moved from stppy.analysis.metadata)
# ======================================================================

def create_avalanche_index(
    all_avalanches: List[List[np.ndarray]],
    all_avalanche_times: List[List[Tuple[int, int]]],
    dataset_names: List[str]
) -> List[Dict]:
    """Create an index linking each avalanche to its dataset, timing, and name.

    Parameters
    ----------
    all_avalanches : list of lists
        List containing avalanche lists for each dataset.
    all_avalanche_times : list of lists
        List containing avalanche timing lists for each dataset.
        Each timing is a tuple (start_idx, end_idx).
    dataset_names : list of str
        Name for each dataset (typically filename).

    Returns
    -------
    list of dict
        Each entry: {'dataset_idx', 'dataset_name', 'avalanche_idx',
        'start_time', 'end_time', 'duration'}
    """
    if len(all_avalanches) != len(all_avalanche_times):
        raise ValueError(
            f"Number of avalanche lists ({len(all_avalanches)}) must match "
            f"number of timing lists ({len(all_avalanche_times)})"
        )

    if len(dataset_names) != len(all_avalanches):
        raise ValueError(
            f"Number of dataset names ({len(dataset_names)}) must match "
            f"number of avalanche lists ({len(all_avalanches)})"
        )

    avalanche_index = []

    for dataset_idx, (dataset_avalanches, dataset_times, dataset_name) in enumerate(
            zip(all_avalanches, all_avalanche_times, dataset_names)):

        if len(dataset_avalanches) != len(dataset_times):
            raise ValueError(
                f"Dataset {dataset_idx} ({dataset_name}): Number of avalanches "
                f"({len(dataset_avalanches)}) must match number of timing entries "
                f"({len(dataset_times)})"
            )

        for avalanche_idx, (avalanche, avalanche_time) in enumerate(
                zip(dataset_avalanches, dataset_times)):

            start_time, end_time = avalanche_time
            duration = end_time - start_time

            entry = {
                'dataset_idx': dataset_idx,
                'dataset_name': dataset_name,
                'avalanche_idx': avalanche_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }

            avalanche_index.append(entry)

    print(f"Created index for {len(avalanche_index)} avalanches "
          f"across {len(all_avalanches)} datasets")

    return avalanche_index


def analyze_cluster_composition(
    clusters: np.ndarray,
    avalanche_index: List[Dict]
) -> Dict[int, Dict]:
    """Analyze which datasets contribute avalanches to each cluster.

    Parameters
    ----------
    clusters : np.ndarray
        Cluster assignments for each avalanche.
    avalanche_index : list of dict
        Avalanche index from create_avalanche_index().

    Returns
    -------
    dict
        {cluster_id: {'size', 'datasets', 'datasets_pct'}}
    """
    unique_clusters = np.unique(clusters)
    valid_clusters = unique_clusters[unique_clusters != -1]

    composition = {}

    for cluster_id in valid_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)

        dataset_counts = defaultdict(int)

        for idx in cluster_indices:
            if idx < len(avalanche_index):
                dataset_name = avalanche_index[idx].get('dataset_name')
                if dataset_name:
                    dataset_counts[dataset_name] += 1

        total = sum(dataset_counts.values())
        dataset_pct = {k: (v/total)*100 for k, v in dataset_counts.items()} if total > 0 else {}

        composition[int(cluster_id)] = {
            'size': cluster_size,
            'datasets': dict(dataset_counts),
            'datasets_pct': dataset_pct
        }

    print(f"\n=== Cluster Composition Analysis ===")
    print(f"Analyzed {len(valid_clusters)} clusters")
    for cluster_id in sorted(composition.keys()):
        stats = composition[cluster_id]
        print(f"\nCluster {cluster_id}: {stats['size']} avalanches")
        if stats['datasets']:
            counts_str = ', '.join([f"{k}: {v}" for k, v in sorted(stats['datasets'].items())])
            print(f"  Datasets: {counts_str}")

    return composition


def summarize_sequence_statistics(
    sequence_stats: List[Dict]
) -> Dict[str, Dict]:
    """Summarize sequence detection statistics by dataset.

    Parameters
    ----------
    sequence_stats : list of dict
        List of sequence statistics with 'dataset_name', 'n_positive', 'rate' keys.

    Returns
    -------
    dict
        {dataset_name: {'n_sequences', 'rate'}}
    """
    summary = {}

    for entry in sequence_stats:
        dataset_name = entry.get('dataset_name')
        if dataset_name:
            summary[dataset_name] = {
                'n_sequences': entry.get('n_positive', 0),
                'rate': entry.get('rate', 0.0)
            }

    print(f"\n=== Sequence Statistics Summary ===")
    for dataset_name in sorted(summary.keys()):
        stats = summary[dataset_name]
        print(f"{dataset_name}: {stats['n_sequences']} sequences (rate: {stats['rate']:.2f} sequences/sec)")

    return summary


def parse_dataset_metadata(
    file_paths: List[str],
    drug_conditions: Tuple[str, ...] = ('Psilocybin', 'Control'),
    task_conditions: Tuple[str, ...] = ('Meditation', 'Music', 'Resting state', 'Video'),
    trained_prefix: str = '2'
) -> List[Dict]:
    """Parse dataset metadata from file paths.

    Extracts drug condition, task condition, subject ID, and subject number
    from the directory structure and filename.

    Parameters
    ----------
    file_paths : list of str
        Full file paths to EEG datasets.
    drug_conditions : tuple of str
        Known drug condition names to search for in path components.
    task_conditions : tuple of str
        Known task condition names to search for in path components.
    trained_prefix : str
        Subject numbers starting with this character are considered trained.

    Returns
    -------
    list of dict
        Each dict: {'drug_condition', 'task', 'subject_id', 'subject_number',
        'is_trained', 'file_path'}
    """
    metadata = []

    for file_path in file_paths:
        path_parts = file_path.split('/')

        drug_condition = drug_conditions[-1]
        for condition in drug_conditions[:-1]:
            if condition in path_parts:
                drug_condition = condition
                break

        task = None
        for part in path_parts:
            if part in task_conditions:
                task = part
                break

        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        tokens = name_without_ext.split('_')

        subject_id = tokens[0] if tokens else name_without_ext

        subject_number = None
        for token in tokens[1:]:
            if token.startswith('sub'):
                subject_number = token[3:]
                break

        is_trained = None
        if subject_number and subject_number[0].isdigit():
            is_trained = subject_number.startswith(trained_prefix)

        metadata.append({
            'drug_condition': drug_condition,
            'task': task,
            'subject_id': subject_id,
            'subject_number': subject_number,
            'is_trained': is_trained,
            'file_path': file_path
        })

    drugs = set(m['drug_condition'] for m in metadata)
    tasks = set(m['task'] for m in metadata if m['task'] is not None)
    subjects = set(m['subject_number'] for m in metadata if m['subject_number'] is not None)
    print(f"Parsed metadata for {len(metadata)} datasets")
    print(f"  Drug conditions: {sorted(drugs)}")
    print(f"  Task conditions: {sorted(tasks)}")
    print(f"  Subjects: {len(subjects)} unique")
    n_trained = sum(1 for m in metadata if m['is_trained'] is True)
    n_untrained = sum(1 for m in metadata if m['is_trained'] is False)
    print(f"  Trained: {n_trained} datasets, Untrained: {n_untrained} datasets")

    return metadata
