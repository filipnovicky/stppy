"""
Resampling utilities for EEG data.
"""

import mne
from typing import Union


def resample_data(
    raw_or_epochs: Union[mne.io.Raw, mne.Epochs],
    target_sfreq: float,
    npad: str = 'auto'
) -> Union[mne.io.Raw, mne.Epochs]:
    """
    Resample EEG data to target sampling frequency.

    Parameters
    ----------
    raw_or_epochs : mne.Raw or mne.Epochs
        Input EEG data
    target_sfreq : float
        Target sampling frequency in Hz
    npad : str, optional
        Padding parameter for resampling (default: 'auto')

    Returns
    -------
    resampled_data : mne.Raw or mne.Epochs
        Resampled EEG data

    Examples
    --------
    >>> raw = mne.io.read_raw_eeglab('data.set', preload=True)
    >>> raw_250 = resample_data(raw, 250)
    """
    resampled = raw_or_epochs.copy()
    resampled.resample(target_sfreq, npad=npad)
    return resampled
