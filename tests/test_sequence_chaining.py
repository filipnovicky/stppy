"""Regression tests for stppy.detection.detect_sequences_in_clusters (sequence chaining).

The chaining rule is defined in frames: a candidate overlapping the current window is a
duplicate, the first candidate starting 0..tolerance_samples frames after the window's end is
appended. In 0.1.0 the comparisons were made in float seconds, so whether a zero-gap or a
one-frame-gap candidate was accepted depended on the window's absolute position.
"""
import numpy as np
import pytest

from stppy.detection import detect_sequences_in_clusters

SFREQ = 250.0
L = 11  # template length in samples


def _assemble(starts, pols, length=L, sfreq=SFREQ, unit="seconds", tol=1):
    starts = np.asarray(starts)
    times = starts / sfreq if unit == "seconds" else starts.astype(float)
    acts = {0: [(0, times, np.asarray(pols, dtype=np.int8), np.full(len(starts), 0.9))]}
    dur = {0: length / sfreq if unit == "seconds" else float(length)}
    seqs = detect_sequences_in_clusters(acts, dur, dataset_names=["rec.set"],
                                        tolerance_samples=tol,
                                        sfreq=sfreq if unit == "seconds" else 1.0)
    return seqs[0]


@pytest.mark.parametrize("last", [0, 25, 40, 1898, 8539, 10636])
@pytest.mark.parametrize("gap", [0, 1])
def test_gap_within_tolerance_is_chained_at_any_position(last, gap):
    seqs = _assemble([last, last + L + gap], [1, -1])
    assert len(seqs) == 1
    assert seqs[0]["length"] == 2
    assert seqs[0]["polarity"] == "rotating"


@pytest.mark.parametrize("last", [0, 25, 1898])
def test_gap_beyond_tolerance_is_not_chained(last):
    assert _assemble([last, last + L + 2], [1, 1]) == []


def test_overlapping_candidates_are_skipped_not_chained():
    seqs = _assemble([100, 105, 111, 122], [1, 1, -1, 1])
    assert len(seqs) == 1
    assert seqs[0]["length"] == 3                      # 100 -> 111 -> 122, 105 is a duplicate
    assert seqs[0]["polarity"] == "rotating"           # +1, -1, +1 (the duplicate's +1 is not used)


def test_end_time_is_window_end_of_last_activation():
    seqs = _assemble([1898, 1910], [1, 1])
    assert seqs[0]["start_time"] == 1898 / SFREQ
    assert seqs[0]["end_time"] == (1910 + L) / SFREQ


def test_result_does_not_depend_on_time_unit():
    rng = np.random.default_rng(0)
    starts = np.sort(rng.choice(75000, size=4000, replace=False))
    starts = np.unique(np.concatenate([starts, starts + L, starts + L + 1]))
    pols = rng.choice([-1, 1], size=starts.size)
    in_seconds = _assemble(starts, pols, unit="seconds")
    in_samples = _assemble(starts, pols, unit="samples")
    key = lambda s, f: (round(s["start_time"] * f), round(s["end_time"] * f), s["length"], s["polarity"])  # noqa: E731
    assert [key(s, SFREQ) for s in in_seconds] == [key(s, 1.0) for s in in_samples]
