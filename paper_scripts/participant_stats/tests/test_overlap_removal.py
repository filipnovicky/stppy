"""Tests for main_analysis.remove_overlapping_bursts (44 ms cross-cluster overlap rule)."""

import numpy as np
import pytest

from main_analysis import remove_overlapping_bursts

SF = 250.0
AVERAGES = {0: np.zeros((64, 11)), 1: np.zeros((64, 12))}   # minimum template length 11 samples


def _seq(start, end, old_style=False):
    """A sequence from start to end sample; old_style = 0.1.0 pickle (seconds only)."""
    if old_style:
        return {"dataset": "rec.set", "start_time": start / SF,
                "end_time": (end - 11) / SF + 11 / SF}       # last start + duration, as in 0.1.0
    return {"dataset": "rec.set", "start_time": start / SF, "end_time": end / SF,
            "start_sample": start, "end_sample": end}


@pytest.mark.parametrize("s", [14, 19, 21, 22, 1000, 30000])
@pytest.mark.parametrize("old_style", [False, True])
def test_overlap_of_exactly_min_length_is_removed_at_any_position(s, old_style):
    bursts = {0: [_seq(s, s + 22, old_style)], 1: [_seq(s + 11, s + 33, old_style)]}
    clean = remove_overlapping_bursts(bursts, AVERAGES)
    assert clean == {0: [], 1: []}


@pytest.mark.parametrize("s", [14, 19, 1000])
def test_overlap_one_sample_short_is_kept(s):
    bursts = {0: [_seq(s, s + 22)], 1: [_seq(s + 12, s + 34)]}
    clean = remove_overlapping_bursts(bursts, AVERAGES)
    assert [len(clean[0]), len(clean[1])] == [1, 1]


def test_same_cluster_overlaps_are_ignored():
    bursts = {0: [_seq(100, 144), _seq(110, 154)], 1: []}
    clean = remove_overlapping_bursts(bursts, AVERAGES)
    assert len(clean[0]) == 2
