"""Paths shared by stppy_quickstart.ipynb and every paper script.

Each path is taken, in this order, from an environment variable of the same name, from an
untracked ``paper_config_local.py`` next to this file (one line per variable, e.g.
``RAW_DIR = "/data/psiconnect"``), or from the defaults below (folders inside the repository).

RAW_DIR          preprocessed PsiConnect EEGLAB .set files (input of the notebook)
RUN_DIR          pipeline products written by the notebook and read by the scripts:
                 datasets.pkl, files.pkl, peak_avalanches.pkl, peak_avalanche_times.pkl,
                 avalanches.pkl, avalanche_times.pkl, similarity_matrix.pkl,
                 cluster_assignments.pkl, cluster_averages.pkl, sequences_by_cluster.pkl
DISTANCE_DIR     UOT + subDTW distances (similarity_matrix.pkl, path_matrix.pkl and the
                 peak_avalanches.pkl they belong to): the notebook loads them from here if
                 present, otherwise computes them (~370 CPU-hours) and saves them here
RESULTS_DIR      figures, tables and statistics (one subfolder per analysis)
SURROGATE_DIR    IAAFT surrogate pipeline (surrogates/README.md): pools/, full/
TEST_RETEST_DIR  test-retest dataset run (test_retest/README.md)
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POS_FILE = REPO / "electrode_positions.json"

try:
    import paper_config_local as _local  # untracked, machine-specific paths
except ImportError:
    _local = None


def _path(name: str, default) -> Path:
    if name in os.environ:
        return Path(os.environ[name])
    if _local is not None and hasattr(_local, name):
        return Path(getattr(_local, name))
    return Path(default)


RAW_DIR = _path("RAW_DIR", REPO / "data" / "psiconnect")
RUN_DIR = _path("RUN_DIR", REPO / "runs" / "psiconnect")
DISTANCE_DIR = _path("DISTANCE_DIR", REPO / "runs" / "distances")
RESULTS_DIR = _path("RESULTS_DIR", RUN_DIR / "results")
SURROGATE_DIR = _path("SURROGATE_DIR", REPO / "runs" / "surrogates")
TEST_RETEST_DIR = _path("TEST_RETEST_DIR", REPO / "runs" / "test_retest")


def results(name: str) -> Path:
    """Output folder of one analysis, created on first use."""
    out = RESULTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return out
