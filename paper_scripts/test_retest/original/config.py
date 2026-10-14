"""Shared settings of the test-retest (ds004148) run: stppy quickstart, broadband 1-40 Hz, 250 Hz.

Detection, filtering, alignment and clustering use the quickstart notebook's own values
(stppy_quickstart.ipynb in the repository root). The only addition is the pool cap: ~10k arrays
including polarity flips, i.e. N_ORIG originals drawn evenly across subjects.
"""
from __future__ import annotations

import contextlib
import os
import io as _io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                  # paper_scripts (paper_config)
sys.path.insert(0, str(HERE.parents[2]))                  # repository root (stppy)

import paper_config                                                       # noqa: E402

# The analysis ran with stppy 0.1.0; its detection, alignment and clustering code is identical
# to 0.2.0 (see ../README.md), so either version reproduces the results.
import stppy                                                              # noqa: E402

if not str(getattr(stppy, "__version__", "")).startswith(("0.1.", "0.2.")):
    raise ImportError(f"expected stppy 0.1.x or 0.2.x, got {stppy.__file__}")

# RUN selects dataset x stage; outputs go to paper_config.TEST_RETEST_DIR with a positions JSON
# built from the recordings themselves.
RUNS = {
    # ds004148 rest, cleaned by preprocess_ds004148.py (62 ch, 1-40 Hz, 250 Hz) into
    # paper_config.DS4148_PREPROCESSED. Positions are mm in the MNE head frame already
    # (x=right, y=anterior), so no plot rotation.
    "ds4148_EC": dict(data_dir=paper_config.DS4148_PREPROCESSED, stage="EC", key="data",
                      mm_to_cm=False, positions=None, frame="mne"),
}
RUN = os.environ.get("RUN", "ds4148_EC")
if RUN not in RUNS:
    raise ValueError(f"RUN must be one of {sorted(RUNS)}, got {RUN!r}")
_R = RUNS[RUN]
OUT = paper_config.TEST_RETEST_DIR
DATA_DIR = Path(_R["data_dir"])
STAGE = _R["stage"]
SIGNAL_KEY = _R["key"]
MM_TO_CM = _R["mm_to_cm"]
POSITIONS = _R["positions"] or str(OUT / "electrode_positions.json")
SFREQ = 250.0

CROSSOVER_THRESHOLDS = [-3, 3]
MIN_DURATION = 11
MIN_ACTIVE_NEIGHBORS = 5
N_ORIG = 5000                     # x2 with flips = 10,000 arrays
SEED = 0

# The notebook's 3.5 leaves every event a singleton on this data (median distance 19.4). The
# paper's candidates use CLUSTER_THRESHOLD=3.84 (the same distance quantile as 3.5, README step 6-7).
CLUSTER_THRESHOLD = float(os.environ.get("CLUSTER_THRESHOLD", 5.0))
_TTAG = f"t{CLUSTER_THRESHOLD:g}"
CLUSTER_METHOD = "average"
MIN_CLUSTER_SIZE = 30
DEDUP_THRESHOLD = -0.999

PER_SUBJECT_DIR = OUT / "per_subject"
PEAKS_PKL = OUT / "peak_avalanches.pkl"
TIMES_PKL = OUT / "peak_avalanche_times.pkl"
SUBJECTS_TXT = OUT / "subjects.txt"
BLOCK_DIR = OUT / "blocks"
SIMILARITY = OUT / "similarity.npz"
CLUSTERS_NPZ = OUT / f"clusters_{_TTAG}.npz"
CLUSTERS_CSV = OUT / f"clusters_{_TTAG}.csv"
FIGDIR = OUT / "figures" / _TTAG


def recordings() -> list[Path]:
    files = sorted(DATA_DIR.glob(f"*_{STAGE}.npz"))
    if not files:
        raise FileNotFoundError(f"no {STAGE} recordings in {DATA_DIR}")
    return files


def write_positions() -> None:
    """Positions JSON in data-row order, identical across all recordings of this run."""
    if _R["positions"]:
        return
    import json
    import numpy as np
    names = xyz = None
    for f in recordings():
        with np.load(f, allow_pickle=True) as d:
            n = [str(c) for c in d["channel_names"]]
            if names is None:
                names, xyz = n, np.asarray(d["electrode_xyz"], float)
            elif n != names:
                raise ValueError(f"{f.name}: channel order differs from {recordings()[0].name}")
    if MM_TO_CM:
        xyz = xyz / 10.0
    OUT.mkdir(parents=True, exist_ok=True)
    pos = {c: dict(x=float(p[0]), y=float(p[1]), z=float(p[2])) for c, p in zip(names, xyz)}
    Path(POSITIONS).write_text(json.dumps(pos, indent=1))
    # The same positions in metres (as the PsiConnect montage), read by plot_test_retest_clusters.py.
    pos_m = {c: {k: v / 1000.0 for k, v in p.items()} for c, p in pos.items()}
    (OUT / "electrode_positions_m.json").write_text(json.dumps(pos_m, indent=1))


def plot_positions() -> str:
    """Positions JSON for stppy topomaps, which read x/y/z as the MNE head frame (x=right,
    y=anterior). Positions in a .pos frame (x=anterior, y=left) are rotated here:
    x_mne = -y_pos, y_mne = x_pos; ds004148 (frame='mne') needs no rotation. Distances
    (ground metric) are unaffected."""
    import json
    out = OUT / "electrode_positions_mne_plot.json"
    pos = json.loads(Path(POSITIONS).read_text())
    if _R.get("frame", "pos") == "mne":
        return POSITIONS
    rot = {c: dict(x=-p["y"], y=p["x"], z=p["z"]) for c, p in pos.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rot, indent=1))
    return str(out)


def ground_metric_norm():
    with contextlib.redirect_stdout(_io.StringIO()):
        K, _names = stppy.utils.create_eeg_ground_metric(POSITIONS)
    return K, stppy.utils.normalize_distance_matrix(K, method="mean")


def load_pool():
    """Originals + flips, in load_avalanche_data order (flips are the second half)."""
    with contextlib.redirect_stdout(_io.StringIO()):
        return stppy.io.load_avalanche_data(
            str(PEAKS_PKL), min_time_length=MIN_DURATION, include_flipped=True,
            times_filepath=str(TIMES_PKL),
        )
