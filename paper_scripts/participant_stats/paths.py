"""Input and output locations shared by the participant-level scripts."""

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PAPER_SCRIPTS = PACKAGE_DIR.parent
REPO_ROOT = PAPER_SCRIPTS.parent

# main_analysis / psiconnect_utils live in paper_scripts/, stppy in the repo root.
for _p in (PAPER_SCRIPTS, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import paper_config  # noqa: E402  (all paths: paper_config.py)

PAPER_DIR = paper_config.RUN_DIR      # notebook products (sequences, clusters, ...)
RAW_DIR = paper_config.RAW_DIR        # preprocessed .set recordings

RESULTS = paper_config.RESULTS_DIR
OUT_PARTICIPANT_LEVEL = RESULTS / "participant_level"
OUT_COUNTS = RESULTS / "participant_counts"
OUT_TABLE1 = RESULTS / "table1"
RECORDINGS_CSV = OUT_PARTICIPANT_LEVEL / "recordings.csv"
