#!/usr/bin/env python3
"""Table 4: recordings retained after preprocessing, mean +- SD duration (minutes) and n,
per session, training group and task.

SD is the population SD (numpy default, ddof=0), as in the paper's table; the sample SD
(ddof=1) is 0.1 larger in two cells (psilocybin, trained: video 0.8, rest 0.7).
Needs recordings.csv (``python -m participant_stats.build_recordings``).

    cd paper_scripts && python -m participant_stats.table4_recordings

Output -> RESULTS_DIR/table4/table4.md
"""

from __future__ import annotations

import pandas as pd

from . import paths
from .data import tag_conditions

TASKS = [("video", "Video"), ("restingstate", "Rest"), ("meditation", "Meditation"), ("music", "Music")]
SESSIONS = [("ctrl", "Baseline"), ("psi", "Psilocybin")]
GROUPS = [(0, "Untrained"), (1, "Trained")]


def table4(rec: pd.DataFrame) -> str:
    rec = tag_conditions(rec).assign(minutes=lambda d: d["duration_s"] / 60.0)
    rows = ["| Session | Group | " + " | ".join(t for _, t in TASKS) + " |",
            "|---|---|" + "---|" * len(TASKS)]
    for drug, session in SESSIONS:
        for trained, group in GROUPS:
            cells = []
            for task, _ in TASKS:
                m = rec.loc[(rec.drug == drug) & (rec.trained == trained) & (rec.task == task), "minutes"]
                cells.append(f"{m.mean():.1f}±{m.std(ddof=0):.1f} (n={m.size})")
            rows.append(f"| {session} | {group} | " + " | ".join(cells) + " |")
    n = rec.groupby("trained")["subj"].nunique()
    rows.append(f"\n{len(rec)} recordings, {rec['subj'].nunique()} participants "
                f"(untrained {n.get(0, 0)}, trained {n.get(1, 0)}).")
    return "\n".join(rows)


def main() -> None:
    md = table4(pd.read_csv(paths.RECORDINGS_CSV))
    out = paths.RESULTS / "table4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table4.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
