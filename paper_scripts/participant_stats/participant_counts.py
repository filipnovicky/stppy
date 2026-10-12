#!/usr/bin/env python3
"""Participant counts for every drug comparison (Table 2, Figure 3, Results text).

For each slice of the sequences (overall, per task, per cluster) reports how
many participants contribute sequences in control, in psilocybin, in either,
and in both. "Both" is the N of the within-subject permutation test in
main_analysis.py, which drops single-session participants.

    cd paper_scripts && python -m participant_stats.participant_counts

Outputs -> RESULTS_DIR/participant_counts/participant_counts.csv (paper_config.py)
"""

from __future__ import annotations

import pandas as pd

from . import paths
from .data import BINARY, load_sequences

TASK_ORDER = ["meditation", "music", "restingstate", "video"]


def slice_counts(seqs: pd.DataFrame, label: str) -> dict:
    """Participant and sequence counts of one slice, by session."""
    ctrl = set(seqs.loc[seqs["drug"] == "ctrl", "subj"])
    psi = set(seqs.loc[seqs["drug"] == "psi", "subj"])
    paired = ctrl & psi
    return {"slice": label,
            "participants_ctrl": len(ctrl), "participants_psi": len(psi),
            "participants_any": len(ctrl | psi), "participants_paired": len(paired),
            "sequences": len(seqs), "sequences_paired": int(seqs["subj"].isin(paired).sum())}


def count_table(seqs: pd.DataFrame) -> pd.DataFrame:
    seqs = seqs[seqs["polarity"].isin(BINARY)]
    rows = [slice_counts(seqs, "overall")]
    rows += [slice_counts(seqs[seqs["task"] == t], f"task:{t}") for t in TASK_ORDER]
    rows += [slice_counts(seqs[seqs["cluster"] == c], f"cluster:{c}")
             for c in sorted(seqs["cluster"].unique())]
    return pd.DataFrame(rows)


def main() -> None:
    table = count_table(load_sequences())
    paths.OUT_COUNTS.mkdir(parents=True, exist_ok=True)
    out = paths.OUT_COUNTS / "participant_counts.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
