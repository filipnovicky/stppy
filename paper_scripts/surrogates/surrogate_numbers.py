#!/usr/bin/env python
"""Section 2.1: how real avalanches differ from IAAFT avalanches.

- durations of the analysed avalanches (>= 11 samples, after the 5/8 filter): 90th percentile
  and longest, real (RUN_DIR) vs IAAFT (SURROGATE_DIR/pools);
- share of event pairs within the clustering threshold 3.5 (upper triangle of the
  polarity-doubled distance matrix): real RUN_DIR/similarity_matrix.pkl, IAAFT the stored
  UOT + subDTW blocks of SURROGATE_DIR/full/iaaft, merged as in full_route.py;
- share of events that joined a retained cluster: real (12 clusters) and IAAFT (all kept
  clusters, and only the 12 matched to the paper templates).

    cd paper_scripts && python surrogates/surrogate_numbers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_scripts" / "surrogates"))
sys.path.insert(0, str(ROOT / "paper_scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

import compare  # noqa: E402
import paper_config  # noqa: E402
import stppy.io  # noqa: E402

THRESHOLD = 3.5
N_BLOCKS = 5


def _load(path: Path):
    return stppy.io.load_pickle(str(path))


def durations(peaks) -> np.ndarray:
    return np.array([a.shape[1] for rec in peaks for a in rec])


def merged_iaaft_matrix(folder: Path) -> np.ndarray:
    """[[A, B], [B, A]] of the IAAFT pool from the full-route blocks (original vs flipped)."""
    files = sorted(folder.glob(f"block_*of{N_BLOCKS:02d}.npz"))
    with np.load(files[0]) as z:
        n = int(z["n_orig"])
    A, B = np.zeros((n, n)), np.zeros((n, n))
    for f in files:
        with np.load(f) as z:
            A[z["rows"]], B[z["rows"]] = z["A"], z["B"]
    A = A + A.T
    B = np.triu(B) + np.triu(B, 1).T
    return np.block([[A, B], [B, A]])


def within_threshold(S: np.ndarray) -> float:
    return float(np.mean(S[np.triu_indices(S.shape[0], 1)] <= THRESHOLD))


def main() -> int:
    run, surr_dir = paper_config.RUN_DIR, paper_config.SURROGATE_DIR
    real = durations(_load(run / "peak_avalanches.pkl"))
    iaaft = durations(_load(surr_dir / "pools" / "iaaft_peak_avalanches.pkl"))
    p_real = within_threshold(np.asarray(_load(run / "similarity_matrix.pkl")))
    p_iaaft = within_threshold(merged_iaaft_matrix(surr_dir / "full" / "iaaft"))

    labels = np.asarray(_load(run / "cluster_assignments.pkl"))
    templates = _load(run / "cluster_averages.pkl")
    joined_real = int(np.isin(labels, sorted(templates)).sum())   # mirror clusters excluded: each event once
    z = np.load(surr_dir / "full" / "iaaft" / "clusters.npz")
    surr = [z[f"consensus_{i:02d}"] for i in range(len(z["keep"]))]
    _, cols = linear_sum_assignment(-compare.similarity_matrix([templates[k] for k in sorted(templates)], surr))
    joined_all, joined_12 = int(z["sizes"].sum()), int(z["sizes"][cols].sum())

    text = "\n".join([
        f"avalanches: real {real.size}, IAAFT {iaaft.size}",
        f"duration (samples): 90th percentile real {np.percentile(real, 90):.0f} vs IAAFT "
        f"{np.percentile(iaaft, 90):.0f}; longest {real.max()} vs {iaaft.max()}",
        f"pairs within the threshold {THRESHOLD}: real {100 * p_real:.2f}% vs IAAFT {100 * p_iaaft:.2f}%",
        f"events in retained clusters: real {joined_real}/{real.size} = {100 * joined_real / real.size:.1f}% "
        f"(12 clusters); IAAFT {joined_all}/{iaaft.size} = {100 * joined_all / iaaft.size:.1f}% "
        f"({len(z['keep'])} kept clusters), {joined_12}/{iaaft.size} = {100 * joined_12 / iaaft.size:.1f}% "
        f"(the 12 matched)",
    ])
    print(text)
    (paper_config.results("surrogates") / "section_2_1_numbers.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
