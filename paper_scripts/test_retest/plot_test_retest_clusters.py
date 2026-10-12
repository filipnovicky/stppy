#!/usr/bin/env python
"""Appendix A figure: the paper's 12 templates recovered in the test-retest dataset.

Dataset: Wang, Duan, Dong, Ding & Lei (2022), "A test-retest resting, and cognitive state EEG
dataset during multiple subject-driven states", Scientific Data 9, 566
(doi:10.1038/s41597-022-01607-9; OpenNeuro ds004148), eyes-closed resting recordings.

Parameters as in the paper (RUN=ds4148_EC, cut 3.84, minimum size 10). Hungarian one-to-one matching
on the back-fit similarity (best-shift frame-wise |r|, surrogates/compare.py) over the shared
channels, each matched consensus flipped to the template's polarity, pole arrows classified
against the PsiConnect repertoire's longest arrow at 0.7, null = electrode-label permutation
of all candidates with the matching redone (N_PERM, default 50 as in the paper).

Inputs (read only): the ds004148 run directory (paper_config.TEST_RETEST_DIR:
candidates_t3.84_min10.npz, electrode_positions_m.json) and the paper templates
(paper_config.RUN_DIR). Output: paper_config.RESULTS_DIR / test_retest.

    cd paper_scripts && python test_retest/plot_test_retest_clusters.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_scripts" / "surrogates"))
sys.path.insert(0, str(ROOT / "paper_scripts"))
sys.path.insert(0, str(ROOT))

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402
from scipy.optimize import linear_sum_assignment                          # noqa: E402

import compare                                                            # noqa: E402
import stppy.io                                                           # noqa: E402
import stppy.visualization                                                # noqa: E402
from linear_distance import best_shift_correlation, frame_normalise       # noqa: E402
from stppy.jacobian import classify_propagation, pole_arrows              # noqa: E402
from stppy.visualization.cluster_arrows import topomap_coords             # noqa: E402

import paper_config  # noqa: E402  (all paths: paper_config.py)

PAPER_DIR = paper_config.RUN_DIR            # the paper's 12 templates (cluster_averages.pkl)
RUN_DIR = paper_config.TEST_RETEST_DIR      # the ds004148 run (candidates, positions in metres)
OUT_DIR = paper_config.RESULTS_DIR / "test_retest"
T = os.environ.get("T", "3.84")
MIN_SIZE = os.environ.get("MIN_SIZE", "10")
TRAVEL_FRAC = float(os.environ.get("TRAVEL_FRAC", 0.7))
N_PERM = int(os.environ.get("N_PERM", 50))
TITLE = "Clusters recovered from the test-retest dataset (eyes closed)"  # cut T is in the caption


def shared_index(paper_pos: str, run_pos: str) -> tuple[list[int], list[int]]:
    """Indices of the channels present in both montages (case-insensitive names)."""
    p = list(json.loads(Path(paper_pos).read_text()))
    r = list(json.loads(Path(run_pos).read_text()))
    ru = {c.upper(): i for i, c in enumerate(r)}
    pairs = [(i, ru[c.upper()]) for i, c in enumerate(p) if c.upper() in ru]
    return [a for a, _ in pairs], [b for _, b in pairs]


def main() -> int:
    templates = stppy.io.load_pickle(str(PAPER_DIR / "cluster_averages.pkl"))
    keys = sorted(templates)
    ref = [templates[k] for k in keys]
    reference_arrow = float(compare.arrow_magnitudes(ref).max())
    z = np.load(RUN_DIR / f"candidates_t{T}_min{MIN_SIZE}.npz")
    sizes = z["sizes"]
    cands = [z[f"consensus_{i:03d}"] for i in range(len(z["keep"]))]
    # Arrow length is a gradient per unit distance: the montage must be in metres like the paper's.
    run_pos = str(RUN_DIR / "electrode_positions_m.json")
    ip, ir = shared_index(compare.POS_FILE, run_pos)
    sim = compare.similarity_matrix([t[ip] for t in ref], [s[ir] for s in cands])
    rows, cols = linear_sum_assignment(-sim)
    matched, sizes_k, r_k = {}, {}, {}
    for i, j in sorted(zip(rows, cols)):
        cmax, cmin, _ = best_shift_correlation(frame_normalise(ref[i][ip]), frame_normalise(cands[j][ir]))
        matched[keys[i]] = (1 if cmax >= -cmin else -1) * cands[j]
        sizes_k[keys[i]] = int(sizes[j])
        r_k[keys[i]] = round(float(sim[i, j]), 2)
    if len(matched) != len(keys):
        raise RuntimeError("fewer candidates than templates")

    fig = stppy.visualization.plot_clusters_with_arrows(
        matched, run_pos, cluster_sizes=sizes_k, title=TITLE, classify_frac=TRAVEL_FRAC,
        reference=reference_arrow)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / f"test_retest_eyes_closed_t{T}_min{MIN_SIZE}"
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)

    _info, xy = topomap_coords(run_pos, cands[0].shape[0])
    travelling = [k for k in matched
                  if classify_propagation(pole_arrows(matched[k], xy), reference_arrow, TRAVEL_FRAC) == "travelling"]
    null = compare.permutation_null([t[ip] for t in ref], [s[ir] for s in cands],
                                    np.random.default_rng(0), n_perm=N_PERM)
    m = sim[rows, cols]
    print(f"test-retest (ds004148) eyes closed, cut {T}, min size {MIN_SIZE}: {len(cands)} candidate "
          f"clusters, {len(ip)} shared channels")
    print(f"  matched |r| {m.min():.3f}-{m.max():.3f}, mean {m.mean():.4f}; null mean {null.mean():.4f}, "
          f"p95 {np.percentile(null, 95):.4f} ({N_PERM} permutations)")
    print(f"  travelling at {TRAVEL_FRAC} of the PsiConnect reference arrow: {travelling}")
    print(f"  per template |r|: {r_k}")
    print(f"  sizes: {sizes_k}")
    print(f"  -> {base}.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
