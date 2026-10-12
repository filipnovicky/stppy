#!/usr/bin/env python
"""Match a candidate repertoire to the paper's 12 templates and draw it like
stppy020/review/results/full/iaaft/iaaft_matched_to_paper.png.

Same recipe as that figure: Hungarian one-to-one on the back-fit similarity (best-shift
frame-wise |r|, review/code/compare.py), each matched consensus flipped to the template's
polarity, pole arrows classified against the PAPER repertoire's longest arrow at 0.7.
Only difference: the paper is 64-ch PsiConnect, the run has its own montage, so similarity is
computed on the shared channels (case-insensitive names). Drawing uses the run's own montage.

    RUN=ds4148_EC T=3.5 MIN_SIZE=2 python match_paper.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, "/home/novicky/stppy020/review/code")
sys.path.insert(0, "/home/novicky/stppy020")

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
import numpy as np                                                        # noqa: E402
from scipy.optimize import linear_sum_assignment                          # noqa: E402

import compare                                                            # noqa: E402
import stppy.visualization                                                # noqa: E402
from linear_distance import best_shift_correlation, frame_normalise       # noqa: E402

PAPER = Path("/home/novicky/storage_stppy/avgref_off/cluster_averages.pkl")
PAPER_POS = compare.POS_FILE
TRAVEL_FRAC = float(os.environ.get("TRAVEL_FRAC", 0.7))
N_PERM = int(os.environ.get("N_PERM", 50))
RUN = os.environ["RUN"]
T = os.environ["T"]
MIN_SIZE = os.environ.get("MIN_SIZE", "2")
OUT = Path("/home/novicky/stppy010_anphy61_n2/runs") / RUN
TITLES = {"ds4148_EC": "ds004148 eyes closed"}


def shared_index(paper_pos: str, run_pos: str) -> tuple[list[int], list[int]]:
    p = list(json.loads(Path(paper_pos).read_text()))
    r = list(json.loads(Path(run_pos).read_text()))
    ru = {c.upper(): i for i, c in enumerate(r)}
    pairs = [(i, ru[c.upper()]) for i, c in enumerate(p) if c.upper() in ru]
    return [a for a, _ in pairs], [b for _, b in pairs]


def blank_rows(fig, row_keys: list[int], empty: list[int], n_cols: int) -> None:
    """Clear the rows of unmatched templates and label them 'no match'."""
    for k in empty:
        ri = row_keys.index(k)
        axes = fig.axes[ri * n_cols:(ri + 1) * n_cols]
        for ax in axes:
            ax.clear()
            ax.set_axis_off()
        axes[0].text(-0.08, 0.5, f"Cluster {k}\nno match", transform=axes[0].transAxes,
                     ha="right", va="center", fontsize=15, color="0.45")


def main() -> int:
    P = pickle.load(open(PAPER, "rb"))
    pk = sorted(P)
    ref = [P[k] for k in pk]
    R = float(compare.arrow_magnitudes(ref).max())
    z = np.load(OUT / f"candidates_t{T}_min{MIN_SIZE}.npz")
    sizes = z["sizes"]
    S = [z[f"consensus_{i:03d}"] for i in range(len(z["keep"]))]
    # Arrow length is a gradient per unit distance, so the montage must be in the paper's units
    # (metres, head radius ~0.1). ds004148 positions are mm: unconverted, arrows are ~1000x short.
    run_pos = str(OUT / "electrode_positions_m.json")
    ip, ir = shared_index(PAPER_POS, run_pos)
    M = compare.similarity_matrix([t[ip] for t in ref], [s[ir] for s in S])
    r, c = linear_sum_assignment(-M)
    avgs, ns, rs = {}, {}, {}
    for i, j in sorted(zip(r, c)):
        cmax, cmin, _ = best_shift_correlation(frame_normalise(ref[i][ip]), frame_normalise(S[j][ir]))
        avgs[pk[i]] = (1 if cmax >= -cmin else -1) * S[j]
        ns[pk[i]] = int(sizes[j])
        rs[pk[i]] = round(float(M[i, j]), 2)
    matched = dict(avgs)
    # Fewer candidates than templates: Hungarian matches each candidate to its best template and
    # the remaining templates stay empty. They are drawn as blank rows labelled "no match".
    empty = [k for k in pk if k not in avgs]
    n_frames = max(a.shape[1] for a in avgs.values())
    for k in empty:
        avgs[k] = np.zeros((S[0].shape[0], n_frames))
        ns[k] = 0
    title = f"Clusters recovered from {TITLES.get(RUN, RUN)} (threshold {T})"
    fig = stppy.visualization.plot_clusters_with_arrows(
        avgs, run_pos, cluster_sizes=ns, title=title, classify_frac=TRAVEL_FRAC, reference=R)
    blank_rows(fig, sorted(avgs), empty, n_frames)
    base = OUT / f"matched_to_paper_t{T}_min{MIN_SIZE}" if TRAVEL_FRAC == 0.7 else \
        OUT / f"matched_to_paper_t{T}_min{MIN_SIZE}_travel{TRAVEL_FRAC:g}"
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    from stppy.jacobian import classify_propagation, pole_arrows
    from stppy.visualization.cluster_arrows import topomap_coords
    _info, xy = topomap_coords(run_pos, S[0].shape[0])
    travel = [k for k in matched if classify_propagation(pole_arrows(matched[k], xy), R, TRAVEL_FRAC) == "travelling"]
    # Chance level with the SAME number of candidates: best-of-N selection inflates |r|, and the
    # electrode-label permutation (compare.permutation_null) keeps N, so the null inflates equally.
    null = compare.permutation_null([t[ip] for t in ref], [s[ir] for s in S],
                                    np.random.default_rng(0), n_perm=N_PERM)
    print(f"{RUN} t={T} min={MIN_SIZE}: {len(S)} candidates, {len(ip)} shared channels; "
          f"matched |r| mean {M[r, c].mean():.3f} (min {M[r, c].min():.2f}); "
          f"null mean {null.mean():.3f}, p95 {np.percentile(null, 95):.3f} ({N_PERM} perms); "
          f"travelling {travel}")
    print("  per template |r|:", rs, " unmatched:", empty)
    print("  sizes:", ns)
    print(f"  -> {base}.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
