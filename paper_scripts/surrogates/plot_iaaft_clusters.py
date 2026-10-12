#!/usr/bin/env python
"""Figure 9 and the template-match numbers of Section 2.1 (IAAFT surrogates).

The IAAFT repertoire (SURROGATE_DIR/full/iaaft/clusters.npz) comes from the original surrogate
pipeline (surrogates/README.md: pools.py -> full_route.py). Its clusters are matched one-to-one
to the paper's 12 templates (Hungarian on the back-fit similarity, surrogates/compare.py), each
matched consensus is flipped to the template's polarity, and arrows are classified against the
paper repertoire's longest arrow at 0.7. Null: electrode labels of all surrogate clusters
permuted and the matching redone (200 permutations).

    cd paper_scripts && python surrogates/plot_iaaft_clusters.py
"""
from __future__ import annotations

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
import paper_config                                                       # noqa: E402
import stppy.io                                                           # noqa: E402
import stppy.visualization                                                # noqa: E402
from linear_distance import best_shift_correlation, frame_normalise       # noqa: E402

TRAVEL_FRAC = 0.7
N_PERM = 200


def main() -> int:
    templates = stppy.io.load_pickle(str(paper_config.RUN_DIR / "cluster_averages.pkl"))
    keys = sorted(templates)
    ref = [templates[k] for k in keys]
    reference_arrow = float(compare.arrow_magnitudes(ref).max())
    z = np.load(paper_config.SURROGATE_DIR / "full" / "iaaft" / "clusters.npz")
    sizes = z["sizes"]
    surr = [z[f"consensus_{i:02d}"] for i in range(len(z["keep"]))]
    sim = compare.similarity_matrix(ref, surr)
    rows, cols = linear_sum_assignment(-sim)
    matched, n_k, r_k = {}, {}, {}
    for i, j in sorted(zip(rows, cols)):
        cmax, cmin, _ = best_shift_correlation(frame_normalise(ref[i]), frame_normalise(surr[j]))
        matched[keys[i]] = (1 if cmax >= -cmin else -1) * surr[j]
        n_k[keys[i]] = int(sizes[j])
        r_k[keys[i]] = round(float(sim[i, j]), 2)

    fig = stppy.visualization.plot_clusters_with_arrows(
        matched, str(compare.POS_FILE), cluster_sizes=n_k,
        title="Clusters recovered from IAAFT surrogate data",
        classify_frac=TRAVEL_FRAC, reference=reference_arrow)
    out = paper_config.results("surrogates")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"figure9_iaaft_clusters.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    null = compare.permutation_null(ref, surr, np.random.default_rng(0), n_perm=N_PERM)
    m = sim[rows, cols]
    text = "\n".join([
        f"IAAFT repertoire: {len(surr)} clusters matched to the {len(ref)} paper templates",
        f"  matched |r| {m.min():.4f}-{m.max():.4f}, mean {m.mean():.4f}",
        f"  null (electrode labels permuted, {N_PERM}x): mean {null.mean():.4f}, 95th percentile {np.percentile(null, 95):.4f}",
        f"  per template |r|: {r_k}",
        f"  cluster sizes: {n_k}",
    ])
    print(text)
    (out / "figure9_iaaft_clusters.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
