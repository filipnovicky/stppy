#!/usr/bin/env python
"""Appendix A: the dendrogram cut of the test-retest dataset at the paper's percentile.

The paper cuts its dendrogram at distance 3.5; this script finds which fraction of the pairwise
distances lies at or below 3.5 (upper triangle of the polarity-doubled matrix, RUN_DIR) and
returns the distance at the same quantile of the test-retest matrix (TEST_RETEST_DIR/similarity.npz,
same layout), i.e. the cut 3.84 used for Figure 8.

    cd paper_scripts && python test_retest/percentile_cut.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_scripts"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import paper_config  # noqa: E402
import stppy.io  # noqa: E402

PAPER_CUT = 3.5


def upper(S: np.ndarray) -> np.ndarray:
    return S[np.triu_indices(S.shape[0], 1)]


def main() -> int:
    q = float(np.mean(upper(np.asarray(stppy.io.load_pickle(str(paper_config.RUN_DIR / "similarity_matrix.pkl"))))
                      <= PAPER_CUT))
    with np.load(paper_config.TEST_RETEST_DIR / "similarity.npz") as z:
        cut = float(np.quantile(upper(z["similarity_matrix"]), q))
    text = (f"{100 * q:.2f} % of the PsiConnect distances lie at or below the paper cut {PAPER_CUT}; "
            f"the same quantile of the test-retest distances is {cut:.2f}")
    print(text)
    (paper_config.results("test_retest") / "percentile_cut.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
