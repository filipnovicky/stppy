"""Compare a cluster repertoire against the paper's 12 templates.

Similarity between two templates is the back-fit's own number (mean frame-by-frame spatial
correlation at the best offset, see linear_distance), taken as |r| because polarity
duplicates are removed before any repertoire is reported. Templates are matched one-to-one
with the Hungarian algorithm; the null permutes the electrode labels of the arm's templates,
which keeps every template's value distribution and destroys its topography.

Propagation is the pole-arrow measure of stppy.jacobian. `classify_propagation` is RELATIVE to the
longest arrow in one figure, so every arm is classified against the PAPER repertoire's
reference -- otherwise "travelling" would mean something different in each arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root (stppy)

from linear_distance import best_shift_correlation, frame_normalise      # noqa: E402
from stppy.jacobian import classify_propagation, pole_arrows              # noqa: E402
from stppy.visualization.cluster_arrows import topomap_coords             # noqa: E402

POS_FILE = str(Path(__file__).resolve().parents[2] / "electrode_positions.json")  # same file as the paper run's
N_PERM = 200


def similarity_matrix(ref: list[np.ndarray], arm: list[np.ndarray]) -> np.ndarray:
    fr = [frame_normalise(t) for t in ref]
    fa = [frame_normalise(t) for t in arm]
    M = np.zeros((len(fr), len(fa)))
    for i, a in enumerate(fr):
        for j, b in enumerate(fa):
            cmax, cmin, _ = best_shift_correlation(a, b)
            M[i, j] = max(abs(cmax), abs(cmin))
    return M


def hungarian(M: np.ndarray) -> np.ndarray:
    """Best one-to-one |r| for every reference template (0 when the arm has fewer)."""
    out = np.zeros(M.shape[0])
    if M.shape[1] == 0:
        return out
    r, c = linear_sum_assignment(-M)
    out[r] = M[r, c]
    return out


def permutation_null(ref: list[np.ndarray], arm: list[np.ndarray], rng,
                     n_perm: int = N_PERM) -> np.ndarray:
    """Mean Hungarian |r| with the arm's electrode labels shuffled, n_perm times."""
    if not arm:
        return np.zeros(n_perm)
    n_el = arm[0].shape[0]
    return np.array([hungarian(similarity_matrix(ref, [t[p] for t in arm])).mean()
                     for p in (rng.permutation(n_el) for _ in range(n_perm))])


def arrow_magnitudes(templates: list[np.ndarray]) -> np.ndarray:
    if not templates:
        return np.zeros(0)
    _info, xy = topomap_coords(POS_FILE, templates[0].shape[0])
    return np.array([max(a.magnitude for a in pole_arrows(t, xy).values())
                     for t in templates])


def classify(templates: list[np.ndarray], reference: float) -> list[str]:
    if not templates:
        return []
    _info, xy = topomap_coords(POS_FILE, templates[0].shape[0])
    return [classify_propagation(pole_arrows(t, xy), reference) for t in templates]


def summarise(ref: list[np.ndarray], arm: list[np.ndarray], ref_arrow_max: float,
              seed: int = 0) -> dict:
    M = similarity_matrix(ref, arm)
    best = hungarian(M)
    null = permutation_null(ref, arm, np.random.default_rng(seed))
    labels = classify(arm, ref_arrow_max)
    return dict(n_templates=len(arm), match=best, match_mean=float(best.mean()),
                match_any=M.max(1) if M.size else np.zeros(len(ref)),
                null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)),
                n_travelling=sum(lbl == "travelling" for lbl in labels),
                arrows=arrow_magnitudes(arm), labels=labels)
