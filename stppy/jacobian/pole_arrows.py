"""One net-movement arrow per polarity of a spatiotemporal pattern.

For each frame and each pole (positive / negative part of the pattern) the
propagation field -grad(phi) is averaged over the pole's electrodes with |V|^2
weights, and the pole's |V|^2 centroid is recorded. The per-frame pole vectors are
then averaged over time, weighted by the pole's energy sum(|V|^2) in each frame, to
give a single "global movement" vector per pole.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .phase_gradient import TEMPORAL_SMOOTH_SIGMA, propagation_field

# A pole whose peak is below this fraction of the pattern's peak |V| is treated as
# absent in that frame: it gets no centroid and does not contribute to the average.
POLE_VALID_FRAC = 0.15


@dataclass(frozen=True)
class PoleArrow:
    """Time-averaged propagation of one pole.

    Attributes
    ----------
    centroids : (T, 2) array
        Per-frame |V|^2 centroid of the pole; NaN where the pole is absent.
    vector : (2,) array
        Energy-weighted time average of the pole-mean -grad(phi).
    """

    centroids: np.ndarray
    vector: np.ndarray

    @property
    def magnitude(self) -> float:
        return float(np.hypot(*self.vector))


def pole_arrows(
    pattern: np.ndarray,
    xy: np.ndarray,
    valid_frac: float = POLE_VALID_FRAC,
    temporal_sigma: float = TEMPORAL_SMOOTH_SIGMA,
) -> Dict[str, PoleArrow]:
    """Net-movement arrow for the positive and the negative pole of a pattern.

    Parameters
    ----------
    pattern : (n_elec, T) array
        Spatiotemporal pattern, e.g. a cluster consensus average.
    xy : (n_elec, 2) array
        2D electrode coordinates (use the plotting frame, e.g. MNE topomap coords).
    valid_frac : float
        Pole-presence threshold as a fraction of the pattern's peak |V|.
    temporal_sigma : float
        Temporal smoothing before the Hilbert transform (samples).

    Returns
    -------
    dict with keys ``'pos'`` and ``'neg'`` mapping to :class:`PoleArrow`.
    """
    pattern = np.asarray(pattern, dtype=float)
    xy = np.asarray(xy, dtype=float)
    u, v = propagation_field(pattern, xy, temporal_sigma=temporal_sigma)
    floor = valid_frac * float(np.abs(pattern).max())
    n_frames = pattern.shape[1]

    arrows = {}
    for name, sign in (('pos', 1.0), ('neg', -1.0)):
        centroids = np.full((n_frames, 2), np.nan)
        vectors = np.zeros((n_frames, 2))
        energy = np.zeros(n_frames)
        for t in range(n_frames):
            pole = np.clip(sign * pattern[:, t], 0.0, None)
            if floor <= 0 or pole.max() < floor:
                continue
            energy[t] = np.sum(pole ** 2)
            w = pole ** 2 / energy[t]
            centroids[t] = w @ xy
            vectors[t] = (w @ u[:, t], w @ v[:, t])
        total = energy.sum()
        vector = energy @ vectors / total if total > 0 else np.zeros(2)
        arrows[name] = PoleArrow(centroids=centroids, vector=vector)
    return arrows


# A cluster is travelling if its stronger pole's time-averaged arrow reaches this
# fraction of the reference (the longest arrow among the clusters being compared).
TRAVELLING_FRAC = 0.75


def classify_propagation(
    arrows: Dict[str, PoleArrow], reference: float, standing_frac: float = TRAVELLING_FRAC,
) -> str:
    """'travelling' if either pole's net arrow reaches standing_frac * reference.

    The threshold is relative: ``reference`` is normally the longest pole arrow across
    all clusters shown together, so labels compare clusters within one repertoire.
    """
    if reference <= 0:
        return 'standing'
    strongest = max(a.magnitude for a in arrows.values())
    return 'travelling' if strongest >= standing_frac * reference else 'standing'
