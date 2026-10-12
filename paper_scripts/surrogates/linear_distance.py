"""The back-fit's own similarity, used as a clustering distance (the "linear route").

The paper's back-fit scores a data window against a template with
`stppy.detection.correlation._vectorized_frame_correlations`: the spatial Pearson correlation
of every frame (across electrodes), averaged over frames, with constant frames counted as 0.
Here the same number is computed between two avalanches, sliding the shorter inside the
longer and keeping the best offset -- exactly how a template slides along continuous data.

    D(A, B)  = 1 - max_s c(A, B, s)          in [0, 2]
    D(A, -B) = 1 + min_s c(A, B, s)          the polarity-flipped copy, for free

so the matrix over originals + flips has the same [[A, B], [B, A]] layout as the paper's
UOT+subDTW matrix and goes through `simple_hierarchical_clustering` unchanged. The cut is the
back-fit's threshold, D = 1 - 0.825: two events share a cluster when, on average, they would
back-fit each other.
"""

from __future__ import annotations

import numpy as np

VALID_STD = 1e-6          # _vectorized_frame_correlations' constant-frame rule


def frame_normalise(x: np.ndarray) -> np.ndarray:
    """Per-frame z-scores scaled so that a column dot product is the frame's Pearson r."""
    x = np.asarray(x, dtype=np.float64)
    c = x - x.mean(0, keepdims=True)
    sd = x.std(0)
    out = np.zeros_like(c)
    ok = sd > VALID_STD
    out[:, ok] = c[:, ok] / (sd[ok] * np.sqrt(x.shape[0]))
    return out


def _diag_means(C: np.ndarray) -> np.ndarray:
    """Mean of C[k, k+s] over k, for every offset s of the shorter (rows) in the longer."""
    Ts, Tl = C.shape
    k = np.arange(Ts)[:, None]
    s = np.arange(Tl - Ts + 1)[None, :]
    return C[k, k + s].mean(0)


def best_shift_correlation(fa: np.ndarray, fb: np.ndarray) -> tuple[float, float, int]:
    """(max_s c, min_s c, argmax offset of the shorter inside the longer)."""
    if fa.shape[1] > fb.shape[1]:
        fa, fb = fb, fa
    c = _diag_means(fa.T @ fb)
    return float(c.max()), float(c.min()), int(c.argmax())


def linear_distance_rows(normed: list[np.ndarray], rows) -> list[tuple[int, np.ndarray,
                                                                         np.ndarray]]:
    """Upper-triangle rows of the originals block A and the flip block B."""
    n = len(normed)
    out = []
    for i in rows:
        A = np.zeros(n)
        B = np.zeros(n)
        for j in range(i, n):
            cmax, cmin, _ = best_shift_correlation(normed[i], normed[j])
            if j > i:
                A[j] = 1.0 - cmax
            B[j] = 1.0 + cmin
        out.append((i, A, B))
    return out


def align_to_medoid(medoid: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x on the medoid's frame grid at the best offset; frames x does not cover are NaN."""
    Tm, Tx = medoid.shape[1], x.shape[1]
    _cmax, _cmin, s = best_shift_correlation(frame_normalise(medoid), frame_normalise(x))
    if Tx >= Tm:
        return x[:, s:s + Tm].astype(np.float64)
    out = np.full(medoid.shape, np.nan)
    out[:, s:s + Tx] = x
    return out
