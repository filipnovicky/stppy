"""Participant-level test statistics (no data loading, no plotting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

N_PERM = 10000
N_BOOT = 10000


def signflip_p(delta: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> float:
    """Two-sided sign-flip permutation p for mean(delta) = 0."""
    delta = np.asarray(delta, dtype=float)
    if delta.size == 0:
        raise ValueError("signflip_p needs at least one paired difference")
    obs = abs(delta.mean())
    flips = rng.choice([-1.0, 1.0], size=(n_perm, delta.size))
    null = np.abs((flips * delta).mean(axis=1))
    return (np.count_nonzero(null >= obs - 1e-12) + 1) / (n_perm + 1)


def boot_ci(x: np.ndarray, rng: np.random.Generator, fn=np.mean,
            n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of fn(x)."""
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    b = np.apply_along_axis(fn, 1, x[idx])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation of the paired differences."""
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def paired_summary(name: str, ctrl: np.ndarray, psi: np.ndarray,
                   rng: np.random.Generator) -> dict:
    """Unweighted paired contrast: each participant contributes one difference."""
    d = psi - ctrl
    w = stats.wilcoxon(d, zero_method="wilcox") if np.any(d != 0) else None
    lo, hi = boot_ci(d, rng)
    return {"contrast": name, "n_participants": int(d.size),
            "ctrl_mean": ctrl.mean(), "psi_mean": psi.mean(),
            "ctrl_median": np.median(ctrl), "psi_median": np.median(psi),
            "mean_diff": d.mean(), "diff_ci_lo": lo, "diff_ci_hi": hi,
            "n_decrease": int((d < 0).sum()), "n_increase": int((d > 0).sum()),
            "wilcoxon_p": w.pvalue if w else 1.0, "rank_biserial": rank_biserial(d),
            "signflip_p": signflip_p(d, rng)}


def contingency_chi2(rows: np.ndarray, cols: np.ndarray) -> float:
    """Pearson chi-square of the rows x cols contingency table (no continuity correction)."""
    tab = pd.crosstab(rows, cols).to_numpy(float)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return float(((tab - exp) ** 2 / exp).sum())
