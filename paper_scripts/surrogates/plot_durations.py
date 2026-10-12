"""Figure 10: log-log duration distribution of every supra-threshold run, real vs IAAFT.

Input: SURROGATE_DIR/pools/<arm>_durations.npz from pools.py (all runs with any |z| >= 3,
before the minimum-duration and neighbour filters); output: RESULTS_DIR/surrogates.
Panel a: P(d) per sample duration; panel b: P_real(d) / P_IAAFT(d) with a 95% band from a paired bootstrap over recordings (each recording's IAAFT draw is
resampled together with the recording). The onset is the shortest duration from which the
band lies above 1 at every longer plotted duration.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper_config  # noqa: E402

POOLS = paper_config.SURROGATE_DIR / "pools"
OUT = paper_config.RESULTS_DIR / "surrogates"
SFREQ = 250.0
MIN_DURATION = 11
MIN_COUNT = 10          # durations with fewer runs than this in either arm are not plotted
N_BOOT = 2000
SEED = 0
ARMS = {"real": ("Real", "#1f1f1f"), "iaaft": ("IAAFT", "#2a6fdb")}


def per_recording_counts(arm: str, n_bins: int) -> np.ndarray:
    """(n_recordings, n_bins) histogram of run durations, one row per recording."""
    z = np.load(POOLS / f"{arm}_durations.npz")
    rows = np.split(z["durations"], np.cumsum(z["per_recording"])[:-1])
    return np.stack([np.bincount(np.minimum(r, n_bins - 1), minlength=n_bins) for r in rows])


def ratio(real: np.ndarray, iaaft: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return (real / real.sum(-1, keepdims=True)) / (iaaft / iaaft.sum(-1, keepdims=True))


def bootstrap(h_real: np.ndarray, h_iaaft: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    n = len(h_real)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    return np.stack([ratio(h_real[i].sum(0), h_iaaft[i].sum(0)) for i in idx])


def onset(lo: np.ndarray, ok: np.ndarray) -> int:
    """Index of the shortest duration from which lo > 1 at every longer plotted duration."""
    above = np.where(ok, lo > 1, True)
    bad = np.flatnonzero(~above)
    return int(bad[-1] + 1) if bad.size else 0


def main() -> dict:
    n_bins = 1 + max(int(np.load(POOLS / f"{a}_durations.npz")["durations"].max()) for a in ARMS)
    h = {a: per_recording_counts(a, n_bins) for a in ARMS}
    tot = {a: h[a].sum(0)[1:].astype(float) for a in ARMS}
    dur_ms = np.arange(1, n_bins) * 1000 / SFREQ
    ok = (tot["real"] >= MIN_COUNT) & (tot["iaaft"] >= MIN_COUNT)
    r = ratio(tot["real"], tot["iaaft"])
    boot = bootstrap(h["real"], h["iaaft"])[:, 1:]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    k = onset(lo, ok)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for a, (label, col) in ARMS.items():
        keep = tot[a] >= MIN_COUNT
        ax1.plot(dur_ms[keep], tot[a][keep] / tot[a].sum(), "o-", ms=3, lw=1.5,
                 color=col, label=label)
    ax1.set_yscale("log")
    ax1.set_ylabel("P(duration)")
    ax1.legend(frameon=False)
    ax1.set_title("a  Duration distribution (429 recordings)", loc="left", fontsize=10)

    ax2.fill_between(dur_ms[ok], lo[ok], hi[ok], color="0.8", lw=0, label="95% bootstrap")
    ax2.plot(dur_ms[ok], r[ok], "o-", ms=3, lw=1.5, color="#1f1f1f", label="real / IAAFT")
    ax2.axhline(1, color="0.6", lw=1, ls="--")
    ax2.set_yscale("log")
    ax2.set_ylabel("P_real / P_IAAFT")
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_title("b  Real relative to IAAFT", loc="left", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("duration (ms)")
        ax.axvline(MIN_DURATION * 1000 / SFREQ, color="0.6", lw=1, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"duration_loglog_real_vs_iaaft.{ext}", dpi=300)
    return dict(dur_ms=dur_ms, r=r, lo=lo, hi=hi, onset=k)


if __name__ == "__main__":
    s = main()
    lines = [f"onset of real > IAAFT (lower 95% bound > 1): {s['dur_ms'][s['onset']]:.0f} ms "
             f"({s['onset'] + 1} samples)"]
    lines += [f"{d:3d} samples ({d * 4:3d} ms): ratio {s['r'][d - 1]:.3f} "
              f"[{s['lo'][d - 1]:.3f}, {s['hi'][d - 1]:.3f}]" for d in range(1, 16)]
    print("\n".join(lines))
    (OUT / "figure10_durations.txt").write_text("\n".join(lines) + "\n")
