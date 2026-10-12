#!/usr/bin/env python3
"""Figure and statistics of the sequence spectral analysis (reads sequence_spectra.npz).

  a    power spectrum around oscillating and stable sequences and around random windows
  b,c  time-frequency power around oscillating / stable sequences, dB relative to random windows
  d    height of the alpha peak above the aperiodic (1/f) background, per cluster

Unit of analysis = participant: spectra are averaged over a participant's sequences
(>= MIN_SEQUENCES) before any feature or test. Tests are paired sign-flip permutations over
participants, marked * p < 0.05, ** p < 0.01, *** p < 0.001 (psiconnect_utils.sig_stars).
Alpha peak heights and peak frequencies, with all tests, go to sequence_spectra_stats.txt.
"""

import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from psiconnect_utils import sig_stars  # noqa: E402
from spectral_tools import ALPHA_BAND, alpha_features, sign_flip_test, unit_means  # noqa: E402

import paper_config  # noqa: E402  (all paths: paper_config.py)

OUT_DIR = paper_config.results("spectral")

POLARITIES = ("oscillating", "stable")
FREQ_RANGE = (1.0, 40.0)
MIN_SEQUENCES = 3  # per participant and group
MIN_PARTICIPANTS = 5  # fewer: dots only, no violin, no test
N_PERM = 10_000
SEED = 0
COLORS = {"oscillating": "#eb6834", "stable": "#2a78d6", "random": "#8a8985"}
INK, INK2, BAND_SHADE = "#0b0b0b", "#52514e", "#f1efe8"
FEATURES = (
    "alpha height (dB above 1/f, max in 8-13 Hz)",
    "peak frequency (Hz, peaks >= 1 dB above 1/f)",
)


class Spectra(NamedTuple):
    freqs: np.ndarray
    log_seq: np.ndarray  # (n_seq, n_freq) log10 power around each sequence
    log_null: np.ndarray  # (n_seq, n_freq) log10 power of its random windows
    tfr_db: np.ndarray  # (n_seq, n_tfreq, n_time) sequence / random windows, dB
    tfr_freqs: np.ndarray
    tfr_times: np.ndarray
    polarity: np.ndarray
    subject: np.ndarray
    cluster: np.ndarray


def load_spectra(path: Path) -> Spectra:
    z = np.load(path)
    return Spectra(
        z["freqs"],
        np.log10(z["psd_seq"]),
        np.log10(z["psd_null"]),
        10 * np.log10(z["tfr_seq"] / z["tfr_null"]),
        z["tfr_freqs"],
        z["tfr_times"],
        z["polarity"],
        z["subject"],
        z["cluster"],
    )


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def plot_spectrum(ax: plt.Axes, sp: Spectra) -> None:
    band = (sp.freqs >= FREQ_RANGE[0]) & (sp.freqs <= FREQ_RANGE[1])
    f = sp.freqs[band]
    nulls = []
    for pol in POLARITIES:
        mask = sp.polarity == pol
        seq, ids = unit_means(sp.log_seq, sp.subject, mask, MIN_SEQUENCES)
        nulls.append(unit_means(sp.log_null, sp.subject, mask, MIN_SEQUENCES)[0])
        mean, sem = seq[:, band].mean(0), seq[:, band].std(0, ddof=1) / np.sqrt(len(seq))
        ax.plot(f, mean, color=COLORS[pol], lw=2, label=f"{pol} (n={len(ids)})")
        ax.fill_between(f, mean - sem, mean + sem, color=COLORS[pol], alpha=0.18, lw=0)
    ax.plot(
        f,
        np.concatenate(nulls)[:, band].mean(0),
        color=COLORS["random"],
        lw=2,
        ls="--",
        label="random windows",
    )
    ax.axvspan(*ALPHA_BAND, color=BAND_SHADE, zorder=0)
    ax.set_xlabel("Frequency (Hz)", color=INK)
    ax.set_ylabel(r"log$_{10}$ power (V$^2$/Hz)", color=INK)
    ax.set_title("a  Power spectrum around sequences", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)


def plot_tfr(fig: plt.Figure, axes: list[plt.Axes], sp: Spectra) -> None:
    flat = sp.tfr_db.reshape(len(sp.tfr_db), -1)
    maps = [
        unit_means(flat, sp.subject, sp.polarity == pol, MIN_SEQUENCES)[0]
        .mean(0)
        .reshape(len(sp.tfr_freqs), len(sp.tfr_times))
        for pol in POLARITIES
    ]
    lim = max(np.abs(m).max() for m in maps)
    for ax, pol, tfr, tag in zip(axes, POLARITIES, maps, "bc"):
        im = ax.pcolormesh(
            sp.tfr_times * 1000,
            sp.tfr_freqs,
            tfr,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            shading="nearest",
        )
        ax.axvline(0, color=INK, lw=0.6, ls=":")
        ax.set_xlabel("Time from sequence centre (ms)", color=INK)
        ax.set_ylabel("Frequency (Hz)", color=INK)
        ax.set_title(f"{tag}  Time-frequency, {pol} sequences", loc="left", color=INK, fontsize=11)
        ax.tick_params(colors=INK2, labelsize=9)
    fig.colorbar(im, ax=axes[-1], fraction=0.05, pad=0.03).set_label(
        "Sequence / random windows (dB)", color=INK
    )


def participant_features(sp: Spectra) -> dict[tuple, dict[str, tuple[float, float]]]:
    """{group: {participant: (alpha height, peak frequency)}}; groups are 'random',
    ('all', polarity) and (cluster, polarity)."""

    def features(log_psd: np.ndarray, mask: np.ndarray) -> dict[str, tuple[float, float]]:
        means, ids = unit_means(log_psd, sp.subject, mask, MIN_SEQUENCES)
        return {u: alpha_features(sp.freqs, row) for u, row in zip(ids, means)}

    groups = {("random",): features(sp.log_null, np.ones(len(sp.polarity), bool))}
    for pol in POLARITIES:
        groups[("all", pol)] = features(sp.log_seq, sp.polarity == pol)
        for c in np.unique(sp.cluster):
            by_cluster = features(sp.log_seq, (sp.cluster == c) & (sp.polarity == pol))
            if by_cluster:
                groups[(int(c), pol)] = by_cluster
    return groups


def paired_test(
    a: dict[str, float], b: dict[str, float], rng: np.random.Generator
) -> tuple[int, float, float]:
    """(n, median of a - b, p) over participants with finite values in both; p = NaN if
    n < MIN_PARTICIPANTS."""
    diff = np.array([a[u] - b[u] for u in a if u in b and np.isfinite(a[u]) and np.isfinite(b[u])])
    if len(diff) < MIN_PARTICIPANTS:
        return len(diff), float("nan"), float("nan")
    return len(diff), float(np.median(diff)), sign_flip_test(diff, N_PERM, rng)


def comparisons(groups: dict) -> list[tuple[tuple, tuple]]:
    """Oscillating vs random windows, oscillating vs stable (pooled and within each cluster)."""
    pairs = [(("all", "oscillating"), ("random",)), (("all", "oscillating"), ("all", "stable"))]
    clusters = sorted(g[0] for g in groups if isinstance(g[0], int) and g[1] == "stable")
    return pairs + [
        ((c, "oscillating"), (c, "stable")) for c in clusters if (c, "oscillating") in groups
    ]


def _name(group: tuple) -> str:
    if group == ("random",):
        return "random windows"
    return f"{'all clusters' if group[0] == 'all' else f'cluster {group[0]}'} {group[1]}"


def feature_stats(groups: dict, k: int, rng: np.random.Generator) -> tuple[list[str], dict]:
    """Report lines for feature ``k`` (0 = height, 1 = peak) and {(a, b): p} of the comparisons."""
    lines, pvals = [f"{FEATURES[k]}: median [IQR] (participants with a value / n)"], {}
    for group, per_unit in groups.items():
        v = np.array([x[k] for x in per_unit.values() if np.isfinite(x[k])])
        q = np.percentile(v, [50, 25, 75]) if len(v) else [np.nan] * 3
        lines.append(
            f"  {_name(group):<26s} {q[0]:6.2f} [{q[1]:6.2f}, {q[2]:6.2f}]  "
            f"({len(v)}/{len(per_unit)})"
        )
    lines.append("  paired sign-flip tests, two-sided:")
    for a, b in comparisons(groups):
        n, med, p = paired_test(
            {u: x[k] for u, x in groups[a].items()}, {u: x[k] for u, x in groups[b].items()}, rng
        )
        if np.isfinite(p):
            pvals[(a, b)] = p
            lines.append(
                f"    {_name(a)} - {_name(b)}: n={n} median diff={med:+.2f} "
                f"p={p:.4f} {sig_stars(p)}"
            )
        else:
            lines.append(f"    {_name(a)} - {_name(b)}: n={n} paired participants, not tested")
    return lines, pvals


def _violin(
    ax: plt.Axes, x: float, values: list[float], color: str, rng: np.random.Generator
) -> None:
    v = np.array(values)
    if len(v) >= MIN_PARTICIPANTS:
        body = ax.violinplot([v], positions=[x], widths=0.34, showextrema=False)["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.28)
        ax.hlines(np.median(v), x - 0.12, x + 0.12, color=color, lw=2.2, zorder=4)
    ax.scatter(
        x + rng.uniform(-0.06, 0.06, len(v)), v, s=11, color=color, lw=0, alpha=0.8, zorder=3
    )


def _bracket(ax: plt.Axes, x1: float, x2: float, y: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y - 0.6, y, y, y - 0.6], color=INK2, lw=0.9)
    ax.text((x1 + x2) / 2, y + 0.2, text, ha="center", va="bottom", fontsize=12, color=INK)


def plot_alpha_height(
    ax: plt.Axes, clusters: list[int], groups: dict, pvals: dict, rng: np.random.Generator
) -> None:
    """Violins of the alpha peak height; every cluster gets a column, empty if it has no data."""
    x = {("random",): 0.0, ("all", "oscillating"): 0.8, ("all", "stable"): 1.2}
    for i, c in enumerate(clusters):
        x[(c, "oscillating")], x[(c, "stable")] = i + 1.8, i + 2.2
    for group, per_unit in groups.items():
        color = COLORS["random" if group == ("random",) else group[1]]
        _violin(ax, x[group], [h for h, _ in per_unit.values()], color, rng)
    for (a, b), p in pvals.items():
        _bracket(ax, x[a], x[b], 25.0 if b == ("random",) else 22.0, sig_stars(p))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(1.6, color="#d6d4cc", lw=0.8)
    ax.set_xticks(
        range(len(clusters) + 2), ["random\nwindows", "all\nclusters"] + [str(c) for c in clusters]
    )
    ax.set_xlim(-0.6, len(clusters) + 1.6)
    ax.set_ylim(-4, 28)
    ax.set_xlabel("Cluster", color=INK)
    ax.set_ylabel("Alpha power above 1/f\nbackground (dB, 8-13 Hz)", color=INK)
    ax.set_title(
        "d  Height of the alpha peak per cluster (one dot per participant)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    for name, color in COLORS.items():
        ax.scatter([], [], color=color, s=18, label="random windows" if name == "random" else name)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3)
    _style(ax)


def main() -> int:
    sp = load_spectra(OUT_DIR / "sequence_spectra.npz")
    groups = participant_features(sp)
    test_rng = np.random.default_rng(SEED)
    height_lines, height_p = feature_stats(groups, 0, test_rng)
    peak_lines, _ = feature_stats(groups, 1, test_rng)

    fig = plt.figure(figsize=(14, 8.4), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1, 1.05], width_ratios=[1, 1, 1.12], hspace=0.42, wspace=0.3
    )
    plot_spectrum(fig.add_subplot(grid[0, 0]), sp)
    plot_tfr(fig, [fig.add_subplot(grid[0, i]) for i in (1, 2)], sp)
    clusters = [int(c) for c in np.unique(sp.cluster)]
    plot_alpha_height(
        fig.add_subplot(grid[1, :]), clusters, groups, height_p, np.random.default_rng(SEED)
    )
    fig.savefig(OUT_DIR / "sequence_spectra.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "sequence_spectra.pdf", bbox_inches="tight")

    header = [
        f"{len(sp.polarity)} sequences; unit = participant (>= {MIN_SEQUENCES} sequences); "
        f"{N_PERM} permutations",
        "",
    ]
    report = "\n".join(header + height_lines + [""] + peak_lines) + "\n"
    (OUT_DIR / "sequence_spectra_stats.txt").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
