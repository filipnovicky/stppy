#!/usr/bin/env python3
"""Does the session change in alpha power track the change in sequence polarity? (Figure 4)

  a  power spectrum around oscillating and stable sequences and around random windows
     (unchanged from plot_sequence_spectra.py)
  b  alpha-band (8-13 Hz) time course around oscillating and stable sequences, dB relative to
     random windows: the existing Morlet power of sequence_spectra.npz averaged over 8-13 Hz,
     then over each participant's sequences (>= 3). Shows whether alpha is raised only during
     the ~90 ms sequence or for hundreds of ms around it; the flanks (>= FLANK_S from the
     centre) contain no sample of the sequence, so their alpha cannot be produced by it.
  c  one dot per participant: change in whole-recording alpha power (psilocybin - baseline)
     against the change in the share of oscillating sequences (psilocybin - baseline)

Alpha power is the participant's ordinary EEG alpha, independent of any sequence: the height of
the alpha peak above the aperiodic (1/f) background (spectral_tools.alpha_features) of the
whole-recording spectrum (compute_recording_alpha.py). Sequences are the paper's back-fitted
sequences after overlap removal, exactly as in Table 2.

Pairing. A participant enters if both sessions exist; only the tasks recorded in BOTH sessions
are used, so a task missing in one session cannot masquerade as a session effect. The share of
oscillating sequences = oscillating / (oscillating + stable) over those tasks, and needs at
least MIN_SEQUENCES sequences in each session. Rates (per minute of recording) of oscillating
and stable sequences are reported in the statistics file.

Statistics, participant = unit: Spearman correlation with a permutation p (the change in
polarity shuffled across participants); session change of alpha by a paired sign-flip test.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[1])]
import stppy.io  # noqa: E402
from main_analysis import POLARITY_RELABEL, remove_overlapping_bursts  # noqa: E402
from plot_sequence_spectra import COLORS, INK, load_spectra, plot_spectrum, _style  # noqa: E402
from psiconnect_utils import parse_condition  # noqa: E402
from spectral_tools import ALPHA_BAND, alpha_features, sign_flip_test, unit_means  # noqa: E402

import paper_config  # noqa: E402  (all paths: paper_config.py)

PAPER_DIR = paper_config.RUN_DIR
OUT_DIR = paper_config.results("spectral")
POLARITIES = ("oscillating", "stable")
SESSIONS = ("ctrl", "psi")
MIN_SEQUENCES = 3  # per participant and session, for the share of oscillating sequences
N_PERM = 10_000
SEED = 0
DOT = "#eb6834"
SEQUENCE_HALF_S = 0.044  # a two-detection sequence spans ~2 x 44 ms around its centre
PANEL_TITLE_SIZE = 13  # bold "(a) Title" like the main results figure
FLANK_S = 0.5            # flank windows: 0.5-1.0 s before and after the centre


def sequence_counts() -> dict[str, dict[str, int]]:
    """dataset -> {'oscillating': n, 'stable': n} after the paper's overlap removal."""
    raw = stppy.io.load_pickle(str(PAPER_DIR / "sequences_by_cluster.pkl"))
    averages = stppy.io.load_pickle(str(PAPER_DIR / "cluster_averages.pkl"))
    for seqs in raw.values():
        for s in seqs:
            s["polarity"] = POLARITY_RELABEL.get(s["polarity"], s["polarity"])
    counts: dict[str, dict[str, int]] = {}
    for seqs in remove_overlapping_bursts(raw, averages).values():
        for s in seqs:
            if s["polarity"] in POLARITIES:
                c = counts.setdefault(s["dataset"], dict.fromkeys(POLARITIES, 0))
                c[s["polarity"]] += 1
    return counts


def participant_sessions(rec: dict, counts: dict) -> dict[str, dict[str, dict]]:
    """subject -> session -> {alpha_db, oscillating, stable, minutes} over shared tasks."""
    out = {}
    for subj in np.unique(rec["subject"]):
        tasks = [set(rec["task"][(rec["subject"] == subj) & (rec["drug"] == s)]) for s in SESSIONS]
        shared = tasks[0] & tasks[1]
        if not shared:
            continue
        out[subj] = {}
        for s in SESSIONS:
            idx = np.flatnonzero((rec["subject"] == subj) & (rec["drug"] == s)
                                 & np.isin(rec["task"], sorted(shared)))
            log_psd = np.log10(rec["psd"][idx].mean(axis=0))
            n = {p: sum(counts.get(rec["dataset"][i], {}).get(p, 0) for i in idx)
                 for p in POLARITIES}
            out[subj][s] = dict(alpha_db=alpha_features(rec["freqs"], log_psd)[0],
                                minutes=rec["duration_s"][idx].sum() / 60, **n)
    return out


def changes(sessions: dict) -> dict[str, np.ndarray]:
    """Per-participant psilocybin - baseline changes; share needs MIN_SEQUENCES per session."""
    rows = []
    for subj, d in sorted(sessions.items()):
        c, p = d["ctrl"], d["psi"]
        tot_c, tot_p = c["oscillating"] + c["stable"], p["oscillating"] + p["stable"]
        share = (p["oscillating"] / tot_p - c["oscillating"] / tot_c) * 100 \
            if min(tot_c, tot_p) >= MIN_SEQUENCES else np.nan
        rows.append((subj, p["alpha_db"] - c["alpha_db"], share,
                     p["oscillating"] / p["minutes"] - c["oscillating"] / c["minutes"],
                     p["stable"] / p["minutes"] - c["stable"] / c["minutes"]))
    subj, alpha, share, osc, stab = zip(*rows)
    return dict(subject=np.array(subj), alpha=np.array(alpha), share=np.array(share),
                osc_rate=np.array(osc), stable_rate=np.array(stab))


def spearman_perm(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[int, float, float]:
    """(n, Spearman rho, two-sided permutation p) over the finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y)[0]
    null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(N_PERM)])
    return int(ok.sum()), float(rho), float((1 + np.sum(np.abs(null) >= abs(rho))) / (N_PERM + 1))


def alpha_time_course(sp) -> dict[str, np.ndarray]:
    """polarity -> (n_participants, n_times) alpha-band dB relative to random windows."""
    band = (sp.tfr_freqs >= ALPHA_BAND[0]) & (sp.tfr_freqs <= ALPHA_BAND[1])
    alpha = sp.tfr_db[:, band, :].mean(axis=1)
    return {pol: unit_means(alpha, sp.subject, sp.polarity == pol, MIN_SEQUENCES)[0]
            for pol in POLARITIES}


def flank_stats(sp, course: dict, rng: np.random.Generator) -> list[str]:
    flank = np.abs(sp.tfr_times) >= FLANK_S
    lines = [f"Alpha (8-13 Hz) vs random windows in the flanks (|t| >= {FLANK_S} s), "
             "participant means, sign-flip test:"]
    for pol, rows in course.items():
        v = rows[:, flank].mean(axis=1)
        centre = rows[:, np.argmin(np.abs(sp.tfr_times))]
        lines.append(f"  {pol:<12} n={len(v):>2}  flanks median {np.median(v):+.2f} dB "
                     f"(p = {sign_flip_test(v, N_PERM, rng):.4f}); centre median "
                     f"{np.median(centre):+.2f} dB")
    return lines


def _panel_title(ax: plt.Axes, text: str) -> None:
    """Panel title in the style of the paper's main results figure (Fig. 3): '(a) Title', bold."""
    ax.set_title("", loc="left")  # drop the left-aligned title a shared plotter may have set
    ax.set_title(text, loc="center", color=INK, fontsize=PANEL_TITLE_SIZE, fontweight="bold")


def plot_time_course(ax: plt.Axes, sp, course: dict) -> None:
    t = sp.tfr_times * 1000
    ax.axvspan(-SEQUENCE_HALF_S * 1000, SEQUENCE_HALF_S * 1000, color="#f1efe8", zorder=0,
               label="sequence")
    ax.axhline(0, color="#b9b7ae", lw=0.8)
    for pol, rows in course.items():
        mean, sem = rows.mean(0), rows.std(0, ddof=1) / np.sqrt(len(rows))
        ax.plot(t, mean, color=COLORS[pol], lw=2, label=f"{pol} ({len(rows)} participants)")
        ax.fill_between(t, mean - sem, mean + sem, color=COLORS[pol], alpha=0.18, lw=0)
    ax.set_xlabel("Time from sequence centre (ms)", color=INK)
    ax.set_ylabel("Alpha power (8-13 Hz),\nsequence / random windows (dB)", color=INK)
    _panel_title(ax, "(b) Alpha Around Sequences")
    ax.legend(frameon=False, fontsize=8.5, loc="center right", bbox_to_anchor=(1.0, 0.36))
    _style(ax)


def plot_change(ax: plt.Axes, ch: dict, n: int, rho: float, p: float) -> None:
    ok = np.isfinite(ch["share"])
    ax.axhline(0, color="#b9b7ae", lw=0.8)
    ax.axvline(0, color="#b9b7ae", lw=0.8)
    ax.scatter(ch["alpha"][ok], ch["share"][ok], s=28, color=DOT, edgecolor="white", lw=0.6,
               zorder=3)
    ax.text(0.03, 0.95, f"one dot per participant\nSpearman ρ = {rho:.2f}, p = {p:.3f} (n = {n})",
            transform=ax.transAxes, fontsize=9, color=INK, va="top")
    ax.set_xlabel("Δ alpha power above 1/f (dB), psilocybin − baseline", color=INK)
    ax.set_ylabel("Δ oscillating share of sequences (pp)\npsilocybin − baseline", color=INK)
    _panel_title(ax, "(c) Change in Alpha vs Change in Polarity")
    _style(ax)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)


def main() -> int:
    z = np.load(OUT_DIR / "recording_spectra.npz")
    rec = {k: z[k] for k in z.files}
    ch = changes(participant_sessions(rec, sequence_counts()))
    rng = np.random.default_rng(SEED)
    stats = {k: spearman_perm(ch["alpha"], ch[k], rng) for k in ("share", "osc_rate", "stable_rate")}
    p_alpha = sign_flip_test(ch["alpha"], N_PERM, rng)

    sp = load_spectra(OUT_DIR / "sequence_spectra.npz")
    course = alpha_time_course(sp)
    fig = plt.figure(figsize=(19, 4.8), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.32)
    ax_a = fig.add_subplot(grid[0, 0])
    plot_spectrum(ax_a, sp)
    _panel_title(ax_a, "(a) Power Spectrum Around Sequences")
    plot_time_course(fig.add_subplot(grid[0, 1]), sp, course)
    plot_change(fig.add_subplot(grid[0, 2]), ch, *stats["share"])
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"sequence_spectra_alpha_polarity.{ext}", dpi=300,
                    bbox_inches="tight")

    lines = [f"{len(ch['subject'])} participants with both sessions (shared tasks only); "
             f"{N_PERM} permutations",
             f"Δ alpha (psilocybin - baseline): median {np.median(ch['alpha']):+.2f} dB "
             f"[IQR {np.percentile(ch['alpha'], 25):+.2f}, {np.percentile(ch['alpha'], 75):+.2f}], "
             f"{np.sum(ch['alpha'] < 0)}/{len(ch['alpha'])} decrease, sign-flip p = {p_alpha:.4f}",
             "Spearman correlation with Δ alpha (permutation p):"]
    labels = {"share": "Δ oscillating share (pp, >= 3 sequences per session)",
              "osc_rate": "Δ oscillating rate (per min)", "stable_rate": "Δ stable rate (per min)"}
    for k, (n, rho, p) in stats.items():
        med = np.nanmedian(ch[k])
        lines.append(f"  {labels[k]:<52} n={n:>2}  rho={rho:+.3f}  p={p:.4f}  "
                     f"(median change {med:+.2f})")
    lines += [""] + flank_stats(sp, course, rng)
    report = "\n".join(lines) + "\n"
    (OUT_DIR / "sequence_spectra_alpha_polarity_stats.txt").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
