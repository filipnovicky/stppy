#!/usr/bin/env python3
"""Participant-level polarity proportions, occurrence rates and models.

Both with the participant (not the sequence) as the unit:

1. Stable/oscillating PROPORTIONS per participant and session, tested so that
   every participant weighs the same (paired Wilcoxon + sign-flip permutation
   on the unweighted mean difference), plus models with participant as a
   cluster / random effect (binomial GEE, binomial mixed GLM).

2. OCCURRENCE RATES (sequences per minute of retained recording) per
   participant and session. Every recording counts, including those with zero
   sequences. Task enters as a covariate, missing recordings are handled by
   (a) a task-matched paired contrast that uses only tasks a participant has
   in both sessions and (b) Poisson GEE / mixed GLM with log(duration) offset
   on all available recordings.

Needs recordings.csv (``python -m participant_stats.build_recordings``).

    cd paper_scripts && python -m participant_stats.participant_level

Outputs -> RESULTS_DIR/participant_level/ (paper_config.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import paths
from .data import load_sequences, recording_table
from .stats import paired_summary

OUT = paths.OUT_PARTICIPANT_LEVEL
SEED = 0
COLORS = {"ctrl": "#2a78d6", "psi": "#eb6834"}
LABELS = {"ctrl": "Control", "psi": "Psilocybin"}
TASKS = ["meditation", "music", "restingstate", "video"]
TASK_LABELS = {"meditation": "Meditation", "music": "Music",
               "restingstate": "Resting state", "video": "Video"}


# ----------------------------------------------------------------------
# 1. proportions
# ----------------------------------------------------------------------

def session_proportions(rec: pd.DataFrame) -> pd.DataFrame:
    s = rec.groupby(["subj", "drug"])[["stable", "oscillating", "total"]].sum().reset_index()
    s["n_binary"] = s["stable"] + s["oscillating"]
    s["p_stable"] = np.where(s["n_binary"] > 0, s["stable"] / s["n_binary"].clip(lower=1), np.nan)
    return s


def wide_paired(s: pd.DataFrame, value: str) -> pd.DataFrame:
    w = s.pivot(index="subj", columns="drug", values=value)
    return w.dropna(subset=["ctrl", "psi"])


def proportion_models(rec: pd.DataFrame) -> list[dict]:
    """Binomial GEE (exchangeable, participant cluster) and binomial mixed GLM."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    d = rec[(rec["stable"] + rec["oscillating"]) > 0].copy()
    d["n_bin"] = d["stable"] + d["oscillating"]
    d["prop"] = d["stable"] / d["n_bin"]
    out = []
    gee = smf.gee("prop ~ psi + C(task)", groups="subj", data=d, weights=None,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable(),
                  freq_weights=None, var_weights=d["n_bin"]).fit()
    out.append(_coef_row("P(stable): binomial GEE, participant cluster", gee, "psi",
                         n=d["subj"].nunique(), k=len(d)))

    # Sequence-level Bernoulli rows for the mixed model.
    long = d.loc[d.index.repeat(d["n_bin"])].copy()
    long["y"] = np.concatenate([np.r_[np.ones(s), np.zeros(o)]
                                for s, o in zip(d["stable"], d["oscillating"])])
    vc = {"subj": "0 + C(subj)", "subj_drug": "0 + C(subj):psi"}
    glmm = BinomialBayesMixedGLM.from_formula("y ~ psi + C(task)", vc, long).fit_vb()
    i = list(glmm.model.exog_names).index("psi")
    m, sd = glmm.fe_mean[i], glmm.fe_sd[i]
    out.append({"model": "P(stable): binomial mixed GLM, random intercept + drug slope (VB)",
                "effect": "odds ratio psi/ctrl", "estimate": np.exp(m),
                "ci_lo": np.exp(m - 1.96 * sd), "ci_hi": np.exp(m + 1.96 * sd),
                "p": 2 * stats.norm.sf(abs(m / sd)),
                "n_participants": d["subj"].nunique(), "n_obs": len(long)})
    return out


def _coef_row(name: str, res, term: str, n: int, k: int, effect="odds ratio psi/ctrl") -> dict:
    b, (lo, hi) = res.params[term], res.conf_int().loc[term]
    return {"model": name, "effect": effect, "estimate": np.exp(b),
            "ci_lo": np.exp(lo), "ci_hi": np.exp(hi), "p": res.pvalues[term],
            "n_participants": n, "n_obs": k}


# ----------------------------------------------------------------------
# 2. rates
# ----------------------------------------------------------------------

def task_matched_rates(rec: pd.DataFrame) -> pd.DataFrame:
    """Per participant x session rate over the tasks present in BOTH sessions."""
    have = rec.groupby(["subj", "task"])["drug"].nunique()
    both = have[have == 2].reset_index()[["subj", "task"]]
    m = rec.merge(both, on=["subj", "task"])
    cols = ["stable", "oscillating", "total", "n_avalanches", "duration_min"]
    g = m.groupby(["subj", "drug"])[cols].sum().reset_index()
    for c in ["stable", "oscillating", "total", "n_avalanches"]:
        g[f"{c}_rate"] = g[c] / g["duration_min"]
    g["n_tasks"] = g["subj"].map(both.groupby("subj").size())
    return g


def rate_models(rec: pd.DataFrame) -> list[dict]:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM

    d = rec.copy()
    d["log_dur"] = np.log(d["duration_min"])
    out = []
    for y in ["total", "stable", "oscillating", "n_avalanches"]:
        gee = smf.gee(f"{y} ~ psi + C(task)", groups="subj", data=d, offset=d["log_dur"],
                      family=sm.families.Poisson(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        out.append(_coef_row(f"{y} rate: Poisson GEE, log-duration offset, participant cluster",
                             gee, "psi", d["subj"].nunique(), len(d),
                             effect="rate ratio psi/ctrl"))
        # The VB mixed GLM takes no offset, so log-duration enters as a covariate;
        # an observation-level random effect absorbs Poisson overdispersion.
        glmm = PoissonBayesMixedGLM.from_formula(
            f"{y} ~ psi + C(task) + log_dur",
            {"subj": "0 + C(subj)", "obs": "0 + C(dataset)"}, d).fit_vb()
        i = list(glmm.model.exog_names).index("psi")
        mu, sd = glmm.fe_mean[i], glmm.fe_sd[i]
        out.append({"model": f"{y} rate: Poisson mixed GLM, random intercept + obs-level RE, log-duration covariate (VB)",
                    "effect": "rate ratio psi/ctrl", "estimate": np.exp(mu),
                    "ci_lo": np.exp(mu - 1.96 * sd), "ci_hi": np.exp(mu + 1.96 * sd),
                    "p": 2 * stats.norm.sf(abs(mu / sd)),
                    "n_participants": d["subj"].nunique(), "n_obs": len(d)})

    # Does oscillating drop MORE than stable? polarity x drug interaction.
    long = pd.concat([d.assign(count=d["stable"], osc=0), d.assign(count=d["oscillating"], osc=1)])
    gee = smf.gee("count ~ psi * osc + C(task)", groups="subj", data=long,
                  offset=long["log_dur"], family=sm.families.Poisson(),
                  cov_struct=sm.cov_struct.Independence()).fit()
    out.append(_coef_row("oscillating vs stable: drug x polarity interaction (Poisson GEE)",
                         gee, "psi:osc", d["subj"].nunique(), len(long),
                         effect="ratio of rate ratios (osc RR / stable RR)"))
    return out


def retention_summary(rec: pd.DataFrame, rng) -> tuple[pd.DataFrame, list[dict]]:
    table = rec.pivot_table(index="task", columns="drug", values="dataset",
                            aggfunc="count").fillna(0).astype(int)
    table.loc["participants"] = [rec.loc[rec.drug == c, "subj"].nunique() for c in table.columns]
    dur = rec.groupby(["subj", "drug"])["duration_min"].agg(["sum", "mean"]).reset_index()
    w_sum = dur.pivot(index="subj", columns="drug", values="sum").dropna()
    w_mean = dur.pivot(index="subj", columns="drug", values="mean").dropna()
    res = [paired_summary("total retained minutes per participant (all tasks)",
                          w_sum["ctrl"].values, w_sum["psi"].values, rng),
           paired_summary("mean retained minutes per recording",
                          w_mean["ctrl"].values, w_mean["psi"].values, rng)]
    return table, res


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#52514e")
    ax.spines["bottom"].set_color("#52514e")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def paired_panel(ax, w: pd.DataFrame, ylabel: str, title: str, sizes=None, stat=None):
    """Slopegraph: one grey line per participant, session dots, session median bar."""
    x = np.array([0.0, 1.0])
    for k, (_, r) in enumerate(w.iterrows()):
        ax.plot(x, [r["ctrl"], r["psi"]], color="#b5b4ad", lw=0.8, zorder=1)
    for j, drug in enumerate(["ctrl", "psi"]):
        s = 12 if sizes is None else sizes[drug]
        ax.scatter(np.full(len(w), x[j]), w[drug], s=s, color=COLORS[drug],
                   edgecolor="#fcfcfb", lw=0.6, zorder=2, alpha=0.9)
        med = np.median(w[drug])
        ax.plot([x[j] - 0.18, x[j] + 0.18], [med, med], color="#0b0b0b", lw=2, zorder=3)
    ax.set_xticks(x, [LABELS["ctrl"], LABELS["psi"]])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, loc="left")
    if stat:
        ax.text(0.5, 1.0, stat, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.5, color="#52514e")
    _style(ax)


def _stat_text(row: dict, fmt="{:+.3f}") -> str:
    return (f"n={row['n_participants']}  Δ={fmt.format(row['mean_diff'])} "
            f"[{fmt.format(row['diff_ci_lo'])}, {fmt.format(row['diff_ci_hi'])}]\n"
            f"Wilcoxon p={row['wilcoxon_p']:.3g}  ↓{row['n_decrease']}/↑{row['n_increase']}")


def figure_main(prop_w, n_w, rates, stats_rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.2), facecolor="#fcfcfb")
    sizes = {d: 6 + 2.0 * np.sqrt(n_w.loc[prop_w.index, d].values) for d in ["ctrl", "psi"]}
    sr = {r["contrast"]: r for r in stats_rows}
    paired_panel(axes[0, 0], prop_w, "P(stable) among stable+oscillating",
                 "a  Stable proportion per participant\n(dot area ∝ n sequences)",
                 sizes, _stat_text(sr["P(stable)"]))
    ax = axes[0, 1]
    d = (prop_w["psi"] - prop_w["ctrl"]).values
    jitter = np.random.default_rng(1).uniform(-0.12, 0.12, d.size)
    ax.scatter(jitter, d, s=sizes["ctrl"] * 0 + 14, color="#52514e", alpha=0.7, lw=0)
    ax.axhline(0, color="#0b0b0b", lw=0.8)
    r = sr["P(stable)"]
    ax.errorbar([0.35], [r["mean_diff"]], yerr=[[r["mean_diff"] - r["diff_ci_lo"]],
                [r["diff_ci_hi"] - r["mean_diff"]]], fmt="o", color="#0b0b0b", ms=6, capsize=4)
    ax.set_xticks([0, 0.35], ["participants", "mean ± 95% CI"])
    ax.set_xlim(-0.4, 0.6)
    ax.set_ylabel("Δ P(stable)  (psilocybin − control)", fontsize=9)
    ax.set_title("b  Within-participant change, unweighted", fontsize=9, loc="left")
    _style(ax)
    keys = [("total_rate", "Total", "c"), ("stable_rate", "Stable", "d"),
            ("oscillating_rate", "Oscillating", "e"), ("n_avalanches_rate", "Avalanche", "f")]
    slots = [axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    for ax, (col, name, letter) in zip(slots, keys):
        w = rates.pivot(index="subj", columns="drug", values=col).dropna()
        unit = "avalanches" if name == "Avalanche" else "sequences"
        paired_panel(ax, w, f"{unit} / min retained recording",
                     f"{letter}  {name} rate (task-matched)", None,
                     _stat_text(sr[f"{col}"], "{:+.2f}"))
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    fig.savefig(path.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)


def figure_per_task(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = rec.assign(stable_rate=rec["stable"] / rec["duration_min"],
                     oscillating_rate=rec["oscillating"] / rec["duration_min"])
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), facecolor="#fcfcfb", sharey="row")
    for i, col in enumerate(["stable_rate", "oscillating_rate"]):
        for j, task in enumerate(TASKS):
            w = rec[rec.task == task].pivot(index="subj", columns="drug", values=col).dropna()
            d = w["psi"] - w["ctrl"]
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            paired_panel(axes[i, j], w, f"{col.split('_')[0]} / min" if j == 0 else "",
                         f"{TASK_LABELS[task]} — {col.split('_')[0]}", None,
                         f"n={len(w)} paired  Δmed={np.median(d):+.2f}  p={p:.3g}")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    seqs = load_sequences()
    rec = recording_table(seqs)
    rec.to_csv(OUT / "per_recording.csv", index=False)

    s = session_proportions(rec)
    s.to_csv(OUT / "per_participant_session.csv", index=False)
    prop_w = wide_paired(s, "p_stable")
    n_w = wide_paired(s, "n_binary")
    rows = [paired_summary("P(stable)", prop_w["ctrl"].values, prop_w["psi"].values, rng)]
    ge5 = (n_w["ctrl"] >= 5) & (n_w["psi"] >= 5)
    rows.append(paired_summary("P(stable), participants with >=5 seq per session",
                               prop_w.loc[ge5, "ctrl"].values, prop_w.loc[ge5, "psi"].values, rng))

    rates = task_matched_rates(rec)
    rates.to_csv(OUT / "per_participant_rates_task_matched.csv", index=False)
    for col in ["total_rate", "stable_rate", "oscillating_rate", "n_avalanches_rate"]:
        w = rates.pivot(index="subj", columns="drug", values=col).dropna()
        rows.append(paired_summary(col, w["ctrl"].values, w["psi"].values, rng))

    ret_table, ret_rows = retention_summary(rec, rng)
    rows += ret_rows
    pd.DataFrame(rows).to_csv(OUT / "paired_tests.csv", index=False)
    ret_table.to_csv(OUT / "recordings_per_task_session.csv")

    models = proportion_models(rec) + rate_models(rec)
    pd.DataFrame(models).to_csv(OUT / "models.csv", index=False)

    figure_main(prop_w, n_w, rates, rows, OUT / "participant_level.png")
    figure_per_task(rec, OUT / "participant_level_per_task.png")

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("sequences by drug:", seqs.groupby(["drug", "polarity"]).size().to_dict())
    print(ret_table, "\n")
    print(pd.DataFrame(rows).round(4).to_string(), "\n")
    print(pd.DataFrame(models).round(4).to_string())


if __name__ == "__main__":
    main()
