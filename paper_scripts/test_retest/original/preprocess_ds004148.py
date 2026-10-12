#!/usr/bin/env python
"""ds004148 resting EEG (eyes closed / eyes open) -> clean broadband 1-40 Hz at 250 Hz.

Mirrors the anphy v4 pipeline (stppy_version_1.0.0/scripts/preprocessing_v4_anphy.py) so the
two datasets are cleaned the same way:
 * load raw BIDS BrainVision (61 EEG, online reference FCz, 500 Hz, 300 s),
 * restore FCz as a channel and average-reference (62 channels, as the authors' chanlocs62),
 * resample to 250 Hz,
 * bad channels (on a 1-40 Hz copy): |robust z of log-variance| > 5, or mean corr with 4 nearest
   neighbours < 0.4 -> spherical-spline interpolate,
 * FIR 1-40 Hz, average reference,
 * 20-component extended-infomax ICA; reject ICLabel eye >= 0.8, muscle >= 0.8, and
   high-kurtosis (> median + 3 SD) sources.
No segment rejection: the authors screened all 4-s epochs by eye and rejected none, and the
dataset has no artefact matrix. Eyes-open raw is already eye-ICA-cleaned by the authors (the
published derivatives are byte-identical to raw); the ICA here is applied to both conditions.

PBS_ARRAY_INDEX (1-based) -> (subject 1..60, session 1..3, task EC/EO).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mne
import numpy as np
from mne_icalabel import label_components
from scipy.stats import kurtosis

BIDS = Path("/store/projects/PUBLIC_DATASETS/ds004148")
OUT = Path("/home/novicky/ds004148_preprocessed/data")
TASKS = [("EC", "eyesclosed"), ("EO", "eyesopen")]
SFREQ_TARGET = 250
FILTER_LO, FILTER_HI = 1.0, 40.0
ICLABEL_EYE_THR = 0.8
ICLABEL_MUSCLE_THR = 0.8
N_SUBJECTS, N_SESSIONS = 60, 3


def unit(idx1: int) -> tuple[int, int, str, str]:
    idx = idx1 - 1
    subj, rest = divmod(idx, N_SESSIONS * len(TASKS))
    ses, t = divmod(rest, len(TASKS))
    return subj + 1, ses + 1, TASKS[t][0], TASKS[t][1]


def positions(eeg_dir: Path, subj: int, ses: int) -> dict[str, np.ndarray]:
    """BIDS electrodes.tsv (mm, x=right, y=anterior) + FCz placed between Fz and Cz on the shell."""
    tsv = eeg_dir / f"sub-{subj:02d}_ses-session{ses}_electrodes.tsv"
    rows = [ln.split("\t") for ln in tsv.read_text().splitlines()[1:] if ln.strip()]
    pos = {r[0]: np.array([float(r[1]), float(r[2]), float(r[3])]) for r in rows}
    mid = (pos["Fz"] + pos["Cz"]) / 2.0
    radius = np.mean([np.linalg.norm(v) for v in pos.values()])
    pos["FCz"] = mid / np.linalg.norm(mid) * radius
    return pos


def bad_channels(data: np.ndarray, names: list[str], xyz: np.ndarray) -> list[str]:
    # Robust z of LOG variance > 5. The v4 rule (|var - median| > 5 MAD on raw variance) flags
    # 4-7 frontal (eye) and parieto-occipital (eyes-closed alpha) channels per recording here:
    # variance is right-skewed, so MAD of raw variance is tiny. Log-variance z never exceeds 2.7
    # on subs 01-05, matching the authors' "no bad channels".
    lv = np.log(np.var(data, axis=1))
    z = (lv - np.median(lv)) / (1.4826 * np.median(np.abs(lv - np.median(lv))))
    cd = np.sqrt(((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(-1))
    bads = []
    for i, ch in enumerate(names):
        if np.abs(z[i]) > 5:
            bads.append(ch)
            continue
        dd = cd[i].copy()
        dd[i] = np.inf
        nn = np.argsort(dd)[:4]
        if np.mean([np.corrcoef(data[i], data[n])[0, 1] for n in nn]) < 0.4:
            bads.append(ch)
    return bads


def run_ica(r: mne.io.BaseRaw) -> dict[str, list[int]]:
    ica = mne.preprocessing.ICA(n_components=20, method="infomax",
                                fit_params=dict(extended=True), random_state=42)
    ica.fit(r, picks="eeg", verbose=False)
    lab = label_components(r, ica, method="iclabel")
    labels, probs = lab["labels"], lab["y_pred_proba"]
    eye = [i for i, (l, p) in enumerate(zip(labels, probs)) if l == "eye blink" and p >= ICLABEL_EYE_THR]
    muscle = [i for i, (l, p) in enumerate(zip(labels, probs))
              if l == "muscle artifact" and p >= ICLABEL_MUSCLE_THR]
    kt = kurtosis(ica.get_sources(r).get_data(), axis=1)
    kthr = np.median(kt) + 3 * np.std(kt)
    kurt = [i for i in range(ica.n_components_) if kt[i] > kthr and i not in eye + muscle]
    ica.exclude = sorted(set(eye + muscle + kurt))
    ica.apply(r, verbose=False)
    return dict(eye=eye, muscle=muscle, kurtosis=kurt, labels=list(labels))


def preprocess(subj: int, ses: int, tag: str, task: str) -> None:
    out = OUT / f"sub-{subj:02d}_ses-{ses}_{tag}.npz"
    if out.exists():
        print(f"[SKIP] {out}")
        return
    eeg_dir = BIDS / f"sub-{subj:02d}" / f"ses-session{ses}" / "eeg"
    vhdr = eeg_dir / f"sub-{subj:02d}_ses-session{ses}_task-{task}_eeg.vhdr"
    if not vhdr.exists():
        print(f"[MISS] {vhdr}")
        return
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose=False)
    raw.pick("eeg")
    pos = positions(eeg_dir, subj, ses)
    # 80 FieldTrip-written headers spell CPz as "Cpz" or Fpz as "FPz" (same order everywhere);
    # map case-insensitively onto the electrodes.tsv template names.
    canon = {c.upper(): c for c in pos}
    raw.rename_channels({c: canon[c.upper()] for c in raw.ch_names
                         if c.upper() in canon and c != canon[c.upper()]})
    mne.add_reference_channels(raw, "FCz", copy=False)
    names = list(raw.ch_names)
    missing = [c for c in names if c not in pos]
    if missing:
        raise ValueError(f"no position for {missing}")
    xyz = np.array([pos[c] for c in names])
    raw.set_montage(mne.channels.make_dig_montage(
        ch_pos={c: pos[c] / 1000.0 for c in names}, coord_frame="head"), verbose=False)
    raw.set_eeg_reference("average", verbose=False)        # gives FCz real data
    raw.resample(SFREQ_TARGET, verbose=False)

    # Detect on a 1-40 Hz copy: on unfiltered data slow drift drives both rules and flagged
    # 13-31 of 62 channels on sub-01 (authors: none bad); on the filtered copy, 0-1.
    bads = bad_channels(raw.copy().filter(FILTER_LO, FILTER_HI, method="fir", verbose=False)
                        .get_data(), names, xyz)
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(verbose=False)
    raw.filter(FILTER_LO, FILTER_HI, method="fir", verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    rej = run_ica(raw)
    data = raw.get_data(picks="eeg")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, data=data.astype(np.float32), channel_names=np.asarray(names),
        electrode_xyz=xyz, sfreq=np.asarray(SFREQ_TARGET), subject=np.asarray(f"sub-{subj:02d}"),
        session=np.asarray(ses), task=np.asarray(tag), bad_channels=np.asarray(bads, dtype=object),
        ica_eye=np.asarray(rej["eye"]), ica_muscle=np.asarray(rej["muscle"]),
        ica_kurtosis=np.asarray(rej["kurtosis"]), ica_labels=np.asarray(rej["labels"]),
    )
    n_rej = len(set(rej["eye"] + rej["muscle"] + rej["kurtosis"]))
    print(f"[OK] {out.name} {data.shape} bads={bads} ICA eye={rej['eye']} "
          f"muscle={rej['muscle']} kurt={rej['kurtosis']} ({n_rej}/20 removed)")


def main() -> None:
    idx = int(os.environ.get("PBS_ARRAY_INDEX", sys.argv[1] if len(sys.argv) > 1 else "1"))
    subj, ses, tag, task = unit(idx)
    print(f"=== ds004148 preprocess idx={idx} sub-{subj:02d} ses-{ses} {tag} ===")
    preprocess(subj, ses, tag, task)


if __name__ == "__main__":
    main()
