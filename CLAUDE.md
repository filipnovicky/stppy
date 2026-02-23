# CLAUDE.md — stppy

## Project
**STPPY** — Spatiotemporal Pattern Processing for Python.
EEG avalanche detection and analysis package.
Author: Filip Novicky (novicky@cs.cas.cz)

## Layout
```
stppy/           # installable package (pip install -e .)
  io/            # EEG file loading, electrode position export
  preprocessing/ # electrode interpolation, resampling
  detection/     # threshold-based avalanche detection, correlation-based sequence detection
  alignment/     # spatial (optimal transport) + temporal (DTW/subDTW) alignment
  clustering/    # hierarchical clustering, consensus alignment
  visualization/ # topomaps, videos
  utils/         # pairwise distances

paper_scripts/   # standalone analysis scripts for the paper — NOT part of the package
                 # excluded from desloppify (scripts/ is an old name for the same folder)
```

## Domain vocabulary
- **Avalanche**: a spatiotemporal EEG event (array of shape `[n_channels, n_timepoints]`)
- **Sequence**: a correlation-detected pattern in continuous EEG (replaces old term "burst")
- **Polarity types**: `'standing'`, `'rotating'`, `'long_rotating'`, `'mixed'`, `'single'`
- **Medoid**: the most central avalanche in a cluster (used as alignment reference)
- **DTW path**: a list of `(i, j)` frame-correspondence tuples from dynamic time warping

## Public API
All `__init__.py` files are intentional re-export facades — not accidents.
The public API is `stppy.<module>.<function>`, e.g. `stppy.detection.detect_avalanches(...)`.
Do not flag these as unnecessary indirection.

## Coding conventions
- Private helpers: prefix with `_` (e.g. `_find_medoid`, `_render_title_card_frame`)
- No emojis in print statements
- `except ValueError` preferred over `except Exception` when MNE raises `ValueError` for channel mismatches
- `except Exception` is acceptable for `scipy.spatial.ConvexHull` (raises `QhullError`, non-standard to import)
- Verbose output controlled by a `verbose: bool` parameter — never `print()` unconditionally in library code

## Desloppify
```bash
desloppify scan          # re-scan (paper_scripts/ and scripts/ are permanently excluded)
desloppify status        # score dashboard
desloppify show <file>   # findings for a specific file
```
Current score: **93.9 / 100** (2026-02-18).
Remaining open items (all stppy/):
- 7 facade findings — intentional, mark wontfix
- 5 large files — known, not currently being addressed
- monster functions in `interpolation.py`, `videos.py` — partially decomposed, still flagged
- broad except in `filtering.py:128` — acceptable (QhullError)
- magic numbers in `similarity.py:29,373,404` — low priority

## Key dependencies
- `mne` — EEG data structures (`Raw`, `Epochs`, `create_info`, etc.)
- `numpy`, `scipy` — numerics
- `matplotlib` — figures
- `ffmpeg` (system) — video assembly in `visualization/videos.py`
