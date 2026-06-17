# HANDOFF — DRL for Quantum Circuit Shot Allocation

_Last updated: 2026-06-16_

## ⚠️ Update 2026-06-16 (ICSOC realignment)
The paper was **realigned to v1** (`program-main-enhanced.ipynb` → `generic-enhanced/`,
MAE **2,049**), per user decision — it had previously described enhanced-2. The paper is now
formatted for **ICSOC 2026** (Springer LNCS, 10–15 pp incl. refs, double-blind): Andrea Cossu
added as author (names kept visible per user); prior work phrased neutrally; easy/hard renamed
**small/big** (text, tables, figures); Fig. 1 redrawn in TikZ; Section 2 split into quantum
related-work vs DL background; Section 3.2 made mathematical; reward/method reflect v1 (9 feats,
std DQN+MSE, absolute overshoot margin 200, no Double DQN / stability voting); incremental
execution cites arXiv:2606.16965; repo presented as a service; `paper/README.md` added.
`make_tables.py` now prefers `generic-enhanced/`. NOTE: the v1 `dashboard-generic.pdf` still
shows easy/hard labels baked in as vector paths (re-render from a fresh run for camera-ready).

## Goal
Improve the DQN agent that decides **when to stop** executing quantum-circuit shots, so its
shot count hugs the a-posteriori Oracle as tightly as possible, with a deliberate bias toward
**slight overshoot over any undershoot** (overshoot wastes resources; undershoot invalidates
results). Keep the paper in `paper/` consistent with the latest results.

## Project layout (key facts)
- Three program versions, each a **single giant code cell** in a notebook:
  - `program-main.ipynb` — original. Reward `HARD_ASYMMETRIC`, 8-feature state. Outputs → `generic/`.
  - `program-main-enhanced.ipynb` — **v1**. Reward `PRECISION`, 9-feature state. Outputs → `generic-enhanced/`.
  - `program-main-enhanced-2.ipynb` — **v1 + v2**. 10-feature state, Double DQN + Huber, stability
    voting, relative reward margin, forced-cap penalty, 2000 episodes. Outputs → `generic-enhanced-2/`.
- **Output folders are version-separated** so runs can't overwrite each other. `paper/make_tables.py`
  picks the CSV in order: `generic-enhanced-2/` > `generic-enhanced/` > `generic/`, and prints which.
- SVG headers embed the config label (e.g. `Reward: PRECISION` vs `HARD_ASYMMETRIC`) — use this to
  identify which version produced any given run's charts.
- Dataset: test = exact 36 paper traces (held out); train/val = 80/20 stratified split of the
  remaining 828 traces (`grover-noancilla` excluded in enhanced versions). Oracle = TVD suffix-stable,
  eps=0.1, batch=50, B=20000. Env runs `qsimbench`; Python env is conda env **`qdrl`**.
- Source PDFs in `pdfs/`: `Thesis.pdf` (earlier work), `IncExc_to_TQC.pdf` (Bisicchia et al., the
  Oracle / IncrementalExecution baseline). `pdfs/tables.tex` holds the SOTA reference numbers that
  `make_tables.py` merges with our CSV.

## Current progress
- **v1 (enhanced) implemented & validated.** Fixes + improvements vs original:
  dropout disabled at inference (deterministic greedy), step-based target sync (every 2000 grad steps),
  per-episode ε-decay, `noancilla` exclusion fixed (was checking backend string, now algorithm),
  `step()` fetch-failure warnings, `PRECISION` reward (full-reward overshoot plateau then oracle-relative
  decay; undershoot keyed to oracle difficulty), entropy normalized by qubits, variance→concentration
  (Σp²), rate-of-change = successive cumulative TVD at two lags, progress denom size·1500, γ=0.997,
  3-pass validation averaging, oracle-stratified sampling in all split metrics.
- **v2 (enhanced-2) implemented & validated** on top of v1: `OVERSHOOT_REL_DECAY` 2→4, relative margin
  `max(50, 0.03·oracle)`, forced-cap reward ×0.5, Double DQN + Huber loss, 10th feature = convergence
  streak (consecutive batches with long-lag TVD < eps), stability voting (2 consecutive STOPs at eval),
  2000 episodes / checkpoint from ep 800. **`N_EVAL_RUNS` left at 1** per user request.
- **Paper is fully aligned to the enhanced-2 run** (the canonical result): method section describes v2,
  tables regenerated, figures converted, prose/limitations rewritten.

## Canonical results (enhanced-2, in `generic-enhanced-2/multirun_eval-generic.csv`)
- DRL mean |shots − Oracle| = **1,154 shots (5.8% of budget)**.
- Beats Inc-Hell (2,706) and Inc-JS (3,336); within ~280 shots of Inc-TVD (871).
- Only **2/36 undershoots**; residual error = budget-cap rides on near-budget (12–14q) instances.

## What worked
- Building each notebook by applying exact-match string edits to the extracted source cell via a Python
  script (`/tmp/make_enhanced.py`, `/tmp/make_enhanced2.py`), then `compile()`-checking and writing the
  cell back. Reliable and reviewable.
- Verifying logic by importing the extracted `.py` in the `qdrl` conda env and unit-testing reward math
  / feature math before trusting a full run.
- `cairosvg` (after `conda install -n qdrl cairo`) to convert the `generic*/` SVGs → `paper/figures/*.pdf`.
- Regenerating tables with `cd paper && python3 make_tables.py`.

## What didn't work / gotchas
- **Editor stale-buffer clobbering:** twice, notebooks I edited on disk were reverted when the user's
  editor saved an old buffer over them (both `program-main-enhanced.ipynb` and the originally-untouched
  files reverted to original code at one point). **Always tell the user to reload notebooks from disk
  before running**, and re-verify on-disk content rather than trusting prior edits.
- `pdftoppm`/poppler not installed → can't render PDFs via the Read tool; use `pymupdf` (`fitz`) text
  extraction instead. `cairosvg` needs the native `cairo` lib (conda), not just the pip package.
- The `generic/` folder is "last run only" — an original-code run will overwrite enhanced results there.
  That's why version-separated output folders were introduced.
- Paper prose had stale claims after each new run (e.g. "undershooting" was the old failure mode; the
  current one is cap-region overshoot). Always re-check Results + Limitations against the actual CSV.

## Next steps
1. **Run enhanced-2 fresh** (reload notebook from disk first) and confirm it reproduces ≈1,154 MAE /
   tighter band. Inspect `generic-enhanced-2/agent_vs_oracle-generic-all36.svg`.
2. If a few points dip **below** the diagonal, first lever is `OVERSHOOT_REL_DECAY` 4.0 → 3.0
   (don't touch the margin yet).
3. Address the residual **budget-cap overshoot** on near-budget (12–14q) hard instances — the gap to
   Inc-TVD. Candidate levers: cap-region reward shaping, richer near-budget convergence descriptors.
4. Bump **`N_EVAL_RUNS`** (≥10) when ready for camera-ready, then re-run `make_tables.py` to get the
   mean±std columns the paper's reproducibility note promises.
5. Recompile `paper/paper.pdf` (no LaTeX toolchain on this machine — build elsewhere). Everything it
   `\input`s is already current.
6. Consider restoring v1 code into `program-main-enhanced.ipynb` if it ever gets clobbered again
   (the v1 source was saved this session at `/tmp/program-main-enhanced-check.py`).

## Useful commands
```bash
# extract a notebook's single code cell to a .py for diffing/testing
python3 -c "import json;print(''.join(json.load(open('program-main-enhanced.ipynb'))['cells'][0]['source']))" > /tmp/x.py

# regenerate paper tables from the best available CSV
cd paper && python3 make_tables.py

# SVG -> PDF for paper figures (needs conda cairo in qdrl)
conda run -n qdrl python -c "import cairosvg; cairosvg.svg2pdf(url='generic-enhanced-2/dashboard-generic.svg', write_to='paper/figures/dashboard-generic.pdf')"
```
