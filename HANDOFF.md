# HANDOFF — DRL for Quantum Circuit Shot Allocation

_Last updated: 2026-06-17_

## Goal
A DQN agent decides **when to stop** executing quantum-circuit shots so its shot count hugs the
a-posteriori Oracle as tightly as possible, with a deliberate bias toward **slight overshoot over
any undershoot** (overshoot wastes resources; undershoot invalidates results). The active work is
the **conference paper in `paper/`**, being prepared for **ICSOC 2026** (double-blind), which
presents the agent as a deployable **shot-allocation service**.

## Current state of the paper (most important section)
The paper is **aligned to v1** (`program-main-enhanced.ipynb` → results in `generic-enhanced/`),
per an explicit user decision this session. It previously described enhanced-2; that is now undone.
**Canonical headline numbers** (from `generic-enhanced/multirun_eval-generic.csv`):
- DRL mean |shots − Oracle| = **2,049 shots (10.2% of budget)**.
- Undershoots on only **3/36** traces, each within ≤200 shots.
- Beats Inc-Hell (2,706) and Inc-JS (3,336); trails Inc-TVD (871) by ≈1,180 shots.
- Dominant error mode = **over-conservative overshoot on mid-range 6–8q `qaoa`/`qft`/`qnn`**
  circuits (NOT cap-riding). Both extremes (tiny + largest near-budget) are stopped tightly.
- ⚠️ The user said the paper's results table does **not yet** report his latest/better run — he
  will update the numbers later. Treat 2,049 as the current-but-soon-to-improve baseline.

### Everything done to the paper this session (2026-06-16 → 06-17)
- **Venue/template:** header + venue notes set to **ICSOC 2026** (Springer LNCS, **10–15 pp incl.
  references**, double-blind). LNCS class was already correct.
- **Authors:** added **Andrea Cossu** (no ORCID — needs his real one; I refused to fabricate).
  Names kept **visible** per user choice (so the compiled PDF is NOT anonymized — if submitting
  blind, the author block must be hidden; the body text is already anonymity-safe).
- **Terminology:** `easy`/`hard` → **`small`/`big`** in prose, reward, dataset, tables, and the
  relabel-able figures.
- **Double-blind phrasing:** removed all "Bisicchia et al." attributions and any "we compare
  ourselves with…" framing; prior work is now neutral ("a recent study/framework").
- **arXiv:** incremental-execution work cites **arXiv:2606.16965** (`bisicchia2026shots` is now an
  `@misc` with that eprint).
- **Service framing (reworked twice):** abstract, intro, and contribution 1 now LEAD with the
  *service* (an intermediary gateway between user and quantum cloud) that *internally* uses the
  learned agent. New **Fig. 1 (`fig:service`)**: high-level gateway diagram (User ↔ Service ↔
  QPU), wrapped in `\resizebox{\textwidth}`. The old agent/env/oracle TikZ diagram is now **Fig. 2
  (`fig:arch`)**, recaptioned "Training-time learning loop inside the service", with the Oracle
  moved **below** the environment (vertical arrows) so it no longer overflows the right margin.
- **Section 2 split:** 2.2 *Related Work* (quantum) and 2.3 *Background: RL & DQN* (deep learning).
  Learning-based methods expanded with explicit 3-way differentiation (static vs variational;
  black-box vs gradient/algorithm signals; general stopping rule vs algorithm-specific allocator).
  The "Adaptive allocation for variational algorithms" paragraph now describes **iCANS**
  (`kubler2020icans`), **gCANS** (`gu2021gcans`), **Rosalin** (`arrasmith2020rosalin`).
- **Section 3.2 (State Space):** rewritten formally with equations for all **9** features
  (entropy/$n$, concentration Σp², two-lag TVD, size-relative progress).
- **Method/reward reflect v1:** 9 features, standard DQN + **MSE** loss, **absolute** overshoot
  margin `m=200`, decay `λ=2`, no Double DQN, no Huber, no stability voting, no convergence-streak
  feature, 1,000 episodes / checkpoint from ep 400, γ=0.997, target sync every 2,000 grad steps.
- **Model selection:** own paragraph (20% stratified val split, de-noised 3-pass val-MAE
  checkpointing, early stopping patience 5, learning-curve rationale).
- **Open source:** inline anonymized repo link (`anonymous.4open.science` placeholder). Originally a
  footnote that overflowed; moved inline + added `\Urlmuskip` so long URLs break. **Footnotes
  removed entirely** (user reported footnote issues).
- **Figures:** `agent_vs_oracle-generic-all36.pdf` (**Fig. 4**) **regenerated from scratch with
  matplotlib** at large fonts (script at `/tmp/fig4.py`) — legible, SMALL/BIG legend with avg MAE,
  y=x diagonal. `problem_overview.pdf` relabeled SMALL/BIG. `dashboard-generic.pdf` converted from
  the v1 SVG but **still shows easy/hard** (matplotlib exported its text as vector paths — 0
  `<text>` nodes — so it can't be string-relabeled; needs a fresh notebook run to fix).
- **Bibliography fix:** reference [1] was showing "anonymous" because the RQAOA entry
  (`rqaoa2026rlshots`, arXiv:2605.26544) had a placeholder author and splncs04 sorts alphabetically.
  Fixed with real authors **Lee, Euimin and Kim, Shiho**.
- **Style:** removed all parenthetical em-dashes (` -- `) from prose (kept numeric/section ranges
  and TikZ path `--`); per user, they "look too AI-written".
- **`make_tables.py`:** candidate order now prefers `generic-enhanced/`; table captions say
  small/big; neutral phrasing. Re-ran it → `tables_generated.tex` now shows DRL **2,049**.
- **READMEs:** wrote `paper/README.md` (full repo explainer) and rewrote root `README.md`.

## Project layout (key facts)
- Three program versions, each a **single giant code cell** in a notebook:
  - `program-main.ipynb` — original. Reward `HARD_ASYMMETRIC`, 8-feature state. Outputs → `generic/`.
  - `program-main-enhanced.ipynb` — **v1, the version the paper now describes**. Reward `PRECISION`,
    9-feature state, standard DQN+MSE. Outputs → `generic-enhanced/`.
  - `program-main-enhanced-2.ipynb` — v2 (10-feature, Double DQN+Huber, stability voting, relative
    margin, 2000 eps). Outputs → `generic-enhanced-2/`. **No longer used by the paper** (stronger
    MAE 1,154 run still exists in that folder if ever needed).
- Output folders are version-separated. `paper/make_tables.py` now picks the CSV in order
  `generic-enhanced/` > `generic/` and prints which.
- Dataset: test = exact 36 paper traces (held out); train/val = 80/20 stratified split of the
  remaining 828 traces (`grover-noancilla` excluded). Oracle = TVD suffix-stable, eps=0.1,
  batch=50, B=20000. Env runs `qsimbench`; Python env is conda env **`qdrl`**.
- Source PDFs in `pdfs/`: `Thesis.pdf` (earlier work), `IncExc_to_TQC.pdf` (the Oracle /
  IncrementalExecution baseline; cite as arXiv:2606.16965). `pdfs/tables.tex` holds the SOTA
  reference numbers that `make_tables.py` merges with our CSV.

## Files changed this session
`paper/paper.tex`, `paper/references.bib`, `paper/make_tables.py`, `paper/tables_generated.tex`,
`paper/figures/{problem_overview,agent_vs_oracle-generic-all36,dashboard-generic}.pdf`,
`paper/README.md` (new), root `README.md`, this `HANDOFF.md`.

## What worked
- Extracting a notebook's single code cell to `/tmp/*.py` to read/grep config and logic
  (`python3 -c "import json;print(''.join(json.load(open('program-main-enhanced.ipynb'))['cells'][0]['source']))" > /tmp/enh.py`).
- Regenerating Figure 4 directly from the run CSV with matplotlib (full control over legibility),
  rather than hacking the SVG.
- `make_tables.py` to regenerate tables; `cairosvg` (conda `cairo` in `qdrl`) for SVG→PDF.
- Structural LaTeX checks without a compiler: a Python brace/`\begin`–`\end` balance scan and an
  `align`-ampersand count (there is **no LaTeX toolchain on this machine** — paper must build
  elsewhere).

## What didn't work / gotchas
- **No `pdflatex`/`tectonic`/`latexmk` here** → cannot compile the paper. All paper checks are
  structural; the user must build elsewhere. Verify the two new TikZ figures + URL render on build.
- **`conda run` swallows stdout** — use `conda run --no-capture-output -n qdrl ...`.
- **PDF rendering:** `pdftoppm`/poppler missing → Read tool can't open the source PDFs; use
  `pymupdf` (`fitz`) text extraction. `cairosvg` needs native `cairo` (conda), not just pip.
- **Editor stale-buffer clobbering (historical):** notebooks edited on disk were twice reverted by
  the user's editor saving an old buffer. Tell the user to reload notebooks from disk before running.
- `dashboard-generic.pdf` text is vector paths → can't relabel easy/hard without re-running.

## Next steps
1. **Update the results table** once the user provides his latest/better run CSV (he flagged the
   table is behind). Drop the CSV into `generic-enhanced/` (or repoint `_CSV_CANDIDATES`) and re-run
   `cd paper && python3 make_tables.py`. Re-check Results/Discussion/Limitations prose against the
   new numbers (the mid-range-overshoot narrative may change).
2. **Get the IncExc arXiv reference** — the user will send the official incremental-execution arXiv
   citation/oracle once available; until then `bisicchia2026shots` points to arXiv:2606.16965.
3. **Camera-ready:** bump `N_EVAL_RUNS` (≥10) and re-run `make_tables.py` for the mean±std the
   reproducibility note promises; re-render `dashboard-generic` from a fresh v1 run to fix
   easy/hard labels; add Andrea Cossu's real ORCID; replace the anonymized repo URL with the real
   GitHub link; decide whether to anonymize the author block for the blind submission.
4. **Compile `paper/paper.pdf` elsewhere** (`pdflatex paper && bibtex paper && pdflatex ×2`) and
   eyeball Fig. 1 (`fig:service`) and Fig. 2 (`fig:arch`) for layout/overflow.

## Open research directions the user is weighing (not yet acted on)
These came up as the user's intended next experiments — capture them, don't assume done:
- **Reward ablation:** start from **MAE-only** reward, then add components incrementally, to see
  what each shaping term buys.
- **Did training actually help?** Snapshot the model **after the first ~500 episodes** vs the final
  model and compare (agent-vs-oracle, a few steps) to confirm learning occurred vs a stable-by-luck
  policy. The reward looked "stable" — establish whether that was training or initialization.
- **Cleaner train/val/test separation:** select different-but-similar circuits/problems for
  validation so the split is a genuine generalization test.
- **Run-to-run variance:** two back-to-back runs of the same program gave very different outcomes
  (stochastic quantum sampling + seed/checkpoint sensitivity) — worth pinning down before trusting
  single-run tables (hence `N_EVAL_RUNS`).

## Useful commands
```bash
# extract a notebook's single code cell to a .py for diffing/grepping
python3 -c "import json;print(''.join(json.load(open('program-main-enhanced.ipynb'))['cells'][0]['source']))" > /tmp/enh.py

# regenerate paper tables from the preferred (generic-enhanced) CSV
cd paper && python3 make_tables.py

# extract source PDFs as text (no poppler; fitz is in base python)
python3 -c "import fitz;d=fitz.open('pdfs/IncExc_to_TQC.pdf');print(d[0].get_text())"

# SVG -> PDF for figures (needs conda cairo in qdrl; --no-capture-output to see prints)
conda run --no-capture-output -n qdrl python -c "import cairosvg; cairosvg.svg2pdf(url='generic-enhanced/dashboard-generic.svg', write_to='paper/figures/dashboard-generic.pdf')"

# regenerate the legible Figure 4 from the run CSV
conda run --no-capture-output -n qdrl python /tmp/fig4.py
```
