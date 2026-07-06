# HANDOFF — DRL for Quantum Circuit Shot Allocation

_Last updated: 2026-07-06_

## Goal
A DQN agent decides **when to stop** executing quantum-circuit shots so its shot count hugs the
a-posteriori Oracle as tightly as possible, with a deliberate bias toward **slight overshoot over
any undershoot** (overshoot wastes resources; undershoot invalidates results). The active work is
the **conference paper in `paper/`**, prepared for **ICSOC 2026** (double-blind, Springer LNCS,
**10–15 pp incl. references**), which presents the agent as a deployable **shot-allocation
service** (an intermediary gateway between the user and the quantum cloud).

---

## ⚠️ Two things the next agent MUST know first

1. **Results numbers in the paper are STALE / user-owned.** `paper/paper.tex` and
   `paper/tables_generated.tex` report DRL mean |shots−Oracle| = **2,049 (10.2%), 3/36 undershoots
   each ≤200 shots** (safety-first narrative). But the current
   `generic-enhanced/multirun_eval-generic.csv` actually yields **2,303 (11.5%), 4 undershoots
   including two LARGE ones (qft-12 ≈ −1,800, qft-14 ≈ −1,500)** — which breaks the "rarely
   undershoots, each ≤200" claim. The user maintains these numbers himself and said the table is
   behind a better run he'll supply. **Do NOT re-run `make_tables.py` or edit headline numbers
   without his go-ahead** (it would overwrite 2,049 with 2,303 and contradict the prose). Also saved
   as memory `paper-results-numbers-stale`.

2. **No LaTeX toolchain on this machine** (no `pdflatex`/`tectonic`/`latexmk`/`xelatex`). The paper
   cannot be compiled here — all checks are structural (brace/`\begin`–`\end` balance, anchor
   greps). The user must build elsewhere and eyeball figures/overflow.

---

## Current progress (this session, 2026-07-06)

### A. Service-oriented paper editing pass (`paper/paper.tex`, `paper/references.bib`)
A large edit pass per an explicit user list. All done:
- **Title** → service-oriented: *"Learning When to Stop: A Deep Reinforcement Learning **Service**
  for Adaptive Shot Allocation in Quantum Circuit Execution"* (running title updated).
- **Abstract**: generalised the "36 traces / strict split" sentence to "hundreds of configurations +
  clean train/validation/held-out-test separation"; rephrased "safety-first behaviour" and the
  "surpasses two of three / trails strongest" sentence into plain language.
- **Intro**: added a **Design rationale** paragraph (motivates black-box + learned + service up
  front); rewrote **Contribution 3** to be generic about experiment count + val/test.
- **Background**: new subsection **2.1 "Quantum Circuits, Shots, and Noise"** (drawn from
  `pdfs/background.tex`); added **undershoot/overshoot/optimum** explanation in 2.1 (basic) and 2.2
  (formal, after the Oracle def).
- **Related work**: added that the service can be applied **independently at each iteration** of a
  variational loop.
- **Section 3 (service focus)**: expanded "The service" (how the user submits data); new **3.2
  "Deploying the Service"** (pre-trained model + **administrator** role can retrain/hot-swap;
  CPU-vs-QPU note — QSimBench emulates today, swap one backend param for a real QPU; **REST/Flask**
  `POST /execute` shown as a `verbatim` code listing `fig:api` + **Docker** + long-running/async
  caveat); new **3.3 "Practical Applicability on Real Quantum Clouds"** (recast of IncExc §7 —
  sessions/reservations/hybrid runtimes — in service terms). **Fig. 1** now shows the budget cap
  **B=20,000**.
- **3.5 (dataset)**: opens by explaining **what QSimBench is / is for**; talks more generally about
  the number of experiments.
- **Results**: simplified the over-technical subsection title → **"How the Agent Learns to Stop"**;
  **removed Figure 3** (`problem_overview`); trimmed the dashboard — first to 3 panels, then (per a
  follow-up) **removed panel (c) stopping-error distribution**, so `fig:dashboard` is now just
  **(a) training reward + (b) validation MAE**; added an "Implications for the service" paragraph
  to the Discussion.
- **Conclusion**: reworded opening to lead with the **service** ("We presented a shot-allocation
  service … whose decision logic is a DRL agent…").
- **references.bib**: **reference [2] (QSimBench)** converted from a bare GitHub link to the proper
  article citation — `@inproceedings`, *QSimBench: An Execution-Level Benchmark Suite for QSE*,
  **IEEE QCE 2025**, authors Bisicchia, Bocci, García-Alonso, Murillo, Brogi (found via web search).
- Structural checks pass: figures 5↔5, minipages 3↔3, verbatim 1↔1, braces balanced, no dangling
  `fig:overview`/`problem_overview` refs.
- **"Stray Claude comment after 2026 in Section 4"**: the user reported one, but it is **NOT** in
  the committed `paper.tex` (grepped all paper files — nothing). Likely was in his editor's unsaved
  buffer. Flagged to him.

### B. δ (tolerance) analysis — why the comparison looks unfavourable
- The reference paper reports **δ ∈ {0.05, 0.10, 0.25}**; `pdfs/tables.tex` has the SOTA numbers
  (incl. the Oracle column) for all three. `paper/make_tables.py` already has a `delta_label` arg.
- δ=0.10 was chosen simply as the **middle default** matching the Oracle's hardcoded `eps=0.1`; no
  deeper documented rationale. (The reference paper frames δ=0.05 as its "strict accuracy" headline.)
- **The agent's stops are δ-agnostic** (δ is NOT one of its 9 state features), so I re-scored the
  *current* agent against the other-δ Oracles by parsing `pdfs/tables.tex`. Per-sample-row
  aggregate mean |shots−Oracle|:
  - **δ=0.05**: Agent **2842**, Inc-TVD 1913, Inc-Hell 3016, Inc-JS 4075 → **agent 2nd, beats Hell &
    JS** (best relative standing).
  - δ=0.10: Agent 2389, Inc-TVD 1165, Inc-Hell 1701, Inc-JS 2291 → agent last (this row-level
    aggregation differs from the paper's 36-trace mean).
  - δ=0.25: Agent 4919 vs ~1340–1766 → agent far worst.
  - **Reason**: the agent overshoots; stricter δ pushes the Oracle up → shrinks the agent's error.
- **Caveats stated to user**: (i) this re-scores a δ=0.10-trained agent off-target — the fair path
  is to **retrain per δ** (0.05 could look even better after retraining); (ii) picking δ post-hoc to
  win is a **reviewer red flag** — recommended framing is "report all three δ as a sensitivity
  analysis; the agent's conservatism is a feature in the strict-accuracy (0.05) regime."

### C. Notebook made δ-switchable (`program-main-enhanced.ipynb`)
Single giant code cell edited (4 surgical string replacements via a Python JSON script; cell
re-parsed as valid Python; JSON valid):
- **`ACTUAL_ORACLE_EPS` (≈line 213) is now the documented single δ switch**, limited to
  `{0.05, 0.10, 0.25}` with an `assert`. It **must** stay at module scope (before the defs) because
  it is bound as the default `eps=` of every function/class below — setting δ in `__main__` would
  NOT propagate.
- **Per-δ output folder**: `__main__` now builds `outdir = f"{tag}-enhanced-{ORACLE_EPS}"`
  (e.g. `generic-enhanced-0.05`, `generic-enhanced-0.1`, `generic-enhanced-0.25`) and routes **all**
  artefacts into it, including `problem_overview.svg` (previously written to repo root). The old
  single `outdir = f"{tag}-enhanced"` block was removed.
- Folder uses the **raw float** → δ=0.10 gives `generic-enhanced-0.1` (not `-0.10`), matching the
  existing Oracle-cache naming (`actual_oracle_eps0.1_batch50`). Offered zero-padding if wanted.
- Switching δ correctly re-derives Oracle labels, the reward target `n*`, the per-δ Oracle cache
  file, and the folder — i.e. it **retrains/re-evaluates cleanly** at the new δ (not just re-scoring).

### D. Page budget
Paper is currently **~20 pp**, must be **15**. Gave the user a prioritised trim plan (no edits made):
- Tier 1 (~3pp): move the two **per-trace tables** out of the PDF to the repo/appendix (keep
  aggregate table); remove **Fig. 2** (training-loop diagram); compress **3.4 Network/hyperparams**.
- Tier 2 (~1.5pp): trim **2.3 RL/DQN background**, **3.2 State Space** justifications, model-selection
  paragraph, fold **Threats to validity** into Limitations.
- Tier 3 (~0.5pp): de-duplicate the under/overshoot explanation (now stated 3×).
- Protect: Fig. 1 (gateway), REST API listing, deployment + applicability subsections, Design
  rationale, aggregate results table, agent-vs-Oracle scatter.

---

## What worked
- **Editing the single-cell notebook via a Python JSON script** (load `nb['cells'][0]['source']`,
  join → `str.replace` unique anchors → `splitlines(keepends=True)` → `json.dump(indent=1,
  ensure_ascii=False)`). Verified anchors are unique first; validated with `json.load` + `ast.parse`.
  This is far safer than the Edit tool on `.ipynb` JSON (newline-escaping hell) or NotebookEdit
  (whole-cell replace).
- **Re-scoring the agent at other δ by parsing `pdfs/tables.tex`** (it contains the Oracle column at
  each δ) — no retraining needed to get a directional answer, because the agent is δ-agnostic.
- **Web search** found the real QSimBench citation (IEEE QCE 2025) to replace the GitHub link.
- Structural LaTeX validation without a compiler (brace / environment balance in Python; anchor
  grep counts).
- `pymupdf` (`fitz`) / the Read tool's `pages=` for the source PDFs (poppler is missing).

## What didn't work / gotchas
- **No LaTeX toolchain** → cannot compile; cannot measure real page count. All page estimates are
  rough.
- **Editor stale-buffer clobbering**: notebooks edited on disk have twice been reverted by the
  user's editor saving an old buffer. **Tell the user to reload the notebook from disk before
  running.**
- **δ cannot be switched from `__main__`** — Python binds default args at def-time, so the switch
  had to stay at module scope (line ~213).
- **Row-level vs 36-trace aggregation** give different absolute MAE numbers; don't mix them. Use
  `make_tables.py` for authoritative figures.
- First `Edit`-tool attempt on the abstract failed on a multi-line block; splitting into a single
  clean anchor worked.

## Next steps
1. **Decide the δ story** with the user: either (a) keep δ=0.10 and report 0.05/0.25 as a sensitivity
   table, or (b) **retrain at δ=0.05** for a fair best-case. To retrain: set `ACTUAL_ORACLE_EPS =
   0.05` (line ~213), reload notebook from disk, run → results land in `generic-enhanced-0.05/`.
2. **Wire `make_tables.py` to the new per-δ folders** (`generic-enhanced-{δ}/multirun_eval-*.csv`)
   and to `parse_sota("0.05")` / `"0.25"`, so comparison tables regenerate per δ. Bump `N_EVAL_RUNS`
   (currently **1**) to ≥10 for the mean±std the paper's reproducibility note promises.
3. **Cut the paper to 15 pp** using the Tier plan above (only once the user OKs which parts go).
4. **Reconcile the stale results numbers** once the user supplies his intended CSV; then re-check the
   Results/Discussion/Limitations prose (the safety-first / "≤200 undershoot" narrative may need
   softening — see the qft-12/14 large undershoots in the current CSV).
5. **Regenerate `dashboard-*.pdf`** from a fresh run if labels still say easy/hard (matplotlib
   exported text as vector paths → can't string-relabel; needs a re-run). Convert SVG→PDF with
   `cairosvg` (conda `cairo` in env `qdrl`).
6. **Camera-ready**: Andrea Cossu's real ORCID; de-anonymise repo URL; decide whether to hide the
   author block for blind submission; verify Fig. 1/Fig. 2/API listing render without overflow.

---

## Project layout (durable facts)
- **Three program versions**, each a single giant code cell in a notebook:
  - `program-main.ipynb` — original (reward `HARD_ASYMMETRIC`, 8-feature state) → `generic/`.
  - `program-main-enhanced.ipynb` — **v1, the version the paper describes** (reward `PRECISION`,
    9-feature state, standard DQN+MSE, absolute overshoot margin m=200, decay λ=2, 1,000 episodes,
    γ=0.997, target sync 2,000 steps). Now **δ-switchable** → `generic-enhanced-{δ}/`.
  - `program-main-enhanced-2.ipynb` — v2 (10-feature, Double DQN+Huber, etc.). **Not used by the
    paper.**
- **δ / Oracle**: a-posteriori optimum (Alg. 1 of the IncExc paper), TVD-based, δ=`ACTUAL_ORACLE_EPS`,
  batch 50, budget **B=20,000**. Cache files `actual_oracle_eps{δ}_batch50.{json,csv}` in repo root.
- **Data split**: test = the 36 paper traces (held out); train/val = 80/20 stratified split of the
  rest (`grover-noancilla` excluded). Env runs `qsimbench`. Python env: conda env **`qdrl`**.
- **Paper build**: `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper` (needs
  `llncs.cls` + `splncs04.bst`, shipped in `paper/`). Tables: `cd paper && python3 make_tables.py`
  (prefers `generic-enhanced/`).
- **Source PDFs** in `pdfs/`: `Thesis.pdf` (bachelor thesis, earlier 8-feature work),
  `IncExc_to_TQC.pdf` (the Oracle / IncrementalExecution baseline; cite as `bisicchia2026shots`,
  arXiv:2606.16965; its §7 "Practical Applicability" seeded our service applicability subsection),
  `pdfs/tables.tex` (SOTA reference numbers at δ∈{0.05,0.10,0.25}), `pdfs/background.tex` (quantum
  basics draft — seeded paper §2.1).

## Useful commands
```bash
# extract the notebook's single code cell for reading/grepping
python3 -c "import json;print(''.join(json.load(open('program-main-enhanced.ipynb'))['cells'][0]['source']))" > /tmp/enh.py

# switch delta & run  (edit ACTUAL_ORACLE_EPS to 0.05/0.1/0.25 first, then reload from disk)
# results -> generic-enhanced-<delta>/

# regenerate paper tables from the preferred CSV
cd paper && python3 make_tables.py

# read source PDFs as text (no poppler)
python3 -c "import fitz;d=fitz.open('pdfs/IncExc_to_TQC.pdf');print(d[0].get_text())"

# SVG -> PDF for figures (needs conda cairo in qdrl)
conda run --no-capture-output -n qdrl python -c "import cairosvg; cairosvg.svg2pdf(url='IN.svg', write_to='OUT.pdf')"
```
