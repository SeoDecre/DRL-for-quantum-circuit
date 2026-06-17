# Learning When to Stop: Deep Reinforcement Learning for Adaptive Shot Allocation

This repository accompanies the paper *"Learning When to Stop: Deep
Reinforcement Learning for Adaptive Shot Allocation in Quantum Circuit
Execution"* (submitted to **ICSOC 2026**, double-blind).

It contains a Deep Q-Network (DQN) agent that learns, under a strict **black-box**
assumption, *when to stop* executing shots of a static quantum circuit: that is,
how many repeated measurements ("shots") are enough to estimate the circuit's
output distribution. The agent never sees the circuit structure or the backend
noise model: it observes only the running statistics of the measured outcomes
and decides, batch by batch, whether to **continue** or **stop**. It is trained
against an *a-posteriori* optimal stop derived from the **point of diminishing
returns**, and evaluated against five reference methods on a strict, leakage-free
split of the [QSimBench](https://github.com/GBisi/qsimbench) benchmark.

> Once trained, the policy is a lightweight network that acts as an online,
> black-box **shot-allocation service**: a quantum-execution platform can query
> it after every batch to decide whether to keep sampling.

---

## Repository layout

```
.
├── program-main.ipynb            # Original agent  (8-feature state, HARD_ASYMMETRIC reward)   -> generic/
├── program-main-enhanced.ipynb   # v1  *** the version described in the paper ***             -> generic-enhanced/
│                                  #     9-feature state, PRECISION reward, standard DQN (MSE)
├── program-main-enhanced-2.ipynb # v2  (10-feature state, Double DQN + Huber, stability vote)  -> generic-enhanced-2/
├── program-parallel.py           # Stand-alone, parallelised training/eval script
├── oracle.ipynb                  # A-posteriori Oracle exploration (Algorithm 1, suffix-stable TVD)
│
├── generic/  generic-enhanced/  generic-enhanced-2/   # version-separated run outputs (CSV + SVG)
├── easy*/  hard*/  unbalanced*/  history/             # additional AgentType runs / checkpoints
├── actual_oracle_eps0.1_batch50.{json,csv}            # cached Oracle labels (eps=0.1, batch=50)
│
├── paper/                        # LaTeX sources (Springer LNCS)
│   ├── paper.tex
│   ├── references.bib
│   ├── make_tables.py            # regenerates paper/tables_generated.tex from a run's CSV
│   ├── tables_generated.tex      # auto-generated tables (do not edit by hand)
│   └── figures/                  # PDF figures used by the paper
└── pdfs/                         # background material (thesis, reference framework, SOTA tables)
```

### Program versions

Each `program-main*.ipynb` is a **single self-contained code cell**. The three
versions differ only in the agent design and write to **separate output folders**
so runs cannot overwrite each other:

| Notebook | State | Reward | Learner | Output folder |
|---|---|---|---|---|
| `program-main.ipynb` | 8 features | `HARD_ASYMMETRIC` | DQN (MSE) | `generic/` |
| **`program-main-enhanced.ipynb`** | **9 features** | **`PRECISION`** | **DQN (MSE)** | **`generic-enhanced/`** |
| `program-main-enhanced-2.ipynb` | 10 features | `PRECISION` + relative margin | Double DQN (Huber) | `generic-enhanced-2/` |

**The paper describes `program-main-enhanced.ipynb` (v1)** and reports the results
in `generic-enhanced/`.

---

## Setup

The code targets **Python 3.12** and uses PyTorch, OpenAI Gym/Gymnasium, NumPy,
and QSimBench. A [conda](https://docs.conda.io/) environment named `qdrl` is the
reference environment:

```bash
conda create -n qdrl python=3.12
conda activate qdrl
pip install torch gym numpy tqdm qsimbench
# for figure conversion only:
conda install -n qdrl cairo && pip install cairosvg
```

---

## Running a training + evaluation

Open the notebook of the version you want (the paper uses
`program-main-enhanced.ipynb`), set the configuration block at the bottom of the
cell, and run the single cell:

```python
NUM_EPISODES = 1000          # production agent in the paper
AGENT_TYPE   = AgentType.GENERIC   # 50/50 small (n<10) / big (n>=10) mix
REWARD_TYPE  = RewardType.PRECISION
SEED         = 42
```

The run will:

1. Build (or load from cache) the **Oracle** labels for all training/validation
   instances (suffix-stable Algorithm 1, TVD, `eps=0.1`, `batch=50`,
   `B=20000`).
2. Train the DQN, checkpointing from episode 400 by **de-noised validation MAE**
   (3 stochastic passes, early stopping with patience 5) and restoring the
   best-validation weights.
3. Evaluate the selected agent on the **36 held-out test traces** and write
   `multirun_eval-generic.{csv,svg}`, the agent-vs-Oracle scatter, and a
   training/evaluation dashboard into the version's output folder.

> **Tip:** if you edit a notebook on disk and also have it open in an editor,
> reload it from disk before running so a stale editor buffer cannot overwrite
> your changes.

### Dataset and split

- **Test set:** the exact 36 benchmark traces of the reference study (18 *small*
  with `n<10`, 18 *big* with `n>=10`), held out and never seen in training.
- **Train/validation:** the remaining 828 traces (after excluding
  `grover-noancilla`), with a stratified 80/20 split for model selection.
- Outcomes are sampled stochastically, so even a deterministic (argmax) policy
  produces a different trajectory each evaluation run.

---

## Reproducing the paper's tables and figures

**Tables.** `make_tables.py` reads the best available run CSV (preferring
`generic-enhanced/` for the paper) and the reference numbers in
`pdfs/tables.tex`, and regenerates `paper/tables_generated.tex`:

```bash
cd paper && python3 make_tables.py
```

**Figures.** Convert a run's SVG dashboards/scatter to the PDFs used by the
paper (requires the native `cairo` library, e.g. via conda):

```bash
conda run -n qdrl python -c "import cairosvg; \
  cairosvg.svg2pdf(url='generic-enhanced/agent_vs_oracle-generic-all36.svg', \
                   write_to='paper/figures/agent_vs_oracle-generic-all36.pdf')"
```

**Building the paper** (needs a LaTeX toolchain with `llncs.cls`/`splncs04.bst`,
both shipped in `paper/`):

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

---

## Results at a glance (paper / `generic-enhanced`)

On the 36 held-out traces (`δ=0.10`, budget `B=20000`), the learned policy
reaches a mean absolute deviation from the Oracle of **2,049 shots (10.2 % of the
budget)**, undershooting on only **3/36** traces (each within 200 shots). It
beats the Hellinger and Jensen-Shannon online policies (2,706 and 3,336) and the
a-priori Weissman/Hoeffding bounds by orders of magnitude, while trailing only
the strongest hand-tuned online policy, Inc-TVD (871). Its residual error is
conservative over-sampling on **mid-range** circuits.

---

## License

Released as open source to accompany the paper. (License to be added with the
de-anonymised, camera-ready release.)
