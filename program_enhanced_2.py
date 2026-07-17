###############################################################################
# DRL for Quantum Circuit Shot Allocation  —  ENHANCED-2 (+extrap) MODULE
# ----------------------------------------
# This module IS the program: program-main-enhanced-2.ipynb is now a thin
# launcher that does `from program_enhanced_2 import *; main()`.
# Edit THIS file, not the notebook.
# ----------------------------------------
# Aligned with the IncExc_to_TQC paper (Bisicchia et al., 2026):
#   • TEST SET = exact 36 traces from Tables 8-13 of the paper
#     (each algorithm uses specific size/backend pairs, NOT a full cartesian product)
#   • TRAIN SET = 864 other traces from qsimbench — zero data leakage
#   • AgentType controls training AND mid-training validation distribution:
#       EASY       → 100% easy training & validation
#       HARD       → 100% hard training & validation
#       GENERIC    → 50/50 balanced mix (train & validation)
#       UNBALANCED → 20% easy / 80% hard (train & validation)
#   • Final evaluation ALWAYS uses ALL available problems (50% easy, 50% hard):
#       - Eval set: all 36 paper problems (18 easy + 18 hard)
#       - Train-set eval: all 864 training problems (overfitting diagnostic)
#   • Small/Large classification from Ch. 6.1:
#       SplitMetric.SIZE   → size < 10 → easy, size >= 10 → hard
#       SplitMetric.ORACLE → oracle <= 5000 → easy, oracle > 5000 → hard
#   • Best-model checkpointing (every 100 eps from ep 2000, keep lowest val MAE)
#   • Early stopping with patience parameter
#   • MAE evaluated on the held-out 36 paper problems
#   • Error percentage = MAE / max_shots
#   • SVG outputs with easy/hard tagging
#   • Oracle = a posteriori optimal (paper definition, §6.1):
#       "minimum n such that D(P̂_n, P̂_B) ≤ δ"  with B=20000, δ=0.1
###############################################################################
# ENHANCED VERSION — changes vs program-main.ipynb:
#   1. Inference fix: dropout disabled during greedy action selection
#      (q_net.eval() in act(), target_net kept in eval mode) — the greedy
#      policy is now deterministic given a trajectory.
#   2. Target network synced every TARGET_SYNC_STEPS replay steps (default
#      2000) instead of every 500 EPISODES (which meant a single sync in a
#      1000-episode run, bootstrapping off a frozen random net until ep 500).
#   3. ε-greedy decays once per EPISODE (0.995/ep → ≈0.05 by ep 600) instead
#      of per replay step (which ended exploration after ~150 episodes and
#      coupled the schedule to episode length).
#   4. Discount factor γ raised 0.99 → 0.997 so the terminal reward survives
#      long-horizon credit assignment (γ^400 ≈ 0.30 instead of 0.018).
#   5. New RewardType.PRECISION: full reward on [0, +OVERSHOOT_MARGIN] shots
#      of overshoot (explicit slight-overshoot incentive), then exponential
#      decay RELATIVE to the oracle value (a +800 overshoot on a 50-shot
#      problem is punished hard, +800 on a 15k problem stays mild); the
#      undershoot base penalty is keyed to ORACLE difficulty, not circuit
#      size (HARD_ASYMMETRIC punished dj_14 with oracle=150 as if hard).
#   6. State vector 8 → 9 features (STATE_SIZE):
#      • entropy normalised by circuit size (max entropy = n qubits in bits;
#        the old /4.0 clipped to 1.0 on every large circuit)
#      • concentration Σp² replaces var(p)·10 — scale-free, does not vanish
#        on large outcome spaces
#      • rate-of-change = TVD between successive CUMULATIVE distributions
#        (the paper's Def. 3.3 diminishing-returns proxy) at two lags
#        (1 batch and RATE_LONG_LAG batches); the old version compared the
#        cumulative against the last 50-shot batch alone, which stays ≈1
#        forever on large outcome spaces
#      • relative-progress heuristic size·1500 (was size·500, which pegged
#        at 1.0 from 7000 shots for 14-qubit circuits whose oracles sit at
#        13k–17.8k)
#   7. grover-noancilla exclusion fixed: the check was on the BACKEND string,
#      so 36 grover-noancilla traces silently leaked into the train/val pool.
#   8. Oracle-stratified hard/easy sampling enabled for BOTH split metrics
#      (was active only when SPLIT_METRIC == ORACLE).
#   9. Snapshot validation averages VALIDATION_RUNS stochastic passes per
#      checkpoint to de-noise best-model selection.
#  10. step(): failed get_outcomes batches are counted and reported instead
#      of a silent `except: pass`; the CONTINUE reward is the explicit
#      STEP_PENALTY constant (0.0 — keep the paper text in sync).
###############################################################################
# ENHANCED-2 — precision pass (keep the band above the diagonal, but tighter):
#   A. OVERSHOOT_REL_DECAY 2.0 → 4.0 — steeper relative decay narrows the
#      overshoot band the expected-reward optimum sits in.
#   B. Relative reward plateau: margin = max(50, 0.03·oracle) instead of an
#      absolute 200 shots (which made a 5× overshoot "perfect" on oracle=50).
#   C. Forced budget-cap terminations beyond the margin earn only
#      FORCED_CAP_FACTOR (0.5) of the reward — "ride the cap and never
#      decide" is no longer a comfortable default on hard problems.
#   D. Double DQN + Huber (smooth-L1) loss — removes the max-operator
#      overestimation that inflates Q(CONTINUE) and stops the undershoot-cliff
#      targets from dominating the gradients.
#   E. 10th state feature: convergence STREAK — consecutive batches with the
#      long-lag cumulative TVD below the Oracle's eps (sustained stability,
#      analogous to the IE framework's k-consecutive stability criterion and
#      a near-sufficient statistic for the Oracle's stopping decision).
#   F. Stability voting at evaluation/validation: STOP executes only after
#      EVAL_STOP_VOTES (2) consecutive greedy STOP decisions — implemented in
#      run_greedy_episode(), used by ALL eval loops incl. checkpoint selection.
#   G. Longer training: 2000 episodes, ε-decay 0.997/episode, checkpoints from
#      ep 800 (multi-pass validation is denoised, so longer training is safe).
#   H. (extrap pass) 11th state feature: ORACLE EXTRAPOLATION — each valid
#      rate_long observation yields a √n-law coefficient c ≈ rate_long·n/√l
#      (TVD(P̂_n, P̂_{n-l}) ≈ c·√l/n under IID sampling, while the Oracle's
#      signal TVD(P̂_n, P̂_B) ≈ c/√n); the median c predicts the stopping
#      point n̂* = 1/(ε²/c² + 1/B), exposed as min(n̂*/B, 1). Motivated
#      by the oracle-variance study (oracle_variance.py): n* is stable across
#      realizations (MAD ≈ 141 shots), so a statistical extrapolation of the
#      convergence curve is highly informative. Outputs → *-enhanced-2-extrap/.
###############################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, Counter
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gym
from gym import spaces
from qsimbench import get_outcomes, get_index
from typing import Dict, List, Tuple, Optional
from enum import Enum
from tqdm import tqdm
import os
import json
from ast import literal_eval
import copy
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# Agent type: controls training distribution bias
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(Enum):
    EASY       = "easy"        # trains ONLY on easy problems (size < 10)
    HARD       = "hard"        # trains ONLY on hard problems (size >= 10)
    GENERIC    = "generic"     # trains on a balanced 50/50 mix of easy and hard
    UNBALANCED = "unbalanced"  # trains on 20% easy, 80% hard

class RewardType(Enum):
    MAE             = "mae"              # exp(-0.003 * mae) for overshoot, -1 - mae/optimal for undershoot
    ASYMMETRIC      = "asymmetric"       # exp(-0.00025 * error) for overshoot, (-1.85 - |e|/opt) * scale(1-1.5) for undershoot
    HARD_ASYMMETRIC = "hard_asymmetric"  # same as ASYMMETRIC for easy; harder undershoot penalty for hard problems
    PRECISION       = "precision"        # full reward on a RELATIVE overshoot plateau, steep oracle-relative decay beyond it; undershoot keyed to ORACLE difficulty; forced-cap stops penalised

class SplitMetric(Enum):
    SIZE   = "size"    # easy/hard split by circuit size (size < 10)
    ORACLE = "oracle"  # easy/hard split by oracle optimal shots (oracle <= 5000)

# ─────────────────────────────────────────────────────────────────────────────
# Paper-aligned constants (Ch. 6.1 of IncExc_to_TQC)
# ─────────────────────────────────────────────────────────────────────────────
# Exact 36 traces from Tables 8-13 (each algorithm uses specific size/backend pairs)
PAPER_TRACES: List[Tuple[str, int, str]] = [
    ("dj", 4, "fake_fez"), ("dj", 4, "fake_torino"),
    ("dj", 8, "fake_fez"), ("dj", 8, "fake_marrakesh"),
    ("dj", 10, "fake_fez"), ("dj", 10, "fake_torino"),
    ("dj", 14, "fake_fez"), ("dj", 14, "fake_marrakesh"),
    ("qaoa", 4, "fake_sherbrooke"),
    ("qaoa", 6, "fake_marrakesh"),
    ("qaoa", 8, "fake_kyiv"), ("qaoa", 8, "fake_sherbrooke"),
    ("qaoa", 10, "fake_sherbrooke"),
    ("qaoa", 12, "fake_marrakesh"),
    ("qaoa", 14, "fake_kyiv"), ("qaoa", 14, "fake_sherbrooke"),
    ("qft", 6, "fake_fez"), ("qft", 6, "fake_torino"),
    ("qft", 8, "fake_fez"), ("qft", 8, "fake_torino"),
    ("qft", 12, "fake_fez"), ("qft", 12, "fake_torino"),
    ("qft", 14, "fake_fez"), ("qft", 14, "fake_torino"),
    ("qnn", 8, "fake_fez"), ("qnn", 8, "fake_kyiv"),
    ("qnn", 14, "fake_fez"), ("qnn", 14, "fake_kyiv"),
    ("random", 6, "fake_fez"),
    ("random", 8, "fake_fez"),
    ("random", 12, "fake_fez"),
    ("random", 14, "fake_fez"),
    ("vqe", 6, "fake_kyiv"),
    ("vqe", 8, "fake_kyiv"),
    ("vqe", 12, "fake_kyiv"),
    ("vqe", 14, "fake_kyiv"),
]

SMALL_LARGE_THRESHOLD = 10   # size < 10 → easy, size >= 10 → hard
ORACLE_EASY_THRESHOLD = 5_000  # oracle <= 5000 → easy (when using SplitMetric.ORACLE)
MAX_SHOTS = 20_000
LOW_SHOT_THRESHOLD = 5_000   # counter: problems solved with < 5000 shots

# ── Enhanced-version constants ───────────────────────────────────────────────
STEP_PENALTY        = 0.0    # reward per CONTINUE step (paper §Reward Design says -0.02 — keep the two in sync)
OVERSHOOT_REL_DECAY = 4.0    # decay on (overshoot-margin)/oracle beyond the margin (v2: was 2.0 — steeper, tighter band)
OVERSHOOT_MARGIN_MIN  = 50    # v2: absolute floor of the full-reward overshoot plateau
OVERSHOOT_MARGIN_FRAC = 0.03  # v2: plateau as a fraction of the oracle — margin = max(MIN, FRAC·oracle)
FORCED_CAP_FACTOR   = 0.5    # v2: positive-reward multiplier when the budget cap forces termination beyond the margin
RATE_LONG_LAG       = 10     # batches (×50 shots) lag for the long-horizon rate-of-change feature
STREAK_CAP          = 20     # v2: batches of sustained sub-eps stability that saturate the streak feature
EXTRAP_MIN_SAMPLES  = 3      # v3: min valid rate_long samples before the extrapolation feature activates
EXTRAP_WINDOW       = 40     # v3: recent c samples used for the median (drops the biased early transient)
EVAL_STOP_VOTES     = 2      # v2: consecutive greedy STOP decisions required to stop at evaluation (1 = voting off)
TARGET_SYNC_STEPS   = 2000   # replay (gradient) steps between target-network hard syncs
VALIDATION_RUNS     = 3      # stochastic validation passes averaged per snapshot/checkpoint
DOUBLE_DQN          = True   # v2: online net selects a', target net evaluates it (kills max-bias toward CONTINUE)
HUBER_LOSS          = True   # v2: smooth-L1 instead of MSE (undershoot-cliff targets stop dominating gradients)
GAMMA               = 0.997  # discount factor (was 0.99; γ^400 ≈ 0.30 instead of 0.018)
STATE_SIZE          = 11     # v3: 10 v2 features + oracle-extrapolation feature

# Active split metric — set in __main__ config block
SPLIT_METRIC = SplitMetric.SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Cached qsimbench index  — called once, reused everywhere
# ─────────────────────────────────────────────────────────────────────────────
_INDEX_CACHE: Optional[dict] = None

def get_cached_index() -> dict:
    """Return qsimbench index, fetching it only once."""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = get_index()
    return _INDEX_CACHE

# ─────────────────────────────────────────────────────────────────────────────
# Global feature-encoding maps  — built ONCE from the full qsimbench index
# so that every environment (train / validation / eval) encodes the same
# algorithm / backend with the same normalised value.
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_ALG_MAP: Optional[Dict[str, int]] = None
_GLOBAL_BACKEND_MAP: Optional[Dict[str, int]] = None

def get_global_mappings() -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (alg_map, backend_map) derived from the FULL qsimbench index.
    Cached after the first call."""
    global _GLOBAL_ALG_MAP, _GLOBAL_BACKEND_MAP
    if _GLOBAL_ALG_MAP is not None:
        return _GLOBAL_ALG_MAP, _GLOBAL_BACKEND_MAP
    index = get_cached_index()
    all_algs: set = set()
    all_backends: set = set()
    for alg in index:
        all_algs.add(alg)
        for sz in index[alg]:
            val = index[alg][sz]
            if isinstance(val, dict):
                all_backends.update(val.keys())
            else:
                all_backends.update(val)
    _GLOBAL_ALG_MAP     = {n: i for i, n in enumerate(sorted(all_algs))}
    _GLOBAL_BACKEND_MAP = {n: i for i, n in enumerate(sorted(all_backends))}
    return _GLOBAL_ALG_MAP, _GLOBAL_BACKEND_MAP

# ─────────────────────────────────────────────────────────────────────────────
# Actual Oracle — Algorithm 1 (A Posteriori Optimal Sample Count,
# Bisicchia et al. 2026, IncExc_to_TQC §3.1)
# ─────────────────────────────────────────────────────────────────────────────

# Default parameters — change here to rerun with different settings
ACTUAL_ORACLE_EPS:       float = 0.1      # ε threshold (δ in the paper)
ACTUAL_ORACLE_BATCH:     int   = 50       # shots per batch
ACTUAL_ORACLE_MAX_SHOTS: int   = MAX_SHOTS  # hard budget B (=20000)

# In-memory cache: (algorithm, size, backend) → oracle_shots
ACTUAL_ORACLE_CACHE: Dict[Tuple[str, int, str], int] = {}


def _actual_oracle_cache_paths(
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
) -> Tuple[str, str]:
    """Return (json_path, csv_path) for the given (eps, batch) combo."""
    base = f"actual_oracle_eps{eps}_batch{batch}"
    return base + ".json", base + ".csv"


def save_actual_oracle_cache(
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
) -> None:
    """Persist ACTUAL_ORACLE_CACHE to JSON + human-readable CSV.

    CSV columns: algorithm, size, backend, oracle_shots, epsilon, batch
    """
    json_path, csv_path = _actual_oracle_cache_paths(eps, batch)

    # JSON (for program use)
    string_key_cache = {str(k): v for k, v in ACTUAL_ORACLE_CACHE.items()}
    with open(json_path, "w") as f:
        json.dump(string_key_cache, f, indent=2)

    # CSV (human-readable, for comparing with paper tables)
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "size", "backend", "oracle_shots", "epsilon", "batch"])
        for (alg, sz, be), shots in sorted(ACTUAL_ORACLE_CACHE.items()):
            writer.writerow([alg, sz, be, shots, eps, batch])
    print(f"Actual oracle cache saved → {json_path}  ({len(ACTUAL_ORACLE_CACHE)} entries)")
    print(f"Human-readable CSV saved  → {csv_path}")


_oracle_cache_loaded: set = set()   # tracks which (eps, batch) files we already loaded

def load_actual_oracle_cache(
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
) -> None:
    """Load ACTUAL_ORACLE_CACHE from JSON if it exists.
    Skips if the same (eps, batch) file was already loaded this session."""
    cache_id = (eps, batch)
    if cache_id in _oracle_cache_loaded:
        return                       # already loaded — nothing to do

    json_path, _ = _actual_oracle_cache_paths(eps, batch)
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                raw = json.load(f)
            n_loaded = 0
            for k_str, v in raw.items():
                ACTUAL_ORACLE_CACHE[literal_eval(k_str)] = int(v)
                n_loaded += 1
            print(f"Loaded {n_loaded} actual oracle entries from {json_path}")
        except Exception as e:
            print(f"Warning: could not read actual oracle cache. Error: {e}")

    _oracle_cache_loaded.add(cache_id)


def compute_actual_oracle(
    algorithm: str,
    size: int,
    backend: str,
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
    max_shots: int = ACTUAL_ORACLE_MAX_SHOTS,
    seed: int = 42,
    force: bool = False,
) -> int:
    """
    Algorithm 1 — A Posteriori Optimal Sample Count (Suffix-Stable).
    Bisicchia et al. 2026, IncExc_to_TQC §3.1, page 8.

    Given B = max_shots total measurement outcomes (collected in batches of
    `batch` shots each):

      Step 1: Build reference distribution P̂_B from all B outcomes.
      Step 2: Compute d[m] = TVD(P̂_m, P̂_B) for every prefix m.
      Step 3: n* = earliest m such that max_{m'>=m} d[m'] <= eps.

    Returns n* (oracle shots, always a multiple of `batch`).
    Falls back to max_shots if the suffix condition is never satisfied.

    Skips any algorithm whose name contains 'noancilla' (case-insensitive).
    Result is stored in ACTUAL_ORACLE_CACHE.
    """
    if "noancilla" in algorithm.lower():
        return -1  # excluded algorithm variant (fix: the check was on the backend string)

    cache_key = (algorithm, size, backend)
    if not force and cache_key in ACTUAL_ORACLE_CACHE:
        return ACTUAL_ORACLE_CACHE[cache_key]

    effective_max = (max_shots // batch) * batch
    if effective_max == 0:
        return max_shots

    # ── Step 1 & prep for Step 2: collect snapshots ──────────────────
    cumulative: Counter = Counter()
    snapshots:  List[Dict[str, int]] = []
    shot_counts: List[int] = []

    for bi in range(effective_max // batch):
        try:
            new_outcomes = get_outcomes(
                algorithm, size, backend, shots=batch, seed=seed + 1 + bi
            )
        except Exception:
            # If the backend errors out fall back to max_shots
            ACTUAL_ORACLE_CACHE[cache_key] = max_shots
            return max_shots
        for k, v in new_outcomes.items():
            cumulative[k] = cumulative.get(k, 0) + int(v)
        snapshots.append(dict(cumulative))
        shot_counts.append((bi + 1) * batch)

    if not snapshots:
        ACTUAL_ORACLE_CACHE[cache_key] = max_shots
        return max_shots

    # ── Step 1: P̂_B = last snapshot (full-budget reference) ─────────
    ref = snapshots[-1]
    ref_total = sum(ref.values())
    if ref_total == 0:
        ACTUAL_ORACLE_CACHE[cache_key] = max_shots
        return max_shots
    ref_norm = {k: v / ref_total for k, v in ref.items()}

    # ── Step 2: d[m] = TVD(P̂_m, P̂_B) for all m ────────────────────
    dist_values: List[float] = []
    for snap in snapshots:
        snap_total = sum(snap.values())
        if snap_total == 0:
            dist_values.append(1.0)
            continue
        snap_norm = {k: v / snap_total for k, v in snap.items()}
        all_keys = set(ref_norm) | set(snap_norm)
        tvd = 0.5 * sum(
            abs(snap_norm.get(k, 0.0) - ref_norm.get(k, 0.0)) for k in all_keys
        )
        dist_values.append(tvd)

    # ── Step 3: find earliest n s.t. suffix max d[m] <= eps ──────────
    # Walk backwards: maintain running suffix_max; whenever it stays <= eps,
    # update n* to the current index (i.e. the earliest we can walk back to).
    n_star = shot_counts[-1]   # worst case: use the full budget
    suffix_max = 0.0
    for idx in range(len(dist_values) - 1, -1, -1):
        suffix_max = max(suffix_max, dist_values[idx])
        if suffix_max <= eps:
            n_star = shot_counts[idx]

    ACTUAL_ORACLE_CACHE[cache_key] = n_star
    return n_star


def build_actual_oracle_dataset(
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
    force: bool = False,
) -> List[Tuple[str, int, str]]:
    """Compute the actual oracle for every qsimbench triplet, excluding:
      • PAPER_TRACES  (reserved for test evaluation)
      • Algorithms containing 'noancilla' (case-insensitive)

    Results are cached to JSON + CSV files.  Pass force=True to recompute.
    Returns the sorted list of successfully computed (algorithm, size, backend)
    triplets, which forms the dataset used for training and validation.
    """
    load_actual_oracle_cache(eps, batch)

    index = get_cached_index()
    paper_set = set(PAPER_TRACES)

    all_triplets: List[Tuple[str, int, str]] = []
    for alg in sorted(index.keys()):
        if "noancilla" in alg.lower():
            continue  # excluded algorithm variant (fix: the check was on the backend string)
        for sz in sorted(index[alg].keys()):
            val = index[alg][sz]
            backends = sorted(val.keys()) if isinstance(val, dict) else sorted(val)
            for be in backends:
                t = (alg, sz, be)
                if t in paper_set:
                    continue
                all_triplets.append(t)

    uncached = [t for t in all_triplets if t not in ACTUAL_ORACLE_CACHE]
    if uncached or force:
        print(f"Computing actual oracle (Algorithm 1) for {len(uncached)} triplets "
              f"(eps={eps}, batch={batch}, max_shots={ACTUAL_ORACLE_MAX_SHOTS})...")
        for t in tqdm(uncached, desc="Actual oracle"):
            compute_actual_oracle(*t, eps=eps, batch=batch, force=force)
        save_actual_oracle_cache(eps, batch)

    return [t for t in all_triplets if t in ACTUAL_ORACLE_CACHE]


def find_optimal_shots(
    algorithm: str,
    size: int,
    backend: str,
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
    max_shots: int = ACTUAL_ORACLE_MAX_SHOTS,
) -> int:
    """Look up the actual oracle from ACTUAL_ORACLE_CACHE.
    If not cached yet, computes it on demand (should only happen for paper traces
    during evaluation, since training triplets are pre-computed by
    build_actual_oracle_dataset())."""
    cache_key = (algorithm, size, backend)
    if cache_key in ACTUAL_ORACLE_CACHE:
        return ACTUAL_ORACLE_CACHE[cache_key]
    return compute_actual_oracle(
        algorithm, size, backend, eps=eps, batch=batch, max_shots=max_shots
    )

# ─────────────────────────────────────────────────────────────────────────────
# Problem sets: paper test set (36), training set, validation set
# Training + validation come from the ACTUAL oracle dataset (Algorithm 1),
# stratified 80/20 split, excluding paper traces and noancilla backends.
# ─────────────────────────────────────────────────────────────────────────────

def classify_problem(triplet: Tuple[str, int, str]) -> str:
    """Classify a problem as easy or hard based on the active SPLIT_METRIC.
      SIZE   → size < 10 → easy  (Paper Ch. 6.1)
      ORACLE → oracle ≤ 5000 → easy
    """
    if SPLIT_METRIC == SplitMetric.SIZE:
        return "easy" if triplet[1] < SMALL_LARGE_THRESHOLD else "hard"
    else:  # ORACLE
        oracle = find_optimal_shots(*triplet)
        return "easy" if oracle <= ORACLE_EASY_THRESHOLD else "hard"

_paper_triplets_cache: Optional[List[Tuple[str, int, str]]] = None

def build_paper_triplets() -> List[Tuple[str, int, str]]:
    """The exact 36 traces from Tables 8-13 of the paper.
    Each algorithm uses specific (size, backend) pairs — NOT a full cartesian product.
    These are used ONLY for evaluation — never seen during training.
    Result is cached after the first call."""
    global _paper_triplets_cache
    if _paper_triplets_cache is not None:
        return _paper_triplets_cache
    _paper_triplets_cache = list(PAPER_TRACES)
    return _paper_triplets_cache

# ── Stratified train/val split (cached after first call) ─────────────────────
_actual_train_split_cache: Optional[List[Tuple[str, int, str]]] = None
_actual_val_split_cache:   Optional[List[Tuple[str, int, str]]] = None

def build_stratified_split(
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    """Build a stratified 80/20 train/val split from the actual oracle dataset.

    • Easy and hard problems are split *separately* so the easy/hard ratio is
      preserved in both subsets.
    • The full actual oracle dataset is pre-computed (or loaded from cache)
      before the split.
    • Paper traces and noancilla algorithms are already excluded by
      build_actual_oracle_dataset().

    Returns (train_triplets, val_triplets).
    """
    global _actual_train_split_cache, _actual_val_split_cache
    if _actual_train_split_cache is not None:
        return _actual_train_split_cache, _actual_val_split_cache

    dataset = build_actual_oracle_dataset(eps=eps, batch=batch)

    easy = [t for t in dataset if classify_problem(t) == "easy"]
    hard = [t for t in dataset if classify_problem(t) == "hard"]

    rng = random.Random(seed)
    rng.shuffle(easy)
    rng.shuffle(hard)

    n_easy_train = int(len(easy) * train_ratio)
    n_hard_train = int(len(hard) * train_ratio)

    easy_train, easy_val = easy[:n_easy_train],  easy[n_easy_train:]
    hard_train, hard_val = hard[:n_hard_train],  hard[n_hard_train:]

    _actual_train_split_cache = easy_train + hard_train
    _actual_val_split_cache   = easy_val   + hard_val

    print(f"Stratified split  →  train: {len(_actual_train_split_cache)} "
          f"({len(easy_train)}E + {len(hard_train)}H),  "
          f"val: {len(_actual_val_split_cache)} "
          f"({len(easy_val)}E + {len(hard_val)}H)")
    return _actual_train_split_cache, _actual_val_split_cache


_training_triplets_cache: Dict[str, List[Tuple[str, int, str]]] = {}

def build_training_triplets(
    agent_type: AgentType = AgentType.GENERIC,
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """Return the 80% training split of the actual oracle dataset.

    The pool is adjusted to match the agent type's target distribution:
      GENERIC    → exactly 50/50 easy/hard (trim larger group to match smaller)
      EASY       → only easy problems
      HARD       → only hard problems
      UNBALANCED → all problems (20/80 sampling handled in env.reset)

    Stratified by easy/hard so the ratio is preserved.  Cached by agent_type."""
    cache_key = agent_type.value
    if cache_key in _training_triplets_cache:
        return _training_triplets_cache[cache_key]

    train, _ = build_stratified_split(eps=eps, batch=batch, seed=seed)

    easy = [t for t in train if classify_problem(t) == "easy"]
    hard = [t for t in train if classify_problem(t) == "hard"]

    if agent_type == AgentType.GENERIC:
        # Exactly 50/50: trim the larger group to match the smaller
        n = min(len(easy), len(hard))
        rng = random.Random(seed + 100)
        rng.shuffle(easy)
        rng.shuffle(hard)
        result = easy[:n] + hard[:n]
    elif agent_type == AgentType.EASY:
        result = easy
    elif agent_type == AgentType.HARD:
        result = hard
    else:  # UNBALANCED — keep full pool, sampling ratio in env.reset()
        result = train

    _training_triplets_cache[cache_key] = result
    return result

_validation_set_cache: Dict[str, List[Tuple[str, int, str]]] = {}

def build_validation_set(
    agent_type: AgentType = AgentType.GENERIC,
    seed: int = 42,
    eps: float = ACTUAL_ORACLE_EPS,
    batch: int = ACTUAL_ORACLE_BATCH,
) -> List[Tuple[str, int, str]]:
    """Build a validation set from the 20% held-out split of the actual oracle
    dataset for mid-training checkpointing.

    The easy/hard selection mirrors the AgentType training distribution so
    that the validation MAE tracks the kind of problems the agent trains on:
      EASY       → only easy problems from the val split
      HARD       → only hard problems from the val split
      GENERIC    → exactly 50/50 easy/hard (trim larger group to match smaller)
      UNBALANCED → ~20% easy, ~80% hard sampled from the val split

    These problems come from the 20% held-out split (never used for training),
    so the validation signal is a genuine out-of-sample check.

    Cached by agent_type.value."""
    cache_key = agent_type.value
    if cache_key in _validation_set_cache:
        return _validation_set_cache[cache_key]

    _, val = build_stratified_split(eps=eps, batch=batch, seed=seed)
    easy = [t for t in val if classify_problem(t) == "easy"]
    hard = [t for t in val if classify_problem(t) == "hard"]

    if agent_type == AgentType.EASY:
        result = easy
    elif agent_type == AgentType.HARD:
        result = hard
    elif agent_type == AgentType.UNBALANCED:
        rng = random.Random(seed)
        n_total = len(easy) + len(hard)
        n_easy  = max(1, round(n_total * 0.2))
        n_hard  = n_total - n_easy
        rng.shuffle(easy)
        rng.shuffle(hard)
        result = easy[:min(n_easy, len(easy))] + hard[:min(n_hard, len(hard))]
    else:  # GENERIC → exactly 50/50 (trim larger group)
        n = min(len(easy), len(hard))
        rng_g = random.Random(seed + 200)
        rng_g.shuffle(easy)
        rng_g.shuffle(hard)
        result = easy[:n] + hard[:n]

    _validation_set_cache[cache_key] = result
    return result

_eval_set_cache: Dict[Tuple[str, int], List[Tuple[str, int, str]]] = {}

def build_eval_set(
    agent_type: AgentType,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """Build the evaluation set from the 36 paper problems.

    Always uses ALL 18 easy + ALL 18 hard = 36 problems so that every
    paper problem is evaluated regardless of the agent type.

    Cached by (agent_type.value, seed)."""
    cache_key = (agent_type.value, seed)
    if cache_key in _eval_set_cache:
        return _eval_set_cache[cache_key]

    paper = build_paper_triplets()
    easy = [t for t in paper if classify_problem(t) == "easy"]
    hard = [t for t in paper if classify_problem(t) == "hard"]

    # Use ALL easy + ALL hard = full 36 paper problems
    selected = easy + hard
    _eval_set_cache[cache_key] = selected
    return selected


def build_filtered_eval_set(
    agent_type: AgentType,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """Return the agent-type-filtered SUBSET of the 36 paper test problems.

    EASY       → only the 18 easy paper problems
    HARD       → only the 18 hard paper problems
    GENERIC    → all 36
    UNBALANCED → 20% easy / 80% hard sample (seeded)

    This is used for a focused scatter chart alongside the full-36 chart."""
    paper = build_paper_triplets()
    easy = [t for t in paper if classify_problem(t) == "easy"]
    hard = [t for t in paper if classify_problem(t) == "hard"]

    if agent_type == AgentType.EASY:
        return easy
    elif agent_type == AgentType.HARD:
        return hard
    elif agent_type == AgentType.UNBALANCED:
        rng = random.Random(seed)
        n_total = len(easy) + len(hard)
        n_easy  = max(1, round(n_total * 0.2))
        n_hard  = n_total - n_easy
        rng.shuffle(easy)
        rng.shuffle(hard)
        return easy[:min(n_easy, len(easy))] + hard[:min(n_hard, len(hard))]
    else:  # GENERIC → all 36
        return easy + hard


def get_config_label(agent_type: AgentType = None, for_svg: bool = False) -> str:
    """Return a one-line label describing the current run configuration.
    Used in SVG headers, plot titles, and dashboard suptitles.
    If for_svg=True, uses XML-safe characters (&lt; instead of <)."""
    atype = (agent_type or AGENT_TYPE).value.upper()
    reward = REWARD_TYPE.value.upper()
    if SPLIT_METRIC == SplitMetric.SIZE:
        lt = "&lt;" if for_svg else "<"
        split = f"SIZE (size {lt} {SMALL_LARGE_THRESHOLD})"
    else:
        lte = "&lt;=" if for_svg else "≤"
        split = f"ORACLE (oracle {lte} {ORACLE_EASY_THRESHOLD})"
    return f"Agent: {atype}  |  Reward: {reward}  |  Split: {split}"


def format_training_split_stats(
    train_triplets: List[Tuple[str, int, str]],
    indent: str = "   ",
) -> str:
    """Return a formatted string describing the training set composition
    based on the active SPLIT_METRIC.

      ORACLE → single line:  < 5000 shots (oracle) : N/total (N%)  [NE + NH]
      SIZE   → table by circuit size with count and percentage
    """
    n_total = len(train_triplets)

    if SPLIT_METRIC == SplitMetric.ORACLE:
        n_low = sum(1 for t in train_triplets
                    if ACTUAL_ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        n_low_easy = sum(1 for t in train_triplets
                         if classify_problem(t) == "easy"
                         and ACTUAL_ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        n_low_hard = sum(1 for t in train_triplets
                         if classify_problem(t) == "hard"
                         and ACTUAL_ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        return (f"{indent}< {LOW_SHOT_THRESHOLD} shots (oracle) : {n_low}/{n_total} "
                f"({n_low/n_total*100:.1f}%)  "
                f"[{n_low_easy}E + {n_low_hard}H]")
    else:  # SIZE
        from collections import Counter as Ctr
        size_counts = Ctr(t[1] for t in train_triplets)
        sizes = sorted(size_counts.keys())
        lines = [f"{indent}Training problems by circuit size:"]
        lines.append(f"{indent}  {'Size':>6s}  {'Count':>5s}  {'%':>6s}  {'Class':>5s}")
        lines.append(f"{indent}  {'─'*30}")
        for sz in sizes:
            cnt = size_counts[sz]
            pct = cnt / n_total * 100
            cat = "easy" if sz < SMALL_LARGE_THRESHOLD else "hard"
            lines.append(f"{indent}  q={sz:<4d}  {cnt:>5d}  {pct:>5.1f}%  [{cat}]")
        lines.append(f"{indent}  {'─'*30}")
        n_easy = sum(c for s, c in size_counts.items() if s < SMALL_LARGE_THRESHOLD)
        n_hard = sum(c for s, c in size_counts.items() if s >= SMALL_LARGE_THRESHOLD)
        lines.append(f"{indent}  Total : {n_total}  "
                     f"[{n_easy}E + {n_hard}H]")
        return "\n".join(lines)


def format_training_split_stats_compact(
    train_triplets: List[Tuple[str, int, str]],
) -> str:
    """Compact version for the dashboard text box (monospace, no indent)."""
    n_total = len(train_triplets)

    if SPLIT_METRIC == SplitMetric.ORACLE:
        n_low = sum(1 for t in train_triplets
                    if ACTUAL_ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        n_low_easy = sum(1 for t in train_triplets
                         if classify_problem(t) == "easy"
                         and ACTUAL_ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        n_low_hard = sum(1 for t in train_triplets
                         if classify_problem(t) == "hard"
                         and ACTUAL_ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        return (f"Train < {LOW_SHOT_THRESHOLD} (oracle) : {n_low:>5d}/{n_total}  ({n_low/n_total*100:.1f}%)"
                f"  [{n_low_easy}E+{n_low_hard}H]")
    else:  # SIZE
        from collections import Counter as Ctr
        size_counts = Ctr(t[1] for t in train_triplets)
        sizes = sorted(size_counts.keys())
        lines = ["Train by size:"]
        for sz in sizes:
            cnt = size_counts[sz]
            pct = cnt / n_total * 100
            cat = "E" if sz < SMALL_LARGE_THRESHOLD else "H"
            lines.append(f"  q={sz:<3d}: {cnt:>3d} ({pct:>4.1f}%) [{cat}]")
        n_easy = sum(c for s, c in size_counts.items() if s < SMALL_LARGE_THRESHOLD)
        n_hard = sum(c for s, c in size_counts.items() if s >= SMALL_LARGE_THRESHOLD)
        lines.append(f"  Total: {n_total} [{n_easy}E+{n_hard}H]")
        return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# SVG problem overview
# ─────────────────────────────────────────────────────────────────────────────

def generate_problem_svg(
    triplets: List[Tuple[str, int, str]],
    output_path: str = "problem_overview.svg",
) -> str:
    """Create an SVG listing every problem with its easy/hard tag and Oracle value."""
    for t in triplets:
        find_optimal_shots(*t)

    easy = sorted([t for t in triplets if classify_problem(t) == "easy"],
                  key=lambda x: (x[0], x[1], x[2]))
    hard = sorted([t for t in triplets if classify_problem(t) == "hard"],
                  key=lambda x: (x[0], x[1], x[2]))

    row_h, col_w, header_h, padding = 22, 420, 80, 16
    n_rows = max(len(easy), len(hard))
    svg_w  = 2 * col_w + 3 * padding
    svg_h  = header_h + n_rows * row_h + 2 * padding

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="monospace" font-size="13">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="#fafafa"/>',
        f'<text x="{svg_w//2}" y="20" font-size="14" fill="#37474f" text-anchor="middle">'
        f'{get_config_label(for_svg=True)}</text>',
        f'<line x1="{padding}" y1="28" x2="{svg_w - padding}" y2="28" stroke="#ccc"/>',
    ]

    if SPLIT_METRIC == SplitMetric.SIZE:
        easy_label = f'EASY (size &lt; {SMALL_LARGE_THRESHOLD})'
        hard_label = f'HARD (size &gt;= {SMALL_LARGE_THRESHOLD})'
    else:
        easy_label = f'EASY (oracle &lt;= {ORACLE_EASY_THRESHOLD})'
        hard_label = f'HARD (oracle &gt; {ORACLE_EASY_THRESHOLD})'

    lines += [
        f'<text x="{padding}" y="50" font-size="16" font-weight="bold" fill="#2e7d32">'
        f'{easy_label}  [{len(easy)} problems]</text>',
        f'<text x="{col_w + 2*padding}" y="50" font-size="16" font-weight="bold" fill="#c62828">'
        f'{hard_label}  [{len(hard)} problems]</text>',
        f'<line x1="0" y1="60" x2="{svg_w}" y2="60" stroke="#bbb"/>',
    ]

    def _row(t, idx, x_off):
        alg, sz, be = t
        oracle = ACTUAL_ORACLE_CACHE.get((alg, sz, be), "?")
        y = header_h + idx * row_h
        cat = classify_problem(t)
        clr = "#2e7d32" if cat == "easy" else "#c62828"
        tag = "EASY" if cat == "easy" else "HARD"
        return (f'<text x="{x_off}" y="{y}" fill="#333">'
                f'{alg:>8s}  q={sz:<3d}  {be:<18s}  oracle={str(oracle):>6s}'
                f'  <tspan fill="{clr}" font-weight="bold">[{tag}]</tspan></text>')

    for i, t in enumerate(easy):
        lines.append(_row(t, i, padding))
    for i, t in enumerate(hard):
        lines.append(_row(t, i, col_w + 2 * padding))

    lines.append("</svg>")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"SVG problem overview saved -> {output_path}")
    return output_path

# ─────────────────────────────────────────────────────────────────────────────
# Quantum RL Environment  — with incremental cumulative caching
# ─────────────────────────────────────────────────────────────────────────────

class IterativeQuantumEnv(gym.Env):
    """Gym environment for iterative shot allocation.
    Maintains an incremental cumulative counter and caches entropy/variance
    to avoid recomputing from scratch every step."""

    def __init__(
        self,
        triplets: List[Tuple[str, int, str]],
        max_shots: int = MAX_SHOTS,
        step_size: int = 50,
        agent_type: AgentType = AgentType.GENERIC,
        reward_type: RewardType = RewardType.MAE,
        label: str = "env",
    ):
        super().__init__()
        self.max_shots = max_shots
        self.step_size = step_size
        self.agent_type = agent_type
        self.reward_type = reward_type
        self.label = label

        self.active_triplets = triplets
        self.alg_map, self.backend_map = get_global_mappings()

        for t in self.active_triplets:
            find_optimal_shots(*t)

        self.easy_triplets = [t for t in self.active_triplets if classify_problem(t) == "easy"]
        self.hard_triplets = [t for t in self.active_triplets if classify_problem(t) == "hard"]


        # ── Oracle-stratified bins (ENHANCED: enabled for BOTH split metrics)
        # The SIZE-split HARD pool mixes oracle≈150 and oracle≈17.8k problems,
        # so plain random sampling under-trains the high-oracle region (where
        # the generic agent was observed to undershoot).  Stratified sampling
        # across oracle-range bins balances the stopping points the agent sees.
        self._oracle_bins: Dict[str, Dict[int, List[Tuple[str, int, str]]]] = {}
        if label == "train":
            bin_edges = [0, 5000, 8000, 11000, 14000, 17000, MAX_SHOTS + 1]
            for pool_name, pool in [("easy", self.easy_triplets), ("hard", self.hard_triplets)]:
                bins: Dict[int, List[Tuple[str, int, str]]] = {}
                for t in pool:
                    oracle_val = find_optimal_shots(*t)
                    for j in range(len(bin_edges) - 1):
                        if bin_edges[j] <= oracle_val < bin_edges[j + 1]:
                            bins.setdefault(j, []).append(t)
                            break
                # Store only non-empty bins
                self._oracle_bins[pool_name] = {k: v for k, v in bins.items() if v}

        # Show effective training pool based on agent_type for "train" envs
        if label == "train":
            if agent_type == AgentType.EASY:
                eff = len(self.easy_triplets)
                desc = f"trains on {eff} EASY of {len(self.active_triplets)} pool"
            elif agent_type == AgentType.HARD:
                eff = len(self.hard_triplets)
                desc = f"trains on {eff} HARD of {len(self.active_triplets)} pool"
            elif agent_type == AgentType.UNBALANCED:
                desc = (f"trains 20/80 mix from {len(self.easy_triplets)}E + "
                        f"{len(self.hard_triplets)}H pool")
            else:
                desc = (f"trains 50/50 mix from {len(self.easy_triplets)}E + "
                        f"{len(self.hard_triplets)}H pool")
            # Show oracle-stratified bin distribution
            if self._oracle_bins:
                for pk, pk_bins in self._oracle_bins.items():
                    bin_str = ", ".join(f"bin{k}:{len(v)}" for k, v in sorted(pk_bins.items()))
                    desc += f"\n         Oracle bins ({pk}): {bin_str}"
            print(f"[{label.upper()}] {desc}")
        else:
            print(f"[{label.upper()}] {len(self.active_triplets)} problems "
                  f"(easy={len(self.easy_triplets)}, hard={len(self.hard_triplets)})")

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_SIZE,), dtype=np.float32)
        self.outcome_history: List[Dict] = []
        self.current_triplet = self.active_triplets[0]
        self.optimal_shots = 0
        self.current_shots = 0

        # ── Incremental caches ──────────────────────────────────────────
        self._cumulative: Counter = Counter()        # running total of all outcomes
        self._cached_entropy: float = 0.5            # last computed entropy
        self._cached_variance: float = 0.5           # last computed variance
        self._cache_valid: bool = False               # True when entropy/variance are up-to-date
        self._fetch_failures: int = 0                 # failed get_outcomes calls this episode
        self._streak: int = 0                         # v2: consecutive batches with rate_long <= oracle eps
        self._streak_seen: int = 0                    # v2: history length at last streak update
        self._c_samples: List[float] = []             # v3: √n-law coefficient samples (extrapolation feature)

    def _stratified_sample(self, pool_name: str) -> Tuple[str, int, str]:
        """Pick a problem from the named pool using oracle-stratified sampling.
        First pick a random bin (uniform), then a random problem within it.
        Falls back to plain random.choice if no bins are available."""
        bins = self._oracle_bins.get(pool_name, {})
        if bins:
            bin_key = random.choice(list(bins.keys()))
            return random.choice(bins[bin_key])
        pool = self.hard_triplets if pool_name == "hard" else self.easy_triplets
        return random.choice(pool or self.active_triplets)

    def reset(self) -> np.ndarray:
        if self.label == "train":
            use_stratified = bool(self._oracle_bins)

            if self.agent_type == AgentType.EASY:
                if self.easy_triplets:
                    self.current_triplet = (self._stratified_sample("easy")
                                            if use_stratified else
                                            random.choice(self.easy_triplets))
                else:
                    self.current_triplet = random.choice(self.active_triplets)
            elif self.agent_type == AgentType.HARD:
                if self.hard_triplets:
                    self.current_triplet = (self._stratified_sample("hard")
                                            if use_stratified else
                                            random.choice(self.hard_triplets))
                else:
                    self.current_triplet = random.choice(self.active_triplets)
            elif self.agent_type == AgentType.UNBALANCED:
                if random.random() < 0.2 and self.easy_triplets:
                    self.current_triplet = (self._stratified_sample("easy")
                                            if use_stratified else
                                            random.choice(self.easy_triplets))
                elif self.hard_triplets:
                    self.current_triplet = (self._stratified_sample("hard")
                                            if use_stratified else
                                            random.choice(self.hard_triplets))
                else:
                    self.current_triplet = random.choice(self.active_triplets)
            else:  # GENERIC — balanced 50/50
                if random.random() < 0.5 and self.easy_triplets:
                    self.current_triplet = (self._stratified_sample("easy")
                                            if use_stratified else
                                            random.choice(self.easy_triplets))
                elif self.hard_triplets:
                    self.current_triplet = (self._stratified_sample("hard")
                                            if use_stratified else
                                            random.choice(self.hard_triplets))
                else:
                    self.current_triplet = random.choice(self.active_triplets)
        else:
            self.current_triplet = random.choice(self.active_triplets)

        self.optimal_shots = find_optimal_shots(*self.current_triplet)
        self.current_shots = 0
        self.outcome_history = []
        # Reset incremental caches
        self._cumulative = Counter()
        self._cached_entropy = 0.5
        self._cached_variance = 0.5
        self._cache_valid = False
        self._fetch_failures = 0
        self._streak = 0
        self._streak_seen = 0
        self._c_samples = []
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action == 1:  # STOP
            return self._terminate()

        self.current_shots += self.step_size
        try:
            alg, size, backend = self.current_triplet
            batch = get_outcomes(alg, size, backend,
                                 shots=self.step_size, strategy="random", exact=True)
            self.outcome_history.append(batch)
            # Incrementally update cumulative counter — no need to rebuild
            self._cumulative.update(batch)
            self._cache_valid = False  # invalidate entropy/variance cache
        except Exception:
            # Shots are still counted (guarantees episode termination), but the
            # failure is recorded instead of silently passing
            self._fetch_failures += 1
            if self._fetch_failures == 1:
                print(f"Warning: get_outcomes failed for {self.current_triplet} "
                      f"at {self.current_shots} shots (shots counted anyway)")

        if self.current_shots >= self.max_shots:
            return self._terminate(forced=True)   # v2: cap-forced stop is penalised

        return self._get_state(), STEP_PENALTY, False, {}

    def _terminate(self, forced: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
        error = self.current_shots - self.optimal_shots
        mae = abs(error)

        if self.reward_type == RewardType.PRECISION:
            # Difficulty keyed to the ORACLE value, not circuit size: size
            # correlates poorly with oracle shots (e.g. dj_14 has oracle=150)
            oracle_hard = self.optimal_shots > ORACLE_EASY_THRESHOLD
            if error < 0:
                scale = 1.0 + 0.5 * self.optimal_shots / self.max_shots
                base = -2.5 if oracle_hard else -1.85
                final_reward = (base - (mae / max(self.optimal_shots, 1))) * scale
            else:
                # Full reward on a RELATIVE plateau [0, margin] — a slight
                # overshoot is the unambiguous optimum and the undershoot
                # cliff sits away from the reward peak.  The relative margin
                # (v2) stops tiny-oracle problems from getting a free pass:
                # the old absolute 200 made oracle=50 "perfect" up to 250.
                margin = max(OVERSHOOT_MARGIN_MIN,
                             OVERSHOOT_MARGIN_FRAC * self.optimal_shots)
                over = max(0.0, error - margin)
                final_reward = np.exp(-OVERSHOOT_REL_DECAY * over / max(self.optimal_shots, 1))
                # v2: riding the budget cap must not be a comfortable default.
                # If the cap FORCED termination beyond the margin, cut the
                # reward (genuinely near-budget oracles are unaffected since
                # their error stays within the margin).
                if forced and over > 0:
                    final_reward *= FORCED_CAP_FACTOR
        elif self.reward_type in (RewardType.ASYMMETRIC, RewardType.HARD_ASYMMETRIC):
            if error < 0:
                scale = 1.0 + 0.5 * self.optimal_shots / self.max_shots
                base = -1.85
                # HARD_ASYMMETRIC: extra undershoot penalty for hard problems
                if (self.reward_type == RewardType.HARD_ASYMMETRIC
                        and classify_problem(self.current_triplet) == "hard"):
                    base = -2.5
                    scale *= 1.3
                final_reward = (base - (mae / max(self.optimal_shots, 1))) * scale
            else:
                final_reward = np.exp(-0.00025 * error)
        else:
            if error < 0:
                final_reward = -1.0 - (mae / max(self.optimal_shots, 1))
            else:
                final_reward = np.exp(-0.003 * mae)

        info = {
            "shots_used": self.current_shots,
            "optimal_shots": self.optimal_shots,
            "error": error,
            "mae": mae,
            "error_pct": mae / self.max_shots * 100,
            "final_reward": final_reward,
            "triplet": self.current_triplet,
            "difficulty": classify_problem(self.current_triplet),
            "forced_stop": forced,
        }
        return self._get_state(), final_reward, True, info

    # ── state features (with caching) ───────────────────────────────────

    def _compute_distribution_entropy(self, outcomes: Dict[str, int]) -> float:
        total = sum(outcomes.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in outcomes.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        # Normalise by the max possible entropy (= n qubits, in bits) so the
        # feature does not saturate at 1.0 for every large circuit (the old
        # /4.0 pegged at 1.0 for any circuit wider than 4 effective bits)
        max_bits = max(self.current_triplet[1], 1)
        return min(entropy / max_bits, 1.0)

    def _compute_distribution_variance(self, outcomes: Dict[str, int]) -> float:
        """Concentration = collision probability Σp² — a scale-free measure of
        how peaked the distribution is: ≈1 for a single dominant outcome, →0
        for a flat distribution, independent of the outcome-space size.
        (The old var(p)·10 vanished for large outcome spaces because all
        probabilities become tiny.)"""
        if not outcomes:
            return 0.0
        total = sum(outcomes.values())
        if total == 0:
            return 0.0
        return min(sum((c / total) ** 2 for c in outcomes.values()), 1.0)

    def _update_cached_features(self):
        """Recompute entropy & variance from the incremental cumulative counter
        only when the cache has been invalidated."""
        if self._cache_valid:
            return
        if self._cumulative:
            self._cached_entropy = self._compute_distribution_entropy(self._cumulative)
            self._cached_variance = self._compute_distribution_variance(self._cumulative)
        else:
            self._cached_entropy = 0.5
            self._cached_variance = 0.5
        self._cache_valid = True

    def _compute_rate_of_change(self, lag_batches: int = 1) -> float:
        """TVD between successive CUMULATIVE distributions, P̂_n vs P̂_{n-lag}
        — the paper's Def. 3.3 diminishing-returns proxy, i.e. the same signal
        family the Oracle's optimality is defined on.  (The old version
        compared the cumulative against the LAST 50-SHOT BATCH ALONE, which
        stays ≈1 forever on large outcome spaces and never signals
        convergence on exactly the hard circuits.)"""
        if len(self.outcome_history) < lag_batches + 1:
            return 1.0

        removed: Counter = Counter()
        for b in self.outcome_history[-lag_batches:]:
            removed.update(b)
        prev_cumulative = self._cumulative - removed
        prev_total = sum(prev_cumulative.values())
        cum_total = sum(self._cumulative.values())
        if prev_total == 0 or cum_total == 0:
            return 1.0

        # prev keys ⊆ cumulative keys, so iterating the cumulative covers both
        tvd = 0.0
        for k, v in self._cumulative.items():
            tvd += abs(v / cum_total - prev_cumulative.get(k, 0) / prev_total)
        return 0.5 * tvd

    def _get_state(self) -> np.ndarray:
        alg, size, backend = self.current_triplet
        n_algs  = max(len(self.alg_map), 2)
        n_backs = max(len(self.backend_map), 2)
        alg_norm     = self.alg_map.get(alg, 0) / (n_algs - 1)
        size_norm    = size / 15.0
        backend_norm = self.backend_map.get(backend, 0) / (n_backs - 1)
        shots_norm   = self.current_shots / self.max_shots

        # Use cached entropy/variance (recomputed only when invalidated)
        self._update_cached_features()
        entropy  = self._cached_entropy
        variance = self._cached_variance

        rate_short = self._compute_rate_of_change(lag_batches=1)
        rate_long  = self._compute_rate_of_change(lag_batches=RATE_LONG_LAG)

        # v2: convergence STREAK — consecutive batches with the long-lag
        # cumulative TVD below the Oracle's eps: a sustained-stability signal
        # analogous to the IE framework's k-consecutive stability criterion
        # and a near-sufficient statistic for the Oracle's stopping decision.
        # Updated only when a new batch has arrived, so the repeated
        # _get_state calls at reset/terminate cannot corrupt the count.
        n_hist = len(self.outcome_history)
        if n_hist == 0:
            self._streak = 0
        elif n_hist != self._streak_seen:
            self._streak = self._streak + 1 if rate_long <= ACTUAL_ORACLE_EPS else 0
            # v3: collect a √n-law coefficient sample for the extrapolation
            # feature. Under IID sampling TVD(P̂_n, P̂_{n-l}) ≈ c·√l/n while
            # the Oracle's signal TVD(P̂_n, P̂_B) ≈ c/√n, so a valid
            # rate_long measurement gives c ≈ rate_long·n/√l.
            if n_hist > RATE_LONG_LAG and rate_long < 1.0:
                lag_shots = RATE_LONG_LAG * self.step_size
                self._c_samples.append(
                    rate_long * self.current_shots / (lag_shots ** 0.5))
        self._streak_seen = n_hist
        streak_norm = min(self._streak / STREAK_CAP, 1.0)

        # v3: oracle-extrapolation feature — predicted stopping point as a
        # fraction of the budget. The Oracle's signal is TVD(P̂_n, P̂_B) with
        # P̂_n a PREFIX of P̂_B, so it follows c·√(1/n − 1/B) rather than
        # c/√n; solving c·√(1/n̂ − 1/B) = ε gives the finite-budget-corrected
        # n̂* = 1/(ε²/c² + 1/B). Median over a recent window: robust to noise
        # yet free of the biased early-transient samples. Feature stays at 1.0
        # ("stop point far away") until enough samples exist, matching the
        # rate features' conservative default.
        if len(self._c_samples) >= EXTRAP_MIN_SAMPLES:
            c_med = float(np.median(self._c_samples[-EXTRAP_WINDOW:]))
            naive = (c_med / ACTUAL_ORACLE_EPS) ** 2   # (c/ε)², valid for n ≪ B
            n_hat = 1.0 / (1.0 / max(naive, 1e-9) + 1.0 / self.max_shots)
            extrap = min(n_hat / self.max_shots, 1.0)
        else:
            extrap = 1.0

        # size·1500 keeps the feature unsaturated across the real decision
        # region (size·500 pegged at 1.0 from 7000 shots for 14-qubit circuits
        # whose oracles sit at 13k–17.8k)
        progress = min(self.current_shots / (size * 1500), 1.0)

        return np.array(
            [alg_norm, size_norm, backend_norm, shots_norm,
             entropy, variance, rate_short, rate_long, streak_norm, progress,
             extrap],
            dtype=np.float32,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DQN Network & Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Agent:
    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = GAMMA,
        target_sync_steps: int = TARGET_SYNC_STEPS,
    ):
        self.q_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.update_target()
        self.target_net.eval()  # never trained directly — keep its dropout off
        self.memory: deque = deque(maxlen=200_000)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.997   # applied once per EPISODE (≈0.05 by ep 1000; v2 trains 2000 eps)
        self.epsilon_min = 0.05
        self.action_size = action_size
        self.target_sync_steps = target_sync_steps
        self.train_steps = 0         # replay() calls — drives target syncing

    def act(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        self.q_net.eval()  # disable dropout — the greedy action must be deterministic
        with torch.no_grad():
            q = self.q_net(state_t)
        return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        with torch.no_grad():
            if DOUBLE_DQN:
                # v2 Double DQN: the ONLINE net selects a', the TARGET net
                # evaluates it — removes the max-operator overestimation that
                # systematically inflates Q(CONTINUE) and biases late stops
                self.q_net.eval()
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        self.q_net.train()  # re-enable dropout for the learning forward pass
        cur_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # v2: Huber/smooth-L1 keeps the large negative undershoot-cliff targets
        # from dominating the gradients (smoother Q near the stopping boundary)
        loss = F.smooth_l1_loss(cur_q, target_q) if HUBER_LOSS else F.mse_loss(cur_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Step-based target sync (was every 500 EPISODES — a single sync in a
        # 1000-episode run, so the Bellman targets bootstrapped off a frozen
        # random network for the whole first half of training)
        self.train_steps += 1
        if self.train_steps % self.target_sync_steps == 0:
            self.update_target()

    def decay_epsilon(self):
        """Called once per episode by the training loop (was per replay step,
        which ended exploration after ~150 episodes and coupled the schedule
        to episode length)."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_weights(self) -> dict:
        return copy.deepcopy(self.q_net.state_dict())

    def load_weights(self, state_dict: dict):
        self.q_net.load_state_dict(state_dict)
        self.update_target()

# ─────────────────────────────────────────────────────────────────────────────
# Greedy episode runner with stability voting (v2)
# ─────────────────────────────────────────────────────────────────────────────

def run_greedy_episode(
    agent: Agent,
    env: IterativeQuantumEnv,
    triplet: Tuple[str, int, str],
    stop_votes: int = EVAL_STOP_VOTES,
) -> Tuple[Dict, float]:
    """Run one greedy (evaluate=True) episode on *triplet* and return
    (info, episode_reward).

    Stability voting: STOP is executed only after `stop_votes` CONSECUTIVE
    greedy STOP decisions (a CONTINUE vote resets the count), so a single
    noisy state can no longer trigger a premature stop.  The cost is at most
    (stop_votes - 1) extra batches of upward bias; the gain is much lower
    variance of the stopping point.  Set EVAL_STOP_VOTES = 1 to disable.

    Used by EVERY evaluation/validation loop so that checkpoint selection and
    final evaluation see exactly the same policy behaviour.
    """
    env.current_triplet = triplet
    env.optimal_shots = find_optimal_shots(*triplet)
    env.current_shots = 0
    env.outcome_history = []
    env._cumulative = Counter()
    env._cache_valid = False
    env._fetch_failures = 0
    env._streak = 0
    env._streak_seen = 0
    env._c_samples = []
    state = env._get_state()

    done, info = False, {}
    ep_reward, votes = 0.0, 0
    while not done:
        action = agent.act(state, evaluate=True)
        if action == 1:
            votes += 1
            if votes < stop_votes:
                action = 0   # not enough consecutive confirmations yet
        else:
            votes = 0
        state, reward, done, info = env.step(action)
        ep_reward += reward
    return info, ep_reward

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Validator  (uses paper problems filtered by AgentType for checkpointing)
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotValidator:
    """Validates agent on a fixed validation subset; returns MAE.
    Iterates deterministically through every triplet (not random sampling)
    so that the validation MAE is reproducible across episodes."""

    def __init__(self, validation_env: IterativeQuantumEnv):
        self.env = validation_env
        self.history: List[Dict] = []

    def validate(self, agent: Agent, episode: int, runs: int = VALIDATION_RUNS) -> float:
        """MAE averaged over *runs* passes of the validation set.  Each pass
        iterates every triplet deterministically, but trajectories differ
        because get_outcomes samples randomly — averaging de-noises the
        best-model selection signal."""
        total_mae, total_reward = 0.0, 0.0
        runs = max(runs, 1)
        n = len(self.env.active_triplets) * runs
        orig_eps = agent.epsilon
        agent.epsilon = 0.0

        for _ in range(runs):
            for triplet in self.env.active_triplets:
                # Deterministic triplet order; reset + stability voting are
                # handled inside run_greedy_episode (same policy as final eval)
                info, ep_reward = run_greedy_episode(agent, self.env, triplet)
                total_mae += abs(info.get("error", 0))
                total_reward += ep_reward

        agent.epsilon = orig_eps

        avg_mae    = total_mae / max(n, 1)
        avg_reward = total_reward / max(n, 1)
        error_pct  = avg_mae / self.env.max_shots * 100

        self.history.append({
            "episode": episode, "mae": avg_mae,
            "error_pct": error_pct, "reward": avg_reward,
        })
        return avg_mae

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(
    agent_type: AgentType = AgentType.GENERIC,
    reward_type: RewardType = RewardType.MAE,
    num_episodes: int = 3000,
    target_sync_steps: int = TARGET_SYNC_STEPS,
    snapshot_interval: int = 200,
    checkpoint_start: int = 2000,
    checkpoint_interval: int = 100,
    patience: int = 5,
    seed: int = 42,
) -> Tuple[Agent, IterativeQuantumEnv, List[float], List[Dict], int]:
    """
    Train a DQN agent on the 80% training split of the actual oracle dataset.
    agent_type controls training distribution:
      EASY       → 100% easy problems
      HARD       → 100% hard problems
      GENERIC    → 50/50 balanced mix
      UNBALANCED → 20% easy, 80% hard
    Validation for checkpointing uses the 20% held-out split from the
    actual-oracle dataset (same easy/hard distribution as AGENT_TYPE).
    Returns (agent, env, rewards_history, validation_history, best_episode).
    """
    # Actual oracle cache is loaded automatically by build_training_triplets()
    train_triplets = build_training_triplets(agent_type=agent_type)
    val_triplets   = build_validation_set(agent_type=agent_type, seed=seed)

    env     = IterativeQuantumEnv(train_triplets, agent_type=agent_type, reward_type=reward_type, label="train")
    val_env = IterativeQuantumEnv(val_triplets,   reward_type=reward_type, label="validation")

    agent     = Agent(target_sync_steps=target_sync_steps)
    validator = SnapshotValidator(val_env)

    rewards_history: List[float] = []
    best_mae: float = float("inf")
    best_weights: Optional[dict] = None
    best_episode: int = -1
    no_improve_count: int = 0
    stopped_early = False

    # ── Describe effective training distribution ───────────────────────
    n_easy_pool = len(env.easy_triplets)
    n_hard_pool = len(env.hard_triplets)
    if agent_type == AgentType.EASY:
        train_desc = f"{n_easy_pool} EASY problems (100% easy)"
    elif agent_type == AgentType.HARD:
        train_desc = f"{n_hard_pool} HARD problems (100% hard)"
    elif agent_type == AgentType.UNBALANCED:
        train_desc = f"20% easy / 80% hard from {n_easy_pool}E + {n_hard_pool}H pool"
    else:
        train_desc = f"50/50 balanced from {n_easy_pool}E + {n_hard_pool}H pool"

    print(f"\n{'='*60}")
    print(f" Agent Type  : {agent_type.value.upper()}")
    print(f" Reward Type : {reward_type.value.upper()}")
    print(f" Split Metric: {SPLIT_METRIC.value.upper()}")
    print(f" Train       : {train_desc}")
    print(f" Validation  : {len(val_triplets)} problems from 20%% held-out split (actual oracle)")
    print(f" Test        : {len(build_paper_triplets())} paper problems (ALL 36, evaluated after training)")
    print(f" Episodes    : {num_episodes}  |  Patience : {patience}")
    print(f"{'='*60}\n")

    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay(batch_size=64)
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        agent.decay_epsilon()   # per-episode exploration decay (target sync is step-based, inside replay)

        # ── Periodic snapshot validation ────────────────────────────────
        # Skip snapshot if checkpoint will also fire this episode (avoids
        # duplicate validator.validate() calls and double history entries)
        checkpoint_fires = (episode >= checkpoint_start
                            and episode % checkpoint_interval == 0)
        if episode % snapshot_interval == 0 and not checkpoint_fires:
            current_mae = validator.validate(agent, episode)
            error_pct = current_mae / env.max_shots * 100

            if len(validator.history) > 1:
                prev = validator.history[-2]
                delta = current_mae - prev["mae"]
                icon = "↓" if delta < 0 else "↑"
                print(f"\n[Snapshot Ep {episode}] Val MAE: {current_mae:.1f} "
                      f"({error_pct:.2f}%)  Δ: {delta:+.1f}  {icon}")
            else:
                print(f"\n[Snapshot Ep {episode}] Val MAE: {current_mae:.1f} "
                      f"({error_pct:.2f}%)  (Baseline)")

        # ── Best-model checkpointing ────────────────────────────────────
        if episode >= checkpoint_start and episode % checkpoint_interval == 0:
            ckpt_mae = validator.validate(agent, episode)

            if ckpt_mae < best_mae:
                best_mae = ckpt_mae
                best_weights = agent.get_weights()
                best_episode = episode
                no_improve_count = 0
                print(f"  ★ New best @ ep {episode}: MAE={best_mae:.1f} "
                      f"({best_mae/env.max_shots*100:.2f}%)")
            else:
                no_improve_count += 1
                print(f"  ✗ No improvement ({no_improve_count}/{patience}) "
                      f"current={ckpt_mae:.1f} vs best={best_mae:.1f}")

            if no_improve_count >= patience:
                print(f"\n>> Early stopping at episode {episode} (patience={patience})")
                stopped_early = True
                break

    if best_weights is not None:
        agent.load_weights(best_weights)
        print(f"\n>> Restored best model weights (MAE={best_mae:.1f})")
    else:
        print("\n>> No checkpoint taken (training ended before checkpoint_start)")

    print(f"Training {'ended early at ep ' + str(episode) if stopped_early else 'completed all ' + str(num_episodes) + ' episodes'}.")

    return agent, env, rewards_history, validator.history, best_episode

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on the 36 paper problems
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent: Agent, agent_type: AgentType = AgentType.GENERIC, seed: int = 42) -> List[Dict]:
    """Evaluate the trained agent on ALL 36 paper problems (18 easy + 18 hard).
    The full test set is always used regardless of agent type so that results
    are directly comparable across training configurations.
    The agent has never seen these traces during training."""
    eval_triplets = build_eval_set(agent_type, seed=seed)
    n_easy = sum(1 for t in eval_triplets if classify_problem(t) == "easy")
    n_hard = sum(1 for t in eval_triplets if classify_problem(t) == "hard")

    env_eval = IterativeQuantumEnv(eval_triplets, label=f"eval ({agent_type.value})")

    results: List[Dict] = []
    for triplet in tqdm(env_eval.active_triplets, desc=f"Evaluating ({agent_type.value}, {len(eval_triplets)} problems)"):
        info, _ = run_greedy_episode(agent, env_eval, triplet)
        results.append(info)

    # Summary
    maes = [r["mae"] for r in results]
    easy_maes = [r["mae"] for r in results if r["difficulty"] == "easy"]
    hard_maes = [r["mae"] for r in results if r["difficulty"] == "hard"]
    avg_mae = np.mean(maes)
    pct = avg_mae / env_eval.max_shots * 100

    train_triplets = build_training_triplets(agent_type=agent_type)

    print(f"\n{'='*55}")
    print(f"  TEST EVALUATION — {agent_type.value.upper()} agent")
    print(f"  {len(eval_triplets)} problems  ({n_easy} easy + {n_hard} hard)")
    print(f"  {'─'*50}")
    print(f"  Overall MAE   : {avg_mae:.1f} shots  ({pct:.2f}%)")
    if easy_maes:
        print(f"  Easy MAE      : {np.mean(easy_maes):.1f} shots  ({len(easy_maes)} problems)")
    if hard_maes:
        print(f"  Hard MAE      : {np.mean(hard_maes):.1f} shots  ({len(hard_maes)} problems)")
    print(f"  {'─'*50}")
    print(format_training_split_stats(train_triplets, indent="  "))
    print(f"{'='*55}")
    return results


def evaluate_on_triplets(agent: Agent, triplets: List[Tuple[str, int, str]],
                         desc: str = "Evaluating") -> List[Dict]:
    """Run the trained agent greedily on *triplets* and return per-problem info dicts.
    This is a lightweight evaluation loop (no printed summary) suitable for
    getting scatter-plot data on the training set or any arbitrary set."""
    env_tmp = IterativeQuantumEnv(triplets, label=desc)
    results: List[Dict] = []
    for triplet in tqdm(env_tmp.active_triplets, desc=desc):
        info, _ = run_greedy_episode(agent, env_tmp, triplet)
        results.append(info)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Multi-run evaluation (N runs on eval set for mean ± std per problem)
# ─────────────────────────────────────────────────────────────────────────────

def multi_run_evaluation(
    agent: Agent,
    triplets: List[Tuple[str, int, str]],
    n_runs: int = 10,
    desc: str = "Multi-run eval",
) -> Dict[Tuple[str, int, str], Dict]:
    """Run the agent *n_runs* times on every problem in *triplets*.

    Because get_outcomes(strategy="random") samples differently each call,
    each run produces a different trajectory and potentially a different
    STOP decision, even though the policy is deterministic (argmax Q).

    Returns a dict keyed by (alg, size, backend) with per-problem stats:
        shots_mean, shots_std, mae_mean, mae_std, optimal_shots, difficulty,
        all_shots (list of raw agent shots from each run)
    """
    # Collect per-problem shots across runs
    per_problem: Dict[Tuple, List[int]] = {tuple(t): [] for t in triplets}
    optimal_cache: Dict[Tuple, int] = {}
    diff_cache: Dict[Tuple, str] = {}

    for run_idx in range(n_runs):
        run_results = evaluate_on_triplets(
            agent, triplets, desc=f"{desc} [{run_idx+1}/{n_runs}]"
        )
        for r in run_results:
            key = tuple(r["triplet"])
            per_problem[key].append(r["shots_used"])
            optimal_cache[key] = r["optimal_shots"]
            diff_cache[key] = r["difficulty"]

    # Aggregate
    summary: Dict[Tuple, Dict] = {}
    for key in triplets:
        key_t = tuple(key)
        shots_arr = np.array(per_problem[key_t], dtype=float)
        opt = optimal_cache[key_t]
        mae_arr = np.abs(shots_arr - opt)
        summary[key_t] = {
            "algorithm": key_t[0],
            "num_qubits": key_t[1],
            "backend": key_t[2],
            "optimal_shots": opt,
            "difficulty": diff_cache[key_t],
            "shots_mean": float(np.mean(shots_arr)),
            "shots_std": float(np.std(shots_arr)),
            "mae_mean": float(np.mean(mae_arr)),
            "mae_std": float(np.std(mae_arr)),
            "all_shots": [int(s) for s in shots_arr],
        }

    # Print summary
    all_mae_means = [v["mae_mean"] for v in summary.values()]
    all_mae_stds  = [v["mae_std"]  for v in summary.values()]
    easy_mae = [v["mae_mean"] for v in summary.values() if v["difficulty"] == "easy"]
    hard_mae = [v["mae_mean"] for v in summary.values() if v["difficulty"] == "hard"]

    print(f"\n{'='*60}")
    print(f"  MULTI-RUN EVALUATION — {n_runs} runs × {len(triplets)} problems")
    print(f"  {'─'*55}")
    print(f"  Overall MAE  : {np.mean(all_mae_means):>8.1f} ± {np.mean(all_mae_stds):.1f} shots")
    print(f"  Error %      : {np.mean(all_mae_means)/20000*100:>8.2f}%")
    if easy_mae:
        print(f"  Easy MAE     : {np.mean(easy_mae):>8.1f} shots  ({len(easy_mae)} probs)")
    if hard_mae:
        print(f"  Hard MAE     : {np.mean(hard_mae):>8.1f} shots  ({len(hard_mae)} probs)")
    print(f"{'='*60}")

    return summary


def generate_multirun_svg(
    summary: Dict[Tuple, Dict],
    output_path: str = "multirun_eval.svg",
    n_runs: int = 10,
    agent_type: AgentType = AgentType.GENERIC,
) -> str:
    """Scatter plot of agent mean shots vs oracle shots with std-dev error bars,
    plus a per-problem table. Suitable for paper publication."""

    items = sorted(summary.values(), key=lambda v: v["optimal_shots"])
    easy_items = [v for v in items if v["difficulty"] == "easy"]
    hard_items = [v for v in items if v["difficulty"] == "hard"]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # ── Left: scatter with error bars ──────────────────────────────────
    ax = axes[0]
    for label, group, color, marker in [
        ("Easy", easy_items, "#2196F3", "o"),
        ("Hard", hard_items, "#F44336", "s"),
    ]:
        if not group:
            continue
        oracle = [v["optimal_shots"] for v in group]
        means  = [v["shots_mean"]    for v in group]
        stds   = [v["shots_std"]     for v in group]
        ax.errorbar(oracle, means, yerr=stds, fmt=marker, color=color,
                    ecolor=color, alpha=0.7, capsize=3, markersize=6,
                    label=f"{label} ({len(group)})")

    lim_max = max(v["optimal_shots"] for v in items)
    lim_max = max(lim_max, max(v["shots_mean"] + v["shots_std"] for v in items))
    lim_max *= 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.4, label="Ideal")
    ax.set_xlabel("Oracle Optimal Shots", fontsize=12)
    ax.set_ylabel("Agent Shots (mean ± std)", fontsize=12)
    ax.set_title(f"Multi-Run Eval ({n_runs} runs) — {agent_type.value.upper()}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal")

    # ── Right: summary text ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    all_mae_means = [v["mae_mean"] for v in items]
    all_mae_stds  = [v["mae_std"]  for v in items]
    easy_mae_m = [v["mae_mean"] for v in easy_items]
    hard_mae_m = [v["mae_mean"] for v in hard_items]

    lines = [
        f"MULTI-RUN EVALUATION SUMMARY",
        f"{'─'*40}",
        f"Runs         : {n_runs}",
        f"Problems     : {len(items)} ({len(easy_items)}E + {len(hard_items)}H)",
        f"Agent Type   : {agent_type.value.upper()}",
        f"{'─'*40}",
        f"Overall MAE  : {np.mean(all_mae_means):.1f} ± {np.mean(all_mae_stds):.1f}",
        f"Error %      : {np.mean(all_mae_means)/20000*100:.2f}%",
        f"{'─'*40}",
    ]
    if easy_mae_m:
        lines.append(f"Easy MAE     : {np.mean(easy_mae_m):.1f} ({len(easy_mae_m)} probs)")
    if hard_mae_m:
        lines.append(f"Hard MAE     : {np.mean(hard_mae_m):.1f} ({len(hard_mae_m)} probs)")

    lines += [
        f"{'─'*40}",
        f"",
        f"{'Problem':<30} {'Oracle':>7} {'Mean':>7} {'Std':>7} {'MAE':>7}",
        f"{'─'*62}",
    ]

    for v in items:
        tag = "E" if v["difficulty"] == "easy" else "H"
        name = f"{v['algorithm']}_{v['num_qubits']}q_{v['backend'][:8]}"
        lines.append(
            f"[{tag}] {name:<27} {v['optimal_shots']:>7d} "
            f"{v['shots_mean']:>7.0f} {v['shots_std']:>7.1f} {v['mae_mean']:>7.1f}"
        )

    txt = "\n".join(lines)
    ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, fontsize=7.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved multi-run evaluation SVG → {output_path}")
    return output_path


def save_multirun_csv(
    summary: Dict[Tuple, Dict],
    output_path: str = "multirun_eval.csv",
) -> str:
    """Save per-problem multi-run results to CSV for paper tables."""
    import csv
    rows = sorted(summary.values(), key=lambda v: (v["difficulty"], v["optimal_shots"]))
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "algorithm", "num_qubits", "backend", "difficulty", "optimal_shots",
            "shots_mean", "shots_std", "mae_mean", "mae_std", "all_shots",
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved multi-run CSV → {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Agent vs Oracle comparison SVG (all 36 paper problems)
# ─────────────────────────────────────────────────────────────────────────────

def generate_comparison_svg(
    agent: Agent,
    triplets: List[Tuple[str, int, str]],
    output_path: str = "agent_vs_oracle.svg",
) -> str:
    """
    Run the trained agent on EVERY problem in `triplets` and produce an SVG
    table comparing agent shots vs oracle shots for each trace.
    """
    env = IterativeQuantumEnv(triplets, label="comparison")
    comparison: List[Dict] = []

    for triplet in tqdm(triplets, desc="Agent vs Oracle comparison"):
        info, _ = run_greedy_episode(agent, env, triplet)
        comparison.append(info)

    result_map: Dict[Tuple, Dict] = {tuple(r["triplet"]): r for r in comparison}

    easy_trips = sorted([t for t in triplets if classify_problem(t) == "easy"],
                        key=lambda x: (x[0], x[1], x[2]))
    hard_trips = sorted([t for t in triplets if classify_problem(t) == "hard"],
                        key=lambda x: (x[0], x[1], x[2]))

    row_h, col_w, header_h, padding = 20, 540, 100, 16
    n_rows = max(len(easy_trips), len(hard_trips))
    svg_w  = 2 * col_w + 3 * padding
    svg_h  = header_h + n_rows * row_h + 2 * padding

    all_maes  = [r["mae"] for r in comparison]
    easy_maes = [result_map[t]["mae"] for t in easy_trips]
    hard_maes = [result_map[t]["mae"] for t in hard_trips]
    avg_mae  = np.mean(all_maes)
    avg_easy = np.mean(easy_maes) if easy_maes else 0
    avg_hard = np.mean(hard_maes) if hard_maes else 0

    config_lbl = get_config_label(for_svg=True)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="monospace" font-size="12">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="#fafafa"/>',
        f'<text x="{svg_w//2}" y="18" font-size="12" fill="#37474f" '
        f'text-anchor="middle">{config_lbl}</text>',
        f'<text x="{svg_w//2}" y="38" font-size="16" font-weight="bold" fill="#1a237e" '
        f'text-anchor="middle">Agent vs Oracle — {len(triplets)} Problems  '
        f'(MAE: {avg_mae:.0f} shots, {avg_mae/MAX_SHOTS*100:.2f}%)</text>',
        f'<text x="{padding}" y="62" font-size="14" font-weight="bold" fill="#2e7d32">'
        f'EASY [{len(easy_trips)}]  avg MAE={avg_easy:.0f}</text>',
        f'<text x="{col_w + 2*padding}" y="62" font-size="14" font-weight="bold" fill="#c62828">'
        f'HARD [{len(hard_trips)}]  avg MAE={avg_hard:.0f}</text>',
        f'<text x="{padding}" y="79" font-size="11" fill="#666">'
        f'{"Algorithm":>8s}  {"q":>3s}  {"Backend":<18s}  {"Oracle":>6s}  {"Agent":>6s}  {"Δ":>7s}</text>',
        f'<text x="{col_w + 2*padding}" y="79" font-size="11" fill="#666">'
        f'{"Algorithm":>8s}  {"q":>3s}  {"Backend":<18s}  {"Oracle":>6s}  {"Agent":>6s}  {"Δ":>7s}</text>',
        f'<line x1="0" y1="85" x2="{svg_w}" y2="85" stroke="#bbb"/>',
    ]

    def _row(triplet, idx, x_off):
        alg, sz, be = triplet
        r = result_map.get(tuple(triplet), {})
        oracle = r.get("optimal_shots", ACTUAL_ORACLE_CACHE.get((alg, sz, be), 0))
        agent_shots = r.get("shots_used", 0)
        delta = agent_shots - oracle
        y = header_h + idx * row_h
        ratio = abs(delta) / oracle if oracle > 0 else 0.0
        if ratio <= 0.10:
            clr = "#2e7d32"
        elif ratio <= 0.25:
            clr = "#e65100"
        else:
            clr = "#c62828"
        return (f'<text x="{x_off}" y="{y}" fill="#333">'
                f'{alg:>8s}  {sz:>3d}  {be:<18s}  {oracle:>6d}  {agent_shots:>6d}'
                f'  <tspan fill="{clr}" font-weight="bold">{delta:+7d}</tspan></text>')

    for i, t in enumerate(easy_trips):
        lines.append(_row(t, i, padding))
    for i, t in enumerate(hard_trips):
        lines.append(_row(t, i, col_w + 2 * padding))

    lines.append("</svg>")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Agent vs Oracle comparison SVG saved -> {output_path}")
    return output_path

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisDashboard:
    def __init__(self, training_rewards, evaluation_results, validation_history,
                 val_results: Optional[List[Dict]] = None,
                 train_results: Optional[List[Dict]] = None,
                 filtered_eval_results: Optional[List[Dict]] = None,
                 best_episode: int = -1,
                 agent_type: AgentType = AgentType.GENERIC, max_shots=MAX_SHOTS):
        self.rewards     = training_rewards
        self.results     = evaluation_results
        self.val_history = validation_history
        self.val_results = val_results or []
        self.train_results = train_results or []
        self.filtered_eval_results = filtered_eval_results or []
        self.best_episode = best_episode
        self.agent_type  = agent_type
        self.max_shots   = max_shots

        self.shots_used    = [r["shots_used"] for r in self.results]
        self.optimal_shots = [r["optimal_shots"] for r in self.results]
        self.errors        = np.array([r["error"] for r in self.results])
        self.maes          = np.abs(self.errors)
        self.difficulties  = [r["difficulty"] for r in self.results]

    def plot_dashboard(self, save_path: str = "dashboard.svg"):
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(36, 22))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.30)

        config_lbl = get_config_label(self.agent_type)
        n_eval = len(self.results)
        train_triplets = build_training_triplets(agent_type=self.agent_type)
        n_train = len(train_triplets)
        n_val = len(build_validation_set(self.agent_type))
        best_ep_str = f"  |  Best ep: {self.best_episode}" if self.best_episode >= 0 else ""
        fig.suptitle(f"DRL for Shot Allocation — {config_lbl}\n"
                     f"Eval: {n_eval} paper problems (test)  |  "
                     f"Train: {n_train} problems (80%)  |  Val: {n_val} problems (20%)"
                     f"{best_ep_str}",
                     fontsize=18, y=0.98)

        # Row 0: training curves + snapshot
        self._plot_training_rewards(fig.add_subplot(gs[0, 0]))
        self._plot_snapshot_evolution(fig.add_subplot(gs[0, 1]))
        self._plot_performance_scatter(fig.add_subplot(gs[0, 2]))
        self._plot_validation_scatter(fig.add_subplot(gs[0, 3]))
        # Row 1: distributions + summary
        self._plot_error_distribution(fig.add_subplot(gs[1, 0]))
        self._plot_efficiency_curve(fig.add_subplot(gs[1, 1]))
        self._plot_mae_summary(fig.add_subplot(gs[1, 2:]))
        # Row 2: scatter diagnostics
        self._plot_training_scatter(fig.add_subplot(gs[2, 0]))
        self._plot_filtered_eval_scatter(fig.add_subplot(gs[2, 1]))
        # Leave gs[2,2] and gs[2,3] empty
        fig.add_subplot(gs[2, 2]).axis("off")
        fig.add_subplot(gs[2, 3]).axis("off")

        fig.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"Dashboard saved -> {save_path}")
        plt.close(fig)

    def _plot_training_rewards(self, ax):
        ax.plot(self.rewards, alpha=0.3, color="gray", label="Raw")
        if len(self.rewards) >= 100:
            ma = np.convolve(self.rewards, np.ones(100) / 100, mode="valid")
            ax.plot(ma, color="blue", label="Moving Avg (100)")
        ax.set_title("Training Rewards", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_snapshot_evolution(self, ax):
        eps  = [x["episode"] for x in self.val_history]
        maes = [x["mae"] for x in self.val_history]
        pcts = [x.get("error_pct", m / self.max_shots * 100)
                for x, m in zip(self.val_history, maes)]
        ax.plot(eps, maes, marker="o", color="green", linewidth=2)
        ax.set_title("Snapshot Validation: Model Evolution", fontsize=14)
        ax.set_ylabel("MAE on Validation Set (shots)")
        ax.set_xlabel("Training Episode")
        ax.grid(True, alpha=0.5)

        ax2 = ax.twinx()
        ax2.plot(eps, pcts, marker="x", color="orange", linewidth=1,
                 alpha=0.6, label="Error %")
        ax2.set_ylabel("Error %  (MAE / max_shots × 100)")
        ax2.legend(loc="upper right")

        if len(maes) > 1:
            ax.text(0.05, 0.95,
                    f"Best MAE: {min(maes):.1f}\nImprov: {maes[0]-min(maes):.1f}",
                    transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.8),
                    verticalalignment="top")

    def _plot_performance_scatter(self, ax):
        """Policy vs Oracle scatter on EVALUATION set — color-coded by easy/hard."""
        easy_idx = [i for i, d in enumerate(self.difficulties) if d == "easy"]
        hard_idx = [i for i, d in enumerate(self.difficulties) if d == "hard"]

        opt = np.array(self.optimal_shots)
        used = np.array(self.shots_used)

        if easy_idx:
            ax.scatter(opt[easy_idx], used[easy_idx], alpha=0.6,
                       edgecolors="k", c="#4caf50", label=f"Easy ({len(easy_idx)})",
                       s=40)
        if hard_idx:
            ax.scatter(opt[hard_idx], used[hard_idx], alpha=0.6,
                       edgecolors="k", c="#e53935", label=f"Hard ({len(hard_idx)})",
                       s=40)

        m = max(max(self.optimal_shots), max(self.shots_used))
        ax.plot([0, m], [0, m], "k--", alpha=0.5, label="Ideal")

        mae_val = float(np.mean(self.maes))
        pct = mae_val / self.max_shots * 100
        n_eval = len(self.results)
        ax.set_title(f"EVAL SET ({n_eval} probs)  —  MAE: {mae_val:.0f}  ({pct:.2f}%)",
                     fontsize=14)
        ax.set_xlabel("Oracle Optimal Shots")
        ax.set_ylabel("Agent Shots")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_validation_scatter(self, ax):
        """Policy vs Oracle scatter on the 20% VALIDATION set — color-coded by easy/hard.
        Shows how well the agent generalises to held-out (non-paper) problems."""
        if not self.val_results:
            ax.text(0.5, 0.5, "No validation-set\nevaluation data",
                    transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray")
            ax.set_title("VAL SET — No Data", fontsize=14)
            ax.axis("off")
            return

        vr_shots   = [r["shots_used"] for r in self.val_results]
        vr_optimal = [r["optimal_shots"] for r in self.val_results]
        vr_diff    = [r["difficulty"] for r in self.val_results]
        vr_errors  = np.array([r["error"] for r in self.val_results])
        vr_maes    = np.abs(vr_errors)

        easy_idx = [i for i, d in enumerate(vr_diff) if d == "easy"]
        hard_idx = [i for i, d in enumerate(vr_diff) if d == "hard"]
        opt  = np.array(vr_optimal)
        used = np.array(vr_shots)

        if easy_idx:
            ax.scatter(opt[easy_idx], used[easy_idx], alpha=0.6,
                       edgecolors="k", c="#4caf50", label=f"Easy ({len(easy_idx)})",
                       s=40)
        if hard_idx:
            ax.scatter(opt[hard_idx], used[hard_idx], alpha=0.6,
                       edgecolors="k", c="#e53935", label=f"Hard ({len(hard_idx)})",
                       s=40)

        m = max(max(vr_optimal), max(vr_shots))
        ax.plot([0, m], [0, m], "k--", alpha=0.5, label="Ideal")

        val_mae = float(np.mean(vr_maes))
        val_pct = val_mae / self.max_shots * 100
        eval_mae = float(np.mean(self.maes))
        eval_pct = eval_mae / self.max_shots * 100
        gap = val_pct - eval_pct

        ax.set_title(f"VAL SET ({len(self.val_results)} probs)  —  MAE: {val_mae:.0f}  ({val_pct:.2f}%)",
                     fontsize=14)
        ax.set_xlabel("Oracle Optimal Shots")
        ax.set_ylabel("Agent Shots")

        icon = "≈" if abs(gap) < 2.0 else ("↑" if gap > 0 else "↓")
        ax.text(0.05, 0.95,
                f"Val err : {val_pct:.2f}%\nTest err: {eval_pct:.2f}%\nΔ: {gap:+.2f}%  {icon}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_error_distribution(self, ax):
        ax.hist(self.errors, bins=40, color="purple", alpha=0.7, edgecolor="black")
        ax.axvline(0, color="r", linestyle="--")
        ax.set_title("Error Distribution (Residuals)", fontsize=14)
        ax.set_xlabel("Error (Agent − Oracle)")

    def _plot_efficiency_curve(self, ax):
        safe_opt = np.where(np.array(self.optimal_shots) == 0, 1, self.optimal_shots)
        ratios = np.array(self.shots_used) / safe_opt
        ax.hist(ratios, bins=50, color="orange", alpha=0.7, edgecolor="black")
        ax.axvline(1.0, color="r", linewidth=2, linestyle="--", label="Ideal (1.0)")
        ax.set_title("Efficiency Ratio Distribution", fontsize=14)
        ax.set_xlabel("Ratio (Agent / Oracle)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_mae_summary(self, ax):
        mae_val    = float(np.mean(self.maes))
        median_mae = float(np.median(self.maes))
        pct        = mae_val / self.max_shots * 100
        pct_med    = median_mae / self.max_shots * 100

        easy_maes = [self.maes[i] for i, d in enumerate(self.difficulties) if d == "easy"]
        hard_maes = [self.maes[i] for i, d in enumerate(self.difficulties) if d == "hard"]

        n_under = int(np.sum(self.errors < 0))
        n_over  = int(np.sum(self.errors > 0))
        n_exact = int(np.sum(self.errors == 0))

        n_easy = len(easy_maes)
        n_hard = len(hard_maes)
        n_total = n_easy + n_hard

        train_triplets = build_training_triplets(agent_type=self.agent_type)
        split_stats = format_training_split_stats_compact(train_triplets)

        train_triplets_count = len(train_triplets)
        val_triplets_count   = len(build_validation_set(self.agent_type))
        txt = (
            f"EVAL: {n_total} problems ({n_easy}E + {n_hard}H)\n"
            f"Agent Type   : {self.agent_type.value.upper()}\n"
            f"Split Metric : {SPLIT_METRIC.value.upper()}\n"
            f"Oracle eps   : {ACTUAL_ORACLE_EPS}  batch: {ACTUAL_ORACLE_BATCH}\n"
            f"{'─'*36}\n"
            f"Mean MAE     : {mae_val:>8.1f} shots\n"
            f"Median MAE   : {median_mae:>8.1f} shots\n"
            f"Error %      : {pct:>8.2f}%  (mean)\n"
            f"Error % med  : {pct_med:>8.2f}%  (median)\n"
            f"{'─'*36}\n"
        )
        if easy_maes:
            txt += f"Easy MAE     : {np.mean(easy_maes):>8.1f}  ({n_easy} probs)\n"
        if hard_maes:
            txt += f"Hard MAE     : {np.mean(hard_maes):>8.1f}  ({n_hard} probs)\n"
        txt += (
            f"{'─'*36}\n"
            f"Undershoot   : {n_under:>5d}  ({n_under/len(self.errors)*100:.1f}%)\n"
            f"Overshoot    : {n_over:>5d}  ({n_over/len(self.errors)*100:.1f}%)\n"
            f"Exact        : {n_exact:>5d}  ({n_exact/len(self.errors)*100:.1f}%)\n"
            f"{'─'*36}\n"
            f"{split_stats}\n"
            f"{'─'*36}\n"
            f"Train set    : {train_triplets_count} problems (80%)\n"
            f"Val set      : {val_triplets_count} problems (20%)\n"
            f"max_shots    : {self.max_shots}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=12, fontfamily="monospace", verticalalignment="top",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.set_title("MAE & Error %", fontsize=14)
        ax.axis("off")

    def _plot_training_scatter(self, ax):
        """Policy vs Oracle scatter on the TRAINING set — overfitting diagnostic."""
        if not self.train_results:
            ax.text(0.5, 0.5, "No training-set\nevaluation data",
                    transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray")
            ax.set_title("TRAIN SET — No Data", fontsize=14)
            ax.axis("off")
            return

        tr_shots   = [r["shots_used"] for r in self.train_results]
        tr_optimal = [r["optimal_shots"] for r in self.train_results]
        tr_diff    = [r["difficulty"] for r in self.train_results]
        tr_errors  = np.array([r["error"] for r in self.train_results])
        tr_maes    = np.abs(tr_errors)

        easy_idx = [i for i, d in enumerate(tr_diff) if d == "easy"]
        hard_idx = [i for i, d in enumerate(tr_diff) if d == "hard"]
        opt  = np.array(tr_optimal)
        used = np.array(tr_shots)

        if easy_idx:
            ax.scatter(opt[easy_idx], used[easy_idx], alpha=0.4,
                       edgecolors="k", c="#4caf50", label=f"Easy ({len(easy_idx)})",
                       s=20)
        if hard_idx:
            ax.scatter(opt[hard_idx], used[hard_idx], alpha=0.4,
                       edgecolors="k", c="#e53935", label=f"Hard ({len(hard_idx)})",
                       s=20)

        m = max(max(tr_optimal), max(tr_shots))
        ax.plot([0, m], [0, m], "k--", alpha=0.5, label="Ideal")

        train_mae = float(np.mean(tr_maes))
        train_pct = train_mae / self.max_shots * 100
        eval_mae  = float(np.mean(self.maes))
        eval_pct  = eval_mae / self.max_shots * 100
        gap = train_pct - eval_pct

        ax.set_title(f"TRAIN SET ({len(self.train_results)} probs)  —  MAE: {train_mae:.0f}  ({train_pct:.2f}%)",
                     fontsize=14)
        ax.set_xlabel("Oracle Optimal Shots")
        ax.set_ylabel("Agent Shots")

        icon = "≈" if abs(gap) < 2.0 else ("↑" if gap > 0 else "↓")
        ax.text(0.05, 0.95,
                f"Train err: {train_pct:.2f}%\nTest err : {eval_pct:.2f}%\nΔ: {gap:+.2f}%  {icon}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_filtered_eval_scatter(self, ax):
        """Policy vs Oracle scatter on the agent-type-FILTERED subset of the 36 paper problems."""
        if not self.filtered_eval_results:
            ax.text(0.5, 0.5, "No filtered-eval\nevaluation data",
                    transform=ax.transAxes, ha="center", va="center", fontsize=14, color="gray")
            ax.set_title("FILTERED EVAL — No Data", fontsize=14)
            ax.axis("off")
            return

        fr_shots   = [r["shots_used"] for r in self.filtered_eval_results]
        fr_optimal = [r["optimal_shots"] for r in self.filtered_eval_results]
        fr_diff    = [r["difficulty"] for r in self.filtered_eval_results]
        fr_errors  = np.array([r["error"] for r in self.filtered_eval_results])
        fr_maes    = np.abs(fr_errors)

        easy_idx = [i for i, d in enumerate(fr_diff) if d == "easy"]
        hard_idx = [i for i, d in enumerate(fr_diff) if d == "hard"]
        opt  = np.array(fr_optimal)
        used = np.array(fr_shots)

        if easy_idx:
            ax.scatter(opt[easy_idx], used[easy_idx], alpha=0.6,
                       edgecolors="k", c="#4caf50", label=f"Easy ({len(easy_idx)})",
                       s=40)
        if hard_idx:
            ax.scatter(opt[hard_idx], used[hard_idx], alpha=0.6,
                       edgecolors="k", c="#e53935", label=f"Hard ({len(hard_idx)})",
                       s=40)

        m = max(max(fr_optimal), max(fr_shots))
        ax.plot([0, m], [0, m], "k--", alpha=0.5, label="Ideal")

        filt_mae = float(np.mean(fr_maes))
        filt_pct = filt_mae / self.max_shots * 100
        n_easy = len(easy_idx)
        n_hard = len(hard_idx)
        suffix = f"{self.agent_type.value.upper()} filter: {n_easy}E + {n_hard}H"

        ax.set_title(f"FILTERED EVAL ({len(self.filtered_eval_results)} probs)  —  MAE: {filt_mae:.0f}  ({filt_pct:.2f}%)",
                     fontsize=14)
        ax.set_xlabel("Oracle Optimal Shots")
        ax.set_ylabel("Agent Shots")

        ax.text(0.05, 0.95, suffix,
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Full training + evaluation pipeline (was the notebook's __main__ block)."""
    global SPLIT_METRIC   # rebound by the config below and read by classify_problem & co.

    # ── Configuration ───────────────────────────────────────────────────
    NUM_EPISODES        = 1000  # v2: longer run — best-model checkpointing keeps the peak anyway
    CHECKPOINT_START    = 400 # first episode eligible for best-model checkpointing (ε ≈ 0.09 by then)
    CHECKPOINT_INTERVAL = 100
    PATIENCE            = 5
    SEED                = 42
    AGENT_TYPE          = AgentType.GENERIC   # EASY / HARD / GENERIC / UNBALANCED
    REWARD_TYPE         = RewardType.ASYMMETRIC   # MAE / ASYMMETRIC / HARD_ASYMMETRIC / PRECISION
    SPLIT_METRIC        = SplitMetric.SIZE  # SIZE / ORACLE
    # Actual oracle parameters (Algorithm 1) — change to rerun with different settings
    ORACLE_EPS          = ACTUAL_ORACLE_EPS   # default 0.1
    ORACLE_BATCH        = ACTUAL_ORACLE_BATCH # default 50
    # ────────────────────────────────────────────────────────────────────

    # 0. Pre-compute actual oracle for all training+validation triplets
    #    (loads from cache if already computed; writes JSON + CSV otherwise)
    print("\nStep 0: Building actual oracle dataset (Algorithm 1)...")
    _ = build_actual_oracle_dataset(eps=ORACLE_EPS, batch=ORACLE_BATCH)

    # 0b. Generate SVG of all 36 paper problems (test set overview)
    paper_triplets = build_paper_triplets()
    # Ensure paper traces have oracle values for the overview SVG
    for t in paper_triplets:
        find_optimal_shots(*t, eps=ORACLE_EPS, batch=ORACLE_BATCH)
    generate_problem_svg(paper_triplets, output_path="problem_overview.svg")

    # 1. Train on 80% actual-oracle dataset (distribution bias set by AGENT_TYPE)
    agent, train_env, rewards, val_history, best_episode = train_agent(
        agent_type=AGENT_TYPE,
        reward_type=REWARD_TYPE,
        num_episodes=NUM_EPISODES,
        target_sync_steps=TARGET_SYNC_STEPS,
        snapshot_interval=200,
        checkpoint_start=CHECKPOINT_START,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        patience=PATIENCE,
        seed=SEED,
    )

    # 2. Evaluate on ALL 36 paper problems
    results = evaluate_agent(agent, agent_type=AGENT_TYPE, seed=SEED)

    # 2b. Evaluate on the 20% validation set (for dashboard scatter)
    val_triplets = build_validation_set(agent_type=AGENT_TYPE, seed=SEED)
    val_results  = evaluate_on_triplets(agent, val_triplets, desc="Eval on val set (20%)")

    # 2c. Evaluate on the training set (overfitting diagnostic)
    train_triplets = build_training_triplets(agent_type=AGENT_TYPE)
    train_results  = evaluate_on_triplets(agent, train_triplets, desc="Eval on train set (80%)")

    # 2d. Evaluate on the agent-type-filtered subset of the 36 paper problems
    filtered_triplets = build_filtered_eval_set(AGENT_TYPE, seed=SEED)
    filtered_eval_results = evaluate_on_triplets(agent, filtered_triplets,
                                                  desc=f"Eval on filtered test ({AGENT_TYPE.value})")

    tag = AGENT_TYPE.value  # "easy", "hard", "generic", or "unbalanced"
    # Per-version output folder so different program versions cannot
    # overwrite each other's results (file names keep the agent-type tag)
    outdir = f"{tag}-enhanced-2-extrap"
    os.makedirs(outdir, exist_ok=True)
    # 2e. Multi-run evaluation (N runs for mean ± std per problem)
    N_EVAL_RUNS = 1
    print(f"\nStep 2e: Running {N_EVAL_RUNS}-run evaluation on 36 paper problems...")
    multirun_summary = multi_run_evaluation(
        agent, paper_triplets, n_runs=N_EVAL_RUNS,
        desc=f"Multi-run eval ({AGENT_TYPE.value})",
    )
    generate_multirun_svg(
        multirun_summary,
        output_path=f"{outdir}/multirun_eval-{tag}.svg",
        n_runs=N_EVAL_RUNS,
        agent_type=AGENT_TYPE,
    )
    save_multirun_csv(multirun_summary, output_path=f"{outdir}/multirun_eval-{tag}.csv")

    # 3. Dashboard (includes validation, training, and filtered-eval scatters)
    dash = AnalysisDashboard(rewards, results, val_history,
                             val_results=val_results,
                             train_results=train_results,
                             filtered_eval_results=filtered_eval_results,
                             best_episode=best_episode,
                             agent_type=AGENT_TYPE)
    dash.plot_dashboard(save_path=f"{outdir}/dashboard-{tag}.svg")

    # 4. Agent vs oracle on ALL 36 paper problems
    generate_comparison_svg(
        agent, paper_triplets,
        output_path=f"{outdir}/agent_vs_oracle-{tag}-all36.svg",
    )

    # 4b. Agent vs oracle on the agent-type-filtered subset
    chart_a_suffix = f"{AGENT_TYPE.value}-filtered"
    generate_comparison_svg(
        agent, filtered_triplets,
        output_path=f"{outdir}/agent_vs_oracle-{chart_a_suffix}.svg",
    )


if __name__ == "__main__":
    main()
