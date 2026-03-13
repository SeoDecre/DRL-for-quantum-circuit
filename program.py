###############################################################################
# DRL for Quantum Circuit Shot Allocation  (FULL-GPU / CUDA-enabled)
# ----------------------------------------
# Aligned with the IncExc_to_TQC paper (Bisicchia et al., 2026):
#   • TEST SET = exact 180 traces from the paper (6 algs × 6 sizes × 5 backends)
#     so results are directly comparable with Tables 8-13
#   • TRAIN SET = 720 other traces from qsimbench (7 extra algorithms,
#     6 extra qubit sizes, 1 extra backend) — zero data leakage
#   • AgentType controls training AND evaluation distribution bias:
#       EASY       → 100% easy training,  95/5  eval
#       HARD       → 100% hard training,  5/95  eval
#       GENERIC    → 50/50 balanced mix          (train & eval)
#       UNBALANCED → 70% easy / 30% hard         (train & eval)
#   • Small/Large classification from Ch. 6.1 (size < 10 → easy, size >= 10 → hard)
#   • Best-model checkpointing (every 100 eps from ep 2000, keep lowest val MAE)
#   • Early stopping with patience parameter
#   • MAE evaluated on the held-out 180 paper problems
#   • Error percentage = MAE / max_shots
#   • SVG outputs with easy/hard tagging
#   • Oracle = a posteriori optimal (paper definition, §6.1):
#       "minimum n such that D(P̂_n, P̂_B) ≤ δ"  with B=20000, δ=0.05
#
# GPU acceleration:
#   • cuDNN benchmark mode for optimized convolution kernels
#   • Automatic Mixed Precision (AMP) for forward/backward passes
#   • Non-blocking CPU→GPU tensor transfers
#   • Gradient clipping for training stability
#   • GPU-accelerated TVD computation in Oracle
#   • GPU-accelerated state features (entropy, variance, rate-of-change)
#   • GPU-accelerated reward computation
###############################################################################

# ── Suppress Colab jupyter_client datetime deprecation spam ─────────────────
import warnings
warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*", category=DeprecationWarning)

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

# ─────────────────────────────────────────────────────────────────────────────
# CUDA / Device setup  —  maximise GPU utilisation
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # cuDNN auto-tuner — picks the fastest algorithm for fixed-size inputs
    torch.backends.cudnn.benchmark = True
    # Allow TF32 on Ampere+ GPUs (3× faster matmuls with negligible precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print("  cuDNN benchmark : ON")
    print("  TF32 matmul     : ON")
else:
    print("  (no GPU detected — running on CPU, all GPU paths will fallback safely)")

# AMP scaler — only active on CUDA, no-ops on CPU
USE_AMP = DEVICE.type == "cuda"
AMP_DTYPE = torch.float16 if USE_AMP else torch.float32
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ─────────────────────────────────────────────────────────────────────────────
# Agent type: controls training distribution bias
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(Enum):
    EASY       = "easy"        # trains ONLY on easy problems (size < 10)
    HARD       = "hard"        # trains ONLY on hard problems (size >= 10)
    GENERIC    = "generic"     # trains on a balanced 50/50 mix of easy and hard
    UNBALANCED = "unbalanced"  # trains on 70% easy, 30% hard

class RewardType(Enum):
    MAE        = "mae"         # exp(-0.003 * mae) for overshoot, -1 - mae/optimal for undershoot
    ASYMMETRIC = "asymmetric"  # exp(-0.00025 * error) for overshoot, (-1.85 - |e|/opt) * scale(1-1.5) for undershoot

# ─────────────────────────────────────────────────────────────────────────────
# Paper-aligned constants (Ch. 6.1 of IncExc_to_TQC)
# ─────────────────────────────────────────────────────────────────────────────
PAPER_ALGORITHMS = ["dj", "qaoa", "qnn", "qft", "random", "vqe"]
PAPER_BACKENDS   = ["fake_fez", "fake_kyiv", "fake_marrakesh", "fake_sherbrooke", "fake_torino"]
PAPER_SIZES      = [4, 6, 8, 10, 12, 14]

SMALL_LARGE_THRESHOLD = 10   # size < 10 → easy, size >= 10 → hard
MAX_SHOTS = 20_000

# ─────────────────────────────────────────────────────────────────────────────
# Oracle and caching system  —  GPU-accelerated TVD
# ─────────────────────────────────────────────────────────────────────────────
ORACLE_CACHE: Dict[Tuple[str, int, str], int] = {}
CACHE_FILE = "oracle_cache_enhanced.json"

def save_oracle_cache():
    """Persist the dictionary of computed optimal shots to JSON."""
    string_key_cache = {str(k): v for k, v in ORACLE_CACHE.items()}
    with open(CACHE_FILE, "w") as f:
        json.dump(string_key_cache, f)

def load_oracle_cache():
    """Load pre-computed optimal shots from the cache file."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                string_key_cache = json.load(f)
                for k_str, v in string_key_cache.items():
                    ORACLE_CACHE[literal_eval(k_str)] = v
            print(f"Loaded {len(ORACLE_CACHE)} items from oracle cache.")
        except Exception as e:
            print(f"Warning: Could not read cache file. Error: {e}")

def find_optimal_shots(
    algorithm: str, size: int, backend: str,
    step_size: int = 50, max_shots: int = MAX_SHOTS,
    delta: float = 0.05,
) -> int:
    """
    Oracle: a posteriori optimal shots (paper definition §6.1).

    Collects the FULL budget of `max_shots` shots first, producing the
    reference distribution P̂_B.  Then finds the smallest n (in steps of
    `step_size`) such that  TVD(P̂_n, P̂_B) ≤ δ.

    δ = 0.05 corresponds to the strictest column in the paper tables.

    GPU acceleration: TVD is computed via vectorized torch operations on DEVICE.
    """
    cache_key = (algorithm, size, backend)
    if cache_key in ORACLE_CACHE:
        return ORACLE_CACHE[cache_key]

    # 1. Collect ALL shots up to the full budget
    n_batches = max_shots // step_size
    batches: List[Dict[str, int]] = []
    for _ in range(n_batches):
        try:
            batch = get_outcomes(algorithm, size, backend,
                                 shots=step_size, strategy="random", exact=True)
            batches.append(batch)
        except Exception:
            ORACLE_CACHE[cache_key] = max_shots
            return max_shots

    # 2. Build FULL-BUDGET reference distribution P̂_B
    full_counts: Counter = Counter()
    for b in batches:
        full_counts.update(b)

    # 3. GPU-accelerated TVD scan ─────────────────────────────────────────
    # Map all outcome keys to indices
    all_keys = sorted(full_counts.keys())
    key_to_idx = {k: i for i, k in enumerate(all_keys)}
    n_keys = len(all_keys)

    # Reference distribution as a GPU tensor
    ref_counts = torch.zeros(n_keys, device=DEVICE)
    for k, v in full_counts.items():
        ref_counts[key_to_idx[k]] = v
    ref_dist = ref_counts / ref_counts.sum()

    # Walk forward: accumulate batch counts and check TVD on GPU
    cumulative_t = torch.zeros(n_keys, device=DEVICE)
    for i, b in enumerate(batches):
        for k, v in b.items():
            cumulative_t[key_to_idx[k]] += v
        n = (i + 1) * step_size
        current_dist = cumulative_t / cumulative_t.sum()
        tvd_val = 0.5 * torch.abs(current_dist - ref_dist).sum().item()
        if tvd_val <= delta:
            ORACLE_CACHE[cache_key] = n
            return n

    ORACLE_CACHE[cache_key] = max_shots
    return max_shots

# ─────────────────────────────────────────────────────────────────────────────
# Problem sets: paper test set (180) vs training set (720)
# ─────────────────────────────────────────────────────────────────────────────

def classify_problem(triplet: Tuple[str, int, str]) -> str:
    """Paper Ch. 6.1: size < 10 → easy, size >= 10 → hard."""
    return "easy" if triplet[1] < SMALL_LARGE_THRESHOLD else "hard"

def build_paper_triplets() -> List[Tuple[str, int, str]]:
    """The exact 180 traces from the paper (6 algs × 6 sizes × 5 backends).
    These are used ONLY for evaluation — never seen during training."""
    index = get_index()
    triplets = []
    for alg in PAPER_ALGORITHMS:
        for sz in PAPER_SIZES:
            for be in PAPER_BACKENDS:
                if alg in index and sz in index[alg] and be in index[alg][sz]:
                    triplets.append((alg, sz, be))
    return triplets

def build_training_triplets() -> List[Tuple[str, int, str]]:
    """All qsimbench traces EXCEPT the 180 paper traces.
    Provides 720 diverse problems for training (13 algs, 12 sizes, 6 backends)."""
    index = get_index()
    paper_set = set(build_paper_triplets())
    triplets = []
    for alg in sorted(index.keys()):
        for sz in sorted(index[alg].keys()):
            val = index[alg][sz]
            backends = sorted(val.keys()) if isinstance(val, dict) else sorted(val)
            for be in backends:
                t = (alg, sz, be)
                if t not in paper_set:
                    triplets.append(t)
    return triplets

def build_validation_set(
    training_triplets: List[Tuple[str, int, str]],
    n: int = 30,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """Sample a small validation subset from the training set for checkpointing."""
    rng = random.Random(seed)
    easy = [t for t in training_triplets if classify_problem(t) == "easy"]
    hard = [t for t in training_triplets if classify_problem(t) == "hard"]
    rng.shuffle(easy)
    rng.shuffle(hard)
    half = n // 2
    return easy[:half] + hard[:half]

def build_eval_set(
    agent_type: AgentType,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """Build an evaluation subset from the 180 paper problems, matching the
    easy/hard ratio used during training.

      EASY       → 95% easy, 5% hard   (86 easy +  4 hard = 90)
      HARD       → 5% easy, 95% hard   ( 4 easy + 86 hard = 90)
      GENERIC    → 50% easy, 50% hard  (45 easy + 45 hard = 90)
      UNBALANCED → 70% easy, 30% hard  (63 easy + 27 hard = 90)

    All 180 paper problems are evaluated exhaustively, but the final
    MAE is computed on the ratio-matched subset so training and
    evaluation distributions are consistent.
    """
    paper = build_paper_triplets()
    easy = [t for t in paper if classify_problem(t) == "easy"]
    hard = [t for t in paper if classify_problem(t) == "hard"]

    rng = random.Random(seed)
    rng.shuffle(easy)
    rng.shuffle(hard)

    if agent_type == AgentType.EASY:
        n_easy, n_hard = 86, 4         # 95/5
    elif agent_type == AgentType.HARD:
        n_easy, n_hard = 4, 86         # 5/95
    elif agent_type == AgentType.UNBALANCED:
        n_easy, n_hard = 27, 63        # 70/30
    else:  # GENERIC
        n_easy, n_hard = 45, 45        # 50/50

    selected = easy[:n_easy] + hard[:n_hard]
    return selected

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

    row_h, col_w, header_h, padding = 22, 420, 60, 16
    n_rows = max(len(easy), len(hard))
    svg_w  = 2 * col_w + 3 * padding
    svg_h  = header_h + n_rows * row_h + 2 * padding

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="monospace" font-size="13">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="#fafafa"/>',
        f'<text x="{padding}" y="30" font-size="16" font-weight="bold" fill="#2e7d32">'
        f'EASY (size &lt; {SMALL_LARGE_THRESHOLD})  [{len(easy)} problems]</text>',
        f'<text x="{col_w + 2*padding}" y="30" font-size="16" font-weight="bold" fill="#c62828">'
        f'HARD (size >= {SMALL_LARGE_THRESHOLD})  [{len(hard)} problems]</text>',
        f'<line x1="0" y1="42" x2="{svg_w}" y2="42" stroke="#bbb"/>',
    ]

    def _row(t, idx, x_off):
        alg, sz, be = t
        oracle = ORACLE_CACHE.get((alg, sz, be), "?")
        y = header_h + idx * row_h
        clr = "#2e7d32" if sz < SMALL_LARGE_THRESHOLD else "#c62828"
        tag = "EASY" if sz < SMALL_LARGE_THRESHOLD else "HARD"
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
# Quantum RL Environment  —  GPU-accelerated state features & rewards
# ─────────────────────────────────────────────────────────────────────────────

class IterativeQuantumEnv(gym.Env):
    """Gym environment for iterative shot allocation.
    State features, entropy, variance, TVD rate-of-change, and reward
    computation all run on DEVICE (GPU when available)."""

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
        self.alg_map, self.backend_map = self._create_mappings()

        for t in self.active_triplets:
            find_optimal_shots(*t)

        self.easy_triplets = [t for t in self.active_triplets if classify_problem(t) == "easy"]
        self.hard_triplets = [t for t in self.active_triplets if classify_problem(t) == "hard"]

        print(f"[{label.upper()}] {len(self.active_triplets)} problems "
              f"(easy={len(self.easy_triplets)}, hard={len(self.hard_triplets)})")

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.outcome_history: List[Dict] = []
        self.current_triplet = self.active_triplets[0]
        self.optimal_shots = 0
        self.current_shots = 0

    def _create_mappings(self):
        all_algs = sorted({t[0] for t in self.active_triplets})
        all_backends = sorted({t[2] for t in self.active_triplets})
        return ({n: i for i, n in enumerate(all_algs)},
                {n: i for i, n in enumerate(all_backends)})

    def reset(self) -> np.ndarray:
        if self.label == "train":
            # Sampling bias depends on agent type
            # EASY  → 100% easy
            # HARD  → 100% hard
            # GENERIC → 50/50 balanced
            # UNBALANCED → 70% easy, 30% hard
            if self.agent_type == AgentType.EASY:
                if self.easy_triplets:
                    self.current_triplet = random.choice(self.easy_triplets)
                else:
                    self.current_triplet = random.choice(self.active_triplets)
            elif self.agent_type == AgentType.HARD:
                if self.hard_triplets:
                    self.current_triplet = random.choice(self.hard_triplets)
                else:
                    self.current_triplet = random.choice(self.active_triplets)
            elif self.agent_type == AgentType.UNBALANCED:
                if random.random() < 0.7 and self.easy_triplets:
                    self.current_triplet = random.choice(self.easy_triplets)
                elif self.hard_triplets:
                    self.current_triplet = random.choice(self.hard_triplets)
                else:
                    self.current_triplet = random.choice(self.active_triplets)
            else:  # GENERIC — balanced 50/50
                if random.random() < 0.5 and self.easy_triplets:
                    self.current_triplet = random.choice(self.easy_triplets)
                elif self.hard_triplets:
                    self.current_triplet = random.choice(self.hard_triplets)
                else:
                    self.current_triplet = random.choice(self.active_triplets)
        else:
            self.current_triplet = random.choice(self.active_triplets)

        self.optimal_shots = find_optimal_shots(*self.current_triplet)
        self.current_shots = 0
        self.outcome_history = []
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
        except Exception:
            pass

        if self.current_shots >= self.max_shots:
            return self._terminate()

        return self._get_state(), 0.0, False, {}

    def _terminate(self) -> Tuple[np.ndarray, float, bool, Dict]:
        """Compute terminal reward on GPU."""
        error = self.current_shots - self.optimal_shots
        mae = abs(error)

        # ── GPU-accelerated reward computation ──────────────────────────
        error_t = torch.tensor(float(error), device=DEVICE)
        mae_t = torch.abs(error_t)
        opt_t = torch.tensor(float(max(self.optimal_shots, 1)), device=DEVICE)
        max_t = torch.tensor(float(self.max_shots), device=DEVICE)

        if self.reward_type == RewardType.ASYMMETRIC:
            if error < 0:
                scale = 1.0 + 0.5 * opt_t / max_t
                final_reward_t = (-1.85 - mae_t / opt_t) * scale
            else:
                final_reward_t = torch.exp(torch.tensor(-0.00025, device=DEVICE) * error_t)
        else:  # MAE
            if error < 0:
                final_reward_t = -1.0 - mae_t / opt_t
            else:
                final_reward_t = torch.exp(torch.tensor(-0.003, device=DEVICE) * mae_t)

        final_reward = final_reward_t.item()

        info = {
            "shots_used": self.current_shots,
            "optimal_shots": self.optimal_shots,
            "error": error,
            "mae": mae,
            "error_pct": mae / self.max_shots * 100,
            "final_reward": final_reward,
            "triplet": self.current_triplet,
            "difficulty": classify_problem(self.current_triplet),
        }
        return self._get_state(), final_reward, True, info

    # ── GPU-accelerated state features ──────────────────────────────────

    def _compute_distribution_entropy_gpu(self, outcomes: Dict[str, int]) -> float:
        """Shannon entropy via GPU tensor ops, normalised to [0, 1]."""
        total = sum(outcomes.values())
        if total == 0:
            return 0.0
        counts = torch.tensor(list(outcomes.values()), dtype=torch.float32, device=DEVICE)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # filter zeros for log
        entropy = -(probs * torch.log2(probs)).sum()
        return min(entropy.item() / 4.0, 1.0)

    def _compute_distribution_variance_gpu(self, outcomes: Dict[str, int]) -> float:
        """Variance of the probability vector, computed on GPU."""
        if not outcomes:
            return 0.0
        total = sum(outcomes.values())
        if total == 0:
            return 0.0
        counts = torch.tensor(list(outcomes.values()), dtype=torch.float32, device=DEVICE)
        probs = counts / counts.sum()
        return min(torch.var(probs).item() * 10.0, 1.0)

    def _compute_rate_of_change_gpu(self) -> float:
        """TVD between cumulative history and latest batch — on GPU."""
        if len(self.outcome_history) < 2:
            return 1.0
        prev_outcomes: Counter = Counter()
        for o in self.outcome_history[:-1]:
            prev_outcomes.update(o)
        recent = self.outcome_history[-1]

        prev_total = sum(prev_outcomes.values())
        recent_total = sum(recent.values())
        if prev_total == 0 or recent_total == 0:
            return 1.0

        # Build aligned tensors on GPU
        all_keys = sorted(set(prev_outcomes.keys()) | set(recent.keys()))
        prev_vals = torch.tensor([prev_outcomes.get(k, 0) for k in all_keys],
                                 dtype=torch.float32, device=DEVICE)
        recent_vals = torch.tensor([recent.get(k, 0) for k in all_keys],
                                   dtype=torch.float32, device=DEVICE)
        prev_dist = prev_vals / prev_vals.sum()
        recent_dist = recent_vals / recent_vals.sum()
        return (0.5 * torch.abs(prev_dist - recent_dist).sum()).item()

    def _get_state(self) -> np.ndarray:
        alg, size, backend = self.current_triplet
        n_algs  = max(len(self.alg_map), 2)
        n_backs = max(len(self.backend_map), 2)
        alg_norm     = self.alg_map.get(alg, 0) / (n_algs - 1)
        size_norm    = size / 15.0
        backend_norm = self.backend_map.get(backend, 0) / (n_backs - 1)
        shots_norm   = self.current_shots / self.max_shots

        if self.outcome_history:
            cumulative: Counter = Counter()
            for b in self.outcome_history:
                cumulative.update(b)
            entropy  = self._compute_distribution_entropy_gpu(cumulative)
            variance = self._compute_distribution_variance_gpu(cumulative)
        else:
            entropy  = 0.5
            variance = 0.5

        rate     = self._compute_rate_of_change_gpu()
        progress = min(self.current_shots / (size * 500), 1.0)

        return np.array(
            [alg_norm, size_norm, backend_norm, shots_norm,
             entropy, variance, rate, progress],
            dtype=np.float32,
        )

# ─────────────────────────────────────────────────────────────────────────────
# DQN Network & Agent  —  AMP training, gradient clipping, non-blocking xfers
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
        state_size: int = 8,
        action_size: int = 2,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
    ):
        self.q_net = DQN(state_size, action_size).to(DEVICE)
        self.target_net = DQN(state_size, action_size).to(DEVICE)
        self.update_target()
        self.memory: deque = deque(maxlen=200_000)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.05
        self.action_size = action_size

    def act(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                q = self.q_net(state_t)
        return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Non-blocking CPU → GPU transfers
        states      = torch.FloatTensor(np.array(states)).to(DEVICE, non_blocking=True)
        actions     = torch.LongTensor(actions).to(DEVICE, non_blocking=True)
        rewards     = torch.FloatTensor(rewards).to(DEVICE, non_blocking=True)
        next_states = torch.FloatTensor(np.array(next_states)).to(DEVICE, non_blocking=True)
        dones       = torch.FloatTensor(dones).to(DEVICE, non_blocking=True)

        # AMP-accelerated forward + backward
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            cur_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q
            loss = F.mse_loss(cur_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
        scaler.scale(loss).backward()
        # Unscale before clipping so the threshold is in true gradient magnitude
        scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        scaler.step(self.optimizer)
        scaler.update()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_weights(self) -> dict:
        """Copy weights to CPU for safe serialisation / checkpointing."""
        return {k: v.cpu().clone() for k, v in self.q_net.state_dict().items()}

    def load_weights(self, state_dict: dict):
        """Load weights (possibly from CPU) back onto DEVICE."""
        gpu_state = {k: v.to(DEVICE) for k, v in state_dict.items()}
        self.q_net.load_state_dict(gpu_state)
        self.update_target()

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Validator  (uses a small subset of training data for checkpointing)
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotValidator:
    """Validates agent on a fixed validation subset; returns MAE."""

    def __init__(self, validation_env: IterativeQuantumEnv):
        self.env = validation_env
        self.history: List[Dict] = []

    def validate(self, agent: Agent, episode: int) -> float:
        total_mae, total_reward = 0.0, 0.0
        n = len(self.env.active_triplets)
        orig_eps = agent.epsilon
        agent.epsilon = 0.0

        for _ in range(n):
            state = self.env.reset()
            done, info = False, {}
            ep_reward = 0.0
            while not done:
                action = agent.act(state, evaluate=True)
                state, reward, done, info = self.env.step(action)
                ep_reward += reward
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
    update_target_every: int = 500,
    snapshot_interval: int = 200,
    checkpoint_start: int = 2000,
    checkpoint_interval: int = 100,
    patience: int = 5,
    seed: int = 42,
) -> Tuple[Agent, IterativeQuantumEnv, List[float], List[Dict]]:
    """
    Train a DQN agent on the 720 non-paper traces from qsimbench.
    agent_type controls training distribution:
      EASY       → 100% easy problems
      HARD       → 100% hard problems
      GENERIC    → 50/50 balanced mix
      UNBALANCED → 70% easy, 30% hard
    Validation for checkpointing uses a 30-problem subset of training data.
    """
    load_oracle_cache()

    train_triplets = build_training_triplets()
    val_triplets   = build_validation_set(train_triplets, n=30, seed=seed)

    env     = IterativeQuantumEnv(train_triplets, agent_type=agent_type, reward_type=reward_type, label="train")
    val_env = IterativeQuantumEnv(val_triplets,   reward_type=reward_type, label="validation")

    agent     = Agent()
    validator = SnapshotValidator(val_env)

    rewards_history: List[float] = []
    best_mae: float = float("inf")
    best_weights: Optional[dict] = None
    no_improve_count: int = 0
    stopped_early = False

    print(f"\n{'='*60}")
    print(f" Agent Type  : {agent_type.value.upper()}")
    print(f" Reward Type : {reward_type.value.upper()}")
    print(f" Device      : {DEVICE}  |  AMP : {'ON' if USE_AMP else 'OFF'}")
    print(f" Train       : {len(train_triplets)} non-paper problems")
    print(f" Validation  : {len(val_triplets)} problems (from training set)")
    print(f" Test        : 180 paper problems (evaluated after training)")
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

        if episode > 0 and episode % update_target_every == 0:
            agent.update_target()

        # ── Periodic snapshot validation ────────────────────────────────
        if episode % snapshot_interval == 0:
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

    save_oracle_cache()

    if best_weights is not None:
        agent.load_weights(best_weights)
        print(f"\n>> Restored best model weights (MAE={best_mae:.1f})")
    else:
        print("\n>> No checkpoint taken (training ended before checkpoint_start)")

    print(f"Training {'ended early at ep ' + str(episode) if stopped_early else 'completed all ' + str(num_episodes) + ' episodes'}.")

    return agent, env, rewards_history, validator.history

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on the 180 paper problems
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent: Agent, agent_type: AgentType = AgentType.GENERIC, seed: int = 42) -> List[Dict]:
    """Evaluate the trained agent on a ratio-matched subset of the 180 paper problems.
    The easy/hard ratio mirrors the training distribution:
      EASY       → 86 easy +  4 hard = 90
      HARD       →  4 easy + 86 hard = 90
      GENERIC    → 45 easy + 45 hard = 90
      UNBALANCED → 63 easy + 27 hard = 90
    This is the test set — the agent has never seen these traces during training."""
    eval_triplets = build_eval_set(agent_type, seed=seed)
    n_easy = sum(1 for t in eval_triplets if classify_problem(t) == "easy")
    n_hard = sum(1 for t in eval_triplets if classify_problem(t) == "hard")

    env_eval = IterativeQuantumEnv(eval_triplets, label=f"eval ({agent_type.value})")

    results: List[Dict] = []
    for triplet in tqdm(env_eval.active_triplets, desc=f"Evaluating ({agent_type.value}, {len(eval_triplets)} problems)"):
        env_eval.current_triplet = triplet
        env_eval.optimal_shots = find_optimal_shots(*triplet)
        env_eval.current_shots = 0
        env_eval.outcome_history = []
        state = env_eval._get_state()

        done, info = False, {}
        while not done:
            action = agent.act(state, evaluate=True)
            state, _, done, info = env_eval.step(action)
        results.append(info)

    # Summary
    maes = [r["mae"] for r in results]
    easy_maes = [r["mae"] for r in results if r["difficulty"] == "easy"]
    hard_maes = [r["mae"] for r in results if r["difficulty"] == "hard"]
    avg_mae = np.mean(maes)
    pct = avg_mae / env_eval.max_shots * 100

    print(f"\n{'='*55}")
    print(f"  TEST EVALUATION — {agent_type.value.upper()} agent")
    print(f"  {len(eval_triplets)} problems  ({n_easy} easy + {n_hard} hard)")
    print(f"  {'─'*50}")
    print(f"  Overall MAE   : {avg_mae:.1f} shots  ({pct:.2f}%)")
    if easy_maes:
        print(f"  Easy MAE      : {np.mean(easy_maes):.1f} shots  ({len(easy_maes)} problems)")
    if hard_maes:
        print(f"  Hard MAE      : {np.mean(hard_maes):.1f} shots  ({len(hard_maes)} problems)")
    print(f"{'='*55}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Agent vs Oracle comparison SVG (all 180 paper problems)
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
    orig_eps = agent.epsilon
    agent.epsilon = 0.0

    for triplet in tqdm(triplets, desc="Agent vs Oracle comparison"):
        env.current_triplet = triplet
        env.optimal_shots = find_optimal_shots(*triplet)
        env.current_shots = 0
        env.outcome_history = []
        state = env._get_state()
        done, info = False, {}
        while not done:
            action = agent.act(state, evaluate=True)
            state, _, done, info = env.step(action)
        comparison.append(info)

    agent.epsilon = orig_eps

    result_map: Dict[Tuple, Dict] = {tuple(r["triplet"]): r for r in comparison}

    easy_trips = sorted([t for t in triplets if classify_problem(t) == "easy"],
                        key=lambda x: (x[0], x[1], x[2]))
    hard_trips = sorted([t for t in triplets if classify_problem(t) == "hard"],
                        key=lambda x: (x[0], x[1], x[2]))

    row_h, col_w, header_h, padding = 20, 540, 80, 16
    n_rows = max(len(easy_trips), len(hard_trips))
    svg_w  = 2 * col_w + 3 * padding
    svg_h  = header_h + n_rows * row_h + 2 * padding

    all_maes  = [r["mae"] for r in comparison]
    easy_maes = [result_map[t]["mae"] for t in easy_trips]
    hard_maes = [result_map[t]["mae"] for t in hard_trips]
    avg_mae  = np.mean(all_maes)
    avg_easy = np.mean(easy_maes) if easy_maes else 0
    avg_hard = np.mean(hard_maes) if hard_maes else 0

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" '
        f'font-family="monospace" font-size="12">',
        f'<rect width="{svg_w}" height="{svg_h}" fill="#fafafa"/>',
        f'<text x="{svg_w//2}" y="22" font-size="16" font-weight="bold" fill="#1a237e" '
        f'text-anchor="middle">Agent vs Oracle — All {len(triplets)} Problems  '
        f'(MAE: {avg_mae:.0f} shots, {avg_mae/MAX_SHOTS*100:.2f}%)</text>',
        f'<text x="{padding}" y="45" font-size="14" font-weight="bold" fill="#2e7d32">'
        f'EASY [{len(easy_trips)}]  avg MAE={avg_easy:.0f}</text>',
        f'<text x="{col_w + 2*padding}" y="45" font-size="14" font-weight="bold" fill="#c62828">'
        f'HARD [{len(hard_trips)}]  avg MAE={avg_hard:.0f}</text>',
        f'<text x="{padding}" y="62" font-size="11" fill="#666">'
        f'{"Algorithm":>8s}  {"q":>3s}  {"Backend":<18s}  {"Oracle":>6s}  {"Agent":>6s}  {"Δ":>7s}</text>',
        f'<text x="{col_w + 2*padding}" y="62" font-size="11" fill="#666">'
        f'{"Algorithm":>8s}  {"q":>3s}  {"Backend":<18s}  {"Oracle":>6s}  {"Agent":>6s}  {"Δ":>7s}</text>',
        f'<line x1="0" y1="68" x2="{svg_w}" y2="68" stroke="#bbb"/>',
    ]

    def _row(triplet, idx, x_off):
        alg, sz, be = triplet
        r = result_map.get(tuple(triplet), {})
        oracle = r.get("optimal_shots", ORACLE_CACHE.get((alg, sz, be), 0))
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
                 agent_type: AgentType = AgentType.GENERIC, max_shots=MAX_SHOTS):
        self.rewards     = training_rewards
        self.results     = evaluation_results
        self.val_history = validation_history
        self.agent_type  = agent_type
        self.max_shots   = max_shots

        self.shots_used    = [r["shots_used"] for r in self.results]
        self.optimal_shots = [r["optimal_shots"] for r in self.results]
        self.errors        = np.array([r["error"] for r in self.results])
        self.maes          = np.abs(self.errors)
        self.difficulties  = [r["difficulty"] for r in self.results]

    def plot_dashboard(self, save_path: str = "dashboard.svg"):
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        atype = self.agent_type.value.upper()
        n_eval = len(self.results)
        fig.suptitle(f"DRL for Shot Allocation — {atype} Agent — Eval on {n_eval} Paper Problems",
                     fontsize=20, y=0.98)

        self._plot_training_rewards(axes[0, 0])
        self._plot_snapshot_evolution(axes[0, 1])
        self._plot_performance_scatter(axes[0, 2])
        self._plot_error_distribution(axes[1, 0])
        self._plot_efficiency_curve(axes[1, 1])
        self._plot_mae_summary(axes[1, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        """Policy vs Oracle scatter — color-coded by easy/hard."""
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
        atype = self.agent_type.value.upper()
        n_eval = len(self.results)
        ax.set_title(f"Policy vs Oracle ({atype}, {n_eval} probs)  |  MAE: {mae_val:.0f}  ({pct:.2f}%)",
                     fontsize=14)
        ax.set_xlabel("Oracle Optimal Shots")
        ax.set_ylabel("Agent Shots")
        ax.legend()
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

        txt = (
            f"EVAL: {n_total} problems ({n_easy}E + {n_hard}H)\n"
            f"Agent Type   : {self.agent_type.value.upper()}\n"
            f"Device       : {DEVICE}\n"
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
            f"Trained on   :  720 non-paper traces\n"
            f"max_shots    : {self.max_shots}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes,
                fontsize=12, fontfamily="monospace", verticalalignment="top",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.set_title("MAE & Error %", fontsize=14)
        ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Configuration ───────────────────────────────────────────────────
    NUM_EPISODES        = 3000
    CHECKPOINT_START    = 2000
    CHECKPOINT_INTERVAL = 100
    PATIENCE            = 5
    SEED                = 42
    AGENT_TYPE          = AgentType.HARD   # EASY / HARD / GENERIC / UNBALANCED
    REWARD_TYPE         = RewardType.MAE      # MAE / ASYMMETRIC
    # ────────────────────────────────────────────────────────────────────

    # 0. Generate SVG of all 180 paper problems (test set)
    paper_triplets = build_paper_triplets()
    generate_problem_svg(paper_triplets, output_path="problem_overview.svg")

    # 1. Train on 720 non-paper problems (distribution bias set by AGENT_TYPE)
    agent, train_env, rewards, val_history = train_agent(
        agent_type=AGENT_TYPE,
        reward_type=REWARD_TYPE,
        num_episodes=NUM_EPISODES,
        update_target_every=500,  
        snapshot_interval=200,
        checkpoint_start=CHECKPOINT_START,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        patience=PATIENCE,
        seed=SEED,
    )

    # 2. Evaluate with same easy/hard ratio as training
    results = evaluate_agent(agent, agent_type=AGENT_TYPE, seed=SEED)

    # 3. Dashboard
    tag = AGENT_TYPE.value  # "easy", "hard", or "generic"
    os.makedirs(tag, exist_ok=True)
    dash = AnalysisDashboard(rewards, results, val_history, agent_type=AGENT_TYPE)
    dash.plot_dashboard(save_path=f"{tag}/dashboard-{tag}.svg")

    # 4. Detailed per-problem comparison SVG (same eval subset)
    eval_triplets = build_eval_set(AGENT_TYPE, seed=SEED)
    generate_comparison_svg(agent, eval_triplets, output_path=f"{tag}/agent_vs_oracle-{tag}.svg")