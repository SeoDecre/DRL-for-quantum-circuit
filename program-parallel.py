#!/usr/bin/env python3
###############################################################################
# DRL for Quantum Circuit Shot Allocation — PARALLEL version
# -----------------------------------------------------------
# Runs ALL 4 agent-type configurations simultaneously:
#   EASY / HARD / GENERIC / UNBALANCED
#
# Designed for multi-core VMs.  Uses multiprocessing to train + evaluate
# each agent type in its own process.  Shared oracle cache is pre-computed
# once before forking.
#
# Usage:
#   python program-parallel.py
#   python program-parallel.py --episodes 5000 --patience 8
#   python program-parallel.py --split oracle --reward mae
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
import sys
import json
from ast import literal_eval
import copy
import multiprocessing as mp
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class AgentType(Enum):
    EASY       = "easy"
    HARD       = "hard"
    GENERIC    = "generic"
    UNBALANCED = "unbalanced"

class RewardType(Enum):
    MAE        = "mae"
    ASYMMETRIC = "asymmetric"

class SplitMetric(Enum):
    SIZE   = "size"
    ORACLE = "oracle"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
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

SMALL_LARGE_THRESHOLD = 10
ORACLE_EASY_THRESHOLD = 5_000
MAX_SHOTS = 20_000
LOW_SHOT_THRESHOLD = 5_000
CACHE_FILE = "oracle_cache_enhanced.json"

# These are set from CLI args before forking
SPLIT_METRIC: SplitMetric = SplitMetric.SIZE
REWARD_TYPE: RewardType   = RewardType.ASYMMETRIC

# ─────────────────────────────────────────────────────────────────────────────
# Cached index + global feature maps
# ─────────────────────────────────────────────────────────────────────────────
_INDEX_CACHE: Optional[dict] = None

def get_cached_index() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = get_index()
    return _INDEX_CACHE

_GLOBAL_ALG_MAP: Optional[Dict[str, int]] = None
_GLOBAL_BACKEND_MAP: Optional[Dict[str, int]] = None

def get_global_mappings() -> Tuple[Dict[str, int], Dict[str, int]]:
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
# Oracle cache
# ─────────────────────────────────────────────────────────────────────────────
ORACLE_CACHE: Dict[Tuple[str, int, str], int] = {}

def save_oracle_cache():
    string_key_cache = {str(k): v for k, v in ORACLE_CACHE.items()}
    with open(CACHE_FILE, "w") as f:
        json.dump(string_key_cache, f)

def load_oracle_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                string_key_cache = json.load(f)
                for k_str, v in string_key_cache.items():
                    ORACLE_CACHE[literal_eval(k_str)] = v
        except Exception:
            pass

def find_optimal_shots(
    algorithm: str, size: int, backend: str,
    step_size: int = 50, max_shots: int = MAX_SHOTS,
    delta: float = 0.05,
) -> int:
    cache_key = (algorithm, size, backend)
    if cache_key in ORACLE_CACHE:
        return ORACLE_CACHE[cache_key]

    def normalize(counts):
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total > 0 else {}

    def tvd(p, q):
        all_keys = set(p) | set(q)
        return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in all_keys)

    n_batches = max_shots // step_size
    batches = []
    for _ in range(n_batches):
        try:
            batch = get_outcomes(algorithm, size, backend,
                                 shots=step_size, strategy="random", exact=True)
            batches.append(batch)
        except Exception:
            ORACLE_CACHE[cache_key] = max_shots
            return max_shots

    full_counts: Counter = Counter()
    for b in batches:
        full_counts.update(b)
    ref_dist = normalize(full_counts)

    cumulative: Counter = Counter()
    for i, b in enumerate(batches):
        cumulative.update(b)
        n = (i + 1) * step_size
        if tvd(normalize(cumulative), ref_dist) <= delta:
            ORACLE_CACHE[cache_key] = n
            return n

    ORACLE_CACHE[cache_key] = max_shots
    return max_shots

# ─────────────────────────────────────────────────────────────────────────────
# Problem sets
# ─────────────────────────────────────────────────────────────────────────────

def classify_problem(triplet: Tuple[str, int, str]) -> str:
    if SPLIT_METRIC == SplitMetric.SIZE:
        return "easy" if triplet[1] < SMALL_LARGE_THRESHOLD else "hard"
    else:
        oracle = find_optimal_shots(*triplet)
        return "easy" if oracle <= ORACLE_EASY_THRESHOLD else "hard"

def build_paper_triplets() -> List[Tuple[str, int, str]]:
    return list(PAPER_TRACES)


def build_training_triplets() -> List[Tuple[str, int, str]]:
    index = get_cached_index()
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

def build_validation_set(agent_type: AgentType, seed: int = 42) -> List[Tuple[str, int, str]]:
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
        n_easy = max(1, round(n_total * 0.2))
        n_hard = n_total - n_easy
        rng.shuffle(easy)
        rng.shuffle(hard)
        return easy[:min(n_easy, len(easy))] + hard[:min(n_hard, len(hard))]
    else:
        return easy + hard

def build_eval_set(agent_type: AgentType, seed: int = 42) -> List[Tuple[str, int, str]]:
    paper = build_paper_triplets()
    easy = [t for t in paper if classify_problem(t) == "easy"]
    hard = [t for t in paper if classify_problem(t) == "hard"]
    return easy + hard

def get_config_label(agent_type: AgentType, for_svg: bool = False) -> str:
    atype = agent_type.value.upper()
    reward = REWARD_TYPE.value.upper()
    if SPLIT_METRIC == SplitMetric.SIZE:
        lt = "&lt;" if for_svg else "<"
        split = f"SIZE (size {lt} {SMALL_LARGE_THRESHOLD})"
    else:
        lte = "&lt;=" if for_svg else "≤"
        split = f"ORACLE (oracle {lte} {ORACLE_EASY_THRESHOLD})"
    return f"Agent: {atype}  |  Reward: {reward}  |  Split: {split}"

def format_training_split_stats_compact(train_triplets):
    n_total = len(train_triplets)
    if SPLIT_METRIC == SplitMetric.ORACLE:
        n_low = sum(1 for t in train_triplets
                    if ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        n_low_easy = sum(1 for t in train_triplets
                         if classify_problem(t) == "easy"
                         and ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        n_low_hard = sum(1 for t in train_triplets
                         if classify_problem(t) == "hard"
                         and ORACLE_CACHE.get(t, find_optimal_shots(*t)) < LOW_SHOT_THRESHOLD)
        return (f"Train < {LOW_SHOT_THRESHOLD} (oracle) : {n_low:>5d}/{n_total}  ({n_low/n_total*100:.1f}%)"
                f"  [{n_low_easy}E+{n_low_hard}H]")
    else:
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
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class IterativeQuantumEnv(gym.Env):
    def __init__(self, triplets, max_shots=MAX_SHOTS, step_size=50,
                 agent_type=AgentType.GENERIC, reward_type=RewardType.MAE,
                 label="env", silent=False):
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

        # ── Oracle-stratified bins for ORACLE split ─────────────────────
        # When SPLIT_METRIC=ORACLE, the HARD pool is dominated by near-max
        # oracle values.  Stratified sampling across oracle-range bins ensures
        # the agent sees a balanced distribution of stopping points.
        self._oracle_bins: Dict[str, Dict[int, List[Tuple[str, int, str]]]] = {}
        if SPLIT_METRIC == SplitMetric.ORACLE and label == "train":
            bin_edges = [0, 5000, 8000, 11000, 14000, 17000, MAX_SHOTS + 1]
            for pool_name, pool in [("easy", self.easy_triplets), ("hard", self.hard_triplets)]:
                bins: Dict[int, List[Tuple[str, int, str]]] = {}
                for t in pool:
                    oracle_val = find_optimal_shots(*t)
                    for j in range(len(bin_edges) - 1):
                        if bin_edges[j] <= oracle_val < bin_edges[j + 1]:
                            bins.setdefault(j, []).append(t)
                            break
                self._oracle_bins[pool_name] = {k: v for k, v in bins.items() if v}

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.outcome_history: List[Dict] = []
        self.current_triplet = self.active_triplets[0]
        self.optimal_shots = 0
        self.current_shots = 0
        self._cumulative: Counter = Counter()
        self._cached_entropy: float = 0.5
        self._cached_variance: float = 0.5
        self._cache_valid: bool = False

    def _stratified_sample(self, pool_name: str) -> Tuple[str, int, str]:
        """Pick a problem using oracle-stratified sampling (uniform across bins)."""
        bins = self._oracle_bins.get(pool_name, {})
        if bins:
            bin_key = random.choice(list(bins.keys()))
            return random.choice(bins[bin_key])
        pool = self.hard_triplets if pool_name == "hard" else self.easy_triplets
        return random.choice(pool or self.active_triplets)

    def reset(self):
        if self.label == "train":
            use_stratified = (SPLIT_METRIC == SplitMetric.ORACLE and self._oracle_bins)

            if self.agent_type == AgentType.EASY:
                pool = self.easy_triplets or self.active_triplets
                self.current_triplet = (self._stratified_sample("easy")
                                        if use_stratified and self.easy_triplets else
                                        random.choice(pool))
            elif self.agent_type == AgentType.HARD:
                pool = self.hard_triplets or self.active_triplets
                self.current_triplet = (self._stratified_sample("hard")
                                        if use_stratified and self.hard_triplets else
                                        random.choice(pool))
            elif self.agent_type == AgentType.UNBALANCED:
                if random.random() < 0.2 and self.easy_triplets:
                    self.current_triplet = (self._stratified_sample("easy")
                                            if use_stratified else
                                            random.choice(self.easy_triplets))
                else:
                    pool = self.hard_triplets or self.active_triplets
                    self.current_triplet = (self._stratified_sample("hard")
                                            if use_stratified and self.hard_triplets else
                                            random.choice(pool))
            else:
                if random.random() < 0.5 and self.easy_triplets:
                    self.current_triplet = (self._stratified_sample("easy")
                                            if use_stratified else
                                            random.choice(self.easy_triplets))
                else:
                    pool = self.hard_triplets or self.active_triplets
                    self.current_triplet = (self._stratified_sample("hard")
                                            if use_stratified and self.hard_triplets else
                                            random.choice(pool))
        else:
            self.current_triplet = random.choice(self.active_triplets)

        self.optimal_shots = find_optimal_shots(*self.current_triplet)
        self.current_shots = 0
        self.outcome_history = []
        self._cumulative = Counter()
        self._cached_entropy = 0.5
        self._cached_variance = 0.5
        self._cache_valid = False
        return self._get_state()

    def step(self, action):
        if action == 1:
            return self._terminate()
        self.current_shots += self.step_size
        try:
            alg, size, backend = self.current_triplet
            batch = get_outcomes(alg, size, backend,
                                 shots=self.step_size, strategy="random", exact=True)
            self.outcome_history.append(batch)
            self._cumulative.update(batch)
            self._cache_valid = False
        except Exception:
            pass
        if self.current_shots >= self.max_shots:
            return self._terminate()
        return self._get_state(), 0.0, False, {}

    def _terminate(self):
        error = self.current_shots - self.optimal_shots
        mae = abs(error)
        if self.reward_type == RewardType.ASYMMETRIC:
            if error < 0:
                scale = 1.0 + 0.5 * self.optimal_shots / self.max_shots
                final_reward = (-1.85 - (mae / max(self.optimal_shots, 1))) * scale
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
            "error": error, "mae": mae,
            "error_pct": mae / self.max_shots * 100,
            "final_reward": final_reward,
            "triplet": self.current_triplet,
            "difficulty": classify_problem(self.current_triplet),
        }
        return self._get_state(), final_reward, True, info

    def _compute_distribution_entropy(self, outcomes):
        total = sum(outcomes.values())
        if total == 0: return 0.0
        entropy = 0.0
        for count in outcomes.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return min(entropy / 4.0, 1.0)

    def _compute_distribution_variance(self, outcomes):
        if not outcomes: return 0.0
        total = sum(outcomes.values())
        if total == 0: return 0.0
        probs = [count / total for count in outcomes.values()]
        return min(float(np.var(probs)) * 10.0, 1.0)

    def _update_cached_features(self):
        if self._cache_valid: return
        if self._cumulative:
            self._cached_entropy = self._compute_distribution_entropy(self._cumulative)
            self._cached_variance = self._compute_distribution_variance(self._cumulative)
        else:
            self._cached_entropy = 0.5
            self._cached_variance = 0.5
        self._cache_valid = True

    def _compute_rate_of_change(self):
        if len(self.outcome_history) < 2: return 1.0
        recent = self.outcome_history[-1]
        recent_total = sum(recent.values())
        if recent_total == 0: return 1.0
        prev_cumulative = self._cumulative - Counter(recent)
        prev_total = sum(prev_cumulative.values())
        if prev_total == 0: return 1.0
        all_keys = set(prev_cumulative.keys()) | set(recent.keys())
        tvd = 0.0
        for k in all_keys:
            tvd += abs(prev_cumulative.get(k, 0) / prev_total - recent.get(k, 0) / recent_total)
        return 0.5 * tvd

    def _get_state(self):
        alg, size, backend = self.current_triplet
        n_algs  = max(len(self.alg_map), 2)
        n_backs = max(len(self.backend_map), 2)
        alg_norm     = self.alg_map.get(alg, 0) / (n_algs - 1)
        size_norm    = size / 15.0
        backend_norm = self.backend_map.get(backend, 0) / (n_backs - 1)
        shots_norm   = self.current_shots / self.max_shots
        self._update_cached_features()
        entropy  = self._cached_entropy
        variance = self._cached_variance
        rate     = self._compute_rate_of_change()
        progress = min(self.current_shots / (size * 500), 1.0)
        return np.array(
            [alg_norm, size_norm, backend_norm, shots_norm,
             entropy, variance, rate, progress], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# DQN Network & Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, input_size=8, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, output_size),
        )
    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, state_size=8, action_size=2, learning_rate=1e-4, gamma=0.99):
        self.q_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.update_target()
        self.memory = deque(maxlen=200_000)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.05
        self.action_size = action_size

    def act(self, state, evaluate=False):
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state_t)
        return int(torch.argmax(q).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.FloatTensor(np.array(states))
        actions     = torch.LongTensor(actions)
        rewards     = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones       = torch.FloatTensor(dones)

        cur_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = F.mse_loss(cur_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_weights(self):
        return copy.deepcopy(self.q_net.state_dict())

    def load_weights(self, state_dict):
        self.q_net.load_state_dict(state_dict)
        self.update_target()

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot Validator
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotValidator:
    def __init__(self, validation_env):
        self.env = validation_env
        self.history = []

    def validate(self, agent, episode):
        total_mae = 0.0
        total_reward = 0.0
        n = len(self.env.active_triplets)
        orig_eps = agent.epsilon
        agent.epsilon = 0.0

        for triplet in self.env.active_triplets:
            self.env.current_triplet = triplet
            self.env.optimal_shots = find_optimal_shots(*triplet)
            self.env.current_shots = 0
            self.env.outcome_history = []
            self.env._cumulative = Counter()
            self.env._cache_valid = False
            state = self.env._get_state()
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
        self.history.append({"episode": episode, "mae": avg_mae,
                             "error_pct": error_pct, "reward": avg_reward})
        return avg_mae

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers (silent)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_triplets_silent(agent, triplets, label="eval"):
    """Run agent greedily on triplets, return list of info dicts. No output."""
    env_tmp = IterativeQuantumEnv(triplets, label=label, silent=True)
    results = []
    for triplet in env_tmp.active_triplets:
        env_tmp.current_triplet = triplet
        env_tmp.optimal_shots = find_optimal_shots(*triplet)
        env_tmp.current_shots = 0
        env_tmp.outcome_history = []
        env_tmp._cumulative = Counter()
        env_tmp._cache_valid = False
        state = env_tmp._get_state()
        done, info = False, {}
        while not done:
            action = agent.act(state, evaluate=True)
            state, _, done, info = env_tmp.step(action)
        results.append(info)
    return results

# ─────────────────────────────────────────────────────────────────────────────
# SVG generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_problem_svg(triplets, output_path="problem_overview.svg"):
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
        f'Split: {SPLIT_METRIC.value.upper()}</text>',
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
        oracle = ORACLE_CACHE.get((alg, sz, be), "?")
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


def generate_comparison_svg(agent, triplets, output_path, agent_type):
    env = IterativeQuantumEnv(triplets, label="cmp", silent=True)
    comparison = []
    orig_eps = agent.epsilon
    agent.epsilon = 0.0
    for triplet in triplets:
        env.current_triplet = triplet
        env.optimal_shots = find_optimal_shots(*triplet)
        env.current_shots = 0
        env.outcome_history = []
        env._cumulative = Counter()
        env._cache_valid = False
        state = env._get_state()
        done, info = False, {}
        while not done:
            action = agent.act(state, evaluate=True)
            state, _, done, info = env.step(action)
        comparison.append(info)
    agent.epsilon = orig_eps

    result_map = {tuple(r["triplet"]): r for r in comparison}
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
    config_lbl = get_config_label(agent_type, for_svg=True)

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
        oracle = r.get("optimal_shots", ORACLE_CACHE.get((alg, sz, be), 0))
        agent_shots = r.get("shots_used", 0)
        delta = agent_shots - oracle
        y = header_h + idx * row_h
        ratio = abs(delta) / oracle if oracle > 0 else 0.0
        if ratio <= 0.10:   clr = "#2e7d32"
        elif ratio <= 0.25: clr = "#e65100"
        else:               clr = "#c62828"
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

# ─────────────────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisDashboard:
    def __init__(self, training_rewards, evaluation_results, validation_history,
                 agent_type=AgentType.GENERIC, max_shots=MAX_SHOTS,
                 train_results=None):
        self.rewards     = training_rewards
        self.results     = evaluation_results
        self.val_history = validation_history
        self.agent_type  = agent_type
        self.max_shots   = max_shots
        self.train_results = train_results or []
        self.shots_used    = [r["shots_used"] for r in self.results]
        self.optimal_shots = [r["optimal_shots"] for r in self.results]
        self.errors        = np.array([r["error"] for r in self.results])
        self.maes          = np.abs(self.errors)
        self.difficulties  = [r["difficulty"] for r in self.results]

    def plot_dashboard(self, save_path="dashboard.svg"):
        fig, axes = plt.subplots(2, 4, figsize=(32, 14))
        config_lbl = get_config_label(self.agent_type)
        n_eval = len(self.results)
        fig.suptitle(f"DRL for Shot Allocation — {config_lbl}\n"
                     f"Eval: {n_eval} paper problems  |  Train: 864 non-paper traces",
                     fontsize=18, y=0.99)
        self._plot_training_rewards(axes[0, 0])
        self._plot_snapshot_evolution(axes[0, 1])
        self._plot_performance_scatter(axes[0, 2])
        self._plot_training_scatter(axes[0, 3])
        self._plot_error_distribution(axes[1, 0])
        self._plot_efficiency_curve(axes[1, 1])
        self._plot_mae_summary(axes[1, 2])
        axes[1, 3].axis("off")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        plt.close(fig)

    def _plot_training_rewards(self, ax):
        ax.plot(self.rewards, alpha=0.3, color="gray", label="Raw")
        if len(self.rewards) >= 100:
            ma = np.convolve(self.rewards, np.ones(100) / 100, mode="valid")
            ax.plot(ma, color="blue", label="MA(100)")
        ax.set_title("Training Rewards", fontsize=14)
        ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_snapshot_evolution(self, ax):
        eps  = [x["episode"] for x in self.val_history]
        maes = [x["mae"] for x in self.val_history]
        pcts = [x.get("error_pct", m / self.max_shots * 100)
                for x, m in zip(self.val_history, maes)]
        ax.plot(eps, maes, marker="o", color="green", linewidth=2)
        ax.set_title("Snapshot Validation: Model Evolution", fontsize=14)
        ax.set_ylabel("MAE (shots)"); ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.5)
        ax2 = ax.twinx()
        ax2.plot(eps, pcts, marker="x", color="orange", linewidth=1, alpha=0.6, label="Error %")
        ax2.set_ylabel("Error %"); ax2.legend(loc="upper right")
        if len(maes) > 1:
            ax.text(0.05, 0.95, f"Best MAE: {min(maes):.1f}\nImprov: {maes[0]-min(maes):.1f}",
                    transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.8),
                    verticalalignment="top")

    def _plot_performance_scatter(self, ax):
        easy_idx = [i for i, d in enumerate(self.difficulties) if d == "easy"]
        hard_idx = [i for i, d in enumerate(self.difficulties) if d == "hard"]
        opt = np.array(self.optimal_shots); used = np.array(self.shots_used)
        if easy_idx:
            ax.scatter(opt[easy_idx], used[easy_idx], alpha=0.6, edgecolors="k",
                       c="#4caf50", label=f"Easy ({len(easy_idx)})", s=40)
        if hard_idx:
            ax.scatter(opt[hard_idx], used[hard_idx], alpha=0.6, edgecolors="k",
                       c="#e53935", label=f"Hard ({len(hard_idx)})", s=40)
        m = max(max(self.optimal_shots), max(self.shots_used))
        ax.plot([0, m], [0, m], "k--", alpha=0.5, label="Ideal")
        mae_val = float(np.mean(self.maes)); pct = mae_val / self.max_shots * 100
        ax.set_title(f"EVAL SET ({len(self.results)} probs)  MAE: {mae_val:.0f} ({pct:.2f}%)", fontsize=14)
        ax.set_xlabel("Oracle"); ax.set_ylabel("Agent"); ax.legend(); ax.grid(True, alpha=0.3)

    def _plot_training_scatter(self, ax):
        if not self.train_results:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=14, color="gray")
            ax.set_title("TRAIN SET — No Data", fontsize=14); ax.axis("off"); return
        tr_shots = [r["shots_used"] for r in self.train_results]
        tr_optimal = [r["optimal_shots"] for r in self.train_results]
        tr_diff = [r["difficulty"] for r in self.train_results]
        tr_maes = np.abs(np.array([r["error"] for r in self.train_results]))
        easy_idx = [i for i, d in enumerate(tr_diff) if d == "easy"]
        hard_idx = [i for i, d in enumerate(tr_diff) if d == "hard"]
        opt = np.array(tr_optimal); used = np.array(tr_shots)
        if easy_idx:
            ax.scatter(opt[easy_idx], used[easy_idx], alpha=0.4, edgecolors="k",
                       c="#4caf50", label=f"Easy ({len(easy_idx)})", s=25)
        if hard_idx:
            ax.scatter(opt[hard_idx], used[hard_idx], alpha=0.4, edgecolors="k",
                       c="#e53935", label=f"Hard ({len(hard_idx)})", s=25)
        m = max(max(tr_optimal), max(tr_shots))
        ax.plot([0, m], [0, m], "k--", alpha=0.5, label="Ideal")
        train_mae = float(np.mean(tr_maes)); train_pct = train_mae / self.max_shots * 100
        eval_mae = float(np.mean(self.maes)); eval_pct = eval_mae / self.max_shots * 100
        gap = train_pct - eval_pct
        ax.set_title(f"TRAIN SET ({len(self.train_results)} probs)  MAE: {train_mae:.0f} ({train_pct:.2f}%)", fontsize=14)
        ax.set_xlabel("Oracle"); ax.set_ylabel("Agent")
        icon = "⚠ OVERFIT" if gap < -2.0 else "✓ OK"
        ax.text(0.05, 0.95, f"Train: {train_pct:.2f}%\nEval : {eval_pct:.2f}%\nΔ: {gap:+.2f}%  {icon}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    def _plot_error_distribution(self, ax):
        ax.hist(self.errors, bins=40, color="purple", alpha=0.7, edgecolor="black")
        ax.axvline(0, color="r", linestyle="--")
        ax.set_title("Error Distribution", fontsize=14); ax.set_xlabel("Error (Agent − Oracle)")

    def _plot_efficiency_curve(self, ax):
        safe_opt = np.where(np.array(self.optimal_shots) == 0, 1, self.optimal_shots)
        ratios = np.array(self.shots_used) / safe_opt
        ax.hist(ratios, bins=50, color="orange", alpha=0.7, edgecolor="black")
        ax.axvline(1.0, color="r", linewidth=2, linestyle="--", label="Ideal")
        ax.set_title("Efficiency Ratio", fontsize=14); ax.set_xlabel("Agent / Oracle")
        ax.legend(); ax.grid(True, alpha=0.3)

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
        n_easy = len(easy_maes); n_hard = len(hard_maes); n_total = n_easy + n_hard
        train_triplets = build_training_triplets()
        split_stats = format_training_split_stats_compact(train_triplets)
        txt = (
            f"EVAL: {n_total} problems ({n_easy}E + {n_hard}H)\n"
            f"Agent Type   : {self.agent_type.value.upper()}\n"
            f"Split Metric : {SPLIT_METRIC.value.upper()}\n"
            f"{'─'*36}\n"
            f"Mean MAE     : {mae_val:>8.1f} shots\n"
            f"Median MAE   : {median_mae:>8.1f} shots\n"
            f"Error %      : {pct:>8.2f}%  (mean)\n"
            f"Error % med  : {pct_med:>8.2f}%  (median)\n"
            f"{'─'*36}\n"
        )
        if easy_maes: txt += f"Easy MAE     : {np.mean(easy_maes):>8.1f}  ({n_easy} probs)\n"
        if hard_maes: txt += f"Hard MAE     : {np.mean(hard_maes):>8.1f}  ({n_hard} probs)\n"
        txt += (
            f"{'─'*36}\n"
            f"Undershoot   : {n_under:>5d}  ({n_under/len(self.errors)*100:.1f}%)\n"
            f"Overshoot    : {n_over:>5d}  ({n_over/len(self.errors)*100:.1f}%)\n"
            f"Exact        : {n_exact:>5d}  ({n_exact/len(self.errors)*100:.1f}%)\n"
            f"{'─'*36}\n"
            f"{split_stats}\n"
            f"{'─'*36}\n"
            f"max_shots    : {self.max_shots}"
        )
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=12,
                fontfamily="monospace", verticalalignment="top",
                bbox=dict(facecolor="lightyellow", alpha=0.9, edgecolor="gray"))
        ax.set_title("MAE & Error %", fontsize=14); ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
# Training function (silent, returns results dict)
# ─────────────────────────────────────────────────────────────────────────────

def train_single_agent(
    agent_type: AgentType,
    reward_type: RewardType,
    num_episodes: int = 3000,
    update_target_every: int = 500,
    snapshot_interval: int = 200,
    checkpoint_start: int = 1500,
    checkpoint_interval: int = 100,
    patience: int = 5,
    seed: int = 42,
) -> Tuple[Agent, List[float], List[Dict]]:
    """Train one agent. Returns (agent, rewards_history, val_history)."""

    train_triplets = build_training_triplets()
    val_triplets   = build_validation_set(agent_type=agent_type, seed=seed)

    env     = IterativeQuantumEnv(train_triplets, agent_type=agent_type,
                                  reward_type=reward_type, label="train", silent=True)
    val_env = IterativeQuantumEnv(val_triplets, reward_type=reward_type,
                                  label="validation", silent=True)

    agent     = Agent()
    validator = SnapshotValidator(val_env)

    rewards_history = []
    best_mae = float("inf")
    best_weights = None
    no_improve_count = 0
    stopped_early = False
    tag = agent_type.value.upper()

    for episode in range(num_episodes):
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

        # Skip snapshot if checkpoint will also fire (avoids duplicate validation)
        checkpoint_fires = (episode >= checkpoint_start
                            and episode % checkpoint_interval == 0)
        if episode % snapshot_interval == 0 and not checkpoint_fires:
            validator.validate(agent, episode)

        if episode >= checkpoint_start and episode % checkpoint_interval == 0:
            ckpt_mae = validator.validate(agent, episode)
            if ckpt_mae < best_mae:
                best_mae = ckpt_mae
                best_weights = agent.get_weights()
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                stopped_early = True
                break

        # Minimal progress log every 500 episodes
        if episode % 500 == 0:
            val_mae = validator.history[-1]["mae"] if validator.history else float("inf")
            print(f"  [{tag}] ep {episode:>5d}/{num_episodes}  "
                  f"val_mae={val_mae:.0f}  best={best_mae:.0f}  "
                  f"eps={agent.epsilon:.4f}", flush=True)

    save_oracle_cache()

    if best_weights is not None:
        agent.load_weights(best_weights)

    final_ep = episode if stopped_early else num_episodes
    print(f"  [{tag}] Done: {'early stop @ ep ' + str(final_ep) if stopped_early else str(num_episodes) + ' eps'}  "
          f"best_mae={best_mae:.0f}", flush=True)

    return agent, rewards_history, validator.history

# ─────────────────────────────────────────────────────────────────────────────
# Worker: train + evaluate + generate outputs for one AgentType
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(args_tuple):
    """Full pipeline for one agent type. Designed to run in a separate process."""
    (agent_type, reward_type, split_metric,
     num_episodes, checkpoint_start, checkpoint_interval,
     patience, seed, oracle_cache_snapshot) = args_tuple

    # Restore shared state in this process
    global SPLIT_METRIC, REWARD_TYPE, ORACLE_CACHE
    SPLIT_METRIC = split_metric
    REWARD_TYPE  = reward_type
    ORACLE_CACHE.update(oracle_cache_snapshot)

    # Force re-init of global maps in this process
    global _GLOBAL_ALG_MAP, _GLOBAL_BACKEND_MAP, _INDEX_CACHE
    _GLOBAL_ALG_MAP = None
    _GLOBAL_BACKEND_MAP = None
    _INDEX_CACHE = None
    get_global_mappings()

    tag = agent_type.value
    t0 = time.time()
    print(f"[{tag.upper()}] Starting training...", flush=True)

    # 1. Train
    agent, rewards, val_history = train_single_agent(
        agent_type=agent_type,
        reward_type=reward_type,
        num_episodes=num_episodes,
        checkpoint_start=checkpoint_start,
        checkpoint_interval=checkpoint_interval,
        patience=patience,
        seed=seed,
    )

    # 2. Evaluate on ALL 36 paper problems
    print(f"[{tag.upper()}] Evaluating on 36 paper problems...", flush=True)
    eval_triplets = build_eval_set(agent_type, seed=seed)
    eval_results = evaluate_on_triplets_silent(agent, eval_triplets, label="eval")

    # 3. Evaluate on ALL 864 training problems (overfitting check)
    print(f"[{tag.upper()}] Evaluating on 864 training problems...", flush=True)
    train_triplets = build_training_triplets()
    train_results = evaluate_on_triplets_silent(agent, train_triplets, label="train-eval")

    # 4. Generate outputs
    os.makedirs(tag, exist_ok=True)

    dash = AnalysisDashboard(rewards, eval_results, val_history,
                             agent_type=agent_type, train_results=train_results)
    dash.plot_dashboard(save_path=f"{tag}/dashboard-{tag}.svg")

    generate_comparison_svg(agent, eval_triplets,
                            output_path=f"{tag}/agent_vs_oracle-{tag}.svg",
                            agent_type=agent_type)

    elapsed = time.time() - t0
    eval_maes = [r["mae"] for r in eval_results]
    avg_mae = np.mean(eval_maes)
    pct = avg_mae / MAX_SHOTS * 100

    summary = {
        "agent_type": tag,
        "eval_mae": avg_mae,
        "eval_pct": pct,
        "train_mae": np.mean([r["mae"] for r in train_results]),
        "elapsed_min": elapsed / 60,
        "n_episodes": len(rewards),
        "best_val_mae": min(v["mae"] for v in val_history) if val_history else float("inf"),
    }

    print(f"[{tag.upper()}] ✓ Finished in {elapsed/60:.1f} min  |  "
          f"Eval MAE: {avg_mae:.0f} ({pct:.2f}%)  |  "
          f"Train MAE: {summary['train_mae']:.0f}", flush=True)

    return summary

# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute oracle cache for ALL 900 triplets (before forking)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_oracle_cache():
    """Compute oracle optimal shots for all 900 triplets sequentially.
    This must happen BEFORE forking workers so all processes share the cache."""
    load_oracle_cache()
    all_triplets = build_paper_triplets() + build_training_triplets()
    already = len(ORACLE_CACHE)
    needed = [t for t in all_triplets if t not in ORACLE_CACHE]

    if not needed:
        print(f"Oracle cache: all {len(all_triplets)} triplets already cached.")
        return

    print(f"Oracle cache: {already} cached, computing {len(needed)} remaining...")
    for t in tqdm(needed, desc="Pre-computing oracle"):
        find_optimal_shots(*t)
    save_oracle_cache()
    print(f"Oracle cache complete: {len(ORACLE_CACHE)} triplets.")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global SPLIT_METRIC, REWARD_TYPE

    parser = argparse.ArgumentParser(description="DRL Shot Allocation — Parallel (4 agents)")
    parser.add_argument("--episodes", type=int, default=3000, help="Training episodes per agent")
    parser.add_argument("--checkpoint-start", type=int, default=1500)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["size", "oracle"], default="size")
    parser.add_argument("--reward", choices=["mae", "asymmetric"], default="asymmetric")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    SPLIT_METRIC = SplitMetric.SIZE if args.split == "size" else SplitMetric.ORACLE
    REWARD_TYPE  = RewardType.MAE if args.reward == "mae" else RewardType.ASYMMETRIC

    print("="*70)
    print("  DRL Shot Allocation — PARALLEL (4 agent types)")
    print(f"  Split: {SPLIT_METRIC.value.upper()}  |  Reward: {REWARD_TYPE.value.upper()}")
    print(f"  Episodes: {args.episodes}  |  Patience: {args.patience}  |  Workers: {args.workers}")
    print(f"  CPUs available: {mp.cpu_count()}")
    print("="*70)

    # Phase 1: Pre-compute oracle cache (sequential, must finish before forking)
    t0 = time.time()
    precompute_oracle_cache()

    # Phase 2: Generate problem overview SVG (once)
    paper_triplets = build_paper_triplets()
    generate_problem_svg(paper_triplets, output_path="problem_overview.svg")
    print(f"Problem overview SVG saved.")

    # Ensure global mappings are built
    get_global_mappings()

    # Phase 3: Launch 4 workers in parallel
    agent_types = [AgentType.EASY, AgentType.HARD, AgentType.GENERIC, AgentType.UNBALANCED]

    worker_args = [
        (at, REWARD_TYPE, SPLIT_METRIC,
         args.episodes, args.checkpoint_start, args.checkpoint_interval,
         args.patience, args.seed, dict(ORACLE_CACHE))
        for at in agent_types
    ]

    n_workers = min(args.workers, len(agent_types))
    print(f"\nLaunching {n_workers} parallel workers...\n")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_pipeline, wa): wa[0].value for wa in worker_args}
        summaries = []
        for future in as_completed(futures):
            tag = futures[future]
            try:
                summary = future.result()
                summaries.append(summary)
            except Exception as e:
                print(f"[{tag.upper()}] ✗ FAILED: {e}")

    # Phase 4: Final summary
    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {elapsed_total/60:.1f} min")
    print(f"{'─'*70}")
    print(f"  {'Agent':<12s} {'Eval MAE':>10s} {'Eval %':>8s} {'Train MAE':>10s} {'Time':>8s} {'Eps':>6s}")
    print(f"  {'─'*58}")
    for s in sorted(summaries, key=lambda x: x["eval_mae"]):
        print(f"  {s['agent_type']:<12s} {s['eval_mae']:>10.0f} {s['eval_pct']:>7.2f}% "
              f"{s['train_mae']:>10.0f} {s['elapsed_min']:>7.1f}m {s['n_episodes']:>6d}")
    print(f"{'='*70}")

    # Save summary JSON
    summary_path = "parallel_results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"Summary saved -> {summary_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
