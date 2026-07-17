#!/usr/bin/env python3
"""
Oracle variance analysis — how much does n* move across trace realizations?

The paper's oracle n* (Algorithm 1 of IncExc, suffix-stable TVD) is cached from
a SINGLE realization of each trace (seed 42, qsimbench strategy="sequential").
The agent, however, observes RANDOM realizations (strategy="random", unseeded)
both in training and evaluation. If n* varies a lot between realizations, that
variance is an irreducible floor on the MAE of ANY stopping method scored
against a fixed per-triplet label.

This script measures that floor: for each of the 36 paper test traces it
recomputes n* on K independent random realizations and reports, per triplet,
mean/std/min/max/median and the MAD (mean absolute deviation from the median),
plus the aggregate MAE floor = mean per-triplet MAD.

Usage:
    conda run -n qdrl python oracle_variance.py               # eps=0.1, 20 seeds
    conda run -n qdrl python oracle_variance.py --eps 0.25 --n-seeds 30
    python oracle_variance.py --analyze-only                  # re-analyze CSV only

Writes rows incrementally to oracle_variance_eps<eps>.csv so an interrupted run
loses nothing and can be resumed (already-computed (triplet, seed) pairs are
skipped on restart).
"""
import argparse
import ast
import csv
import json
import os
import statistics
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ── Paper-aligned constants (same as program-main-enhanced.ipynb) ────────────
BATCH = 50
MAX_SHOTS = 20_000

# Exact 36 traces from Tables 8-13 of the IncExc paper (= PAPER_TRACES in the
# notebook; kept as a copy so this script stays standalone).
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

HERE = os.path.dirname(os.path.abspath(__file__))

# Reference numbers for the printed comparison (paper, delta=0.10, 36 traces)
DRL_MAE = 2_049
INC_TVD_MAE = 871


def compute_oracle_random(algorithm: str, size: int, backend: str,
                          eps: float, base_seed: int) -> int:
    """Algorithm 1 (suffix-stable a-posteriori optimum) on ONE random
    realization of the trace.

    Identical to find_optimal_shots in program-main-enhanced.ipynb, except that
    batches are drawn with strategy="random" (what the agent's environment
    actually sees) and a controllable seed, instead of the notebook's default
    strategy="sequential" with seed 42.
    """
    from qsimbench import get_outcomes

    cumulative: Counter = Counter()
    snapshots: List[Dict[str, int]] = []
    shot_counts: List[int] = []

    n_batches = MAX_SHOTS // BATCH
    for bi in range(n_batches):
        new_outcomes = get_outcomes(
            algorithm, size, backend, shots=BATCH,
            strategy="random", exact=True,
            seed=base_seed * 1000 + bi,
        )
        for k, v in new_outcomes.items():
            cumulative[k] += int(v)
        snapshots.append(dict(cumulative))
        shot_counts.append((bi + 1) * BATCH)

    # Step 1: full-budget reference distribution
    ref = snapshots[-1]
    ref_total = sum(ref.values())
    ref_norm = {k: v / ref_total for k, v in ref.items()}

    # Step 2: d[m] = TVD(P_m, P_B) for every prefix m
    dist_values: List[float] = []
    for snap in snapshots:
        snap_total = sum(snap.values())
        snap_norm = {k: v / snap_total for k, v in snap.items()}
        all_keys = set(ref_norm) | set(snap_norm)
        tvd = 0.5 * sum(abs(snap_norm.get(k, 0.0) - ref_norm.get(k, 0.0))
                        for k in all_keys)
        dist_values.append(tvd)

    # Step 3: earliest n whose suffix max stays <= eps
    n_star = shot_counts[-1]
    suffix_max = 0.0
    for idx in range(len(dist_values) - 1, -1, -1):
        suffix_max = max(suffix_max, dist_values[idx])
        if suffix_max <= eps:
            n_star = shot_counts[idx]
    return n_star


def load_cached_oracle(eps: float) -> Dict[Tuple[str, int, str], int]:
    """Load the notebook's cached (sequential, seed-42) oracle values.

    The batch cache JSON excludes the 36 paper test traces (they are computed
    on demand during evaluation), so for those we fall back to the
    `optimal_shots` column of the multirun evaluation CSV — the same source
    the paper tables are generated from.
    """
    out: Dict[Tuple[str, int, str], int] = {}
    path = os.path.join(HERE, f"actual_oracle_eps{eps:g}_batch50.json")
    if os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        for key, val in raw.items():
            # keys are stringified tuples: "('dj', 4, 'fake_kyiv')"
            try:
                alg, size, backend = ast.literal_eval(key)
                out[(alg, int(size), backend)] = int(val)
            except (ValueError, SyntaxError):
                continue
    # fallback for the paper test traces (only relevant at eps=0.1)
    mr = os.path.join(HERE, "generic-enhanced", "multirun_eval-generic.csv")
    if abs(eps - 0.1) < 1e-9 and os.path.exists(mr):
        with open(mr) as f:
            for r in csv.DictReader(f):
                key = (r["algorithm"], int(r["num_qubits"]), r["backend"])
                out.setdefault(key, int(float(r["optimal_shots"])))
    return out


def existing_rows(out_csv: str) -> set:
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["algorithm"], int(r["size"]), r["backend"],
                          int(r["seed"])))
    return done


def run(eps: float, n_seeds: int, out_csv: str) -> None:
    done = existing_rows(out_csv)
    new_file = not os.path.exists(out_csv)
    total = len(PAPER_TRACES) * n_seeds
    t0 = time.time()
    n_done_start = len(done)

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["algorithm", "size", "backend", "seed", "n_star"])
            f.flush()
        count = 0
        for (alg, size, backend) in PAPER_TRACES:
            for seed in range(1, n_seeds + 1):
                count += 1
                if (alg, size, backend, seed) in done:
                    continue
                t1 = time.time()
                n_star = compute_oracle_random(alg, size, backend, eps, seed)
                writer.writerow([alg, size, backend, seed, n_star])
                f.flush()
                elapsed = time.time() - t0
                done_now = count - sum(1 for _ in [None]) + 1  # progress incl. skips
                rate = (count - n_done_start) / max(elapsed, 1e-9)
                remaining = (total - count) / max(rate, 1e-9)
                print(f"[{count}/{total}] {alg}_{size}_{backend} seed={seed} "
                      f"n*={n_star}  ({time.time()-t1:.1f}s, "
                      f"~{remaining/60:.0f} min left)", flush=True)


def analyze(eps: float, out_csv: str) -> None:
    cached = load_cached_oracle(eps)
    per_triplet: Dict[Tuple[str, int, str], List[int]] = {}
    with open(out_csv) as f:
        for r in csv.DictReader(f):
            key = (r["algorithm"], int(r["size"]), r["backend"])
            per_triplet.setdefault(key, []).append(int(r["n_star"]))

    print(f"\n{'='*100}")
    print(f"ORACLE VARIANCE ANALYSIS  (eps={eps}, batch={BATCH}, B={MAX_SHOTS})")
    print(f"{'='*100}")
    header = (f"{'trace':<24}{'cache':>7}{'median':>8}{'mean':>8}{'std':>7}"
              f"{'min':>7}{'max':>7}{'MAD':>7}{'|cache-med|':>12}{'K':>4}")
    print(header)
    print("-" * len(header))

    mads, cache_gaps, stds = [], [], []
    small_mads, big_mads = [], []
    for (alg, size, backend) in PAPER_TRACES:
        vals = per_triplet.get((alg, size, backend))
        if not vals:
            continue
        med = statistics.median(vals)
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        mad = statistics.mean(abs(v - med) for v in vals)
        c = cached.get((alg, size, backend))
        gap = abs(c - med) if c is not None else float("nan")
        mads.append(mad)
        stds.append(std)
        (small_mads if size < 10 else big_mads).append(mad)
        if c is not None:
            cache_gaps.append(gap)
        cstr = f"{c}" if c is not None else "--"
        print(f"{alg}_{size}_{backend:<14}{cstr:>7}{med:>8.0f}{mean:>8.0f}"
              f"{std:>7.0f}{min(vals):>7}{max(vals):>7}{mad:>7.0f}"
              f"{gap:>12.0f}{len(vals):>4}")

    if not mads:
        print("No data yet.")
        return

    floor = statistics.mean(mads)
    print("-" * len(header))
    print(f"\nAGGREGATE over {len(mads)} traces:")
    print(f"  MAE floor (mean per-trace MAD around median): {floor:>8,.0f} shots"
          f"  ({floor/MAX_SHOTS*100:.1f}% of budget)")
    if small_mads:
        print(f"    - small circuits (n<10): {statistics.mean(small_mads):>8,.0f}")
    if big_mads:
        print(f"    - big circuits  (n>=10): {statistics.mean(big_mads):>8,.0f}")
    print(f"  Mean per-trace std of n*:                     "
          f"{statistics.mean(stds):>8,.0f} shots")
    if cache_gaps:
        print(f"  Mean |cached(sequential) - median(random)|:   "
              f"{statistics.mean(cache_gaps):>8,.0f} shots"
              f"   <- systematic label bias")
    print(f"\n  For reference (paper, delta=0.10, same 36 traces):")
    print(f"    DRL agent MAE : {DRL_MAE:>6,}")
    print(f"    Inc-TVD  MAE  : {INC_TVD_MAE:>6,}")
    print(f"\n  Interpretation: no method scored against the FIXED cached label"
          f"\n  can be expected to beat ~the floor above; the gap between a"
          f"\n  method's MAE and the floor is the part the model can still fix.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eps", type=float, default=0.1,
                    help="oracle tolerance delta (default 0.1, as in the paper)")
    ap.add_argument("--n-seeds", type=int, default=20,
                    help="independent random realizations per trace (default 20)")
    ap.add_argument("--out", default=None, help="output CSV path")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip computation, just analyze the existing CSV")
    args = ap.parse_args()

    out_csv = args.out or os.path.join(HERE, f"oracle_variance_eps{args.eps:g}.csv")
    if not args.analyze_only:
        run(args.eps, args.n_seeds, out_csv)
    analyze(args.eps, out_csv)


if __name__ == "__main__":
    main()
