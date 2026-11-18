#!/usr/bin/env python3
"""2D multi-horizon dataset (n = nh (nh - 1), default nh = 100 -> n = 9900), constant target y_d = 1 on omega.

mu = (w, c_rel, c_y, log10 alpha, log10 beta): omega = [xa, xb] x [c_y - 0.2, c_y + 0.2] with
xa = c_rel (1 - w), xb = xa + w. Horizons T in {1, 2, 4, 8}, dt = 1/320. Stores U_T, u_bar
(static optimum), lambda_1(beta) (separable), per-sample exact-solve timings.
Keys follow turnpike_data.npz so turnpike_analysis.py and baselines_multihorizon.py work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D, assemble, horizon_family, lambda1, static_optimum  # noqa: E402

T_LIST = (1, 2, 4, 8)
COLS = ["w", "c_rel", "c_y", "log10_alpha", "log10_beta"]
RANGES = {"w": (0.1, 0.5), "c_rel": (0.0, 1.0), "c_y": (0.25, 0.75), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0)}
NH = 100


def geometry(row):
    w, c_rel, cy, la, lb = row
    xa = c_rel * (1 - w)
    return {"xa": xa, "xb": xa + w, "ya": cy - 0.2, "yb": cy + 0.2, "alpha": 10.0**la, "beta": 10.0**lb}


def solve_one(row):
    g = geometry(row)
    t0 = time.perf_counter()
    ops = assemble(Cfg2D(nh=NH, T=8.0, Nt=320 * 8, **g))
    timing = {}
    fam = horizon_family(ops, T_LIST, timing=timing)
    timing["total_exact_all_T"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    ub = static_optimum(ops)
    timing["static_optimum"] = time.perf_counter() - t1
    return {**{T: fam[T] for T in T_LIST}, "u_bar": ub, "lam1": lambda1(NH, g["beta"]), "timing": timing, **g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        rows = pool.map(solve_one, list(X), chunksize=1)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
    tim = {k: np.array([r["timing"][k] for r in rows]) for k in rows[0]["timing"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, **{f"U_T{T}": np.array([r[T] for r in rows]) for T in T_LIST},
        u_bar=np.array([r["u_bar"] for r in rows]), lam1=np.array([r["lam1"] for r in rows]),
        xa=np.array([r["xa"] for r in rows]), xb=np.array([r["xb"] for r in rows]),
        ya=np.array([r["ya"] for r in rows]), yb=np.array([r["yb"] for r in rows]),
        alpha=np.array([r["alpha"] for r in rows]), beta=np.array([r["beta"] for r in rows]),
        **{f"time_{k}": v for k, v in tim.items()},
        train_idx=np.sort(perm[:n_tr]), val_idx=np.sort(perm[n_tr : n_tr + n_va]), test_idx=np.sort(perm[n_tr + n_va :]),
        dt=1.0 / 320, T_list=np.array(T_LIST),
        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "nh": NH, "n": NH * (NH - 1), "target": "y_d = 1",
                                  "control": "Robin on x=0, uniform along the edge", "seed": args.seed}),
    )
    print(f"{args.n} samples (n = {NH * (NH - 1)}) in {time.perf_counter() - t0:.0f}s; median exact time per mu (all 4 horizons) "
          f"{np.median(tim['total_exact_all_T']):.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
