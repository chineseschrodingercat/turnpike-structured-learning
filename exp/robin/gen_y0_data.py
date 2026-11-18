#!/usr/bin/env python3
"""Multi-horizon dataset with random initial states y0 (1D Robin, nx = 160, dt = 1/320, T_ref = 1 units).

mu = (w, c_rel, log10 alpha, log10 beta[, gamma_eff]); gamma_eff in [0, gamma_max] (gamma_max = 0: linear problem).
y0 = s y_bar(mu) + sum_k c_k cos((k - 1/2) pi x) (y0_family.sample), stored as nodal values, together with
z = 16 cosine coefficients of y0 - y_bar (the network input). Train/val: T = 1, 2; test: T = 1, 2, 4, 8.
Semilinear solves start from the LQ solution with the same y0.
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

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators  # noqa: E402
from semilinear1d import build, cost, solve_kkt, static_ocp_sparse  # noqa: E402
from y0_family import Projector, sample  # noqa: E402

T_LIST = (1, 2, 4, 8)
RANGES = {"w": (0.1, 0.5), "c_rel": (0.0, 1.0), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0)}


def geom(row):
    w, c_rel, la, lb = row[:4]
    c = w / 2 + c_rel * (1 - w)
    return dict(alpha=10.0**la, beta=10.0**lb, xa=c - w / 2, xb=c + w / 2)


def solve_one(args):
    row, horizons, seed, i = args
    g = geom(row)
    gamma = float(row[4]) if len(row) > 4 else 0.0
    cfg1 = RobinConfig(nx=160, Nt=320, **g)
    x = np.arange(cfg1.nx) * cfg1.h
    st = static_ocp_sparse(build(cfg1, gamma, 1.0))
    y0, s, c = sample(np.random.default_rng([seed, i]), x, st["y_bar"])
    proj = Projector(x, assemble_operators(cfg1).M)
    out = {"u_bar": st["u_bar"], "lam1": st["lam1_lin"], "y0": y0, "s": s, "c": c, "z": proj(y0 - st["y_bar"]), "y_bar": st["y_bar"]}
    for T in horizons:
        cfg = RobinConfig(nx=160, Nt=320 * T, T=float(T), **g)
        t0 = time.perf_counter()
        init = None
        if gamma > 0:
            init = solve_kkt(build(cfg, 0.0, 1.0, y0))[:3]
        sm = build(cfg, gamma, 1.0, y0)
        Y, P, U, info = solve_kkt(sm, init=init)
        out[T] = U
        out[f"J_{T}"], out[f"iters_{T}"], out[f"res_{T}"] = cost(sm, Y, U), info["newton_iters"], info["res_rel"]
        out[f"secs_{T}"] = time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--gamma-max", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--y0-per-mu", type=int, default=1, help="initial states per parameter sample (n must be a multiple)")
    ap.add_argument("--all-train", action="store_true", help="extra training pool: every sample in train, T = 1 only")
    args = ap.parse_args()
    cols = list(RANGES) + (["gamma_eff"] if args.gamma_max > 0 else [])
    lo = np.array([RANGES[k][0] for k in RANGES] + ([0.0] if args.gamma_max > 0 else []))
    hi = np.array([RANGES[k][1] for k in RANGES] + ([args.gamma_max] if args.gamma_max > 0 else []))
    X = lo + qmc.LatinHypercube(d=len(cols), seed=args.seed).random(args.n // args.y0_per_mu) * (hi - lo)
    X = np.repeat(X, args.y0_per_mu, axis=0)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = (args.n, 0) if args.all_train else (int(0.8 * args.n), int(0.1 * args.n))
    tr, va, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr : n_tr + n_va]), np.sort(perm[n_tr + n_va :])
    is_test = np.zeros(args.n, bool)
    is_test[te] = True
    order = np.argsort(~is_test)
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(solve_one, [(X[i], T_LIST if is_test[i] else ((1,) if args.all_train else (1, 2)), args.seed, int(i))
                                   for i in order], chunksize=1)
    rows = [None] * args.n
    for k, i in enumerate(order):
        rows[i] = res[k]
    pay = {}
    for T in T_LIST:
        U = np.full((args.n, 320 * T), np.nan)
        for i, r in enumerate(rows):
            if T in r:
                U[i] = r[T]
        pay[f"U_T{T}"] = U
        for key in ("J", "iters", "secs", "res"):
            pay[f"{key}_T{T}"] = np.array([r.get(f"{key}_{T}", np.nan) for r in rows], dtype=float)
    for key in ("u_bar", "lam1", "s"):
        pay[key] = np.array([r[key] for r in rows])
    for key in ("y0", "c", "z", "y_bar"):
        pay[key] = np.stack([r[key] for r in rows])
    gs = [geom(r) for r in X]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, Z=pay["z"], xa=np.array([g["xa"] for g in gs]), xb=np.array([g["xb"] for g in gs]),
        alpha=np.array([g["alpha"] for g in gs]), beta=np.array([g["beta"] for g in gs]),
        gamma=X[:, 4] if args.gamma_max > 0 else np.zeros(args.n), T_ref=np.ones(args.n),
        train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, T_list=np.array(T_LIST),
        metadata_json=json.dumps({"input_columns_X": cols, "ranges": RANGES, "gamma_max": args.gamma_max, "seed": args.seed,
                                  "y0": "s*y_bar + sum_{k<=12} c_k cos((k-1/2) pi x), s~U(0,1.5), c_k~N(0,(0.4/k)^2)",
                                  "Z": "16 cosine coefficients (M-projection) of y0 - y_bar"}),
        **pay)
    Ts = (1,) if args.all_train else T_LIST
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; iters T=1 median {np.nanmedian(pay['iters_T1']):.0f} max "
          f"{np.nanmax(pay['iters_T1']):.0f}; max rel residual {max(np.nanmax(pay[f'res_T{T}']) for T in Ts):.1e}; "
          f"exact seconds T={Ts[-1]} median {np.nanmedian(pay[f'secs_T{Ts[-1]}']):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
