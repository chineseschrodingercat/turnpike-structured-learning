#!/usr/bin/env python3
"""Multi-horizon dataset for the semilinear problem (semilinear1d.py), nx = 160, dt = 1/320.

Raw parameters (w, c_rel, T_ref, log10 alpha, log10 beta, gamma), gamma in [0, 20]. The discrete problem is exactly
invariant under y -> T_ref z, u -> T_ref v with gamma_eff = gamma T_ref^2, so u / u_bar depends only on
X = (w, c_rel, log10 alpha, log10 beta, gamma_eff); X is what every model sees (X_raw is kept too).
Train/val: T = 1, 2; test: T = 1, 2, 4, 8 (U_T4, U_T8 are NaN outside the test split). Newton starts from the LQ
(gamma = 0) solution. Stores u_bar (static nonlinear optimum), lam1 (linearized rate floor at the turnpike),
the optimal cost J*_T, and Newton iterations / seconds per solve.
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
from robin_solver import RobinConfig  # noqa: E402
from semilinear1d import build, cost, solve_kkt, static_ocp  # noqa: E402

COLS_RAW = ["w", "c_rel", "T_ref", "log10_alpha", "log10_beta", "gamma"]
RANGES = {"w": (0.1, 0.5), "c_rel": (0.0, 1.0), "T_ref": (0.5, 2.0), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0),
          "gamma": (0.0, 20.0)}
T_LIST = (1, 2, 4, 8)


def geom(row):
    w, c_rel, T_ref, la, lb, gamma = row
    c = w / 2 + c_rel * (1 - w)
    return dict(alpha=10.0**la, beta=10.0**lb, xa=c - w / 2, xb=c + w / 2), float(T_ref), float(gamma)


def solve_one(args):
    row, horizons = args
    g, T_ref, gamma = geom(row)
    out = {}
    for T in horizons:
        cfg = RobinConfig(nx=160, Nt=320 * T, T=float(T), **g)
        t0 = time.perf_counter()
        lq = solve_kkt(build(cfg, 0.0, T_ref))
        s = build(cfg, gamma, T_ref)
        Y, P, U, info = solve_kkt(s, init=lq[:3])
        out[T] = U
        out[f"J_{T}"] = cost(s, Y, U)
        out[f"iters_{T}"], out[f"secs_{T}"], out[f"res_{T}"] = info["newton_iters"], time.perf_counter() - t0, info["res_rel"]
    st = static_ocp(build(RobinConfig(nx=160, Nt=320, **g), gamma, T_ref))
    out["u_bar"], out["lam1"] = st["u_bar"], st["lam1_lin"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--all-train", action="store_true", help="extra training pool: every sample in train, horizons 1 and 2 only")
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS_RAW])
    hi = np.array([RANGES[c][1] for c in COLS_RAW])
    Xr = lo + qmc.LatinHypercube(d=len(COLS_RAW), seed=args.seed).random(args.n) * (hi - lo)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = (args.n, 0) if args.all_train else (int(0.7 * args.n), int(0.15 * args.n))
    tr, va, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr : n_tr + n_va]), np.sort(perm[n_tr + n_va :])
    is_test = np.zeros(args.n, bool)
    is_test[te] = True
    order = np.argsort(~is_test)
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(solve_one, [(Xr[i], T_LIST if is_test[i] else (1, 2)) for i in order], chunksize=1)
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
    X = np.column_stack([Xr[:, 0], Xr[:, 1], Xr[:, 3], Xr[:, 4], Xr[:, 5] * Xr[:, 2] ** 2])
    gs = [geom(r)[0] for r in Xr]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, X_raw=Xr, u_bar=np.array([r["u_bar"] for r in rows]), lam1=np.array([r["lam1"] for r in rows]),
        xa=np.array([g["xa"] for g in gs]), xb=np.array([g["xb"] for g in gs]), alpha=np.array([g["alpha"] for g in gs]),
        beta=np.array([g["beta"] for g in gs]), T_ref=Xr[:, 2], gamma=Xr[:, 5], gamma_eff=X[:, 4],
        train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, T_list=np.array(T_LIST),
        metadata_json=json.dumps({"input_columns_X": ["w", "c_rel", "log10_alpha", "log10_beta", "gamma_eff"], "raw_columns": COLS_RAW,
                                  "ranges": RANGES, "pde": "y_t - y_xx + gamma y^3 = 0, Robin control at x=0", "seed": args.seed}),
        **pay)
    Ts = (1, 2) if args.all_train else T_LIST
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; Newton iters T=1 median {np.nanmedian(pay['iters_T1']):.0f} max "
          f"{np.nanmax(pay['iters_T1']):.0f}; max rel residual {max(np.nanmax(pay[f'res_T{T}']) for T in Ts):.1e}; "
          f"exact seconds T={Ts[-1]} median {np.nanmedian(pay[f'secs_T{Ts[-1]}']):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
