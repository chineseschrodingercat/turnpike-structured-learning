#!/usr/bin/env python3
"""Competing multi-region targets, multi-input control (nh = 100, n = 9900, m = 99).

mu = (xa_1, xa_2, xa_3, tau_1, tau_2, tau_3, log10 alpha, log10 beta); regions in fixed y-bands, width 0.25 in x.
Train/val solved at T = 1, 2; test at T = 1, 2, 4, 8 (matrix-free PCG, tol 1e-10). Also stores the total static
optimum u_bar(s), the per-region static optima u_bar_k(s) (tau-linear decomposition), lambda_1(beta), PCG stats.
--check: tiny-grid validation (PCG vs KKT, linearity in tau, plateau decomposition).
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
from robin2d import lambda1  # noqa: E402
from robin2d_mi import pcg_solve, static_optimum_mi  # noqa: E402
from robin2d_multi import CfgMulti, assemble_multi, solve_kkt_multi, static_optima_per_region  # noqa: E402

NH, M = 100, 99
COLS = ["xa1", "xa2", "xa3", "tau1", "tau2", "tau3", "log10_alpha", "log10_beta"]
RANGES = {"xa1": (0.0, 0.6), "xa2": (0.0, 0.6), "xa3": (0.0, 0.6), "tau1": (0.0, 2.0), "tau2": (0.0, 2.0), "tau3": (0.0, 2.0),
          "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0)}


def cfg_of(row, T, nh=NH):
    return CfgMulti(nh=nh, alpha=10.0 ** row[6], beta=10.0 ** row[7], xas=tuple(row[:3]), taus=tuple(row[3:6]), T=float(T), Nt=int(320 * T))


def solve_one(args):
    row, horizons = args
    out = {"lam1": lambda1(NH, 10.0 ** row[7])}
    for T in horizons:
        u, info = pcg_solve(assemble_multi(cfg_of(row, T), M), float(T))
        out[T] = u.astype(np.float32)
        out[f"iters_{T}"], out[f"secs_{T}"] = info["iters"], info["seconds"]
    mi = assemble_multi(cfg_of(row, 1), M)
    out["u_bar"] = static_optimum_mi(mi)
    out["u_bar_k"] = static_optima_per_region(mi)
    return out


def check():
    row = np.array([0.05, 0.3, 0.55, 1.6, 0.2, 1.1, np.log10(4e-3), 0.2])
    nh, m = 12, 11
    u_kkt, res = solve_kkt_multi(cfg_of(row, 1, nh), m)
    u_pcg, info = pcg_solve(assemble_multi(cfg_of(row, 1, nh), m), 1.0, tol=1e-12)
    parts = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        parts.append(pcg_solve(assemble_multi(cfg_of(row, 1, nh), m, taus=e), 1.0, tol=1e-12)[0])
    lin = sum(row[3 + k] * parts[k] for k in range(3))
    mi = assemble_multi(cfg_of(row, 1, nh), m)
    ub, ubk = static_optimum_mi(mi), static_optima_per_region(mi)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(json.dumps({"pcg_vs_kkt": rel(u_pcg, u_kkt), "kkt_rel_res": res, "pcg_iters": info["iters"],
                      "linearity_in_tau": rel(lin, u_pcg), "plateau_decomposition": rel(row[3:6] @ ubk, ub)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260707)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
    tr, va, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr : n_tr + n_va]), np.sort(perm[n_tr + n_va :])
    is_test = np.zeros(args.n, bool)
    is_test[te] = True
    order = np.argsort(~is_test)
    jobs = [(X[i], (1, 2, 4, 8) if is_test[i] else (1, 2)) for i in order]
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(solve_one, jobs, chunksize=1)
    rows = [None] * args.n
    for k, i in enumerate(order):
        rows[i] = res[k]
    pay = {"U_T1": np.stack([r[1] for r in rows]), "U_T2": np.stack([r[2] for r in rows]),
           "U_T4_test": np.stack([rows[i][4] for i in te]), "U_T8_test": np.stack([rows[i][8] for i in te]),
           "u_bar": np.stack([r["u_bar"] for r in rows]), "u_bar_k": np.stack([r["u_bar_k"] for r in rows]),
           "lam1": np.array([r["lam1"] for r in rows]), "taus": X[:, 3:6],
           **{f"pcg_iters_T{T}": np.array([r.get(f"iters_{T}", -1) for r in rows]) for T in (1, 2, 4, 8)},
           **{f"pcg_secs_T{T}": np.array([r.get(f"secs_{T}", np.nan) for r in rows]) for T in (1, 2, 4, 8)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=M, nh=NH,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "bands": [list(b) for b in
                                                  ((0.05, 0.30), (0.375, 0.625), (0.70, 0.95))], "wx": 0.25, "seed": args.seed}), **pay)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; PCG iters T=1 median {np.median(pay['pcg_iters_T1']):.0f} max "
          f"{pay['pcg_iters_T1'].max()}; PCG seconds T=8 median {np.nanmedian(pay['pcg_secs_T8']):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
