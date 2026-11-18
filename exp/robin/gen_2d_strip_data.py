#!/usr/bin/env python3
"""Edge-strip target with fine y-variation, multi-input control (nh = 100, n = 9900, m = 99).

omega = [0, w] x [0.05, 0.95] (touches the controlled edge), y_d(y) = 1 + sum_{k=1}^{6} tau_k sin(k pi y).
mu = (w, log10 alpha, log10 beta, tau_1..tau_6). u is linear in coefs = (1, tau_1..tau_6) (K = 7).
Stores per-component static optima u_bar_k (tau-linear plateau) plus the same fields as gen_2d_multi_data.py,
and geo_cols / coefs so multi_turnpike_surrogate.py can read the layout. --check: tiny-grid validation.
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
from robin2d_multi import CfgStrip, assemble_strip, static_optima_per_region  # noqa: E402

NH, M = 100, 99
COLS = ["w", "log10_alpha", "log10_beta"] + [f"tau{k}" for k in range(1, 7)]
RANGES = {"w": (0.05, 0.3), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0), **{f"tau{k}": (-1.0, 1.0) for k in range(1, 7)}}


def coefs_of(row):
    return (1.0,) + tuple(row[3:9])


def cfg_of(row, T, nh=NH):
    return CfgStrip(nh=nh, alpha=10.0 ** row[1], beta=10.0 ** row[2], w=float(row[0]), coefs=coefs_of(row), T=float(T), Nt=int(320 * T))


def solve_one(args):
    row, horizons = args
    out = {"lam1": lambda1(NH, 10.0 ** row[2])}
    for T in horizons:
        u, info = pcg_solve(assemble_strip(cfg_of(row, T), M), float(T))
        out[T] = u.astype(np.float32)
        out[f"iters_{T}"], out[f"secs_{T}"] = info["iters"], info["seconds"]
    mi = assemble_strip(cfg_of(row, 1), M)
    out["u_bar"] = static_optimum_mi(mi)
    out["u_bar_k"] = static_optima_per_region(mi)
    return out


def check():
    from robin2d_mi import solve_kkt_mi  # noqa: F401  (single-region KKT; strip KKT built below)
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    from robin_solver import block_lower

    row = np.array([0.15, np.log10(3e-3), 0.1, 0.5, -0.4, 0.3, 0.8, -0.6, 0.2])
    nh, m = 16, 15
    cfg = cfg_of(row, 1, nh)
    mi = assemble_strip(cfg, m)
    o, N = mi.base, cfg.Nt
    I_t = sp.eye(N, format="csr")
    Ay = block_lower(o.A, (-o.M / cfg.dt).tocsr(), N)
    Bb = -sp.kron(I_t, mi.B, format="csr")
    KKT = sp.bmat([[Ay, None, Bb], [sp.kron(I_t, o.M_obs, format="csr"), Ay.T.tocsr(), None],
                   [None, Bb.T, cfg.alpha * sp.kron(I_t, sp.csr_matrix(mi.Mu), format="csr")]], format="csc")
    rhs = np.concatenate([np.zeros(o.n * N), np.tile(o.G, N), np.zeros(m * N)])
    u_kkt = spla.spsolve(KKT, rhs)[2 * o.n * N :].reshape(N, m)
    u_pcg, info = pcg_solve(mi, 1.0, tol=1e-12)
    ub, ubk = static_optimum_mi(mi), static_optima_per_region(mi)
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    print(json.dumps({"pcg_vs_kkt": rel(u_pcg, u_kkt), "pcg_iters": info["iters"],
                      "plateau_decomposition": rel(np.array(coefs_of(row)) @ ubk, ub)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260708)
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
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(solve_one, [(X[i], (1, 2, 4, 8) if is_test[i] else (1, 2)) for i in order], chunksize=1)
    rows = [None] * args.n
    for k, i in enumerate(order):
        rows[i] = res[k]
    pay = {"U_T1": np.stack([r[1] for r in rows]), "U_T2": np.stack([r[2] for r in rows]),
           "U_T4_test": np.stack([rows[i][4] for i in te]), "U_T8_test": np.stack([rows[i][8] for i in te]),
           "u_bar": np.stack([r["u_bar"] for r in rows]), "u_bar_k": np.stack([r["u_bar_k"] for r in rows]),
           "lam1": np.array([r["lam1"] for r in rows]),
           "coefs": np.concatenate([np.ones((args.n, 1)), X[:, 3:9]], axis=1), "geo_cols": np.array([0, 1, 2]),
           **{f"pcg_iters_T{T}": np.array([r.get(f"iters_{T}", -1) for r in rows]) for T in (1, 2, 4, 8)},
           **{f"pcg_secs_T{T}": np.array([r.get(f"secs_{T}", np.nan) for r in rows]) for T in (1, 2, 4, 8)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=M, nh=NH,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "omega": "[0,w] x [0.05,0.95]",
                                                  "target": "1 + sum_k tau_k sin(k pi y)", "seed": args.seed}), **pay)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; PCG iters T=1 median {np.median(pay['pcg_iters_T1']):.0f} max "
          f"{pay['pcg_iters_T1'].max()}; PCG seconds T=8 median {np.nanmedian(pay['pcg_secs_T8']):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
