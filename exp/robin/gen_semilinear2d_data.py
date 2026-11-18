#!/usr/bin/env python3
"""2D semilinear multi-horizon dataset (nh = 100, n = 9900, dt = 1/320; semilinear2d.py), y_d = 1 on the rectangle.

mu = (w, c_rel, c_y, log10 alpha, log10 beta, gamma_eff), the first five as in gen_2d_turnpike_data.py, gamma_eff in
[0, gamma_max]. Train/val: T = 1, 2; test: T = 1, 2, 4, 8 (U_T4, U_T8 NaN outside the test split). Exact controls by
reduced Newton-CG warm-started from the exact LQ control; u_bar = static nonlinear optimum, lam1 = linearized floor.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_2d_turnpike_data import COLS as COLS5, RANGES as RANGES5, geometry  # noqa: E402
from robin2d import Cfg2D  # noqa: E402
from semilinear2d import build, lq_control, solve, static_ocp  # noqa: E402

NH, T_LIST = 100, (1, 2, 4, 8)


def solve_one(args):
    row, horizons, gtol, glog = args
    g = geometry(row[:5])
    gamma = float(10.0 ** row[5]) if glog else float(row[5])
    out = dict(g)
    lq = lq_control(Cfg2D(nh=NH, T=float(max(horizons)), Nt=320 * max(horizons), **g), horizons)
    for T in horizons:
        u, info = solve(build(Cfg2D(nh=NH, T=float(T), Nt=320 * T, **g), gamma), lq[T], gtol=gtol)
        out[T] = u
        for k in ("J", "newton_iters", "cg_iters", "grad_rel", "seconds"):
            out[f"{k}_{T}"] = info[k]
    t0 = time.perf_counter()
    st = static_ocp(build(Cfg2D(nh=NH, **g), gamma))
    out["u_bar"], out["lam1"], out["static_seconds"] = st["u_bar"], st["lam1_lin"], time.perf_counter() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--gamma-max", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--gtol", type=float, default=1e-8, help="Newton stop: ||g|| / ||g(0)||")
    ap.add_argument("--gamma-log", type=float, nargs=2, default=None, metavar=("GMIN", "GMAX"),
                    help="sample log10 gamma_eff uniformly in [log10 GMIN, log10 GMAX]; X then holds log10 gamma_eff")
    ap.add_argument("--log-alpha", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="override the log10 alpha range")
    args = ap.parse_args()
    if args.log_alpha is not None:
        RANGES5["log10_alpha"] = tuple(args.log_alpha)
    glog = args.gamma_log is not None
    cols = COLS5 + (["log10_gamma_eff"] if glog else ["gamma_eff"])
    g_lo, g_hi = (np.log10(args.gamma_log[0]), np.log10(args.gamma_log[1])) if glog else (0.0, args.gamma_max)
    lo = np.array([RANGES5[c][0] for c in COLS5] + [g_lo])
    hi = np.array([RANGES5[c][1] for c in COLS5] + [g_hi])
    X = lo + qmc.LatinHypercube(d=len(cols), seed=args.seed).random(args.n) * (hi - lo)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    te = np.sort(perm[: args.n_test])
    va = np.sort(perm[args.n_test : args.n_test + args.n_val])
    tr = np.sort(perm[args.n_test + args.n_val :])
    is_test = np.zeros(args.n, bool)
    is_test[te] = True
    order = np.argsort(~is_test)
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(solve_one, [(X[i], T_LIST if is_test[i] else (1, 2), args.gtol, glog) for i in order], chunksize=1)
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
        for k in ("J", "newton_iters", "cg_iters", "grad_rel", "seconds"):
            pay[f"{k}_T{T}"] = np.array([r.get(f"{k}_{T}", np.nan) for r in rows], dtype=float)
    for k in ("u_bar", "lam1", "static_seconds", "xa", "xb", "ya", "yb", "alpha", "beta"):
        pay[k] = np.array([r[k] for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, gamma=10.0 ** X[:, 5] if glog else X[:, 5], T_ref=np.ones(args.n), train_idx=tr, val_idx=va,
                        test_idx=te, dt=1.0 / 320,
                        T_list=np.array(T_LIST), metadata_json=json.dumps({"input_columns": cols, "ranges": {**RANGES5, cols[5]: [g_lo, g_hi]},
                                                                           "nh": NH, "n": NH * (NH - 1), "seed": args.seed}), **pay)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; Newton T=1 median {np.nanmedian(pay['newton_iters_T1']):.0f} max "
          f"{np.nanmax(pay['newton_iters_T1']):.0f}; max grad_rel {max(np.nanmax(pay[f'grad_rel_T{T}']) for T in T_LIST):.1e}; "
          f"exact seconds T=8 median {np.nanmedian(pay['seconds_T8']):.0f}; static median {np.median(pay['static_seconds']):.2f}s -> {args.out}")


if __name__ == "__main__":
    main()
