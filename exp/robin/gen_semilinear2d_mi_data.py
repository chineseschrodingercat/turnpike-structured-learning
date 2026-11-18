#!/usr/bin/env python3
"""Dataset: 2D semilinear problem with distributed edge control u(s, t) (m = 99, n = 9900) and the edge-strip target
y_d(y) = 1 + sum_{k<=6} tau_k sin(k pi y) on [0, w] x [0.05, 0.95] (semilinear2d_mi.py).

mu = (w, log10 alpha, log10 beta, tau_1..tau_6, log10 gamma_eff); train/val solved at T = 1, 2, test at T = 1, 2, 4, 8.
Stores, besides U_T* (float32, (N, m)): u_bar (static nonlinear optimum), u_bar_lq (static LQ optimum), u_bar_k
(per-component LQ static optima, the tau-linear plateau of section 18b), lam1 (linearized floor at the nonlinear
turnpike), the LQ controls of the test samples (U_lq_T*_test), Newton/CG statistics and timings.
Keys follow gen_2d_strip_data.py, so multi_turnpike_surrogate.py reads the file directly (geo_cols include log gamma).
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
from robin2d_mi import static_optimum_mi  # noqa: E402
from robin2d_multi import CfgStrip, static_optima_per_region  # noqa: E402
from semilinear2d_mi import build, lq_control, solve, static_ocp  # noqa: E402

NH, M, T_LIST = 100, 99, (1, 2, 4, 8)
COLS = ["w", "log10_alpha", "log10_beta"] + [f"tau{k}" for k in range(1, 7)] + ["log10_gamma_eff"]


def cfg_of(row, T):
    return CfgStrip(nh=NH, alpha=10.0 ** row[1], beta=10.0 ** row[2], w=float(row[0]), coefs=(1.0,) + tuple(float(v) for v in row[3:9]),
                    T=float(T), Nt=int(320 * T))


def solve_one(args):
    row, horizons, gtol = args
    gamma = float(10.0 ** row[9])
    out = {}
    for T in horizons:
        cfg = cfg_of(row, T)
        t0 = time.perf_counter()
        Clq = lq_control(cfg, M)
        t_lq = time.perf_counter() - t0
        C, info = solve(build(cfg, M, gamma), Clq, gtol=gtol)
        out[T], out[f"lq_{T}"] = C.astype(np.float32), Clq.astype(np.float32)
        out[f"secs_{T}"], out[f"lq_secs_{T}"] = info["seconds"] + t_lq, t_lq
        for k in ("J", "newton_iters", "cg_iters", "grad_rel"):
            out[f"{k}_{T}"] = info[k]
    s1 = build(cfg_of(row, 1), M, gamma)
    t0 = time.perf_counter()
    st = static_ocp(s1)
    out["static_seconds"] = time.perf_counter() - t0
    out["u_bar"], out["lam1"] = st["u_bar"], st["lam1_lin"]
    out["u_bar_lq"], out["u_bar_k"] = static_optimum_mi(s1.mi), static_optima_per_region(s1.mi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--log-alpha", type=float, nargs=2, default=(-3.0, -1.0))
    ap.add_argument("--gamma", type=float, nargs=2, default=(3.0, 300.0))
    ap.add_argument("--gtol", type=float, default=1e-8)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()
    ranges = {"w": (0.05, 0.3), "log10_alpha": tuple(args.log_alpha), "log10_beta": (-1.0, 1.0),
              **{f"tau{k}": (-1.0, 1.0) for k in range(1, 7)}, "log10_gamma_eff": (np.log10(args.gamma[0]), np.log10(args.gamma[1]))}
    lo = np.array([ranges[c][0] for c in COLS])
    hi = np.array([ranges[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)
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
        res = pool.map(solve_one, [(X[i], T_LIST if is_test[i] else (1, 2), args.gtol) for i in order], chunksize=1)
    rows = [None] * args.n
    for k, i in enumerate(order):
        rows[i] = res[k]
    pay = {"U_T1": np.stack([r[1] for r in rows]), "U_T2": np.stack([r[2] for r in rows]),
           "U_T4_test": np.stack([rows[i][4] for i in te]), "U_T8_test": np.stack([rows[i][8] for i in te]),
           **{f"U_lq_T{T}_test": np.stack([rows[i][f"lq_{T}"] for i in te]) for T in T_LIST},
           "u_bar": np.stack([r["u_bar"] for r in rows]), "u_bar_lq": np.stack([r["u_bar_lq"] for r in rows]),
           "u_bar_k": np.stack([r["u_bar_k"] for r in rows]), "lam1": np.array([r["lam1"] for r in rows]),
           "static_seconds": np.array([r["static_seconds"] for r in rows]),
           "coefs": np.concatenate([np.ones((args.n, 1)), X[:, 3:9]], axis=1), "geo_cols": np.array([0, 1, 2, 9]),
           "gamma": 10.0 ** X[:, 9]}
    for T in T_LIST:
        for k in ("J", "newton_iters", "cg_iters", "grad_rel", "secs", "lq_secs"):
            pay[f"{k}_T{T}"] = np.array([r.get(f"{k}_{T}", np.nan) for r in rows], dtype=float)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=M, nh=NH,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": ranges, "omega": "[0,w] x [0.05,0.95]",
                                                  "target": "1 + sum_k tau_k sin(k pi y)", "pde": "y_t - Laplace y + gamma y^3 = 0",
                                                  "seed": args.seed}), **pay)
    nl = np.linalg.norm(pay["u_bar"] - pay["u_bar_lq"], axis=1) / np.linalg.norm(pay["u_bar_lq"], axis=1)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; Newton T=1 median {np.nanmedian(pay['newton_iters_T1']):.0f} max "
          f"{np.nanmax(pay['newton_iters_T1']):.0f}; max grad_rel {max(np.nanmax(pay[f'grad_rel_T{T}']) for T in T_LIST):.1e}; "
          f"exact s T=8 median {np.nanmedian(pay['secs_T8']):.0f}; static median {np.median(pay['static_seconds']):.2f}s; "
          f"|u_bar - u_bar_lq|/|u_bar_lq| median {np.median(nl):.3f} p90 {np.percentile(nl, 90):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
