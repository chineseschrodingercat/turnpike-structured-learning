#!/usr/bin/env python3
"""Dataset: 2D semilinear problem with a periodically moving hot-spot target (semilinear_moving.py), n = 9900, m = 99.

mu = (w, log10 alpha, log10 beta, a, omega, log10 sigma, phi0, log10 gamma_eff). Train/val at T = 1, 2; test at
T = 1, 2, 4, 8. Per sample: exact nonlinear controls (Newton-CG warm-started from the exact LQ control), the LQ
controls of the test samples, the harmonics of the LINEAR periodic turnpike (harmonic balance; zero-padded),
lambda_1(beta) (open-loop floor), Newton/CG statistics and timings.
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
from moving2d import CfgMove  # noqa: E402
from robin2d import lambda1  # noqa: E402
from semilinear_moving import build, lq_control, solve  # noqa: E402

COLS = ["w", "log10_alpha", "log10_beta", "a", "omega", "log10_sigma", "phi0", "log10_gamma_eff"]
T_LIST, NH_MAX = (1, 2, 4, 8), 80


def cfg_of(row, T):
    return CfgMove(nh=100, m=99, w=float(row[0]), alpha=float(10 ** row[1]), beta=float(10 ** row[2]), a=float(row[3]),
                   omega=float(row[4]), sigma=float(10 ** row[5]), phi0=float(row[6]), T=float(T), Nt=int(320 * T))


def solve_one(args):
    row, horizons, gtol = args
    gamma = float(10 ** row[7])
    out = {"lam1": lambda1(100, float(10 ** row[2]))}
    for T in horizons:
        cf = cfg_of(row, T)
        t0 = time.perf_counter()
        Clq, Ulin, _ = lq_control(cf)
        t_lq = time.perf_counter() - t0
        C, info = solve(build(cf, gamma), Clq, gtol=gtol)
        out[T], out[f"lq_{T}"] = C.astype(np.float32), Clq.astype(np.float32)
        out[f"secs_{T}"], out[f"lq_secs_{T}"] = info["seconds"] + t_lq, t_lq
        for k in ("newton_iters", "cg_iters", "grad_rel"):
            out[f"{k}_{T}"] = info[k]
        if T == horizons[0]:
            Upad = np.zeros((NH_MAX + 1, 99), dtype=np.complex64)
            Upad[: min(Ulin.shape[0], NH_MAX + 1)] = Ulin[: NH_MAX + 1]
            out["uhat_lin"], out["n_harm"] = Upad, Ulin.shape[0] - 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--log-alpha", type=float, nargs=2, default=(-3.5, -2.0))
    ap.add_argument("--gamma", type=float, nargs=2, default=(3.0, 300.0))
    ap.add_argument("--gtol", type=float, default=1e-8)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261021)
    args = ap.parse_args()
    ranges = {"w": (0.05, 0.3), "log10_alpha": tuple(args.log_alpha), "log10_beta": (-1.0, 1.0), "a": (0.1, 0.35), "omega": (np.pi, 4 * np.pi),
              "log10_sigma": (np.log10(0.04), np.log10(0.12)), "phi0": (0.0, 2 * np.pi),
              "log10_gamma_eff": (np.log10(args.gamma[0]), np.log10(args.gamma[1]))}
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
           "U_lq_T1": np.stack([r["lq_1"] for r in rows]),
           "uhat_lin": np.stack([r["uhat_lin"] for r in rows]), "n_harm": np.array([r["n_harm"] for r in rows]),
           "lam1": np.array([r["lam1"] for r in rows]), "gamma": 10.0 ** X[:, 7]}
    for T in T_LIST:
        for k in ("newton_iters", "cg_iters", "grad_rel", "secs", "lq_secs"):
            pay[f"{k}_T{T}"] = np.array([r.get(f"{k}_{T}", np.nan) for r in rows], dtype=float)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=99, nh=100,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": ranges, "pde": "y_t - Laplace y + gamma y^3 = 0",
                                                  "target": "exp(-(y - 0.5 - a sin(omega t + phi0))^2 / (2 sigma^2))", "seed": args.seed}), **pay)
    nl = np.linalg.norm((pay["U_T1"] - pay["U_lq_T1"]).reshape(args.n, -1), axis=1) / np.linalg.norm(pay["U_lq_T1"].reshape(args.n, -1), axis=1)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; Newton T=1 median {np.nanmedian(pay['newton_iters_T1']):.0f} max "
          f"{np.nanmax(pay['newton_iters_T1']):.0f}; max grad_rel {max(np.nanmax(pay[f'grad_rel_T{T}']) for T in T_LIST):.1e}; "
          f"exact s T=8 median {np.nanmedian(pay['secs_T8']):.0f}; |u - u_lq|/|u_lq| (T=1) median {np.median(nl):.3f} p90 {np.percentile(nl, 90):.3f} -> {args.out}")


if __name__ == "__main__":
    main()
