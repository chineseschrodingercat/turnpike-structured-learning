#!/usr/bin/env python3
"""Dataset for the moving hot-spot target (moving2d.py): n = 9900, m = 99, dt = 1/320.

mu = (w, log10 alpha, log10 beta, a, omega, log10 sigma, phi0). Per sample: harmonics of the exact discrete periodic
turnpike (u_n, n = 0..N_h, zero-padded to NH_MAX), exact controls by PCG warm-started from u_hat: train/val at
T = 1, 2, test at T = 1, 2, 4, 8 (float32, (N, m)); PCG iterations / seconds, harmonic-balance seconds, lambda_1(beta).
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
from moving2d import CfgMove, assemble, eval_periodic, pcg_solve, periodic_turnpike, target_loads  # noqa: E402
from robin2d import lambda1  # noqa: E402

COLS = ["w", "log10_alpha", "log10_beta", "a", "omega", "log10_sigma", "phi0"]
RANGES = {"w": (0.05, 0.3), "log10_alpha": (-3.5, -2.0), "log10_beta": (-1.0, 1.0), "a": (0.1, 0.35), "omega": (np.pi, 4 * np.pi),
          "log10_sigma": (np.log10(0.04), np.log10(0.12)), "phi0": (0.0, 2 * np.pi)}
T_LIST, NH_MAX = (1, 2, 4, 8), 80


def cfg_of(row, T):
    return CfgMove(nh=100, m=99, w=float(row[0]), alpha=float(10 ** row[1]), beta=float(10 ** row[2]), a=float(row[3]),
                   omega=float(row[4]), sigma=float(10 ** row[5]), phi0=float(row[6]), T=float(T), Nt=int(320 * T))


def solve_one(args):
    row, horizons = args
    out = {"lam1": lambda1(100, float(10 ** row[2]))}
    ops1 = assemble(cfg_of(row, 1))
    U, hinfo = periodic_turnpike(ops1, nmax=NH_MAX)
    Upad = np.zeros((NH_MAX + 1, 99), dtype=np.complex64)
    Upad[: U.shape[0]] = U
    out["uhat"], out["n_harm"], out["harm_seconds"] = Upad, hinfo["n_harm"], hinfo["seconds"]
    for T in horizons:
        cf = cfg_of(row, T)
        ops = assemble(cf)
        GY, _ = target_loads(ops)
        t = np.arange(1, cf.Nt + 1) * cf.dt
        C, info = pcg_solve(ops, GY, C0=eval_periodic(U, cf.omega, cf.phi0, t))
        out[T], out[f"iters_{T}"], out[f"secs_{T}"] = C.astype(np.float32), info["iters"], info["seconds"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=750)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
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
        res = pool.map(solve_one, [(X[i], T_LIST if is_test[i] else (1, 2)) for i in order], chunksize=1)
    rows = [None] * args.n
    for k, i in enumerate(order):
        rows[i] = res[k]
    pay = {"U_T1": np.stack([r[1] for r in rows]), "U_T2": np.stack([r[2] for r in rows]),
           "U_T4_test": np.stack([rows[i][4] for i in te]), "U_T8_test": np.stack([rows[i][8] for i in te]),
           "uhat": np.stack([r["uhat"] for r in rows]), "n_harm": np.array([r["n_harm"] for r in rows]),
           "harm_seconds": np.array([r["harm_seconds"] for r in rows]), "lam1": np.array([r["lam1"] for r in rows])}
    for T in T_LIST:
        pay[f"pcg_iters_T{T}"] = np.array([r.get(f"iters_{T}", np.nan) for r in rows], dtype=float)
        pay[f"pcg_secs_T{T}"] = np.array([r.get(f"secs_{T}", np.nan) for r in rows], dtype=float)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=99, nh=100,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "omega_strip": "[0,w] x [0.05,0.95]",
                                                  "target": "exp(-(y - 0.5 - a sin(omega t + phi0))^2 / (2 sigma^2))", "seed": args.seed}), **pay)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; harmonics median {np.median(pay['n_harm']):.0f} max {pay['n_harm'].max()}; "
          f"PCG iters T=1 median {np.nanmedian(pay['pcg_iters_T1']):.0f}, T=8 median {np.nanmedian(pay['pcg_iters_T8']):.0f}; "
          f"PCG s T=8 median {np.nanmedian(pay['pcg_secs_T8']):.0f}; harmonic balance median {np.median(pay['harm_seconds']):.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
