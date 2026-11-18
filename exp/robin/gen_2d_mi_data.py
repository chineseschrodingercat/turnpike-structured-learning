#!/usr/bin/env python3
"""Multi-input dataset: 2D problem (nh = 100, n = 9900), distributed edge control with m = 99 inputs.

Same parameters as gen_2d_turnpike_data.py. Exact controls by matrix-free PCG (tol 1e-10).
Train/val samples are solved at T = 1, 2 only (what training uses); test samples at T = 1, 2, 4, 8.
Stores U_T1, U_T2 for all samples, U_T4_test, U_T8_test for the test split (float32, shape (., N, m)),
u_bar(s) (static optimum), lambda_1(beta), PCG iterations and seconds.
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
from gen_2d_turnpike_data import COLS, RANGES, geometry  # noqa: E402
from robin2d import Cfg2D, lambda1  # noqa: E402
from robin2d_mi import assemble_mi, pcg_solve, static_optimum_mi  # noqa: E402

NH, M = 100, 99


def solve_one(args):
    row, horizons = args
    g = geometry(row)
    out = {"lam1": lambda1(NH, g["beta"]), **g}
    for T in horizons:
        mi = assemble_mi(Cfg2D(nh=NH, T=float(T), Nt=320 * T, **g), M)
        u, info = pcg_solve(mi, float(T))
        out[T] = u.astype(np.float32)
        out[f"iters_{T}"], out[f"secs_{T}"] = info["iters"], info["seconds"]
    out["u_bar"] = static_optimum_mi(assemble_mi(Cfg2D(nh=NH, **g), M))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260706)
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
    tr, va, te = np.sort(perm[:n_tr]), np.sort(perm[n_tr : n_tr + n_va]), np.sort(perm[n_tr + n_va :])
    is_test = np.zeros(args.n, bool)
    is_test[te] = True
    jobs = [(X[i], (1, 2, 4, 8) if is_test[i] else (1, 2)) for i in range(args.n)]
    jobs_order = np.argsort(~is_test)  # test (long) jobs first for load balance
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res_sorted = pool.map(solve_one, [jobs[i] for i in jobs_order], chunksize=1)
    rows = [None] * args.n
    for k, i in enumerate(jobs_order):
        rows[i] = res_sorted[k]
    pay = {
        "U_T1": np.stack([r[1] for r in rows]), "U_T2": np.stack([r[2] for r in rows]),
        "U_T4_test": np.stack([rows[i][4] for i in te]), "U_T8_test": np.stack([rows[i][8] for i in te]),
        "u_bar": np.stack([r["u_bar"] for r in rows]), "lam1": np.array([r["lam1"] for r in rows]),
        **{k: np.array([r[k] for r in rows]) for k in ("xa", "xb", "ya", "yb", "alpha", "beta")},
        **{f"pcg_iters_T{T}": np.array([r.get(f"iters_{T}", -1) for r in rows]) for T in (1, 2, 4, 8)},
        **{f"pcg_secs_T{T}": np.array([r.get(f"secs_{T}", np.nan) for r in rows]) for T in (1, 2, 4, 8)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=M, nh=NH,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "nh": NH, "m": M,
                                                  "control": "u(s,t) on x=0, m hat functions", "seed": args.seed}), **pay)
    print(f"{args.n} samples in {time.perf_counter() - t0:.0f}s; median PCG iters T=1 {np.median(pay['pcg_iters_T1']):.0f}, "
          f"median PCG seconds T=8 (test) {np.nanmedian(pay['pcg_secs_T8']):.1f} -> {args.out}")


if __name__ == "__main__":
    main()
