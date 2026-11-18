#!/usr/bin/env python3
"""Multi-horizon dataset for the turnpike study (Robin control, constant target y_d = 1).

Parameters mu = (w, c_rel, log10 alpha, log10 beta) on the report's ranges; the
target amplitude T_ref is dropped because u is exactly linear in it (f = y0 = 0).
Every mu is solved for horizons T in {1, 2, 4, 8} with the SAME time step
dt = 1/320, so Nt = 320 T and the layers live on a common physical-time grid.

Horizon family from one impulse response: with R[p, q] = a_p^T M_obs a_q,
    H_T[i, j] = sum_{m=0}^{N_T-1-j} R[m+|j-i|, m],   g_T[j] = sum_{m=0}^{N_T-1-j} a_m^T G,
and u_T = (H_T + alpha I)^{-1} g_T (validated against the monolithic KKT solve
with --check). The turnpike value is the static optimal control
    u_bar = psi^T G / (psi^T M_obs psi + alpha),   K_beta psi = e0.
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
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robin_solver import RobinConfig, assemble_operators, impulse_response, interval_load, solve_kkt  # noqa: E402

T_LIST = (1, 2, 4, 8)
STEPS_PER_UNIT = 320
COLS = ["w", "c_rel", "log10_alpha", "log10_beta"]
RANGES = {"w": (0.1, 0.5), "c_rel": (0.0, 1.0), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0)}
ONE = lambda x, t: np.ones_like(np.asarray(x, float))


def geometry(row: np.ndarray) -> dict:
    w, c_rel, la, lb = row
    c = w / 2 + c_rel * (1 - w)
    return {"xa": c - w / 2, "xb": c + w / 2, "alpha": 10.0**la, "beta": 10.0**lb}


def horizon_family(row: np.ndarray, T_list=T_LIST) -> dict:
    g = geometry(row)
    n_max = STEPS_PER_UNIT * max(T_list)
    cfg = RobinConfig(alpha=g["alpha"], beta=g["beta"], xa=g["xa"], xb=g["xb"], T=float(max(T_list)), Nt=n_max)
    ops = assemble_operators(cfg)
    a = impulse_response(ops)  # (n_max, nx)
    G = interval_load(ONE, cfg, 0.0, cfg.xa, cfg.xb)
    R = a @ (ops.M_obs @ a.T)
    aG_cs = np.cumsum(a @ G)

    sizes = [STEPS_PER_UNIT * T for T in T_list]
    H = {N: np.zeros((N, N)) for N in sizes}
    for d in range(n_max):
        cs = np.cumsum(np.diagonal(R, offset=-d))
        for N in sizes:
            if N > d:
                j = np.arange(d, N)
                vals = cs[N - 1 - j]
                H[N][j - d, j] = vals
                H[N][j, j - d] = vals
    out = {}
    for T, N in zip(T_list, sizes):
        g_T = aG_cs[N - 1 - np.arange(N)]
        out[T] = np.linalg.solve(H[N] + g["alpha"] * np.eye(N), g_T)

    psi = spla.splu(ops.K_beta.tocsc()).solve(ops.e0)
    out["u_bar"] = float(psi @ G / (psi @ (ops.M_obs @ psi) + g["alpha"]))
    return out


def check(rows: np.ndarray) -> list[dict]:
    """Horizon-family controls vs monolithic KKT at T = 1 and T = 2."""
    res = []
    for row in rows:
        fam = horizon_family(row, T_list=(1, 2))
        g = geometry(row)
        for T in (1, 2):
            cfg = RobinConfig(alpha=g["alpha"], beta=g["beta"], xa=g["xa"], xb=g["xb"], T=float(T), Nt=STEPS_PER_UNIT * T)
            kkt = solve_kkt(cfg, yd_func=ONE)
            res.append({"T": T, **{k: float(v) for k, v in g.items()},
                        "rel_diff_vs_kkt": float(np.linalg.norm(fam[T] - kkt.U) / np.linalg.norm(kkt.U)),
                        "kkt_rel_res": kkt.rel_res})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--T-list", type=int, nargs="+", default=list(T_LIST), help="horizons (e.g. a frozen fresh test set)")
    args = ap.parse_args()
    T_list = tuple(args.T_list)

    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=4, seed=args.seed).random(args.n) * (hi - lo)

    if args.check:
        for r in check(X[:3]):
            print(json.dumps(r))
        return

    from multiprocessing import Pool

    t0 = time.perf_counter()
    from functools import partial

    with Pool(args.workers) as pool:
        fams = pool.map(partial(horizon_family, T_list=T_list), list(X), chunksize=2)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
    geo = [geometry(r) for r in X]
    payload = {f"U_T{T}": np.array([f[T] for f in fams]) for T in T_list}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X=X,
        u_bar=np.array([f["u_bar"] for f in fams]),
        xa=np.array([g["xa"] for g in geo]),
        xb=np.array([g["xb"] for g in geo]),
        alpha=np.array([g["alpha"] for g in geo]),
        beta=np.array([g["beta"] for g in geo]),
        train_idx=np.sort(perm[:n_tr]),
        val_idx=np.sort(perm[n_tr : n_tr + n_va]),
        test_idx=np.sort(perm[n_tr + n_va :]),
        dt=1.0 / STEPS_PER_UNIT,
        T_list=np.array(T_list),
        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "target": "y_d = 1 on omega_o",
                                  "f": "0", "y0": "0", "nx": 160, "steps_per_unit_time": STEPS_PER_UNIT,
                                  "seed": args.seed}),
        **payload,
    )
    print(f"{args.n} samples x horizons {T_list} in {time.perf_counter() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
