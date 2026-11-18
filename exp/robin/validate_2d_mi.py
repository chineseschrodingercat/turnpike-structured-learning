#!/usr/bin/env python3
"""Validation and cost growth with the number of control inputs m (single-threaded timings).

V1  PCG vs dense reduced vs monolithic KKT on a tiny grid (nh = 12, m = 11 = fine edge space, T = 1)
V2  plateau: static multi-input optimum vs the middle of a T = 4 solution (nh = 30, m = 29)
C1  at nh = 100 (n = 9900), one parameter point, m in {1, 3, 9, 33, 99}:
      PCG at T = 1 and T = 8 (iterations, time); dense reduced at T = 1 for m <= --dense-max-m;
      surrogate online pieces (static optimum with m sparse solves)
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D  # noqa: E402
from robin2d_mi import assemble_mi, dense_reduced, pcg_solve, solve_kkt_mi, static_optimum_mi  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--m-list", type=int, nargs="+", default=[1, 3, 9, 33, 99])
    ap.add_argument("--dense-max-m", type=int, default=33)
    args = ap.parse_args()
    base = dict(alpha=3e-3, beta=1.5, xa=0.12, xb=0.47, ya=0.25, yb=0.65)
    rep = {}

    cfg = Cfg2D(nh=12, T=1.0, Nt=320, **base)
    u_kkt, res = solve_kkt_mi(cfg, 11)
    mi = assemble_mi(cfg, 11)
    u_pcg, info = pcg_solve(mi, 1.0, tol=1e-12)
    u_den, _ = dense_reduced(mi, 1.0)
    rep["V1"] = {"n": mi.base.n, "m": 11, "pcg_vs_kkt": rel(u_pcg, u_kkt), "dense_vs_kkt": rel(u_den, u_kkt),
                 "kkt_rel_res": res, "pcg_iters": info["iters"]}
    print(rep["V1"], flush=True)

    mi = assemble_mi(Cfg2D(nh=30, T=4.0, Nt=1280, **base), 29)
    u4, _ = pcg_solve(mi, 4.0, tol=1e-12)
    ub = static_optimum_mi(mi)
    rep["V2"] = {"plateau_rel_err_mid_T4": rel(u4[639], ub)}
    print(rep["V2"], flush=True)

    rows = []
    for m in args.m_list:
        row = {"m": m}
        for T in (1, 8):
            mi = assemble_mi(Cfg2D(nh=100, T=float(T), Nt=320 * T, **base), m)
            _, info = pcg_solve(mi, float(T))
            row[f"pcg_T{T}"] = info
            print(m, T, info, flush=True)
        if m <= args.dense_max_m:
            mi = assemble_mi(Cfg2D(nh=100, T=1.0, Nt=320, **base), m)
            u_d, dinfo = dense_reduced(mi, 1.0)
            u_p, _ = pcg_solve(mi, 1.0, tol=1e-12)
            dinfo["dense_vs_pcg"] = rel(u_d, u_p)
            row["dense_T1"] = dinfo
            print(m, "dense", dinfo, flush=True)
        mi = assemble_mi(Cfg2D(nh=100, **base), m)
        t0 = time.perf_counter()
        static_optimum_mi(mi)
        row["static_optimum_s"] = time.perf_counter() - t0
        rows.append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rep["C1"] = rows
        args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
