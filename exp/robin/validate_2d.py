#!/usr/bin/env python3
"""Validation and cost scaling for robin2d.py (run single-threaded for fair timings).

V1  reduced horizon family vs monolithic KKT (nh = 12, T = 1 and 2)
V2  static optimum u_bar vs the middle of a long horizon (nh = 20, T = 4)
V3  separable lambda_1 vs a direct sparse eigensolve of (K, M) (nh = 30)
C1  dense symplectic-pencil eigendecomposition time vs n (exact ARE method)
C2  monolithic KKT solve time vs n (T = 1)
C3  reduced exact solve (all horizons 1, 2, 4, 8) at nh = 100 (n = 9900), with a timing breakdown
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
import scipy.sparse.linalg as spla  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D, assemble, horizon_family, lambda1, pencil_eig_seconds, solve_kkt, static_optimum  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--pencil-nh", type=int, nargs="+", default=[10, 15, 20, 25, 30, 35])
    ap.add_argument("--kkt-nh", type=int, nargs="+", default=[10, 15, 20, 30, 40])
    args = ap.parse_args()
    rep = {}

    base = dict(alpha=3e-3, beta=1.5, xa=0.12, xb=0.47, ya=0.25, yb=0.65)
    v1 = []
    for T in (1, 2):
        cfg = Cfg2D(nh=12, T=float(T), Nt=320 * T, **base)
        u_kkt, res, ops = solve_kkt(cfg)
        fam = horizon_family(assemble(Cfg2D(nh=12, T=2.0, Nt=640, **base)), (1, 2))
        v1.append({"T": T, "n": ops.n, "rel_diff_reduced_vs_kkt": rel(fam[T], u_kkt), "kkt_rel_res": res})
    rep["V1_reduced_vs_kkt"] = v1
    print(json.dumps(v1), flush=True)

    ops = assemble(Cfg2D(nh=20, T=4.0, Nt=1280, **base))
    u4 = horizon_family(ops, (4,))[4]
    ubar = static_optimum(ops)
    rep["V2_plateau"] = {"n": ops.n, "u_bar": ubar, "rel_err_mid_T4": abs(u4[639] - ubar) / abs(ubar)}
    print(rep["V2_plateau"], flush=True)

    ops = assemble(Cfg2D(nh=30, **base))
    lam_sep = lambda1(30, base["beta"])
    lam_dir = float(spla.eigsh(ops.K.tocsc(), k=1, M=ops.M.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)[0])
    rep["V3_lambda1"] = {"separable": lam_sep, "direct": lam_dir, "rel_diff": abs(lam_sep - lam_dir) / lam_dir}
    print(rep["V3_lambda1"], flush=True)

    c1 = []
    for nh in args.pencil_nh:
        cfg = Cfg2D(nh=nh, **base)
        n = nh * (nh - 1)
        secs = pencil_eig_seconds(cfg)
        c1.append({"nh": nh, "n": n, "pencil_dim": 2 * n, "seconds": secs})
        print(c1[-1], flush=True)
    ns = np.array([r["n"] for r in c1], float)
    ts = np.array([r["seconds"] for r in c1])
    big = ns >= 300
    p = np.polyfit(np.log(ns[big]), np.log(ts[big]), 1)
    rep["C1_pencil_eig"] = {"runs": c1, "fitted_exponent": float(p[0]),
                            "extrapolated_seconds_n9900": float(np.exp(np.polyval(p, np.log(9900.0)))),
                            "dense_matrix_GB_n9900_complex_eigvecs": float((2 * 9900) ** 2 * 16 / 1e9)}
    print(rep["C1_pencil_eig"]["fitted_exponent"], rep["C1_pencil_eig"]["extrapolated_seconds_n9900"], flush=True)

    c2 = []
    for nh in args.kkt_nh:
        cfg = Cfg2D(nh=nh, **base)
        t0 = time.perf_counter()
        _, res, ops = solve_kkt(cfg)
        c2.append({"nh": nh, "n": ops.n, "unknowns": 2 * ops.n * 320 + 320, "seconds_T1": time.perf_counter() - t0, "rel_res": res})
        print(c2[-1], flush=True)
    ns = np.array([r["n"] for r in c2], float)
    ts = np.array([r["seconds_T1"] for r in c2])
    p = np.polyfit(np.log(ns[1:]), np.log(ts[1:]), 1)
    rep["C2_kkt_T1"] = {"runs": c2, "fitted_exponent": float(p[0]),
                        "extrapolated_seconds_n9900_T1": float(np.exp(np.polyval(p, np.log(9900.0))))}

    timing = {}
    cfg = Cfg2D(nh=100, T=8.0, Nt=2560, **base)
    t0 = time.perf_counter()
    ops = assemble(cfg)
    t_asm = time.perf_counter() - t0
    fam = horizon_family(ops, (1, 2, 4, 8), timing=timing)
    t0 = time.perf_counter()
    ub = static_optimum(ops)
    t_static = time.perf_counter() - t0
    rep["C3_reduced_exact_n9900"] = {"n": ops.n, "assembly_s": t_asm, **timing, "static_optimum_s": t_static,
                                     "plateau_rel_err_mid_T8": abs(fam[8][1279] - ub) / abs(ub)}
    print(rep["C3_reduced_exact_n9900"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
