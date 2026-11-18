#!/usr/bin/env python3
"""Validation and calibration for semilinear1d.py (single-threaded).

V1  gamma = 0 reproduces robin_solver.solve_kkt
V2  reduced gradient vs centered finite differences (gamma > 0, small grid)
V3  at the Newton solution: reduced gradient ~ 0, random perturbations increase J (local min),
    Newton from zero and from the LQ solution give the same control (uniqueness spot check)
V4  plateau: static nonlinear OCP vs the middle of a T = 4 solution; linearized rate floor
V5  nonlinearity strength ||u_gamma - u_0|| / ||u_0|| for gamma in a grid (nx = 160, T = 1)
V6  Newton-KKT timing at T = 1 and T = 8 (nx = 160)
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, solve_kkt as solve_kkt_lin  # noqa: E402
from semilinear1d import build, cost, forward, reduced_gradient, solve_kkt, static_ocp  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rep = {}
    geo = dict(alpha=3e-3, beta=1.0, xa=0.25, xb=0.55)
    T_ref = 1.5

    cfg = RobinConfig(nx=160, Nt=320, **geo)
    lin = solve_kkt_lin(cfg, yd_func=lambda x, t: np.full_like(np.asarray(x, float), T_ref))
    _, _, U0, info0 = solve_kkt(build(cfg, 0.0, T_ref))
    rep["V1_gamma0_vs_linear"] = {"rel_diff": rel(U0, lin.U), "newton_iters": info0["newton_iters"]}
    print(rep["V1_gamma0_vs_linear"], flush=True)

    small = RobinConfig(nx=40, Nt=80, **geo)
    s = build(small, 10.0, T_ref)
    rng = np.random.default_rng(0)
    u, d = 2.0 + rng.standard_normal(80), rng.standard_normal(80)
    eps = 1e-4
    fd = (cost(s, forward(s, u + eps * d), u + eps * d) - cost(s, forward(s, u - eps * d), u - eps * d)) / (2 * eps)
    an = float(reduced_gradient(s, u) @ d)
    rep["V2_gradient_fd"] = {"fd": fd, "adjoint": an, "rel_err": abs(fd - an) / abs(an)}
    print(rep["V2_gradient_fd"], flush=True)

    v3 = []
    for gamma in (5.0, 20.0):
        s = build(small, gamma, T_ref)
        Y, P, U, info = solve_kkt(s)
        Ylin, Plin, Ulin, _ = solve_kkt(build(small, 0.0, T_ref))
        _, _, U2, info2 = solve_kkt(s, init=(Ylin, Plin, Ulin))
        J0 = cost(s, forward(s, U), U)
        dJ = [cost(s, forward(s, U + 1e-2 * np.linalg.norm(U) / np.sqrt(80) * dd), U + 1e-2 * np.linalg.norm(U) / np.sqrt(80) * dd) - J0
              for dd in rng.standard_normal((6, 80))]
        g = reduced_gradient(s, U)
        v3.append({"gamma": gamma, "newton_iters_from_zero": info["newton_iters"], "newton_iters_from_lq": info2["newton_iters"],
                   "same_solution": rel(U2, U), "grad_rel": float(np.linalg.norm(g) / np.linalg.norm(reduced_gradient(s, np.zeros(80)))),
                   "min_dJ_random_perturb": float(min(dJ))})
        print(v3[-1], flush=True)
    rep["V3_solution_checks"] = v3

    cfg4 = RobinConfig(nx=160, Nt=1280, T=4.0, **geo)
    s4 = build(cfg4, 10.0, T_ref)
    _, _, U4, info4 = solve_kkt(s4)
    st = static_ocp(s4)
    rep["V4_plateau"] = {"u_bar": st["u_bar"], "rel_err_mid_T4": abs(U4[639] - st["u_bar"]) / abs(st["u_bar"]),
                         "lam1_lin": st["lam1_lin"], "newton_iters_T4": info4["newton_iters"]}
    print(rep["V4_plateau"], flush=True)

    v5 = []
    for T_r in (0.5, 1.0, 2.0):
        _, _, Ul, _ = solve_kkt(build(cfg, 0.0, T_r))
        for gamma in (1.0, 5.0, 10.0, 20.0, 50.0):
            _, _, Ug, ig = solve_kkt(build(cfg, gamma, T_r))
            v5.append({"T_ref": T_r, "gamma": gamma, "rel_change_vs_lq": rel(Ug, Ul), "newton_iters": ig["newton_iters"]})
            print(v5[-1], flush=True)
    rep["V5_nonlinearity"] = v5

    v6 = {}
    for T in (1, 8):
        c = RobinConfig(nx=160, Nt=320 * T, T=float(T), **geo)
        t0 = time.perf_counter()
        _, _, _, inf = solve_kkt(build(c, 10.0, T_ref))
        v6[f"T{T}"] = {"seconds": time.perf_counter() - t0, "newton_iters": inf["newton_iters"]}
        print(T, v6[f"T{T}"], flush=True)
    rep["V6_timing"] = v6
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
