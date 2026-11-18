#!/usr/bin/env python3
"""Checks for nonzero initial states (semilinear1d with y0, y0_family).

V1  gamma = 0 with y0 reproduces robin_solver.solve_kkt(y0_func=...)
V2  gamma = 0: control is affine in y0 (u(a + b) - u(a) - u(b) + u(0) = 0)
V3  gamma = 10, y0 != 0: reduced gradient vs centered differences (small grid)
V4  turnpike with y0 (gamma = 10, T = 4): plateau = u_bar for every y0; with y0 = y_bar the initial layer
    (almost) vanishes; the terminal layer does not depend on y0
V5  input projection: K_in = 16 cosines of y0 - y_bar, effect of the truncation on the optimal control
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators, solve_kkt as solve_kkt_lin  # noqa: E402
from semilinear1d import build, cost, forward, reduced_gradient, solve_kkt, static_ocp_sparse  # noqa: E402
from y0_family import Projector, sample  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rep = {}
    geo = dict(alpha=3e-3, beta=1.0, xa=0.25, xb=0.55)
    cfg = RobinConfig(nx=160, Nt=320, **geo)
    x = np.arange(cfg.nx) * cfg.h
    ybar0 = static_ocp_sparse(build(cfg, 0.0, 1.0))["y_bar"]
    ya, _, _ = sample(rng, x, ybar0)
    yb, _, _ = sample(rng, x, ybar0)

    lin = solve_kkt_lin(cfg, yd_func=lambda xx, t: np.ones_like(np.asarray(xx, float)), y0_func=lambda xx: np.interp(xx, x, ya))
    Ua = solve_kkt(build(cfg, 0.0, 1.0, ya))[2]
    rep["V1_vs_linear_solver"] = rel(Ua, lin.U)
    Ub = solve_kkt(build(cfg, 0.0, 1.0, yb))[2]
    Uab = solve_kkt(build(cfg, 0.0, 1.0, ya + yb))[2]
    U0 = solve_kkt(build(cfg, 0.0, 1.0))[2]
    rep["V2_affine_defect"] = float(np.linalg.norm(Uab - Ua - Ub + U0) / np.linalg.norm(Uab))
    print(rep, flush=True)

    small = RobinConfig(nx=40, Nt=80, **geo)
    xs = np.arange(small.nx) * small.h
    s = build(small, 10.0, 1.0, np.interp(xs, x, ya))
    u, d = 2.0 + rng.standard_normal(80), rng.standard_normal(80)
    eps = 1e-4
    fd = (cost(s, forward(s, u + eps * d), u + eps * d) - cost(s, forward(s, u - eps * d), u - eps * d)) / (2 * eps)
    an = float(reduced_gradient(s, u) @ d)
    rep["V3_gradient_fd_rel_err"] = abs(fd - an) / abs(an)
    print(rep["V3_gradient_fd_rel_err"], flush=True)

    cfg4 = RobinConfig(nx=160, Nt=1280, T=4.0, **geo)
    st = static_ocp_sparse(build(cfg4, 10.0, 1.0))
    ub, ybar = st["u_bar"], st["y_bar"]
    lam = st["lam1_lin"]
    cases = {"zero": np.zeros(160), "y_bar": ybar}
    for j in range(3):
        cases[f"random{j}"] = sample(rng, x, ybar)[0]
    sols = {k: solve_kkt(build(cfg4, 10.0, 1.0, v), init=None)[2] for k, v in cases.items()}
    t = np.arange(1, 1281) / 320
    v4 = {"u_bar": ub, "lam1_lin": lam}
    for k, U in sols.items():
        v4[k] = {"plateau_rel_err_t2": abs(U[639] - ub) / abs(ub),
                 "initial_layer_max_dev_t<1": float(np.max(np.abs(U[t < 1] - ub)) / abs(ub)),
                 "u0_over_ubar": float(U[0] / ub)}
    term = t > 3
    v4["terminal_layer_diff_vs_zero"] = {k: float(np.max(np.abs(U[term] - sols["zero"][term])) / abs(ub)) for k, U in sols.items() if k != "zero"}
    rep["V4_turnpike_with_y0"] = v4
    print(json.dumps(v4, indent=1), flush=True)

    ops = assemble_operators(cfg)
    proj = Projector(x, ops.M)
    v5 = []
    for j in range(5):
        y0 = sample(rng, x, ybar0)[0]
        d = y0 - ybar0
        dp = proj.back(proj(d))
        U = solve_kkt(build(cfg, 0.0, 1.0, y0))[2]
        Up = solve_kkt(build(cfg, 0.0, 1.0, ybar0 + dp))[2]
        v5.append({"state_proj_rel_err": float(np.sqrt((d - dp) @ ops.M @ (d - dp) / (d @ ops.M @ d))), "control_rel_change": rel(Up, U)})
    rep["V5_projection_K16"] = v5
    print(json.dumps(v5), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
