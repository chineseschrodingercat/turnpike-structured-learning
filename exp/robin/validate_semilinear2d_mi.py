#!/usr/bin/env python3
"""Checks for semilinear2d_mi.py, plus a static probe of the nonlinearity strength.

X1  gamma = 0 (nh = 100, m = 99, T = 1): Newton-CG from the LQ control stops at once; from zero it reproduces the LQ control
X2  small grid (nh = 8, m = 7, N = 40), gamma = 20: gradient and Hessian-vector vs finite differences; Newton-CG vs an
    independent all-at-once KKT Newton
X3  probe (static problem, nh = 100, m = 99): ||u_bar - u_bar_LQ|| / ||u_bar_LQ|| over alpha and gamma for one target
X4  full grid, gamma = 30, log10 alpha = -2: T = 1 solve (iterations, time, change vs LQ), plateau of T = 4 vs static OCP
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
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d_mi import static_optimum_mi  # noqa: E402
from robin2d_multi import CfgStrip  # noqa: E402
from semilinear2d import adjoint  # noqa: E402
from semilinear2d_mi import build, cost, forward, gradient, hessvec, lq_control, solve, static_ocp  # noqa: E402

COEFS = (1.0, 0.5, -0.4, 0.3, 0.6, -0.5, 0.3)


def kkt_newton(s, tol=1e-12, maxit=40):
    o, g, N, n, m = s.o, s.gamma, s.cfg.Nt, s.o.n, s.m
    Y, P, C = np.zeros((N, n)), np.zeros((N, n)), np.zeros((N, m))

    def res(Y, P, C):
        Yp = np.vstack([np.zeros(n), Y[:-1]])
        Pn = np.vstack([P[1:], np.zeros(n)])
        F1 = Y @ o.A.T - Yp @ s.Mdt.T + g * s.ML * Y**3 - C @ s.B.T
        F2 = P @ o.A.T + 3 * g * s.ML * Y**2 * P - Pn @ s.Mdt.T - o.G + Y @ o.M_obs.T
        F3 = s.cfg.alpha * C @ s.Mu.T - P @ s.B
        return np.concatenate([F1.ravel(), F2.ravel(), F3.ravel()])

    F = res(Y, P, C)
    r0 = np.linalg.norm(F)
    for _ in range(maxit):
        if np.linalg.norm(F) <= tol * r0:
            break
        J1 = sp.block_diag([o.A + sp.diags(3 * g * s.ML * Y[k] ** 2) for k in range(N)]) - sp.kron(sp.eye(N, k=-1), s.Mdt)
        D2 = sp.block_diag([o.M_obs + sp.diags(6 * g * s.ML * Y[k] * P[k]) for k in range(N)])
        Bb = -sp.kron(sp.eye(N), s.B)
        Km = sp.bmat([[J1, None, Bb], [D2, J1.T, None], [None, Bb.T, s.cfg.alpha * sp.kron(sp.eye(N), sp.csr_matrix(s.Mu))]], format="csc")
        d = spla.spsolve(Km, -F)
        Y = Y + d[: N * n].reshape(N, n)
        P = P + d[N * n : 2 * N * n].reshape(N, n)
        C = C + d[2 * N * n :].reshape(N, m)
        F = res(Y, P, C)
    return C


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rep = {}
    rng = np.random.default_rng(0)

    c1 = CfgStrip(nh=100, alpha=1e-2, beta=0.5, w=0.2, coefs=COEFS, T=1.0, Nt=320)
    s0 = build(c1, 99, 0.0)
    Clq = lq_control(c1, 99, tol=1e-12)
    _, i1 = solve(s0, Clq)
    Cz, i2 = solve(build(c1, 99, 0.0), np.zeros_like(Clq))
    rep["X1_gamma0"] = {"from_lq_iters": i1["newton_iters"], "from_lq_grad": i1["grad_rel"], "from_zero_vs_lq": rel(Cz, Clq),
                        "from_zero_iters": i2["newton_iters"]}
    print(rep["X1_gamma0"], flush=True)

    cs = CfgStrip(nh=8, alpha=1e-2, beta=0.5, w=0.25, coefs=COEFS, T=0.5, Nt=40)
    s = build(cs, 7, 20.0)
    C = 1.0 + 0.3 * rng.standard_normal((40, 7))
    V = rng.standard_normal((40, 7))
    eps = 1e-5
    Jp, Jm = cost(s, forward(s, C + eps * V), C + eps * V), cost(s, forward(s, C - eps * V), C - eps * V)
    Y = forward(s, C)
    P = adjoint(s, Y)
    g = gradient(s, C, P)
    gp = gradient(s, C + eps * V, adjoint(s, forward(s, C + eps * V)))
    gm = gradient(s, C - eps * V, adjoint(s, forward(s, C - eps * V)))
    HV = hessvec(s, Y, P, V)
    Cs, inf = solve(s, np.zeros((40, 7)), gtol=1e-12)
    Ck = kkt_newton(s)
    rep["X2_small"] = {"grad_fd_rel": abs((Jp - Jm) / (2 * eps) - np.sum(g * V)) / abs(np.sum(g * V)),
                       "hessvec_fd_rel": rel((gp - gm) / (2 * eps), HV), "newton_cg_vs_kkt": rel(Cs, Ck), "newton_iters": inf["newton_iters"]}
    print(rep["X2_small"], flush=True)

    probe = []
    for la in (-3.0, -2.0, -1.0):
        for gam in (1.0, 10.0, 100.0):
            cfg = CfgStrip(nh=100, alpha=10**la, beta=0.5, w=0.2, coefs=COEFS)
            sp_ = build(cfg, 99, gam)
            t0 = time.perf_counter()
            st = static_ocp(sp_)
            u0 = static_optimum_mi(sp_.mi)
            probe.append({"log10_alpha": la, "gamma": gam, "rel_change": rel(st["u_bar"], u0), "ymax": float(np.abs(st["y_bar"]).max()),
                          "iters": st["iters"], "res": st["res_rel"], "seconds": time.perf_counter() - t0})
            print(probe[-1], flush=True)
    rep["X3_probe"] = probe

    x4 = {}
    for T in (1, 4):
        cT = CfgStrip(nh=100, alpha=1e-2, beta=0.5, w=0.2, coefs=COEFS, T=float(T), Nt=320 * T)
        t0 = time.perf_counter()
        Clq = lq_control(cT, 99)
        t_lq = time.perf_counter() - t0
        sT = build(cT, 99, 30.0)
        CT, inf = solve(sT, Clq, verbose=True)
        x4[f"T{T}"] = {"lq_seconds": t_lq, **{k: inf[k] for k in ("newton_iters", "cg_iters", "grad_rel", "seconds")}, "rel_change_vs_lq": rel(CT, Clq)}
        if T == 4:
            st = static_ocp(build(CfgStrip(nh=100, alpha=1e-2, beta=0.5, w=0.2, coefs=COEFS), 99, 30.0))
            x4["plateau_rel_err_t2"] = rel(CT[639], st["u_bar"])
            sv = np.linalg.svd(CT, compute_uv=False)
            x4["sv_ratio_2_3"] = [float(sv[1] / sv[0]), float(sv[2] / sv[0])]
        print(T, x4, flush=True)
    rep["X4_full"] = x4
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
