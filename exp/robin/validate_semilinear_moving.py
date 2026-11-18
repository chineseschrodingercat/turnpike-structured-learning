#!/usr/bin/env python3
"""Checks for semilinear_moving.py and a probe of the nonlinearity.

Y1  gamma = 0 (full grid, T = 1): Newton-CG from the LQ control stops at once
Y2  small grid (nh = 8, m = 7, N = 40), gamma = 20: gradient / Hessian-vector vs differences; Newton-CG vs an
    all-at-once KKT Newton with the time-dependent target
Y3  probe (full grid, T = 1): ||u - u_LQ|| / ||u_LQ|| over alpha and gamma for one moving target; Newton/CG/time
Y4  full grid, T = 4, one case: the middle of the nonlinear control is periodic (u(t + P) vs u(t)), its distance to the
    linear periodic turnpike, and the rank of u(s, t)
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
from moving2d import CfgMove, eval_periodic  # noqa: E402
from semilinear_moving import G, adjoint, build, cost, forward, gradient, hessvec, lq_control, solve  # noqa: E402

BASE = dict(a=0.3, omega=2 * np.pi, sigma=0.07, beta=0.5, w=0.2, phi0=1.0)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def kkt_newton(s, tol=1e-12, maxit=40):
    o, g, N, n, m = s.o, s.gamma, s.cfg.Nt, s.o.n, s.cfg.m
    Gs = np.stack([G(s, k) for k in range(N)])
    Y, P, C = np.zeros((N, n)), np.zeros((N, n)), np.zeros((N, m))

    def res(Y, P, C):
        Yp = np.vstack([np.zeros(n), Y[:-1]])
        Pn = np.vstack([P[1:], np.zeros(n)])
        F1 = Y @ o.A.T - Yp @ s.Mdt.T + g * s.ML * Y**3 - C @ o.B.T
        F2 = P @ o.A.T + 3 * g * s.ML * Y**2 * P - Pn @ s.Mdt.T - Gs + Y @ o.M_obs.T
        F3 = s.cfg.alpha * C @ o.Mu.T - P @ o.B
        return np.concatenate([F1.ravel(), F2.ravel(), F3.ravel()])

    F = res(Y, P, C)
    r0 = np.linalg.norm(F)
    for _ in range(maxit):
        if np.linalg.norm(F) <= tol * r0:
            break
        J1 = sp.block_diag([o.A + sp.diags(3 * g * s.ML * Y[k] ** 2) for k in range(N)]) - sp.kron(sp.eye(N, k=-1), s.Mdt)
        D2 = sp.block_diag([o.M_obs + sp.diags(6 * g * s.ML * Y[k] * P[k]) for k in range(N)])
        Bb = -sp.kron(sp.eye(N), o.B)
        Km = sp.bmat([[J1, None, Bb], [D2, J1.T, None], [None, Bb.T, s.cfg.alpha * sp.kron(sp.eye(N), sp.csr_matrix(o.Mu))]], format="csc")
        d = spla.spsolve(Km, -F)
        Y = Y + d[: N * n].reshape(N, n)
        P = P + d[N * n : 2 * N * n].reshape(N, n)
        C = C + d[2 * N * n :].reshape(N, m)
        F = res(Y, P, C)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rep = {}
    rng = np.random.default_rng(0)

    c1 = CfgMove(nh=100, m=99, alpha=1e-3, T=1.0, Nt=320, **BASE)
    Clq, _, _ = lq_control(c1, tol=1e-12)
    _, inf = solve(build(c1, 0.0), Clq)
    rep["Y1_gamma0"] = {"iters": inf["newton_iters"], "grad_rel": inf["grad_rel"]}
    print(rep, flush=True)

    cs = CfgMove(nh=8, m=7, alpha=1e-2, T=0.5, Nt=40, **{**BASE, "sigma": 0.15})
    s = build(cs, 20.0)
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
    rep["Y2_small"] = {"grad_fd_rel": abs((Jp - Jm) / (2 * eps) - np.sum(g * V)) / abs(np.sum(g * V)),
                       "hessvec_fd_rel": rel((gp - gm) / (2 * eps), HV), "newton_cg_vs_kkt": rel(Cs, kkt_newton(s)), "newton_iters": inf["newton_iters"]}
    print(rep["Y2_small"], flush=True)

    probe = []
    for la in (-3.5, -2.75, -2.0):
        cf = CfgMove(nh=100, m=99, alpha=10**la, T=1.0, Nt=320, **BASE)
        Clq, _, _ = lq_control(cf)
        for gam in (3.0, 30.0, 300.0):
            t0 = time.perf_counter()
            Cn, inf = solve(build(cf, gam), Clq)
            probe.append({"log10_alpha": la, "gamma": gam, "rel_change_vs_lq": rel(Cn, Clq), "newton": inf["newton_iters"],
                          "cg": inf["cg_iters"], "seconds": time.perf_counter() - t0})
            print(probe[-1], flush=True)
    rep["Y3_probe"] = probe

    cf = CfgMove(nh=100, m=99, alpha=10**-2.75, T=4.0, Nt=1280, **BASE)
    Clq, Ulin, _ = lq_control(cf)
    t0 = time.perf_counter()
    Cn, inf = solve(build(cf, 30.0), Clq, verbose=True)
    t = np.arange(1, 1281) * cf.dt
    P_steps = int(round(2 * np.pi / cf.omega / cf.dt))
    k0 = 480
    sv = np.linalg.svd(Cn, compute_uv=False)
    rep["Y4_T4"] = {"seconds": time.perf_counter() - t0, "newton": inf["newton_iters"], "cg": inf["cg_iters"],
                    "periodicity_middle": rel(Cn[k0 : k0 + P_steps], Cn[k0 + P_steps : k0 + 2 * P_steps]),
                    "middle_vs_linear_periodic": rel(eval_periodic(Ulin, cf.omega, cf.phi0, t[k0 : k0 + P_steps]), Cn[k0 : k0 + P_steps]),
                    "rel_change_vs_lq": rel(Cn, Clq), "rank_1e-3": int(np.sum(sv > 1e-3 * sv[0])), "sv2_sv1": float(sv[1] / sv[0])}
    print(rep["Y4_T4"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    np.savez(args.out.with_suffix(".npz"), C_nl=Cn.astype(np.float32), C_lq=Clq.astype(np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
