#!/usr/bin/env python3
"""Checks for semilinear2d.py.

W1  gamma = 0: Newton-CG returns the exact LQ control (robin2d) in one step; forward = linear sweep
W2  small grid (nh = 8, N = 40), gamma = 20: reduced gradient vs centered differences; Hessian-vector vs differences
    of the gradient; Newton-CG vs an independent all-at-once KKT Newton (sparse direct)
W3  full grid (nh = 100), gamma = 20: plateau of T = 4 = static OCP; Newton / CG iterations and wall clock at T = 1, 8
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
from robin2d import Cfg2D  # noqa: E402
from semilinear2d import adjoint, build, cost, forward, gradient, hessvec, lq_control, solve, static_ocp  # noqa: E402

GEO = dict(alpha=1e-2, beta=1.0, xa=0.1, xb=0.4, ya=0.3, yb=0.7)


def kkt_newton(s, tol=1e-12, maxit=40):
    """All-at-once Newton on (Y, P, U) for small grids (independent of the reduced solver)."""
    o, g, N, n = s.o, s.gamma, s.cfg.Nt, s.o.n
    Y, P, U = np.zeros((N, n)), np.zeros((N, n)), np.zeros(N)

    def res(Y, P, U):
        Yp = np.vstack([np.zeros(n), Y[:-1]])
        Pn = np.vstack([P[1:], np.zeros(n)])
        F1 = Y @ o.A.T - Yp @ s.Mdt.T + g * s.ML * Y**3 - np.outer(U, o.b)
        F2 = P @ o.A.T + 3 * g * s.ML * Y**2 * P - Pn @ s.Mdt.T - o.G + Y @ o.M_obs.T
        F3 = s.cfg.alpha * U - P @ o.b
        return np.concatenate([F1.ravel(), F2.ravel(), F3])

    F = res(Y, P, U)
    r0 = np.linalg.norm(F)
    for _ in range(maxit):
        if np.linalg.norm(F) <= tol * r0:
            break
        J1 = sp.block_diag([o.A + sp.diags(3 * g * s.ML * Y[k] ** 2) for k in range(N)]) - sp.kron(sp.eye(N, k=-1), s.Mdt)
        D2 = sp.block_diag([o.M_obs + sp.diags(6 * g * s.ML * Y[k] * P[k]) for k in range(N)])
        B = -sp.kron(sp.eye(N), sp.csr_matrix(o.b.reshape(-1, 1)))
        Km = sp.bmat([[J1, None, B], [D2, J1.T, None], [None, B.T, s.cfg.alpha * sp.eye(N)]], format="csc")
        d = spla.spsolve(Km, -F)
        Y, P, U = Y + d[: N * n].reshape(N, n), P + d[N * n : 2 * N * n].reshape(N, n), U + d[2 * N * n :]
        F = res(Y, P, U)
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rep = {}
    rng = np.random.default_rng(0)

    c = Cfg2D(nh=100, T=1.0, Nt=320, **GEO)
    s0 = build(c, 0.0)
    ulq = lq_control(c, (1,))[1]
    u, info = solve(s0, ulq)
    rep["W1_gamma0"] = {"newton_iters": info["newton_iters"], "grad_rel": info["grad_rel"], "rel_change": float(np.linalg.norm(u - ulq) / np.linalg.norm(ulq))}
    print(rep, flush=True)

    cs = Cfg2D(nh=8, T=0.5, Nt=40, **GEO)
    s = build(cs, 20.0)
    uu, dd = 1.0 + 0.3 * rng.standard_normal(40), rng.standard_normal(40)
    eps = 1e-5
    Jp, Jm = cost(s, forward(s, uu + eps * dd), uu + eps * dd), cost(s, forward(s, uu - eps * dd), uu - eps * dd)
    Y = forward(s, uu)
    P = adjoint(s, Y)
    g = gradient(s, uu, P)
    gp = gradient(s, uu + eps * dd, adjoint(s, forward(s, uu + eps * dd)))
    gm = gradient(s, uu - eps * dd, adjoint(s, forward(s, uu - eps * dd)))
    Hv = hessvec(s, Y, P, dd)
    us, inf_s = solve(s, np.zeros(40))
    uk = kkt_newton(s)
    rep["W2_small"] = {"grad_fd_rel": abs((Jp - Jm) / (2 * eps) - g @ dd) / abs(g @ dd),
                       "hessvec_fd_rel": float(np.linalg.norm((gp - gm) / (2 * eps) - Hv) / np.linalg.norm(Hv)),
                       "newton_cg_vs_kkt": float(np.linalg.norm(us - uk) / np.linalg.norm(uk)), "newton_iters": inf_s["newton_iters"]}
    print(rep["W2_small"], flush=True)

    w3 = {}
    for T in (1, 8):
        cT = Cfg2D(nh=100, T=float(T), Nt=320 * T, **GEO)
        t0 = time.perf_counter()
        ulq = lq_control(cT, (T,))[T]
        t_lq = time.perf_counter() - t0
        sT = build(cT, 20.0)
        uT, inf = solve(sT, ulq, verbose=True)
        w3[f"T{T}"] = {"lq_seconds": t_lq, **{k: inf[k] for k in ("newton_iters", "cg_iters", "grad_rel", "seconds", "lu_solves")},
                       "rel_change_vs_lq": float(np.linalg.norm(uT - ulq) / np.linalg.norm(ulq))}
        if T == 8:
            st = static_ocp(build(Cfg2D(nh=100, **GEO), 20.0))
            w3["static"] = {k: st[k] for k in ("u_bar", "lam1_lin", "res_rel", "iters")}
            w3["plateau_rel_err_t4"] = abs(uT[1279] - st["u_bar"]) / abs(st["u_bar"])
            t0 = time.perf_counter()
            static_ocp(build(Cfg2D(nh=100, **GEO), 20.0))
            w3["static_seconds"] = time.perf_counter() - t0
        print(T, w3, flush=True)
    rep["W3_full_grid"] = w3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
