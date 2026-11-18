#!/usr/bin/env python3
"""2D semilinear problem with distributed edge control u(s, t) = sum_l c_l(t) chi_l(s) (m inputs) and the edge-strip
target of robin2d_multi.CfgStrip: y_d(y) = coef_0 + sum_k coef_k sin(k pi y) on omega = [0, w] x [0.05, 0.95].

    A y_k - (M/dt) y_{k-1} + gamma M_L y_k^3 = B c_k,
    J(c) = sum_k dt/2 (y_k^T M_obs y_k - 2 y_k^T G + int_omega y_d^2) + alpha dt/2 sum_k c_k^T Mu c_k
Same reduced-space inexact Newton-CG as semilinear2d.py with the control in R^{N x m}:
    gradient g_k = dt (alpha Mu c_k - B^T p_k),   Hessian H V = dt (alpha V Mu - B^T dP),
CG preconditioned by (dt alpha Mu)^{-1} (the linear PCG of robin2d_mi.py; iteration counts independent of m).
Warm start: exact LQ control (robin2d_mi.pcg_solve). Static (turnpike) problem by sparse Newton on (y, p, u).
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Ops2D  # noqa: E402
from robin2d_mi import pcg_solve, static_optimum_mi  # noqa: E402
from robin2d_multi import CfgStrip, assemble_strip  # noqa: E402
from semilinear2d import adjoint, shifted_solve  # noqa: E402


@dataclass
class SemiMI:
    cfg: CfgStrip
    gamma: float
    o: Ops2D
    B: sp.csr_matrix
    Mu: np.ndarray
    m: int
    ML: np.ndarray
    lu: object
    Mdt: sp.csr_matrix
    c_track: float
    mi: object
    stats: dict


def target_sq_integral(cfg: CfgStrip) -> float:
    x, w = leggauss(200)
    y = 0.5 * (cfg.yb - cfg.ya) * x + 0.5 * (cfg.yb + cfg.ya)
    yd = cfg.coefs[0] + sum(c * np.sin(k * np.pi * y) for k, c in enumerate(cfg.coefs[1:], start=1))
    return float(cfg.w * 0.5 * (cfg.yb - cfg.ya) * np.sum(w * yd**2))


def build(cfg: CfgStrip, m: int, gamma: float) -> SemiMI:
    mi = assemble_strip(cfg, m)
    o = mi.base
    return SemiMI(cfg=cfg, gamma=gamma, o=o, B=mi.B.tocsr(), Mu=mi.Mu, m=m, ML=np.asarray(o.M.sum(axis=1)).ravel(),
                  lu=spla.splu(o.A.tocsc()), Mdt=(o.M / cfg.dt).tocsr(), c_track=target_sq_integral(cfg), mi=mi,
                  stats={"lu_solves": 0})


def forward(s: SemiMI, C: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    N, n, g = C.shape[0], s.o.n, s.gamma
    Y = np.zeros((N, n))
    y = np.zeros(n)
    yprev = y.copy()
    for k in range(N):
        rhs = s.Mdt @ y + s.B @ C[k]
        z = 2 * y - yprev if k > 0 else y.copy()
        nr = max(np.linalg.norm(rhs), 1e-300)
        for it in range(60):
            F = s.o.A @ z + g * s.ML * z**3 - rhs
            if np.linalg.norm(F) <= tol * nr:
                break
            if it < 8:
                z = z - s.lu.solve(F)
                s.stats["lu_solves"] += 1
            else:
                z = z - shifted_solve(s, 3 * g * s.ML * z**2, F)
        yprev, y = y, z
        Y[k] = y
    return Y


def cost(s: SemiMI, Y: np.ndarray, C: np.ndarray) -> float:
    dt = s.cfg.dt
    tr = 0.5 * dt * (float(np.sum(Y * (Y @ s.o.M_obs.T))) - 2 * float(np.sum(Y @ s.o.G)) + C.shape[0] * s.c_track)
    return tr + 0.5 * s.cfg.alpha * dt * float(np.sum(C * (C @ s.Mu.T)))


def gradient(s: SemiMI, C: np.ndarray, P: np.ndarray) -> np.ndarray:
    return s.cfg.dt * (s.cfg.alpha * C @ s.Mu.T - P @ s.B)


def hessvec(s: SemiMI, Y: np.ndarray, P: np.ndarray, V: np.ndarray) -> np.ndarray:
    N, n, g = Y.shape[0], s.o.n, s.gamma
    dY = np.zeros((N, n))
    dy = np.zeros(n)
    for k in range(N):
        dy = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, s.Mdt @ dy + s.B @ V[k])
        dY[k] = dy
    dp = np.zeros(n)
    BtdP = np.zeros((N, s.m))
    BT = s.B.T.tocsr()
    for k in range(N - 1, -1, -1):
        src = s.Mdt @ dp - (s.o.M_obs @ dY[k] + 6 * g * s.ML * Y[k] * P[k] * dY[k])
        dp = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, src)
        BtdP[k] = BT @ dp
    return s.cfg.dt * (s.cfg.alpha * V @ s.Mu.T - BtdP)


def lq_control(cfg: CfgStrip, m: int, tol: float = 1e-10) -> np.ndarray:
    return pcg_solve(assemble_strip(cfg, m), float(cfg.T), tol=tol)[0]


def solve(s: SemiMI, C0: np.ndarray, gtol: float = 1e-8, maxit: int = 15, verbose: bool = False):
    """Inexact Newton-CG (Armijo far from the solution, full steps when ||g||/||g0|| < 1e-5, stop on stagnation)."""
    t0 = time.perf_counter()
    N, m = C0.shape
    g_ref = np.linalg.norm(gradient(s, np.zeros((N, m)), adjoint(s, np.zeros((N, s.o.n)))))
    Minv = np.linalg.inv(s.cfg.dt * s.cfg.alpha * s.Mu)
    C = C0.copy()
    Y = forward(s, C)
    J = cost(s, Y, C)
    hist, cg_total, it = [], 0, 0
    for it in range(1, maxit + 1):
        P = adjoint(s, Y)
        g = gradient(s, C, P)
        gn = np.linalg.norm(g) / g_ref
        hist.append(gn)
        if verbose:
            print(f"  newton {it}: J {J:.12e} |g|/|g0| {gn:.2e}", flush=True)
        if gn <= gtol or (len(hist) > 1 and gn < 1e-7 and gn > 0.5 * hist[-2]):
            break
        eta = min(0.1, np.sqrt(gn))
        X = np.zeros((N, m))
        R = -g.copy()
        Z = R @ Minv.T
        D = Z.copy()
        rz = float(np.sum(R * Z))
        r0 = np.linalg.norm(R)
        for _cg in range(200):
            HD = hessvec(s, Y, P, D)
            cg_total += 1
            curv = float(np.sum(D * HD))
            if curv <= 0:
                if not np.any(X):
                    X = -g @ Minv.T
                break
            a = rz / curv
            X += a * D
            R -= a * HD
            if np.linalg.norm(R) <= eta * r0:
                break
            Z = R @ Minv.T
            rz_new = float(np.sum(R * Z))
            D = Z + (rz_new / rz) * D
            rz = rz_new
        step, slope = 1.0, float(np.sum(g * X))
        while True:
            Cn = C + step * X
            Yn = forward(s, Cn)
            Jn = cost(s, Yn, Cn)
            if gn < 1e-5 or Jn <= J + 1e-4 * step * slope or step < 1e-6:
                break
            step *= 0.5
        C, Y, J = Cn, Yn, Jn
    return C, {"newton_iters": it, "cg_iters": cg_total, "grad_rel": hist[-1], "hist": hist, "J": J,
               "seconds": time.perf_counter() - t0, "lu_solves": s.stats["lu_solves"]}


def static_ocp(s: SemiMI, tol: float = 1e-13, maxit: int = 40):
    """Static optimum (y_bar, p_bar, u_bar in R^m) by sparse Newton from the LQ static solution; lam1_lin."""
    o, g, a = s.o, s.gamma, s.cfg.alpha
    n, m = o.n, s.m
    K, Mo, B, G, ML = o.K.tocsr(), o.M_obs.tocsr(), s.B, o.G, s.ML
    luK = spla.splu(K.tocsc())
    u = static_optimum_mi(s.mi)
    y = luK.solve(B @ u)
    p = luK.solve(G - Mo @ y)
    BT, aMu = B.T.tocsr(), sp.csr_matrix(a * s.Mu)

    def F(y, p, u):
        return np.concatenate([K @ y + g * ML * y**3 - B @ u, K @ p + 3 * g * ML * y**2 * p - G + Mo @ y, a * (s.Mu @ u) - BT @ p])

    Fz = F(y, p, u)
    r = np.linalg.norm(Fz)
    scale = max(np.linalg.norm(G), 1e-300)
    it = 0
    for it in range(1, maxit + 1):
        if r <= tol * scale or g == 0:
            break
        Ky = K + sp.diags(3 * g * ML * y**2)
        Jm = sp.bmat([[Ky, None, -B], [Mo + sp.diags(6 * g * ML * y * p), Ky, None], [None, -BT, aMu]], format="csc")
        dz = spla.spsolve(Jm, -Fz)
        step, ok = 1.0, False
        while step > 1e-8:
            yn, pn, un = y + step * dz[:n], p + step * dz[n : 2 * n], u + step * dz[2 * n :]
            Fn = F(yn, pn, un)
            if np.linalg.norm(Fn) < (1 - 1e-4 * step) * r:
                ok = True
                break
            step *= 0.5
        if not ok:
            break
        y, p, u, Fz = yn, pn, un, Fn
        r = np.linalg.norm(Fz)
    Ky = (K + sp.diags(3 * g * ML * y**2)).tocsc()
    lam1 = float(spla.eigsh(Ky, k=1, M=o.M.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)[0])
    return {"u_bar": u, "y_bar": y, "p_bar": p, "lam1_lin": lam1, "res_rel": float(r / scale), "iters": it}
