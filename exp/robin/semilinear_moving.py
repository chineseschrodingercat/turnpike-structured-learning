#!/usr/bin/env python3
"""2D semilinear problem with a periodically MOVING target: y_t - Laplace y + gamma y^3 = 0, distributed edge control
u(s, t) (m inputs) on x = 0, observation strip [0, w] x [0.05, 0.95],
    y_d(y, t) = exp(-(y - 0.5 - a sin(omega t + phi0))^2 / (2 sigma^2))     (moving2d.py).
Exact solver: reduced inexact Newton-CG of semilinear2d_mi.py with the time-dependent target load G_k = gy(t_k) (x) gx
in the adjoint and in the cost; warm start = exact LQ (gamma = 0) control from moving2d.pcg_solve.
There is no harmonic-balance closed form for the nonlinear periodic turnpike; it is read off long-horizon solutions.
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import CfgMove, MoveOps, assemble, eval_periodic, pcg_solve, periodic_turnpike, target_loads  # noqa: E402
from semilinear2d import shifted_solve  # noqa: E402


@dataclass
class SemiMove:
    cfg: CfgMove
    gamma: float
    o: MoveOps
    ML: np.ndarray
    lu: object
    Mdt: sp.csr_matrix
    GY: np.ndarray
    csq: np.ndarray
    stats: dict


def build(cfg: CfgMove, gamma: float) -> SemiMove:
    o = assemble(cfg)
    GY, csq = target_loads(o)
    return SemiMove(cfg=cfg, gamma=gamma, o=o, ML=np.asarray(o.M.sum(axis=1)).ravel(), lu=spla.splu(o.A.tocsc()),
                    Mdt=(o.M / cfg.dt).tocsr(), GY=GY, csq=csq, stats={"lu_solves": 0})


def G(s: SemiMove, k: int) -> np.ndarray:
    return np.kron(s.GY[k], s.o.gx)


def forward(s: SemiMove, C: np.ndarray, tol: float = 1e-13) -> np.ndarray:
    N, n, g = C.shape[0], s.o.n, s.gamma
    Y = np.zeros((N, n))
    y = np.zeros(n)
    yprev = y.copy()
    for k in range(N):
        rhs = s.Mdt @ y + s.o.B @ C[k]
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


def adjoint(s: SemiMove, Y: np.ndarray) -> np.ndarray:
    N, n, g = Y.shape[0], s.o.n, s.gamma
    P = np.zeros((N, n))
    p = np.zeros(n)
    for k in range(N - 1, -1, -1):
        p = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, s.Mdt @ p + G(s, k) - s.o.M_obs @ Y[k])
        P[k] = p
    return P


def cost(s: SemiMove, Y: np.ndarray, C: np.ndarray) -> float:
    c = s.cfg
    J = 0.0
    for k in range(C.shape[0]):
        y = Y[k]
        J += 0.5 * c.dt * (float(y @ (s.o.M_obs @ y)) - 2.0 * float(y @ G(s, k)) + s.csq[k])
    return J + 0.5 * c.alpha * c.dt * float(np.sum(C * (C @ s.o.Mu.T)))


def gradient(s: SemiMove, C: np.ndarray, P: np.ndarray) -> np.ndarray:
    return s.cfg.dt * (s.cfg.alpha * C @ s.o.Mu.T - P @ s.o.B)


def hessvec(s: SemiMove, Y: np.ndarray, P: np.ndarray, V: np.ndarray) -> np.ndarray:
    N, n, g = Y.shape[0], s.o.n, s.gamma
    dY = np.zeros((N, n))
    dy = np.zeros(n)
    for k in range(N):
        dy = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, s.Mdt @ dy + s.o.B @ V[k])
        dY[k] = dy
    dp = np.zeros(n)
    BtdP = np.zeros((N, V.shape[1]))
    BT = s.o.B.T.tocsr()
    for k in range(N - 1, -1, -1):
        src = s.Mdt @ dp - (s.o.M_obs @ dY[k] + 6 * g * s.ML * Y[k] * P[k] * dY[k])
        dp = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, src)
        BtdP[k] = BT @ dp
    return s.cfg.dt * (s.cfg.alpha * V @ s.o.Mu.T - BtdP)


def lq_control(cfg: CfgMove, tol: float = 1e-10):
    """Exact linear control (PCG warm-started from the harmonic-balance periodic turnpike) and the harmonics."""
    o = assemble(cfg)
    GY, _ = target_loads(o)
    U, _ = periodic_turnpike(o)
    t = np.arange(1, cfg.Nt + 1) * cfg.dt
    C, info = pcg_solve(o, GY, C0=eval_periodic(U, cfg.omega, cfg.phi0, t), tol=tol)
    return C, U, info


def solve(s: SemiMove, C0: np.ndarray, gtol: float = 1e-8, maxit: int = 15, verbose: bool = False):
    """Inexact Newton-CG (Armijo far from the solution, full steps when ||g||/||g0|| < 1e-5, stop on stagnation)."""
    t0 = time.perf_counter()
    N, m = C0.shape
    g_ref = np.linalg.norm(gradient(s, np.zeros((N, m)), adjoint(s, np.zeros((N, s.o.n)))))
    Minv = np.linalg.inv(s.cfg.dt * s.cfg.alpha * s.o.Mu)
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
        for _cg in range(300):
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
    return C, {"newton_iters": it, "cg_iters": cg_total, "grad_rel": hist[-1], "J": J, "seconds": time.perf_counter() - t0,
               "lu_solves": s.stats["lu_solves"]}
