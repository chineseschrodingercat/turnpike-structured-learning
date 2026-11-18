#!/usr/bin/env python3
"""Moving-target edge control (moving2d geometry) with a general reaction term N(y):
    y_t - Laplace y + N(y) = 0,   N = M_L-lumped,
    cubic   N = gamma y^3                                  (monotone, semilinear_moving.py)
    nagumo  N = kappa y (y - theta) (y - 1)                 (bistable: stable states 0 and 1, threshold theta)
Exact solver: the reduced Newton-CG of semilinear_moving.py with N, N', N'' in place of the cubic terms; the reduced
Hessian may be indefinite for the bistable case (negative curvature stops CG; Armijo line search on J). The implicit
step A z + M_L N(z) = rhs stays uniquely solvable while M/dt + M_L min N' > 0 (dt kappa max(-N'/kappa) < 1).
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
from moving2d import CfgMove, MoveOps, assemble, target_loads  # noqa: E402
from semilinear2d import shifted_solve  # noqa: E402


@dataclass
class Reaction:
    kind: str
    k: float  # gamma (cubic) or kappa (nagumo)
    theta: float = 0.25

    def N(self, y):
        if self.kind == "cubic":
            return self.k * y**3
        return self.k * y * (y - self.theta) * (y - 1.0)

    def dN(self, y):
        if self.kind == "cubic":
            return 3 * self.k * y**2
        return self.k * (3 * y**2 - 2 * (1 + self.theta) * y + self.theta)

    def d2N(self, y):
        if self.kind == "cubic":
            return 6 * self.k * y
        return self.k * (6 * y - 2 * (1 + self.theta))


@dataclass
class ReactMove:
    cfg: CfgMove
    rx: Reaction
    o: MoveOps
    ML: np.ndarray
    lu: object
    Mdt: sp.csr_matrix
    GY: np.ndarray
    csq: np.ndarray
    stats: dict


def build(cfg: CfgMove, rx: Reaction) -> ReactMove:
    o = assemble(cfg)
    GY, csq = target_loads(o)
    return ReactMove(cfg=cfg, rx=rx, o=o, ML=np.asarray(o.M.sum(axis=1)).ravel(), lu=spla.splu(o.A.tocsc()),
                     Mdt=(o.M / cfg.dt).tocsr(), GY=GY, csq=csq, stats={"lu_solves": 0})


def solve_shifted(s, d, r):
    """(A + diag(d)) x = r by CG preconditioned with A^{-1}; d < 0 somewhere (bistable) is fine while A + diag(d) is SPD.
    When the reaction dominates (dt |N'| > 4, large states at line-search trial points) CG is slow: direct solve."""
    if s.cfg.dt * np.max(np.abs(d) / s.ML) > 4.0:
        return spla.spsolve((s.o.A + sp.diags(d)).tocsc(), r)
    return shifted_solve(s, d, r, maxit=120)


def G(s, k):
    return np.kron(s.GY[k], s.o.gx)


def forward(s: ReactMove, C: np.ndarray, y0: np.ndarray | None = None, tol: float = 1e-13) -> np.ndarray:
    N, n = C.shape[0], s.o.n
    Y = np.zeros((N, n))
    y = np.zeros(n) if y0 is None else y0.copy()
    yprev = y.copy()
    for k in range(N):
        rhs = s.Mdt @ y + s.o.B @ C[k]
        z = 2 * y - yprev if k > 0 else y.copy()
        nr = max(np.linalg.norm(rhs), 1e-300)
        F = s.o.A @ z + s.ML * s.rx.N(z) - rhs
        nF = np.linalg.norm(F)
        for it in range(80):
            if nF <= tol * nr:
                break
            dN = s.rx.dN(z)
            # chord (A^{-1}) while it contracts (dt |N'| small), else Newton; backtracking on |F|
            dz = s.lu.solve(F) if (it < 8 and s.cfg.dt * np.abs(dN).max() < 0.3) else solve_shifted(s, s.ML * dN, F)
            lam = 1.0
            while True:
                zn = z - lam * dz
                Fn = s.o.A @ zn + s.ML * s.rx.N(zn) - rhs
                nFn = np.linalg.norm(Fn)
                if nFn < (1 - 1e-4 * lam) * nF or lam < 1e-6:
                    break
                lam *= 0.5
            z, F, nF = zn, Fn, nFn
        yprev, y = y, z
        Y[k] = y
    return Y


def adjoint(s: ReactMove, Y: np.ndarray) -> np.ndarray:
    N, n = Y.shape[0], s.o.n
    P = np.zeros((N, n))
    p = np.zeros(n)
    for k in range(N - 1, -1, -1):
        p = solve_shifted(s, s.ML * s.rx.dN(Y[k]), s.Mdt @ p + G(s, k) - s.o.M_obs @ Y[k])
        P[k] = p
    return P


def cost(s: ReactMove, Y: np.ndarray, C: np.ndarray, Mu=None) -> float:
    c = s.cfg
    Mu = s.o.Mu if Mu is None else Mu
    J = 0.0
    for k in range(C.shape[0]):
        y = Y[k]
        J += 0.5 * c.dt * (float(y @ (s.o.M_obs @ y)) - 2.0 * float(y @ G(s, k)) + s.csq[k])
    return J + 0.5 * c.alpha * c.dt * float(np.sum(C * (C @ Mu.T)))


def gradient(s, C, P):
    return s.cfg.dt * (s.cfg.alpha * C @ s.o.Mu.T - P @ s.o.B)


def hessvec(s, Y, P, V):
    N, n = Y.shape[0], s.o.n
    dY = np.zeros((N, n))
    dy = np.zeros(n)
    for k in range(N):
        dy = solve_shifted(s, s.ML * s.rx.dN(Y[k]), s.Mdt @ dy + s.o.B @ V[k])
        dY[k] = dy
    dp = np.zeros(n)
    BtdP = np.zeros((N, V.shape[1]))
    BT = s.o.B.T.tocsr()
    for k in range(N - 1, -1, -1):
        src = s.Mdt @ dp - (s.o.M_obs @ dY[k] + s.ML * s.rx.d2N(Y[k]) * P[k] * dY[k])
        dp = solve_shifted(s, s.ML * s.rx.dN(Y[k]), src)
        BtdP[k] = BT @ dp
    return s.cfg.dt * (s.cfg.alpha * V @ s.o.Mu.T - BtdP)


def solve(s: ReactMove, C0: np.ndarray, gtol: float = 1e-8, maxit: int = 40, verbose: bool = False):
    """Inexact Newton-CG with negative-curvature safeguard and Armijo backtracking (full steps near convergence)."""
    t0 = time.perf_counter()
    N, m = C0.shape
    g_ref = np.linalg.norm(gradient(s, np.zeros((N, m)), adjoint(s, np.zeros((N, s.o.n)))))
    Minv = np.linalg.inv(s.cfg.dt * s.cfg.alpha * s.o.Mu)
    C = C0.copy()
    Y = forward(s, C)
    J = cost(s, Y, C)
    hist, cg_total, neg, it = [], 0, 0, 0
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
                neg += 1
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
        if slope >= 0:  # not a descent direction: fall back to the preconditioned gradient
            X, slope = -g @ Minv.T, float(np.sum(g * (-g @ Minv.T)))
        while True:
            Cn = C + step * X
            Yn = forward(s, Cn)
            Jn = cost(s, Yn, Cn)
            ok = np.isfinite(Jn) and ((gn < 1e-5 and Jn <= J + 1e-12 * abs(J)) or Jn <= J + 1e-4 * step * slope)
            if ok or step < 1e-8:
                break
            step *= 0.5
        if not ok:
            break
        C, Y, J = Cn, Yn, Jn
    return C, {"newton_iters": it, "cg_iters": cg_total, "neg_curv": neg, "grad_rel": hist[-1], "J": J,
               "seconds": time.perf_counter() - t0}
