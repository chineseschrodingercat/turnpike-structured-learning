#!/usr/bin/env python3
"""Semilinear heat equation with Robin boundary control and subzone tracking (1D), Newton on the all-at-once KKT.

    y_t - y_xx + gamma y^3 = 0 in (0,1) x (0,T),  -y_x(0,t) + beta y(0,t) = u(t),  y(1,t) = 0,  y(.,0) = 0,
    J = 1/2 int_0^T int_omega (y - T_ref)^2 + alpha/2 int_0^T u^2.
Discretization: the P1 / backward-Euler setting of robin_solver.py; the cubic term uses the lumped mass M_L.
KKT residuals (k = 1..N, y_0 = 0, p_{N+1} = 0):
    F1_k = A y_k - (M/dt) y_{k-1} + gamma M_L y_k^3 - e0 u_k
    F2_k = J_k p_k - (M/dt) p_{k+1} - G + M_obs y_k,      J_k = A + 3 gamma M_L diag(y_k^2)
    F3_k = alpha u_k - p_k(0)
(gamma = 0 is exactly the linear KKT of robin_solver.solve_kkt). Newton matrix (J_k symmetric):
    [ J1          0     -I(x)e0 ]       J1 = blockdiag(J_k) - subdiag(M/dt)
    [ D2          J1^T   0      ]       D2 = blockdiag(M_obs + 6 gamma M_L diag(y_k p_k))
    [ 0       -I(x)e0^T  alpha I ]
Static (turnpike) problem: K_b y + gamma M_L y^3 = e0 u, (K_b + 3 gamma M_L y^2) p = G - M_obs y, alpha u = p(0).
A nonzero initial state y0 (build(..., y0=...)) only changes F1_1 (y_0 = y0 instead of 0).
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators, interval_load  # noqa: E402


@dataclass(frozen=True)
class SemiOps:
    cfg: RobinConfig
    gamma: float
    T_ref: float
    M: sp.csr_matrix
    Kb: sp.csr_matrix
    A: sp.csr_matrix
    Mdt: sp.csr_matrix
    Mo: sp.csr_matrix
    ML: np.ndarray
    G: np.ndarray
    e0: np.ndarray
    c_track: float  # int_omega T_ref^2
    y0: np.ndarray  # initial state (nodal values on the free dofs)


def build(cfg: RobinConfig, gamma: float, T_ref: float, y0: np.ndarray | None = None) -> SemiOps:
    o = assemble_operators(cfg)
    G = T_ref * interval_load(lambda x, t: np.ones_like(np.asarray(x, float)), cfg, 0.0, cfg.xa, cfg.xb)
    return SemiOps(cfg=cfg, gamma=gamma, T_ref=T_ref, M=o.M, Kb=o.K_beta, A=o.A, Mdt=(o.M / cfg.dt).tocsr(), Mo=o.M_obs,
                   ML=np.asarray(o.M.sum(axis=1)).ravel(), G=G, e0=o.e0, c_track=T_ref**2 * (cfg.xb - cfg.xa),
                   y0=np.zeros(cfg.nx) if y0 is None else np.asarray(y0, float).copy())


def residual(s: SemiOps, Y, P, U):
    N = len(U)
    Yprev = np.vstack([s.y0, Y[:-1]])
    Pnext = np.vstack([P[1:], np.zeros(s.A.shape[0])])
    g = s.gamma
    F1 = Y @ s.A.T - Yprev @ s.Mdt.T + g * s.ML * Y**3 - np.outer(U, s.e0)
    F2 = P @ s.A.T + 3 * g * s.ML * Y**2 * P - Pnext @ s.Mdt.T - s.G + Y @ s.Mo.T
    F3 = s.cfg.alpha * U - P[:, 0]
    return F1, F2, F3


def newton_matrix(s: SemiOps, Y, P):
    N, n = Y.shape
    g = s.gamma
    Jdiag = sp.block_diag([s.A + sp.diags(3 * g * s.ML * Y[k] ** 2) for k in range(N)], format="csr")
    sub = sp.kron(sp.eye(N, k=-1), s.Mdt, format="csr")
    J1 = (Jdiag - sub).tocsr()
    D2 = sp.block_diag([s.Mo + sp.diags(6 * g * s.ML * Y[k] * P[k]) for k in range(N)], format="csr")
    B = -sp.kron(sp.eye(N), sp.csr_matrix(s.e0).T, format="csr")
    return sp.bmat([[J1, None, B], [D2, J1.T, None], [None, B.T, s.cfg.alpha * sp.eye(N)]], format="csc")


def solve_kkt(s: SemiOps, init=None, tol: float = 1e-11, maxit: int = 40):
    """Damped Newton on the all-at-once KKT system. init = (Y, P, U) or None (zero)."""
    N, n = s.cfg.Nt, s.A.shape[0]
    Y, P, U = (np.zeros((N, n)), np.zeros((N, n)), np.zeros(N)) if init is None else (init[0].copy(), init[1].copy(), init[2].copy())
    t0 = time.perf_counter()
    norm = lambda F: float(np.sqrt(sum(np.sum(f**2) for f in F)))
    F = residual(s, Y, P, U)
    r0 = r = norm(F)
    scale = max(r0, float(np.linalg.norm(s.G)) * np.sqrt(N))
    hist, it = [r], 0
    for it in range(1, maxit + 1):
        if r <= tol * scale:
            break
        d = spla.spsolve(newton_matrix(s, Y, P), -np.concatenate([F[0].ravel(), F[1].ravel(), F[2]]))
        dY, dP, dU = d[: N * n].reshape(N, n), d[N * n : 2 * N * n].reshape(N, n), d[2 * N * n :]
        step = 1.0
        while True:
            Fn = residual(s, Y + step * dY, P + step * dP, U + step * dU)
            rn = norm(Fn)
            if rn < (1 - 1e-4 * step) * r or step < 1e-6:
                break
            step *= 0.5
        Y, P, U, F, r = Y + step * dY, P + step * dP, U + step * dU, Fn, rn
        hist.append(r)
    return Y, P, U, {"newton_iters": it, "res_rel": r / scale, "seconds": time.perf_counter() - t0, "hist": hist}


def forward(s: SemiOps, U, tol=1e-13):
    """State for a given control (Newton per time step)."""
    N, n = len(U), s.A.shape[0]
    Y = np.zeros((N, n))
    y = s.y0.copy()
    for k in range(N):
        rhs = s.Mdt @ y + s.e0 * U[k]
        z = y.copy()
        for _ in range(50):
            Fz = s.A @ z + s.gamma * s.ML * z**3 - rhs
            if np.linalg.norm(Fz) <= tol * max(np.linalg.norm(rhs), 1e-300):
                break
            z = z - spla.spsolve((s.A + sp.diags(3 * s.gamma * s.ML * z**2)).tocsc(), Fz)
        y = z
        Y[k] = y
    return Y


def cost(s: SemiOps, Y, U) -> float:
    dt = s.cfg.dt
    track = 0.5 * dt * float(np.sum(Y * (Y @ s.Mo.T)) - 2 * np.sum(Y @ s.G) + len(U) * s.c_track)
    return track + 0.5 * s.cfg.alpha * dt * float(U @ U)


def adjoint(s: SemiOps, Y):
    """Adjoint states (backward sweep of F2 = 0) linearized around the state Y."""
    N, n = Y.shape
    P = np.zeros((N, n))
    p = np.zeros(n)
    for k in range(N - 1, -1, -1):
        rhs = s.G - s.Mo @ Y[k] + s.Mdt @ p
        p = spla.spsolve((s.A + sp.diags(3 * s.gamma * s.ML * Y[k] ** 2)).tocsc(), rhs)
        P[k] = p
    return P


def reduced_gradient(s: SemiOps, U):
    """dt * (alpha u_k - p_k(0)) with p from the (linear) adjoint around the forward state."""
    P = adjoint(s, forward(s, U))
    return s.cfg.dt * (s.cfg.alpha * U - P[:, 0])


def static_ocp(s: SemiOps, tol=1e-13, maxit=60):
    """Turnpike: static optimal (y_bar, p_bar, u_bar) by damped Newton; returns also the linearized rate floor."""
    n = s.A.shape[0]
    g, a = s.gamma, s.cfg.alpha
    Kb = s.Kb.toarray()
    Mo = s.Mo.toarray()

    def F(z):
        y, p, u = z[:n], z[n : 2 * n], z[-1]
        return np.concatenate([Kb @ y + g * s.ML * y**3 - s.e0 * u, (Kb + np.diag(3 * g * s.ML * y**2)) @ p - s.G + Mo @ y, [a * u - p[0]]])

    def Jac(z):
        y, p = z[:n], z[n : 2 * n]
        J = np.zeros((2 * n + 1, 2 * n + 1))
        Ky = Kb + np.diag(3 * g * s.ML * y**2)
        J[:n, :n] = Ky
        J[:n, -1] = -s.e0
        J[n : 2 * n, :n] = Mo + np.diag(6 * g * s.ML * y * p)
        J[n : 2 * n, n : 2 * n] = Ky
        J[-1, n] = -1.0
        J[-1, -1] = a
        return J

    z = np.zeros(2 * n + 1)
    r = np.linalg.norm(F(z))
    for _ in range(maxit):
        if r <= tol * max(np.linalg.norm(s.G), 1e-300):
            break
        d = np.linalg.solve(Jac(z), -F(z))
        step = 1.0
        while np.linalg.norm(F(z + step * d)) >= (1 - 1e-4 * step) * r and step > 1e-8:
            step *= 0.5
        z = z + step * d
        r = np.linalg.norm(F(z))
    y = z[:n]
    lam1 = float(sla.eigh(Kb + np.diag(3 * g * s.ML * y**2), s.M.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0])
    return {"u_bar": float(z[-1]), "y_bar": y, "p_bar": z[n : 2 * n], "lam1_lin": lam1, "res": float(r)}


def static_ocp_sparse(s: SemiOps, tol=1e-13, maxit=60):
    """static_ocp with sparse Newton steps and a shift-invert eigensolve (same equations, O(n) per step)."""
    n = s.A.shape[0]
    g, a = s.gamma, s.cfg.alpha
    e0c = sp.csr_matrix(s.e0.reshape(-1, 1))
    p0 = sp.csr_matrix(([-1.0], ([0], [0])), shape=(1, n))
    au = sp.csr_matrix([[a]])

    def F(y, p, u):
        return np.concatenate([s.Kb @ y + g * s.ML * y**3 - s.e0 * u, s.Kb @ p + 3 * g * s.ML * y**2 * p - s.G + s.Mo @ y, [a * u - p[0]]])

    y, p, u = np.zeros(n), np.zeros(n), 0.0
    Fz = F(y, p, u)
    r = np.linalg.norm(Fz)
    it = 0
    for it in range(1, maxit + 1):
        if r <= tol * max(np.linalg.norm(s.G), 1e-300):
            break
        Ky = s.Kb + sp.diags(3 * g * s.ML * y**2)
        J = sp.bmat([[Ky, None, -e0c], [s.Mo + sp.diags(6 * g * s.ML * y * p), Ky, None], [None, p0, au]], format="csc")
        d = spla.spsolve(J, -Fz)
        step, ok = 1.0, False
        while step > 1e-8:
            yn, pn, un = y + step * d[:n], p + step * d[n : 2 * n], u + step * d[-1]
            Fn = F(yn, pn, un)
            if np.linalg.norm(Fn) < (1 - 1e-4 * step) * r:
                ok = True
                break
            step *= 0.5
        if not ok:  # no decrease possible: the residual is at round-off level
            break
        y, p, u, Fz = yn, pn, un, Fn
        r = np.linalg.norm(Fz)
    Ky = (s.Kb + sp.diags(3 * g * s.ML * y**2)).tocsc()
    lam1 = float(spla.eigsh(Ky, k=1, M=s.M.tocsc(), sigma=0.0, which="LM", return_eigenvectors=False)[0])
    return {"u_bar": float(u), "y_bar": y, "p_bar": p, "lam1_lin": lam1, "res": float(r), "iters": it}
