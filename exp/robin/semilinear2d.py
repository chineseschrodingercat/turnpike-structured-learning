#!/usr/bin/env python3
"""2D semilinear boundary control (robin2d geometry): y_t - Laplace y + gamma y^3 = 0 on (0,1)^2, scalar Robin control
on x = 0 (uniform along the edge), tracking y_d = 1 on the rectangle omega, y(., 0) = 0 (T_ref = 1 units).

Discretization: Q1 tensor FEM of robin2d.assemble + backward Euler; the cubic term uses the lumped mass M_L.
    A y_k - (M/dt) y_{k-1} + gamma M_L y_k^3 = b u_k
    J(u) = sum_k dt/2 (y_k^T M_obs y_k - 2 y_k^T G + |omega|) + alpha dt/2 sum_k u_k^2
Exact solver: reduced-space inexact Newton-CG on u in R^N (the all-at-once KKT has 2 n N ~ 5e7 unknowns at T = 8),
    gradient   g_k = dt (alpha u_k - b^T p_k),  (A + 3 gamma M_L y_k^2) p_k = (M/dt) p_{k+1} + G - M_obs y_k
    Hessian    H v = dt (alpha v - b^T dp),     J_k dy_k = (M/dt) dy_{k-1} + b v_k,
                                                J_k dp_k = (M/dt) dp_{k+1} - (M_obs + 6 gamma M_L y_k p_k) dy_k
Per-step solves with J_k = A + diag(3 gamma M_L y_k^2): CG preconditioned by the LU of A (the cubic term is a small
diagonal shift relative to the stiffness), the nonlinear state step by a chord iteration with the same LU.
Warm start: the exact LQ control (robin2d.horizon_family). Static (turnpike) problem by sparse Newton.
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
from robin2d import Cfg2D, Ops2D, assemble, horizon_family  # noqa: E402


@dataclass
class Semi2D:
    cfg: Cfg2D
    gamma: float
    o: Ops2D
    ML: np.ndarray
    lu: object
    Mdt: sp.csr_matrix
    area: float
    stats: dict


def build(cfg: Cfg2D, gamma: float) -> Semi2D:
    o = assemble(cfg)
    return Semi2D(cfg=cfg, gamma=gamma, o=o, ML=np.asarray(o.M.sum(axis=1)).ravel(), lu=spla.splu(o.A.tocsc()),
                  Mdt=(o.M / cfg.dt).tocsr(), area=(cfg.xb - cfg.xa) * (cfg.yb - cfg.ya), stats={"lu_solves": 0})


def shifted_solve(s: Semi2D, d: np.ndarray, r: np.ndarray, tol: float = 1e-13, maxit: int = 60) -> np.ndarray:
    """(A + diag(d)) x = r, d >= 0, by CG preconditioned with A^{-1}."""
    x = s.lu.solve(r)
    s.stats["lu_solves"] += 1
    if not np.any(d):
        return x
    nr = np.linalg.norm(r)
    res = r - (s.o.A @ x + d * x)
    if np.linalg.norm(res) <= tol * nr:
        return x
    z = s.lu.solve(res)
    s.stats["lu_solves"] += 1
    pd, rz = z.copy(), float(res @ z)
    for _ in range(maxit):
        q = s.o.A @ pd + d * pd
        a = rz / float(pd @ q)
        x += a * pd
        res -= a * q
        if np.linalg.norm(res) <= tol * nr:
            break
        z = s.lu.solve(res)
        s.stats["lu_solves"] += 1
        rz_new = float(res @ z)
        pd = z + (rz_new / rz) * pd
        rz = rz_new
    return x


def forward(s: Semi2D, u: np.ndarray, y0: np.ndarray | None = None, tol: float = 1e-13) -> np.ndarray:
    """Nonlinear state; each step by a chord iteration z <- z - A^{-1} F(z) (Newton fallback if slow)."""
    N, n, g = len(u), s.o.n, s.gamma
    Y = np.zeros((N, n))
    y = np.zeros(n) if y0 is None else y0.copy()
    yprev = y.copy()
    for k in range(N):
        rhs = s.Mdt @ y + s.o.b * u[k]
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


def adjoint(s: Semi2D, Y: np.ndarray) -> np.ndarray:
    N, n, g = Y.shape[0], s.o.n, s.gamma
    P = np.zeros((N, n))
    p = np.zeros(n)
    for k in range(N - 1, -1, -1):
        p = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, s.Mdt @ p + s.o.G - s.o.M_obs @ Y[k])
        P[k] = p
    return P


def cost(s: Semi2D, Y: np.ndarray, u: np.ndarray) -> float:
    dt = s.cfg.dt
    tr = 0.5 * dt * (float(np.sum(Y * (Y @ s.o.M_obs.T))) - 2 * float(np.sum(Y @ s.o.G)) + len(u) * s.area)
    return tr + 0.5 * s.cfg.alpha * dt * float(u @ u)


def gradient(s: Semi2D, u: np.ndarray, P: np.ndarray) -> np.ndarray:
    return s.cfg.dt * (s.cfg.alpha * u - P @ s.o.b)


def hessvec(s: Semi2D, Y: np.ndarray, P: np.ndarray, v: np.ndarray) -> np.ndarray:
    N, n, g = Y.shape[0], s.o.n, s.gamma
    dY = np.zeros((N, n))
    dy = np.zeros(n)
    for k in range(N):
        dy = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, s.Mdt @ dy + s.o.b * v[k])
        dY[k] = dy
    dp = np.zeros(n)
    btdp = np.zeros(N)
    for k in range(N - 1, -1, -1):
        src = s.Mdt @ dp - (s.o.M_obs @ dY[k] + 6 * g * s.ML * Y[k] * P[k] * dY[k])
        dp = shifted_solve(s, 3 * g * s.ML * Y[k] ** 2, src)
        btdp[k] = s.o.b @ dp
    return s.cfg.dt * (s.cfg.alpha * v - btdp)


def lq_control(cfg: Cfg2D, T_list) -> dict:
    """Exact linear (gamma = 0) controls for every T in T_list, same dt (robin2d reduced method)."""
    return horizon_family(assemble(cfg), T_list, steps_per_unit=int(round(1 / cfg.dt)))


def solve(s: Semi2D, u0: np.ndarray, gtol: float = 1e-10, maxit: int = 15, verbose: bool = False):
    """Inexact Newton-CG with Armijo backtracking on J far from the solution and full Newton steps in the quadratic
    regime (||g|| / ||g(0)|| < 1e-5, where the decrease of J is below its round-off). Stops when ||g|| <= gtol ||g(0)||
    or when the gradient stagnates below 1e-7 (round-off floor of long horizons)."""
    t0 = time.perf_counter()
    N = len(u0)
    g_ref = np.linalg.norm(gradient(s, np.zeros(N), adjoint(s, np.zeros((N, s.o.n)))))
    u = u0.copy()
    Y = forward(s, u)
    J = cost(s, Y, u)
    hist, cg_total, it = [], 0, 0
    for it in range(1, maxit + 1):
        P = adjoint(s, Y)
        g = gradient(s, u, P)
        gn = np.linalg.norm(g) / g_ref
        hist.append(gn)
        if verbose:
            print(f"  newton {it}: J {J:.12e} |g|/|g0| {gn:.2e}", flush=True)
        if gn <= gtol or (len(hist) > 1 and gn < 1e-7 and gn > 0.5 * hist[-2]):
            break
        eta = min(0.1, np.sqrt(gn))  # Eisenstat-Walker-type forcing
        dt_alpha = s.cfg.dt * s.cfg.alpha
        x = np.zeros(N)
        r = -g.copy()
        z = r / dt_alpha
        d = z.copy()
        rz = float(r @ z)
        r0 = np.linalg.norm(r)
        for _cg in range(200):
            Hd = hessvec(s, Y, P, d)
            cg_total += 1
            curv = float(d @ Hd)
            if curv <= 0:
                if not np.any(x):
                    x = -g / dt_alpha
                break
            a = rz / curv
            x += a * d
            r -= a * Hd
            if np.linalg.norm(r) <= eta * r0:
                break
            z = r / dt_alpha
            rz_new = float(r @ z)
            d = z + (rz_new / rz) * d
            rz = rz_new
        step, slope = 1.0, float(g @ x)
        while True:
            un = u + step * x
            Yn = forward(s, un)
            Jn = cost(s, Yn, un)
            if gn < 1e-5 or Jn <= J + 1e-4 * step * slope or step < 1e-6:
                break
            step *= 0.5
        u, Y, J = un, Yn, Jn
    return u, {"newton_iters": it, "cg_iters": cg_total, "grad_rel": hist[-1], "hist": hist, "J": J,
               "seconds": time.perf_counter() - t0, "lu_solves": s.stats["lu_solves"]}


def static_ocp(s: Semi2D, tol: float = 1e-13, maxit: int = 40):
    """Static optimum (y_bar, p_bar, u_bar) by sparse Newton from the LQ static solution; lam1_lin by shift-invert."""
    o, g, a = s.o, s.gamma, s.cfg.alpha
    n = o.n
    K, Mo, b, G, ML = o.K.tocsr(), o.M_obs.tocsr(), o.b, o.G, s.ML
    luK = spla.splu(K.tocsc())
    psi = luK.solve(b)
    u = float(psi @ G / (psi @ (Mo @ psi) + a))
    y = u * psi
    p = luK.solve(G - Mo @ y)
    bc, bt, au = sp.csr_matrix(b.reshape(-1, 1)), sp.csr_matrix(b.reshape(1, -1)), sp.csr_matrix([[a]])

    def F(y, p, u):
        return np.concatenate([K @ y + g * ML * y**3 - b * u, K @ p + 3 * g * ML * y**2 * p - G + Mo @ y, [a * u - b @ p]])

    Fz = F(y, p, u)
    r = np.linalg.norm(Fz)
    scale = max(np.linalg.norm(G), 1e-300)
    it = 0
    for it in range(1, maxit + 1):
        if r <= tol * scale or g == 0:
            break
        Ky = K + sp.diags(3 * g * ML * y**2)
        Jm = sp.bmat([[Ky, None, -bc], [Mo + sp.diags(6 * g * ML * y * p), Ky, None], [None, -bt, au]], format="csc")
        dz = spla.spsolve(Jm, -Fz)
        step, ok = 1.0, False
        while step > 1e-8:
            yn, pn, un = y + step * dz[:n], p + step * dz[n : 2 * n], u + step * dz[-1]
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
    return {"u_bar": float(u), "y_bar": y, "p_bar": p, "lam1_lin": lam1, "res_rel": float(r / scale), "iters": it}
