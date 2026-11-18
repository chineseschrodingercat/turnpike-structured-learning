#!/usr/bin/env python3
"""2D heat equation, distributed edge control u(s, t) on x = 0 (m hat functions), observation strip
omega = [0, w] x [0.05, 0.95] (robin2d_multi geometry) and a periodically MOVING target (hot spot):
    y_d(y, t) = exp(-(y - c(t))^2 / (2 sigma^2)),   c(t) = 0.5 + a sin(omega t + phi0).
Exact finite-horizon control: reduced PCG (as robin2d_mi.pcg_solve) with a time-dependent target load
G_k = gy(t_k) (x) gx, preconditioner (alpha Mu)^{-1}.
Exact periodic turnpike (harmonic balance): G(theta) = sum_n G_n e^{i n theta}, theta = omega t + phi0; for each
harmonic the discrete BE symbol z_n = (1 - e^{-i n omega dt}) / dt gives
    Psi_n = (z_n M + K)^{-1} B,   (alpha Mu + Psi_n^H M_obs Psi_n) u_n = Psi_n^H G_n,
and u_hat(t_k) = Re[u_0 + 2 sum_{n>=1} u_n e^{i n theta_k}] solves the discrete optimality system exactly
(the generalization of the single-frequency closed form of gen_osc_turnpike_data.py).
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
from robin2d import one_d  # noqa: E402
from robin2d_multi import CfgStrip, assemble_strip  # noqa: E402
from robin_solver import RobinConfig, interval_load  # noqa: E402


@dataclass(frozen=True)
class CfgMove:
    nh: int = 100
    m: int = 99
    alpha: float = 1e-3
    beta: float = 1.0
    w: float = 0.15
    a: float = 0.25
    omega: float = 2 * np.pi
    sigma: float = 0.08
    phi0: float = 0.0
    T: float = 1.0
    Nt: int = 320
    ya: float = 0.05
    yb: float = 0.95

    @property
    def dt(self) -> float:
        return self.T / self.Nt


def target(cfg: CfgMove, y, theta):
    c = 0.5 + cfg.a * np.sin(theta)
    return np.exp(-0.5 * ((np.asarray(y, float) - c) / cfg.sigma) ** 2)


@dataclass
class MoveOps:
    cfg: CfgMove
    M: sp.csr_matrix
    K: sp.csr_matrix
    A: sp.csr_matrix
    M_obs: sp.csr_matrix
    B: sp.csr_matrix
    Mu: np.ndarray
    gx: np.ndarray
    S: np.ndarray
    n: int
    cy: RobinConfig


def assemble(cfg: CfgMove) -> MoveOps:
    mi = assemble_strip(CfgStrip(nh=cfg.nh, alpha=cfg.alpha, beta=cfg.beta, w=cfg.w, T=cfg.T, Nt=cfg.Nt, ya=cfg.ya, yb=cfg.yb), cfg.m)
    o = mi.base
    gx = one_d(cfg.nh, 0.0, cfg.w, drop_first=False)[3]
    cy = RobinConfig(alpha=1.0, beta=0.0, xa=0.0, xb=1.0, nx=cfg.nh, Nt=1)
    S = np.unique(np.concatenate([o.M_obs.nonzero()[0], np.nonzero(np.kron(np.ones(cfg.nh - 1), gx))[0]]))
    return MoveOps(cfg=cfg, M=o.M, K=o.K, A=o.A, M_obs=o.M_obs, B=mi.B.tocsr(), Mu=mi.Mu, gx=gx, S=S, n=o.n, cy=cy)


def gy_of_theta(ops: MoveOps, theta: float) -> np.ndarray:
    c = ops.cfg
    return interval_load(lambda y, t: target(c, y, theta), ops.cy, 0.0, c.ya, c.yb)[1:]


def target_loads(ops: MoveOps) -> tuple[np.ndarray, np.ndarray]:
    """gy at t_k = k dt (k = 1..N), and int_omega y_d^2 per step."""
    c = ops.cfg
    th = c.omega * np.arange(1, c.Nt + 1) * c.dt + c.phi0
    GY = np.stack([gy_of_theta(ops, t) for t in th])
    xq, wq = leggauss(200)
    yq = 0.5 * (c.yb - c.ya) * xq + 0.5 * (c.yb + c.ya)
    csq = np.array([c.w * 0.5 * (c.yb - c.ya) * np.sum(wq * target(c, yq, t) ** 2) for t in th])
    return GY, csq


class Sweeps:
    def __init__(self, ops: MoveOps, GY: np.ndarray):
        c = ops.cfg
        self.ops, self.N, self.GY = ops, c.Nt, GY
        self.lu = spla.splu(ops.A.tocsc())
        self.Mdt = (ops.M / c.dt).tocsr()
        self.MoS = ops.M_obs[:, ops.S].tocsr()
        self.BT = ops.B.T.tocsr()

    def forward(self, C):
        o = self.ops
        y = np.zeros(o.n)
        yS = np.zeros((self.N, len(o.S)))
        for k in range(self.N):
            y = self.lu.solve(self.Mdt @ y + o.B @ C[k])
            yS[k] = y[o.S]
        return yS

    def forward_full(self, C):
        o = self.ops
        y = np.zeros(o.n)
        Y = np.zeros((self.N, o.n))
        for k in range(self.N):
            y = self.lu.solve(self.Mdt @ y + o.B @ C[k])
            Y[k] = y
        return Y

    def adjoint(self, yS, with_target: bool):
        o = self.ops
        p = np.zeros(o.n)
        out = np.zeros((self.N, o.m if hasattr(o, "m") else o.cfg.m))
        for k in range(self.N - 1, -1, -1):
            src = self.Mdt @ p
            if with_target:
                src = src + np.kron(self.GY[k], o.gx)
            if yS is not None:
                src = src - self.MoS @ yS[k]
            p = self.lu.solve(src)
            out[k] = self.BT @ p
        return out


def pcg_solve(ops: MoveOps, GY: np.ndarray, C0: np.ndarray | None = None, tol: float = 1e-10, maxit: int = 2000):
    c = ops.cfg
    sw = Sweeps(ops, GY)
    t0 = time.perf_counter()
    g = sw.adjoint(None, True)
    Minv = np.linalg.inv(c.alpha * ops.Mu)

    def op(X):
        return c.alpha * X @ ops.Mu.T - sw.adjoint(sw.forward(X), False)

    C = np.zeros((c.Nt, c.m)) if C0 is None else C0.copy()
    r = g - op(C) if C0 is not None else g.copy()
    z = r @ Minv.T
    d = z.copy()
    rz = float(np.sum(r * z))
    g_norm = float(np.sqrt(np.sum(g * (g @ Minv.T))))
    it = 0
    for it in range(1, maxit + 1):
        if np.sqrt(max(rz, 0.0)) <= tol * g_norm:
            break
        Ad = op(d)
        a = rz / float(np.sum(d * Ad))
        C += a * d
        r -= a * Ad
        z = r @ Minv.T
        rz_new = float(np.sum(r * z))
        d = z + (rz_new / rz) * d
        rz = rz_new
    return C, {"iters": it, "seconds": time.perf_counter() - t0}


def cost(ops: MoveOps, GY: np.ndarray, csq: np.ndarray, C: np.ndarray) -> float:
    c = ops.cfg
    sw = Sweeps(ops, GY)
    Y = sw.forward_full(C)
    J = 0.0
    for k in range(c.Nt):
        y = Y[k]
        J += 0.5 * c.dt * (float(y @ (ops.M_obs @ y)) - 2.0 * float(y @ np.kron(GY[k], ops.gx)) + csq[k])
    return J + 0.5 * c.alpha * c.dt * float(np.einsum("ki,ij,kj->", C, ops.Mu, C))


def periodic_turnpike(ops: MoveOps, n_theta: int = 256, tol: float = 1e-8, nmax: int = 80):
    """Harmonics u_n (n = 0..N_h) of the exact discrete periodic optimal control; returns (u_n, info)."""
    c = ops.cfg
    t0 = time.perf_counter()
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    GYt = np.stack([gy_of_theta(ops, t) for t in th])
    Gn = np.fft.fft(GYt, axis=0) / n_theta  # Gn[n] multiplies e^{i n theta}
    norms = np.linalg.norm(Gn, axis=1)
    nh = int(min(nmax, max(n for n in range(n_theta // 2) if norms[n] > tol * norms[0])))
    Bd = ops.B.toarray()
    U = np.zeros((nh + 1, c.m), dtype=complex)
    for n in range(nh + 1):
        z = (1.0 - np.exp(-1j * n * c.omega * c.dt)) / c.dt
        lu = spla.splu((z * ops.M + ops.K).tocsc().astype(complex))
        Psi = lu.solve(Bd.astype(complex))
        H = c.alpha * ops.Mu + Psi.conj().T @ (ops.M_obs @ Psi)
        U[n] = np.linalg.solve(H, Psi.conj().T @ np.kron(Gn[n], ops.gx))
    return U, {"n_harm": nh, "seconds": time.perf_counter() - t0, "tail": float(norms[nh + 1] / norms[0]) if nh + 1 < n_theta // 2 else 0.0}


def eval_periodic(U: np.ndarray, omega: float, phi0: float, t: np.ndarray) -> np.ndarray:
    th = omega * np.asarray(t)[:, None] * np.arange(U.shape[0])[None, :] + phi0 * np.arange(U.shape[0])[None, :]
    E = np.exp(1j * th)
    E[:, 1:] *= 2.0
    return np.real(E @ U)
