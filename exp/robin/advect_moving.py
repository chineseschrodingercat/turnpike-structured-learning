#!/usr/bin/env python3
"""Moving hot-spot edge control (moving2d geometry) for advection-diffusion:
    y_t - eps Laplace y + b . grad y = 0           in (0,1)^2,  b = (bx, by) constant, axis-aligned
    eps dy/dn + beta y = u(s, t)                   on x = 0 (distributed control, m hat functions along s)
    y = 0 on the other edges,  J = 1/2 int int_strip (y - y_d)^2 + alpha/2 int int u^2   (as in moving2d).
Q1 tensor FEM, node index j * nh + i (j along s, i along x):
    K = (eps + dy) Ky (x) Mx + (eps + dx) My (x) Kx + beta My (x) e0 e0^T + bx My (x) Cx + by Cy (x) Mx,
Cx, Cy the 1D P1 convection matrices int phi_j' phi_i, (dx, dy) streamline artificial diffusion along the flow
(Il'in-Allen-Southwell, |b| h/2 (coth Pe_h - 1/Pe_h), Pe_h = |b| h / (2 eps)). eps = 1, b = 0 is moving2d exactly.
K is nonsymmetric, so the adjoint sweep solves with A^T; the reduced Hessian alpha Mu + S^* C S is still SPD (PCG).
The harmonic-balance periodic turnpike of moving2d holds unchanged for nonsymmetric K.
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import moving2d  # noqa: E402
from robin2d import one_d  # noqa: E402
from robin2d_mi import control_basis  # noqa: E402
from robin_solver import RobinConfig  # noqa: E402


@dataclass(frozen=True)
class CfgAdv:
    nh: int = 100
    m: int = 99
    alpha: float = 1e-3
    beta: float = 0.5
    w: float = 0.2
    a: float = 0.3
    omega: float = 2 * np.pi
    sigma: float = 0.07
    phi0: float = 1.0
    T: float = 2.0
    Nt: int = 640
    ya: float = 0.05
    yb: float = 0.95
    eps: float = 1.0
    bx: float = 0.0
    by: float = 0.0
    stab: bool = True
    shear: str = ""  # "" uniform bx; "lin": bx (0.25 + 1.5 s); "pois": bx 6 s (1 - s); "gen": bx (1 + gamma (s - 0.5))
    xo: float = 0.0  # observation strip x in [xo, xo + w] (xo > 0: downstream of the actuator)
    gamma: float = 0.0  # shear strength for shear = "gen"

    @property
    def dt(self) -> float:
        return self.T / self.Nt


def conv1d(n: int, half_first: bool) -> sp.csr_matrix:
    """int phi_j' phi_i on a uniform P1 grid (h-independent); node 0 has half support if half_first."""
    main = np.zeros(n)
    if half_first:
        main[0] = -0.5
    return sp.diags([-0.5 * np.ones(n - 1), main, 0.5 * np.ones(n - 1)], [-1, 0, 1], format="csr")


def art_diff(b, h: float, eps: float, on: bool):
    """Il'in-Allen-Southwell streamline diffusion |b| h/2 (coth Pe_h - 1/Pe_h); b scalar or array."""
    b = np.abs(np.asarray(b, float))
    if not on:
        return np.zeros_like(b)
    pe = np.maximum(b * h / (2 * eps), 1e-8)
    return np.where(b > 0, b * h / 2 * (1.0 / np.tanh(pe) - 1.0 / pe), 0.0)


SHEAR = {"lin": lambda s: 0.25 + 1.5 * s, "pois": lambda s: 6.0 * s * (1.0 - s)}


def weighted_mass_y(nh: int, f) -> sp.csr_matrix:
    """int f(s) psi_i psi_j over the interior P1 nodes s_j = j h (j = 1..nh-1), 3-point Gauss per element."""
    h = 1.0 / nh
    xq, wq = np.polynomial.legendre.leggauss(3)
    rows, cols, vals = [], [], []
    for k in range(nh):
        sq = (k + 0.5 * (xq + 1)) * h
        fw = f(sq) * wq * h / 2
        phi = [1 - (sq / h - k), sq / h - k]  # local shape functions of nodes k, k+1
        for a in range(2):
            for b in range(2):
                i, j = k + a - 1, k + b - 1  # interior index (node 0 and node nh are Dirichlet)
                if 0 <= i < nh - 1 and 0 <= j < nh - 1:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(np.sum(fw * phi[a] * phi[b])))
    return sp.csr_matrix((vals, (rows, cols)), shape=(nh - 1, nh - 1))


def assemble(cfg: CfgAdv) -> moving2d.MoveOps:
    nh, h = cfg.nh, 1.0 / cfg.nh
    Mx, Kx, Mxo, gx, _ = one_d(nh, cfg.xo, cfg.xo + cfg.w, drop_first=False)
    My, Ky, Myo, _, _ = one_d(nh, cfg.ya, cfg.yb, drop_first=True)
    E = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, nh))
    dy = float(art_diff(cfg.by, h, cfg.eps, cfg.stab))
    if cfg.shear:  # bx(s): convection and streamline diffusion weighted along s
        prof = (lambda s: 1.0 + cfg.gamma * (s - 0.5)) if cfg.shear == "gen" else SHEAR[cfg.shear]
        Xconv = sp.kron(weighted_mass_y(nh, lambda s: cfg.bx * prof(s)), conv1d(nh, True))
        Xdiff = sp.kron(cfg.eps * My + weighted_mass_y(nh, lambda s: art_diff(cfg.bx * prof(s), h, cfg.eps, cfg.stab)), Kx)
    else:
        Xconv = cfg.bx * sp.kron(My, conv1d(nh, True))
        Xdiff = (cfg.eps + float(art_diff(cfg.bx, h, cfg.eps, cfg.stab))) * sp.kron(My, Kx)
    K = ((cfg.eps + dy) * sp.kron(Ky, Mx) + Xdiff + cfg.beta * sp.kron(My, E)
         + Xconv + cfg.by * sp.kron(conv1d(nh - 1, False), Mx)).tocsr()
    M = sp.kron(My, Mx, format="csr")
    M_obs = sp.kron(Myo, Mxo, format="csr")
    P = control_basis(nh, cfg.m)
    e0c = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, 1))
    B = sp.kron(sp.csr_matrix(My @ P), e0c, format="csr")
    S = np.unique(np.concatenate([M_obs.nonzero()[0], np.nonzero(np.kron(np.ones(nh - 1), gx))[0]]))
    cy = RobinConfig(alpha=1.0, beta=0.0, xa=0.0, xb=1.0, nx=nh, Nt=1)
    return moving2d.MoveOps(cfg=cfg, M=M, K=K, A=(M / cfg.dt + K).tocsr(), M_obs=M_obs, B=B, Mu=P.T @ (My @ P), gx=gx, S=S,
                            n=M.shape[0], cy=cy)


class Sweeps(moving2d.Sweeps):
    """moving2d sweeps with the adjoint solved by A^T (nonsymmetric state operator)."""

    def adjoint(self, yS, with_target: bool):
        o = self.ops
        p = np.zeros(o.n)
        out = np.zeros((self.N, o.cfg.m))
        for k in range(self.N - 1, -1, -1):
            src = self.Mdt @ p
            if with_target:
                src = src + np.kron(self.GY[k], o.gx)
            if yS is not None:
                src = src - self.MoS @ yS[k]
            p = self.lu.solve(src, trans="T")
            out[k] = self.BT @ p
        return out

    def forward_strip(self, C, n_snap: int = 8):
        """Strip-averaged state (x in [xo, xo + w]) over (t, s) and n_snap full snapshots, without storing all of Y."""
        o, c = self.ops, self.ops.cfg
        y = np.zeros(o.n)
        ix0, ix1 = int(round(c.xo * c.nh)), int(round((c.xo + c.w) * c.nh)) + 1
        strip = np.zeros((self.N, c.nh - 1))
        ks = np.linspace(self.N // n_snap, self.N, n_snap).astype(int) - 1
        snaps = []
        for k in range(self.N):
            y = self.lu.solve(self.Mdt @ y + o.B @ C[k])
            strip[k] = y.reshape(c.nh - 1, c.nh)[:, ix0:ix1].mean(axis=1)
            if k in ks:
                snaps.append(y.copy())
        return strip, ks, np.stack(snaps)


def pcg_solve(ops, GY, C0=None, tol: float = 1e-10, maxit: int = 3000):
    """moving2d.pcg_solve with the transposed adjoint."""
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
