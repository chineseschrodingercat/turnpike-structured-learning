#!/usr/bin/env python3
"""Multi-input boundary control on x = 0 for the 2D problem of robin2d.py.

Control u(s, t) = sum_l c_l(t) chi_l(s), with chi_l hat functions on m equispaced interior points of the
edge (m = nh - 1 gives the fine boundary P1 space; m = 1 is the uniform profile). Discretely
    A y_k - (M/dt) y_{k-1} - B c_k = 0,            B = (My P) (x) e0  (n x m),
    A p_k - (M/dt) p_{k+1} = G - M_obs y_k,
    alpha Mu c_k = B^T p_k,                        Mu = P^T My P  (control mass on the edge),
so in the reduced form (alpha I (x) Mu + S^T C S) c = S^T G with S: c -> y. Exact solvers:
    pcg_solve       matrix-free PCG, one forward + one adjoint sweep per iteration, preconditioner (alpha Mu)^{-1};
    dense_reduced   build the (N m) x (N m) Hessian from m impulse responses (block Toeplitz, small N m only);
    solve_kkt_mi    monolithic KKT (tiny grids, validation).
Turnpike plateau (static optimum): (alpha Mu + Psi^T M_obs Psi) u_bar = Psi^T G, Psi = K^{-1} B.
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
from robin2d import Cfg2D, Ops2D, assemble, one_d  # noqa: E402
from robin_solver import block_lower  # noqa: E402


def control_basis(nh: int, m: int) -> np.ndarray:
    y = np.arange(1, nh) / nh
    if m == 1:
        return np.ones((nh - 1, 1))
    s = np.arange(1, m + 1) / (m + 1)
    return np.maximum(0.0, 1.0 - np.abs(y[:, None] - s[None, :]) * (m + 1))


@dataclass(frozen=True)
class MIOps:
    base: Ops2D
    B: sp.csr_matrix
    Mu: np.ndarray
    P: np.ndarray
    m: int


def assemble_mi(cfg: Cfg2D, m: int) -> MIOps:
    ops = assemble(cfg)
    My = one_d(cfg.nh, 0.0, 1.0, drop_first=True)[0]
    P = control_basis(cfg.nh, m)
    e0 = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(cfg.nh, 1))
    B = sp.kron(sp.csr_matrix(My @ P), e0, format="csr")
    return MIOps(base=ops, B=B, Mu=P.T @ (My @ P), P=P, m=m)


class Sweeps:
    """Forward / adjoint sweeps with one sparse LU (A is symmetric, so the adjoint reuses it)."""

    def __init__(self, mi: MIOps, N: int):
        o = mi.base
        self.mi, self.N = mi, N
        self.lu = spla.splu(o.A.tocsc())
        self.Mdt = (o.M / o.cfg.dt).tocsr()
        self.S = o.S
        self.MoS = o.M_obs[:, o.S].tocsr()
        self.BT = mi.B.T.tocsr()
        self.n_solves = 0

    def forward(self, c: np.ndarray) -> np.ndarray:
        y = np.zeros(self.mi.base.n)
        yS = np.zeros((self.N, len(self.S)))
        for k in range(self.N):
            y = self.lu.solve(self.Mdt @ y + self.mi.B @ c[k])
            yS[k] = y[self.S]
        self.n_solves += self.N
        return yS

    def adjoint(self, yS: np.ndarray | None, G: np.ndarray | None) -> np.ndarray:
        """Returns B^T p_k for the source (G if given) - M_obs y_k (if yS given)."""
        p = np.zeros(self.mi.base.n)
        out = np.zeros((self.N, self.mi.m))
        for k in range(self.N - 1, -1, -1):
            src = self.Mdt @ p
            if G is not None:
                src = src + G
            if yS is not None:
                src = src - self.MoS @ yS[k]
            p = self.lu.solve(src)
            out[k] = self.BT @ p
        self.n_solves += self.N
        return out


def pcg_solve(mi: MIOps, T: float, tol: float = 1e-10, maxit: int = 1000, c0: np.ndarray | None = None):
    o = mi.base
    alpha = o.cfg.alpha
    N = int(round(T / o.cfg.dt))
    sw = Sweeps(mi, N)
    t0 = time.perf_counter()
    g = sw.adjoint(None, o.G)
    Minv = np.linalg.inv(alpha * mi.Mu)

    def op(c):
        return alpha * c @ mi.Mu.T - sw.adjoint(sw.forward(c), None)

    c = np.zeros((N, mi.m)) if c0 is None else c0.copy()
    r = g - op(c) if c0 is not None else g.copy()
    z = r @ Minv.T
    d = z.copy()
    rz = float(np.sum(r * z))
    g_norm = float(np.sqrt(np.sum(g * (g @ Minv.T))))
    it = 0
    for it in range(1, maxit + 1):
        Ad = op(d)
        a = rz / float(np.sum(d * Ad))
        c += a * d
        r -= a * Ad
        z = r @ Minv.T
        rz_new = float(np.sum(r * z))
        if np.sqrt(max(rz_new, 0.0)) <= tol * g_norm:
            break
        d = z + (rz_new / rz) * d
        rz = rz_new
    return c, {"iters": it, "seconds": time.perf_counter() - t0, "sparse_solves": sw.n_solves, "N": N}


def dense_reduced(mi: MIOps, T: float):
    """Exact solve through the dense (N m) x (N m) reduced Hessian (block Toeplitz, from impulse responses)."""
    o = mi.base
    alpha, m = o.cfg.alpha, mi.m
    N = int(round(T / o.cfg.dt))
    t0 = time.perf_counter()
    lu = spla.splu(o.A.tocsc())
    Mdt = (o.M / o.cfg.dt).tocsr()
    S = o.S
    Y = lu.solve(mi.B.toarray())  # (n, m), response one step after a unit impulse in each input
    aS = np.zeros((N, len(S), m))
    aS[0] = Y[S]
    for k in range(1, N):
        Y = lu.solve(Mdt @ Y)
        aS[k] = Y[S]
    t1 = time.perf_counter()
    Amat = aS.transpose(0, 2, 1).reshape(N * m, len(S))
    Moo = o.M_obs[S][:, S].toarray()
    R4 = (Amat @ (Moo @ Amat.T)).reshape(N, m, N, m)
    t2 = time.perf_counter()
    H4 = np.zeros((N, m, N, m))
    for dd in range(N):
        D = np.diagonal(R4, offset=-dd, axis1=0, axis2=2)  # (m, m, N - dd): D[l, k, q] = R4[q + dd, l, q, k]
        cs = np.cumsum(D, axis=2)
        j = np.arange(dd, N)
        blk = cs[:, :, N - 1 - j].transpose(2, 0, 1)  # (len j, m, m) = H[(j - dd, l), (j, k)]
        H4[j - dd, :, j, :] = blk
        H4[j, :, j - dd, :] = blk.transpose(0, 2, 1)
    Hm = H4.reshape(N * m, N * m) + alpha * np.kron(np.eye(N), mi.Mu)
    t3 = time.perf_counter()
    sw = Sweeps(mi, N)
    g = sw.adjoint(None, o.G).reshape(-1)
    c = sla.solve(Hm, g, assume_a="pos").reshape(N, m)
    t4 = time.perf_counter()
    return c, {"impulse_s": t1 - t0, "R_s": t2 - t1, "H_assembly_s": t3 - t2, "solve_s": t4 - t3, "total_s": t4 - t0,
               "hessian_GB": (N * m) ** 2 * 8 / 1e9}


def static_optimum_mi(mi: MIOps) -> np.ndarray:
    o = mi.base
    Psi = spla.splu(o.K.tocsc()).solve(mi.B.toarray())
    Hs = o.cfg.alpha * mi.Mu + Psi.T @ (o.M_obs @ Psi)
    return np.linalg.solve(Hs, Psi.T @ o.G)


def solve_kkt_mi(cfg: Cfg2D, m: int):
    mi = assemble_mi(cfg, m)
    o = mi.base
    N = cfg.Nt
    I_t = sp.eye(N, format="csr")
    Ay = block_lower(o.A, (-o.M / cfg.dt).tocsr(), N)
    Ap = Ay.T.tocsr()
    Bb = -sp.kron(I_t, mi.B, format="csr")
    C = sp.kron(I_t, o.M_obs, format="csr")
    Au = cfg.alpha * sp.kron(I_t, sp.csr_matrix(mi.Mu), format="csr")
    KKT = sp.bmat([[Ay, None, Bb], [C, Ap, None], [None, Bb.T, Au]], format="csc")
    rhs = np.concatenate([np.zeros(o.n * N), np.tile(o.G, N), np.zeros(m * N)])
    sol = spla.spsolve(KKT, rhs)
    return sol[2 * o.n * N :].reshape(N, m), float(np.linalg.norm(KKT @ sol - rhs) / np.linalg.norm(rhs))
