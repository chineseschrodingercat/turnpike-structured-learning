#!/usr/bin/env python3
"""2D Robin boundary control of the heat equation on the unit square, subzone tracking (Q1 tensor FEM).

    y_t - Laplace y = 0                      in (0,1)^2 x (0,T)
    -dy/dn + beta y = u(t)                   on x = 0      (scalar control, uniform along the edge)
    y = 0                                    on x = 1, y = 0, y = 1
    y(.,0) = 0,  J = 1/2 int_0^T int_omega (y - 1)^2 + alpha/2 int_0^T u^2,  omega = [xa,xb] x [ya,yb]

Free nodes: x_i = i h (i = 0..nh-1, x = 1 eliminated), y_j = j h (j = 1..nh-1). With the validated 1D
P1 matrices (robin_solver) and index = j * nh + i:
    M = My (x) Mx,   K = Ky (x) Mx + My (x) Kx + beta My (x) e0 e0^T,   b = (int psi_j) (x) e0,
    M_obs = My_obs (x) Mx_obs,   G = gy (x) gx                      (exact for Q1 on a tensor grid).
Backward Euler as in 1D: A y_k - (M/dt) y_{k-1} - b u_k = 0, adjoint A p_k - (M/dt) p_{k+1} = G - M_obs y_k,
alpha u_k = b^T p_k. The reduced Hessian is N x N (scalar control); only nodes in the support S of M_obs
enter it, so H_T needs N sparse solves and an N x N x |S| product.
First eigenvalue of (K, M) separates: lambda_1 = lambda_1^x(beta) + lambda_1^y.
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
from robin_solver import RobinConfig, assemble_operators, block_lower, interval_load, interval_mass  # noqa: E402

ONE = lambda x, t: np.ones_like(np.asarray(x, float))


@dataclass(frozen=True)
class Cfg2D:
    nh: int = 100
    alpha: float = 1e-2
    beta: float = 1.0
    xa: float = 0.3
    xb: float = 0.6
    ya: float = 0.3
    yb: float = 0.7
    T: float = 1.0
    Nt: int = 320

    @property
    def dt(self) -> float:
        return self.T / self.Nt


@dataclass(frozen=True)
class Ops2D:
    cfg: Cfg2D
    M: sp.csr_matrix
    K: sp.csr_matrix
    A: sp.csr_matrix
    M_obs: sp.csr_matrix
    b: np.ndarray
    G: np.ndarray
    S: np.ndarray          # support of M_obs / G
    n: int


def one_d(nh: int, a: float, b: float, drop_first: bool):
    c = RobinConfig(alpha=1.0, beta=0.0, xa=0.0, xb=1.0, nx=nh, Nt=1)
    o = assemble_operators(c)
    sl = slice(1, None) if drop_first else slice(None)
    M, K = o.M[sl, sl], o.K[sl, sl]
    Mo = interval_mass(c, a, b)[sl, sl]
    g = interval_load(ONE, c, 0.0, a, b)[sl]
    full = interval_load(ONE, c, 0.0, 0.0, 1.0)[sl]
    return M.tocsr(), K.tocsr(), Mo.tocsr(), g, full


def assemble(cfg: Cfg2D) -> Ops2D:
    nh = cfg.nh
    Mx, Kx, Mxo, gx, _ = one_d(nh, cfg.xa, cfg.xb, drop_first=False)  # x: Robin at 0 (node kept), Dirichlet at 1
    My, Ky, Myo, gy, by = one_d(nh, cfg.ya, cfg.yb, drop_first=True)  # y: Dirichlet at 0 and 1
    E = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, nh))
    e0 = np.zeros(nh)
    e0[0] = 1.0
    M = sp.kron(My, Mx, format="csr")
    K = (sp.kron(Ky, Mx) + sp.kron(My, Kx) + cfg.beta * sp.kron(My, E)).tocsr()
    M_obs = sp.kron(Myo, Mxo, format="csr")
    G = np.kron(gy, gx)
    b = np.kron(by, e0)
    S = np.unique(np.concatenate([M_obs.nonzero()[0], np.nonzero(G)[0]]))
    A = (M / cfg.dt + K).tocsr()
    return Ops2D(cfg=cfg, M=M, K=K, A=A, M_obs=M_obs, b=b, G=G, S=S, n=M.shape[0])


def lambda1(nh: int, beta: float) -> float:
    Mx, Kx, _, _, _ = one_d(nh, 0.0, 1.0, drop_first=False)
    My, Ky, _, _, _ = one_d(nh, 0.0, 1.0, drop_first=True)
    Kxb = Kx.toarray()
    Kxb[0, 0] += beta
    lx = sla.eigh(Kxb, Mx.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0]
    ly = sla.eigh(Ky.toarray(), My.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0]
    return float(lx + ly)


def static_optimum(ops: Ops2D) -> float:
    psi = spla.splu(ops.K.tocsc()).solve(ops.b)
    return float(psi @ ops.G / (psi @ (ops.M_obs @ psi) + ops.cfg.alpha))


def horizon_family(ops: Ops2D, T_list, steps_per_unit: int = 320, timing: dict | None = None) -> dict:
    """Exact controls u_T for every T in T_list (same dt), via the reduced N x N system."""
    t0 = time.perf_counter()
    n_max = steps_per_unit * max(T_list)
    lu = spla.splu(ops.A.tocsc())
    Mdt = (ops.M / ops.cfg.dt).tocsr()
    S = ops.S
    aS = np.zeros((n_max, len(S)))
    y = lu.solve(ops.b)
    aS[0] = y[S]
    for m in range(1, n_max):
        y = lu.solve(Mdt @ y)
        aS[m] = y[S]
    t1 = time.perf_counter()
    Mo = ops.M_obs[S][:, S].toarray()
    R = aS @ (Mo @ aS.T)
    aG_cs = np.cumsum(aS @ ops.G[S])
    t2 = time.perf_counter()
    sizes = [steps_per_unit * T for T in T_list]
    H = {N: np.zeros((N, N)) for N in sizes}
    for d in range(n_max):
        cs = np.cumsum(np.diagonal(R, offset=-d))
        for N in sizes:
            if N > d:
                j = np.arange(d, N)
                H[N][j - d, j] = cs[N - 1 - j]
                H[N][j, j - d] = cs[N - 1 - j]
    t3 = time.perf_counter()
    out = {}
    for T, N in zip(T_list, sizes):
        g = aG_cs[N - 1 - np.arange(N)]
        out[T] = np.linalg.solve(H[N] + ops.cfg.alpha * np.eye(N), g)
    t4 = time.perf_counter()
    if timing is not None:
        timing.update({"impulse_solves": t1 - t0, "R_product": t2 - t1, "H_assembly": t3 - t2, "dense_solves": t4 - t3, "S_size": int(len(S))})
    return out


def solve_kkt(cfg: Cfg2D):
    """Monolithic KKT (small grids only), same block structure as the 1D solver."""
    ops = assemble(cfg)
    N = cfg.Nt
    I_t = sp.eye(N, format="csr")
    Ay = block_lower(ops.A, (-ops.M / cfg.dt).tocsr(), N)
    Ap = Ay.T.tocsr()
    B = -sp.kron(I_t, sp.csr_matrix(ops.b).T, format="csr")
    C = sp.kron(I_t, ops.M_obs, format="csr")
    KKT = sp.bmat([[Ay, None, B], [C, Ap, None], [None, B.T, cfg.alpha * I_t]], format="csc")
    rhs = np.concatenate([np.zeros(ops.n * N), np.tile(ops.G, N), np.zeros(N)])
    sol = spla.spsolve(KKT, rhs)
    rel_res = float(np.linalg.norm(KKT @ sol - rhs) / np.linalg.norm(rhs))
    return sol[2 * ops.n * N :], rel_res, ops


def pencil_eig_seconds(cfg: Cfg2D) -> float:
    """Time of the dense symplectic-pencil eigendecomposition used by the exact ARE method (1D: hamiltonian_exact.py)."""
    ops = assemble(cfg)
    n = ops.n
    Mdt = (ops.M / cfg.dt).toarray()
    A = ops.A.toarray()
    Mo = ops.M_obs.toarray()
    Bb = np.outer(ops.b, ops.b) / cfg.alpha
    Z = np.zeros((n, n))
    L = np.block([[-Mdt, Z], [Mo, A]])
    R = np.block([[-A, Bb], [Z, Mdt]])
    t0 = time.perf_counter()
    sla.eig(L, R, right=True)
    return time.perf_counter() - t0
