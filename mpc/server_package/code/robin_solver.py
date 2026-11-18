#!/usr/bin/env python3
"""P1 FEM / backward-Euler KKT solver for Robin boundary control with subzone observation.

Re-implemented from equations (2)-(7) of the report "Robin Boundary Control of
the 1D Heat Equation: a Four-Configuration Study" (4 August 2026). The upstream
``ctrl1d/solver_robin_subzone_p1.py`` was not available, so this file follows
the report and the conventions of ``ctrl1d/solver_p1.py``.

    y_t - kappa y_xx = f                      in (0,L) x (0,T),
    -kappa y_x(0,t) + beta y(0,t) = u(t),     beta >= 0,
    y(L,t) = 0,   y(x,0) = y0(x),

    J(y,u) = 1/2 int_0^T int_{omega_o} (y - y_d)^2 dx dt + alpha/2 int_0^T u(t)^2 dt,
    omega_o = (x_a, x_b).

Free P1 dofs are x_i = i h, i = 0..nx-1 (x = L is eliminated). With
K_beta = K + beta e0 e0^T, B = -I_Nt (x) e0 and C = I_Nt (x) M_obs(omega_o):

    [A_y  0    B ] [Y]   [F]
    [C    A_p  0 ] [P] = [G],      A_p = A_y^T,
    [0    B^T aI ] [U]   [0]

so the control row reads alpha u_k - p_k(0) = 0 (physical sign, report eq. (5)).
The KKT system is exactly the discrete optimality system of the dt-weighted cost
implemented in ``eval_cost``.

Reduced form. With f = 0 and y0 = 0 the state is Y = S U, where S is block
lower-triangular Toeplitz (backward Euler with constant A). In the inner
products dt*u^T v (control) and dt*sum_k y_k^T M_obs z_k (observation) the
reduced optimality condition is (H + alpha I) u = g with H = S^T C S and
g = S^T (G - C Y_free). ``reduced_system`` builds H and g from a single impulse
response, without forming S.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss

ScalarField = Callable[[np.ndarray, float], np.ndarray]
InitialField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RobinConfig:
    L: float = 1.0
    T: float = 1.0
    kappa: float = 1.0
    alpha: float = 1e-2
    beta: float = 1.0
    xa: float = 0.25
    xb: float = 0.75
    nx: int = 160
    Nt: int = 320
    n_gauss: int = 4

    @property
    def h(self) -> float:
        return self.L / self.nx

    @property
    def dt(self) -> float:
        return self.T / self.Nt


@dataclass(frozen=True)
class RobinOperators:
    cfg: RobinConfig
    x_free: np.ndarray
    M: sp.csr_matrix
    K: sp.csr_matrix
    K_beta: sp.csr_matrix
    M_obs: sp.csr_matrix
    A: sp.csr_matrix
    A_sub: sp.csr_matrix
    e0: np.ndarray


@dataclass(frozen=True)
class ProblemData:
    F_list: list[np.ndarray]
    G_list: list[np.ndarray]
    yd_sq: np.ndarray
    y0: np.ndarray


@dataclass(frozen=True)
class KKTResult:
    cfg: RobinConfig
    operators: RobinOperators
    data: ProblemData
    Y: np.ndarray
    P: np.ndarray
    U: np.ndarray
    rel_res: float
    J_track: float
    J_ctrl: float
    J_total: float


@dataclass(frozen=True)
class ReducedSystem:
    impulse: np.ndarray
    H: np.ndarray
    g: np.ndarray


def _as_array(values: np.ndarray | float, like: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != like.shape:
        return np.broadcast_to(arr, like.shape).astype(float)
    return arr


def _overlaps(cfg: RobinConfig, a: float, b: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elements overlapping [a, b] and the overlap endpoints."""
    xl = np.arange(cfg.nx, dtype=float) * cfg.h
    xr = xl + cfg.h
    lo = np.maximum(a, xl)
    hi = np.minimum(b, xr)
    elems = np.nonzero(hi - lo > 1e-14 * cfg.h)[0]
    return elems, lo[elems], hi[elems]


def interval_mass(cfg: RobinConfig, a: float, b: float) -> sp.csr_matrix:
    """Exact int_a^b phi_i phi_j dx on the free dofs (partial elements included)."""
    xi, w = leggauss(2)  # exact for products of two linear functions
    elems, lo, hi = _overlaps(cfg, a, b)
    h = cfg.h
    xl = elems * h
    xr = xl + h
    xg = 0.5 * (hi - lo)[:, None] * xi + 0.5 * (hi + lo)[:, None]
    wg = 0.5 * (hi - lo)[:, None] * w
    phis = [(xr[:, None] - xg) / h, (xg - xl[:, None]) / h]
    rows, cols, vals = [], [], []
    for i in range(2):
        for j in range(2):
            rows.append(elems + i)
            cols.append(elems + j)
            vals.append(np.sum(wg * phis[i] * phis[j], axis=1))
    n = cfg.nx + 1
    full = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    # x = L carries the homogeneous Dirichlet condition and is not a free dof.
    return full[: cfg.nx, : cfg.nx].tocsr()


def interval_load(func: ScalarField, cfg: RobinConfig, t: float, a: float, b: float) -> np.ndarray:
    """int_a^b func(x,t) phi_i dx on the free dofs."""
    xi, w = leggauss(cfg.n_gauss)
    elems, lo, hi = _overlaps(cfg, a, b)
    h = cfg.h
    xl = elems * h
    xr = xl + h
    xg = 0.5 * (hi - lo)[:, None] * xi + 0.5 * (hi + lo)[:, None]
    wg = 0.5 * (hi - lo)[:, None] * w
    fg = _as_array(func(xg, t), xg)
    out = np.zeros(cfg.nx + 1)
    np.add.at(out, elems, np.sum(wg * fg * (xr[:, None] - xg) / h, axis=1))
    np.add.at(out, elems + 1, np.sum(wg * fg * (xg - xl[:, None]) / h, axis=1))
    return out[: cfg.nx]


def interval_integral(func: ScalarField, cfg: RobinConfig, t: float, a: float, b: float) -> float:
    """int_a^b func(x,t) dx with the same element-wise Gauss rule."""
    xi, w = leggauss(cfg.n_gauss)
    _, lo, hi = _overlaps(cfg, a, b)
    xg = 0.5 * (hi - lo)[:, None] * xi + 0.5 * (hi + lo)[:, None]
    wg = 0.5 * (hi - lo)[:, None] * w
    return float(np.sum(wg * _as_array(func(xg, t), xg)))


def assemble_operators(cfg: RobinConfig) -> RobinOperators:
    """Analytic P1 stiffness, exact (sub)interval mass matrices, Robin term on e0."""
    if not (0.0 <= cfg.xa < cfg.xb <= cfg.L):
        raise ValueError(f"need 0 <= xa < xb <= L, got ({cfg.xa}, {cfg.xb})")
    n = cfg.nx
    h = cfg.h
    M = interval_mass(cfg, 0.0, cfg.L)

    diag_k = np.full(n, 2.0 * cfg.kappa / h)
    diag_k[0] = cfg.kappa / h
    off_k = np.full(n - 1, -cfg.kappa / h)
    K = sp.diags([off_k, diag_k, off_k], [-1, 0, 1], shape=(n, n), format="csr")

    e0 = np.zeros(n)
    e0[0] = 1.0
    K_beta = (K + cfg.beta * sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(n, n))).tocsr()
    M_obs = interval_mass(cfg, cfg.xa, cfg.xb)

    A = (M / cfg.dt + K_beta).tocsr()
    A_sub = (-M / cfg.dt).tocsr()
    x_free = np.arange(n, dtype=float) * h
    return RobinOperators(cfg=cfg, x_free=x_free, M=M, K=K, K_beta=K_beta, M_obs=M_obs, A=A, A_sub=A_sub, e0=e0)


def assemble_problem_data(
    cfg: RobinConfig,
    ops: RobinOperators,
    yd_func: ScalarField,
    f_func: ScalarField | None = None,
    y0_func: InitialField | None = None,
    h_func: ScalarField | None = None,
) -> ProblemData:
    """State loads F_k, adjoint loads G_k = int_omega y_d phi (+ int_0^L h phi), and int_omega y_d^2.

    ``h_func`` is an extra adjoint source on the whole domain; production data
    leave it at None. It exists for manufactured-solution tests.
    """
    y0 = np.zeros(cfg.nx)
    if y0_func is not None:
        y0 = _as_array(y0_func(ops.x_free), ops.x_free)

    F_list: list[np.ndarray] = []
    G_list: list[np.ndarray] = []
    yd_sq = np.zeros(cfg.Nt)
    for k in range(1, cfg.Nt + 1):
        tk = k * cfg.dt
        Fk = np.zeros(cfg.nx) if f_func is None else interval_load(f_func, cfg, tk, 0.0, cfg.L)
        Gk = interval_load(yd_func, cfg, tk, cfg.xa, cfg.xb)
        if h_func is not None:
            Gk = Gk + interval_load(h_func, cfg, tk, 0.0, cfg.L)
        F_list.append(Fk)
        G_list.append(Gk)
        yd_sq[k - 1] = interval_integral(lambda x, t: _as_array(yd_func(x, t), x) ** 2, cfg, tk, cfg.xa, cfg.xb)

    F_list[0] = F_list[0] + (ops.M @ y0) / cfg.dt
    return ProblemData(F_list=F_list, G_list=G_list, yd_sq=yd_sq, y0=y0)


def block_lower(A: sp.spmatrix, Bsub: sp.spmatrix, nblocks: int) -> sp.csr_matrix:
    rows = []
    for i in range(nblocks):
        rows.append([Bsub if j == i - 1 else (A if j == i else None) for j in range(nblocks)])
    return sp.bmat(rows, format="csr")


def build_kkt(cfg: RobinConfig, ops: RobinOperators) -> sp.csr_matrix:
    """Monolithic KKT matrix of report eq. (7)."""
    Nt = cfg.Nt
    I_t = sp.eye(Nt, format="csr")
    Ay = block_lower(ops.A, ops.A_sub, Nt)
    Ap = Ay.T.tocsr()
    B = -sp.kron(I_t, sp.csr_matrix(ops.e0).T, format="csr")
    C = sp.kron(I_t, ops.M_obs, format="csr")
    Au = cfg.alpha * I_t
    return sp.bmat([[Ay, None, B], [C, Ap, None], [None, B.T, Au]], format="csr")


def solve_kkt(
    cfg: RobinConfig,
    yd_func: ScalarField,
    f_func: ScalarField | None = None,
    y0_func: InitialField | None = None,
    h_func: ScalarField | None = None,
) -> KKTResult:
    """Solve the full all-at-once KKT system."""
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, yd_func, f_func, y0_func, h_func)
    KKT = build_kkt(cfg, ops)
    n, Nt = cfg.nx, cfg.Nt
    rhs = np.concatenate([np.concatenate(data.F_list), np.concatenate(data.G_list), np.zeros(Nt)])
    sol = spla.spsolve(KKT.tocsc(), rhs)
    rel_res = float(np.linalg.norm(KKT @ sol - rhs) / max(np.linalg.norm(rhs), 1.0))
    Y = sol[: n * Nt].reshape(Nt, n)
    P = sol[n * Nt : 2 * n * Nt].reshape(Nt, n)
    U = sol[2 * n * Nt :]
    J_total, J_track, J_ctrl = eval_cost(cfg, ops, data, Y, U)
    return KKTResult(cfg=cfg, operators=ops, data=data, Y=Y, P=P, U=U, rel_res=rel_res,
                     J_track=J_track, J_ctrl=J_ctrl, J_total=J_total)


def solve_state(ops: RobinOperators, data: ProblemData, u: np.ndarray) -> np.ndarray:
    """Forward solve: A y_k = F_k + (M/dt) y_{k-1} + e0 u_k."""
    cfg = ops.cfg
    A_lu = spla.splu(ops.A.tocsc())
    Mdt = ops.M / cfg.dt
    Y = np.zeros((cfg.Nt, cfg.nx))
    y_prev = np.zeros(cfg.nx)
    for k in range(cfg.Nt):
        rhs = data.F_list[k] + ops.e0 * u[k]
        if k > 0:
            rhs = rhs + Mdt @ y_prev
        y_prev = A_lu.solve(rhs)
        Y[k] = y_prev
    return Y


def solve_adjoint(ops: RobinOperators, data: ProblemData, Y: np.ndarray) -> np.ndarray:
    """Backward solve: A p_k = G_k - M_obs y_k + (M/dt) p_{k+1}, p_{Nt+1} = 0."""
    cfg = ops.cfg
    A_lu = spla.splu(ops.A.tocsc())
    Mdt = ops.M / cfg.dt
    P = np.zeros((cfg.Nt, cfg.nx))
    p_next = np.zeros(cfg.nx)
    for k in range(cfg.Nt - 1, -1, -1):
        rhs = data.G_list[k] - ops.M_obs @ Y[k] + Mdt @ p_next
        p_next = A_lu.solve(rhs)
        P[k] = p_next
    return P


def eval_cost(cfg: RobinConfig, ops: RobinOperators, data: ProblemData, Y: np.ndarray, U: np.ndarray) -> tuple[float, float, float]:
    """(J_total, J_track, J_ctrl) of the dt-weighted discrete cost."""
    J_track = 0.0
    for k in range(cfg.Nt):
        y = Y[k]
        J_track += 0.5 * cfg.dt * (float(y @ (ops.M_obs @ y)) - 2.0 * float(y @ data.G_list[k]) + data.yd_sq[k])
    J_ctrl = 0.5 * cfg.alpha * cfg.dt * float(U @ U)
    return J_track + J_ctrl, J_track, J_ctrl


def eval_gradient(ops: RobinOperators, data: ProblemData, u: np.ndarray) -> np.ndarray:
    """Euclidean gradient of the dt-weighted cost: dt * (alpha u_k - p_k(0))."""
    Y = solve_state(ops, data, u)
    P = solve_adjoint(ops, data, Y)
    return ops.cfg.dt * (ops.cfg.alpha * u - P[:, 0])


def impulse_response(ops: RobinOperators) -> np.ndarray:
    """State response a_m (m = 0..Nt-1 steps after) to a unit control in the first step, f = y0 = 0."""
    cfg = ops.cfg
    A_lu = spla.splu(ops.A.tocsc())
    Mdt = ops.M / cfg.dt
    a = np.zeros((cfg.Nt, cfg.nx))
    a[0] = A_lu.solve(ops.e0)
    for m in range(1, cfg.Nt):
        a[m] = A_lu.solve(Mdt @ a[m - 1])
    return a


def dense_control_to_state(ops: RobinOperators, a: np.ndarray | None = None) -> np.ndarray:
    """Dense S with Y.ravel() = S @ U (only for small checks)."""
    cfg = ops.cfg
    a = impulse_response(ops) if a is None else a
    n, Nt = cfg.nx, cfg.Nt
    S = np.zeros((Nt * n, Nt))
    for j in range(Nt):
        S[j * n :, j] = a[: Nt - j].ravel()
    return S


def reduced_system(ops: RobinOperators, data: ProblemData) -> ReducedSystem:
    """H = S^T C S and g = S^T (G - C Y_free) from one impulse response.

    For i <= j: H[i, j] = sum_{m=0}^{Nt-1-j} a_{m+j-i}^T M_obs a_m.
    g_j = sum_{m=0}^{Nt-1-j} a_m^T (G - C Y_free)_{m+j}.
    """
    cfg = ops.cfg
    Nt = cfg.Nt
    a = impulse_response(ops)
    R = a @ (ops.M_obs @ a.T)  # R[p, q] = a_p^T M_obs a_q
    H = np.zeros((Nt, Nt))
    for d in range(Nt):
        cs = np.cumsum(np.diagonal(R, offset=-d))  # cs[L] = sum_{m<=L} R[m+d, m]
        j = np.arange(d, Nt)
        vals = cs[Nt - 1 - j]
        H[j - d, j] = vals
        H[j, j - d] = vals

    G = np.asarray(data.G_list)
    if any(np.any(Fk) for Fk in data.F_list):
        Y_free = solve_state(ops, data, np.zeros(Nt))
        G = G - Y_free @ ops.M_obs.T
    E = a @ G.T  # E[m, k] = a_m^T G_k
    g = np.array([np.trace(E, offset=j) for j in range(Nt)])
    return ReducedSystem(impulse=a, H=H, g=g)


def reduced_control(red: ReducedSystem, alpha: float, rank: int | None = None) -> np.ndarray:
    """u = V_k diag(1/(sigma^2 + alpha)) V_k^T g; rank=None gives the exact solve."""
    if rank is None:
        return np.linalg.solve(red.H + alpha * np.eye(len(red.g)), red.g)
    s2, V = np.linalg.eigh(red.H)
    order = np.argsort(s2)[::-1][:rank]
    Vk = V[:, order]
    return Vk @ ((Vk.T @ red.g) / (np.clip(s2[order], 0.0, None) + alpha))
