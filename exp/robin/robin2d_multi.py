#!/usr/bin/env python3
"""Competing multi-region targets for the multi-input 2D problem (robin2d_mi.py).

Tracking on omega = union of K disjoint rectangles omega_k = [xa_k, xa_k + wx] x [band_k] with target tau_k
on omega_k:  J = 1/2 sum_k int int_{omega_k} (y - tau_k)^2 + alpha/2 int int u^2 ds dt.
    M_obs = sum_k My_k (x) Mx_k,   G = sum_k tau_k G_k,  G_k = gy_k (x) gx_k.
u is linear in tau: u = sum_k tau_k u^(k), where u^(k) solves the problem with target 1 on omega_k and 0 on the
other regions (all regions still observed). The static plateau of each u^(k) is closed form.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Ops2D, one_d  # noqa: E402
from robin2d_mi import MIOps, control_basis  # noqa: E402

BANDS = ((0.05, 0.30), (0.375, 0.625), (0.70, 0.95))
WX = 0.25


@dataclass(frozen=True)
class CfgMulti:
    nh: int = 100
    alpha: float = 1e-2
    beta: float = 1.0
    xas: tuple = (0.1, 0.3, 0.5)
    taus: tuple = (1.0, 0.0, 2.0)
    T: float = 1.0
    Nt: int = 320
    bands: tuple = BANDS
    wx: float = WX

    @property
    def dt(self) -> float:
        return self.T / self.Nt


def assemble_multi(cfg: CfgMulti, m: int, taus=None) -> MIOps:
    """Multi-input ops; G uses `taus` (default cfg.taus). Returns MIOps whose base.G is the weighted target load."""
    nh = cfg.nh
    Mx, Kx, _, _, _ = one_d(nh, 0.0, 1.0, drop_first=False)
    My, Ky, _, _, by = one_d(nh, 0.0, 1.0, drop_first=True)
    E = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, nh))
    M = sp.kron(My, Mx, format="csr")
    K = (sp.kron(Ky, Mx) + sp.kron(My, Kx) + cfg.beta * sp.kron(My, E)).tocsr()
    M_obs = None
    Gk = []
    for xa, (ya, yb) in zip(cfg.xas, cfg.bands):
        _, _, Mxo, gx, _ = one_d(nh, xa, xa + cfg.wx, drop_first=False)
        _, _, Myo, gy, _ = one_d(nh, ya, yb, drop_first=True)
        blk = sp.kron(Myo, Mxo, format="csr")
        M_obs = blk if M_obs is None else (M_obs + blk).tocsr()
        Gk.append(np.kron(gy, gx))
    taus = cfg.taus if taus is None else taus
    G = sum(t * g for t, g in zip(taus, Gk))
    S = np.unique(np.concatenate([M_obs.nonzero()[0]] + [np.nonzero(g)[0] for g in Gk]))
    A = (M / cfg.dt + K).tocsr()
    e0 = np.zeros(nh)
    e0[0] = 1.0
    base = Ops2D(cfg=cfg, M=M, K=K, A=A, M_obs=M_obs, b=np.kron(by, e0), G=G, S=S, n=M.shape[0])
    P = control_basis(nh, m)
    e0c = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, 1))
    B = sp.kron(sp.csr_matrix(My @ P), e0c, format="csr")
    mi = MIOps(base=base, B=B, Mu=P.T @ (My @ P), P=P, m=m)
    object.__setattr__(mi, "Gk", Gk)  # per-region target loads (for the tau-linear decomposition)
    return mi


@dataclass(frozen=True)
class CfgStrip:
    """Observation strip next to the controlled edge: omega = [0, w] x [0.05, 0.95],
    target y_d(y) = coef_0 + sum_{k>=1} coef_k sin(k pi y) (constant in x and t)."""
    nh: int = 100
    alpha: float = 1e-2
    beta: float = 1.0
    w: float = 0.1
    coefs: tuple = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    T: float = 1.0
    Nt: int = 320
    ya: float = 0.05
    yb: float = 0.95

    @property
    def dt(self) -> float:
        return self.T / self.Nt


def assemble_strip(cfg: CfgStrip, m: int, coefs=None) -> MIOps:
    """Multi-input ops for the edge strip; Gk[k] is the load of the k-th target profile (1, sin(pi y), ...)."""
    from robin_solver import RobinConfig, interval_load

    nh = cfg.nh
    Mx, Kx, Mxo, gx, _ = one_d(nh, 0.0, cfg.w, drop_first=False)
    My, Ky, Myo, _, by = one_d(nh, cfg.ya, cfg.yb, drop_first=True)
    E = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, nh))
    M = sp.kron(My, Mx, format="csr")
    K = (sp.kron(Ky, Mx) + sp.kron(My, Kx) + cfg.beta * sp.kron(My, E)).tocsr()
    M_obs = sp.kron(Myo, Mxo, format="csr")
    cy = RobinConfig(alpha=1.0, beta=0.0, xa=0.0, xb=1.0, nx=nh, Nt=1)
    profiles = [lambda y, t: np.ones_like(np.asarray(y, float))] + [
        (lambda kk: (lambda y, t: np.sin(kk * np.pi * np.asarray(y, float))))(k) for k in range(1, len(cfg.coefs))]
    Gk = [np.kron(interval_load(f, cy, 0.0, cfg.ya, cfg.yb)[1:], gx) for f in profiles]
    coefs = cfg.coefs if coefs is None else coefs
    G = sum(c * g for c, g in zip(coefs, Gk))
    S = np.unique(np.concatenate([M_obs.nonzero()[0]] + [np.nonzero(g)[0] for g in Gk]))
    A = (M / cfg.dt + K).tocsr()
    e0 = np.zeros(nh)
    e0[0] = 1.0
    base = Ops2D(cfg=cfg, M=M, K=K, A=A, M_obs=M_obs, b=np.kron(by, e0), G=G, S=S, n=M.shape[0])
    P = control_basis(nh, m)
    e0c = sp.csr_matrix((np.array([1.0]), (np.array([0]), np.array([0]))), shape=(nh, 1))
    mi = MIOps(base=base, B=sp.kron(sp.csr_matrix(My @ P), e0c, format="csr"), Mu=P.T @ (My @ P), P=P, m=m)
    object.__setattr__(mi, "Gk", Gk)
    return mi


def solve_kkt_multi(cfg: CfgMulti, m: int) -> tuple[np.ndarray, float]:
    """Monolithic KKT for validation on tiny grids (same block structure as robin2d_mi.solve_kkt_mi)."""
    from robin_solver import block_lower

    mi = assemble_multi(cfg, m)
    o = mi.base
    N = cfg.Nt
    I_t = sp.eye(N, format="csr")
    Ay = block_lower(o.A, (-o.M / cfg.dt).tocsr(), N)
    Bb = -sp.kron(I_t, mi.B, format="csr")
    KKT = sp.bmat([[Ay, None, Bb], [sp.kron(I_t, o.M_obs, format="csr"), Ay.T.tocsr(), None],
                   [None, Bb.T, cfg.alpha * sp.kron(I_t, sp.csr_matrix(mi.Mu), format="csr")]], format="csc")
    rhs = np.concatenate([np.zeros(o.n * N), np.tile(o.G, N), np.zeros(m * N)])
    sol = spla.spsolve(KKT, rhs)
    return sol[2 * o.n * N :].reshape(N, m), float(np.linalg.norm(KKT @ sol - rhs) / np.linalg.norm(rhs))


def static_optima_per_region(mi: MIOps) -> np.ndarray:
    """(K, m): static optimum for target 1 on region k only (all regions observed)."""
    o = mi.base
    Psi = spla.splu(o.K.tocsc()).solve(mi.B.toarray())
    Hs = o.cfg.alpha * mi.Mu + Psi.T @ (o.M_obs @ Psi)
    return np.linalg.solve(Hs, np.stack([Psi.T @ g for g in mi.Gk], axis=1)).T
