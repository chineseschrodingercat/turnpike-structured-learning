#!/usr/bin/env python3
"""Moving-target edge control (moving2d geometry, linear heat equation) with box constraints u_a <= c_l(t) <= u_b on
the nodal control values, lumped control mass m_l (so the optimality condition is c = clip(B^T p / (alpha m_l))).

Projected Newton (Bertsekas): with gradient G = alpha m c - B^T p(c), the binding set is {c at a bound, G pointing
out}; on the free set the Newton direction comes from CG on (alpha m + S') (S' = S^T C S, one forward + one adjoint
sweep), on the binding set the scaled gradient; Armijo along the projected arc clip(c + s d). Monotone in J (plain
primal-dual active set cycles here for small alpha). J is quadratic, so J and G at a trial point cost one S' product.
u_a = -inf, u_b = +inf gives the unconstrained control (same lumped mass).
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import CfgMove, Sweeps, assemble, target_loads  # noqa: E402


def solve_box(cfg: CfgMove, ua: float, ub: float, C0=None, tol: float = 1e-9, maxit: int = 80, verbose: bool = False):
    o = assemble(cfg)
    GY, _ = target_loads(o)
    sw = Sweeps(o, GY)
    am = cfg.alpha * np.asarray(o.Mu.sum(axis=1)).ravel()[None, :]
    t0 = time.perf_counter()
    g = sw.adjoint(None, True)  # B^T p for c = 0 (target only)

    def Sp(X):  # S^T C S X (observation part of the reduced Hessian)
        return -sw.adjoint(sw.forward(X), False)

    def JG(X):  # J (up to a constant, without the dt factor) and its gradient
        HX = am * X + Sp(X)
        return 0.5 * float(np.sum(X * HX)) - float(np.sum(X * g)), HX - g

    C = np.zeros((cfg.Nt, cfg.m)) if C0 is None else np.clip(C0, ua, ub)
    J, G = JG(C)
    cg_total, it, res = 0, 0, np.inf
    for it in range(1, maxit + 1):
        step_fp = C - np.clip(C - G / am, ua, ub)
        res = float(np.linalg.norm(step_fp) / max(np.linalg.norm(C), 1e-300))
        if verbose:
            print(f"  pn {it}: J {J:.12e} fixed-point res {res:.2e} upper {np.mean(C >= ub):.3f} lower {np.mean(C <= ua):.3f} "
                  f"cg {cg_total}", flush=True)
        if res <= tol:
            break
        eps = min(1e-3 * max(np.abs(C).max(), 1e-12), float(np.abs(step_fp).max()))
        Bd = ((C <= ua + eps) & (G > 0)) | ((C >= ub - eps) & (G < 0))
        F = ~Bd
        eta = min(0.1, np.sqrt(res))
        X = np.zeros_like(C)
        R = np.where(F, -G, 0.0)
        Z = R / am
        D = Z.copy()
        rz = float(np.sum(R * Z))
        r0 = np.sqrt(rz)
        for _ in range(500):
            if np.sqrt(max(rz, 0.0)) <= eta * r0:
                break
            HD = np.where(F, am * D + Sp(D), 0.0)
            cg_total += 1
            al = rz / float(np.sum(D * HD))
            X += al * D
            R -= al * HD
            Z = R / am
            rz_new = float(np.sum(R * Z))
            D = Z + (rz_new / rz) * D
            rz = rz_new
        Dir = np.where(F, X, -G / am)
        s = 1.0
        while True:
            Cs = np.clip(C + s * Dir, ua, ub)
            Js, Gs = JG(Cs)
            if Js <= J + 1e-4 * float(np.sum(G * (Cs - C))) or s < 1e-10:
                break
            s *= 0.5
        C, J, G = Cs, Js, Gs
    return C, {"pn_iters": it, "cg_iters": cg_total, "fixed_point_res": res, "seconds": time.perf_counter() - t0,
               "frac_upper": float((C >= ub).mean()), "frac_lower": float((C <= ua).mean())}
