#!/usr/bin/env python3
"""Small-grid checks for reaction_moving.py (Nagumo gradient / Hessian-vector by finite differences, cubic case equals
semilinear_moving.py) and box_moving.py (unconstrained = lumped-mass PCG, KKT fixed point, J below random feasible
perturbations)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reaction_moving as rm  # noqa: E402
import semilinear_moving as sm  # noqa: E402
from box_moving import solve_box  # noqa: E402
from moving2d import CfgMove  # noqa: E402

cf = CfgMove(nh=20, m=19, alpha=1e-3, beta=0.5, w=0.2, a=0.3, omega=2 * np.pi, sigma=0.1, phi0=1.0, T=0.5, Nt=40)
rng = np.random.default_rng(0)
C_lq, _, _ = sm.lq_control(cf)
REACT = "--box-only" not in sys.argv

# --- Nagumo derivatives
s = rm.build(cf, rm.Reaction("nagumo", 200.0, 0.4))
C = 3.0 * rng.standard_normal((cf.Nt, cf.m)) + 4.0
Y = rm.forward(s, C)
print(f"state range [{Y.min():.3f}, {Y.max():.3f}]")
P = rm.adjoint(s, Y)
g = rm.gradient(s, C, P)
V = rng.standard_normal(C.shape)
for h in (1e-3, 1e-4, 1e-5) if REACT else ():
    Jp = rm.cost(s, rm.forward(s, C + h * V), C + h * V)
    Jm = rm.cost(s, rm.forward(s, C - h * V), C - h * V)
    print(f"grad FD h={h:.0e}: rel err {abs((Jp - Jm) / (2 * h) - np.sum(g * V)) / abs(np.sum(g * V)):.2e}")
Hv = rm.hessvec(s, Y, P, V)
for h in (1e-3, 1e-4, 1e-5) if REACT else ():
    gp = rm.gradient(s, C + h * V, rm.adjoint(s, rm.forward(s, C + h * V)))
    gm = rm.gradient(s, C - h * V, rm.adjoint(s, rm.forward(s, C - h * V)))
    print(f"hessvec FD h={h:.0e}: rel err {np.linalg.norm((gp - gm) / (2 * h) - Hv) / np.linalg.norm(Hv):.2e}")

# --- cubic: same optimum as semilinear_moving
if REACT:
    Ca, ia = sm.solve(sm.build(cf, 100.0), C_lq)
    Cb, ib = rm.solve(rm.build(cf, rm.Reaction("cubic", 100.0)), C_lq)
    print(f"cubic: |C_rm - C_sm|/|C_sm| = {np.linalg.norm(Ca - Cb) / np.linalg.norm(Ca):.2e}  (rm {ib['newton_iters']} it, grad {ib['grad_rel']:.1e})")

# --- Nagumo solve from LQ and from 0
for th in (0.3, 0.5) if REACT else ():
    sn = rm.build(cf, rm.Reaction("nagumo", 100.0, th))
    for nm, C0 in (("LQ", C_lq), ("zero", np.zeros_like(C_lq))):
        Cn, info = rm.solve(sn, C0, maxit=60)
        print(f"nagumo theta={th} from {nm}: J {info['J']:.6e} it {info['newton_iters']} cg {info['cg_iters']} neg {info['neg_curv']} "
              f"grad {info['grad_rel']:.1e} max|u| {np.abs(Cn).max():.2f}")

# --- box
Cu, iu = solve_box(cf, -np.inf, np.inf)
print(f"box unconstrained: it {iu['pn_iters']} fixed-point res {iu['fixed_point_res']:.1e}; vs consistent-mass LQ "
      f"{np.linalg.norm(Cu - C_lq) / np.linalg.norm(C_lq):.2e} (lumped vs consistent mass)")
ub = 0.5 * Cu.max()
Cb, ib = solve_box(cf, 0.0, ub, C0=Cu, verbose=True)
print(f"box [0, {ub:.3f}]: it {ib['pn_iters']} res {ib['fixed_point_res']:.1e} upper {ib['frac_upper']:.3f} lower {ib['frac_lower']:.3f}")
Cz, iz = solve_box(cf, 0.0, ub)
print(f"box from zero: it {iz['pn_iters']} cg {iz['cg_iters']}; |C - C_fromclip|/|C| = {np.linalg.norm(Cz - Cb) / np.linalg.norm(Cb):.1e}")
from moving2d import Sweeps, assemble, target_loads  # noqa: E402

o = assemble(cf)
GY, csq = target_loads(o)
sw = Sweeps(o, GY)
ml = np.asarray(o.Mu.sum(axis=1)).ravel()


def Jl(Cx):
    Yx = sw.forward_full(Cx)
    J = sum(0.5 * cf.dt * (Yx[k] @ (o.M_obs @ Yx[k]) - 2 * Yx[k] @ np.kron(GY[k], o.gx) + csq[k]) for k in range(cf.Nt))
    return J + 0.5 * cf.alpha * cf.dt * float(np.sum(Cx * Cx * ml[None, :]))


J0 = Jl(Cb)
worse = [Jl(np.clip(Cb + 0.05 * ub * rng.standard_normal(Cb.shape), 0.0, ub)) - J0 for _ in range(10)]
print(f"box: J* {J0:.6e}, min increase over 10 random feasible perturbations {min(worse):.2e} (must be > 0)")
