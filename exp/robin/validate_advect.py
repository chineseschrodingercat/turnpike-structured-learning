#!/usr/bin/env python3
"""Small-grid checks for advect_moving.py: (1) eps = 1, b = 0 equals moving2d; (2) reduced gradient by finite
differences for flows along / into the domain (transposed adjoint); (3) harmonic-balance periodic turnpike equals the
middle of a long-horizon optimum; (4) transport sign: constant heating, state at x = 0.5 for bx = +1 / 0 / -1."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import advect_moving as am  # noqa: E402
import moving2d  # noqa: E402

base = dict(nh=20, m=19, alpha=1e-3, beta=0.5, w=0.2, a=0.3, omega=2 * np.pi, sigma=0.1, phi0=1.0, T=1.0, Nt=80)

# (1) reduction to moving2d
cm = moving2d.CfgMove(**base)
om = moving2d.assemble(cm)
GYm, _ = moving2d.target_loads(om)
Cm, _ = moving2d.pcg_solve(om, GYm)
ca = am.CfgAdv(**base)
oa = am.assemble(ca)
GYa, _ = moving2d.target_loads(oa)
Ca, _ = am.pcg_solve(oa, GYa)
print(f"(1) |K - K_moving2d| = {abs(oa.K - om.K).max():.1e}, |C - C_moving2d|/|C| = {np.linalg.norm(Ca - Cm) / np.linalg.norm(Cm):.1e}")

# (2) gradient by finite differences
rng = np.random.default_rng(0)
for lab, kw in (("along s (by = 1)", dict(by=1.0)), ("into domain (bx = 1)", dict(bx=1.0))):
    c = am.CfgAdv(**base, eps=0.01, **kw)
    o = am.assemble(c)
    GY, csq = moving2d.target_loads(o)
    sw = am.Sweeps(o, GY)
    C = rng.standard_normal((c.Nt, c.m))
    V = rng.standard_normal(C.shape)
    g = c.dt * (c.alpha * C @ o.Mu.T - sw.adjoint(sw.forward(C), True))
    h = 1e-4
    fd = (moving2d.cost(o, GY, csq, C + h * V) - moving2d.cost(o, GY, csq, C - h * V)) / (2 * h)
    print(f"(2) {lab}: grad FD rel err {abs(fd - np.sum(g * V)) / abs(fd):.1e};  |K - K^T|/|K| = "
          f"{abs(o.K - o.K.T).max() / abs(o.K).max():.2f}")

# (3) periodic turnpike vs the middle of a long horizon
for lab, kw in (("along s", dict(by=1.0)), ("into domain", dict(bx=1.0))):
    c = am.CfgAdv(**dict(base, T=6.0, Nt=480), eps=0.01, **kw)
    o = am.assemble(c)
    GY, _ = moving2d.target_loads(o)
    C, info = am.pcg_solve(o, GY)
    U, hinfo = moving2d.periodic_turnpike(o)
    t = np.arange(1, c.Nt + 1) * c.dt
    Up = moving2d.eval_periodic(U, c.omega, c.phi0, t)
    mid = (t > 2.5) & (t < 3.5)
    print(f"(3) {lab}: PCG {info['iters']} it; |u - u_per|/|u_per| on t in (2.5, 3.5) = "
          f"{np.linalg.norm(C[mid] - Up[mid]) / np.linalg.norm(Up[mid]):.1e} ({hinfo['n_harm']} harmonics)")

# (4) transport sign: u = 1 on the whole edge for t in (0, 3)
for bx in (1.0, 0.0, -1.0):
    c = am.CfgAdv(**dict(base, nh=40, m=39, T=3.0, Nt=240), eps=0.01, bx=bx)
    o = am.assemble(c)
    sw = am.Sweeps(o, np.zeros((c.Nt, c.nh - 1)))
    y = np.zeros(o.n)
    for k in range(c.Nt):
        y = sw.lu.solve(sw.Mdt @ y + o.B @ np.ones(c.m))
    F = y.reshape(c.nh - 1, c.nh)
    print(f"(4) bx = {bx:+.0f}: y(x = 0, s = 0.5) = {F[c.nh // 2 - 1, 0]:.3f}, y(x = 0.5, s = 0.5) = {F[c.nh // 2 - 1, c.nh // 2]:.3f}"
          f"  (expect ~2 = u/beta at x = 0.5 only for bx = +1)")

if "--shear" in sys.argv:
    from robin2d import one_d

    # (5) shear flow bx(s): weighted mass, gradient, s-dependent transit time
    My = one_d(20, 0.05, 0.95, drop_first=True)[0]
    print(f"(5) |weighted mass(f = 1) - My| = {abs(am.weighted_mass_y(20, lambda s: np.ones_like(s)) - My).max():.1e}")
    c = am.CfgAdv(**base, eps=0.01, bx=1.0, shear="lin")
    o = am.assemble(c)
    GY, csq = moving2d.target_loads(o)
    sw = am.Sweeps(o, GY)
    C = rng.standard_normal((c.Nt, c.m))
    V = rng.standard_normal(C.shape)
    g = c.dt * (c.alpha * C @ o.Mu.T - sw.adjoint(sw.forward(C), True))
    fd = (moving2d.cost(o, GY, csq, C + 1e-4 * V) - moving2d.cost(o, GY, csq, C - 1e-4 * V)) / 2e-4
    print(f"(5) shear 'lin' grad FD rel err {abs(fd - np.sum(g * V)) / abs(fd):.1e}")
    c = am.CfgAdv(**dict(base, nh=40, m=39, T=0.4, Nt=80), eps=0.01, bx=1.0, shear="lin")
    o = am.assemble(c)
    sw = am.Sweeps(o, np.zeros((c.Nt, c.nh - 1)))
    y = np.zeros(o.n)
    for k in range(c.Nt):
        y = sw.lu.solve(sw.Mdt @ y + o.B @ np.ones(c.m))
    F = y.reshape(c.nh - 1, c.nh)
    print(f"(5) t = 0.4, y(x = 0.5): s = 0.1 (bx = 0.4) {F[3, 20]:.3f}, s = 0.5 (bx = 1.0) {F[19, 20]:.3f}, "
          f"s = 0.9 (bx = 1.6) {F[35, 20]:.3f}  (expect ~0, ~0 to 1, ~2)")
