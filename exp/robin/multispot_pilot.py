#!/usr/bin/env python3
"""Pilot for richer moving targets: several hot spots moving along the controlled edge with incommensurate frequencies,
    y_d(s, t) = sum_j A_j exp(-(s - c_j - a_j sin(omega_j t + phi_j))^2 / (2 sigma_j^2)),
so the target is quasi-periodic and there is no periodic turnpike (moving2d.py geometry: strip [0, w] x [0.05, 0.95],
distributed edge control, n = 9900, m = 99, dt = 1/320).
Per case and T in {2, 8}: exact linear control (PCG, warm start = quasi-periodic reference) and exact semilinear control
(Newton-CG of semilinear_moving.py, gamma y^3, warm start = linear control). The linear quasi-periodic reference is exact:
harmonic balance per spot (frequencies n omega_j), superposed, with the constant parts of all spots solved together.
Writes runs/multispot/<case>.npz and runs/multispot/summary.json."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import CfgMove, assemble, pcg_solve  # noqa: E402
from robin_solver import interval_load  # noqa: E402
from semilinear_moving import SemiMove, solve as newton_solve  # noqa: E402

GOLD = (1 + 5**0.5) / 2
# spot = (A, c, a, omega, sigma, phi)
CASES = {
    "two_spots": dict(alpha=10**-2.75, beta=1.0, w=0.15, gamma=100.0,
                      spots=[(1.0, 0.3, 0.15, 2 * np.pi, 0.07, 0.0), (1.0, 0.7, 0.15, 2 * np.pi * GOLD, 0.07, 1.0)]),
    "three_spots": dict(alpha=1e-3, beta=1.0, w=0.15, gamma=100.0,
                        spots=[(1.0, 0.25, 0.1, np.pi, 0.05, 0.0), (1.0, 0.5, 0.1, np.pi * 2**0.5, 0.05, 2.0),
                               (1.0, 0.75, 0.1, np.pi * 3**0.5, 0.05, 4.0)]),
    "crossing": dict(alpha=1.5e-3, beta=1.0, w=0.15, gamma=100.0,
                     spots=[(1.0, 0.5, 0.3, 2 * np.pi, 0.06, 0.0), (1.0, 0.5, 0.3, 2 * np.pi / GOLD, 0.06, np.pi / 2)]),
    "hot_cold": dict(alpha=1.5e-3, beta=1.0, w=0.15, gamma=100.0,
                     spots=[(1.0, 0.35, 0.15, 2 * np.pi, 0.07, 0.0), (-0.6, 0.65, 0.15, 2 * np.pi * 2**0.5, 0.07, 1.0)]),
}
OUT = pathlib.Path("runs/multispot")


def spot(y, th, sp_):
    A, c, a, _, sg, _ = sp_
    return A * np.exp(-0.5 * ((np.asarray(y, float) - c - a * np.sin(th)) / sg) ** 2)


def target(y, t, spots):
    return sum(spot(y, sp_[3] * t + sp_[5], sp_) for sp_ in spots)


def cfg_for(case, T):
    p = CASES[case]
    return CfgMove(alpha=p["alpha"], beta=p["beta"], w=p["w"], T=float(T), Nt=int(round(320 * T)))


def gy(ops, f):
    c = ops.cfg
    return interval_load(lambda y, tt: f(y), ops.cy, 0.0, c.ya, c.yb)[1:]


def loads(ops, spots):
    c = ops.cfg
    t = np.arange(1, c.Nt + 1) * c.dt
    GY = np.stack([gy(ops, lambda y, tk=tk: target(y, tk, spots)) for tk in t])
    xq, wq = leggauss(200)
    yq = 0.5 * (c.yb - c.ya) * xq + 0.5 * (c.yb + c.ya)
    csq = np.array([c.w * 0.5 * (c.yb - c.ya) * np.sum(wq * target(yq, tk, spots) ** 2) for tk in t])
    return GY, csq


def harm(ops, Bd, freq, Gvec):
    """Control harmonic for the load Gvec e^{i freq t} (backward-Euler symbol), exact for the discrete problem."""
    c = ops.cfg
    z = (1.0 - np.exp(-1j * freq * c.dt)) / c.dt
    lu = spla.splu((z * ops.M + ops.K).tocsc().astype(complex))
    Psi = lu.solve(Bd)
    H = c.alpha * ops.Mu + Psi.conj().T @ (ops.M_obs @ Psi)
    return np.linalg.solve(H, Psi.conj().T @ np.kron(Gvec, ops.gx))


def qp_reference(ops, spots, n_theta=256, tol=1e-8, nmax=80):
    Bd = ops.B.toarray().astype(complex)
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    G0, parts, nharm = 0.0, [], []
    for sp_ in spots:
        GYt = np.stack([gy(ops, lambda y, a=a: spot(y, a, sp_)) for a in th])
        Gn = np.fft.fft(GYt, axis=0) / n_theta
        norms = np.linalg.norm(Gn, axis=1)
        nh = int(min(nmax, max([n for n in range(1, n_theta // 2) if norms[n] > tol * norms[0]], default=1)))
        G0 = G0 + Gn[0]
        U = np.zeros((nh + 1, ops.cfg.m), complex)
        for n in range(1, nh + 1):
            U[n] = harm(ops, Bd, n * sp_[3], Gn[n])
        parts.append((sp_[3], sp_[5], U))
        nharm.append(nh)
    u0 = np.real(harm(ops, Bd, 0.0, G0))
    return u0, parts, nharm


def eval_ref(u0, parts, t):
    R = np.tile(u0, (len(t), 1))
    for om, ph, U in parts:
        n = np.arange(U.shape[0])
        E = np.exp(1j * (om * t[:, None] + ph) * n[None, :])
        E[:, 0] = 0.0
        R = R + 2 * np.real(E @ U)
    return R


def run(job):
    case, T = job
    p = CASES[case]
    cf = cfg_for(case, T)
    ops = assemble(cf)
    GY, csq = loads(ops, p["spots"])
    t = np.arange(1, cf.Nt + 1) * cf.dt
    t0 = time.perf_counter()
    u0, parts, nharm = qp_reference(ops, p["spots"])
    R = eval_ref(u0, parts, t)
    t_ref = time.perf_counter() - t0
    C_lin, info_l = pcg_solve(ops, GY, C0=R, tol=1e-10)
    s = SemiMove(cfg=cf, gamma=p["gamma"], o=ops, ML=np.asarray(ops.M.sum(axis=1)).ravel(), lu=spla.splu(ops.A.tocsc()),
                 Mdt=(ops.M / cf.dt).tocsr(), GY=GY, csq=csq, stats={"lu_solves": 0})
    C_nl, info_n = newton_solve(s, C_lin, gtol=1e-8)
    info = {"ref_seconds": t_ref, "n_harm": nharm, "pcg_iters": info_l["iters"], "pcg_seconds": info_l["seconds"],
            "newton_iters": info_n["newton_iters"], "newton_seconds": info_n["seconds"], "grad_rel": info_n["grad_rel"]}
    print(case, T, json.dumps(info), flush=True)
    return case, T, C_lin, C_nl, R, info


def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, T) for T in (8, 2) for c in CASES]
    with Pool(len(jobs)) as pool:
        res = pool.map(run, jobs, chunksize=1)
    by = {(c, T): (Cl, Cn, R, info) for c, T, Cl, Cn, R, info in res}
    summary = {}
    s = np.arange(1, 100) / 100
    for c in CASES:
        Cl8, Cn8, R8, i8 = by[(c, 8)]
        Cl2, Cn2, R2, i2 = by[(c, 2)]
        t8 = np.arange(1, Cl8.shape[0] + 1) / 320
        Yd = target(s[None, :], t8[:, None], CASES[c]["spots"])
        mid = (t8 > 2) & (t8 < 6)
        w0 = t8 <= 0.5
        sv_l, sv_n = np.linalg.svd(Cl8, compute_uv=False), np.linalg.svd(Cn8, compute_uv=False)
        summary[c] = {
            "params": {k: v for k, v in CASES[c].items() if k != "spots"}, "spots": CASES[c]["spots"],
            "T8": i8, "T2": i2,
            "lin_vs_ref_interior": rel(Cl8[mid], R8[mid]),
            "lin_vs_ref_whole": rel(Cl8, R8),
            "nl_vs_lin_interior": rel(Cn8[mid], Cl8[mid]),
            "nl_vs_linref_interior": rel(Cn8[mid], R8[mid]),
            "init_window_T2_vs_T8_lin": rel(Cl2[w0[:Cl2.shape[0]]], Cl8[w0]),
            "init_window_T2_vs_T8_nl": rel(Cn2[w0[:Cn2.shape[0]]], Cn8[w0]),
            "rank_lin": int(np.sum(sv_l > 1e-3 * sv_l[0])), "s2s1_lin": float(sv_l[1] / sv_l[0]),
            "rank_nl": int(np.sum(sv_n > 1e-3 * sv_n[0])), "s2s1_nl": float(sv_n[1] / sv_n[0]),
            "rank_target": int(np.sum((sv_t := np.linalg.svd(Yd, compute_uv=False)) > 1e-3 * sv_t[0])),
        }
        np.savez_compressed(OUT / f"{c}.npz", C_lin_T8=Cl8.astype(np.float32), C_nl_T8=Cn8.astype(np.float32), R_T8=R8.astype(np.float32),
                            C_lin_T2=Cl2.astype(np.float32), C_nl_T2=Cn2.astype(np.float32), Yd_T8=Yd.astype(np.float32))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({c: {k: v for k, v in d.items() if k not in ("params", "spots", "T8", "T2")} for c, d in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
