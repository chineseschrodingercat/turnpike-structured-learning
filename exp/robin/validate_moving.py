#!/usr/bin/env python3
"""Checks for moving2d.py and a pilot of the exact controls.

V1  small grid (nh = 10, m = 9, N = 80): PCG vs the monolithic KKT with the time-dependent target
V2  small grid, T = 6: exact control in the middle vs the harmonic-balance periodic turnpike
V3  full grid (nh = 100, m = 99), three moving targets, T = 4: PCG cold vs warm start from u_hat (iterations, time),
    middle of the exact control vs u_hat, singular values of u(s, t); saves the controls for plotting
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import CfgMove, assemble, eval_periodic, pcg_solve, periodic_turnpike, target_loads  # noqa: E402
from robin_solver import block_lower  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def kkt(ops, GY):
    c = ops.cfg
    N, n, m = c.Nt, ops.n, c.m
    I_t = sp.eye(N, format="csr")
    Ay = block_lower(ops.A, (-ops.M / c.dt).tocsr(), N)
    Bb = -sp.kron(I_t, ops.B, format="csr")
    Km = sp.bmat([[Ay, None, Bb], [sp.kron(I_t, ops.M_obs, format="csr"), Ay.T.tocsr(), None],
                  [None, Bb.T, c.alpha * sp.kron(I_t, sp.csr_matrix(ops.Mu), format="csr")]], format="csc")
    rhs = np.concatenate([np.zeros(n * N), np.concatenate([np.kron(GY[k], ops.gx) for k in range(N)]), np.zeros(m * N)])
    return spla.spsolve(Km, rhs)[2 * n * N :].reshape(N, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    rep = {}
    c = CfgMove(nh=10, m=9, alpha=1e-3, beta=1.0, w=0.3, a=0.3, omega=2 * np.pi, sigma=0.12, phi0=0.7, T=1.0, Nt=80)
    ops = assemble(c)
    GY, _ = target_loads(ops)
    Cp, info = pcg_solve(ops, GY, tol=1e-12)
    rep["V1_pcg_vs_kkt"] = {"rel": rel(Cp, kkt(ops, GY)), "iters": info["iters"]}
    print(rep, flush=True)

    c6 = CfgMove(**{**c.__dict__, "T": 6.0, "Nt": 480})
    ops6 = assemble(c6)
    GY6, _ = target_loads(ops6)
    C6, _ = pcg_solve(ops6, GY6, tol=1e-12)
    U, hinfo = periodic_turnpike(ops6)
    t = np.arange(1, 481) * c6.dt
    mid = (t > 2.5) & (t < 3.5)
    rep["V2_periodic_turnpike_small"] = {"rel_err_middle": rel(eval_periodic(U, c6.omega, c6.phi0, t[mid]), C6[mid]), **hinfo}
    print(rep["V2_periodic_turnpike_small"], flush=True)

    pilots = [dict(a=0.15, omega=np.pi, sigma=0.12, alpha=10**-2.5, beta=1.0, w=0.15, phi0=0.0),
              dict(a=0.3, omega=2 * np.pi, sigma=0.07, alpha=10**-3, beta=0.5, w=0.2, phi0=1.0),
              dict(a=0.35, omega=4 * np.pi, sigma=0.04, alpha=10**-3.5, beta=0.3, w=0.1, phi0=2.0)]
    v3, save = [], {}
    for j, pdict in enumerate(pilots):
        cf = CfgMove(nh=100, m=99, T=4.0, Nt=1280, **pdict)
        o = assemble(cf)
        GYf, csq = target_loads(o)
        U, hinfo = periodic_turnpike(o)
        tt = np.arange(1, 1281) * cf.dt
        uh = eval_periodic(U, cf.omega, cf.phi0, tt)
        Cc, ic = pcg_solve(o, GYf)
        Cw, iw = pcg_solve(o, GYf, C0=uh)
        mid = (tt > 1.5) & (tt < 2.5)
        sv = np.linalg.svd(Cc, compute_uv=False)
        svh = np.linalg.svd(uh, compute_uv=False)
        r = {"pilot": j, **{k: float(v) for k, v in pdict.items()}, "harmonics": hinfo["n_harm"], "harm_seconds": hinfo["seconds"],
             "pcg_cold": ic, "pcg_warm": iw, "warm_vs_cold": rel(Cw, Cc), "u_hat_vs_exact_middle": rel(uh[mid], Cc[mid]),
             "sv_ratios_u": [float(x) for x in sv[1:6] / sv[0]], "rank_1e-3_u": int(np.sum(sv > 1e-3 * sv[0])),
             "rank_1e-3_u_hat": int(np.sum(svh > 1e-3 * svh[0]))}
        v3.append(r)
        print(json.dumps(r), flush=True)
        save[f"C{j}"], save[f"uh{j}"], save[f"GY{j}"] = Cc.astype(np.float32), uh.astype(np.float32), GYf.astype(np.float32)
    rep["V3_pilots"] = v3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    np.savez(args.out.with_suffix(".npz"), pilots=json.dumps(pilots, default=float), **save)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
