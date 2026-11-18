#!/usr/bin/env python3
"""One case of the advection-diffusion moving hot-spot problem: exact optimal edge control u(s, t) (= q up to the
control mass, unconstrained), exact periodic turnpike, and rank diagnostics:
  rank@1e-3 / 1e-4 of u over the horizon and of one period of the periodic turnpike, POD floors of u,
  rank of the target itself, layer profile |u(t) - u_per(t)|, strip tracking error."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import advect_moving as am  # noqa: E402
import moving2d  # noqa: E402

RANKS = (5, 10, 20, 40)


def sv_stats(Z):
    sv = np.linalg.svd(Z, compute_uv=False)
    return {"rank3": int(np.sum(sv > 1e-3 * sv[0])), "rank4": int(np.sum(sv > 1e-4 * sv[0])), "s2s1": float(sv[1] / sv[0]),
            "sv": (sv[:60] / sv[0]).tolist()}


def pod_floor(Z):
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    tot = np.sqrt(np.sum(s**2))
    return {r: float(np.sqrt(np.sum(s[r:] ** 2)) / tot) for r in RANKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--bx", type=float, default=0.0)
    ap.add_argument("--by", type=float, default=0.0)
    ap.add_argument("--nh", type=int, default=100)
    ap.add_argument("--Nt", type=int, default=640)
    ap.add_argument("--shear", default="", choices=["", "lin", "pois"])
    ap.add_argument("--xo", type=float, default=0.0, help="observation strip x in [xo, xo + w]")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    cf = am.CfgAdv(nh=args.nh, Nt=args.Nt, eps=args.eps, bx=args.bx, by=args.by, shear=args.shear, xo=args.xo)
    t0 = time.perf_counter()
    o = am.assemble(cf)
    GY, csq = moving2d.target_loads(o)
    U, hinfo = moving2d.periodic_turnpike(o)
    t_hb = time.perf_counter() - t0
    t = np.arange(1, cf.Nt + 1) * cf.dt
    Up = moving2d.eval_periodic(U, cf.omega, cf.phi0, t)
    C, info = am.pcg_solve(o, GY, C0=Up)
    sw = am.Sweeps(o, GY)
    strip, ks, snaps = sw.forward_strip(C)
    J = moving2d.cost(o, GY, csq, C)
    J0 = 0.5 * cf.dt * float(np.sum(csq))
    Nper = int(round(2 * np.pi / cf.omega / cf.dt))
    Uper1 = moving2d.eval_periodic(U, cf.omega, cf.phi0, np.arange(Nper) * cf.dt)
    s = np.arange(1, cf.m + 1) / (cf.m + 1)
    Yd = moving2d.target(cf, s[None, :], cf.omega * t[:, None] + cf.phi0)
    sy = np.arange(1, cf.nh) / cf.nh
    Yd_strip = moving2d.target(cf, sy[None, :], cf.omega * t[:, None] + cf.phi0)
    k0 = cf.Nt // 4
    layer = np.linalg.norm(C - Up, axis=1) / np.linalg.norm(Up, axis=1).max()
    res = {"eps": cf.eps, "bx": cf.bx, "by": cf.by, "shear": cf.shear, "xo": cf.xo, "nh": cf.nh, "Nt": cf.Nt,
           "Pe": max(abs(cf.bx), abs(cf.by)) / cf.eps,
           "art_diff": float(am.art_diff(max(abs(cf.bx), abs(cf.by)), 1.0 / cf.nh, cf.eps, cf.stab)),
           "pcg_iters": info["iters"], "pcg_seconds": info["seconds"], "hb_seconds": t_hb, "n_harm": hinfo["n_harm"],
           "u": sv_stats(C), "u_per": sv_stats(Uper1), "target": sv_stats(Yd), "pod_u": pod_floor(C), "pod_u_per": pod_floor(Uper1),
           "J_over_J0": J / J0, "track": float(np.linalg.norm(strip[k0:] - Yd_strip[k0:]) / np.linalg.norm(Yd_strip[k0:])),
           "max_u": float(np.abs(C).max()), "per_vs_u_mid": float(np.linalg.norm((C - Up)[cf.Nt // 4: 3 * cf.Nt // 4])
                                                                  / np.linalg.norm(Up[cf.Nt // 4: 3 * cf.Nt // 4])),
           "total_seconds": time.perf_counter() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, C=C.astype(np.float32), Up=Up.astype(np.float32), strip=strip.astype(np.float32), snap_k=ks,
             snaps=snaps.astype(np.float32), layer=layer, res=json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}
                     | {"rank_u": res["u"]["rank3"], "rank_u_per": res["u_per"]["rank3"], "rank_target": res["target"]["rank3"]}))


if __name__ == "__main__":
    main()
