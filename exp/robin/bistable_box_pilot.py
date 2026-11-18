#!/usr/bin/env python3
"""One ground-truth case of the moving hot-spot problem (n = 9900, m = 99) with either
  --kind nagumo : y_t - Laplace y + kappa y (y - theta)(y - 1) = 0  (exact optimum by Newton-CG from --init lq|zero), or
  --kind box    : linear heat, ua <= u <= ub (projected Newton; bounds as fractions of max|u_unc|).
Saves the control, the reference control (LQ / unconstrained), strip-averaged state traces y(s, t) and 2D state
snapshots for the optimal control and for the reference control applied to the same plant."""

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
import moving2d  # noqa: E402
import reaction_moving as rm  # noqa: E402
from box_moving import solve_box  # noqa: E402
from moving2d import CfgMove  # noqa: E402
from semilinear_moving import lq_control  # noqa: E402

BASE = dict(nh=100, m=99, alpha=1e-3, beta=0.5, w=0.2, a=0.3, omega=2 * np.pi, sigma=0.07, phi0=1.0, T=2.0, Nt=640)


def front_target(delta, c0=0.5):
    """Moving front: hot (1) for s < c(t) = c0 + a sin(theta), cold (0) above, logistic width delta (Nagumo-front shaped)."""
    def target(cfg, y, theta):
        c = c0 + cfg.a * np.sin(theta)
        return 1.0 / (1.0 + np.exp((np.asarray(y, float) - c) / delta))
    return target


def traces(cf, Y):
    """Strip-averaged state (x in [0, w]) over (t, s), and 16 full snapshots."""
    nx = cf.nh
    F = Y.reshape(Y.shape[0], cf.nh - 1, nx)
    strip = F[:, :, : int(round(cf.w * cf.nh)) + 1].mean(axis=2)
    ks = np.linspace(cf.Nt // 16, cf.Nt, 16).astype(int) - 1
    return strip.astype(np.float32), ks, Y[ks].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["nagumo", "box"], required=True)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--init", choices=["lq", "zero"], default="lq")
    ap.add_argument("--ua", type=float, default=-np.inf, help="fraction of max|u_unc| (or -inf / 0)")
    ap.add_argument("--ub", type=float, default=np.inf)
    ap.add_argument("--alpha", type=float, default=BASE["alpha"])
    ap.add_argument("--target", choices=["spot", "front"], default="spot")
    ap.add_argument("--delta", type=float, default=0.05, help="front width (--target front)")
    ap.add_argument("--c0", type=float, default=0.5, help="mean front position (--target front)")
    ap.add_argument("--a", type=float, default=BASE["a"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if args.target == "front":  # moving2d.target_loads / gy_of_theta look the target up at call time
        moving2d.target = front_target(args.delta, args.c0)
    prm = dict(BASE, alpha=args.alpha, a=args.a)
    cf = CfgMove(**prm)
    t0 = time.perf_counter()
    out = {}
    if args.kind == "nagumo":
        Clq, _, _ = lq_control(cf)
        s = rm.build(cf, rm.Reaction("nagumo", args.kappa, args.theta))
        C, info = rm.solve(s, Clq if args.init == "lq" else np.zeros_like(Clq), maxit=60, verbose=True)
        Y = rm.forward(s, C)
        Yr = rm.forward(s, Clq)  # LQ control applied to the bistable plant
        info["J_lq_on_plant"] = rm.cost(s, Yr, Clq)
        s0 = rm.build(cf, rm.Reaction("cubic", 0.0))
        info["J_lq_linear"] = rm.cost(s0, rm.forward(s0, Clq), Clq)
        Cref = Clq
        meta = {"kappa": args.kappa, "theta": args.theta, "init": args.init, "target": args.target, "delta": args.delta, "c0": args.c0}
    else:
        Cu, iu = solve_box(cf, -np.inf, np.inf)
        vmax = np.abs(Cu).max()
        ua = args.ua * vmax if np.isfinite(args.ua) else args.ua
        ub = args.ub * vmax if np.isfinite(args.ub) else args.ub
        C, info = solve_box(cf, ua, ub, C0=Cu, verbose=True)
        s0 = rm.build(cf, rm.Reaction("cubic", 0.0))
        Y, Yr = rm.forward(s0, C), rm.forward(s0, Cu)
        ml = np.asarray(s0.o.Mu.sum(axis=1)).ravel()
        Mlump = np.diag(ml)
        info["J"] = rm.cost(s0, Y, C, Mu=Mlump)
        info["J_unc"] = rm.cost(s0, Yr, Cu, Mu=Mlump)
        info["J_clip_unc"] = rm.cost(s0, rm.forward(s0, np.clip(Cu, ua, ub)), np.clip(Cu, ua, ub), Mu=Mlump)
        Cref = Cu
        meta = {"ua": float(ua), "ub": float(ub), "u_unc_max": float(vmax), "target": args.target, "delta": args.delta}
    strip, ks, snaps = traces(cf, Y)
    strip_r, _, snaps_r = traces(cf, Yr)
    info["total_seconds"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, C=C.astype(np.float32), C_ref=Cref.astype(np.float32), strip=strip, strip_ref=strip_r, snap_k=ks,
             snaps=snaps, snaps_ref=snaps_r, prm=json.dumps(prm), meta=json.dumps(meta),
             info=json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in info.items()}))
    print(json.dumps({**meta, **{k: v for k, v in info.items() if np.isscalar(v)}}, default=float))


if __name__ == "__main__":
    main()
