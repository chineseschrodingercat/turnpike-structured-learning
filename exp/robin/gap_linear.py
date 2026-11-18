#!/usr/bin/env python3
"""Optimality gap (J(u) - J*) / J* for the linear cases, J from a forward solve with the same discretization.

kinds: const1d (turnpike_data.npz, y_d = 1), osc1d (osc_data.npz, y_d = 1.25 + A sin(omega t)),
       scalar2d (data_2d.npz, y_d = 1), mi2d (data_mi.npz, u(s,t) with m inputs, control cost dt alpha c^T Mu c / 2).
Predictions: --preds label=path.npz ... (rows = test_idx order); horizons present in every file are evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators, assemble_problem_data, eval_cost, solve_state  # noqa: E402

C_OFF = 1.25


def target(spec):
    if spec[0] == "const":
        return lambda x, t: np.ones_like(np.asarray(x, float))
    _, A, om = spec
    return lambda x, t: np.full_like(np.asarray(x, float), C_OFF + A * np.sin(om * t))


def cost_1d(task):
    geo, T, spec, controls = task
    yd = target(spec)
    cfg = RobinConfig(nx=160, Nt=320 * T, T=float(T), **geo)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, yd)
    return {k: eval_cost(cfg, ops, data, solve_state(ops, data, u), u)[0] for k, u in controls.items()}


def cost_2d(task):
    from robin2d import Cfg2D, assemble
    from robin2d_mi import assemble_mi

    geo, T, m, controls = task
    cfg = Cfg2D(nh=100, T=float(T), Nt=320 * T, **{k: geo[k] for k in ("xa", "xb", "ya", "yb", "alpha", "beta")})
    if m is None:
        o = assemble(cfg)
        Bop, Mu = (lambda c: o.b * c), None
    else:
        mi = assemble_mi(cfg, m)
        o, Mu = mi.base, mi.Mu
        Bop = lambda c: mi.B @ c  # noqa: E731
    lu = spla.splu(o.A.tocsc())
    Mdt = (o.M / cfg.dt).tocsr()
    area = (cfg.xb - cfg.xa) * (cfg.yb - cfg.ya)
    out = {}
    for k, u in controls.items():
        y = np.zeros(o.n)
        J = 0.0
        for j in range(len(u)):
            y = lu.solve(Mdt @ y + Bop(u[j]))
            J += 0.5 * cfg.dt * (float(y @ (o.M_obs @ y)) - 2.0 * float(y @ o.G) + area)
        J += 0.5 * cfg.alpha * cfg.dt * (float(np.sum(u * u)) if Mu is None else float(np.einsum("ki,ij,kj->", u, Mu, u)))
        out[k] = J
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["const1d", "osc1d", "scalar2d", "mi2d"], required=True)
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--T", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    d = np.load(args.data)
    te = d["test_idx"]
    preds = {}
    for spec in args.preds:
        lab, path = spec.split("=", 1)
        z = np.load(path)
        assert np.array_equal(z["test_idx"], te), path
        preds[lab] = {T: z[f"U_T{T}"].astype(np.float64) for T in args.T}
    tasks, keys = [], []
    for r, i in enumerate(te):
        geo = {k: float(d[k][i]) for k in ("xa", "xb", "alpha", "beta") + (("ya", "yb") if "ya" in d.files else ())}
        for T in args.T:
            if args.kind == "const1d":
                ex, yd = d[f"U_T{T}"][i], ("const",)
            elif args.kind == "osc1d":
                A, om = float(d["A"][i]), float(d["omega"][i])
                ex = C_OFF * d[f"U_const_T{T}"][i] + A * d[f"U_sin_T{T}"][i]
                yd = ("osc", A, om)
            elif args.kind == "scalar2d":
                ex, yd = d[f"U_T{T}"][i], None
            else:
                ex = (d[f"U_T{T}"][i] if T <= 2 else d[f"U_T{T}_test"][r]).astype(np.float64)
                yd = None
            ctrl = {"exact": ex, **{lab: preds[lab][T][r] for lab in preds}}
            tasks.append((geo, T, yd, ctrl) if args.kind.endswith("1d") else (geo, T, None if args.kind == "scalar2d" else int(d["m"]), ctrl))
            keys.append((r, T))
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(cost_1d if args.kind.endswith("1d") else cost_2d, tasks, chunksize=1)
    rep = {"kind": args.kind, "n_test": len(te), "seconds": time.perf_counter() - t0, "by_T": {}}
    raw = {}
    for T in args.T:
        rows = [res[k] for k, (r, TT) in enumerate(keys) if TT == T]
        rep["by_T"][str(T)] = {}
        for lab in preds:
            g = np.array([(q[lab] - q["exact"]) / q["exact"] for q in rows])
            raw[f"gap_{lab}_T{T}"] = g
            rep["by_T"][str(T)][lab] = {"median": float(np.median(g)), "p90": float(np.percentile(g, 90)), "max": float(np.max(g)),
                                        "min": float(np.min(g))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    np.savez(args.out.with_suffix(".npz"), test_idx=te, **raw)
    for T in args.T:
        print(f"T={T}: " + "  ".join(f"{lab} med {v['median']:.1e} max {v['max']:.1e} min {v['min']:.1e}" for lab, v in rep["by_T"][str(T)].items()))
    print(f"wrote {args.out} ({rep['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
