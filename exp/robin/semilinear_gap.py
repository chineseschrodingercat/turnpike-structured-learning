#!/usr/bin/env python3
"""Optimality gap (J(u) - J*) / J* of predicted controls on the semilinear test split, J evaluated with the nonlinear
forward solve (semilinear1d.forward + cost); J* uses the same evaluation of the Newton-KKT control.

Reference controls without learning: 'plateau' u = u_bar (static nonlinear OCP), 'lq' = optimal control of the
gamma = 0 problem, 'lq_rescaled' = lq * u_bar / u_bar_lq (linear shape, nonlinear plateau level).
Predictions: --preds label=path.npz ... (files from baselines_multihorizon.py / y0_turnpike_surrogate.py --save-pred).
Datasets with a 'y0' array (gen_y0_data.py) use that initial state for every solve, the references included.
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig  # noqa: E402
from semilinear1d import build, cost, forward, solve_kkt, static_ocp  # noqa: E402

T_LIST = (1, 2, 4, 8)


def one(task):
    geo, gamma, T_ref, T, ubar, controls, y0 = task
    cfg = RobinConfig(nx=160, Nt=320 * T, T=float(T), **geo)
    s = build(cfg, gamma, T_ref, y0)
    lq = solve_kkt(build(cfg, 0.0, T_ref, y0))[2]
    ubar0 = static_ocp(build(RobinConfig(nx=160, Nt=320, **geo), 0.0, T_ref))["u_bar"]
    controls = dict(controls, plateau=np.full(cfg.Nt, ubar), lq=lq, lq_rescaled=lq * ubar / ubar0)
    Ustar = controls.pop("exact")
    Jstar = cost(s, forward(s, Ustar), Ustar)
    out = {"J_star": Jstar}
    for k, u in controls.items():
        out[k] = {"gap": (cost(s, forward(s, u), u) - Jstar) / Jstar, "rel_err": float(np.linalg.norm(u - Ustar) / np.linalg.norm(Ustar))}
    return out


def summ(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(np.max(x)), "min": float(np.min(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    d = np.load(args.data)
    te = d["test_idx"]
    preds = {}
    for spec in args.preds:
        label, path = spec.split("=", 1)
        z = np.load(path)
        assert np.array_equal(z["test_idx"], te), path
        preds[label] = {T: z[f"U_T{T}"] for T in T_LIST}
    tasks, keys = [], []
    for r, i in enumerate(te):
        geo = dict(alpha=float(d["alpha"][i]), beta=float(d["beta"][i]), xa=float(d["xa"][i]), xb=float(d["xb"][i]))
        for T in T_LIST:
            ctrl = {"exact": d[f"U_T{T}"][i]}
            ctrl.update({lab: preds[lab][T][r] for lab in preds})
            tasks.append((geo, float(d["gamma"][i]), float(d["T_ref"][i]), T, float(d["u_bar"][i]), ctrl,
                          d["y0"][i] if "y0" in d.files else None))
            keys.append((r, T))
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(one, tasks, chunksize=1)
    labels = list(preds) + ["plateau", "lq", "lq_rescaled"]
    rep = {"n_test": len(te), "seconds": time.perf_counter() - t0, "by_T": {}}
    raw = {}
    for T in T_LIST:
        rows = [res[k] for k, (r, TT) in enumerate(keys) if TT == T]
        rep["by_T"][str(T)] = {lab: {"gap": summ([q[lab]["gap"] for q in rows]), "rel_err": summ([q[lab]["rel_err"] for q in rows]),
                                     "gap_over_relerr_sq_median": float(np.median([q[lab]["gap"] / max(q[lab]["rel_err"], 1e-300) ** 2 for q in rows]))}
                               for lab in labels}
        for lab in labels:
            raw[f"gap_{lab}_T{T}"] = np.array([q[lab]["gap"] for q in rows])
            raw[f"relerr_{lab}_T{T}"] = np.array([q[lab]["rel_err"] for q in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    np.savez(args.out.with_suffix(".npz"), test_idx=te, **raw)
    for T in T_LIST:
        print(f"T={T}: " + "  ".join(f"{lab} gap med {rep['by_T'][str(T)][lab]['gap']['median']:.1e} max {rep['by_T'][str(T)][lab]['gap']['max']:.1e}"
                                     for lab in labels))
    print(f"wrote {args.out} ({rep['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
