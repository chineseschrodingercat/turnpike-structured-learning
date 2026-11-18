#!/usr/bin/env python3
"""Optimality gap (J(u) - J*) / J* for the 2D semilinear test split, J by the nonlinear forward solve of
semilinear2d.py. References without learning: plateau u = u_bar, LQ control (gamma = 0 optimum).
--preds label=path.npz (rows = test_idx order)."""

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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D  # noqa: E402
from semilinear2d import build, cost, forward, lq_control  # noqa: E402


def one(task):
    geo, gamma, T, ub, controls = task
    cfg = Cfg2D(nh=100, T=float(T), Nt=320 * T, **geo)
    s = build(cfg, gamma)
    controls = dict(controls, plateau=np.full(cfg.Nt, ub), lq=lq_control(cfg, (T,))[T])
    ex = controls.pop("exact")
    Jstar = cost(s, forward(s, ex), ex)
    return {k: {"gap": (cost(s, forward(s, u), u) - Jstar) / Jstar, "rel_err": float(np.linalg.norm(u - ex) / np.linalg.norm(ex))}
            for k, u in controls.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--T", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    d = np.load(args.data)
    te = d["test_idx"]
    preds = {}
    for spec in args.preds:
        lab, path = spec.split("=", 1)
        z = np.load(path)
        assert np.array_equal(z["test_idx"], te), path
        preds[lab] = {T: z[f"U_T{T}"] for T in args.T}
    tasks, keys = [], []
    for r, i in enumerate(te):
        geo = {k: float(d[k][i]) for k in ("xa", "xb", "ya", "yb", "alpha", "beta")}
        for T in args.T:
            tasks.append((geo, float(d["gamma"][i]), T, float(d["u_bar"][i]), {"exact": d[f"U_T{T}"][i], **{lab: preds[lab][T][r] for lab in preds}}))
            keys.append(T)
    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = pool.map(one, tasks, chunksize=1)
    labels = list(preds) + ["plateau", "lq"]
    rep = {"n_test": len(te), "seconds": time.perf_counter() - t0, "by_T": {}}
    for T in args.T:
        rows = [q for q, TT in zip(res, keys) if TT == T]
        rep["by_T"][str(T)] = {}
        for lab in labels:
            g = np.array([q[lab]["gap"] for q in rows])
            e = np.array([q[lab]["rel_err"] for q in rows])
            rep["by_T"][str(T)][lab] = {"gap_median": float(np.median(g)), "gap_max": float(np.max(g)), "gap_min": float(np.min(g)),
                                        "rel_err_median": float(np.median(e)), "rel_err_max": float(np.max(e))}
        print(f"T={T}: " + "  ".join(f"{lab} gap {v['gap_median']:.1e}/{v['gap_max']:.1e} (min {v['gap_min']:.1e})"
                                     for lab, v in rep["by_T"][str(T)].items()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({rep['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
