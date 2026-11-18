#!/usr/bin/env python3
"""Uniform timing for the 2D semilinear problem (n = 9900): one process, one thread (BLAS and torch), per mu.

exact:      LQ warm start (robin2d reduced method) + reduced Newton-CG, T = 1 and T = 8
surrogate:  static nonlinear OCP (sparse Newton) + linearized floor (shift-invert eigsh) + TurnpikeNet query (r = 4/8,
            untrained weights; cost is weight-independent), batch 1, T = 8
"""

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
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D  # noqa: E402
from semilinear2d import build, lq_control, solve, static_ocp  # noqa: E402
from turnpike_surrogate import TurnpikeNet, grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    d = np.load(args.data)
    te = d["test_idx"]
    ge = d["gamma"][te]
    picks = [int(te[np.argmin(np.abs(ge - np.quantile(ge, q)))]) for q in np.linspace(0.2, 0.8, args.k)]
    rows = []
    for i in picks:
        geo = {k: float(d[k][i]) for k in ("xa", "xb", "ya", "yb", "alpha", "beta")}
        gamma = float(d["gamma"][i])
        r = {"idx": i, "gamma_eff": gamma}
        for T in (1, 8):
            cfg = Cfg2D(nh=100, T=float(T), Nt=320 * T, **geo)
            t0 = time.perf_counter()
            ulq = lq_control(cfg, (T,))[T]
            t1 = time.perf_counter()
            u, info = solve(build(cfg, gamma), ulq)
            t2 = time.perf_counter()
            r[f"T{T}"] = {"lq_s": t1 - t0, "newton_cg_s": t2 - t1, "total_s": t2 - t0, "newton_iters": info["newton_iters"],
                          "cg_iters": info["cg_iters"], "rel_err_vs_data": float(np.linalg.norm(u - d[f"U_T{T}"][i]) / np.linalg.norm(d[f"U_T{T}"][i]))}
        t0 = time.perf_counter()
        st = static_ocp(build(Cfg2D(nh=100, **geo), gamma))
        r["static_ocp_and_floor_s"] = time.perf_counter() - t0
        r["static_iters"] = st["iters"]
        t, s = grid(8, 1 / 320)
        for rr in (4, 8):
            net = TurnpikeNet(r=rr, shared=True, use_floor=True, n_in=d["X"].shape[1]).eval()
            x, fl = torch.randn(1, d["X"].shape[1]), torch.tensor([st["lam1_lin"]])
            with torch.no_grad():
                net(x, t, s, fl)
                ts = []
                for _ in range(20):
                    t0 = time.perf_counter()
                    net(x, t, s, fl)
                    ts.append(time.perf_counter() - t0)
            r[f"net_r{rr}_T8_s"] = float(np.median(ts))
        r["surrogate_total_T8_s"] = r["static_ocp_and_floor_s"] + r["net_r8_T8_s"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    rep = {"protocol": "single process, 1 thread (BLAS + torch), wall clock per mu", "samples": rows,
           "median": {"exact_T1_s": float(np.median([q["T1"]["total_s"] for q in rows])), "exact_T8_s": float(np.median([q["T8"]["total_s"] for q in rows])),
                      "surrogate_T8_s": float(np.median([q["surrogate_total_T8_s"] for q in rows])),
                      "static_s": float(np.median([q["static_ocp_and_floor_s"] for q in rows]))}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["median"], indent=1))


if __name__ == "__main__":
    main()
