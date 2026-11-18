#!/usr/bin/env python3
"""Uniform timing protocol for the semilinear problem: one process, one thread (BLAS and torch), same machine,
wall clock per parameter sample, nx = 160, dt = 1/320.

Exact:      Newton-KKT from zero; LQ solve + Newton from the LQ solution (what the generator does);
            Newton warm-started from a surrogate prediction (forward + adjoint sweep + Newton), if --pred is given.
Surrogate:  static nonlinear OCP (u_bar, lam1_lin) + network inference. Network cost does not depend on the
            trained weights, so untrained networks of the trained architectures are timed (float64, as trained);
            batch 1 (one query) and batch = test-set size (amortized).
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
from baselines_multihorizon import DeepONet, DilatedCNN, FNO1d  # noqa: E402
from robin_solver import RobinConfig  # noqa: E402
from semilinear1d import adjoint, build, forward, solve_kkt, static_ocp  # noqa: E402
from turnpike_surrogate import TurnpikeNet, grid  # noqa: E402


def clock(f, reps=1):
    ts, out = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--pred", type=pathlib.Path, default=None, help="surrogate predictions for the warm start")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    d = np.load(args.data)
    te = d["test_idx"]
    ge = d["gamma_eff"][te]
    rows = [int(np.argmin(np.abs(ge - np.quantile(ge, q)))) for q in np.linspace(0.1, 0.9, args.k)]
    pred = np.load(args.pred) if args.pred else None
    rep = {"protocol": "single process, OMP/OPENBLAS/MKL/torch threads = 1, wall clock per sample", "samples": [], "nets": {}}
    for r in rows:
        i = int(te[r])
        geo = dict(alpha=float(d["alpha"][i]), beta=float(d["beta"][i]), xa=float(d["xa"][i]), xb=float(d["xb"][i]))
        gamma, T_ref = float(d["gamma"][i]), float(d["T_ref"][i])
        row = {"idx": i, "gamma_eff": float(ge[r])}
        row["static_ocp_s"], _ = clock(lambda: static_ocp(build(RobinConfig(nx=160, Nt=320, **geo), gamma, T_ref)), reps=5)
        for T in (1, 8):
            cfg = RobinConfig(nx=160, Nt=320 * T, T=float(T), **geo)
            s = build(cfg, gamma, T_ref)
            sec, (_, _, U0, inf0) = clock(lambda: solve_kkt(s))
            row[f"T{T}_newton_zero"] = {"s": sec, "iters": inf0["newton_iters"]}

            def lq_then_newton():
                lq = solve_kkt(build(cfg, 0.0, T_ref))
                return solve_kkt(s, init=lq[:3])

            sec, (_, _, U1, inf1) = clock(lq_then_newton)
            row[f"T{T}_lq_then_newton"] = {"s": sec, "iters": inf1["newton_iters"], "same": float(np.linalg.norm(U1 - U0) / np.linalg.norm(U0))}
            if pred is not None:
                up = pred[f"U_T{T}"][r]

                def warm():
                    Y = forward(s, up)
                    return solve_kkt(s, init=(Y, adjoint(s, Y), up))

                sec, (_, _, U2, inf2) = clock(warm)
                row[f"T{T}_surrogate_warm_newton"] = {"s": sec, "iters": inf2["newton_iters"],
                                                      "same": float(np.linalg.norm(U2 - U0) / np.linalg.norm(U0)),
                                                      "pred_rel_err": float(np.linalg.norm(up - U0) / np.linalg.norm(U0))}
        rep["samples"].append(row)
        print(json.dumps(row), flush=True)

    n_par, nb = d["X"].shape[1], len(te)
    torch.manual_seed(0)
    nets = {"turnpike": TurnpikeNet(r=4, shared=True, use_floor=True, n_in=n_par), "cnn": DilatedCNN(n_par),
            "deeponet": DeepONet(n_par), "fno": FNO1d(n_par)}
    for name, net in nets.items():
        net.eval()
        rep["nets"][name] = {}
        for T in (1, 8):
            t, s_ = grid(T, float(d["dt"]))
            for b in (1, nb):
                x, yd, fl = torch.randn(b, n_par), torch.ones(b, len(t)), torch.full((b,), 5.0)
                f = (lambda: net(x, t, s_, fl)) if name == "turnpike" else (lambda: net(x, yd, t, s_))
                with torch.no_grad():
                    for _ in range(3):
                        f()
                    sec, _ = clock(f, reps=20)
                rep["nets"][name][f"T{T}_batch{b}_per_sample_s"] = sec / b
        print(name, rep["nets"][name], flush=True)

    keys = [k for k in rep["samples"][0] if k not in ("idx", "gamma_eff")]
    rep["median_over_samples"] = {k: (float(np.median([q[k] for q in rep["samples"]])) if not isinstance(rep["samples"][0][k], dict)
                                      else {kk: float(np.median([q[k][kk] for q in rep["samples"]])) for kk in rep["samples"][0][k]})
                                  for k in keys}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep["median_over_samples"], indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
