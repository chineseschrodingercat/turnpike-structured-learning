#!/usr/bin/env python3
"""Uniform timing for the linear 1D problem (constant target): one process, one thread (BLAS and torch),
wall clock per parameter sample mu, median over --k test samples.

exact:      monolithic KKT (T = 1, 8); reduced N x N system (T = 8); symplectic pencil / ARE
            (setup = generalized eigenproblem + static particular solution, query = one 2n x 2n solve, T = 8)
surrogate:  closed-form plateau u_bar (one sparse solve) + rate floor lambda_1(beta) + TurnpikeNet (r = 4) query,
            network weights untrained (cost is weight-independent), float64, batch 1
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
import scipy.sparse.linalg as spla  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hamiltonian_exact import build, control, particular, pencil_modes  # noqa: E402
from robin_solver import RobinConfig, assemble_operators, assemble_problem_data, reduced_control, reduced_system, solve_kkt  # noqa: E402
from turnpike_surrogate import TurnpikeNet, first_eig, grid  # noqa: E402

ONE = lambda x, t: np.ones_like(np.asarray(x, float))  # noqa: E731


def clock(f, reps=1):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    d = np.load(args.data)
    te = d["test_idx"][: args.k]
    net = TurnpikeNet(r=4, shared=True, use_floor=True, n_in=d["X"].shape[1]).eval()
    rows = []
    for i in te:
        geo = {k: float(d[k][i]) for k in ("alpha", "beta", "xa", "xb")}
        r = {}
        for T in (1, 8):
            cfg = RobinConfig(T=float(T), Nt=320 * T, **geo)
            r[f"kkt_T{T}"] = clock(lambda: solve_kkt(cfg, yd_func=ONE))
        cfg8 = RobinConfig(T=8.0, Nt=2560, **geo)

        def reduced():
            ops = assemble_operators(cfg8)
            red = reduced_system(ops, assemble_problem_data(cfg8, ops, ONE))
            return reduced_control(red, geo["alpha"])

        r["reduced_T8"] = clock(reduced)
        st = {}

        def setup():
            cfg, ops, G = build(geo)
            st["m"], st["p"], st["dt"] = pencil_modes(ops, geo["alpha"]), particular(ops, G, geo["alpha"], 0.0), cfg.dt

        r["pencil_setup"] = clock(setup)
        r["pencil_query_T8"] = clock(lambda: control(st["m"], st["p"], geo["alpha"], 0.0, 8, st["dt"], True), reps=5)

        def plateau():
            cfg, ops, G = build(geo)
            psi = spla.splu(ops.K_beta.tocsc()).solve(ops.e0)
            return float(psi @ G / (psi @ (ops.M_obs @ psi) + geo["alpha"]))

        r["surrogate_plateau"] = clock(plateau, reps=5)
        r["surrogate_floor"] = clock(lambda: first_eig(geo["beta"]), reps=5)
        t, s = grid(8, 1 / 320)
        x, fl = torch.randn(1, d["X"].shape[1]), torch.tensor([5.0])
        with torch.no_grad():
            net(x, t, s, fl)
            r["surrogate_net_T8"] = clock(lambda: net(x, t, s, fl), reps=20)
        r["surrogate_total_T8"] = r["surrogate_plateau"] + r["surrogate_floor"] + r["surrogate_net_T8"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    med = {k: float(np.median([q[k] for q in rows])) for k in rows[0]}
    rep = {"protocol": "single process, 1 thread (BLAS + torch), wall clock per mu", "median": med, "samples": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(med, indent=1))


if __name__ == "__main__":
    main()
