#!/usr/bin/env python3
"""Classical structure-exploiting solver: exact any-horizon controls and ARE turnpike layers.

Deviations from the (static or periodic) turnpike solve the homogeneous discrete optimality system
    A dy_k - (M/dt) dy_{k-1} - (1/alpha) E dp_k = 0,      E = e0 e0^T,
    A dp_k - (M/dt) dp_{k+1} + M_obs dy_k = 0,
so modes dy_k = mu^k Y, dp_k = mu^k P satisfy the generalized eigenproblem (symplectic pencil)
    [-M/dt, 0; M_obs, A] z = mu [-A, E/alpha; 0, M/dt] z,
whose eigenvalues pair as (mu, 1/mu). The n stable modes span the stable deflating subspace
(the one the Schur method uses to solve the ARE); the n unstable modes run backward from T.
Boundary conditions dy_0 = -y_hat and dp_{N+1} = -p_hat e^{i w t_{N+1}} give
    exact      : one 2n x 2n solve coupling both ends (any T, cost independent of T),
    ARE layers : stable part from dy_0 only, unstable part from dp_{N+1} only (T-invariant layers,
                 i.e. the turnpike decomposition with exact layers; error = end-to-end interaction).
The control is u_k = u_hat e^{i w t_k} + e0^T dp_k / alpha.
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
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin_solver import RobinConfig, assemble_operators, interval_load, solve_kkt  # noqa: E402

STEPS = 320
T_EVAL = (1, 2, 4, 8)
ONE = lambda x, t: np.ones_like(np.asarray(x, float))


def build(row_geo: dict):
    cfg = RobinConfig(alpha=row_geo["alpha"], beta=row_geo["beta"], xa=row_geo["xa"], xb=row_geo["xb"], T=1.0, Nt=STEPS)
    ops = assemble_operators(cfg)
    G = interval_load(ONE, cfg, 0.0, cfg.xa, cfg.xb)
    return cfg, ops, G


def pencil_modes(ops, alpha: float):
    n = ops.cfg.nx
    Mdt = (ops.M / ops.cfg.dt).toarray()
    A = ops.A.toarray()
    Mo = ops.M_obs.toarray()
    E = np.zeros((n, n))
    E[0, 0] = 1.0
    Z = np.zeros((n, n))
    L = np.block([[-Mdt, Z], [Mo, A]])
    R = np.block([[-A, E / alpha], [Z, Mdt]])
    mu, V = sla.eig(L, R)
    st = np.abs(mu) < 1.0
    if st.sum() != n:
        raise RuntimeError(f"expected {n} stable modes, got {st.sum()}")
    return {"mus": mu[st], "Vs": V[:, st], "muu": mu[~st], "Vu": V[:, ~st], "n": n,
            "slowest_stable_rate": float(-np.log(np.max(np.abs(mu[st]))) / ops.cfg.dt)}


def particular(ops, G: np.ndarray, alpha: float, omega: float):
    dt = ops.cfg.dt
    z = (1.0 - np.exp(-1j * omega * dt)) / dt
    e0 = ops.e0.astype(complex)
    psi = spla.splu((z * ops.M + ops.K_beta).tocsc().astype(complex)).solve(e0)
    u = np.vdot(psi, G) / (alpha + np.real(np.vdot(psi, ops.M_obs @ psi)))
    y = psi * u
    p = spla.splu((np.conj(z) * ops.M + ops.K_beta).tocsc().astype(complex)).solve(G - ops.M_obs @ y)
    return u, y, p


def control(modes: dict, part, alpha: float, omega: float, T: int, dt: float, exact: bool) -> np.ndarray:
    n, mus, Vs, muu, Vu = modes["n"], modes["mus"], modes["Vs"], modes["muu"], modes["Vu"]
    N = STEPS * T
    u_hat, y_hat, p_hat = part
    rhs = np.concatenate([-y_hat, -p_hat * np.exp(1j * omega * (N + 1) * dt)])
    if exact:
        Mtx = np.block([[Vs[:n], Vu[:n] * muu[None, :] ** (-(N + 1))],
                        [Vs[n:] * mus[None, :] ** (N + 1), Vu[n:]]])
        cd = np.linalg.solve(Mtx, rhs)
        c, d = cd[:n], cd[n:]
    else:
        c = np.linalg.solve(Vs[:n], rhs[:n])
        d = np.linalg.solve(Vu[n:], rhs[n:])
    k = np.arange(1, N + 1)
    dp0 = (mus[None, :] ** k[:, None]) @ (c * Vs[n]) + (muu[None, :] ** (k[:, None] - N - 1)) @ (d * Vu[n])
    return u_hat * np.exp(1j * omega * k * dt) + dp0 / alpha


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def summarize(e):
    e = np.asarray(e)
    return {"median": float(np.median(e)), "p90": float(np.percentile(e, 90)), "max": float(np.max(e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const-data", type=pathlib.Path, required=True)
    ap.add_argument("--osc-data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--kkt-timing-samples", type=int, default=3)
    args = ap.parse_args()
    report = {}

    for kind, path in (("const", args.const_data), ("osc", args.osc_data)):
        d = np.load(path)
        te = d["test_idx"]
        dt = float(d["dt"])
        err_exact = {T: [] for T in T_EVAL}
        err_layers = {T: [] for T in T_EVAL}
        t_eig, t_eval, rates = [], {T: [] for T in T_EVAL}, []
        for i in te:
            geo = {"alpha": float(d["alpha"][i]), "beta": float(d["beta"][i]), "xa": float(d["xa"][i]), "xb": float(d["xb"][i])}
            t0 = time.perf_counter()
            cfg, ops, G = build(geo)
            modes = pencil_modes(ops, geo["alpha"])
            part0 = particular(ops, G, geo["alpha"], 0.0)
            if kind == "osc":
                om = float(d["omega"][i])
                partw = particular(ops, G, geo["alpha"], om)
            t_eig.append(time.perf_counter() - t0)
            rates.append(modes["slowest_stable_rate"])
            for T in T_EVAL:
                t1 = time.perf_counter()
                for exact, store in ((True, err_exact), (False, err_layers)):
                    uc = control(modes, part0, geo["alpha"], 0.0, T, dt, exact).real
                    if kind == "const":
                        u, truth = uc, d[f"U_T{T}"][i]
                    else:
                        uw = control(modes, partw, geo["alpha"], om, T, dt, exact)
                        u = 1.25 * uc + d["A"][i] * uw.imag
                        truth = 1.25 * d[f"U_const_T{T}"][i] + d["A"][i] * d[f"U_sin_T{T}"][i]
                    store[T].append(rel(u, truth))
                    if exact:
                        t_eval[T].append(time.perf_counter() - t1)
        kkt_times = {}
        for T in (1, 8):
            ts = []
            for i in te[: args.kkt_timing_samples]:
                cfg = RobinConfig(alpha=float(d["alpha"][i]), beta=float(d["beta"][i]), xa=float(d["xa"][i]), xb=float(d["xb"][i]),
                                  T=float(T), Nt=STEPS * T)
                t0 = time.perf_counter()
                solve_kkt(cfg, yd_func=ONE)
                ts.append(time.perf_counter() - t0)
            kkt_times[str(T)] = float(np.median(ts))
        report[kind] = {
            "n_test": int(len(te)),
            "exact_any_T_rel_err": {str(T): summarize(err_exact[T]) for T in T_EVAL},
            "are_turnpike_layers_rel_err": {str(T): summarize(err_layers[T]) for T in T_EVAL},
            "seconds_per_mu_setup_eig_and_turnpike": float(np.median(t_eig)),
            "seconds_per_query_after_setup": {str(T): float(np.median(t_eval[T])) for T in T_EVAL},
            "seconds_kkt_single_target_single_thread": kkt_times,
            "slowest_closed_loop_rate": summarize(rates),
        }
        print(kind, json.dumps(report[kind], indent=1), flush=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
