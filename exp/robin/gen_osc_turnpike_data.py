#!/usr/bin/env python3
"""Multi-horizon data for time-varying targets y_d(t) = c + A sin(omega t) on omega_o.

By linearity u = c * U_const + A * U_sin, so per sample we store the controls for
the three targets 1, sin(omega t), cos(omega t) at horizons T in {1, 2, 4, 8}
(dt = 1/320). V = U_cos + i U_sin is the control for the complex target e^{i omega t}.

Periodic turnpike (closed form, discrete-exact for backward Euler): with
z = (1 - exp(-i omega dt)) / dt and (z M + K_beta) psi = e0,
    u_hat(omega) = psi^H G / (alpha + psi^H M_obs psi),      u_hat(0) = u_bar,
and the periodic optimal control for the target e^{i omega t} is u_hat e^{i omega t}.

Parameters: (w, c_rel, log10 alpha, log10 beta) as in gen_turnpike_data.py, omega in
[pi, 4 pi], A in [0.25, 0.75]; c = 1.25 at evaluation (report's oscillatory config).
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
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robin_solver import RobinConfig, assemble_operators, impulse_response, interval_load, solve_kkt  # noqa: E402

T_LIST = (1, 2, 4, 8)
STEPS = 320
COLS = ["w", "c_rel", "log10_alpha", "log10_beta", "omega", "A"]
RANGES = {"w": (0.1, 0.5), "c_rel": (0.0, 1.0), "log10_alpha": (-3.0, 0.0), "log10_beta": (-1.0, 1.0),
          "omega": (np.pi, 4 * np.pi), "A": (0.25, 0.75)}
ONE = lambda x, t: np.ones_like(np.asarray(x, float))


def setup(row: np.ndarray, T_max: int):
    w, c_rel, la, lb, om, _ = row
    c = w / 2 + c_rel * (1 - w)
    cfg = RobinConfig(alpha=10.0**la, beta=10.0**lb, xa=c - w / 2, xb=c + w / 2, T=float(T_max), Nt=STEPS * T_max)
    return cfg, assemble_operators(cfg), om


def turnpike_amplitude(ops, G: np.ndarray, alpha: float, omega: float) -> complex:
    dt = ops.cfg.dt
    z = (1.0 - np.exp(-1j * omega * dt)) / dt
    psi = spla.splu((z * ops.M + ops.K_beta).tocsc().astype(complex)).solve(ops.e0.astype(complex))
    return complex(np.vdot(psi, G) / (alpha + np.real(np.vdot(psi, ops.M_obs @ psi))))


def solve_family(row: np.ndarray, T_list=T_LIST) -> dict:
    T_max = max(T_list)
    cfg, ops, om = setup(row, T_max)
    n_max = STEPS * T_max
    a = impulse_response(ops)
    G = interval_load(ONE, cfg, 0.0, cfg.xa, cfg.xb)
    h = a @ G  # h[m] = a_m^T G
    R = a @ (ops.M_obs @ a.T)
    sizes = [STEPS * T for T in T_list]
    H = {N: np.zeros((N, N)) for N in sizes}
    for d in range(n_max):
        cs = np.cumsum(np.diagonal(R, offset=-d))
        for N in sizes:
            if N > d:
                j = np.arange(d, N)
                H[N][j - d, j] = cs[N - 1 - j]
                H[N][j, j - d] = cs[N - 1 - j]
    out = {}
    for T, N in zip(T_list, sizes):
        t = np.arange(1, N + 1) * cfg.dt
        F = np.stack([np.ones(N), np.sin(om * t), np.cos(om * t)], axis=1)  # target time profiles
        # g[j] = sum_{m=0}^{N-1-j} h[m] f[m+j]
        g = np.stack([np.array([h[: N - j] @ F[j:, q] for j in range(N)]) for q in range(3)], axis=1)
        U = np.linalg.solve(H[N] + cfg.alpha * np.eye(N), g)
        out[T] = U  # (N, 3): const, sin, cos
    out["u_bar"] = turnpike_amplitude(ops, G, cfg.alpha, 0.0).real
    out["u_hat"] = turnpike_amplitude(ops, G, cfg.alpha, om)
    return out


def check(rows: np.ndarray) -> None:
    for row in rows:
        fam = solve_family(row, T_list=(1, 2, 8))
        cfg, ops, om = setup(row, 1)
        kkt = solve_kkt(cfg, yd_func=lambda x, t: np.full_like(np.asarray(x, float), np.sin(om * t)))
        V8 = fam[8][:, 2] + 1j * fam[8][:, 1]
        t8 = np.arange(1, 8 * STEPS + 1) / STEPS
        mid = (t8 >= 3) & (t8 <= 5)
        per = fam["u_hat"] * np.exp(1j * om * t8[mid])
        print(json.dumps({
            "omega": float(om), "alpha": cfg.alpha, "beta": cfg.beta,
            "sin_T1_vs_kkt": float(np.linalg.norm(fam[1][:, 1] - kkt.U) / np.linalg.norm(kkt.U)),
            "periodic_turnpike_vs_T8_mid": float(np.linalg.norm(V8[mid] - per) / np.linalg.norm(V8[mid])),
            "u_bar_vs_T8_mid_const": float(np.max(np.abs(fam[8][mid, 0] - fam["u_bar"])) / abs(fam["u_bar"])),
        }))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)
    if args.check:
        check(X[:3])
        return

    from multiprocessing import Pool

    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        fams = pool.map(solve_family, list(X), chunksize=2)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
    payload = {}
    for T in T_LIST:
        arr = np.array([f[T] for f in fams])  # (n, N, 3)
        payload[f"U_const_T{T}"], payload[f"U_sin_T{T}"], payload[f"U_cos_T{T}"] = arr[..., 0], arr[..., 1], arr[..., 2]
    w, c_rel = X[:, 0], X[:, 1]
    c = w / 2 + c_rel * (1 - w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, X=X, xa=c - w / 2, xb=c + w / 2, alpha=10.0 ** X[:, 2], beta=10.0 ** X[:, 3], omega=X[:, 4], A=X[:, 5],
        u_bar=np.array([f["u_bar"] for f in fams]), u_hat=np.array([f["u_hat"] for f in fams]),
        train_idx=np.sort(perm[:n_tr]), val_idx=np.sort(perm[n_tr : n_tr + n_va]), test_idx=np.sort(perm[n_tr + n_va :]),
        dt=1.0 / STEPS, T_list=np.array(T_LIST), offset_c=1.25,
        metadata_json=json.dumps({"input_columns": COLS, "ranges": RANGES, "targets": ["1", "sin(omega t)", "cos(omega t)"],
                                  "evaluation_target": "1.25 + A sin(omega t)", "seed": args.seed}),
        **payload,
    )
    print(f"{args.n} samples x {T_LIST} x 3 targets in {time.perf_counter() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
