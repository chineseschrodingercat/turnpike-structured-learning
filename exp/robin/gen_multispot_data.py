#!/usr/bin/env python3
"""Dataset: two hot spots moving along the controlled edge with independent (generically incommensurate) frequencies,
so the target is quasi-periodic (multispot_pilot.py geometry: strip [0, w] x [0.05, 0.95], n = 9900, m = 99, dt = 1/320).

mu = (w, log10 alpha, log10 beta, log10 gamma, and per spot j = 1, 2: A_j, c_j, a_j, omega_j, log10 sigma_j, phi_j).
Per sample and horizon: exact linear control (PCG, warm start = quasi-periodic reference) and exact semilinear control
(gamma y^3, Newton-CG warm-started from the linear control); the quasi-periodic linear reference as harmonics
(constant part u0 and, per spot, U_j[n] multiplying e^{i n theta_j}, zero-padded); lambda_1(beta); solver statistics.
Train/val at T = 1, 2; test at T = 1, 2, 4, 8. Samples are written one file each (resumable); --merge builds data.npz.
"""

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
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.stats import qmc  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import CfgMove, assemble, pcg_solve  # noqa: E402
from multispot_pilot import eval_ref, loads, qp_reference  # noqa: E402
from robin2d import lambda1  # noqa: E402
from semilinear_moving import SemiMove, solve as newton_solve  # noqa: E402

SPOT_KEYS = ["A", "c", "a", "omega", "log10_sigma", "phi"]
COLS = ["w", "log10_alpha", "log10_beta", "log10_gamma"] + [f"{k}{j}" for j in (1, 2) for k in SPOT_KEYS]
RANGES = {"w": (0.05, 0.3), "log10_alpha": (-3.25, -2.25), "log10_beta": (-1.0, 1.0), "log10_gamma": (np.log10(20.0), np.log10(1000.0)),
          "A1": (0.5, 1.0), "A2": (-1.0, 1.0)}
for j in (1, 2):
    RANGES.update({f"c{j}": (0.3, 0.7), f"a{j}": (0.05, 0.25), f"omega{j}": (np.pi, 4 * np.pi),
                   f"log10_sigma{j}": (np.log10(0.04), np.log10(0.1)), f"phi{j}": (0.0, 2 * np.pi)})
T_TEST, T_TRAIN, NH_MAX = (1, 2, 4, 8), (1, 2), 48


def spots_of(row):
    out = []
    for j in range(2):
        A, c, a, om, ls, ph = row[4 + 6 * j : 10 + 6 * j]
        out.append((float(A), float(c), float(a), float(om), float(10**ls), float(ph)))
    return out


def solve_one(args):
    i, row, horizons, outdir = args
    f = outdir / f"sample_{i:04d}.npz"
    if f.exists():
        return i, "skip"
    spots, gamma = spots_of(row), float(10 ** row[3])
    pay = {"row": row, "lam1": lambda1(100, float(10 ** row[2]))}
    ref = None
    for T in horizons:
        cf = CfgMove(alpha=float(10 ** row[1]), beta=float(10 ** row[2]), w=float(row[0]), T=float(T), Nt=int(320 * T))
        ops = assemble(cf)
        GY, csq = loads(ops, spots)
        if ref is None:
            t0 = time.perf_counter()
            ref = qp_reference(ops, spots, nmax=NH_MAX)
            pay["ref_secs"] = time.perf_counter() - t0
        u0, parts, nharm = ref
        t = np.arange(1, cf.Nt + 1) * cf.dt
        t0 = time.perf_counter()
        C_lin, il = pcg_solve(ops, GY, C0=eval_ref(u0, parts, t), tol=1e-10)
        t_lin = time.perf_counter() - t0
        s = SemiMove(cfg=cf, gamma=gamma, o=ops, ML=np.asarray(ops.M.sum(axis=1)).ravel(), lu=spla.splu(ops.A.tocsc()),
                     Mdt=(ops.M / cf.dt).tocsr(), GY=GY, csq=csq, stats={"lu_solves": 0})
        C_nl, inl = newton_solve(s, C_lin, gtol=1e-8)
        pay.update({f"lin_{T}": C_lin.astype(np.float32), f"nl_{T}": C_nl.astype(np.float32), f"lin_secs_{T}": t_lin,
                    f"pcg_iters_{T}": il["iters"], f"nl_secs_{T}": inl["seconds"], f"newton_iters_{T}": inl["newton_iters"],
                    f"grad_rel_{T}": inl["grad_rel"]})
    U = np.zeros((2, NH_MAX + 1, 99), np.complex64)
    for j, (_, _, Uj) in enumerate(ref[1]):
        U[j, : Uj.shape[0]] = Uj
    pay.update({"u0": ref[0].astype(np.float64), "uhat": U, "nharm": np.array(ref[2])})
    np.savez(f, **pay)
    return i, "ok"


def merge(args, X, tr, va, te):
    S = args.out.parent / "samples"
    rows = [dict(np.load(S / f"sample_{i:04d}.npz")) for i in range(args.n)]
    pay = {}
    for kind in ("nl", "lin"):
        for T in T_TRAIN:
            pay[f"U_{kind}_T{T}"] = np.stack([r[f"{kind}_{T}"] for r in rows])
        for T in (4, 8):
            pay[f"U_{kind}_T{T}_test"] = np.stack([rows[i][f"{kind}_{T}"] for i in te])
    pay.update({"u0": np.stack([r["u0"] for r in rows]), "uhat": np.stack([r["uhat"] for r in rows]), "nharm": np.stack([r["nharm"] for r in rows]),
                "lam1": np.array([float(r["lam1"]) for r in rows]), "gamma": 10.0 ** X[:, 3]})
    for T in T_TEST:
        for k in ("lin_secs", "pcg_iters", "nl_secs", "newton_iters", "grad_rel"):
            pay[f"{k}_T{T}"] = np.array([float(r.get(f"{k}_{T}", np.nan)) for r in rows])
    np.savez_compressed(args.out, X=X, train_idx=tr, val_idx=va, test_idx=te, dt=1.0 / 320, m=99, nh=100,
                        metadata_json=json.dumps({"input_columns": COLS, "ranges": {k: list(v) for k, v in RANGES.items()},
                                                  "pde": "y_t - Laplace y + gamma y^3 = 0 (lin: gamma = 0)",
                                                  "target": "sum_j A_j exp(-(s - c_j - a_j sin(omega_j t + phi_j))^2 / (2 sigma_j^2))",
                                                  "seed": args.seed}), **pay)
    nl = np.linalg.norm((pay["U_nl_T1"] - pay["U_lin_T1"]).reshape(args.n, -1), axis=1) / np.linalg.norm(pay["U_lin_T1"].reshape(args.n, -1), axis=1)
    print(f"merged {args.n} -> {args.out}; Newton T=1 median {np.nanmedian(pay['newton_iters_T1']):.0f} max {np.nanmax(pay['newton_iters_T1']):.0f}; "
          f"max grad_rel {max(np.nanmax(pay[f'grad_rel_T{T}']) for T in T_TEST):.1e}; nl secs T=8 median {np.nanmedian(pay['nl_secs_T8']):.0f}; "
          f"|u_nl - u_lin|/|u_lin| (T=1) median {np.median(nl):.3f} p90 {np.percentile(nl, 90):.3f}; max nharm {pay['nharm'].max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/multispot_data/data.npz"))
    ap.add_argument("--n", type=int, default=1350)
    ap.add_argument("--n-val", type=int, default=150)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--only", type=int, nargs="*", default=None, help="solve only these sample indices (smoke test)")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    lo = np.array([RANGES[c][0] for c in COLS])
    hi = np.array([RANGES[c][1] for c in COLS])
    X = lo + qmc.LatinHypercube(d=len(COLS), seed=args.seed).random(args.n) * (hi - lo)
    perm = np.random.default_rng(args.seed + 1).permutation(args.n)
    te, va, tr = np.sort(perm[: args.n_test]), np.sort(perm[args.n_test : args.n_test + args.n_val]), np.sort(perm[args.n_test + args.n_val :])
    if args.merge:
        merge(args, X, tr, va, te)
        return
    S = args.out.parent / "samples"
    S.mkdir(parents=True, exist_ok=True)
    is_test = np.zeros(args.n, bool)
    is_test[te] = True
    idx = args.only if args.only is not None else list(te) + list(np.concatenate([va, tr]))
    jobs = [(int(i), X[i], T_TEST if is_test[i] else T_TRAIN, S) for i in idx]
    from multiprocessing import Pool

    t0 = time.perf_counter()
    done = 0
    with Pool(min(args.workers, len(jobs))) as pool:
        for i, st in pool.imap_unordered(solve_one, jobs, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} done ({time.perf_counter() - t0:.0f}s), last {i} {st}", flush=True)


if __name__ == "__main__":
    main()
