#!/usr/bin/env python3
"""Dataset for the downstream-observation advection problem (advect_moving.py):
    y_t - eps Laplace y + bx(s) dy/dx = 0,  bx(s) = b0 (1 + gamma (s - 0.5)),  edge control at x = 0 (m = 99),
    observation strip x in [0.6, 0.8], moving hot spot c(t) = 0.5 + a sin(omega t + phi0), width sigma.
mu = (gamma, log10 eps, phi0, a, log10 sigma); fixed b0 = 2, alpha = 1e-3, beta = 0.5, omega = 2 pi, dt = 1/320.
Per sample: exact periodic turnpike (harmonic balance, harmonics stored) and the exact optimal control (PCG warm-started
from it) at T = 2; test samples also at T = 4. One file per sample in --work (resumable, progress logged), then
--merge builds the dataset npz (train / val / test = first n_train / next n_val / last n_test)."""

from __future__ import annotations

import argparse
import dataclasses
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
import advect_moving as am  # noqa: E402
import moving2d  # noqa: E402

FIX = dict(nh=100, m=99, alpha=1e-3, beta=0.5, w=0.2, omega=2 * np.pi, xo=0.6, bx=2.0, shear="gen", ya=0.05, yb=0.95)
STEPS_PER_UNIT = 320
NH_MAX = 80
MU_NAMES = ["gamma", "log10_eps", "phi0", "a", "log10_sigma"]


def sample_mu(rng, n):
    return np.stack([rng.uniform(0.0, 1.0, n), rng.uniform(-3.0, -1.5, n), rng.uniform(0.0, 2 * np.pi, n),
                     rng.uniform(0.15, 0.3, n), rng.uniform(np.log10(0.05), np.log10(0.1), n)], axis=1)


def cfg_of(mu, T):
    g, le, ph, a, ls = mu
    return am.CfgAdv(**FIX, gamma=float(g), eps=float(10.0**le), phi0=float(ph), a=float(a), sigma=float(10.0**ls),
                     T=float(T), Nt=int(round(STEPS_PER_UNIT * T)))


def solve_one(task):
    i, mu, horizons, out = task
    path = pathlib.Path(out) / f"sample_{i:04d}.npz"
    if path.exists():
        return i, "skip"
    t0 = time.perf_counter()
    c0 = cfg_of(mu, horizons[0])
    ops = am.assemble(c0)
    U, hinfo = moving2d.periodic_turnpike(ops, nmax=NH_MAX - 1)
    t_hb = time.perf_counter() - t0
    uhat = np.zeros((NH_MAX, c0.m), dtype=np.complex64)
    uhat[: U.shape[0]] = U
    rec = {"mu": mu, "uhat": uhat, "n_harm": hinfo["n_harm"], "hb_seconds": t_hb}
    for T in horizons:
        c = cfg_of(mu, T)
        o = dataclasses.replace(ops, cfg=c, A=ops.A)  # same M, K, dt; only the horizon changes
        GY, csq = moving2d.target_loads(o)
        t = np.arange(1, c.Nt + 1) * c.dt
        Up = moving2d.eval_periodic(U, c.omega, c.phi0, t)
        t1 = time.perf_counter()
        C, info = am.pcg_solve(o, GY, C0=Up, tol=1e-9)
        rec[f"U_T{T}"] = C.astype(np.float32)
        rec[f"J_T{T}"] = moving2d.cost(o, GY, csq, C)
        rec[f"iters_T{T}"] = info["iters"]
        rec[f"pcg_seconds_T{T}"] = time.perf_counter() - t1
    rec["total_seconds"] = time.perf_counter() - t0
    np.savez(path, **rec)
    return i, f"{rec['total_seconds']:.0f}s"


def pilot(args):
    """Worst-case layer lengths: deviation from the periodic turnpike along t at T = 2 and T = 4, and the
    T = 2 vs T = 4 initial-layer consistency (a floor for any superposition surrogate trained on T = 2)."""
    cases = ([1.0, -3.0, 0.5, 0.3, np.log10(0.05)], [0.0, -3.0, 0.5, 0.3, np.log10(0.05)], [0.5, -1.5, 2.0, 0.2, np.log10(0.07)])
    for mu in [cases[args.pilot]]:
        mu = np.array(mu)
        res = {}
        c0 = cfg_of(mu, 2)
        ops = am.assemble(c0)
        U, hinfo = moving2d.periodic_turnpike(ops, nmax=NH_MAX - 1)
        dev = {}
        for T in (2, 4):
            c = cfg_of(mu, T)
            o = dataclasses.replace(ops, cfg=c)
            GY, _ = moving2d.target_loads(o)
            t = np.arange(1, c.Nt + 1) * c.dt
            Up = moving2d.eval_periodic(U, c.omega, c.phi0, t)
            C, info = am.pcg_solve(o, GY, C0=Up, tol=1e-9)
            nrm = np.linalg.norm(Up) / np.sqrt(c.Nt)
            dev[T] = np.linalg.norm(C - Up, axis=1) / nrm
            res[f"T{T}"] = {"iters": info["iters"], "seconds": round(info["seconds"]),
                            "dev_at": {f"{tt:.1f}": float(dev[T][int(tt * STEPS_PER_UNIT) - 1]) for tt in np.arange(0.2, T, 0.2)}}
            res[f"T{T}"]["rel_periodic_only"] = float(np.linalg.norm(C - Up) / np.linalg.norm(C))
            if T == 4:
                k = int(0.8 * STEPS_PER_UNIT)
                res["layer_T2_vs_T4_on_[0,0.8]"] = float(np.linalg.norm((C4 := C)[:k] - C2[:k]) / np.linalg.norm(C4[:k] - Up[:k]))
            else:
                C2 = C
        print(json.dumps({"mu": mu.round(3).tolist(), "n_harm": hinfo["n_harm"], **res}), flush=True)


def merge(args):
    files = sorted(pathlib.Path(args.work).glob("sample_*.npz"))
    n = args.n_train + args.n_val + args.n_test
    assert len(files) == n, f"{len(files)} of {n} samples present"
    recs = [np.load(f) for f in files]
    te0 = args.n_train + args.n_val
    out = {"X": np.stack([r["mu"] for r in recs]), "mu_names": np.array(MU_NAMES), "uhat": np.stack([r["uhat"] for r in recs]),
           "n_harm": np.array([int(r["n_harm"]) for r in recs]), "U_T2": np.stack([r["U_T2"] for r in recs]),
           "U_T4_test": np.stack([recs[i]["U_T4"] for i in range(te0, n)]),
           "J_T2": np.array([float(r["J_T2"]) for r in recs]), "J_T4_test": np.array([float(recs[i]["J_T4"]) for i in range(te0, n)]),
           "iters_T2": np.array([int(r["iters_T2"]) for r in recs]), "hb_seconds": np.array([float(r["hb_seconds"]) for r in recs]),
           "pcg_seconds_T2": np.array([float(r["pcg_seconds_T2"]) for r in recs]),
           "pcg_seconds_T4_test": np.array([float(recs[i]["pcg_seconds_T4"]) for i in range(te0, n)]),
           "train_idx": np.arange(args.n_train), "val_idx": np.arange(args.n_train, te0), "test_idx": np.arange(te0, n),
           "dt": 1.0 / STEPS_PER_UNIT, "m": FIX["m"], "fixed": json.dumps(FIX)}
    np.savez(args.out, **out)
    print(f"wrote {args.out}: {n} samples; PCG iters T=2 median {np.median(out['iters_T2']):.0f}; exact seconds T=2 median "
          f"{np.median(out['hb_seconds'] + out['pcg_seconds_T2']):.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=pathlib.Path, default=pathlib.Path("runs/advect_data/samples"))
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/advect_data/advect_data.npz"))
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pilot", type=int, default=-1, help="run worst-case layer check number k (0..2) and exit")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.pilot >= 0:
        return pilot(args)
    if args.merge:
        return merge(args)
    n_tv, n = args.n_train + args.n_val, args.n_train + args.n_val + args.n_test
    mu = sample_mu(np.random.default_rng(args.seed), n)
    args.work.mkdir(parents=True, exist_ok=True)
    tasks = [(i, mu[i], (2, 4), str(args.work)) for i in range(n_tv, n)] + [(i, mu[i], (2,), str(args.work)) for i in range(n_tv)]
    from multiprocessing import Pool

    t0, done = time.perf_counter(), 0
    with Pool(args.workers) as pool:
        for i, msg in pool.imap_unordered(solve_one, tasks):
            done += 1
            el = time.perf_counter() - t0
            print(f"[{done}/{n}] sample {i} {msg}  elapsed {el / 60:.1f} min  ETA {el / done * (n - done) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
