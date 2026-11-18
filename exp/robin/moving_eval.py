#!/usr/bin/env python3
"""Evaluation for the moving-target dataset (gen_moving_data.py).

analysis  rank of u(s, t) (T = 8), share of the layers (||u - u_hat|| / ||u||), u_hat vs exact in the middle,
          harmonics and PCG statistics
gap       (J(u) - J*) / J* with the linear forward solve, for --preds label=path.npz (T = 1, 8); 'periodic' = u_hat
timing    single thread: exact PCG (warm start from u_hat) at T = 1, 8 vs harmonic balance + TurnpikeNet query
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_moving_data import cfg_of  # noqa: E402
from moving2d import assemble, cost, eval_periodic, pcg_solve, periodic_turnpike, target_loads  # noqa: E402


def summ(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "min": float(np.min(x))}


def relf(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def uhat_of(d, i, T):
    row = d["X"][i]
    cf = cfg_of(row, T)
    t = np.arange(1, cf.Nt + 1) * cf.dt
    return eval_periodic(d["uhat"][i].astype(np.complex128), cf.omega, cf.phi0, t)


def analysis(d):
    te = d["test_idx"]
    rk, s21, lay, mid = [], [], [], []
    for r, i in enumerate(te):
        U = d["U_T8_test"][r].astype(np.float64)
        sv = np.linalg.svd(U, compute_uv=False)
        rk.append(int(np.sum(sv > 1e-3 * sv[0])))
        s21.append(sv[1] / sv[0])
        uh = uhat_of(d, i, 8)
        lay.append(relf(uh, U))
        mid.append(relf(uh[1100:1460], U[1100:1460]))
    return {"rank_1e-3_T8": summ(rk), "sv2_over_sv1": summ(s21), "layer_share_T8": summ(lay), "u_hat_vs_exact_middle": summ(mid),
            "n_harm": summ(d["n_harm"]), "harm_seconds": summ(d["harm_seconds"]),
            "pcg_iters": {str(T): float(np.nanmedian(d[f"pcg_iters_T{T}"][te])) for T in (1, 2, 4, 8)},
            "pcg_seconds": {str(T): float(np.nanmedian(d[f"pcg_secs_T{T}"][te])) for T in (1, 2, 4, 8)}}


def gap_one(task):
    row, T, controls = task
    ops = assemble(cfg_of(row, T))
    GY, csq = target_loads(ops)
    ex = controls.pop("exact")
    Js = cost(ops, GY, csq, ex)
    return {k: {"gap": (cost(ops, GY, csq, u) - Js) / Js, "rel_err": relf(u, ex)} for k, u in controls.items()}


def gap(d, preds, workers):
    te = d["test_idx"]
    tasks, keys = [], []
    for r, i in enumerate(te):
        for T in (1, 8):
            ex = (d["U_T1"][i] if T == 1 else d["U_T8_test"][r]).astype(np.float64)
            tasks.append((d["X"][i], T, {"exact": ex, "periodic": uhat_of(d, i, T), **{lab: P[T][r] for lab, P in preds.items()}}))
            keys.append(T)
    from multiprocessing import Pool

    with Pool(workers) as pool:
        res = pool.map(gap_one, tasks, chunksize=1)
    out = {}
    for T in (1, 8):
        rows = [q for q, TT in zip(res, keys) if TT == T]
        out[str(T)] = {lab: {"gap_median": float(np.median([q[lab]["gap"] for q in rows])), "gap_max": float(np.max([q[lab]["gap"] for q in rows])),
                             "gap_min": float(np.min([q[lab]["gap"] for q in rows])),
                             "rel_err_median": float(np.median([q[lab]["rel_err"] for q in rows])),
                             "rel_err_max": float(np.max([q[lab]["rel_err"] for q in rows]))} for lab in rows[0]}
        print(f"T={T}: " + "  ".join(f"{lab} gap {v['gap_median']:.1e}/{v['gap_max']:.1e} (min {v['gap_min']:.1e}) err {100 * v['rel_err_median']:.3f}%/{100 * v['rel_err_max']:.2f}%"
                                     for lab, v in out[str(T)].items()), flush=True)
    return out


def timing(d, k):
    import torch

    from moving_surrogate import MoveTurnpike
    from turnpike_surrogate import grid

    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    te = d["test_idx"]
    om = d["X"][te, 4]
    picks = [int(te[np.argmin(np.abs(om - np.quantile(om, q)))]) for q in np.linspace(0.2, 0.8, k)]
    net = MoveTurnpike(99).eval()
    rows = []
    for i in picks:
        row = d["X"][i]
        r = {"idx": i, "omega_over_pi": float(row[4] / np.pi), "sigma": float(10 ** row[5])}
        t0 = time.perf_counter()
        U, hinfo = periodic_turnpike(assemble(cfg_of(row, 1)))
        r["harmonic_balance_s"], r["n_harm"] = time.perf_counter() - t0, hinfo["n_harm"]
        for T in (1, 8):
            cf = cfg_of(row, T)
            t0 = time.perf_counter()
            ops = assemble(cf)
            GY, _ = target_loads(ops)
            tt = np.arange(1, cf.Nt + 1) * cf.dt
            _, info = pcg_solve(ops, GY, C0=eval_periodic(U, cf.omega, cf.phi0, tt))
            r[f"exact_T{T}_s"], r[f"pcg_iters_T{T}"] = time.perf_counter() - t0, info["iters"]
        Ut = torch.tensor(U[None])
        t, s = grid(8, 1 / 320)
        p = {"mu": torch.randn(1, 6), "phi0": torch.tensor([row[6]]), "omega": torch.tensor([row[4]]), "floor": torch.tensor([10.0]), "dt": 1 / 320,
             "row": {8: torch.tensor([0])}}

        def query():
            from moving_surrogate import periodic

            p["P"] = {8: periodic(Ut, p["omega"], p["phi0"], t)}
            return net(p, torch.tensor([0]), 8)

        with torch.no_grad():
            query()
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                query()
                ts.append(time.perf_counter() - t0)
        r["net_and_u_hat_eval_T8_s"] = float(np.median(ts))
        r["surrogate_T8_s"] = r["harmonic_balance_s"] + r["net_and_u_hat_eval_T8_s"]
        rows.append(r)
        print(json.dumps(r), flush=True)
    return {"samples": rows, "median": {k2: float(np.median([q[k2] for q in rows])) for k2 in ("exact_T1_s", "exact_T8_s", "harmonic_balance_s", "surrogate_T8_s")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--what", choices=["analysis", "gap", "timing"], required=True)
    ap.add_argument("--preds", nargs="*", default=[])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()
    d = np.load(args.data)
    if args.what == "analysis":
        rep = analysis(d)
    elif args.what == "gap":
        preds = {}
        for spec in args.preds:
            lab, path = spec.split("=", 1)
            z = np.load(path)
            assert np.array_equal(z["test_idx"], d["test_idx"]), path
            preds[lab] = {T: z[f"U_T{T}"].astype(np.float64) for T in (1, 8)}
        rep = gap(d, preds, args.workers)
    else:
        rep = timing(d, args.k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if args.what != "gap":
        print(json.dumps(rep, indent=1)[:3000])


if __name__ == "__main__":
    main()
