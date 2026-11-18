#!/usr/bin/env python3
"""Evaluation for the nonlinear moving-target dataset (gen_semilinear_moving_data.py).

analysis  rank of u(s, t) at T = 8; periodicity of the middle (u(t + P) vs u(t) on t in [3, 5]); distance of the middle
          to the LINEAR periodic turnpike (how nonlinear the periodic orbit is); u vs the LQ control
gap       (J(u) - J*) / J* with the nonlinear forward solve (T = 1, 8); references: LQ control, linear periodic turnpike
timing    single thread: exact (LQ PCG with harmonic-balance warm start + Newton-CG) at T = 1, 8 vs harmonic balance
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
from gen_semilinear_moving_data import cfg_of  # noqa: E402
from moving2d import assemble, eval_periodic, periodic_turnpike  # noqa: E402
from semilinear_moving import build, cost, forward, lq_control, solve  # noqa: E402


def summ(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "min": float(np.min(x))}


def relf(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def ulin_of(d, i, T):
    cf = cfg_of(d["X"][i], T)
    t = np.arange(1, cf.Nt + 1) * cf.dt
    return eval_periodic(d["uhat_lin"][i].astype(np.complex128), cf.omega, cf.phi0, t)


def analysis(d):
    te = d["test_idx"]
    rk, s21, per, nlp, nlq = [], [], [], [], []
    for r, i in enumerate(te):
        U = d["U_T8_test"][r].astype(np.float64)
        cf = cfg_of(d["X"][i], 8)
        Pk = int(round(2 * np.pi / cf.omega / cf.dt))
        k0 = 960
        sv = np.linalg.svd(U, compute_uv=False)
        rk.append(int(np.sum(sv > 1e-3 * sv[0])))
        s21.append(sv[1] / sv[0])
        per.append(relf(U[k0 : k0 + Pk], U[k0 + Pk : k0 + 2 * Pk]))
        nlp.append(relf(ulin_of(d, i, 8)[k0 : k0 + Pk], U[k0 : k0 + Pk]))
        nlq.append(relf(d["U_lq_T8_test"][r].astype(np.float64), U))
    return {"rank_1e-3_T8": summ(rk), "sv2_over_sv1": summ(s21), "periodicity_middle": summ(per),
            "middle_vs_linear_periodic_turnpike": summ(nlp), "u_vs_lq_control_T8": summ(nlq), "n_harm": summ(d["n_harm"]),
            "exact_seconds": {str(T): float(np.nanmedian(d[f"secs_T{T}"][te])) for T in (1, 2, 4, 8)},
            "newton": {str(T): float(np.nanmedian(d[f"newton_iters_T{T}"][te])) for T in (1, 2, 4, 8)},
            "cg": {str(T): float(np.nanmedian(d[f"cg_iters_T{T}"][te])) for T in (1, 2, 4, 8)}}


def gap_one(task):
    row, T, controls = task
    s = build(cfg_of(row, T), float(10 ** row[7]))
    ex = controls.pop("exact")
    Js = cost(s, forward(s, ex), ex)
    return {k: {"gap": (cost(s, forward(s, u), u) - Js) / Js, "rel_err": relf(u, ex)} for k, u in controls.items()}


def gap(d, preds, workers):
    te = d["test_idx"]
    tasks, keys = [], []
    for r, i in enumerate(te):
        for T in (1, 8):
            ex = (d["U_T1"][i] if T == 1 else d["U_T8_test"][r]).astype(np.float64)
            ctrl = {"exact": ex, "lq": d[f"U_lq_T{T}_test"][r].astype(np.float64), "lin_periodic": ulin_of(d, i, T),
                    **{lab: P[T][r] for lab, P in preds.items()}}
            tasks.append((d["X"][i], T, ctrl))
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
    te = d["test_idx"]
    lg = d["X"][te, 7]
    picks = [int(te[np.argmin(np.abs(lg - np.quantile(lg, q)))]) for q in np.linspace(0.2, 0.8, k)]
    rows = []
    for i in picks:
        row = d["X"][i]
        r = {"idx": i, "gamma": float(10 ** row[7]), "omega_over_pi": float(row[4] / np.pi)}
        t0 = time.perf_counter()
        _, hinfo = periodic_turnpike(assemble(cfg_of(row, 1)))
        r["harmonic_balance_s"] = time.perf_counter() - t0
        for T in (1, 8):
            cf = cfg_of(row, T)
            t0 = time.perf_counter()
            Clq, _, linfo = lq_control(cf)
            t1 = time.perf_counter()
            _, info = solve(build(cf, r["gamma"]), Clq)
            r[f"T{T}"] = {"lq_s": t1 - t0, "total_s": time.perf_counter() - t0, "pcg": linfo["iters"], "newton": info["newton_iters"], "cg": info["cg_iters"]}
        rows.append(r)
        print(json.dumps(r), flush=True)
    return {"samples": rows, "median": {"exact_T1_s": float(np.median([q["T1"]["total_s"] for q in rows])),
                                        "exact_T8_s": float(np.median([q["T8"]["total_s"] for q in rows])),
                                        "harmonic_balance_s": float(np.median([q["harmonic_balance_s"] for q in rows]))}}


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
