#!/usr/bin/env python3
"""Evaluation for the 2D semilinear distributed-control dataset (gen_semilinear2d_mi_data.py).

analysis  turnpike structure on the test split: plateau vs static nonlinear OCP, T-invariant superposition with layers
          from T = 8 (relative L2 over (t, s)), nonlinearity measures (u_bar vs u_bar_LQ = sum_k tau_k u_bar_k; u vs the LQ
          control), per-sample rank of u(s, t) and of the layer part u - u_bar
gap       (J(u) - J*) / J* with the nonlinear forward solve; references: LQ control, plateau u_bar(s) for all t
timing    single thread: LQ PCG + Newton-CG at T = 1, 8 vs static nonlinear OCP (+ network, timed separately)
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
from gen_semilinear2d_mi_data import M, cfg_of  # noqa: E402
from semilinear2d_mi import build, cost, forward, lq_control, solve, static_ocp  # noqa: E402


def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "min": float(np.min(x))}


def relf(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def analysis(d):
    te = d["test_idx"]
    U8 = d["U_T8_test"].astype(np.float64)
    ub, ubl = d["u_bar"][te], d["u_bar_lq"][te]
    half = U8.shape[1] // 2
    rep = {"plateau_rel_err_mid_T8": summ([relf(U8[r, half - 1], ub[r]) for r in range(len(te))])}
    L0 = U8[:, :half] - ub[:, None, :]
    L1 = U8[:, ::-1][:, :half] - ub[:, None, :]
    for T in (1, 2, 4):
        UT = (d[f"U_T{T}"][te] if T <= 2 else d[f"U_T{T}_test"]).astype(np.float64)
        N = UT.shape[1]
        k = np.arange(N)
        rep[f"superposition_T{T}"] = summ([relf(ub[r][None, :] + L0[r, k] + L1[r, N - 1 - k], UT[r]) for r in range(len(te))])
    rep["plateau_nonlinear_vs_lq"] = summ([relf(ub[r], ubl[r]) for r in range(len(te))])
    rep["control_T8_nonlinear_vs_lq"] = summ([relf(d["U_lq_T8_test"][r].astype(np.float64), U8[r]) for r in range(len(te))])
    s21, l21, lrank = [], [], []
    for r in range(len(te)):
        sv = np.linalg.svd(U8[r], compute_uv=False)
        s21.append(sv[1] / sv[0])
        sl = np.linalg.svd(U8[r] - ub[r][None, :], compute_uv=False)
        l21.append(sl[1] / sl[0])
        lrank.append(int(np.sum(sl > 1e-3 * sl[0])))
    rep["sv2_over_sv1_u"] = summ(s21)
    rep["sv2_over_sv1_layers"] = summ(l21)
    rep["layer_rank_1e-3"] = summ(lrank)
    Ub = d["u_bar"][d["train_idx"]]
    sv = np.linalg.svd(Ub - 0 * Ub.mean(0), compute_uv=False)
    e = np.cumsum(sv**2) / np.sum(sv**2)
    rep["plateau_family_modes_99_999_9999"] = [int(np.searchsorted(e, q) + 1) for q in (0.99, 0.999, 0.9999)]
    rep["exact_seconds_median"] = {str(T): float(np.nanmedian(d[f"secs_T{T}"][te])) for T in (1, 2, 4, 8)}
    rep["newton_iters_median"] = {str(T): float(np.nanmedian(d[f"newton_iters_T{T}"][te])) for T in (1, 2, 4, 8)}
    rep["cg_iters_median"] = {str(T): float(np.nanmedian(d[f"cg_iters_T{T}"][te])) for T in (1, 2, 4, 8)}
    return rep


def gap_one(task):
    row, T, ub, controls = task
    s = build(cfg_of(row, T), M, float(10.0 ** row[9]))
    controls = dict(controls, plateau=np.tile(ub, (int(320 * T), 1)))
    ex = controls.pop("exact")
    Jstar = cost(s, forward(s, ex), ex)
    return {k: {"gap": (cost(s, forward(s, u), u) - Jstar) / Jstar, "rel_err": relf(u, ex)} for k, u in controls.items()}


def gap(d, preds, workers):
    te = d["test_idx"]
    tasks, keys = [], []
    for r, i in enumerate(te):
        for T in (1, 8):
            ex = (d["U_T1"][i] if T == 1 else d["U_T8_test"][r]).astype(np.float64)
            ctrl = {"exact": ex, "lq": d[f"U_lq_T{T}_test"][r].astype(np.float64), **{lab: P[T][r] for lab, P in preds.items()}}
            tasks.append((d["X"][i], T, d["u_bar"][i], ctrl))
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
        print(f"T={T}: " + "  ".join(f"{lab} gap {v['gap_median']:.1e}/{v['gap_max']:.1e} err {100 * v['rel_err_median']:.3f}%/{100 * v['rel_err_max']:.2f}%"
                                     for lab, v in out[str(T)].items()), flush=True)
    return out


def timing(d, k):
    te = d["test_idx"]
    lg = d["X"][te, 9]
    picks = [int(te[np.argmin(np.abs(lg - np.quantile(lg, q)))]) for q in np.linspace(0.2, 0.8, k)]
    rows = []
    for i in picks:
        row = d["X"][i]
        r = {"idx": i, "gamma": float(10.0 ** row[9])}
        for T in (1, 8):
            cfg = cfg_of(row, T)
            t0 = time.perf_counter()
            Clq = lq_control(cfg, M)
            t1 = time.perf_counter()
            _, info = solve(build(cfg, M, r["gamma"]), Clq)
            r[f"T{T}"] = {"lq_s": t1 - t0, "total_s": time.perf_counter() - t0, "newton_iters": info["newton_iters"], "cg_iters": info["cg_iters"]}
        t0 = time.perf_counter()
        static_ocp(build(cfg_of(row, 1), M, r["gamma"]))
        r["static_s"] = time.perf_counter() - t0
        rows.append(r)
        print(json.dumps(r), flush=True)
    return {"samples": rows, "median": {"exact_T1_s": float(np.median([q["T1"]["total_s"] for q in rows])),
                                        "exact_T8_s": float(np.median([q["T8"]["total_s"] for q in rows])),
                                        "static_s": float(np.median([q["static_s"] for q in rows]))}}


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
    print(json.dumps(rep if args.what != "gap" else "done", indent=1)[:4000])


if __name__ == "__main__":
    main()
