#!/usr/bin/env python3
"""Turnpike structure of the 2D semilinear dataset (test split): plateau vs static nonlinear OCP, T-invariant
superposition with layers from T = 8, layer rates vs the linearized floor lam1_lin and the open-loop lambda_1(beta),
and the plateau shift u_bar(gamma) / u_bar(0)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D, assemble, lambda1, static_optimum  # noqa: E402
from turnpike_analysis import width  # noqa: E402


def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "min": float(np.min(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    te, dt = d["test_idx"], float(d["dt"])
    ub, lam, ge = d["u_bar"][te], d["lam1"][te], d["gamma"][te]
    U8 = d["U_T8"][te]
    half = U8.shape[1] // 2
    plateau = np.abs(U8[:, half - 1] - ub) / np.abs(ub)
    L0 = U8[:, :half] - ub[:, None]
    L1 = U8[:, ::-1][:, :half] - ub[:, None]
    sup = {}
    for T in (1, 2, 4):
        UT = d[f"U_T{T}"][te]
        k = np.arange(UT.shape[1])
        sup[T] = np.linalg.norm(ub[:, None] + L0[:, k] + L1[:, UT.shape[1] - 1 - k] - UT, axis=1) / np.linalg.norm(UT, axis=1)
    n = len(te)
    w0 = {e: np.array([width(L0[i], ub[i], e, dt) for i in range(n)]) for e in (1e-2, 1e-3)}
    w1 = {e: np.array([width(L1[i], ub[i], e, dt) - dt for i in range(n)]) for e in (1e-2, 1e-3)}
    with np.errstate(divide="ignore", invalid="ignore"):
        nu0 = np.log(10.0) / (w0[1e-3] - w0[1e-2])
        nu1 = np.log(10.0) / (w1[1e-3] - w1[1e-2])
    lam_open = np.array([lambda1(100, float(b)) for b in d["beta"][te]])
    ub0 = np.array([static_optimum(assemble(Cfg2D(nh=100, **{k: float(d[k][i]) for k in ("xa", "xb", "ya", "yb", "alpha", "beta")}))) for i in te])
    ok = np.isfinite(nu0) & np.isfinite(nu1)
    rep = {"n_test": n, "gamma_eff": summ(ge), "plateau_rel_err_mid_T8": summ(plateau),
           "superposition_rel_err": {str(T): summ(v) for T, v in sup.items()},
           "plateau_shift_ubar_over_ubar_lq": summ(ub / ub0), "lam1_lin_over_lam1_open": summ(lam / lam_open),
           "initial_rate_over_lam1_lin": summ(nu0 / lam), "terminal_rate_over_lam1_lin": summ(nu1 / lam),
           "frac_initial_below_lam1_lin": float(np.mean(nu0[ok] < lam[ok])), "frac_terminal_below_lam1_lin": float(np.mean(nu1[ok] < lam[ok])),
           "initial_amp_over_ubar": summ(np.max(np.abs(L0), 1) / np.abs(ub)),
           "spearman_superposition_T1_vs_gamma": float(spearmanr(sup[1], ge).statistic),
           "exact_seconds_median": {str(T): float(np.nanmedian(d[f"seconds_T{T}"][te])) for T in (1, 2, 4, 8)},
           "newton_iters_median": {str(T): float(np.nanmedian(d[f"newton_iters_T{T}"][te])) for T in (1, 2, 4, 8)},
           "cg_iters_median": {str(T): float(np.nanmedian(d[f"cg_iters_T{T}"][te])) for T in (1, 2, 4, 8)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
