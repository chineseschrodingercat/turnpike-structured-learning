#!/usr/bin/env python3
"""Turnpike structure of the semilinear dataset (test split, which has T = 1, 2, 4, 8) as a function of gamma_eff.

Same metrics as turnpike_analysis.py: plateau error against the static nonlinear OCP, T-invariant superposition
u_T(t) ~ u_bar + l0(t) + l1(T - t) with layers taken from T = 8, layer decay rates from the 1e-2 -> 1e-3 widths.
New here: rates are compared with the linearized floor lam1_lin (at the nonlinear turnpike) and with the open-loop
linear floor lambda_1(beta); the plateau shift u_bar(gamma) / u_bar(0) measures how nonlinear the turnpike itself is.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scipy.linalg as sla  # noqa: E402

from robin_solver import RobinConfig, assemble_operators  # noqa: E402
from semilinear1d import build, static_ocp  # noqa: E402
from turnpike_analysis import width  # noqa: E402


def first_eig(beta: float) -> float:
    """Open-loop floor lambda_1(beta) on the nx = 160 grid (turnpike_surrogate.first_eig without the torch import)."""
    o = assemble_operators(RobinConfig(nx=160, beta=beta))
    return float(sla.eigh(o.K_beta.toarray(), o.M.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0])


def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
            "max": float(np.max(x)), "min": float(np.min(x))}


def by_quartile(v, key):
    edges = np.quantile(key, [0, .25, .5, .75, 1])
    return [{"range": [float(lo), float(hi)], "median": float(np.median(v[(key >= lo) & (key <= hi)]))}
            for lo, hi in zip(edges[:-1], edges[1:])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data)
    te = d["test_idx"]
    dt = float(d["dt"])
    ubar, lam1, ge = d["u_bar"][te], d["lam1"][te], d["gamma_eff"][te]
    beta, alpha, xa, xb, T_ref = d["beta"][te], d["alpha"][te], d["xa"][te], d["xb"][te], d["T_ref"][te]
    U8 = d["U_T8"][te]
    N8 = U8.shape[1]
    half = N8 // 2
    n = len(te)

    plateau_err = np.abs(U8[:, half - 1] - ubar) / np.abs(ubar)
    L0 = U8[:, :half] - ubar[:, None]
    L1 = U8[:, ::-1][:, :half] - ubar[:, None]
    resid = {}
    for T in (1, 2, 4):
        UT = d[f"U_T{T}"][te]
        N = UT.shape[1]
        k = np.arange(N)
        resid[T] = np.linalg.norm(ubar[:, None] + L0[:, k] + L1[:, N - 1 - k] - UT, axis=1) / np.linalg.norm(UT, axis=1)
    w0 = {e: np.array([width(L0[i], ubar[i], e, dt) for i in range(n)]) for e in (1e-2, 1e-3)}
    w1 = {e: np.array([width(L1[i], ubar[i], e, dt) - dt for i in range(n)]) for e in (1e-2, 1e-3)}
    with np.errstate(divide="ignore", invalid="ignore"):
        nu0 = np.log(10.0) / (w0[1e-3] - w0[1e-2])
        nu1 = np.log(10.0) / (w1[1e-3] - w1[1e-2])
    lam1_open = np.array([first_eig(b) for b in beta])
    ubar0 = np.array([static_ocp(build(RobinConfig(nx=160, Nt=320, alpha=alpha[i], beta=beta[i], xa=xa[i], xb=xb[i]), 0.0,
                                       T_ref[i]))["u_bar"] for i in range(n)])
    shift = ubar / ubar0
    ok = np.isfinite(nu0) & np.isfinite(nu1)
    report = {
        "n_test": n,
        "gamma_eff": summ(ge),
        "plateau_shift_ubar_over_ubar_lq": summ(shift),
        "lam1_lin_over_lam1_open": summ(lam1 / lam1_open),
        "plateau_rel_err_mid_T8": summ(plateau_err),
        "superposition_rel_err": {str(T): summ(resid[T]) for T in resid},
        "superposition_T1_by_gamma_eff_quartile": by_quartile(resid[1], ge),
        "superposition_T2_by_gamma_eff_quartile": by_quartile(resid[2], ge),
        "initial_rate_over_lam1_lin": summ(nu0 / lam1),
        "terminal_rate_over_lam1_lin": summ(nu1 / lam1),
        "initial_rate_over_lam1_open": summ(nu0 / lam1_open),
        "frac_initial_rate_below_lam1_lin": float(np.mean(nu0[ok] < lam1[ok])),
        "frac_terminal_rate_below_lam1_lin": float(np.mean(nu1[ok] < lam1[ok])),
        "initial_amp_over_ubar": summ(np.max(np.abs(L0), axis=1) / np.abs(ubar)),
        "terminal_amp_over_ubar": summ(np.max(np.abs(L1), axis=1) / np.abs(ubar)),
        "spearman_vs_gamma_eff": {
            "resid_T1": float(spearmanr(resid[1], ge).statistic), "resid_T2": float(spearmanr(resid[2], ge).statistic),
            "nu0_over_lam1_lin": float(spearmanr((nu0 / lam1)[ok], ge[ok]).statistic),
            "initial_width": float(spearmanr(w0[1e-3], ge).statistic), "plateau_shift": float(spearmanr(shift, ge).statistic)},
    }
    (args.out_dir / "semilinear_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    t8 = np.arange(1, N8 + 1) * dt
    for q, color in zip([0.05, 0.5, 0.95], ["C0", "C1", "C2"]):
        i = int(np.argmin(np.abs(ge - np.quantile(ge, q))))
        ax.plot(t8, U8[i] / ubar[i], color=color, label=f"T=8, gamma_eff={ge[i]:.1f}")
        u1 = d["U_T1"][te[i]]
        ax.plot(np.arange(1, len(u1) + 1) * dt, u1 / ubar[i], "--", color=color, alpha=0.8)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlim(0, 3); ax.set_xlabel("t"); ax.set_title("u_T / u_bar (solid T=8, dashed T=1), t in [0,3]")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax = axes[1]
    for T in (1, 2, 4):
        ax.scatter(ge, resid[T], s=6, label=f"T={T}")
    ax.set_yscale("log"); ax.set_xlabel("gamma_eff = gamma T_ref^2"); ax.set_ylabel("superposition rel. error")
    ax.set_title("T-interaction vs nonlinearity"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[2]
    sc = ax.scatter(lam1[ok], nu0[ok], c=ge[ok], s=8, cmap="viridis", label="initial")
    ax.scatter(lam1[ok], nu1[ok], c=ge[ok], s=8, cmap="viridis", marker="x", label="terminal")
    lim = [lam1.min() * 0.9, max(nu0[ok].max(), nu1[ok].max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=0.8, label="rate = lam1_lin")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("lam1_lin (linearized at turnpike)"); ax.set_ylabel("fitted layer rate")
    ax.set_title("layer rates vs linearized floor (color gamma_eff)"); ax.legend(fontsize=7); fig.colorbar(sc, ax=ax); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "semilinear_analysis.png", dpi=140)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
