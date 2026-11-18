#!/usr/bin/env python3
"""Quantify the turnpike structure of the multi-horizon dataset.

From the longest horizon (T = 8) define, per sample,
    l0(t) = u_8(t) - u_bar        (initial layer, t < 4)
    l1(s) = u_8(8 - s) - u_bar    (terminal layer, s = T - t < 4)
and test the T-invariant superposition u_T(t) ~ u_bar + l0(t) + l1(T - t) for
T = 1, 2, 4. Its relative error is the "interaction" a T=1-trained structured
model cannot see; theory predicts ~ exp(-nu T).
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


def width(layer: np.ndarray, scale: float, eps: float, dt: float) -> float:
    """Largest t (or s) where |layer| > eps * scale; layer[k] sits at (k+1) dt or k dt."""
    idx = np.nonzero(np.abs(layer) > eps * scale)[0]
    return float((idx[-1] + 1) * dt) if idx.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data)
    dt = float(d["dt"])
    ubar, xa, alpha, beta = d["u_bar"], d["xa"], d["alpha"], d["beta"]
    U8 = d["U_T8"]
    N8 = U8.shape[1]
    half = N8 // 2
    n = len(ubar)

    plateau_err = np.abs(U8[:, half - 1] - ubar) / np.abs(ubar)
    L0 = U8[:, :half] - ubar[:, None]                 # l0 at t_k = (k+1) dt
    L1 = U8[:, ::-1][:, :half] - ubar[:, None]        # l1 at s_m = m dt

    resid = {}
    for T in (1, 2, 4):
        UT = d[f"U_T{T}"]
        N = UT.shape[1]
        k = np.arange(N)
        approx = ubar[:, None] + L0[:, k] + L1[:, N - 1 - k]
        resid[T] = np.linalg.norm(approx - UT, axis=1) / np.linalg.norm(UT, axis=1)

    w0 = {e: np.array([width(L0[i], ubar[i], e, dt) for i in range(n)]) for e in (1e-2, 1e-3)}
    w1 = {e: np.array([width(L1[i], ubar[i], e, dt) - dt for i in range(n)]) for e in (1e-2, 1e-3)}
    with np.errstate(divide="ignore", invalid="ignore"):
        nu0 = np.log(10.0) / (w0[1e-3] - w0[1e-2])
        nu1 = np.log(10.0) / (w1[1e-3] - w1[1e-2])
    amp0 = np.max(np.abs(L0), axis=1) / ubar
    amp1 = np.max(np.abs(L1), axis=1) / ubar

    def summ(x):
        x = x[np.isfinite(x)]
        return {"median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
                "p90": float(np.percentile(x, 90)), "max": float(np.max(x)), "min": float(np.min(x))}

    la = np.log10(alpha)
    report = {
        "n": n,
        "plateau_rel_err_at_t4_T8": summ(plateau_err),
        "superposition_rel_err": {str(T): summ(resid[T]) for T in resid},
        "superposition_rel_err_by_log_alpha_quartile": {
            str(T): [float(np.median(resid[T][(la >= lo) & (la <= hi)]))
                     for lo, hi in zip(np.quantile(la, [0, .25, .5, .75]), np.quantile(la, [.25, .5, .75, 1]))]
            for T in resid},
        "initial_width_1e-3": summ(w0[1e-3]),
        "terminal_width_1e-3": summ(w1[1e-3]),
        "initial_rate_nu0": summ(nu0),
        "terminal_rate_nu1": summ(nu1),
        "initial_amp_over_ubar": summ(amp0),
        "terminal_amp_over_ubar": summ(amp1),
        "spearman": {
            "resid_T1_vs": {"log_alpha": float(spearmanr(resid[1], la).statistic), "xa": float(spearmanr(resid[1], xa).statistic),
                            "log_beta": float(spearmanr(resid[1], np.log10(beta)).statistic)},
            "initial_width_vs": {"log_alpha": float(spearmanr(w0[1e-3], la).statistic), "xa": float(spearmanr(w0[1e-3], xa).statistic),
                                 "log_beta": float(spearmanr(w0[1e-3], np.log10(beta)).statistic)},
            "terminal_width_vs": {"log_alpha": float(spearmanr(w1[1e-3], la).statistic), "xa": float(spearmanr(w1[1e-3], xa).statistic),
                                  "log_beta": float(spearmanr(w1[1e-3], np.log10(beta)).statistic)},
            "nu0_vs_nu1": float(spearmanr(nu0[np.isfinite(nu0) & np.isfinite(nu1)], nu1[np.isfinite(nu0) & np.isfinite(nu1)]).statistic),
        },
    }
    (args.out_dir / "turnpike_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez(args.out_dir / "turnpike_layers.npz", resid1=resid[1], resid2=resid[2], resid4=resid[4],
             w0=w0[1e-3], w1=w1[1e-3], nu0=nu0, nu1=nu1, plateau_err=plateau_err)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    for q, color in zip([0.05, 0.5, 0.95], ["C0", "C1", "C2"]):
        i = int(np.argmin(np.abs(la - np.quantile(la, q))))
        t8 = np.arange(1, N8 + 1) * dt
        ax.plot(t8, U8[i] / ubar[i], color=color, label=f"T=8, log10 a={la[i]:.2f}, x_a={xa[i]:.2f}")
        u1 = d["U_T1"][i]
        ax.plot(np.arange(1, len(u1) + 1) * dt, u1 / ubar[i], "--", color=color, alpha=0.8)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlim(0, 3); ax.set_title("u_T / u_bar (solid T=8, dashed T=1); t in [0,3]")
    ax.set_xlabel("t"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    for T in (1, 2, 4):
        ax.scatter(la, resid[T], s=5, label=f"T={T}")
    ax.set_yscale("log"); ax.set_xlabel("log10 alpha"); ax.set_ylabel("rel. error of u_bar + l0(t) + l1(T-t)")
    ax.set_title("T-interaction (superposition error)"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    sc = ax.scatter(w0[1e-3], w1[1e-3], c=la, s=6, cmap="viridis")
    ax.set_xlabel("initial layer width (|l0| > 1e-3 u_bar)"); ax.set_ylabel("terminal layer width")
    ax.set_title("layer widths, color = log10 alpha"); fig.colorbar(sc, ax=ax); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "turnpike_analysis.png", dpi=140)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
