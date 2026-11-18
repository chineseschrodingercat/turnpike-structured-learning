#!/usr/bin/env python3
"""Figures for the 2D semilinear experiment (semilinear2d.py data).

1) examples: exact vs TurnpikeNet (trained on T = 1) vs CNN (T in {1, 2}) vs LQ control at T = 8 for three test samples
   of the strong set (weak / strong nonlinearity, worst TurnpikeNet sample): whole horizon, layer zooms, pointwise error.
2) overview: plateau shift u_bar / u_bar_LQ vs gamma_eff (colour log10 alpha) for the mild and the strong set, and the
   per-sample T = 8 errors of TurnpikeNet and CNN vs gamma_eff on the strong set.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from robin2d import Cfg2D, assemble, static_optimum  # noqa: E402
from semilinear2d import lq_control  # noqa: E402

GEO = ("xa", "xb", "ya", "yb", "alpha", "beta")


def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)


def examples(d, tp, cnn, out, T=8):
    te, dt = d["test_idx"], float(d["dt"])
    U, Ptp, Pcnn = d[f"U_T{T}"][te], tp[f"U_T{T}"], cnn[f"U_T{T}"]
    e_tp, e_cnn = rel(Ptp, U), rel(Pcnn, U)
    ge = d["gamma"][te]
    rows = [("weak nonlinearity", int(np.argmin(np.abs(ge - np.quantile(ge, 0.1))))),
            ("strong nonlinearity", int(np.argmin(np.abs(ge - np.quantile(ge, 0.95))))),
            ("worst TurnpikeNet sample", int(np.argmax(e_tp)))]
    t = np.arange(1, U.shape[1] + 1) * dt
    fig, axes = plt.subplots(3, 4, figsize=(19, 12.3), gridspec_kw={"width_ratios": [2.2, 1, 1, 1.4]})
    for k, (title, r) in enumerate(rows):
        i = int(te[r])
        g = {key: float(d[key][i]) for key in GEO}
        lq = lq_control(Cfg2D(nh=100, T=float(T), Nt=U.shape[1], **g), (T,))[T]
        ub = float(d["u_bar"][i])
        curves = [("exact (Newton-CG)", U[r], dict(color="k", lw=2.2)),
                  (f"TurnpikeNet, train T=1 ({100 * e_tp[r]:.3f}%)", Ptp[r], dict(color="C3", lw=1.4, ls="--")),
                  (f"CNN, train T=1,2 ({100 * e_cnn[r]:.2f}%)", Pcnn[r], dict(color="C0", lw=1.1, ls="-.")),
                  (f"LQ control, gamma=0 ({100 * rel(lq, U[r]):.1f}%)", lq, dict(color="C2", lw=1.0, ls=":"))]
        for c, (lo, hi) in enumerate([(0, T), (0, 0.5), (T - 0.5, T)]):
            ax = axes[k, c]
            for lab, u, st in curves:
                ax.plot(t, u, label=lab, **st)
            ax.axhline(ub, color="gray", lw=0.8, ls="--", label="plateau u_bar (static nonlinear OCP)" if c == 0 else None)
            ax.set_xlim(lo, hi)
            ax.grid(alpha=0.3)
            ax.set_xlabel("t")
            if c == 0:
                ax.set_ylabel("u(t)")
                ax.set_title(f"{title}: gamma_eff = {ge[r]:.0f}, alpha = {g['alpha']:.2g}, beta = {g['beta']:.2g}, rectangle "
                             f"[{g['xa']:.2f},{g['xb']:.2f}]x[{g['ya']:.2f},{g['yb']:.2f}]", fontsize=9)
                ax.legend(fontsize=7.5)
            else:
                ax.set_title("initial layer (zoom)" if c == 1 else "terminal layer (zoom)", fontsize=9.5)
        ax = axes[k, 3]
        ax.semilogy(t, np.abs(Ptp[r] - U[r]) / abs(ub) + 1e-16, color="C3", lw=1.0, label="TurnpikeNet")
        ax.semilogy(t, np.abs(Pcnn[r] - U[r]) / abs(ub) + 1e-16, color="C0", lw=0.8, alpha=0.8, label="CNN")
        ax.semilogy(t, np.abs(lq - U[r]) / abs(ub) + 1e-16, color="C2", lw=0.8, alpha=0.8, label="LQ control")
        ax.set_ylim(1e-7, 2)
        ax.set_xlim(0, T)
        ax.grid(alpha=0.3)
        ax.set_xlabel("t")
        ax.set_title("pointwise error |u - u*| / |u_bar|", fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle("2D semilinear heat equation y_t - Laplace y + gamma y^3 = 0 (n = 9900), scalar Robin control on x = 0: "
                 "test samples of the strong set at T = 8 (TurnpikeNet r = 8 trained on T = 1 only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=125)
    print("examples:", [(title, int(te[r]), float(ge[r]), float(e_tp[r]), float(e_cnn[r])) for title, r in rows])


def overview(sets, tp, cnn, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, (name, d) in zip(axes[:2], sets):
        te = d["test_idx"]
        ub0 = np.array([static_optimum(assemble(Cfg2D(nh=100, **{k: float(d[k][i]) for k in GEO}))) for i in te])
        sc = ax.scatter(d["gamma"][te], d["u_bar"][te] / ub0, c=np.log10(d["alpha"][te]), s=14, cmap="viridis", vmin=-3, vmax=0)
        ax.set_xscale("log")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(1, color="k", lw=0.6)
        ax.set_xlabel("gamma_eff")
        ax.set_ylabel("u_bar / u_bar_LQ")
        ax.set_title(f"{name}\nplateau shift u_bar / u_bar_LQ (colour: log10 alpha)", fontsize=9.5)
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax)
        print(name, "median u_bar/u_bar_LQ", float(np.median(d["u_bar"][te] / ub0)))
    d = sets[1][1]
    te = d["test_idx"]
    U = d["U_T8"][te]
    ax = axes[2]
    ax.scatter(d["gamma"][te], 100 * rel(tp["U_T8"], U), s=14, color="C3", label="TurnpikeNet (train T=1)")
    ax.scatter(d["gamma"][te], 100 * rel(cnn["U_T8"], U), s=14, color="C0", marker="x", label="CNN (train T=1,2)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gamma_eff")
    ax.set_ylabel("relative L2 error at T = 8 (%)")
    ax.set_title("strong set: per-sample error at T = 8\n(extrapolated; TurnpikeNet r = 8 seed 0, CNN seed 0)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=125)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mild", type=pathlib.Path, required=True)
    ap.add_argument("--strong", type=pathlib.Path, required=True)
    ap.add_argument("--tp", type=pathlib.Path, required=True)
    ap.add_argument("--cnn", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds, tp, cnn = np.load(args.strong), np.load(args.tp), np.load(args.cnn)
    examples(ds, tp, cnn, args.out_dir / "semilinear2d_examples.png")
    overview([("mild set: gamma_eff <= 50, log10 alpha in [-3, 0]", np.load(args.mild)),
              ("strong set: gamma_eff in [10, 1000], log10 alpha in [-3, -1]", ds)], tp, cnn, args.out_dir / "semilinear2d_overview.png")
    print("wrote", args.out_dir)


if __name__ == "__main__":
    main()
