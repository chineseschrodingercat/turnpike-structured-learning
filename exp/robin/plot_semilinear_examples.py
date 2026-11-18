#!/usr/bin/env python3
"""Exact vs learned controls for a few semilinear test samples at T = 8 (models trained on T = 1 / T in {1, 2}).

Rows: weak nonlinearity, strong nonlinearity, worst TurnpikeNet sample at T = 8.
Columns: whole horizon, initial-layer zoom, terminal-layer zoom, pointwise error / |u_bar|.
Also drawn: the LQ control (gamma = 0 optimum) and the plateau u_bar (static nonlinear OCP).
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
from robin_solver import RobinConfig  # noqa: E402
from semilinear1d import build, solve_kkt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--tp", type=pathlib.Path, required=True, help="TurnpikeNet predictions (trained on T = 1)")
    ap.add_argument("--cnn", type=pathlib.Path, required=True, help="CNN predictions (trained on T in {1, 2})")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()
    d = np.load(args.data)
    te, dt, T = d["test_idx"], float(d["dt"]), args.T
    U = d[f"U_T{T}"][te]
    P_tp, P_cnn = np.load(args.tp)[f"U_T{T}"], np.load(args.cnn)[f"U_T{T}"]
    err_tp = np.linalg.norm(P_tp - U, axis=1) / np.linalg.norm(U, axis=1)
    err_cnn = np.linalg.norm(P_cnn - U, axis=1) / np.linalg.norm(U, axis=1)
    ge = d["gamma_eff"][te]
    rows = [("weak nonlinearity", int(np.argmin(np.abs(ge - np.quantile(ge, 0.1))))),
            ("strong nonlinearity", int(np.argmin(np.abs(ge - np.quantile(ge, 0.95))))),
            ("worst TurnpikeNet sample", int(np.argmax(err_tp)))]
    t = np.arange(1, U.shape[1] + 1) * dt
    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 4.1 * len(rows)), gridspec_kw={"width_ratios": [2.2, 1, 1, 1.4]})
    for k, (title, r) in enumerate(rows):
        i = int(te[r])
        geo = dict(alpha=float(d["alpha"][i]), beta=float(d["beta"][i]), xa=float(d["xa"][i]), xb=float(d["xb"][i]))
        lq = solve_kkt(build(RobinConfig(nx=160, Nt=U.shape[1], T=float(T), **geo), 0.0, float(d["T_ref"][i])))[2]
        ub = float(d["u_bar"][i])
        curves = [("exact (Newton-KKT)", U[r], dict(color="k", lw=2.2)),
                  (f"TurnpikeNet, train T=1 ({100 * err_tp[r]:.3f}%)", P_tp[r], dict(color="C3", lw=1.4, ls="--")),
                  (f"CNN, train T=1,2 ({100 * err_cnn[r]:.2f}%)", P_cnn[r], dict(color="C0", lw=1.1, ls="-.")),
                  (f"LQ control, gamma=0 ({100 * np.linalg.norm(lq - U[r]) / np.linalg.norm(U[r]):.1f}%)", lq, dict(color="C2", lw=1.0, ls=":"))]
        for c, (lo, hi) in enumerate([(0, T), (0, 0.6), (T - 0.6, T)]):
            ax = axes[k, c]
            for lab, u, st in curves:
                ax.plot(t, u, label=lab, **st)
            ax.axhline(ub, color="gray", lw=0.8, ls="--", label="plateau u_bar (static OCP)" if c == 0 else None)
            ax.set_xlim(lo, hi)
            ax.grid(alpha=0.3)
            ax.set_xlabel("t")
            if c == 0:
                ax.set_ylabel("u(t)")
                ax.set_title(f"{title}: gamma_eff={ge[r]:.1f}, alpha={geo['alpha']:.2g}, beta={geo['beta']:.2g}, "
                             f"window [{geo['xa']:.2f},{geo['xb']:.2f}]", fontsize=9.5)
                ax.legend(fontsize=7.5, loc="best")
            else:
                ax.set_title("initial layer (zoom)" if c == 1 else "terminal layer (zoom)", fontsize=9.5)
        ax = axes[k, 3]
        ax.semilogy(t, np.abs(P_tp[r] - U[r]) / abs(ub) + 1e-16, color="C3", lw=1.0, label="TurnpikeNet")
        ax.semilogy(t, np.abs(P_cnn[r] - U[r]) / abs(ub) + 1e-16, color="C0", lw=0.8, alpha=0.8, label="CNN")
        ax.set_ylim(1e-7, 1)
        ax.set_xlim(0, T)
        ax.grid(alpha=0.3)
        ax.set_xlabel("t")
        ax.set_title("pointwise error |u_pred - u*| / |u_bar|", fontsize=9.5)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Semilinear heat equation y_t - y_xx + gamma y^3 = 0, Robin boundary control: test samples at T = {T} "
                 f"(TurnpikeNet r=8, N_train=3400, trained on T=1 only)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("rows:", [(title, int(te[r]), float(ge[r]), float(err_tp[r]), float(err_cnn[r])) for title, r in rows])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
