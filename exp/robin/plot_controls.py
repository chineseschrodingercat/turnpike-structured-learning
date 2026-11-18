#!/usr/bin/env python3
"""Plot representative optimal controls u(t) (normalized by T_ref) across alpha and x_a.

Also re-solves a few constant-target problems on longer horizons T to see whether
the control develops a steady (turnpike) middle part with layers at t = 0 and t = T.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from robin_solver import RobinConfig, assemble_operators, assemble_problem_data, reduced_control, reduced_system  # noqa: E402


def control(alpha, beta, xa, xb, T, Nt, yd=1.0):
    cfg = RobinConfig(alpha=alpha, beta=beta, xa=xa, xb=xb, T=T, Nt=Nt)
    ops = assemble_operators(cfg)
    data = assemble_problem_data(cfg, ops, lambda x, t: np.full_like(np.asarray(x, float), yd))
    return np.arange(1, Nt + 1) * cfg.dt, reduced_control(reduced_system(ops, data), alpha)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    X, U, xa, alpha = d["X"], d["U"], d["xa"], d["alpha"]
    t = d["t"]
    T_ref = X[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    mid = np.nonzero((xa > 0.3) & (xa < 0.45))[0]
    for q in np.quantile(np.log10(alpha[mid]), [0.02, 0.25, 0.5, 0.75, 0.98]):
        i = mid[np.argmin(np.abs(np.log10(alpha[mid]) - q))]
        ax.plot(t, U[i] / T_ref[i], label=f"log10 a={np.log10(alpha[i]):.2f}, x_a={xa[i]:.2f}")
    ax.set_title("robin_beta1: u/T_ref, x_a in (0.3,0.45), vary alpha"); ax.set_yscale("symlog", linthresh=1e-1)
    ax.set_xlabel("t"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    sel = np.nonzero(np.abs(np.log10(alpha) + 2.0) < 0.15)[0]
    for q in np.quantile(xa[sel], [0.02, 0.25, 0.5, 0.75, 0.98]):
        i = sel[np.argmin(np.abs(xa[sel] - q))]
        ax.plot(t, U[i] / T_ref[i], label=f"x_a={xa[i]:.3f}, log10 a={np.log10(alpha[i]):.2f}")
    ax.set_title("robin_beta1: u/T_ref, log10 alpha ~ -2, vary x_a"); ax.set_yscale("symlog", linthresh=1e-1)
    ax.set_xlabel("t"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2]
    for T in (1.0, 2.0, 4.0, 8.0):
        tt, uu = control(1e-2, 1.0, 0.3, 0.6, T, int(320 * T))
        ax.plot(tt, uu, label=f"T={T:g}")
    ax.set_title("constant target 1, alpha=1e-2, beta=1, omega=(0.3,0.6): longer horizons")
    ax.set_xlabel("t"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
