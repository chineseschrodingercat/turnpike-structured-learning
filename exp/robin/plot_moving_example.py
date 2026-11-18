#!/usr/bin/env python3
"""One test sample of the moving-target problem at T = 8: target, exact u(s, t), TurnpikeNet (trained on T = 1),
CNN (trained on T in {1, 2}), errors, profiles in the initial layer and time traces."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_moving_data import cfg_of  # noqa: E402
from moving2d import eval_periodic, target  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--tp", type=pathlib.Path, required=True)
    ap.add_argument("--cnn", type=pathlib.Path, required=True)
    ap.add_argument("--q", type=float, default=0.8, help="quantile of the control rank used to pick the sample")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    te = d["test_idx"]
    U = d["U_T8_test"].astype(np.float64)
    Up, Uc = np.load(args.tp)["U_T8"].astype(np.float64), np.load(args.cnn)["U_T8"].astype(np.float64)
    ranks = np.array([np.sum(np.linalg.svd(U[r], compute_uv=False) > 1e-3 * np.linalg.norm(U[r], 2)) for r in range(len(te))])
    e_tp = np.array([rel(Up[r], U[r]) for r in range(len(te))])
    cand = np.nonzero(ranks >= np.quantile(ranks, args.q))[0]
    r = int(cand[np.argsort(e_tp[cand])[len(cand) // 2]])
    i = int(te[r])
    cf = cfg_of(d["X"][i], 8)
    t = np.arange(1, cf.Nt + 1) * cf.dt
    s = np.arange(1, 100) / 100
    u, up, uc = U[r], Up[r], Uc[r]
    uh = eval_periodic(d["uhat"][i].astype(np.complex128), cf.omega, cf.phi0, t)
    Yd = target(cf, s[None, :], cf.omega * t[:, None] + cf.phi0)
    v = np.abs(u).max()

    fig, axes = plt.subplots(2, 4, figsize=(21, 9.6))

    def heat(ax, Z, lo, hi, ttl, **kw):
        sel = (t >= lo) & (t <= hi)
        pc = ax.pcolormesh(t[sel], s, Z[sel].T, shading="auto", **kw)
        ax.set_xlabel("t")
        ax.set_ylabel("s (along the edge)")
        ax.set_title(ttl, fontsize=9.5)
        fig.colorbar(pc, ax=ax)

    heat(axes[0, 0], Yd, 0, 2, f"target y_d(s, t), t in [0, 2]: a = {cf.a:.2f}, omega = {cf.omega / np.pi:.2f} pi, sigma = {cf.sigma:.3f}", cmap="viridis")
    heat(axes[0, 1], u, 0, 2, "exact u(s, t) (PCG), t in [0, 2]", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(axes[0, 2], up, 0, 2, f"TurnpikeNet, trained on T = 1 only ({100 * rel(up, u):.3f}% on [0, 8])", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(axes[0, 3], np.maximum(np.abs(up - u) / v, 1e-8), 0, 8, f"TurnpikeNet |error| / max|u|, t in [0, 8] (CNN {100 * rel(uc, u):.2f}%)",
         cmap="magma", norm=LogNorm(1e-6, 1e-1))
    heat(axes[1, 0], u, 6, 8, "exact u(s, t), t in [6, 8] (terminal layer)", cmap="RdBu_r", vmin=-v, vmax=v)
    heat(axes[1, 1], up, 6, 8, "TurnpikeNet, t in [6, 8]", cmap="RdBu_r", vmin=-v, vmax=v)
    ax = axes[1, 2]
    for tt, col in zip([0.01, 0.05, 0.15], ["C3", "C1", "C2"]):
        k = int(round(tt / cf.dt)) - 1
        ax.plot(s, u[k], color=col, lw=2.0, label=f"exact, t = {tt}")
        ax.plot(s, up[k], color=col, lw=1.0, ls="--")
        ax.plot(s, uc[k], color=col, lw=0.8, ls=":")
        ax.plot(s, uh[k], color=col, lw=0.6, ls="-.", alpha=0.6)
    ax.set_xlabel("s")
    ax.set_title("initial-layer profiles: solid exact, dashed TurnpikeNet,\ndotted CNN, dash-dot periodic turnpike alone", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax = axes[1, 3]
    j = int(np.argmin(np.abs(s - 0.5)))
    for lo, hi, ls in ((0, 1.0, "-"),):
        sel = (t >= lo) & (t <= hi)
        ax.plot(t[sel], u[sel, j], "k", lw=2.0, label="exact")
        ax.plot(t[sel], up[sel, j], "r--", lw=1.2, label="TurnpikeNet")
        ax.plot(t[sel], uc[sel, j], "C0:", lw=1.0, label="CNN")
        ax.plot(t[sel], uh[sel, j], "C2-.", lw=0.9, label="periodic turnpike alone")
    ax.set_xlabel("t")
    ax.set_title("time trace at s = 0.5, t in [0, 1]", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    sv = np.linalg.svd(u, compute_uv=False)
    fig.suptitle(f"Moving hot-spot target, distributed edge control (n = 9900, m = 99), T = 8, test sample #{i}: w = {cf.w:.2f}, "
                 f"alpha = {cf.alpha:.1e}, beta = {cf.beta:.2f};  rank@1e-3 of u = {int(np.sum(sv > 1e-3 * sv[0]))}, "
                 f"s2/s1 = {sv[1] / sv[0]:.2f}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("sample", i, "tp", rel(up, u), "cnn", rel(uc, u), "periodic", rel(uh, u))


if __name__ == "__main__":
    main()
