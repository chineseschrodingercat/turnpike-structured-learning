#!/usr/bin/env python3
"""Pilot figure for the moving-target problem (validate_moving.py output): target y_d(s, t) on the edge coordinate,
exact control u(s, t) over T = 4, one period in the middle (exact vs periodic turnpike), singular values of u(s, t)."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from moving2d import CfgMove, target  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    z = np.load(args.npz)
    pilots = json.loads(str(z["pilots"]))
    T, N = 4.0, 1280
    t = np.arange(1, N + 1) * T / N
    s = np.arange(1, 100) / 100
    fig, axes = plt.subplots(len(pilots), 4, figsize=(20, 4.3 * len(pilots)), gridspec_kw={"width_ratios": [1, 1.6, 1, 0.8]})
    for j, p in enumerate(pilots):
        cf = CfgMove(**p)
        C, uh = z[f"C{j}"].astype(float), z[f"uh{j}"].astype(float)
        th = cf.omega * t + cf.phi0
        Yd = target(cf, s[None, :], th[:, None])
        ax = axes[j, 0]
        pc = ax.pcolormesh(t, s, Yd.T, shading="auto", cmap="viridis")
        ax.set_ylabel("s (along the edge)")
        ax.set_xlabel("t")
        ax.set_title(f"target y_d(s, t): a = {cf.a}, omega = {cf.omega / np.pi:.0f} pi, sigma = {cf.sigma}", fontsize=9.5)
        fig.colorbar(pc, ax=ax)
        ax = axes[j, 1]
        v = np.abs(C).max()
        pc = ax.pcolormesh(t, s, C.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_xlabel("t")
        ax.set_ylabel("s")
        ax.set_title(f"exact optimal control u(s, t), T = 4 (alpha = {cf.alpha:.1g}, beta = {cf.beta}, w = {cf.w})", fontsize=9.5)
        fig.colorbar(pc, ax=ax)
        ax = axes[j, 2]
        P = 2 * np.pi / cf.omega
        sel = (t > 2.0) & (t < 2.0 + P)
        for tt, col in zip(np.linspace(2.0, 2.0 + P, 5)[:-1], ["C0", "C1", "C2", "C3"]):
            k = int(np.argmin(np.abs(t - tt)))
            ax.plot(s, C[k], color=col, lw=1.8, label=f"t = {t[k]:.2f}")
            ax.plot(s, uh[k], color="k", lw=0.8, ls="--")
        ax.set_xlabel("s")
        ax.set_title("profiles over one period (dashed: periodic turnpike u_hat)", fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
        ax = axes[j, 3]
        sv = np.linalg.svd(C, compute_uv=False)
        ax.semilogy(np.arange(1, 21), sv[:20] / sv[0], "o-", ms=4)
        ax.axhline(1e-3, color="gray", lw=0.7, ls=":")
        ax.set_xlabel("index")
        ax.set_title(f"singular values of u(s,t) (rank@1e-3 = {int(np.sum(sv > 1e-3 * sv[0]))})", fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.set_ylim(1e-6, 1.5)
    fig.suptitle("Moving hot-spot target on the edge strip (n = 9900, m = 99): exact optimal controls", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
