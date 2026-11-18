#!/usr/bin/env python3
"""Ground-truth controls for the nonlinear moving-target problem: target, exact nonlinear u(s, t), exact LQ control,
their difference, profiles over one period, singular values. Input: npz with C_nl, C_lq (+ the CfgMove parameters)."""

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
    ap.add_argument("--cases", nargs="+", required=True, help="label=path.npz=json_params")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(len(args.cases), 5, figsize=(24, 4.6 * len(args.cases)), gridspec_kw={"width_ratios": [1, 1.4, 1.4, 1, 0.8]},
                             squeeze=False)
    for row, spec in enumerate(args.cases):
        lab, path, pj = spec.split("=", 2)
        z = np.load(path)
        prm = json.loads(pj)
        gamma = prm.pop("gamma")
        cf = CfgMove(**prm)
        Cn, Cl = z["C_nl"].astype(float), z["C_lq"].astype(float)
        t = np.arange(1, Cn.shape[0] + 1) * cf.dt
        s = np.arange(1, 100) / 100
        Yd = target(cf, s[None, :], cf.omega * t[:, None] + cf.phi0)
        v = max(np.abs(Cn).max(), np.abs(Cl).max())
        ax = axes[row, 0]
        pc = ax.pcolormesh(t, s, Yd.T, shading="auto", cmap="viridis")
        ax.set_title(f"{lab}: target y_d(s, t)", fontsize=9.5)
        ax.set_xlabel("t")
        ax.set_ylabel("s (along the edge)")
        fig.colorbar(pc, ax=ax)
        for c, (Z, ttl) in enumerate([(Cn, f"exact NONLINEAR u(s, t), gamma = {gamma:g}"), (Cl, "exact LQ control (gamma = 0)")]):
            ax = axes[row, 1 + c]
            pc = ax.pcolormesh(t, s, Z.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(ttl + f"  (alpha = {cf.alpha:.1e})", fontsize=9.5)
            ax.set_xlabel("t")
            ax.set_ylabel("s")
            fig.colorbar(pc, ax=ax)
        ax = axes[row, 3]
        P = 2 * np.pi / cf.omega
        t0 = 0.5 * t[-1]
        for tt, col in zip(np.linspace(t0, t0 + P, 4, endpoint=False), ["C0", "C1", "C2", "C3"]):
            k = int(np.argmin(np.abs(t - tt)))
            ax.plot(s, Cn[k], color=col, lw=1.8, label=f"t = {t[k]:.2f}")
            ax.plot(s, Cl[k], color=col, lw=0.9, ls="--")
        ax.set_xlabel("s")
        ax.set_title(f"one period: solid nonlinear, dashed LQ (|u - u_LQ|/|u_LQ| = {100 * np.linalg.norm(Cn - Cl) / np.linalg.norm(Cl):.0f}%)", fontsize=9)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
        ax = axes[row, 4]
        for Z, st, nm in ((Cn, "o-", "nonlinear"), (Cl, "x--", "LQ")):
            sv = np.linalg.svd(Z, compute_uv=False)
            ax.semilogy(np.arange(1, 21), sv[:20] / sv[0], st, ms=4, label=f"{nm} (rank@1e-3 = {int(np.sum(sv > 1e-3 * sv[0]))})")
        ax.axhline(1e-3, color="gray", lw=0.7, ls=":")
        ax.set_ylim(1e-6, 1.5)
        ax.set_title("singular values of u(s, t)", fontsize=9.5)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Ground truth: optimal edge control for a moving hot-spot target with the nonlinear state equation y_t - Laplace y + gamma y^3 = 0 "
                 "(n = 9900, m = 99)", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=115)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
