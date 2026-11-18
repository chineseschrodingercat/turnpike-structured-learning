#!/usr/bin/env python3
"""T = 4 ground truth vs one surrogate (default tp_transport, trained on T = 2 only) for the advection dataset:
rows = test samples at the median / p90 / worst surrogate error; columns = exact u(s, t), prediction, error map,
profiles at four times, |error(t)| / max|u| on a log scale."""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--pred", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--label", default="tp_transport (ours)")
    args = ap.parse_args()
    d = np.load(args.data)
    U4, X, te = d["U_T4_test"].astype(float), d["X"], d["test_idx"]
    Up = np.load(args.pred)["U_T4"].astype(float)
    err = np.linalg.norm((Up - U4).reshape(len(te), -1), axis=1) / np.linalg.norm(U4.reshape(len(te), -1), axis=1)
    order = np.argsort(err)
    picks = [("median", order[len(order) // 2]), ("90th percentile", order[int(0.9 * (len(order) - 1))]), ("worst", order[-1])]
    N, m = U4.shape[1], U4.shape[2]
    t = np.arange(1, N + 1) / 320.0
    s = np.arange(1, m + 1) / (m + 1)
    fig, axes = plt.subplots(len(picks), 5, figsize=(29, 5.2 * len(picks)), gridspec_kw={"width_ratios": [1.25, 1.25, 1.25, 1, 1]})
    for r, (lab, j) in enumerate(picks):
        U, P = U4[j], Up[j]
        v = np.quantile(np.abs(U), 0.995)
        for c, (Z, ttl) in enumerate([(U, "ground truth (exact optimal control)"), (P, f"{args.label}: prediction")]):
            ax = axes[r, c]
            pc = ax.pcolormesh(t, s, Z.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
            ax.set_title(f"{ttl}" + (f"\n{lab} test sample #{int(te[j])}" if c == 0 else f"\nrelative L2 error {100 * err[j]:.2f}%"), fontsize=10)
            fig.colorbar(pc, ax=ax)
        ax = axes[r, 2]
        D = P - U
        ve = np.quantile(np.abs(D), 0.999)
        pc = ax.pcolormesh(t, s, D.T, shading="auto", cmap="PuOr_r", vmin=-ve, vmax=ve)
        ax.set_title(f"error (prediction - exact), colour scale {ve / v:.1%} of the control's", fontsize=10)
        fig.colorbar(pc, ax=ax)
        for c in range(3):
            ax = axes[r, c]
            ax.set_xlabel("t")
            ax.set_ylabel("s (along the edge)")
            ax.axvline(2.0, color="k", lw=0.8, ls=":")
            for tw in (1.0, 3.0):
                ax.axvline(tw, color="gray", lw=0.6, ls="--")
        ax = axes[r, 3]
        for tt, col in zip((0.03, 1.0, 2.5, 3.85), ("C0", "C1", "C2", "C3")):
            k = int(round(tt * 320)) - 1
            ax.plot(s, U[k], color=col, lw=1.8, label=f"t = {t[k]:.2f}")
            ax.plot(s, P[k], color=col, lw=1.0, ls="--")
        ax.set_xlabel("s")
        ax.set_title("profiles: solid exact, dashed prediction", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[r, 4]
        ax.semilogy(t, np.linalg.norm(U, axis=1), color="k", lw=1.2, label="|u_exact(t)|")
        ax.semilogy(t, np.linalg.norm(P - U, axis=1), color="C3", lw=1.2, label="|error(t)|")
        ax.axvline(2.0, color="k", lw=0.8, ls=":")
        for tw in (1.0, 3.0):
            ax.axvline(tw, color="gray", lw=0.6, ls="--")
        ax.set_ylim(1e-5, None)
        ax.set_xlabel("t")
        ax.set_title("norm over s at each time", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        mu = X[te[j]]
        axes[r, 0].text(0.01, 0.01, f"gamma = {mu[0]:.2f}, eps = {10 ** mu[1]:.1e}, a = {mu[3]:.2f}, sigma = {10 ** mu[4]:.3f}",
                        transform=axes[r, 0].transAxes, fontsize=8, va="bottom", bbox=dict(fc="white", alpha=0.7, lw=0))
    fig.suptitle(f"Advection with downstream observation, horizon T = 4: ground truth vs {args.label}, trained on T = 2 only. "
                 f"Dotted: end of the training horizon; dashed: layer windows [0, 1] and [3, 4] (exact periodic part in between). "
                 f"Test set: median {100 * np.median(err):.2f}%, max {100 * err.max():.2f}%", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=100)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
