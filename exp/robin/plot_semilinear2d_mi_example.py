#!/usr/bin/env python3
"""One strongly nonlinear test sample of the 2D semilinear distributed-control problem (section 22), T = 8:
exact u(s, t) (Newton-CG) vs TurnpikeNet (exact nonlinear plateau + learned layers, trained on T = 1) vs the tau-linear
superposition (what works in the linear case, section 18b) vs the LQ control; plus the nonlinear state at t = 4."""

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
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_semilinear2d_mi_data import M, cfg_of  # noqa: E402
from plot_2d_examples import field  # noqa: E402
from semilinear2d_mi import build, forward  # noqa: E402


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, required=True)
    ap.add_argument("--tp", type=pathlib.Path, required=True)
    ap.add_argument("--tau", type=pathlib.Path, required=True)
    ap.add_argument("--q", type=float, default=0.9, help="quantile of the plateau nonlinearity used to pick the sample")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    d = np.load(args.data)
    te, dt = d["test_idx"], float(d["dt"])
    nl = np.linalg.norm(d["u_bar"][te] - d["u_bar_lq"][te], axis=1) / np.linalg.norm(d["u_bar_lq"][te], axis=1)
    r = int(np.argmin(np.abs(nl - np.quantile(nl, args.q))))
    i = int(te[r])
    row = d["X"][i]
    u = d["U_T8_test"][r].astype(np.float64)
    up = np.load(args.tp)["U_T8"][r].astype(np.float64)
    ut = np.load(args.tau)["U_T8"][r].astype(np.float64)
    ulq = d["U_lq_T8_test"][r].astype(np.float64)
    ub, ubl = d["u_bar"][i], d["u_bar_lq"][i]
    t = np.arange(1, u.shape[0] + 1) * dt
    s = np.arange(1, M + 1) / (M + 1)
    gamma = float(10.0 ** row[9])
    cfg = cfg_of(row, 4)
    sm = build(cfg, M, gamma)
    y4 = forward(sm, u[: cfg.Nt])[-1]
    slq = build(cfg, M, 0.0)
    y4lq = forward(slq, ulq[: cfg.Nt])[-1]

    fig = plt.figure(figsize=(20, 9.8))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    yy = np.linspace(0, 1, 400)
    yd = 1 + sum(row[3 + k] * np.sin((k + 1) * np.pi * yy) for k in range(6))
    ax.plot(yy, yd, "C1", lw=1.6, label="target y_d(y)")
    ax.set_xlabel("y  (= s along the edge)")
    ax.set_ylabel("target", color="C1")
    ax2 = ax.twinx()
    ax2.plot(s, ub, "k", lw=2.0, label="plateau, nonlinear (exact static OCP)")
    ax2.plot(s, ubl, "C2", lw=1.3, ls=":", label="plateau, LQ = tau-linear superposition")
    ax2.plot(s, up[1279], "r--", lw=1.0, label="TurnpikeNet u(s, t=4)")
    ax2.set_ylabel("control")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="best")
    ax.set_title(f"target vs plateau (nonlinear shift {100 * nl[r]:.0f}%)", fontsize=9.5)
    ax.grid(alpha=0.3)

    vmax = max(np.abs(u).max(), np.abs(up).max())
    for c, (Z, ttl) in enumerate([(u, "exact u(s, t) (Newton-CG)"), (up, f"TurnpikeNet, trained on T = 1 ({100 * rel(up, u):.2f}%)")]):
        axh = fig.add_subplot(gs[0, 1 + c])
        pc = axh.pcolormesh(t, s, Z.T, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axh.set_xlabel("t")
        axh.set_ylabel("s (position along the edge x = 0)")
        axh.set_title(ttl, fontsize=9.5)
        fig.colorbar(pc, ax=axh)
    axe = fig.add_subplot(gs[0, 3])
    pc = axe.pcolormesh(t, s, np.maximum(np.abs(up - u) / np.abs(ub).max(), 1e-8).T, shading="auto", cmap="magma", norm=LogNorm(1e-6, 1e-1))
    axe.set_xlabel("t")
    axe.set_ylabel("s")
    axe.set_title(f"TurnpikeNet |error| / max|u_bar|; tau-linear {100 * rel(ut, u):.1f}%, LQ {100 * rel(ulq, u):.1f}%", fontsize=9)
    fig.colorbar(pc, ax=axe)

    ax = fig.add_subplot(gs[1, 0])
    for tt, col in zip([0.006, 0.05, 4.0, 7.95], ["C3", "C1", "k", "C0"]):
        k = int(round(tt / dt)) - 1
        ax.plot(s, u[k], color=col, lw=2.0, label=f"t = {tt}")
        ax.plot(s, up[k], color=col, lw=1.0, ls="--")
        ax.plot(s, ut[k], color=col, lw=0.9, ls=":")
    ax.set_xlabel("s")
    ax.set_ylabel("u(s, t)")
    ax.set_title("profiles: solid exact, dashed TurnpikeNet, dotted tau-linear", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    inner = np.nonzero((s > 0.15) & (s < 0.85))[0]
    j_pick = [int(inner[np.argmax(ub[inner])]), int(inner[np.argmin(ub[inner])])]
    for c, (lo, hi, ttl) in enumerate([(0, 0.3, "time traces, initial layer"), (7.7, 8.0, "time traces, terminal layer")]):
        ax = fig.add_subplot(gs[1, 1 + c])
        for j, col in zip(j_pick, ["C3", "C0"]):
            ax.plot(t, u[:, j], color=col, lw=2.0, label=f"exact, s = {s[j]:.2f}")
            ax.plot(t, up[:, j], color=col, lw=1.0, ls="--", label="TurnpikeNet")
            ax.plot(t, ut[:, j], color=col, lw=0.9, ls=":", label="tau-linear")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("t")
        ax.set_title(ttl, fontsize=9.5)
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 3])
    F = field(y4)
    vm = np.abs(F).max()
    im = ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="bilinear")
    ax.add_patch(Rectangle((0, 0.05), float(row[0]), 0.9, fill=False, ec="k", lw=1.4, ls="--"))
    ax.plot([0, 0], [0, 1], color="lime", lw=5, solid_capstyle="butt")
    ax.set_title(f"nonlinear state at t = 4 (max {np.abs(y4).max():.2f}; LQ state max {np.abs(y4lq).max():.2f})", fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)
    fig.suptitle(f"2D semilinear, distributed edge control u(s,t) (m = 99, n = 9900), edge-strip target, T = 8, test sample #{i}: "
                 f"w = {row[0]:.2f}, alpha = {10**row[1]:.2g}, beta = {10**row[2]:.2g}, gamma_eff = {gamma:.0f}", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=125)
    print("sample", i, "nl", float(nl[r]), "tp", rel(up, u), "tau", rel(ut, u), "lq", rel(ulq, u))


if __name__ == "__main__":
    main()
